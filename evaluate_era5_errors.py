"""Errors of saved predictions against ERA5, per start time, lead time, variable and level, for several models at once.

Usage:
  python evaluate_era5_errors.py \
      --pred esfmS_hydro_24h_ft=/iopsstor/scratch/cscs/$USER/eval_hydro_2021/esfmS_hydro_24h_ft/predictions_val_set_step-1.zarr \
      --pred 'auroraS_hydro_24h_ft=/iopsstor/scratch/cscs/$USER/eval_hydro_2021/auroraS_hydro_24h_ft/part*/rollout_pred_*.zarr' \
      --out errors_eval2021.csv

Predictions are zarrs written by inference.py / inference_rollout.py (utils/merge_predictions.py) or by the old code's
inference_direct.py --format_for_evaluation: dims init_time, lead_time, ensemble, [level], latitude, longitude, ERA5 full
variable names, *_log variables back-transformed to physical units. A glob matching several zarrs (e.g. one per GPU) is
concatenated along init_time. Only start times present for every model are evaluated, so all models share the samples.

Targets are built like utils.dataset builds them: accumulated *_6hr variables are summed over the lead time
(accumulation_offsets_h), all other variables are taken at the valid time.

Metric: MAE weighted by cos(latitude) / mean(cos(latitude)) over grid cells where the target is not NaN, as in the
per-variable entries of utils.losses.WeightedMAELoss (<var>_val in W&B). Atmospheric variables per level and pooled over
levels (level "all", like <var>_val); tp and r also in log space (space "log", transform_data), as the model sees them.

Output: long-format CSV with columns model, init_time, lead_time_h, variable, level, space, mae.
"""
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from utils.transform_data import accumulation_offsets_h, transform_data

WB = "/capstor/store/cscs/swissai/weatherbench"
STORES = {  # ERA5 store of each predicted variable; the rest are in weatherbench2_original
    "accumulated": f"{WB}/ERA5_surface_accumulated_6hr.zarr",
    "soil": f"{WB}/weatherbench2_additionalvariables.zarr",
    "original": f"{WB}/weatherbench2_original",
}
NON_VARS = {"init_time", "lead_time", "ensemble", "level", "latitude", "longitude"}
# variables predicted as *_log (saved back-transformed), with the short name transform_data knows them by
LOG_VARS = {"total_precipitation_6hr": "tp_6hr_log", "runoff_6hr": "r_6hr_log"}


def store_of(full_name):
    if full_name.endswith("_6hr"):
        return "accumulated"
    if full_name.startswith("volumetric_soil_water_layer"):
        return "soil"
    return "original"


def open_predictions(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction zarr matches {pattern}")
    ds = xr.concat([xr.open_zarr(p) for p in paths], dim="init_time") if len(paths) > 1 else xr.open_zarr(paths[0])
    if not np.issubdtype(ds.lead_time.dtype, np.timedelta64):  # hours as numbers
        ds = ds.assign_coords(lead_time=pd.to_timedelta(ds.lead_time.values, unit="h"))
    return ds.sortby("init_time")


def latitude_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def weighted_mae(pred, target, w_lat):
    """Latitude-weighted MAE over the last two axes (lat, lon) and any leading ones, ignoring NaN targets."""
    err = np.abs(pred - target) * w_lat[:, None]
    return float(np.nanmean(np.where(np.isnan(target), np.nan, err)))


def target_field(stores, full_name, valid_time, lead_h, lat, lon, levels):
    """ERA5 target as utils.dataset builds it: a lead-time sum for accumulated variables, else the value at valid_time."""
    da = stores[store_of(full_name)][full_name]
    offsets = accumulation_offsets_h(full_name, lead_h)  # full names work: accumulated 6 h vars end in _6hr
    times = [valid_time - pd.Timedelta(hours=o) for o in offsets]
    da = da.sel(time=times).sel(latitude=lat, longitude=lon, method="nearest", tolerance=1e-6)
    if "level" in da.dims:
        da = da.sel(level=levels)
    return da.values.sum(axis=0)


def errors_for_sample(stores, preds, init_time, lead_time):
    """All error rows of one start time and lead time, for every model."""
    any_pred = next(iter(preds.values()))
    lat, lon, levels = any_pred.latitude.values, any_pred.longitude.values, any_pred.level.values
    w_lat = latitude_weights(lat)
    valid_time = pd.Timestamp(init_time) + pd.Timedelta(lead_time)
    lead_h = int(pd.Timedelta(lead_time) / pd.Timedelta(hours=1))
    variables = sorted(set().union(*[{v for v in p.data_vars if v not in NON_VARS} for p in preds.values()]))

    def load_target(v):
        return v, target_field(stores, v, valid_time, lead_h, lat, lon, levels)

    with ThreadPoolExecutor(8) as pool:
        targets = dict(pool.map(load_target, variables))

    rows = []
    for model, ds in preds.items():
        sample = ds.sel(init_time=init_time, lead_time=lead_time)
        for v in variables:
            if v not in sample:
                continue
            pred = sample[v].mean("ensemble").values if "ensemble" in sample[v].dims else sample[v].values
            target = targets[v]
            spaces = {"phys": (pred, target)}
            if v in LOG_VARS:  # tp and r: also in the log space the model predicts in
                spaces["log"] = (transform_data(pred, LOG_VARS[v]), transform_data(target, LOG_VARS[v]))
            for space, (p, t) in spaces.items():
                row = dict(model=model, init_time=init_time, lead_time_h=lead_h, variable=v, space=space)
                if "level" in sample[v].dims:
                    rows += [dict(row, level=str(lev), mae=weighted_mae(p[i], t[i], w_lat)) for i, lev in enumerate(levels)]
                    rows.append(dict(row, level="all", mae=weighted_mae(p, t, w_lat)))
                else:
                    rows.append(dict(row, level="", mae=weighted_mae(p, t, w_lat)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pred", action="append", required=True, help="NAME=ZARR_OR_GLOB, once per model")
    parser.add_argument("--out", required=True, help="CSV file for the per-sample errors")
    parser.add_argument("--max_inits", type=int, default=None, help="Evaluate only the first N common start times (tests)")
    args = parser.parse_args()

    preds = dict(p.split("=", 1) for p in args.pred)
    preds = {name: open_predictions(pattern) for name, pattern in preds.items()}
    inits = sorted(set.intersection(*[set(ds.init_time.values) for ds in preds.values()]))
    leads = sorted(set.intersection(*[set(ds.lead_time.values) for ds in preds.values()]))
    for name, ds in preds.items():
        print(f"{name}: {ds.sizes['init_time']} start times, lead times {[str(pd.Timedelta(l)) for l in ds.lead_time.values]}")
    inits = inits[:args.max_inits]
    print(f"Evaluating {len(inits)} common start times x {len(leads)} common lead times")

    stores = {k: xr.open_zarr(path, chunks=None) for k, path in STORES.items()}
    rows = []
    for i, init_time in enumerate(inits):
        for lead_time in leads:
            rows += errors_for_sample(stores, preds, init_time, lead_time)
        print(f"start time {i + 1}/{len(inits)}: {pd.Timestamp(init_time)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    summary = df[df.level.isin(["", "all"])].groupby(["variable", "space", "lead_time_h", "model"]).mae.mean().unstack("model")
    print(summary.to_string(float_format="{:.4g}".format))


if __name__ == "__main__":
    main()
