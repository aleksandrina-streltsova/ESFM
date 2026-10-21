"""Delete the per-sample pred_*.pkl files of an inference run once its merged zarr holds all of their init times.

inference.py and inference_rollout.py write one pkl per start time to <log_dir>/tmp_predictions*/ and merge them into
<log_dir>/predictions*_step*.zarr, but keep the pkl files, which doubles the storage of a run.

Usage: python scripts/inference/remove_merged_tmp_predictions.py <log_dir>
"""
import glob
import os
import shutil
import sys

import xarray as xr


def remove_merged_tmp_predictions(log_dir):
    for zarr_path in sorted(glob.glob(os.path.join(log_dir, "predictions*_step*.zarr"))):
        # predictions{suffix}_step{N}.zarr is merged from tmp_predictions{suffix}/ (see utils/merge_predictions.py)
        suffix = os.path.basename(zarr_path)[len("predictions"):].rsplit("_step", 1)[0]
        tmp_dir = os.path.join(log_dir, f"tmp_predictions{suffix}")
        pkl_files = glob.glob(os.path.join(tmp_dir, "pred_*.pkl"))
        if not pkl_files:
            continue
        n_merged = xr.open_zarr(zarr_path).sizes["init_time"]
        if n_merged != len(pkl_files):
            print(f"Keeping {tmp_dir}: {len(pkl_files)} pkl files, but {n_merged} init times in {zarr_path}")
            continue
        shutil.rmtree(tmp_dir)
        print(f"Removed {tmp_dir}: all {len(pkl_files)} start times are in {zarr_path}")


if __name__ == "__main__":
    remove_merged_tmp_predictions(sys.argv[1])
