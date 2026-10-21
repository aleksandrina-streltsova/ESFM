#!/bin/bash
# Analysis of the 2021 evaluation predictions: errors against ERA5, the land water balance and PhysMetrics'
# atmospheric checks. CPU only and outside the container (it uses the esfm conda env), on a compute node:
#   srun -A ab016 --time=00:30:00 --pty bash        # a new allocation, or a second shell in a running job:
#   srun --jobid=<JOBID> --overlap --pty bash       # (without --environment, the container isn't needed)
#   bash run_analysis_eval2021.sh                    # N_STARTS=2 bash run_analysis_eval2021.sh for a quick try
# where JOBID is the first column of `squeue --me`.
#
# Models whose predictions are missing are skipped. The steps run in parallel, each with its own log:
#   errors           - evaluate_era5_errors.py: lat-weighted MAE per start time, variable and level, all models
#                      together on the start times they share
#   land_<model>     - physmetrics-land: E* = P - R - dS, for the hydro fine-tunes (the models predicting land hydrology)
#   physics_<model>  - physmetrics-run: spectra, lapse rates, hydrostatic and geostrophic balance at 24 h
# Outputs in $OUT:
#   errors_eval2021.csv, land_water_balance_<model>.csv, physmetrics/<model>/*.csv, logs/<step>_<MMDD-HHMM>.log

PRED="/iopsstor/scratch/cscs/$USER/eval_hydro_2021"
OUT=${OUT:-$PRED/analysis}
N_STARTS=${N_STARTS:-112}           # start times to evaluate, out of the 112 of era5_*_eval2021
WORKERS=${WORKERS:-12}              # worker processes per step; up to about 4 GB of memory each
STAMP=$(date +%m%d-%H%M)
ERA5="/capstor/store/cscs/swissai/weatherbench/weatherbench2_original"

declare -A PREDS=(
    [esfmS_hydro_24h_ft]="$PRED/esfmS_hydro_24h_ft/predictions*_step*.zarr"
    [auroraS_hydro_24h_ft]="$PRED/auroraS_hydro_24h_ft/part*/rollout_pred_*.zarr"
    [ESFM_s_nm_ft]="$PRED/ESFM_s_nm_ft/predictions*_step*.zarr"
    [auroraS_pretrained]="$PRED/auroraS_pretrained/predictions*_step*.zarr"
)
MODELS="esfmS_hydro_24h_ft auroraS_hydro_24h_ft ESFM_s_nm_ft auroraS_pretrained"
HYDRO_MODELS="esfmS_hydro_24h_ft auroraS_hydro_24h_ft"

cd /users/$USER/ESFM
export PATH="/users/$USER/miniconda3/envs/esfm/bin:$PATH"
export OMP_NUM_THREADS=1            # the parallelism comes from the worker processes
which python                        # should be ~/miniconda3/envs/esfm/bin/python
nproc                               # CPU cores this shell may use; should be well above the workers started
mkdir -p "$OUT/logs"

present=()
for m in $MODELS; do
    if compgen -G "${PREDS[$m]}" > /dev/null; then present+=("$m"); else echo "$m: no predictions at ${PREDS[$m]}, skipped"; fi
done
# the start times of era5_*_eval2021 in dataset_config.yaml: every 78 h from 2021-01-01T00
starts=($(for i in $(seq 0 $((N_STARTS - 1))); do date -u -d "2021-01-01 00:00 UTC + $((i * 78)) hours" +%Y-%m-%dT%H; done))
echo "Models: ${present[*]}; ${#starts[@]} start times; $WORKERS workers per step"

logs=(); pids=()
run() {  # run <step> <command...>: start the command in the background with its own log
    local log="$OUT/logs/$1_$STAMP.log"; shift
    "$@" > "$log" 2>&1 &
    logs+=("$log"); pids+=($!)
}

pred_args=()
for m in "${present[@]}"; do pred_args+=(--pred "$m=${PREDS[$m]}"); done
run errors python evaluate_era5_errors.py "${pred_args[@]}" --out "$OUT/errors_eval2021.csv" \
    --max_inits $N_STARTS --workers $WORKERS

for m in "${present[@]}"; do
    if [[ " $HYDRO_MODELS " == *" $m "* ]]; then
        run land_$m physmetrics-land --prediction-zarr "${PREDS[$m]}" --model $m \
            --output "$OUT/land_water_balance_$m.csv" --dates "${starts[@]}" --workers $WORKERS
    fi
    run physics_$m physmetrics-run --year 2021 --model $m --prediction-zarr "${PREDS[$m]}" --ref-zarr "$ERA5" \
        --dates "${starts[@]}" --lead-times 24h --workers $WORKERS --output-dir "$OUT/physmetrics/$m"
done

echo "Started ${#pids[@]} steps, logs in $OUT/logs/*_$STAMP.log"
for i in "${!pids[@]}"; do
    wait ${pids[$i]} && echo "done:   ${logs[$i]}" || echo "FAILED: ${logs[$i]}"
done
