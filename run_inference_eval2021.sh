#!/bin/bash
# 2021 evaluation inference of the models compared with the hydro fine-tune, from a shell that is already inside
# the container on a GPU node:
#   srun --container-writable --environment=/users/$USER/.edf/torchcontainer_clariden.toml -A ab016 --time=02:00:00 --pty bash
# then one model per run:
#   bash run_inference_eval2021.sh esfmS_hydro_24h_ft   # ESFM-small hydro fine-tune, direct 24 h forecasts
#   bash run_inference_eval2021.sh ESFM_s_nm_ft         # pretrained ESFM-small, 4 x 6 h rollout, 24 h saved
#   bash run_inference_eval2021.sh auroraS_pretrained   # pretrained Aurora-small, 4 x 6 h rollout, 24 h saved
# The AuroraS hydro fine-tune runs in the old code: cd ../auroraS_hydro_24h/code && bash run_inference_auroraS_hydro_eval2021.sh
#
# Open a second shell in a running job
#   srun --jobid=<JOBID> --overlap --container-writable --environment=/users/$USER/.edf/torchcontainer_clariden.toml --pty bash
#   tail -f logs/inference_eval2021_<MODEL>_<MMDD-HHMM>.log # the new shell can't attach to the old one, so follow the run through its log
# where JOBID is the first column of `squeue --me`
#
# Start times: era5_*_eval2021 in dataset_config.yaml (112 starts in 2021, every 78 h).
# Outputs:
# logs/                                       - console output of each run, inference_eval2021_<MODEL>_<MMDD-HHMM>.log
# log_dir (configs/config_eval2021_<MODEL>.yaml) - predictions_val_set_step-1.zarr (inference.py) or
#                                               predictions_rollout_val_set_step*.zarr (inference_rollout.py), about 36 GB;
#                                               the per-start pkl files are deleted once the zarr holds all of them
# utils/                                      - cached sample indices forecast_pairs_*.pkl, reused by runs over the same period
# auroraS_pretrained reads the weights from ~/.cache/huggingface without network access; download them once on the login node:
#   python -c "from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/aurora', 'aurora-0.25-small-pretrained.ckpt')"

MODEL=${1:-$MODEL}
case "$MODEL" in
    esfmS_hydro_24h_ft)
        SCRIPT="inference.py --use_val_set_from_config" ;;
    ESFM_s_nm_ft|auroraS_pretrained)
        SCRIPT="inference_rollout.py --use_val_set_from_config --no-use_inds_subset --NUM_ROLLOUT_STEPS 4 --lead_times_save 24" ;;
    *)
        echo "Unknown MODEL '$MODEL', choose esfmS_hydro_24h_ft, ESFM_s_nm_ft or auroraS_pretrained"; exit 1 ;;
esac
CONFIG="configs/config_eval2021_$MODEL.yaml"
LOG="logs/inference_eval2021_${MODEL}_$(date +%m%d-%H%M).log"

cd /users/$USER/ESFM
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1 # print() goes to the log file immediately, not in chunks
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
export HF_HOME="/users/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1   # use the cached Aurora weights, compute nodes may have no internet

nvidia-smi -L                       # should list 4 GH200 GPUs
env -u OMP_NUM_THREADS nproc        # CPU cores this shell may use; should be well above 4 (nproc reports OMP_NUM_THREADS if set)
which torchrun                      # should be the container's, not ~/miniconda3/...
ls -d $HF_HOME/hub/models--microsoft--aurora  # needed for auroraS_pretrained only
set -o pipefail                     # the exit status of torchrun, not of tee, decides whether the pkl files are deleted
torchrun --standalone --nproc_per_node=4 $SCRIPT \
    --config "$CONFIG" \
    --num_nodes 1 \
    2>&1 | tee "$LOG" &&
python scripts/inference/remove_merged_tmp_predictions.py \
    "$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['log_dir'])")" \
    2>&1 | tee -a "$LOG"
