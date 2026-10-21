#!/bin/bash
# Test run with config_test.yaml, from a shell that is already inside the container on a GPU node:
#   srun --container-writable --environment=/users/$USER/.edf/torchcontainer_clariden.toml -A ab016 --time=01:00:00 --pty bash
#
# Open a second shell in a running job
#   srun --jobid=<JOBID> --overlap --container-writable --environment=/users/$USER/.edf/torchcontainer_clariden.toml --pty bash
#   tail -f logs/test_<MMDD-HHMM>.log # the new shell can't attach to the old one, so follow the run through its log
# where JOBID is the first column of `squeue --me`
#
# Outputs:
# logs/                               - console output of each run, test_<MMDD-HHMM>.log
# log_dir (set in config_test.yaml)   - checkpoints: model_ckpt-step=*, model_ckpt-epoch=*, model_best_val_ckpt-*, last*.ckpt
# log_dir/<YYYYmmdd-HHMMSS>/          - config copy, args.json and log.txt of each start
# log_dir/wandb/                      - local W&B run files, synced to wandb.ai/astreltsova-ethz/ESFM_GHM
# /iopsstor/scratch/cscs/$USER/wandb/ - W&B cache, config and data
# utils/                              - cached sample indices forecast_pairs_*.pkl, reused by runs over the same period

cd /users/$USER/ESFM
export WANDB__SERVICE_WAIT=300
wandb_dir="/iopsstor/scratch/cscs/$USER/wandb"
export WANDB_CACHE_DIR="$wandb_dir/cache"
export WANDB_ARTIFACT_LOCATION="$wandb_dir/artifact_location"
export WANDB_ARTIFACT_DIR="$wandb_dir/artifact_dir"
export WANDB_CONFIG_DIR="$wandb_dir/config"
export WANDB_DATA_DIR="$wandb_dir/data_dir"
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000

nvidia-smi -L                       # should list 4 GH200 GPUs
env -u OMP_NUM_THREADS nproc        # CPU cores this shell may use; should be well above 4 (nproc reports OMP_NUM_THREADS if set)
echo "WANDB_KEY ${WANDB_KEY:+is set}"  # without "is set", train.py won't log to W&B
which torchrun                      # should be the container's, not ~/miniconda3/...
torchrun --standalone --nproc_per_node=4 train.py \
    --config ./config_test.yaml \
    --num_nodes 1 \
    --wnb_id "ESFMs_hydro_test_$(date +%Y-%m-%d--%H-%M-%S)" \
    2>&1 | tee logs/test_$(date +%m%d-%H%M).log
