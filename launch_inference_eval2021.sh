#!/bin/bash
#SBATCH --job-name=eval2021          # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node (torchrun spawns 1 process per GPU)
#SBATCH --gpus-per-node=4           # GPUs per node, must match `devices` in the config
#SBATCH -c 72                       # CPU cores per task
#SBATCH --mem=460000                # memory per node
#SBATCH --exclusive
#SBATCH --time=02:00:00             # total run time (HH:MM:SS)
#SBATCH --account=ab016
#SBATCH --output=logs/inference_eval2021_%j.out  # output log file

# 2021 evaluation inference of the models compared with the hydro fine-tune; MODEL selects the model:
#   MODEL=esfmS_hydro_24h_ft sbatch launch_inference_eval2021.sh   # ESFM-small hydro fine-tune, direct 24 h forecasts
#   MODEL=ESFM_s_nm_ft       sbatch launch_inference_eval2021.sh   # pretrained ESFM-small, 4 x 6 h rollout, 24 h saved
#   MODEL=auroraS_pretrained sbatch launch_inference_eval2021.sh   # pretrained Aurora-small, 4 x 6 h rollout, 24 h saved
# The AuroraS hydro fine-tune runs in the old code: ../auroraS_hydro_24h/code/launch_inference_auroraS_hydro_eval2021.sh
#
# Start times: era5_*_eval2021 in dataset_config.yaml (112 starts in 2021, every 78 h).
# Outputs in log_dir of configs/config_eval2021_$MODEL.yaml:
#   predictions_val_set_step-1.zarr (inference.py) or predictions_rollout_val_set_step*.zarr (inference_rollout.py),
#   about 36 GB; the per-start pkl files are deleted once the zarr holds all of them.
# auroraS_pretrained reads the weights from ~/.cache/huggingface without network access; download them once on the login node:
#   python -c "from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/aurora', 'aurora-0.25-small-pretrained.ckpt')"

case "$MODEL" in
    esfmS_hydro_24h_ft)
        SCRIPT="inference.py --use_val_set_from_config" ;;
    ESFM_s_nm_ft|auroraS_pretrained)
        SCRIPT="inference_rollout.py --use_val_set_from_config --no-use_inds_subset --NUM_ROLLOUT_STEPS 4 --lead_times_save 24" ;;
    *)
        echo "Unknown MODEL '$MODEL', choose esfmS_hydro_24h_ft, ESFM_s_nm_ft or auroraS_pretrained"; exit 1 ;;
esac
export SCRIPT
export CONFIG="configs/config_eval2021_$MODEL.yaml"

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1 # print() goes to the log file immediately, not in chunks
export HF_HOME="/users/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1   # use the cached Aurora weights, compute nodes may have no internet
ulimit -c 0  # disable core dumps
ulimit -t unlimited
export NCCL_CROSS_NIC=1
# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501

echo "================= SLURM Info ================="
echo "Run started at:       $(date)"
echo "SLURM_JOB_ID:         $SLURM_JOB_ID"
echo "SLURM_NODELIST:       $SLURM_JOB_NODELIST"
echo "MODEL:                $MODEL"
echo "CONFIG:               $CONFIG"
echo "SCRIPT:               $SCRIPT"
echo "================================================"
set -x

workdir="/users/$USER/ESFM"
export WORKDIR=${WORKDIR:-$workdir}
tomlpath="/users/$USER/.edf/torchcontainer_clariden.toml"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="6000"
export TORCH_NCCL_BLOCKING_WAIT="1"
export NCCL_TIMEOUT="6000"

# Disable eager messages to avoid NCCL timeouts
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

srun --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --cpus-per-task=72 \
     --export=ALL \
     --environment=$tomlpath \
     -u -l \
  bash -c '
    cd $WORKDIR
    set -x

    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$SLURM_PROCID \
      --nproc_per_node=4 \
      --rdzv_id=42 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      $SCRIPT \
        --config "$CONFIG" \
        --num_nodes $SLURM_JOB_NUM_NODES \
        2>&1 &&
    python scripts/inference/remove_merged_tmp_predictions.py \
      "$(python -c "import yaml; print(yaml.safe_load(open(\"$CONFIG\"))[\"log_dir\"])")"
'

echo "Run finished at: $(date)"
