#!/bin/bash
#SBATCH --job-name=ESFMs_hydro      # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node (torchrun spawns 1 process per GPU)
#SBATCH --gpus-per-node=4           # GPUs per node, must match `devices` in config.yaml
#SBATCH -c 72                       # CPU cores per task
#SBATCH --mem=460000                # memory per node
#SBATCH --exclusive
#SBATCH --time=07:00:00             # total run time (HH:MM:SS)
#SBATCH --account=ab016
#SBATCH --output=logs/train_esfm_small_hydro_24h_%j.out  # output log file

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
export WANDB__SERVICE_WAIT=300
wandb_dir="/iopsstor/scratch/cscs/$USER/wandb"
export WANDB_CACHE_DIR="$wandb_dir/cache"
export WANDB_ARTIFACT_LOCATION="$wandb_dir/artifact_location"
export WANDB_ARTIFACT_DIR="$wandb_dir/artifact_dir"
export WANDB_CONFIG_DIR="$wandb_dir/config"
export WANDB_DATA_DIR="$wandb_dir/data_dir"
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1 # print() goes to the log file immediately, not in chunks
ulimit -c 0  # disable core dumps
ulimit -t unlimited
export NCCL_CROSS_NIC=1
# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501

node_rank=$SLURM_NODEID
nnodes=$SLURM_JOB_NUM_NODES

echo "================= SLURM Info ================="
echo "Run started at:       $(date)"
echo "SLURM_JOB_ID:         $SLURM_JOB_ID"
echo "SLURM_NODELIST:       $SLURM_JOB_NODELIST"
echo "MASTER_ADDR:          $MASTER_ADDR"
echo "MASTER_PORT:          $MASTER_PORT"
echo "Number of nodes:= "   $SLURM_JOB_NUM_NODES
echo "node_rank:            $node_rank"
echo "Ntasks per node:= "   $SLURM_NTASKS_PER_NODE
echo "================================================"
set -x

workdir="/users/$USER/ESFM"
export WORKDIR=${WORKDIR:-$workdir}
tomlpath="/users/$USER/.edf/torchcontainer_clariden.toml"

export CONFIG=${CONFIG:-./config.yaml} # CONFIG=./config_test.yaml for a test run
# If first run, use a new wnb_id; if resuming, add --resume and set WNB_ID to the wnb_id of the previous run
export WNB_ID=${WNB_ID:-"ESFMs_hydro_$(date +"%Y-%m-%d--%H-%M-%S")"}

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="6000"
export NCCL_BLOCKING_WAIT="1"
export NCCL_TIMEOUT="6000"

# Disable eager messages to avoid NCCL timeouts
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0


srun --ntasks=$nnodes \
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
      train.py \
        --config "$CONFIG" \
        --num_nodes $SLURM_JOB_NUM_NODES \
        --wnb_name "ESFMs_hydro_24h/$SLURM_JOB_NUM_NODES" \
        --wnb_id "$WNB_ID" \
        2>&1
'

echo "Run finished at: $(date)"
