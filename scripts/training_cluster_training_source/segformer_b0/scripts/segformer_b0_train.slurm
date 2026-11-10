#!/bin/bash
#SBATCH --job-name=segformer_b0_cropweed
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:L4:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG="${CONFIG:-$PACKAGE_DIR/config/segmentation_segformer_b0_train_lock.yaml}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/segment}"
SEGFORMER_B0_REQUESTED_GPUS="${SEGFORMER_B0_REQUESTED_GPUS:-4}"
SEGFORMER_B0_NODE_COUNT="${SEGFORMER_B0_NODE_COUNT:-${SLURM_JOB_NUM_NODES:-1}}"
SEGFORMER_B0_DDP_LAUNCHER="${SEGFORMER_B0_DDP_LAUNCHER:-torchrun}"

if [[ "$SEGFORMER_B0_REQUESTED_GPUS" =~ ^[0-9]+$ ]] && (( SEGFORMER_B0_REQUESTED_GPUS > 1 )); then
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / SEGFORMER_B0_REQUESTED_GPUS ))
  if (( OMP_NUM_THREADS < 1 )); then
    export OMP_NUM_THREADS=1
  fi
else
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi

echo "[INFO] Hostname     : $(hostname)"
echo "[INFO] Job ID       : ${SLURM_JOB_ID:-none}"
echo "[INFO] Package      : $PACKAGE_DIR"
echo "[INFO] Config       : $CONFIG"
echo "[INFO] Mode         : $PIPELINE_MODE"
echo "[INFO] Venv         : $VENV_PATH"
echo "[INFO] Nodes        : $SEGFORMER_B0_NODE_COUNT"
echo "[INFO] GPUs/node    : $SEGFORMER_B0_REQUESTED_GPUS"
echo "[INFO] Total GPUs   : $(( SEGFORMER_B0_NODE_COUNT * SEGFORMER_B0_REQUESTED_GPUS ))"
echo "[INFO] OMP threads  : $OMP_NUM_THREADS"

cd "$PACKAGE_DIR"
source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
export SEGFORMER_B0_REQUESTED_GPUS
export SEGFORMER_B0_NODE_COUNT
export SEGFORMER_B0_DDP_LAUNCHER
export SEGFORMER_B0_DIST_BACKEND="${SEGFORMER_B0_DIST_BACKEND:-nccl}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
if [[ "${SEGFORMER_B0_NODE_COUNT:-1}" =~ ^[0-9]+$ ]] && (( SEGFORMER_B0_NODE_COUNT <= 1 )); then
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  unset NCCL_SOCKET_IFNAME
else
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
fi
export SEGFORMER_B0_DIST_TIMEOUT_SEC="${SEGFORMER_B0_DIST_TIMEOUT_SEC:-600}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-$(( 15000 + (${SLURM_JOB_ID:-0} % 20000) ))}"
echo "[INFO] MASTER_ADDR  : $MASTER_ADDR"
echo "[INFO] MASTER_PORT  : $MASTER_PORT"

python "$PACKAGE_DIR/scripts/stage2_segformer_b0_pipeline.py" --config "$CONFIG" --mode "$PIPELINE_MODE"
