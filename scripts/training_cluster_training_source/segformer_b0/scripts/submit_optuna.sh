#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OPTUNA_SLURM="$PACKAGE_DIR/scripts/segformer_b0_optuna.slurm"

PARTITION="gpu_p"
GPU_TYPE="L4"
GPU_COUNT="4"
NODE_COUNT="1"
GRES=""
NODELIST=""
TIME_LIMIT="168:00:00"
CPUS="32"
MEMORY="128G"
VENV_PATH="/home/$USER/envs/segment"
CONFIG="$PACKAGE_DIR/config/segmentation_segformer_b0_train_lock.yaml"
AUTO_FULL="yes"

extract_gpu_count() {
  local spec="$1"
  local tail="${spec##*:}"
  if [[ "$tail" =~ ^[0-9]+$ ]]; then
    echo "$tail"
    return
  fi
  if [[ "$spec" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo "1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --gpus) GPU_COUNT="$2"; shift 2 ;;
    --nodes) NODE_COUNT="$2"; shift 2 ;;
    --nodelist) NODELIST="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --auto-full) AUTO_FULL="$2"; shift 2 ;;
    --no-auto-full) AUTO_FULL="no"; shift 1 ;;
    --full-partition|--full-gres|--full-time|--full-cpus|--full-mem)
      echo "[WARN] $1 is ignored in single-job mode."
      shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$GRES" ]]; then
  GRES="gpu:${GPU_TYPE}:${GPU_COUNT}"
fi

REQUESTED_GPUS="$(extract_gpu_count "$GRES")"
if ! [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] || (( NODE_COUNT < 1 )); then
  echo "[ERROR] --nodes must be a positive integer."
  exit 1
fi
TOTAL_GPUS=$(( NODE_COUNT * REQUESTED_GPUS ))

if [[ "$AUTO_FULL" == "yes" ]]; then
  echo "[INFO] Submitting one SegFormer-B0 job for Optuna -> full training partition=$PARTITION nodes=$NODE_COUNT gres=$GRES gpus_per_node=$REQUESTED_GPUS total_gpus=$TOTAL_GPUS time=$TIME_LIMIT cpus_per_node=$CPUS mem_per_node=$MEMORY"
else
  echo "[INFO] Submitting one SegFormer-B0 Optuna-only job partition=$PARTITION nodes=$NODE_COUNT gres=$GRES gpus_per_node=$REQUESTED_GPUS total_gpus=$TOTAL_GPUS time=$TIME_LIMIT cpus_per_node=$CPUS mem_per_node=$MEMORY"
fi

SUBMIT_OUTPUT=$(sbatch \
  --partition="$PARTITION" \
  --nodes="$NODE_COUNT" \
  --ntasks="$NODE_COUNT" \
  --ntasks-per-node=1 \
  ${NODELIST:+--nodelist="$NODELIST"} \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",VENV_PATH="$VENV_PATH",CHAIN_FULL="$AUTO_FULL",FULL_PIPELINE_MODE="full",SEGFORMER_B0_REQUESTED_GPUS="$REQUESTED_GPUS",SEGFORMER_B0_NODE_COUNT="$NODE_COUNT",SEGFORMER_B0_DDP_LAUNCHER="torchrun" \
  "$OPTUNA_SLURM")

echo "$SUBMIT_OUTPUT"
JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
if [[ -z "$JOB_ID" ]]; then
  echo "[ERROR] Could not parse job ID from sbatch output."
  exit 1
fi

if [[ "$AUTO_FULL" == "yes" ]]; then
  echo "[INFO] Single job ID submitted: $JOB_ID"
  echo "[INFO] The same job will run Optuna first, then start full training automatically in the same allocation."
else
  echo "[INFO] Optuna-only job ID: $JOB_ID"
fi
