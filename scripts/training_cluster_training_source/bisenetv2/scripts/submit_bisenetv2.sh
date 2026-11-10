#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SLURM_SCRIPT="$PACKAGE_DIR/scripts/bisenetv2_train.slurm"

MODE="full"
PARTITION="gpu_p"
GPU_TYPE="L4"
GPU_COUNT="4"
NODE_COUNT="1"
GRES=""
NODELIST=""
TIME_LIMIT=""
CPUS="32"
MEMORY=""
VENV_PATH="/home/$USER/envs/segment"
CONFIG="$PACKAGE_DIR/config/segmentation_bisenetv2_train_lock.yaml"

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
    --mode) MODE="$2"; shift 2 ;;
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
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "[ERROR] --mode must be smoke or full"
  exit 1
fi

if [[ -z "$TIME_LIMIT" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    TIME_LIMIT="12:00:00"
  else
    TIME_LIMIT="168:00:00"
  fi
fi

if [[ -z "$MEMORY" ]]; then
  MEMORY="96G"
fi

if [[ -z "$GRES" ]]; then
  GRES="gpu:${GPU_TYPE}:${GPU_COUNT}"
fi

REQUESTED_GPUS="$(extract_gpu_count "$GRES")"
if ! [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] || (( NODE_COUNT < 1 )); then
  echo "[ERROR] --nodes must be a positive integer."
  exit 1
fi
TOTAL_GPUS=$(( NODE_COUNT * REQUESTED_GPUS ))

echo "[INFO] Submitting mode=$MODE partition=$PARTITION nodes=$NODE_COUNT gres=$GRES gpus_per_node=$REQUESTED_GPUS total_gpus=$TOTAL_GPUS time=$TIME_LIMIT cpus_per_node=$CPUS mem_per_node=$MEMORY"

sbatch \
  --partition="$PARTITION" \
  --nodes="$NODE_COUNT" \
  --ntasks="$NODE_COUNT" \
  --ntasks-per-node=1 \
  ${NODELIST:+--nodelist="$NODELIST"} \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",PIPELINE_MODE="$MODE",VENV_PATH="$VENV_PATH",BISENETV2_REQUESTED_GPUS="$REQUESTED_GPUS",BISENETV2_NODE_COUNT="$NODE_COUNT",BISENETV2_DDP_LAUNCHER="torchrun",BISENETV2_OPT_FOREACH="0" \
  "$SLURM_SCRIPT"
