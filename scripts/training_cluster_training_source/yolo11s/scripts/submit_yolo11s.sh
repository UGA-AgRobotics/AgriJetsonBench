#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster}"
SLURM_SCRIPT="$PACKAGE_DIR/scripts/yolo11s_train.slurm"

MODE="full"
PARTITION="gpu_p"
GRES="gpu:A100:1"
TIME_LIMIT=""
CPUS="32"
MEMORY=""
VENV_PATH="/home/$USER/envs/yolo11"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"
CONFIG="$PACKAGE_DIR/config/detection_yolo11s_train_lock.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --pytorch-module) PYTORCH_MODULE="$2"; shift 2 ;;
    --torchvision-module) TORCHVISION_MODULE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
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
    TIME_LIMIT="72:00:00"
  fi
fi

if [[ -z "$MEMORY" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    MEMORY="48G"
  else
    MEMORY="64G"
  fi
fi

echo "[INFO] Submitting mode=$MODE partition=$PARTITION gres=$GRES time=$TIME_LIMIT cpus=$CPUS mem=$MEMORY"

sbatch \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",PIPELINE_MODE="$MODE",VENV_PATH="$VENV_PATH",PYTORCH_MODULE="$PYTORCH_MODULE",TORCHVISION_MODULE="$TORCHVISION_MODULE" \
  "$SLURM_SCRIPT"
