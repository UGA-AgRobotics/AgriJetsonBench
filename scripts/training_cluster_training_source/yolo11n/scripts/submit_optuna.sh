#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster}"
SLURM_SCRIPT="$PACKAGE_DIR/scripts/yolo11n_optuna.slurm"

PARTITION="gpu_p"
GRES="gpu:L4:1"
TIME_LIMIT="72:00:00"
CPUS="8"
MEMORY="48G"
VENV_PATH="/home/$USER/envs/SAM"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"
CONFIG="$PACKAGE_DIR/config/detection_yolo11n_train_lock.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
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

echo "[INFO] Submitting Optuna job partition=$PARTITION gres=$GRES time=$TIME_LIMIT cpus=$CPUS mem=$MEMORY"

sbatch \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",VENV_PATH="$VENV_PATH",PYTORCH_MODULE="$PYTORCH_MODULE",TORCHVISION_MODULE="$TORCHVISION_MODULE" \
  "$SLURM_SCRIPT"
