#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SLURM_SCRIPT="$PACKAGE_DIR/scripts/prepare_runtime.slurm"

PARTITION="batch"
TIME_LIMIT="02:00:00"
CPUS="4"
MEMORY="16G"
VENV_PATH="/home/$USER/envs/segment"
CONFIG="$PACKAGE_DIR/config/segmentation_deeplabv3plus_train_lock.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

echo "[INFO] Submitting prepare job partition=$PARTITION time=$TIME_LIMIT cpus=$CPUS mem=$MEMORY"

sbatch \
  --partition="$PARTITION" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",VENV_PATH="$VENV_PATH" \
  "$SLURM_SCRIPT"
