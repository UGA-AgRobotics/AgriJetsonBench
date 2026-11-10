#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${1:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster}"
MODE="${2:-status}"
CONFIG="${3:-$PACKAGE_DIR/config/detection_yolo11s_train_lock.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/yolo11}"

source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
python "$PACKAGE_DIR/scripts/stage2_yolo11s_pipeline.py" --config "$CONFIG" --mode "$MODE"
