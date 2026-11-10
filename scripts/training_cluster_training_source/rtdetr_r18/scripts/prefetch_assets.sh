#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${1:-$HOME/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster}"
VENV_PATH="${2:-/home/$USER/envs/SAM}"

source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
python "$PACKAGE_DIR/scripts/ensure_rtdetr_repo.py" --package-dir "$PACKAGE_DIR"
python "$PACKAGE_DIR/scripts/ensure_model_weights.py" --package-dir "$PACKAGE_DIR"
