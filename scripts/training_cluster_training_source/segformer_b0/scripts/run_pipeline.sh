#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$DEFAULT_PACKAGE_DIR}"
if [[ $# -gt 0 && -d "$1" ]]; then
  PACKAGE_DIR="$1"
  shift
fi
MODE="${1:-status}"
CONFIG="${2:-$PACKAGE_DIR/config/segmentation_segformer_b0_train_lock.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/segment}"

source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
python "$PACKAGE_DIR/scripts/stage2_segformer_b0_pipeline.py" --config "$CONFIG" --mode "$MODE"
