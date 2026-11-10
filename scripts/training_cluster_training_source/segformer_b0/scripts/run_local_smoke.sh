#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-$PACKAGE_DIR/config/segmentation_segformer_b0_local_subset100.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/segment}"
FAKE_ROOT="$PACKAGE_DIR/runtime_local/fake_cropandweed"

source "$SCRIPT_DIR/local_runtime_env.sh"
sanitize_runtime_env
source "$SCRIPT_DIR/activate_stage2_env.sh" "$VENV_PATH"

if [[ ! -d "$FAKE_ROOT/images/train" ]]; then
  python "$PACKAGE_DIR/scripts/segformer_b0_lib.py" --make-fake-dataset "$FAKE_ROOT" >/dev/null
fi

cd "$PACKAGE_DIR"
exec python "$PACKAGE_DIR/scripts/stage2_segformer_b0_pipeline.py" --config "$CONFIG_PATH" --mode smoke
