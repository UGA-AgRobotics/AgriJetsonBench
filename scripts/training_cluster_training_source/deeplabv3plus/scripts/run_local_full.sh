#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-$PACKAGE_DIR/config/segmentation_deeplabv3plus_local_subset100.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/segment}"
FAKE_ROOT="$PACKAGE_DIR/runtime_local/fake_cropandweed"

source "$SCRIPT_DIR/local_runtime_env.sh"
sanitize_runtime_env
source "$SCRIPT_DIR/activate_stage2_env.sh" "$VENV_PATH"

if [[ ! -d "$FAKE_ROOT/images/train" ]]; then
  python "$PACKAGE_DIR/scripts/deeplabv3plus_lib.py" --make-fake-dataset "$FAKE_ROOT" >/dev/null
fi

cd "$PACKAGE_DIR"
exec python "$PACKAGE_DIR/scripts/stage2_deeplabv3plus_pipeline.py" --config "$CONFIG_PATH" --mode full
