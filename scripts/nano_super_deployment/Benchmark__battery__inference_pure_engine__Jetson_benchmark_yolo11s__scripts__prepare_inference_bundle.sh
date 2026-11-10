#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <experiment_root_with_inference_bundle> <out_root>" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
python "$PKG_ROOT/scripts/prepare_inference_bundle.py" --experiment-root "$1" --out-root "$2"
