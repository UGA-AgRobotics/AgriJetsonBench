#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PACKAGE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG_PATH=${1:-$PACKAGE_DIR/config/detection_rtdetr_r18_local_subset100.yaml}

. "$SCRIPT_DIR/local_runtime_env.sh"
sanitize_runtime_env

export PACKAGE_DIR
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$PACKAGE_DIR/scripts/python_compat:$PACKAGE_DIR/scripts:$PYTHONPATH"
else
  export PYTHONPATH="$PACKAGE_DIR/scripts/python_compat:$PACKAGE_DIR/scripts"
fi

cd "$PACKAGE_DIR"
exec python "$PACKAGE_DIR/scripts/optuna_tune_rtdetr_r18.py" --config "$CONFIG_PATH"
