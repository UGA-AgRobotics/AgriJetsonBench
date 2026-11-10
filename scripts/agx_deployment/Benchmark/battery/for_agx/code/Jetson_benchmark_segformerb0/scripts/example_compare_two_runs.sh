#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <run_root_1> <run_root_2> <outdir>" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
python "$PKG_ROOT/analysis/compare_runs.py" --run-root "$1" --run-root "$2" --outdir "$3"
