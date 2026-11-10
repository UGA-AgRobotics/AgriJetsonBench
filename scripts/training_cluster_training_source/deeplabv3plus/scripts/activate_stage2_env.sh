#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
VENV_PATH="${1:-/home/$USER/envs/segment}"
SEGMENT_ACTIVATE_SCRIPT="${SEGMENT_ACTIVATE_SCRIPT:-$REPO_ROOT/segment_env/activate_env.sh}"

die() {
  echo "[ERROR] $*" >&2
  return 1 2>/dev/null || exit 1
}

append_pythonpath() {
  local path="$1"
  [ -d "$path" ] || return 0
  if [ -n "${PYTHONPATH:-}" ]; then
    case ":$PYTHONPATH:" in
      *":$path:"*) ;;
      *) export PYTHONPATH="$path:$PYTHONPATH" ;;
    esac
  else
    export PYTHONPATH="$path"
  fi
}

if [ ! -f "$SEGMENT_ACTIVATE_SCRIPT" ]; then
  die "Shared segmentation activation script not found at $SEGMENT_ACTIVATE_SCRIPT"
fi

# Force the shared activation helper to stay inside the shared `segment` env
# instead of inheriting a stale base site-packages path from an older env.
if [ -z "${BASE_SITE_PACKAGES:-}" ]; then
  shopt -s nullglob
  env_site_packages_candidates=( "$VENV_PATH"/lib/python*/site-packages )
  shopt -u nullglob
  if [ ${#env_site_packages_candidates[@]} -gt 0 ]; then
    export BASE_SITE_PACKAGES="${env_site_packages_candidates[0]}"
  fi
fi

source "$SEGMENT_ACTIVATE_SCRIPT" "$VENV_PATH"
export PACKAGE_DIR
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

append_pythonpath "$PACKAGE_DIR/scripts/python_compat"
append_pythonpath "$PACKAGE_DIR/scripts"
append_pythonpath "$PACKAGE_DIR/vendor"

echo "[INFO] Active Python : $(command -v python)"
echo "[INFO] VIRTUAL_ENV   : ${VIRTUAL_ENV:-unset}"
echo "[INFO] PACKAGE_DIR   : $PACKAGE_DIR"
