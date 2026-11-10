#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
PI_LOGGER_URL="${PI_LOGGER_URL:-http://${LOGGER_DASHBOARD_HOST}:8000}"
HOLD_S="${HOLD_S:-20}"
PROFILE="${NOISE_GUARD_PROFILE:-safe}"
PYBIN="${PYBIN:-python}"
"$PKG_ROOT/scripts/run_with_sudo_keepalive.sh" \
  "$PYBIN" "$PKG_ROOT/jetson/noise_guard.py" \
  --preserve-url "$PI_LOGGER_URL" \
  --profile "$PROFILE" \
  --hold-s "$HOLD_S"
