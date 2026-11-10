#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
OLED_BUS="${OLED_BUS:-7}"
OLED_ADDR="${OLED_ADDR:-0x3C}"
SCENARIO="${SCENARIO:-ok}"
exec python "$PKG_ROOT/tools/jetson_oled_dl_tester.py" --bus "$OLED_BUS" --addr "$OLED_ADDR" --scenario "$SCENARIO"
