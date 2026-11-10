#!/usr/bin/env bash
set -euo pipefail
OLED_ENABLE=1
export OLED_ENABLE
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/example_run_agx_native.sh" "$@"
