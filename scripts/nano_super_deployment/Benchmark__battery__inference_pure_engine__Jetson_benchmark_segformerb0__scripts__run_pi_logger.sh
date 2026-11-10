#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_pi_env
I2C_BUS="${I2C_BUS:-auto}"
PORT="${PORT:-8000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PI_LOGGER_OUTPUT_ROOT}"
exec python "$PKG_ROOT/pi_logger/logger_server.py" --i2c-bus "$I2C_BUS" --port "$PORT" --output-root "$OUTPUT_ROOT" --fast-temp-sensor-on-init
