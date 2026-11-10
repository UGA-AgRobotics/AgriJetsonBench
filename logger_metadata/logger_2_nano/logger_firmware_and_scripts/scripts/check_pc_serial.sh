#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

echo "Expected QT Py port: ${PC_SENSOR_PORT:-/dev/ttyACM0}"
echo "Nano QT Py port: ${PC_SENSOR_PORT_NANO:-${NANO_SENSOR_PORT:-unset}}"
echo "AGX QT Py port: ${PC_SENSOR_PORT_AGX:-${AGX_SENSOR_PORT:-unset}}"
echo "Port map: ${PC_SENSOR_PORTS:-${SENSOR_PORTS:-unset}}"
echo "--- serial devices ---"
ls -l /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
echo "--- user groups ---"
id

sample_port() {
  local label="$1"
  local port="$2"
  local seconds="${3:-5}"
  [[ -n "$port" ]] || return 0
  echo "--- sample lines: ${label} (${port}) ---"
  stty -F "$port" 115200 raw -echo 2>/dev/null || true
  timeout "$seconds" cat "$port" || true
}

if [[ -n "${PC_SENSOR_PORTS:-${SENSOR_PORTS:-}}" ]]; then
  raw_map="${PC_SENSOR_PORTS:-${SENSOR_PORTS:-}}"
  IFS=',;' read -ra items <<<"$raw_map"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ "$item" == *=* ]] || continue
    label="${item%%=*}"
    port="${item#*=}"
    sample_port "$label" "$port" "${1:-5}"
  done
else
  sample_port "nano" "${PC_SENSOR_PORT_NANO:-${NANO_SENSOR_PORT:-}}" "${1:-5}"
  sample_port "agx" "${PC_SENSOR_PORT_AGX:-${AGX_SENSOR_PORT:-}}" "${1:-5}"
  if [[ -z "${PC_SENSOR_PORT_NANO:-${NANO_SENSOR_PORT:-}}" && -z "${PC_SENSOR_PORT_AGX:-${AGX_SENSOR_PORT:-}}" ]]; then
    sample_port "default" "$PC_SENSOR_PORT" "${1:-5}"
  fi
fi
