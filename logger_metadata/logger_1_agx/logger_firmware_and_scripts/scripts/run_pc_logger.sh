#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
SERIAL_PORT="${PC_SENSOR_PORT:-/dev/ttyACM0}"
SENSOR_PORT_MAP="${PC_SENSOR_PORTS:-${SENSOR_PORTS:-}}"
PORT="${PORT:-8000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PC_LOGGER_OUTPUT_ROOT}"
LOGGER_SERVER="$PKG_ROOT/pi_logger/logger_server.py"
PC_LOGGER_RESTART="${PC_LOGGER_RESTART:-0}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

port_is_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

list_port_pids() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp "sport = :${port}" 2>/dev/null \
      | grep -oE 'pid=[0-9]+' \
      | cut -d= -f2 \
      | sort -u || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | sort -u || true
  fi
}

read_cmdline() {
  local pid="$1"
  tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null \
    || ps -p "$pid" -o command= 2>/dev/null \
    || true
}

if port_is_listening "$PORT"; then
  mapfile -t LISTEN_PIDS < <(list_port_pids "$PORT")
  LOGGER_PIDS=()
  for pid in "${LISTEN_PIDS[@]}"; do
    cmdline="$(read_cmdline "$pid")"
    if [[ "$cmdline" == *"$LOGGER_SERVER"* ]]; then
      LOGGER_PIDS+=("$pid")
    fi
  done

  if (( ${#LOGGER_PIDS[@]} > 0 )); then
    if is_truthy "$PC_LOGGER_RESTART"; then
      echo "[INFO] Restarting existing PC logger on port ${PORT} (pid(s): ${LOGGER_PIDS[*]})."
      kill "${LOGGER_PIDS[@]}"
      for _ in {1..50}; do
        if ! port_is_listening "$PORT"; then
          break
        fi
        sleep 0.1
      done
      if port_is_listening "$PORT"; then
        echo "[ERROR] Existing PC logger did not release port ${PORT}." >&2
        exit 1
      fi
    else
      echo "[INFO] PC logger is already running on port ${PORT} (pid ${LOGGER_PIDS[0]})."
      echo "[INFO] Dashboard: http://127.0.0.1:${PORT}/"
      echo "[INFO] Wired dashboard: http://${LOGGER_DASHBOARD_HOST}:${PORT}/"
      if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[INFO] Health check: OK"
      fi
      echo "[INFO] To reload updated code, run: PC_LOGGER_RESTART=1 bash scripts/run_pc_logger.sh"
      exit 0
    fi
  fi

  if port_is_listening "$PORT"; then
    echo "[ERROR] Port ${PORT} is already in use by another process:" >&2
    if (( ${#LISTEN_PIDS[@]} > 0 )); then
      for pid in "${LISTEN_PIDS[@]}"; do
        echo "  pid ${pid}: $(read_cmdline "$pid")" >&2
      done
    else
      echo "  pid unavailable; try: ss -ltnp 'sport = :${PORT}'" >&2
    fi
    echo "[ERROR] Stop that process or start this logger on another port, for example:" >&2
    echo "  PORT=8001 bash scripts/run_pc_logger.sh" >&2
    exit 1
  fi
fi

activate_pi_env
echo "[INFO] Dashboard: http://127.0.0.1:${PORT}/"
echo "[INFO] Wired dashboard: http://${LOGGER_DASHBOARD_HOST}:${PORT}/"
cmd=(python "$PKG_ROOT/pi_logger/logger_server.py")
if [[ -n "$SENSOR_PORT_MAP" ]]; then
  echo "[INFO] Sensor port map: ${SENSOR_PORT_MAP}"
  cmd+=(--sensor-port-map "$SENSOR_PORT_MAP")
else
  if [[ -n "${PC_SENSOR_PORT_NANO:-}" || -n "${PC_SENSOR_PORT_AGX:-}" || -n "${NANO_SENSOR_PORT:-}" || -n "${AGX_SENSOR_PORT:-}" ]]; then
    echo "[INFO] Sensor ports from PC_SENSOR_PORT_NANO/PC_SENSOR_PORT_AGX env."
  else
    echo "[INFO] Single/default sensor port: ${SERIAL_PORT}"
    cmd+=(--serial-port "$SERIAL_PORT")
  fi
fi
cmd+=(--port "$PORT" --output-root "$OUTPUT_ROOT" --fast-temp-sensor-on-init)
exec "${cmd[@]}"
