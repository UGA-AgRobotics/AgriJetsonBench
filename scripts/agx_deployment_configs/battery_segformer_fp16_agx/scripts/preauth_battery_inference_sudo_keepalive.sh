#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
PID_FILE="${RUNTIME_DIR}/battery_inference_sudo_keepalive_${USER}.pid"
LOG_FILE="${RUNTIME_DIR}/battery_inference_sudo_keepalive_${USER}.log"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    echo "[OK] sudo keepalive is already running: pid=${old_pid}"
    exit 0
  fi
fi

echo "[SUDO] Enter your password once now. A keepalive will refresh sudo until you stop it."
sudo -v

(
  while true; do
    sudo -n -v || exit 1
    sleep 60
  done
) >"${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "[OK] sudo keepalive started: pid=${pid}"
echo "[OK] pid file: ${PID_FILE}"
echo "[OK] log file: ${LOG_FILE}"
echo "[NEXT] Stop later with: bash scripts/stop_battery_inference_sudo_keepalive.sh"
