#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
PID_FILE="${RUNTIME_DIR}/battery_inference_sudo_keepalive_${USER}.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[OK] no sudo keepalive pid file found."
  exit 0
fi

pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
rm -f "${PID_FILE}"

if [[ -z "${pid}" ]]; then
  echo "[OK] empty pid file removed."
  exit 0
fi

if kill -0 "${pid}" >/dev/null 2>&1; then
  kill "${pid}" >/dev/null 2>&1 || true
  echo "[OK] stopped sudo keepalive pid=${pid}"
else
  echo "[OK] sudo keepalive was not running."
fi
