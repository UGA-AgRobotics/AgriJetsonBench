#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "[SUDO] Enter your password once now. This wrapper keeps sudo alive until the run exits."
sudo -v

keepalive_pid=""
cleanup() {
  if [[ -n "${keepalive_pid}" ]] && kill -0 "${keepalive_pid}" >/dev/null 2>&1; then
    kill "${keepalive_pid}" >/dev/null 2>&1 || true
    wait "${keepalive_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

while true; do
  sudo -n -v >/dev/null 2>&1 || exit 1
  sleep 60
done &
keepalive_pid="$!"

python3 scripts/run_battery_inference_agx.py --no-sudo-prompt-once "$@"
