#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLED_MESSAGE_TOOL="$SCRIPT_DIR/../tools/show_oled_message.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command> [args ...]" >&2
  exit 1
fi

KEEPALIVE_SECONDS="${SUDO_KEEPALIVE_SECONDS:-240}"
QUIET_WRAP_ENABLE="${QUIET_WRAP_ENABLE:-1}"
SUDO_PROMPT_ONCE="${SUDO_PROMPT_ONCE:-1}"
SUDO_REQUIRE_SUCCESS="${SUDO_REQUIRE_SUCCESS:-0}"
KEEPALIVE_PID=""
SUDO_NONINTERACTIVE_OK=0

resolve_status_update_url() {
  if [[ -n "${STATUS_UPDATE_URL:-}" ]]; then
    printf '%s' "$STATUS_UPDATE_URL"
    return 0
  fi
  local base_url="${PI_LOGGER_URL:-${LOGGER_URL:-${PC_LOGGER_URL:-}}}"
  case "${base_url,,}" in
    ""|0|false|no|none|off)
      return 0
      ;;
  esac
  printf '%s/api/jetson_status' "${base_url%/}"
}

post_sudo_status() {
  local stage="$1"
  local state="$2"
  local message="$3"
  local url
  url="$(resolve_status_update_url)"
  if [[ -z "$url" ]]; then
    return 0
  fi
  "${PYBIN:-python}" - "$url" "$stage" "$state" "$message" <<'PY' >/dev/null 2>&1 || true
import json
import os
import sys
import time
import urllib.request

url, stage, state, message = sys.argv[1:5]
payload = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "wallclock_ns": time.time_ns(),
    "state": state,
    "stage": stage,
    "message": message,
    "board_name": os.environ.get("BOARD_NAME"),
    "run_id": os.environ.get("SESSION_NAME") or None,
    "session_name": os.environ.get("SESSION_NAME") or None,
    "operation_mode": os.environ.get("SESSION_WORK_MODE") or None,
    "sudo": {
        "stage": stage,
        "password_required": stage == "sudo_password_required",
        "noninteractive_available": stage in {"sudo_ready_noninteractive", "sudo_ready_cached"},
    },
}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=3):
    pass
PY
}

show_oled_message() {
  if [[ "${OLED_ENABLE:-0}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$OLED_MESSAGE_TOOL" ]]; then
    return 0
  fi
  local pybin="${PYBIN:-python}"
  local cmd=("$pybin" "$OLED_MESSAGE_TOOL" --line1 "${1:-}" --line2 "${2:-}" --line3 "${3:-}")
  if [[ -n "${OLED_BUS:-}" ]]; then
    cmd+=(--bus "$OLED_BUS")
  fi
  if [[ -n "${OLED_ADDR:-}" ]]; then
    cmd+=(--addr "$OLED_ADDR")
  fi
  "${cmd[@]}" >/dev/null 2>&1 || true
}

start_keepalive() {
  SUDO_NONINTERACTIVE_OK=1
  (
    while true; do
      sleep "${KEEPALIVE_SECONDS}"
      sudo -n true >/dev/null 2>&1 || exit 0
    done
  ) >/dev/null 2>&1 &
  KEEPALIVE_PID="$!"
}

cleanup() {
  if [[ -n "${KEEPALIVE_PID}" ]]; then
    kill "${KEEPALIVE_PID}" >/dev/null 2>&1 || true
    wait "${KEEPALIVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${QUIET_WRAP_ENABLE}" == "1" ]]; then
  if sudo -n true >/dev/null 2>&1; then
    echo "[INFO] Using existing non-interactive sudo for noise-guard actions."
    post_sudo_status "sudo_ready_noninteractive" "running" "Manual privileged-command access is ready for noise-guard actions."
    start_keepalive
  elif [[ "${SUDO_PROMPT_ONCE}" == "1" && -t 0 && -t 1 ]]; then
    show_oled_message "SUDO NEEDED" "ENTER PASSWORD" "once for run"
    post_sudo_status "sudo_password_required" "waiting" "Sudo password required once before benchmark can start."
    echo "[INFO] Sudo password is needed once so quiet-mode can disable/restore Wi-Fi, Bluetooth, services, and display state."
    if sudo -v; then
      echo "[INFO] Sudo credentials cached for this benchmark session."
      show_oled_message "PASSWORD OK" "starting run" "quiet-mode ready"
      post_sudo_status "sudo_ready_cached" "running" "Sudo password accepted; noise-guard can continue."
      start_keepalive
    else
      show_oled_message "SUDO FAILED" "check terminal" "quiet-mode paused"
      post_sudo_status "sudo_password_failed" "error" "Sudo password failed; benchmark cannot get privileged quiet-mode actions."
      if [[ "${SUDO_REQUIRE_SUCCESS}" == "1" ]]; then
        echo "[ERROR] sudo -v failed. Aborting because SUDO_REQUIRE_SUCCESS=1." >&2
        exit 1
      fi
      echo "[WARN] sudo -v failed. Continuing without sudo keepalive; noise-guard will run best-effort without privileged actions." >&2
    fi
  else
    if [[ "${SUDO_REQUIRE_SUCCESS}" == "1" ]]; then
      post_sudo_status "sudo_password_required" "error" "Manual privileged-command access is unavailable and no interactive terminal is available for a password prompt."
      echo "[ERROR] Manual privileged-command access is unavailable and no interactive TTY is available for a one-time sudo prompt." >&2
      exit 1
    fi
    post_sudo_status "sudo_unavailable" "warning" "Manual privileged-command access is unavailable; noise-guard will run best-effort without privileged actions."
    echo "[WARN] Manual privileged-command access not available. Continuing without sudo keepalive; noise-guard will run best-effort without privileged actions." >&2
    echo "[HINT] Baraye manual privileged-command access-ye mahdood, yekbar bezan: bash scripts/install_jetson_benchmark_privileged-command policy.sh" >&2
  fi
fi

"$@"
