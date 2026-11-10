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
    start_keepalive
  elif [[ "${SUDO_PROMPT_ONCE}" == "1" && -t 0 && -t 1 ]]; then
    show_oled_message "SUDO NEEDED" "ENTER PASSWORD" "once for run"
    echo "[INFO] Sudo password is needed once so quiet-mode can disable/restore Wi-Fi, Bluetooth, services, and display state."
    if sudo -v; then
      echo "[INFO] Sudo credentials cached for this benchmark session."
      show_oled_message "PASSWORD OK" "starting run" "quiet-mode ready"
      start_keepalive
    else
      show_oled_message "SUDO FAILED" "check terminal" "quiet-mode paused"
      if [[ "${SUDO_REQUIRE_SUCCESS}" == "1" ]]; then
        echo "[ERROR] sudo -v failed. Aborting because SUDO_REQUIRE_SUCCESS=1." >&2
        exit 1
      fi
      echo "[WARN] sudo -v failed. Continuing without sudo keepalive; noise-guard will run best-effort without privileged actions." >&2
    fi
  else
    if [[ "${SUDO_REQUIRE_SUCCESS}" == "1" ]]; then
      echo "[ERROR] Manual privileged-command access is unavailable and no interactive TTY is available for a one-time sudo prompt." >&2
      exit 1
    fi
    echo "[WARN] Manual privileged-command access not available. Continuing without sudo keepalive; noise-guard will run best-effort without privileged actions." >&2
    echo "[HINT] To install the limited manual privileged-command access rules, run once: bash scripts/install_jetson_benchmark_privileged-command policy.sh" >&2
  fi
fi

"$@"
