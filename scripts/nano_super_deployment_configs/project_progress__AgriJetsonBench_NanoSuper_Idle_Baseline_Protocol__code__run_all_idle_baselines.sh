#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="both"
HAS_REBOOT_FLAG=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      ARGS+=("$1")
      shift
      ;;
    --require-reboot-after-mode-set|--no-reboot-after-mode-set)
      HAS_REBOOT_FLAG=1
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$MODE" != "15W" && "$MODE" != "25W" && "$MODE" != "both" ]]; then
  echo "[ERROR] --mode must be one of: 15W, 25W, both" >&2
  exit 2
fi

if [[ "$MODE" == "both" ]]; then
  echo "[WARN] Running both modes in one boot. For reboot-clean paper runs, use --mode 15W and --mode 25W separately." >&2
  python3 "${SCRIPT_DIR}/run_idle_baseline_protocol.py" --mode both "${ARGS[@]}"
else
  if [[ "$HAS_REBOOT_FLAG" == "0" ]]; then
    ARGS+=(--require-reboot-after-mode-set)
  fi
  python3 "${SCRIPT_DIR}/run_idle_baseline_protocol.py" "${ARGS[@]}"
fi
