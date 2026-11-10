#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEEDS_SUDO_CHECK=1
for arg in "$@"; do
  case "${arg}" in
    --merge-only|-h|--help)
      NEEDS_SUDO_CHECK=0
      ;;
  esac
done

if [[ "${NEEDS_SUDO_CHECK}" == "1" ]]; then
  echo "[SUDO] using command-specific manual privileged-command access checks"
  sudo -n nvpmodel -q >/dev/null
fi

cd "${ROOT_DIR}"
exec python3 "${SCRIPT_DIR}/run_idle_baseline_agx.py" --root "${ROOT_DIR}" "$@"
