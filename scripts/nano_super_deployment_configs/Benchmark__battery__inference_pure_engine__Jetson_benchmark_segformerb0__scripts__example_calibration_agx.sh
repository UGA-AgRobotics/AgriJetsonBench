#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
PI_LOGGER_URL="${PI_LOGGER_URL:-http://${LOGGER_DASHBOARD_HOST}:8000}"
OUT_ROOT="${OUT_ROOT:-$CALIBRATION_OUT_ROOT}"
DATASET_INPUT="${DATASET_INPUT:-$DATASET_SOURCE}"
PYBIN="${PYBIN:-python}"
RUN_WRAPPER="$PKG_ROOT/scripts/run_with_sudo_keepalive.sh"
NOISE_GUARD_ENABLE="${NOISE_GUARD_ENABLE:-1}"
NOISE_GUARD_PROFILE="${NOISE_GUARD_PROFILE:-safe}"
NOISE_GUARD_DISPLAY_OFF="${NOISE_GUARD_DISPLAY_OFF:-1}"
NOISE_GUARD_STOP_SERVICES="${NOISE_GUARD_STOP_SERVICES:-1}"
NOISE_GUARD_DISABLE_WIFI="${NOISE_GUARD_DISABLE_WIFI:-1}"
NOISE_GUARD_DISABLE_BLUETOOTH="${NOISE_GUARD_DISABLE_BLUETOOTH:-1}"
for SC in idle cpu gpu mixed; do
  CMD=("$PYBIN" "$PKG_ROOT/jetson/run_calibration.py"
    --scenario "$SC"
    --duration-s 180
    --run-name agx_native_calibration
    --board-name "Jetson AGX Orin 64GB"
    --output-root "$OUT_ROOT"
    --logger-url "$PI_LOGGER_URL"
    --model "$MODEL_PATH"
    --dataset-root "$DATASET_INPUT")
  if [[ "$NOISE_GUARD_ENABLE" == "1" ]]; then
    CMD+=(--noise-guard --noise-guard-profile "$NOISE_GUARD_PROFILE")
    if [[ "$NOISE_GUARD_DISPLAY_OFF" != "1" ]]; then CMD+=(--no-noise-guard-display-off); fi
    if [[ "$NOISE_GUARD_STOP_SERVICES" != "1" ]]; then CMD+=(--no-noise-guard-stop-services); fi
    if [[ "$NOISE_GUARD_DISABLE_WIFI" != "1" ]]; then CMD+=(--no-noise-guard-disable-wifi); fi
    if [[ "$NOISE_GUARD_DISABLE_BLUETOOTH" != "1" ]]; then CMD+=(--no-noise-guard-disable-bluetooth); fi
  else
    CMD+=(--no-noise-guard)
  fi
  QUIET_WRAP_ENABLE="$NOISE_GUARD_ENABLE" "$RUN_WRAPPER" "${CMD[@]}"
done
"$PYBIN" "$PKG_ROOT/analysis/analyze_calibration.py" --suite-root "$OUT_ROOT/agx_native_calibration"
