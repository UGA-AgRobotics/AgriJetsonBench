#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
DATASET_INPUT="${DATASET_INPUT:-$DATASET_SOURCE}"
PI_LOGGER_URL="${PI_LOGGER_URL:-http://${LOGGER_DASHBOARD_HOST}:8000}"
OUT_ROOT="${OUT_ROOT:-$BENCHMARK_OUT_ROOT}"
NVP_MODE="${NVP_MODE:-}"
OLED_ENABLE="${OLED_ENABLE:-1}"
OLED_BUS="${OLED_BUS:-}"
OLED_ADDR="${OLED_ADDR:-}"
PYBIN="${PYBIN:-python}"
RUN_WRAPPER="$PKG_ROOT/scripts/run_with_sudo_keepalive.sh"
NOISE_GUARD_ENABLE="${NOISE_GUARD_ENABLE:-1}"
NOISE_GUARD_PROFILE="${NOISE_GUARD_PROFILE:-safe}"
NOISE_GUARD_DISPLAY_OFF="${NOISE_GUARD_DISPLAY_OFF:-1}"
NOISE_GUARD_STOP_SERVICES="${NOISE_GUARD_STOP_SERVICES:-1}"
NOISE_GUARD_DISABLE_WIFI="${NOISE_GUARD_DISABLE_WIFI:-1}"
NOISE_GUARD_DISABLE_BLUETOOTH="${NOISE_GUARD_DISABLE_BLUETOOTH:-1}"

CMD=("$PYBIN" "$PKG_ROOT/jetson/run_benchmark.py"
  --model "$MODEL_PATH"
  --dataset-root "$DATASET_INPUT"
  --run-name "agx_orin_segformerb0_fp16_native_proposal"
  --board-name "Jetson AGX Orin 64GB"
  --output-root "$OUT_ROOT"
  --device 0
  --repeats 3
  --warmup-images 32
  --warmup-seconds 300
  --min-images 500
  --min-duration-s 180
  --pi-logger-url "$PI_LOGGER_URL"
  --expect-external-logger
  --external-sample-hz 10
  --tegrastats-interval-ms 200
  --internal-interval-ms 200
  --start-delay-s 5
  --power-mode-label native)
if [[ -n "$NVP_MODE" ]]; then
  CMD+=(--nvpmodel-mode "$NVP_MODE")
fi
if [[ "$OLED_ENABLE" == "1" ]]; then
  CMD+=(--oled-enable)
fi
if [[ -n "$OLED_BUS" ]]; then
  CMD+=(--oled-bus "$OLED_BUS")
fi
if [[ -n "$OLED_ADDR" ]]; then
  CMD+=(--oled-addr "$OLED_ADDR")
fi
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
"$PYBIN" "$PKG_ROOT/analysis/merge_and_summarize.py" --run-root "$OUT_ROOT/agx_orin_segformerb0_fp16_native_proposal"
