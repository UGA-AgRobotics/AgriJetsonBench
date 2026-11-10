#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
activate_local_tensorrt_stack

PYBIN="${PYBIN:-python}"
RUN_WRAPPER="$PKG_ROOT/scripts/run_with_sudo_keepalive.sh"
POWER_HELPER="$PKG_ROOT/jetson/power_mode_utils.py"
MERGE_SUMMARY="$PKG_ROOT/analysis/merge_and_summarize.py"
COMPARE_RUNS="$PKG_ROOT/analysis/compare_runs.py"
SESSION_NOISE_GUARD="$PKG_ROOT/jetson/run_with_noise_guard.py"
OLED_MESSAGE_TOOL="$PKG_ROOT/tools/show_oled_message.py"
BUILD_ALL_ENGINES_SCRIPT="$PKG_ROOT/scripts/build_engines_int8_fp16_fp32.sh"
BACKFILL_PRECISION_ACCURACY_SCRIPT="$PKG_ROOT/scripts/backfill_precision_specific_accuracy.py"
PT_ACCURACY_MODEL_PATH="${PT_ACCURACY_MODEL_PATH:-}"

BOARD_NAME="${BOARD_NAME:-Jetson AGX Orin 64GB}"
DATASET_INPUT="${DATASET_INPUT:-$DATASET_SOURCE}"
PI_LOGGER_URL="${PI_LOGGER_URL:-http://${LOGGER_DASHBOARD_HOST}:8000}"
DEFAULT_STATUS_UPDATE_URL=""
if [[ -n "$PI_LOGGER_URL" ]]; then
  DEFAULT_STATUS_UPDATE_URL="${PI_LOGGER_URL%/}/api/jetson_status"
fi
STATUS_UPDATE_URL="${STATUS_UPDATE_URL:-$DEFAULT_STATUS_UPDATE_URL}"
OUT_ROOT="${OUT_ROOT:-$BENCHMARK_OUT_ROOT}"
FP16_MODEL_PATH="${FP16_MODEL_PATH:-$MODEL_DIR/best_fp16.engine}"
FP32_MODEL_PATH="${FP32_MODEL_PATH:-$MODEL_DIR/best_fp32.engine}"
INT8_MODEL_PATH="${INT8_MODEL_PATH:-$MODEL_DIR/best_int8.engine}"
INT8_ENABLE="${INT8_ENABLE:-auto}"
AUTO_BUILD_MISSING_ENGINES="${AUTO_BUILD_MISSING_ENGINES:-1}"
RUN_PRECISIONS="${RUN_PRECISIONS:-${RERUN_PRECISIONS:-${PRECISION_FILTER:-all}}}"
RUN_ACCURACY_MODE="${RUN_ACCURACY_MODE:-precision_specific_engine}"
RUN_ACCURACY_MODEL_OVERRIDE="${RUN_ACCURACY_MODEL_OVERRIDE:-}"
if [[ -n "$RUN_ACCURACY_MODEL_OVERRIDE" ]]; then
  RUN_ACCURACY_MODEL="$RUN_ACCURACY_MODEL_OVERRIDE"
elif [[ "$RUN_ACCURACY_MODE" == "precision_specific_engine" ]]; then
  RUN_ACCURACY_MODEL=""
elif [[ -f "$PT_ACCURACY_MODEL_PATH" ]]; then
  RUN_ACCURACY_MODEL="$PT_ACCURACY_MODEL_PATH"
else
  RUN_ACCURACY_MODEL="$ONNX_MODEL_PATH"
fi

REPORT_MODE="${REPORT_MODE:-1}"
REPORT_REPEATS_MIN="${REPORT_REPEATS_MIN:-3}"
REPORT_WARMUP_IMAGES_MIN="${REPORT_WARMUP_IMAGES_MIN:-32}"
REPORT_WARMUP_SECONDS_MIN="${REPORT_WARMUP_SECONDS_MIN:-300}"
REPORT_MIN_IMAGES_MIN="${REPORT_MIN_IMAGES_MIN:-500}"
REPORT_MIN_DURATION_S_MIN="${REPORT_MIN_DURATION_S_MIN:-180}"

REPEATS="${REPEATS:-3}"
WARMUP_IMAGES="${WARMUP_IMAGES:-32}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MIN_IMAGES="${MIN_IMAGES:-500}"
MIN_DURATION_S="${MIN_DURATION_S:-180}"
EXTERNAL_SAMPLE_HZ="${EXTERNAL_SAMPLE_HZ:-10}"
TEGRASTATS_INTERVAL_MS="${TEGRASTATS_INTERVAL_MS:-200}"
INTERNAL_INTERVAL_MS="${INTERNAL_INTERVAL_MS:-200}"
START_DELAY_S="${START_DELAY_S:-5}"
DEVICE="${DEVICE:-0}"
EXPECT_EXTERNAL_LOGGER="${EXPECT_EXTERNAL_LOGGER:-1}"
OLED_ENABLE="${OLED_ENABLE:-1}"
OLED_BUS="${OLED_BUS:-}"
OLED_ADDR="${OLED_ADDR:-}"
SKIP_COMPLETED_PRECISIONS="${SKIP_COMPLETED_PRECISIONS:-1}"
NOISE_GUARD_ENABLE="${NOISE_GUARD_ENABLE:-1}"
NOISE_GUARD_PROFILE="${NOISE_GUARD_PROFILE:-safe}"
NOISE_GUARD_DISPLAY_OFF="${NOISE_GUARD_DISPLAY_OFF:-1}"
NOISE_GUARD_STOP_SERVICES="${NOISE_GUARD_STOP_SERVICES:-1}"
NOISE_GUARD_DISABLE_WIFI="${NOISE_GUARD_DISABLE_WIFI:-1}"
NOISE_GUARD_DISABLE_BLUETOOTH="${NOISE_GUARD_DISABLE_BLUETOOTH:-1}"
SESSION_TIMESTAMP="${SESSION_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_CURRENT_MODE_FP16_FP32_PHASE="${RUN_CURRENT_MODE_FP16_FP32_PHASE:-outer}"
SESSION_NOISE_GUARD_ACTIVE="${SESSION_NOISE_GUARD_ACTIVE:-0}"
SESSION_WORK_MODE="${SESSION_WORK_MODE:-unknown}"
RUN_VARIANTS="${RUN_VARIANTS:-standard}"
YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$OUT_ROOT/ultralytics_config}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_ROOT/matplotlib_config}"
ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS="${ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS:-1}"

export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS

mkdir -p "$OUT_ROOT" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"

numeric_lt() {
  "$PYBIN" - "$1" "$2" <<'PY'
import sys
left = float(sys.argv[1])
right = float(sys.argv[2])
raise SystemExit(0 if left < right else 1)
PY
}

promote_report_floor() {
  local var_name="$1"
  local current_value="$2"
  local minimum_value="$3"
  if [[ "$REPORT_MODE" != "1" ]]; then
    printf '%s' "$current_value"
    return 0
  fi
  if numeric_lt "$current_value" "$minimum_value"; then
    echo "[INFO] REPORT_MODE=1 -> promoted $var_name from $current_value to $minimum_value." >&2
    printf '%s' "$minimum_value"
  else
    printf '%s' "$current_value"
  fi
}

REPEATS="$(promote_report_floor REPEATS "$REPEATS" "$REPORT_REPEATS_MIN")"
WARMUP_IMAGES="$(promote_report_floor WARMUP_IMAGES "$WARMUP_IMAGES" "$REPORT_WARMUP_IMAGES_MIN")"
WARMUP_SECONDS="$(promote_report_floor WARMUP_SECONDS "$WARMUP_SECONDS" "$REPORT_WARMUP_SECONDS_MIN")"
MIN_IMAGES="$(promote_report_floor MIN_IMAGES "$MIN_IMAGES" "$REPORT_MIN_IMAGES_MIN")"
MIN_DURATION_S="$(promote_report_floor MIN_DURATION_S "$MIN_DURATION_S" "$REPORT_MIN_DURATION_S_MIN")"

export PYBIN BOARD_NAME DATASET_INPUT PI_LOGGER_URL OUT_ROOT FP16_MODEL_PATH FP32_MODEL_PATH INT8_MODEL_PATH INT8_ENABLE RUN_ACCURACY_MODEL RUN_ACCURACY_MODE
export REPORT_MODE REPORT_REPEATS_MIN REPORT_WARMUP_IMAGES_MIN REPORT_WARMUP_SECONDS_MIN REPORT_MIN_IMAGES_MIN REPORT_MIN_DURATION_S_MIN
export REPEATS WARMUP_IMAGES WARMUP_SECONDS MIN_IMAGES MIN_DURATION_S EXTERNAL_SAMPLE_HZ TEGRASTATS_INTERVAL_MS INTERNAL_INTERVAL_MS START_DELAY_S DEVICE EXPECT_EXTERNAL_LOGGER
export OLED_ENABLE OLED_BUS OLED_ADDR SKIP_COMPLETED_PRECISIONS NOISE_GUARD_ENABLE NOISE_GUARD_PROFILE NOISE_GUARD_DISPLAY_OFF NOISE_GUARD_STOP_SERVICES NOISE_GUARD_DISABLE_WIFI NOISE_GUARD_DISABLE_BLUETOOTH
export SESSION_TIMESTAMP RUN_CURRENT_MODE_FP16_FP32_PHASE SESSION_NOISE_GUARD_ACTIVE SESSION_WORK_MODE RUN_VARIANTS YOLO_CONFIG_DIR MPLCONFIGDIR ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS
export AUTO_BUILD_MISSING_ENGINES
export RUN_PRECISIONS
export STATUS_UPDATE_URL

PRECISION_FILTER_ACTIVE="0"
SELECTED_PRECISIONS_CANONICAL="all"
SELECTED_PRECISION_LABELS=()
declare -A _SEEN_SELECTED_PRECISIONS=()
if [[ -n "$RUN_PRECISIONS" && "${RUN_PRECISIONS,,}" != "all" ]]; then
  PRECISION_FILTER_ACTIVE="1"
  IFS=',' read -r -a _RAW_SELECTED_PRECISIONS <<< "$RUN_PRECISIONS"
  for _raw_precision in "${_RAW_SELECTED_PRECISIONS[@]}"; do
    _normalized_precision="$(printf '%s' "$_raw_precision" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "$_normalized_precision" in
      "")
        ;;
      fp16|fp32|int8)
        if [[ -z "${_SEEN_SELECTED_PRECISIONS[$_normalized_precision]:-}" ]]; then
          SELECTED_PRECISION_LABELS+=("$_normalized_precision")
          _SEEN_SELECTED_PRECISIONS["$_normalized_precision"]=1
        fi
        ;;
      *)
        echo "Unknown precision in RUN_PRECISIONS: $_raw_precision" >&2
        exit 1
        ;;
    esac
  done
  if [[ ${#SELECTED_PRECISION_LABELS[@]} -eq 0 ]]; then
    echo "RUN_PRECISIONS did not contain any valid precision labels." >&2
    exit 1
  fi
  SELECTED_PRECISIONS_CANONICAL="$(IFS=,; echo "${SELECTED_PRECISION_LABELS[*]}")"
fi
export PRECISION_FILTER_ACTIVE SELECTED_PRECISIONS_CANONICAL

precision_is_selected() {
  local precision="$1"
  if [[ "$PRECISION_FILTER_ACTIVE" != "1" ]]; then
    return 0
  fi
  case ",$SELECTED_PRECISIONS_CANONICAL," in
    *,"$precision",*)
      return 0
      ;;
  esac
  return 1
}

show_oled_message() {
  if [[ "${OLED_ENABLE:-0}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$OLED_MESSAGE_TOOL" ]]; then
    return 0
  fi
  local cmd=("$PYBIN" "$OLED_MESSAGE_TOOL" --line1 "${1:-}" --line2 "${2:-}" --line3 "${3:-}")
  if [[ -n "${OLED_BUS:-}" ]]; then
    cmd+=(--bus "$OLED_BUS")
  fi
  if [[ -n "${OLED_ADDR:-}" ]]; then
    cmd+=(--addr "$OLED_ADDR")
  fi
  "${cmd[@]}" >/dev/null 2>&1 || true
}

post_status_update() {
  local stage="$1"
  local state="${2:-running}"
  local message="${3:-}"
  local precision="${4:-}"
  local run_variant="${5:-session_wrapper}"
  local accuracy_only="${6:-0}"
  if [[ -z "$STATUS_UPDATE_URL" ]]; then
    return 0
  fi
  "$PYBIN" - "$STATUS_UPDATE_URL" "$stage" "$state" "$message" "$precision" "$run_variant" "$accuracy_only" <<'PY' >/dev/null 2>&1 || true
import json
import os
import sys
import time
import urllib.request

url, stage, state, message, precision, run_variant, accuracy_only = sys.argv[1:8]
power_mode = {
    "power_mode_name": os.environ.get("CURRENT_POWER_MODE_NAME"),
    "power_mode_id": os.environ.get("CURRENT_POWER_MODE_ID"),
    "power_mode_class": os.environ.get("CURRENT_POWER_MODE_CLASS"),
    "power_mode_tag": os.environ.get("CURRENT_POWER_MODE_TAG"),
    "power_mode_label": os.environ.get("CURRENT_POWER_MODE_LABEL"),
    "matched_budget_label": os.environ.get("CURRENT_MATCHED_BUDGET_LABEL"),
    "power_budget_w": os.environ.get("CURRENT_POWER_BUDGET_W"),
}
payload = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "wallclock_ns": time.time_ns(),
    "state": state,
    "stage": stage,
    "message": message,
    "board_name": os.environ.get("BOARD_NAME"),
    "power_mode": power_mode,
    "precision_label": precision or None,
    "run_variant": run_variant or "session_wrapper",
    "run_id": os.environ.get("SESSION_NAME") or None,
    "session_name": os.environ.get("SESSION_NAME") or None,
    "session_root": os.environ.get("SESSION_ROOT") or None,
    "operation_mode": os.environ.get("SESSION_WORK_MODE") or None,
    "accuracy_only": accuracy_only == "1",
}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=4):
    pass
PY
}

emit_wrapper_status() {
  local stage="$1"
  local state="$2"
  local message="$3"
  local precision="${4:-}"
  local run_variant="${5:-session_wrapper}"
  local oled_l1="${6:-$stage}"
  local oled_l2="${7:-$message}"
  local oled_l3="${8:-}"
  local accuracy_only="${9:-0}"
  echo "[INFO] $message"
  post_status_update "$stage" "$state" "$message" "$precision" "$run_variant" "$accuracy_only"
  show_oled_message "$oled_l1" "$oled_l2" "$oled_l3"
}

write_session_settings() {
  "$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

session_root = Path(os.environ["SESSION_ROOT"]).expanduser().resolve()
payload = {
    "report_mode": os.environ.get("REPORT_MODE"),
    "run_variants": os.environ.get("RUN_VARIANTS"),
    "repeats": os.environ.get("REPEATS"),
    "warmup_images": os.environ.get("WARMUP_IMAGES"),
    "warmup_seconds": os.environ.get("WARMUP_SECONDS"),
    "min_images": os.environ.get("MIN_IMAGES"),
    "min_duration_s": os.environ.get("MIN_DURATION_S"),
    "noise_guard_enable": os.environ.get("NOISE_GUARD_ENABLE"),
    "noise_guard_session_wide": os.environ.get("SESSION_NOISE_GUARD_ACTIVE"),
    "noise_guard_disable_wifi": os.environ.get("NOISE_GUARD_DISABLE_WIFI"),
    "noise_guard_display_off": os.environ.get("NOISE_GUARD_DISPLAY_OFF"),
    "noise_guard_stop_services": os.environ.get("NOISE_GUARD_STOP_SERVICES"),
    "noise_guard_disable_bluetooth": os.environ.get("NOISE_GUARD_DISABLE_BLUETOOTH"),
    "oled_enable": os.environ.get("OLED_ENABLE"),
    "oled_bus": os.environ.get("OLED_BUS"),
    "oled_addr": os.environ.get("OLED_ADDR"),
    "pi_logger_url": os.environ.get("PI_LOGGER_URL"),
    "run_precisions": os.environ.get("SELECTED_PRECISIONS_CANONICAL"),
    "precision_filter_active": os.environ.get("PRECISION_FILTER_ACTIVE"),
}
(session_root / "effective_session_settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

verify_runtime_python_modules() {
  "$PYBIN" - <<'PY'
import importlib
import sys

missing = []
details = []
modules = ["numpy", "pandas", "yaml", "tqdm", "ultralytics", "tensorrt"]

for module_name in modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(module_name)
        details.append(f"{module_name}: {type(exc).__name__}: {exc}")

if missing:
    print(
        "Missing Jetson benchmark Python modules: "
        + ", ".join(missing)
        + "\nRun: bash scripts/setup_jetson_env.sh",
        file=sys.stderr,
    )
    if details:
        print("\nImport details:", file=sys.stderr)
        for line in details:
            print(f"  - {line}", file=sys.stderr)
    raise SystemExit(1)
PY
}

POWER_MODE_JSON="$("$PYBIN" "$POWER_HELPER" --json)"
eval "$("$PYBIN" "$POWER_HELPER" --shell)"
export CURRENT_POWER_MODE_NAME CURRENT_POWER_MODE_ID CURRENT_POWER_MODE_CLASS CURRENT_POWER_MODE_TAG CURRENT_POWER_MODE_LABEL CURRENT_MATCHED_BUDGET_LABEL CURRENT_POWER_BUDGET_W

ensure_required_engines() {
  local missing_required="0"
  local build_int8="0"

  if precision_is_selected fp16 && [[ ! -f "$FP16_MODEL_PATH" ]]; then
    missing_required="1"
  fi
  if precision_is_selected fp32 && [[ ! -f "$FP32_MODEL_PATH" ]]; then
    missing_required="1"
  fi

  if precision_is_selected int8; then
    case "${INT8_ENABLE,,}" in
      1|true|yes)
        build_int8="1"
        if [[ ! -f "$INT8_MODEL_PATH" ]]; then
          missing_required="1"
        fi
        ;;
      auto)
        if [[ "$PRECISION_FILTER_ACTIVE" == "1" && ! -f "$INT8_MODEL_PATH" ]]; then
          build_int8="1"
          missing_required="1"
        fi
        ;;
      *)
        echo "RUN_PRECISIONS selects int8, but INT8 is disabled by INT8_ENABLE=$INT8_ENABLE" >&2
        return 1
        ;;
    esac
  elif [[ "${INT8_ENABLE,,}" =~ ^(1|true|yes)$ && ! -f "$INT8_MODEL_PATH" ]]; then
    build_int8="1"
    missing_required="1"
  fi

  if [[ "$missing_required" != "1" ]]; then
    return 0
  fi
  if [[ "$AUTO_BUILD_MISSING_ENGINES" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$ONNX_MODEL_PATH" ]]; then
    echo "Missing TensorRT engine(s) detected, but ONNX file not found for auto-build: $ONNX_MODEL_PATH" >&2
    return 1
  fi
  if [[ ! -x "$BUILD_ALL_ENGINES_SCRIPT" && ! -f "$BUILD_ALL_ENGINES_SCRIPT" ]]; then
    echo "Auto-build script not found: $BUILD_ALL_ENGINES_SCRIPT" >&2
    return 1
  fi

  echo "[INFO] Missing TensorRT engine(s) detected. Auto-building required precisions..."
  BUILD_INT8="$build_int8" FORCE_REBUILD=0 bash "$BUILD_ALL_ENGINES_SCRIPT" "$ONNX_MODEL_PATH" "$MODEL_DIR"
}

ensure_required_engines

if [[ "$RUN_CURRENT_MODE_FP16_FP32_PHASE" != "inner" ]]; then
  echo "[INFO] Preparing session and checking whether noise-guard sudo is needed..."
fi

if precision_is_selected fp16 && [[ ! -f "$FP16_MODEL_PATH" ]]; then
  echo "Required file not found: $FP16_MODEL_PATH" >&2
  exit 1
fi
if precision_is_selected fp32 && [[ ! -f "$FP32_MODEL_PATH" ]]; then
  echo "Required file not found: $FP32_MODEL_PATH" >&2
  exit 1
fi
if [[ -n "$RUN_ACCURACY_MODEL" && ! -f "$RUN_ACCURACY_MODEL" ]]; then
  echo "Required accuracy file not found: $RUN_ACCURACY_MODEL" >&2
  exit 1
fi
case "${INT8_ENABLE,,}" in
  1|true|yes)
    if precision_is_selected int8 && [[ ! -f "$INT8_MODEL_PATH" ]]; then
      echo "INT8 model requested but not found: $INT8_MODEL_PATH" >&2
      exit 1
    fi
    ;;
  auto)
    if precision_is_selected int8 && [[ ! -f "$INT8_MODEL_PATH" ]]; then
      echo "Selected INT8 model was not found: $INT8_MODEL_PATH" >&2
      exit 1
    fi
    ;;
esac

SESSION_NAME="${SESSION_NAME:-agx_orin_yolo11s_${CURRENT_POWER_MODE_TAG}_${SESSION_TIMESTAMP}}"
SESSION_ROOT="${SESSION_ROOT:-$OUT_ROOT/$SESSION_NAME}"
LOOKUP_EXCLUDE_SESSION_ROOT="${LOOKUP_EXCLUDE_SESSION_ROOT:-$SESSION_ROOT}"
BACKFILL_SESSION_ROOT="${BACKFILL_SESSION_ROOT:-}"
TARGET_SESSION_ROOT="${TARGET_SESSION_ROOT:-${RERUN_SESSION_ROOT:-${EXPLICIT_SESSION_ROOT:-}}}"
REPLACE_EXISTING_VARIANTS="${REPLACE_EXISTING_VARIANTS:-0}"

if [[ -n "$TARGET_SESSION_ROOT" ]]; then
  TARGET_SESSION_ROOT="$(realpath -m "$TARGET_SESSION_ROOT")"
  if [[ ! -d "$TARGET_SESSION_ROOT" ]]; then
    echo "Explicit TARGET_SESSION_ROOT does not exist: $TARGET_SESSION_ROOT" >&2
    exit 1
  fi
  SESSION_ROOT="$TARGET_SESSION_ROOT"
  SESSION_NAME="$(basename "$SESSION_ROOT")"
  LOOKUP_EXCLUDE_SESSION_ROOT=""
  if [[ "$SESSION_NAME" =~ _([0-9]{8}T[0-9]{6}Z)$ ]]; then
    SESSION_TIMESTAMP="${BASH_REMATCH[1]}"
  fi
fi

export SESSION_NAME SESSION_ROOT LOOKUP_EXCLUDE_SESSION_ROOT BACKFILL_SESSION_ROOT TARGET_SESSION_ROOT REPLACE_EXISTING_VARIANTS SESSION_TIMESTAMP
export CURRENT_POWER_MODE_NAME CURRENT_POWER_MODE_ID CURRENT_POWER_MODE_CLASS CURRENT_POWER_MODE_TAG CURRENT_POWER_MODE_LABEL CURRENT_MATCHED_BUDGET_LABEL CURRENT_POWER_BUDGET_W

ALL_PRECISION_LABELS=()
ALL_PRECISION_MODEL_PATHS=()
PRECISION_LABELS=()
PRECISION_MODEL_PATHS=()
LAST_RUN_ROOT=""
LAST_RUN_STATE=""
CURRENT_RUN_VARIANT=""
VARIANT_ENABLE_PER_IMAGE_METRICS="0"
VARIANT_SESSION_ROOT=""
RUNS_ROOT=""
REPORTS_ROOT=""
RUN_METADATA_TSV=""

append_precision() {
  local precision="$1"
  local model_path="$2"
  ALL_PRECISION_LABELS+=("$precision")
  ALL_PRECISION_MODEL_PATHS+=("$model_path")
  if precision_is_selected "$precision"; then
    PRECISION_LABELS+=("$precision")
    PRECISION_MODEL_PATHS+=("$model_path")
  fi
}

append_precision fp16 "$FP16_MODEL_PATH"
append_precision fp32 "$FP32_MODEL_PATH"
case "${INT8_ENABLE,,}" in
  1|true|yes)
    append_precision int8 "$INT8_MODEL_PATH"
    ;;
  auto)
    if [[ -f "$INT8_MODEL_PATH" ]]; then
      append_precision int8 "$INT8_MODEL_PATH"
    fi
    ;;
esac

if [[ ${#PRECISION_LABELS[@]} -eq 0 ]]; then
  echo "No precisions selected for execution." >&2
  exit 1
fi

VARIANT_NAMES=()
declare -A _SEEN_VARIANTS=()
IFS=',' read -r -a _RAW_VARIANTS <<< "$RUN_VARIANTS"
for _raw_variant in "${_RAW_VARIANTS[@]}"; do
  _normalized="$(printf '%s' "$_raw_variant" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | xargs)"
  case "$_normalized" in
    "" )
      ;;
    standard|summary)
      if [[ -z "${_SEEN_VARIANTS[standard]:-}" ]]; then
        VARIANT_NAMES+=("standard")
        _SEEN_VARIANTS[standard]=1
      fi
      ;;
    per_image|perimage)
      echo "[INFO] RUN_VARIANTS entry per_image ignored; per-image inference artifacts are disabled for the main-run-only protocol." >&2
      ;;
    *)
      echo "Unknown RUN_VARIANTS entry: $_raw_variant" >&2
      exit 1
      ;;
  esac
done
if [[ ${#VARIANT_NAMES[@]} -eq 0 ]]; then
  VARIANT_NAMES=("standard")
fi
RUN_VARIANTS_CANONICAL="$(IFS=,; echo "${VARIANT_NAMES[*]}")"
export RUN_VARIANTS_CANONICAL
echo "[INFO] Run variants: $RUN_VARIANTS_CANONICAL"
declare -A _BACKFILLED_SESSION_ROOTS=()

variant_requires_per_image() {
  [[ "$1" == "per_image" ]]
}

accuracy_mode_requires_precision_specific_for_model() {
  local model_path="$1"
  if [[ -n "$RUN_ACCURACY_MODEL_OVERRIDE" ]]; then
    return 1
  fi
  if [[ "$RUN_ACCURACY_MODE" != "precision_specific_engine" ]]; then
    return 1
  fi
  [[ "$model_path" == *.engine ]]
}

probe_precision_run_state() {
  local precision="$1"
  local model_path="$2"
  local variant_name="$3"
  local require_per_image="$4"
  local require_precision_specific_accuracy="$5"
  local reused_root=""
  reused_root="$(find_reusable_run_root "$precision" "$model_path" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
  if [[ -n "$reused_root" ]]; then
    printf '%s' "reusable"
    return 0
  fi
  if [[ "$require_precision_specific_accuracy" == "1" ]]; then
    local backfillable_root=""
    backfillable_root="$(find_backfillable_run_root "$precision" "$model_path" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
    if [[ -n "$backfillable_root" ]]; then
      printf '%s' "backfill"
      return 0
    fi
  fi
  printf '%s' "full"
}

compute_session_work_mode() {
  if [[ -n "$TARGET_SESSION_ROOT" && "$REPLACE_EXISTING_VARIANTS" == "1" ]]; then
    printf '%s' "full"
    return 0
  fi
  if [[ "$SKIP_COMPLETED_PRECISIONS" != "1" ]]; then
    printf '%s' "full"
    return 0
  fi
  local saw_backfill="0"
  local saw_full="0"
  local variant_name=""
  local idx=""
  for variant_name in "${VARIANT_NAMES[@]}"; do
    local require_per_image="0"
    if variant_requires_per_image "$variant_name"; then
      require_per_image="1"
    fi
    for idx in "${!PRECISION_LABELS[@]}"; do
      local require_precision_specific_accuracy="0"
      if accuracy_mode_requires_precision_specific_for_model "${PRECISION_MODEL_PATHS[$idx]}"; then
        require_precision_specific_accuracy="1"
      fi
      local state=""
      state="$(probe_precision_run_state "${PRECISION_LABELS[$idx]}" "${PRECISION_MODEL_PATHS[$idx]}" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
      case "$state" in
        backfill)
          saw_backfill="1"
          ;;
        full)
          saw_full="1"
          ;;
      esac
    done
  done
  if [[ "$saw_full" == "1" && "$saw_backfill" == "1" ]]; then
    printf '%s' "mixed"
  elif [[ "$saw_full" == "1" ]]; then
    printf '%s' "full"
  elif [[ "$saw_backfill" == "1" ]]; then
    printf '%s' "backfill_only"
  else
    printf '%s' "reuse_only"
  fi
}

resolve_backfill_only_session_root() {
  if [[ -n "$BACKFILL_SESSION_ROOT" ]]; then
    local explicit_root=""
    explicit_root="$(realpath -m "$BACKFILL_SESSION_ROOT")"
    if [[ ! -d "$explicit_root" ]]; then
      echo "Explicit BACKFILL_SESSION_ROOT does not exist: $explicit_root" >&2
      return 1
    fi
    printf '%s\n' "$explicit_root"
    return 0
  fi
  local variant_name=""
  local idx=""
  for variant_name in "${VARIANT_NAMES[@]}"; do
    local require_per_image="0"
    if variant_requires_per_image "$variant_name"; then
      require_per_image="1"
    fi
    for idx in "${!PRECISION_LABELS[@]}"; do
      local require_precision_specific_accuracy="0"
      if accuracy_mode_requires_precision_specific_for_model "${PRECISION_MODEL_PATHS[$idx]}"; then
        require_precision_specific_accuracy="1"
      fi
      [[ "$require_precision_specific_accuracy" == "1" ]] || continue
      local candidate_run_root=""
      candidate_run_root="$(find_backfillable_run_root "${PRECISION_LABELS[$idx]}" "${PRECISION_MODEL_PATHS[$idx]}" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
      if [[ -n "$candidate_run_root" ]]; then
        dirname "$(dirname "$(dirname "$candidate_run_root")")"
        return 0
      fi
    done
  done
  return 1
}

run_backfill_for_session_root() {
  local target_session_root="$1"
  local cmd=("$PYBIN" "$BACKFILL_PRECISION_ACCURACY_SCRIPT" --session-root "$target_session_root" --variant "$CURRENT_RUN_VARIANT")
  local cache_key="${target_session_root}::${CURRENT_RUN_VARIANT}::${SELECTED_PRECISIONS_CANONICAL}"
  local idx=""
  if [[ -n "${_BACKFILLED_SESSION_ROOTS[$cache_key]:-}" ]]; then
    return 0
  fi
  for idx in "${!PRECISION_LABELS[@]}"; do
    cmd+=(--precision "${PRECISION_LABELS[$idx]}")
  done
  emit_wrapper_status \
    "accuracy_backfill_only" \
    "running" \
    "Running precision-specific accuracy-only backfill for existing session. Measured benchmark repeats will not rerun: $target_session_root" \
    "" \
    "session_wrapper" \
    "ACC ONLY" \
    "precision fix" \
    "no full rerun" \
    "1"
  if ! "${cmd[@]}"; then
    emit_wrapper_status \
      "accuracy_backfill_error" \
      "error" \
      "Precision-specific accuracy-only backfill failed for existing session: $target_session_root" \
      "" \
      "session_wrapper" \
      "ACC ERROR" \
      "backfill failed" \
      "check terminal" \
      "1"
    return 1
  fi
  emit_wrapper_status \
    "accuracy_backfill_complete" \
    "running" \
    "Precision-specific accuracy-only backfill completed for existing session: $target_session_root" \
    "" \
    "session_wrapper" \
    "ACC DONE" \
    "session ready" \
    "reuse measured" \
    "1"
  _BACKFILLED_SESSION_ROOTS["$cache_key"]=1
}

find_reusable_run_root() {
  local precision="$1"
  local model_path="$2"
  local variant_name="$3"
  local require_per_image="$4"
  local require_precision_specific_accuracy="$5"
  "$PYBIN" - "$OUT_ROOT" "$LOOKUP_EXCLUDE_SESSION_ROOT" "$precision" "$CURRENT_POWER_MODE_TAG" "$model_path" "$REPEATS" "$EXPECT_EXTERNAL_LOGGER" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy" "$BACKFILL_SESSION_ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path
import yaml

out_root = Path(sys.argv[1]).expanduser().resolve()
exclude_session_root_arg = sys.argv[2].strip()
exclude_session_root = Path(exclude_session_root_arg).expanduser().resolve() if exclude_session_root_arg else None
precision = sys.argv[3]
power_tag = sys.argv[4]
model_path = Path(sys.argv[5]).expanduser().resolve()
required_repeats = int(float(sys.argv[6]))
expect_external = sys.argv[7] == "1"
variant_name = sys.argv[8]
require_per_image = sys.argv[9] == "1"
require_precision_specific_accuracy = sys.argv[10] == "1"
explicit_session_root_arg = sys.argv[11].strip()
explicit_session_root = Path(explicit_session_root_arg).expanduser().resolve() if explicit_session_root_arg else None

if not out_root.exists() or not model_path.exists():
    raise SystemExit(0)

sha = hashlib.sha256()
with model_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        sha.update(chunk)
model_sha = sha.hexdigest()

pattern = f"*/{variant_name}/runs/agx_orin_yolo11s_{precision}_{power_tag}_{variant_name}_*"
candidates = sorted(out_root.glob(pattern), reverse=True)
if explicit_session_root is not None:
    candidates = [candidate for candidate in candidates if explicit_session_root in candidate.parents]

def _normalized_layout_candidates(raw_path: str | None) -> list[Path]:
    if not raw_path:
        return []
    base = Path(raw_path).expanduser().resolve()
    candidates = [base]
    raw = str(base)
    replacements = [
        ("/inference_bundle/model/", "/model/"),
        ("/inference_bundle/dataset/", "/dataset/"),
        ("/inference_bundle/metadata/", "/metadata/"),
        ("/inference_bundle/", "/"),
    ]
    for old, new in replacements:
        if old in raw:
            candidates.append(Path(raw.replace(old, new)).expanduser().resolve())
    if base.name:
        benchmark_root = out_root.parent
        candidates.extend(
            [
                (benchmark_root / "yolo11s" / "inference_bundle" / "model" / base.name).resolve(),
                (benchmark_root / "yolo11s" / "model" / base.name).resolve(),
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique

def _path_matches_model(raw_path: str | None) -> bool:
    return any(candidate == model_path for candidate in _normalized_layout_candidates(raw_path))

def is_complete(run_root: Path) -> bool:
    try:
        if exclude_session_root is not None and exclude_session_root in run_root.parents:
            return False
        final_path = run_root / "FINAL_SUMMARY.json"
        agg_path = run_root / "aggregate_repeat_summary.csv"
        analysis_path = run_root / "analysis_repeat_summary.csv"
        if not final_path.is_file() or not agg_path.is_file() or not analysis_path.is_file():
            return False
        final = json.loads(final_path.read_text(encoding="utf-8"))
        project = final.get("project") or {}
        model = final.get("model") or {}
        power_mode = final.get("power_mode") or {}
        if (project.get("precision_label") or model.get("precision_label")) != precision:
            return False
        if power_mode.get("observed_mode_tag") != power_tag:
            return False
        summary_sha = (model.get("measured_model_sha256") or "").strip()
        if summary_sha and summary_sha != model_sha:
            return False
        summary_path = model.get("measured_model_path")
        if summary_path and not summary_sha:
            if not _path_matches_model(summary_path):
                return False
        if require_precision_specific_accuracy:
            accuracy_precision_specific = bool(model.get("accuracy_precision_specific"))
            accuracy_model_path = model.get("accuracy_model_path")
            accuracy_model_sha = (model.get("accuracy_model_sha256") or "").strip()
            if not accuracy_precision_specific:
                return False
            if not accuracy_model_path or not accuracy_model_sha:
                return False
            if not _path_matches_model(accuracy_model_path):
                return False
            if accuracy_model_sha != model_sha:
                return False
            if model.get("accuracy_backend") != "tensorrt_engine":
                return False
            if (model.get("accuracy_precision_label") or "").lower() != precision.lower():
                return False
            config_path = run_root / "config_snapshot.yaml"
            if not config_path.is_file():
                return False
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            cfg_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
            cfg_accuracy_model_path = cfg_model.get("accuracy_model_path")
            cfg_accuracy_model_sha = (cfg_model.get("accuracy_model_sha256") or "").strip()
            if not cfg_accuracy_model_path or cfg_accuracy_model_sha != model_sha:
                return False
            if not _path_matches_model(cfg_accuracy_model_path):
                return False
            if cfg_model.get("accuracy_backend") != "tensorrt_engine":
                return False
            if (cfg_model.get("accuracy_precision_label") or "").lower() != precision.lower():
                return False
            validation_logs = sorted((run_root / "accuracy").glob("validation_*.log"))
            if not validation_logs:
                return False
            validation_text = validation_logs[0].read_text(encoding="utf-8", errors="replace")
            if str(model_path) not in validation_text:
                return False
            if "tensorrt_engine" not in validation_text:
                return False
            if precision.lower() not in validation_text.lower():
                return False
        if not (final.get("accuracy_metrics") or {}):
            return False

        with agg_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < required_repeats:
            return False
        if expect_external:
            valid = 0
            for row in rows:
                try:
                    float(row.get("avg_power_external_w", ""))
                    valid += 1
                except (TypeError, ValueError):
                    pass
            if valid < required_repeats:
                return False
        if require_per_image:
            per_image = sorted(run_root.glob("benchmark_runs/repeat_*/per_image_metrics.csv"))
            if len(per_image) < required_repeats:
                return False
        else:
            if list(run_root.glob("benchmark_runs/repeat_*/per_image_metrics.csv")):
                return False
            if list(run_root.glob("benchmark_runs/repeat_*/per_image_latency.csv")):
                return False
            if list(run_root.glob("benchmark_runs/repeat_*/trtexec_times.json")):
                return False
        return True
    except Exception:
        return False

for candidate in candidates:
    if is_complete(candidate):
        print(candidate)
        break
PY
}

find_backfillable_run_root() {
  local precision="$1"
  local model_path="$2"
  local variant_name="$3"
  local require_per_image="$4"
  local require_precision_specific_accuracy="$5"
  "$PYBIN" - "$OUT_ROOT" "$LOOKUP_EXCLUDE_SESSION_ROOT" "$precision" "$CURRENT_POWER_MODE_TAG" "$model_path" "$REPEATS" "$EXPECT_EXTERNAL_LOGGER" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy" "$BACKFILL_SESSION_ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path
import yaml

out_root = Path(sys.argv[1]).expanduser().resolve()
exclude_session_root_arg = sys.argv[2].strip()
exclude_session_root = Path(exclude_session_root_arg).expanduser().resolve() if exclude_session_root_arg else None
precision = sys.argv[3]
power_tag = sys.argv[4]
model_path = Path(sys.argv[5]).expanduser().resolve()
required_repeats = int(float(sys.argv[6]))
expect_external = sys.argv[7] == "1"
variant_name = sys.argv[8]
require_per_image = sys.argv[9] == "1"
require_precision_specific_accuracy = sys.argv[10] == "1"
explicit_session_root_arg = sys.argv[11].strip()
explicit_session_root = Path(explicit_session_root_arg).expanduser().resolve() if explicit_session_root_arg else None

if not require_precision_specific_accuracy or not out_root.exists() or not model_path.exists():
    raise SystemExit(0)

sha = hashlib.sha256()
with model_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        sha.update(chunk)
model_sha = sha.hexdigest()

pattern = f"*/{variant_name}/runs/agx_orin_yolo11s_{precision}_{power_tag}_{variant_name}_*"
candidates = sorted(out_root.glob(pattern), reverse=True)
if explicit_session_root is not None:
    candidates = [candidate for candidate in candidates if explicit_session_root in candidate.parents]

def _normalized_layout_candidates(raw_path: str | None) -> list[Path]:
    if not raw_path:
        return []
    base = Path(raw_path).expanduser().resolve()
    candidates = [base]
    raw = str(base)
    replacements = [
        ("/inference_bundle/model/", "/model/"),
        ("/inference_bundle/dataset/", "/dataset/"),
        ("/inference_bundle/metadata/", "/metadata/"),
        ("/inference_bundle/", "/"),
    ]
    for old, new in replacements:
        if old in raw:
            candidates.append(Path(raw.replace(old, new)).expanduser().resolve())
    if base.name:
        benchmark_root = out_root.parent
        candidates.extend(
            [
                (benchmark_root / "yolo11s" / "inference_bundle" / "model" / base.name).resolve(),
                (benchmark_root / "yolo11s" / "model" / base.name).resolve(),
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique

def _path_matches_model(raw_path: str | None) -> bool:
    return any(candidate == model_path for candidate in _normalized_layout_candidates(raw_path))

def is_backfillable(run_root: Path) -> bool:
    try:
        if exclude_session_root is not None and exclude_session_root in run_root.parents:
            return False
        final_path = run_root / "FINAL_SUMMARY.json"
        agg_path = run_root / "aggregate_repeat_summary.csv"
        analysis_path = run_root / "analysis_repeat_summary.csv"
        if not final_path.is_file() or not agg_path.is_file() or not analysis_path.is_file():
            return False
        final = json.loads(final_path.read_text(encoding="utf-8"))
        project = final.get("project") or {}
        model = final.get("model") or {}
        power_mode = final.get("power_mode") or {}
        if (project.get("precision_label") or model.get("precision_label")) != precision:
            return False
        if power_mode.get("observed_mode_tag") != power_tag:
            return False
        summary_sha = (model.get("measured_model_sha256") or "").strip()
        if summary_sha and summary_sha != model_sha:
            return False
        summary_path = model.get("measured_model_path")
        if summary_path:
            if not _path_matches_model(summary_path):
                return False
        if not (final.get("accuracy_metrics") or {}):
            return False
        if model.get("accuracy_precision_specific"):
            accuracy_model_path = model.get("accuracy_model_path")
            accuracy_model_sha = (model.get("accuracy_model_sha256") or "").strip()
            if accuracy_model_path and accuracy_model_sha == model_sha:
                accuracy_model_matches = _path_matches_model(accuracy_model_path)
            else:
                accuracy_model_matches = False
            if accuracy_model_matches and model.get("accuracy_backend") == "tensorrt_engine" and (model.get("accuracy_precision_label") or "").lower() == precision.lower():
                config_path = run_root / "config_snapshot.yaml"
                validation_logs = sorted((run_root / "accuracy").glob("validation_*.log"))
                config_ok = False
                log_ok = False
                if config_path.is_file():
                    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    cfg_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
                    cfg_accuracy_model_path = cfg_model.get("accuracy_model_path")
                    cfg_accuracy_model_sha = (cfg_model.get("accuracy_model_sha256") or "").strip()
                    config_ok = (
                        cfg_accuracy_model_path is not None
                        and _path_matches_model(cfg_accuracy_model_path)
                        and cfg_accuracy_model_sha == model_sha
                        and cfg_model.get("accuracy_backend") == "tensorrt_engine"
                        and (cfg_model.get("accuracy_precision_label") or "").lower() == precision.lower()
                    )
                if validation_logs:
                    validation_text = validation_logs[0].read_text(encoding="utf-8", errors="replace")
                    log_ok = (
                        str(model_path) in validation_text
                        and "tensorrt_engine" in validation_text
                        and precision.lower() in validation_text.lower()
                    )
                if config_ok and log_ok:
                    return False

        with agg_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < required_repeats:
            return False
        if expect_external:
            valid = 0
            for row in rows:
                try:
                    float(row.get("avg_power_external_w", ""))
                    valid += 1
                except (TypeError, ValueError):
                    pass
            if valid < required_repeats:
                return False
        if require_per_image:
            per_image = sorted(run_root.glob("benchmark_runs/repeat_*/per_image_metrics.csv"))
            if len(per_image) < required_repeats:
                return False
        else:
            if list(run_root.glob("benchmark_runs/repeat_*/per_image_metrics.csv")):
                return False
            if list(run_root.glob("benchmark_runs/repeat_*/per_image_latency.csv")):
                return False
            if list(run_root.glob("benchmark_runs/repeat_*/trtexec_times.json")):
                return False
        return True
    except Exception:
        return False

for candidate in candidates:
    if is_backfillable(candidate):
        print(candidate)
        break
PY
}

maybe_backfill_precision_specific_accuracy() {
  local precision="$1"
  local model_path="$2"
  local variant_name="$3"
  local require_per_image="$4"
  local require_precision_specific_accuracy="$5"

  if [[ "$require_precision_specific_accuracy" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$BACKFILL_PRECISION_ACCURACY_SCRIPT" ]]; then
    return 0
  fi

  local candidate_run_root=""
  candidate_run_root="$(find_backfillable_run_root "$precision" "$model_path" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
  if [[ -z "$candidate_run_root" ]]; then
    return 0
  fi

  local candidate_session_root=""
  candidate_session_root="$(dirname "$(dirname "$(dirname "$candidate_run_root")")")"
  if [[ -n "${_BACKFILLED_SESSION_ROOTS[$candidate_session_root]:-}" ]]; then
    return 0
  fi

  emit_wrapper_status \
    "accuracy_backfill_only" \
    "running" \
    "Running precision-specific accuracy-only backfill for existing session. Measured benchmark repeats will not rerun: $candidate_session_root" \
    "$precision" \
    "$variant_name" \
    "ACC ONLY" \
    "${precision^^} ${variant_name}" \
    "no full rerun" \
    "1"
  if ! "$PYBIN" "$BACKFILL_PRECISION_ACCURACY_SCRIPT" --session-root "$candidate_session_root"; then
    emit_wrapper_status \
      "accuracy_backfill_error" \
      "error" \
      "Precision-specific accuracy-only backfill failed for existing session: $candidate_session_root" \
      "$precision" \
      "$variant_name" \
      "ACC ERROR" \
      "${precision^^} ${variant_name}" \
      "check terminal" \
      "1"
    return 1
  fi
  emit_wrapper_status \
    "accuracy_backfill_complete" \
    "running" \
    "Precision-specific accuracy-only backfill completed for existing session: $candidate_session_root" \
    "$precision" \
    "$variant_name" \
    "ACC DONE" \
    "${precision^^} ${variant_name}" \
    "reuse measured" \
    "1"
  _BACKFILLED_SESSION_ROOTS["$candidate_session_root"]=1
}

record_reusable_precision_run() {
  local precision="$1"
  local model_path="$2"
  local require_per_image="0"
  local require_precision_specific_accuracy="0"
  local reused_root=""

  if variant_requires_per_image "$CURRENT_RUN_VARIANT"; then
    require_per_image="1"
  fi
  if accuracy_mode_requires_precision_specific_for_model "$model_path"; then
    require_precision_specific_accuracy="1"
  fi

  reused_root="$(find_reusable_run_root "$precision" "$model_path" "$CURRENT_RUN_VARIANT" "$require_per_image" "$require_precision_specific_accuracy")"
  if [[ -z "$reused_root" ]]; then
    echo "Expected reusable run not found after precision-specific accuracy backfill: precision=$precision variant=$CURRENT_RUN_VARIANT session_root=$SESSION_ROOT" >&2
    return 1
  fi

  echo "[INFO] Reuse $precision ($CURRENT_RUN_VARIANT) from existing complete run: $reused_root"
  LAST_RUN_ROOT="$reused_root"
  LAST_RUN_STATE="reused"
  printf '%s\t%s\t%s\t%s\n' "$precision" "$model_path" "$reused_root" "$LAST_RUN_STATE" >> "$RUN_METADATA_TSV"
}

record_unselected_reusable_precision_runs() {
  local idx=""
  local precision=""
  local model_path=""
  local require_per_image="0"
  local require_precision_specific_accuracy="0"
  local reused_root=""

  if [[ "$PRECISION_FILTER_ACTIVE" != "1" ]]; then
    return 0
  fi
  if variant_requires_per_image "$CURRENT_RUN_VARIANT"; then
    require_per_image="1"
  fi

  for idx in "${!ALL_PRECISION_LABELS[@]}"; do
    precision="${ALL_PRECISION_LABELS[$idx]}"
    model_path="${ALL_PRECISION_MODEL_PATHS[$idx]}"
    if precision_is_selected "$precision"; then
      continue
    fi
    if [[ ! -f "$model_path" ]]; then
      continue
    fi
    require_precision_specific_accuracy="0"
    if accuracy_mode_requires_precision_specific_for_model "$model_path"; then
      require_precision_specific_accuracy="1"
    fi
    reused_root="$(find_reusable_run_root "$precision" "$model_path" "$CURRENT_RUN_VARIANT" "$require_per_image" "$require_precision_specific_accuracy")"
    if [[ -z "$reused_root" ]]; then
      continue
    fi
    echo "[INFO] Keep existing $precision ($CURRENT_RUN_VARIANT) in reports: $reused_root"
    printf '%s\t%s\t%s\t%s\n' "$precision" "$model_path" "$reused_root" "reused" >> "$RUN_METADATA_TSV"
  done
}

have_pending_precision_runs_for_variant() {
  local variant_name="$1"
  if [[ -n "$TARGET_SESSION_ROOT" && "$REPLACE_EXISTING_VARIANTS" == "1" ]]; then
    return 0
  fi
  local require_per_image="0"
  if [[ "$SKIP_COMPLETED_PRECISIONS" != "1" ]]; then
    return 0
  fi
  if variant_requires_per_image "$variant_name"; then
    require_per_image="1"
  fi
  local idx
  for idx in "${!PRECISION_LABELS[@]}"; do
    local state=""
    local require_precision_specific_accuracy="0"
    if accuracy_mode_requires_precision_specific_for_model "${PRECISION_MODEL_PATHS[$idx]}"; then
      require_precision_specific_accuracy="1"
    fi
    state="$(probe_precision_run_state "${PRECISION_LABELS[$idx]}" "${PRECISION_MODEL_PATHS[$idx]}" "$variant_name" "$require_per_image" "$require_precision_specific_accuracy")"
    if [[ "$state" != "reusable" ]]; then
      return 0
    fi
  done
  return 1
}

have_pending_precision_runs_any_variant() {
  local variant_name
  if [[ -n "$TARGET_SESSION_ROOT" && "$REPLACE_EXISTING_VARIANTS" == "1" ]]; then
    return 0
  fi
  if [[ "$SKIP_COMPLETED_PRECISIONS" != "1" ]]; then
    return 0
  fi
  for variant_name in "${VARIANT_NAMES[@]}"; do
    if have_pending_precision_runs_for_variant "$variant_name"; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${FORCED_SESSION_WORK_MODE:-}" ]]; then
  SESSION_WORK_MODE="$FORCED_SESSION_WORK_MODE"
else
  SESSION_WORK_MODE="$(compute_session_work_mode)"
fi
export SESSION_WORK_MODE
if [[ "$SESSION_WORK_MODE" == "backfill_only" ]]; then
  EXISTING_BACKFILL_SESSION_ROOT="$(resolve_backfill_only_session_root || true)"
  if [[ -n "$EXISTING_BACKFILL_SESSION_ROOT" ]]; then
    SESSION_ROOT="$EXISTING_BACKFILL_SESSION_ROOT"
    SESSION_NAME="$(basename "$SESSION_ROOT")"
    LOOKUP_EXCLUDE_SESSION_ROOT=""
  fi
fi
export SESSION_NAME SESSION_ROOT LOOKUP_EXCLUDE_SESSION_ROOT
mkdir -p "$SESSION_ROOT"
printf '%s\n' "$POWER_MODE_JSON" > "$SESSION_ROOT/detected_power_mode.json"
write_session_settings
if [[ "$RUN_CURRENT_MODE_FP16_FP32_PHASE" == "inner" ]]; then
  case "$SESSION_WORK_MODE" in
    backfill_only)
      emit_wrapper_status \
        "accuracy_backfill_only" \
        "running" \
        "Session work mode: precision-specific accuracy-only backfill. No full benchmark rerun will happen." \
        "" \
        "session_wrapper" \
        "ACC ONLY" \
        "precision fix" \
        "no full rerun" \
        "1"
      ;;
    mixed)
      emit_wrapper_status \
        "mixed_session" \
        "running" \
        "Session work mode: mixed. Existing runs with missing precision-specific accuracy will be backfilled; missing runs will execute full benchmark." \
        "" \
        "session_wrapper" \
        "MIXED MODE" \
        "backfill + full" \
        "see terminal" \
        "0"
      ;;
    full)
      echo "[INFO] Session work mode: full benchmark execution required."
      ;;
    reuse_only)
      echo "[INFO] Session work mode: reuse-only. No new benchmark or accuracy backfill is needed."
      ;;
  esac
fi

write_variant_settings() {
  "$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

variant_root = Path(os.environ["VARIANT_SESSION_ROOT"]).expanduser().resolve()
payload = {
    "run_variant": os.environ.get("CURRENT_RUN_VARIANT"),
    "per_image_metrics_enabled": os.environ.get("VARIANT_ENABLE_PER_IMAGE_METRICS"),
    "session_parent_root": os.environ.get("SESSION_ROOT"),
    "report_mode": os.environ.get("REPORT_MODE"),
    "repeats": os.environ.get("REPEATS"),
    "warmup_images": os.environ.get("WARMUP_IMAGES"),
    "warmup_seconds": os.environ.get("WARMUP_SECONDS"),
    "min_images": os.environ.get("MIN_IMAGES"),
    "min_duration_s": os.environ.get("MIN_DURATION_S"),
    "expect_external_logger": os.environ.get("EXPECT_EXTERNAL_LOGGER"),
}
(variant_root / "effective_variant_settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

maybe_prepare_explicit_variant_rerun() {
  local variant_name="$1"
  local target_variant_root="$SESSION_ROOT/$variant_name"
  if [[ -z "$TARGET_SESSION_ROOT" || "$REPLACE_EXISTING_VARIANTS" != "1" || "$SESSION_WORK_MODE" == "backfill_only" ]]; then
    return 0
  fi
  if [[ ! -d "$target_variant_root" ]]; then
    return 0
  fi
  emit_wrapper_status \
    "variant_replace_prepare" \
    "running" \
    "Replacing existing variant contents in explicit target session root: $target_variant_root" \
    "" \
    "$variant_name" \
    "REPLACE" \
    "$variant_name" \
    "clean old files" \
    "0"
  rm -rf "$target_variant_root"
}

set_variant_context() {
  CURRENT_RUN_VARIANT="$1"
  if variant_requires_per_image "$CURRENT_RUN_VARIANT"; then
    VARIANT_ENABLE_PER_IMAGE_METRICS="1"
  else
    VARIANT_ENABLE_PER_IMAGE_METRICS="0"
  fi
  VARIANT_SESSION_ROOT="$SESSION_ROOT/$CURRENT_RUN_VARIANT"
  RUNS_ROOT="$VARIANT_SESSION_ROOT/runs"
  REPORTS_ROOT="$VARIANT_SESSION_ROOT/reports"
  RUN_METADATA_TSV="$VARIANT_SESSION_ROOT/session_run_metadata.tsv"
  mkdir -p "$RUNS_ROOT" "$REPORTS_ROOT"
  printf '%s\n' "$POWER_MODE_JSON" > "$VARIANT_SESSION_ROOT/detected_power_mode.json"
  export CURRENT_RUN_VARIANT VARIANT_ENABLE_PER_IMAGE_METRICS VARIANT_SESSION_ROOT RUNS_ROOT REPORTS_ROOT RUN_METADATA_TSV
  write_variant_settings
}

if [[ "$RUN_CURRENT_MODE_FP16_FP32_PHASE" != "inner" ]]; then
  if [[ "$NOISE_GUARD_ENABLE" == "1" && "$SESSION_WORK_MODE" != "backfill_only" ]] && have_pending_precision_runs_any_variant; then
    export RUN_CURRENT_MODE_FP16_FP32_PHASE="inner"
    export SESSION_NOISE_GUARD_ACTIVE="1"
    write_session_settings

    guard_cmd=(
      "$PYBIN" "$SESSION_NOISE_GUARD"
      --summary-json "$SESSION_ROOT/session_noise_guard_summary.json"
      --profile "$NOISE_GUARD_PROFILE"
    )
    if [[ -n "$PI_LOGGER_URL" ]]; then
      guard_cmd+=(--preserve-url "$PI_LOGGER_URL")
    fi
    if [[ "$NOISE_GUARD_DISPLAY_OFF" != "1" ]]; then guard_cmd+=(--no-display-off); fi
    if [[ "$NOISE_GUARD_STOP_SERVICES" != "1" ]]; then guard_cmd+=(--no-stop-services); fi
    if [[ "$NOISE_GUARD_DISABLE_WIFI" != "1" ]]; then guard_cmd+=(--no-disable-wifi); fi
    if [[ "$NOISE_GUARD_DISABLE_BLUETOOTH" != "1" ]]; then guard_cmd+=(--no-disable-bluetooth); fi
    if [[ "$OLED_ENABLE" == "1" ]]; then
      guard_cmd+=(--oled-enable)
    fi
    if [[ -n "$OLED_BUS" ]]; then
      guard_cmd+=(--oled-bus "$OLED_BUS")
    fi
    if [[ -n "$OLED_ADDR" ]]; then
      guard_cmd+=(--oled-addr "$OLED_ADDR")
    fi
    guard_cmd+=(-- bash "$0")

    exec env \
      PYBIN="$PYBIN" \
      RUN_CURRENT_MODE_FP16_FP32_PHASE="$RUN_CURRENT_MODE_FP16_FP32_PHASE" \
      SESSION_NOISE_GUARD_ACTIVE="$SESSION_NOISE_GUARD_ACTIVE" \
      FORCED_SESSION_WORK_MODE="$SESSION_WORK_MODE" \
      QUIET_WRAP_ENABLE=1 \
      SUDO_PROMPT_ONCE=1 \
      SUDO_REQUIRE_SUCCESS=1 \
      OLED_ENABLE="$OLED_ENABLE" \
      OLED_BUS="$OLED_BUS" \
      OLED_ADDR="$OLED_ADDR" \
      "$RUN_WRAPPER" "${guard_cmd[@]}"
  fi

  export RUN_CURRENT_MODE_FP16_FP32_PHASE="inner"
  export SESSION_NOISE_GUARD_ACTIVE="0"
  write_session_settings
  exec env PYBIN="$PYBIN" RUN_CURRENT_MODE_FP16_FP32_PHASE="$RUN_CURRENT_MODE_FP16_FP32_PHASE" SESSION_NOISE_GUARD_ACTIVE="$SESSION_NOISE_GUARD_ACTIVE" FORCED_SESSION_WORK_MODE="$SESSION_WORK_MODE" bash "$0"
fi

if ! verify_runtime_python_modules; then
  exit 1
fi

echo "[INFO] Measured FP16 model: $FP16_MODEL_PATH"
echo "[INFO] Measured FP32 model: $FP32_MODEL_PATH"
case "${INT8_ENABLE,,}" in
  1|true|yes)
    echo "[INFO] Measured INT8 model requested: $INT8_MODEL_PATH"
    ;;
  auto)
    if [[ -f "$INT8_MODEL_PATH" ]]; then
      echo "[INFO] Measured INT8 model auto-detected: $INT8_MODEL_PATH"
    else
      echo "[INFO] Measured INT8 model not found; INT8 run will be skipped."
    fi
    ;;
  *)
    echo "[INFO] Measured INT8 model disabled."
    ;;
esac
if [[ -n "$RUN_ACCURACY_MODEL_OVERRIDE" ]]; then
  echo "[INFO] Accuracy model override: $RUN_ACCURACY_MODEL_OVERRIDE"
elif [[ "$RUN_ACCURACY_MODE" == "precision_specific_engine" ]]; then
  echo "[INFO] Accuracy mode: precision-specific engine validation"
else
  echo "[INFO] Accuracy model: $RUN_ACCURACY_MODEL"
fi
echo "[INFO] Effective repeats: $REPEATS"
echo "[INFO] Effective warmup: images=$WARMUP_IMAGES seconds=$WARMUP_SECONDS"
echo "[INFO] Effective measured minima: images=$MIN_IMAGES duration_s=$MIN_DURATION_S"
echo "[INFO] Skip completed precisions: $SKIP_COMPLETED_PRECISIONS"
if [[ "$PRECISION_FILTER_ACTIVE" == "1" ]]; then
  echo "[INFO] Selected precisions: $SELECTED_PRECISIONS_CANONICAL"
else
  echo "[INFO] Selected precisions: all available"
fi

run_one_precision() {
  local precision="$1"
  local model_path="$2"
  local run_name="agx_orin_yolo11s_${precision}_${CURRENT_POWER_MODE_TAG}_${CURRENT_RUN_VARIANT}_${SESSION_TIMESTAMP}"
  local run_root="$RUNS_ROOT/$run_name"
  local accuracy_model_for_run="$RUN_ACCURACY_MODEL"

  if [[ -z "$RUN_ACCURACY_MODEL_OVERRIDE" ]] && accuracy_mode_requires_precision_specific_for_model "$model_path"; then
    accuracy_model_for_run="$model_path"
  fi

  local cmd=(
    "$PYBIN" "$PKG_ROOT/jetson/run_benchmark.py"
    --model "$model_path"
    --dataset-root "$DATASET_INPUT"
    --run-name "$run_name"
    --run-variant "$CURRENT_RUN_VARIANT"
    --board-name "$BOARD_NAME"
    --precision-label "$precision"
    --output-root "$RUNS_ROOT"
    --device "$DEVICE"
    --repeats "$REPEATS"
    --warmup-images "$WARMUP_IMAGES"
    --warmup-seconds "$WARMUP_SECONDS"
    --min-images "$MIN_IMAGES"
    --min-duration-s "$MIN_DURATION_S"
    --external-sample-hz "$EXTERNAL_SAMPLE_HZ"
    --tegrastats-interval-ms "$TEGRASTATS_INTERVAL_MS"
    --internal-interval-ms "$INTERNAL_INTERVAL_MS"
    --start-delay-s "$START_DELAY_S"
    --power-mode-label "$CURRENT_POWER_MODE_LABEL"
    --observed-power-mode-name "$CURRENT_POWER_MODE_NAME"
    --observed-power-mode-id "$CURRENT_POWER_MODE_ID"
    --observed-power-mode-class "$CURRENT_POWER_MODE_CLASS"
    --observed-power-mode-tag "$CURRENT_POWER_MODE_TAG"
    --observed-power-mode-source "auto_detected_current_mode"
  )

  if [[ -n "$CURRENT_POWER_BUDGET_W" ]]; then
    cmd+=(--observed-power-budget-w "$CURRENT_POWER_BUDGET_W")
  fi
  if [[ -n "$CURRENT_MATCHED_BUDGET_LABEL" ]]; then
    cmd+=(--matched-budget-label "$CURRENT_MATCHED_BUDGET_LABEL")
  fi
  if [[ -n "$accuracy_model_for_run" ]]; then
    cmd+=(--accuracy-model "$accuracy_model_for_run")
  fi
  if [[ -n "$PI_LOGGER_URL" ]]; then
    cmd+=(--pi-logger-url "$PI_LOGGER_URL")
  fi
  if [[ "$EXPECT_EXTERNAL_LOGGER" == "1" ]]; then
    cmd+=(--expect-external-logger)
  fi
  if [[ "$OLED_ENABLE" == "1" ]]; then
    cmd+=(--oled-enable)
  fi
  if [[ -n "$OLED_BUS" ]]; then
    cmd+=(--oled-bus "$OLED_BUS")
  fi
  if [[ -n "$OLED_ADDR" ]]; then
    cmd+=(--oled-addr "$OLED_ADDR")
  fi
  if [[ "$VARIANT_ENABLE_PER_IMAGE_METRICS" == "1" ]]; then
    cmd+=(--save-per-image-outputs)
  else
    cmd+=(--no-save-per-image-outputs --no-save-predictions)
  fi
  if [[ "$SESSION_NOISE_GUARD_ACTIVE" == "1" ]]; then
    cmd+=(--no-noise-guard)
  elif [[ "$NOISE_GUARD_ENABLE" == "1" ]]; then
    cmd+=(--noise-guard --noise-guard-profile "$NOISE_GUARD_PROFILE")
    if [[ "$NOISE_GUARD_DISPLAY_OFF" != "1" ]]; then cmd+=(--no-noise-guard-display-off); fi
    if [[ "$NOISE_GUARD_STOP_SERVICES" != "1" ]]; then cmd+=(--no-noise-guard-stop-services); fi
    if [[ "$NOISE_GUARD_DISABLE_WIFI" != "1" ]]; then cmd+=(--no-noise-guard-disable-wifi); fi
    if [[ "$NOISE_GUARD_DISABLE_BLUETOOTH" != "1" ]]; then cmd+=(--no-noise-guard-disable-bluetooth); fi
  else
    cmd+=(--no-noise-guard)
  fi

  "${cmd[@]}"
  local merge_cmd=("$PYBIN" "$MERGE_SUMMARY" --run-root "$run_root")
  if [[ "$VARIANT_ENABLE_PER_IMAGE_METRICS" != "1" ]]; then
    merge_cmd+=(--skip-per-image-metrics)
  fi
  "${merge_cmd[@]}"
  LAST_RUN_ROOT="$run_root"
  LAST_RUN_STATE="new"
}

materialize_precision_run() {
  local precision="$1"
  local model_path="$2"
  local reused_root=""
  local require_per_image="0"
  local require_precision_specific_accuracy="0"

  if variant_requires_per_image "$CURRENT_RUN_VARIANT"; then
    require_per_image="1"
  fi
  if accuracy_mode_requires_precision_specific_for_model "$model_path"; then
    require_precision_specific_accuracy="1"
  fi
  maybe_backfill_precision_specific_accuracy "$precision" "$model_path" "$CURRENT_RUN_VARIANT" "$require_per_image" "$require_precision_specific_accuracy"
  if [[ "$SKIP_COMPLETED_PRECISIONS" == "1" ]]; then
    reused_root="$(find_reusable_run_root "$precision" "$model_path" "$CURRENT_RUN_VARIANT" "$require_per_image" "$require_precision_specific_accuracy")"
  fi

  if [[ -n "$reused_root" ]]; then
    echo "[INFO] Skip $precision ($CURRENT_RUN_VARIANT) because a complete run already exists: $reused_root"
    LAST_RUN_ROOT="$reused_root"
    LAST_RUN_STATE="reused"
    printf '%s\t%s\t%s\t%s\n' "$precision" "$model_path" "$reused_root" "$LAST_RUN_STATE" >> "$RUN_METADATA_TSV"
    return 0
  fi

  run_one_precision "$precision" "$model_path"
  printf '%s\t%s\t%s\t%s\n' "$precision" "$model_path" "$LAST_RUN_ROOT" "$LAST_RUN_STATE" >> "$RUN_METADATA_TSV"
}

finalize_variant_session() {
  COMPARE_CMD=("$PYBIN" "$COMPARE_RUNS" --outdir "$REPORTS_ROOT")
  while IFS=$'\t' read -r precision_label model_path run_root run_state; do
    [[ -n "$run_root" ]] || continue
    COMPARE_CMD+=(--run-root "$run_root")
  done < "$RUN_METADATA_TSV"

  COMPARE_OUTPUT_JSON="$("${COMPARE_CMD[@]}")"
  printf '%s\n' "$COMPARE_OUTPUT_JSON" > "$REPORTS_ROOT/compare_runs_output.json"

  "$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

variant_root = Path(os.environ['VARIANT_SESSION_ROOT']).expanduser().resolve()
reports_root = Path(os.environ['REPORTS_ROOT']).expanduser().resolve()
power_mode_json = json.loads((variant_root / 'detected_power_mode.json').read_text(encoding='utf-8'))
run_rows = []
paper_artifacts = [
    str(variant_root / 'detected_power_mode.json'),
    str(Path(os.environ['SESSION_ROOT']).expanduser().resolve() / 'effective_session_settings.json'),
    str(variant_root / 'effective_variant_settings.json'),
    str(reports_root / 'proposal_comparison_table.csv'),
    str(reports_root / 'proposal_comparison_table.json'),
]
readme_lines = [
    f"Keep these files for the paper/report ({os.environ['CURRENT_RUN_VARIANT']}):",
    '- detected_power_mode.json',
    '- effective_variant_settings.json',
]

pairwise_path = reports_root / 'proposal_pairwise_summary.json'
if pairwise_path.exists():
    paper_artifacts.append(str(pairwise_path))

metadata_tsv = variant_root / 'session_run_metadata.tsv'
for raw_line in metadata_tsv.read_text(encoding='utf-8').splitlines():
    if not raw_line.strip():
        continue
    precision_label, model_path, run_root_raw, run_state = raw_line.split('\t')
    run_root = Path(run_root_raw)
    final_summary = run_root / 'FINAL_SUMMARY.json'
    final_payload = json.loads(final_summary.read_text(encoding='utf-8'))
    aggregate_csv = run_root / 'aggregate_repeat_summary.csv'
    benchmark_runs_root = run_root / 'benchmark_runs'
    per_image_metrics = sorted(str(p) for p in benchmark_runs_root.glob('repeat_*/per_image_metrics.csv'))
    merged_traces = sorted(str(p) for p in benchmark_runs_root.glob('repeat_*/merged_trace.csv'))
    complexity_json = (final_payload.get('files') or {}).get('complexity_json')
    complexity_onnx = (final_payload.get('files') or {}).get('complexity_profile_onnx')
    run_rows.append(
        {
            'precision_label': precision_label,
            'model_path': model_path,
            'run_root': str(run_root),
            'run_state': run_state,
            'final_summary_json': str(final_summary),
            'aggregate_csv': str(aggregate_csv),
            'complexity_json': complexity_json,
            'complexity_profile_onnx': complexity_onnx,
            'benchmark_runs_root': str(benchmark_runs_root),
            'per_image_metrics_csvs': per_image_metrics,
            'merged_trace_csvs': merged_traces,
        }
    )
    paper_artifacts.append(str(final_summary))
    paper_artifacts.append(str(aggregate_csv))
    if complexity_json:
        paper_artifacts.append(str(complexity_json))
    paper_artifacts.extend(per_image_metrics)
    paper_artifacts.extend(merged_traces)
    readme_prefix = '(reused) ' if run_state == 'reused' else ''
    readme_lines.append(f'- {readme_prefix}{run_root}/FINAL_SUMMARY.json')
    readme_lines.append(f'- {readme_prefix}{run_root}/aggregate_repeat_summary.csv')
    if complexity_json:
        readme_lines.append(f'- {readme_prefix}{complexity_json}')
    if per_image_metrics:
        readme_lines.append(f'- {readme_prefix}{run_root}/benchmark_runs/repeat_*/per_image_metrics.csv')
    if merged_traces:
        readme_lines.append(f'- {readme_prefix}{run_root}/benchmark_runs/repeat_*/merged_trace.csv')

readme_lines.append('- reports/proposal_comparison_table.csv')
readme_lines.append('- reports/proposal_comparison_table.json')
if pairwise_path.exists():
    readme_lines.append('- reports/proposal_pairwise_summary.json')

manifest = {
    'session_name': os.environ['SESSION_NAME'],
    'run_variant': os.environ['CURRENT_RUN_VARIANT'],
    'session_root': str(variant_root),
    'session_parent_root': str(Path(os.environ['SESSION_ROOT']).expanduser().resolve()),
    'timestamp_utc': os.environ['SESSION_TIMESTAMP'],
    'board_name': os.environ['BOARD_NAME'],
    'power_mode_detection': power_mode_json,
    'runs': run_rows,
    'comparison_outputs': {
        'comparison_csv': str(reports_root / 'proposal_comparison_table.csv'),
        'comparison_json': str(reports_root / 'proposal_comparison_table.json'),
        'pairwise_json': str(reports_root / 'proposal_pairwise_summary.json'),
    },
    'session_noise_guard_summary_json': str(Path(os.environ['SESSION_ROOT']).expanduser().resolve() / 'session_noise_guard_summary.json'),
    'effective_session_settings_json': str(Path(os.environ['SESSION_ROOT']).expanduser().resolve() / 'effective_session_settings.json'),
    'effective_variant_settings_json': str(variant_root / 'effective_variant_settings.json'),
    'paper_artifacts': paper_artifacts,
}

session_guard_path = Path(os.environ['SESSION_ROOT']).expanduser().resolve() / 'session_noise_guard_summary.json'
if session_guard_path.exists():
    manifest['paper_artifacts'].append(str(session_guard_path))

(variant_root / 'paper_submission_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(variant_root / 'README_SESSION.txt').write_text('\n'.join(readme_lines) + '\n', encoding='utf-8')
PY
}

for CURRENT_RUN_VARIANT in "${VARIANT_NAMES[@]}"; do
  maybe_prepare_explicit_variant_rerun "$CURRENT_RUN_VARIANT"
  set_variant_context "$CURRENT_RUN_VARIANT"
  : > "$RUN_METADATA_TSV"
  if [[ "$SESSION_WORK_MODE" == "backfill_only" ]]; then
    echo "[INFO] Processing variant in accuracy-only precision-specific mode: $CURRENT_RUN_VARIANT"
    if [[ "$CURRENT_RUN_VARIANT" == "per_image" ]]; then
      show_oled_message "ACC ONLY" "PERIMG variant" "no full rerun"
    else
      show_oled_message "ACC ONLY" "STD variant" "no full rerun"
    fi
    run_backfill_for_session_root "$SESSION_ROOT"
    for idx in "${!PRECISION_LABELS[@]}"; do
      record_reusable_precision_run "${PRECISION_LABELS[$idx]}" "${PRECISION_MODEL_PATHS[$idx]}"
    done
  else
    echo "[INFO] Starting variant: $CURRENT_RUN_VARIANT"
    if [[ "$CURRENT_RUN_VARIANT" == "per_image" ]]; then
      show_oled_message "VARIANT PERIMG" "per-image logs" "memory+energy"
    else
      show_oled_message "VARIANT STD" "summary only" "no per-image"
    fi
    for idx in "${!PRECISION_LABELS[@]}"; do
      materialize_precision_run "${PRECISION_LABELS[$idx]}" "${PRECISION_MODEL_PATHS[$idx]}"
    done
  fi
  record_unselected_reusable_precision_runs
  finalize_variant_session
done

"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

session_root = Path(os.environ['SESSION_ROOT']).expanduser().resolve()
selected_variant_names = [item for item in os.environ.get('RUN_VARIANTS_CANONICAL', '').split(',') if item]
existing_variant_names = []
for candidate in ('standard', 'per_image'):
    variant_root = session_root / candidate
    if variant_root.is_dir():
        existing_variant_names.append(candidate)
variant_names = []
for name in existing_variant_names + selected_variant_names:
    if name and name not in variant_names:
        variant_names.append(name)
bundle = {
    'session_name': os.environ['SESSION_NAME'],
    'session_root': str(session_root),
    'timestamp_utc': os.environ['SESSION_TIMESTAMP'],
    'board_name': os.environ['BOARD_NAME'],
    'power_mode_detection_json': str(session_root / 'detected_power_mode.json'),
    'effective_session_settings_json': str(session_root / 'effective_session_settings.json'),
    'session_noise_guard_summary_json': str(session_root / 'session_noise_guard_summary.json'),
    'variants': [],
}
readme_lines = [
    'Session bundle outputs:',
    '- detected_power_mode.json',
    '- effective_session_settings.json',
]
for variant_name in variant_names:
    variant_root = session_root / variant_name
    manifest_path = variant_root / 'paper_submission_manifest.json'
    readme_lines.append(f'- {variant_name}/paper_submission_manifest.json')
    bundle['variants'].append(
        {
            'run_variant': variant_name,
            'variant_root': str(variant_root),
            'manifest_json': str(manifest_path),
            'readme_txt': str(variant_root / 'README_SESSION.txt'),
        }
    )

(session_root / 'paper_submission_manifest.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
(session_root / 'README_SESSION.txt').write_text('\n'.join(readme_lines) + '\n', encoding='utf-8')
PY

echo "Session root: $SESSION_ROOT"
for CURRENT_RUN_VARIANT in "${VARIANT_NAMES[@]}"; do
  VARIANT_SESSION_ROOT="$SESSION_ROOT/$CURRENT_RUN_VARIANT"
  echo "Variant root [$CURRENT_RUN_VARIANT]: $VARIANT_SESSION_ROOT"
done
echo "Bundle manifest: $SESSION_ROOT/paper_submission_manifest.json"
