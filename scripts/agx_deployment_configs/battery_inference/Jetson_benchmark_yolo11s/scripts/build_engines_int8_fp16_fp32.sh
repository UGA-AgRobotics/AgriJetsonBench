#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_build_conda_env
activate_local_tensorrt_stack
PYBIN="${PYBIN:-python}"
OLED_MESSAGE_TOOL="$PKG_ROOT/tools/show_oled_message.py"
DEFAULT_STATUS_UPDATE_URL=""
if [[ -n "${PI_LOGGER_URL:-}" ]]; then
  DEFAULT_STATUS_UPDATE_URL="${PI_LOGGER_URL%/}/api/jetson_status"
fi
STATUS_UPDATE_URL="${STATUS_UPDATE_URL:-$DEFAULT_STATUS_UPDATE_URL}"

ONNX_PATH="${1:-$ONNX_MODEL_PATH}"
OUT_DIR="${2:-$MODEL_DIR}"
IMGSZ="${3:-640}"
WORKSPACE_MB="${4:-4096}"
MAX_IMAGES="${5:-256}"
BUILD_INT8="${BUILD_INT8:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CURRENT_BUILD_LABEL=""

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
  local state="${2:-building}"
  local message="${3:-}"
  local precision="${4:-}"
  if [[ -z "$STATUS_UPDATE_URL" ]]; then
    return 0
  fi
  "$PYBIN" - "$STATUS_UPDATE_URL" "$stage" "$state" "$message" "$precision" <<'PY' >/dev/null 2>&1 || true
import json
import sys
import time
import urllib.request
import os

url, stage, state, message, precision = sys.argv[1:6]
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
    "run_variant": "engine_build",
}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=4):
    pass
PY
}

emit_build_status() {
  local stage="$1"
  local state="$2"
  local message="$3"
  local precision="${4:-}"
  local oled_l1="${5:-$stage}"
  local oled_l2="${6:-$message}"
  local oled_l3="${7:-}"
  post_status_update "$stage" "$state" "$message" "$precision"
  show_oled_message "$oled_l1" "$oled_l2" "$oled_l3"
}

on_build_error() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  local label="${CURRENT_BUILD_LABEL:-ENGINE}"
  emit_build_status "BUILD ERROR" "error" "Build failed for ${label}" "$label" "BUILD ERROR" "${label}" "check terminal"
  exit "$rc"
}

trap on_build_error ERR

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ONNX file not found: $ONNX_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

ONNX_BASENAME="$(basename "$ONNX_PATH")"
ONNX_STEM="${ONNX_BASENAME%.*}"
FP16_ENGINE_PATH="$OUT_DIR/${ONNX_STEM}_fp16.engine"
FP32_ENGINE_PATH="$OUT_DIR/${ONNX_STEM}_fp32.engine"
INT8_ENGINE_PATH_LOCAL="$OUT_DIR/${ONNX_STEM}_int8.engine"
INT8_CACHE_PATH_LOCAL="$OUT_DIR/${ONNX_STEM}_int8_calibration.cache"
INT8_META_PATH_LOCAL="$OUT_DIR/${ONNX_STEM}_int8_cache_build_meta.json"

build_fp16() {
  if [[ "$FORCE_REBUILD" != "1" && -s "$FP16_ENGINE_PATH" ]]; then
    echo "[INFO] Skip fp16 build because the engine already exists: $FP16_ENGINE_PATH"
    return 0
  fi
  CURRENT_BUILD_LABEL="FP16"
  echo "[INFO] Building fp16 engine..."
  emit_build_status "BUILD FP16" "building" "Building fp16 TensorRT engine from best.onnx" "fp16" "BUILD FP16" "TensorRT eng" "please wait"
  bash "$SCRIPT_DIR/build_engine_fp16.sh" "$ONNX_PATH" "$OUT_DIR" "$IMGSZ" "$WORKSPACE_MB"
  emit_build_status "FP16 READY" "ready" "fp16 TensorRT engine built successfully" "fp16" "FP16 READY" "engine saved" "$(basename "$FP16_ENGINE_PATH")"
}

build_fp32() {
  if [[ "$FORCE_REBUILD" != "1" && -s "$FP32_ENGINE_PATH" ]]; then
    echo "[INFO] Skip fp32 build because the engine already exists: $FP32_ENGINE_PATH"
    return 0
  fi
  CURRENT_BUILD_LABEL="FP32"
  echo "[INFO] Building fp32 engine..."
  emit_build_status "BUILD FP32" "building" "Building fp32 TensorRT engine from best.onnx" "fp32" "BUILD FP32" "TensorRT eng" "please wait"
  bash "$SCRIPT_DIR/build_engine_fp32.sh" "$ONNX_PATH" "$OUT_DIR" "$IMGSZ" "$WORKSPACE_MB"
  emit_build_status "FP32 READY" "ready" "fp32 TensorRT engine built successfully" "fp32" "FP32 READY" "engine saved" "$(basename "$FP32_ENGINE_PATH")"
}

build_int8() {
  case "${BUILD_INT8,,}" in
    0|false|no)
      echo "[INFO] INT8 build disabled."
      return 0
      ;;
  esac

  if [[ "$FORCE_REBUILD" != "1" && -s "$INT8_ENGINE_PATH_LOCAL" && -s "$INT8_CACHE_PATH_LOCAL" ]]; then
    echo "[INFO] Skip int8 build because the engine/cache already exist: $INT8_ENGINE_PATH_LOCAL"
    return 0
  fi

  CURRENT_BUILD_LABEL="INT8"
  echo "[INFO] Building int8 cache + engine..."
  emit_build_status "BUILD INT8" "building" "Building int8 calibration cache and TensorRT engine" "int8" "BUILD INT8" "cache + eng" "please wait"
  INT8_MODEL_PATH="$INT8_ENGINE_PATH_LOCAL" \
  INT8_CALIB_CACHE_PATH="$INT8_CACHE_PATH_LOCAL" \
  INT8_META_PATH="$INT8_META_PATH_LOCAL" \
  bash "$SCRIPT_DIR/build_int8_cache_and_engine.sh" \
    "$ONNX_PATH" \
    "$OUT_DIR" \
    "${INT8_CALIB_DATASET_ROOT:-$DATASET_SOURCE}" \
    "${INT8_CALIB_MANIFEST:-}" \
    "$IMGSZ" \
    "$WORKSPACE_MB" \
    "$MAX_IMAGES"
  emit_build_status "INT8 READY" "ready" "int8 cache and engine built successfully" "int8" "INT8 READY" "cache+engine" "$(basename "$INT8_ENGINE_PATH_LOCAL")"
}

emit_build_status "ENGINE CHECK" "building" "Checking and building missing TensorRT engines" "" "ENGINE CHECK" "fp16/fp32/int8" "starting"
build_fp16
build_fp32
build_int8
CURRENT_BUILD_LABEL=""
emit_build_status "BUILD COMPLETE" "ready" "Required TensorRT engines are ready" "" "BUILD DONE" "engines ready" ""

echo "[INFO] Engine build summary:"
echo "  fp16: $FP16_ENGINE_PATH"
echo "  fp32: $FP32_ENGINE_PATH"
case "${BUILD_INT8,,}" in
  0|false|no)
    echo "  int8: skipped"
    ;;
  *)
    echo "  int8 engine: $INT8_ENGINE_PATH_LOCAL"
    echo "  int8 cache:  $INT8_CACHE_PATH_LOCAL"
    ;;
esac
