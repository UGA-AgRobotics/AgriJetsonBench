#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_build_conda_env
activate_local_tensorrt_stack

PRECISION="${1:-fp16}"
ONNX_PATH="${2:-$MODEL_DIR/best.onnx}"
OUT_DIR="${3:-$MODEL_DIR}"
IMGSZ="${4:-640}"
WORKSPACE_MB="${5:-2048}"
DEFAULT_CALIB_CACHE="${CALIB_CACHE_PATH:-${INT8_CALIB_CACHE_PATH:-}}"
CALIB_CACHE="${6:-$DEFAULT_CALIB_CACHE}"
ENGINE_PATH_OVERRIDE="${ENGINE_PATH_OVERRIDE:-}"
ENGINE_LOG_PATH_OVERRIDE="${ENGINE_LOG_PATH_OVERRIDE:-}"
ENGINE_META_PATH_OVERRIDE="${ENGINE_META_PATH_OVERRIDE:-}"

CMD=(
  python "$PKG_ROOT/jetson/build_tensorrt_engine.py"
  --onnx "$ONNX_PATH"
  --output-dir "$OUT_DIR"
  --imgsz "$IMGSZ"
  --workspace-mb "$WORKSPACE_MB"
  --precision "$PRECISION"
)

if [[ -n "$CALIB_CACHE" ]]; then
  CMD+=(--calib-cache "$CALIB_CACHE")
fi
if [[ -n "$ENGINE_PATH_OVERRIDE" ]]; then
  CMD+=(--engine-path "$ENGINE_PATH_OVERRIDE")
fi
if [[ -n "$ENGINE_LOG_PATH_OVERRIDE" ]]; then
  CMD+=(--log-path "$ENGINE_LOG_PATH_OVERRIDE")
fi
if [[ -n "$ENGINE_META_PATH_OVERRIDE" ]]; then
  CMD+=(--meta-path "$ENGINE_META_PATH_OVERRIDE")
fi

"${CMD[@]}"
