#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env

PRECISION="${1:-fp16}"
ONNX_PATH="${2:-$MODEL_DIR/best.onnx}"
OUT_DIR="${3:-$MODEL_DIR}"
IMGSZ="${4:-640}"
WORKSPACE_MB="${5:-4096}"
DEFAULT_CALIB_CACHE="${CALIB_CACHE_PATH:-${INT8_CALIB_CACHE_PATH:-}}"
CALIB_CACHE="${6:-$DEFAULT_CALIB_CACHE}"

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

"${CMD[@]}"
