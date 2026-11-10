#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_conda_env
PYBIN="${PYBIN:-python}"

ONNX_PATH="${1:-$MODEL_DIR/best.onnx}"
OUT_DIR="${2:-$MODEL_DIR}"
DATASET_ROOT_ARG="${3:-${INT8_CALIB_DATASET_ROOT:-$DATASET_SOURCE}}"
MANIFEST_ARG="${4:-${INT8_CALIB_MANIFEST:-}}"
IMGSZ="${5:-640}"
WORKSPACE_MB="${6:-4096}"
MAX_IMAGES="${7:-256}"
INT8_BUILD_IMPL="${INT8_BUILD_IMPL:-cpp}"
INT8_CACHE_PATH="${INT8_CALIB_CACHE_PATH:-$MODEL_DIR/best_int8_calibration.cache}"
INT8_ENGINE_PATH="${INT8_MODEL_PATH:-$MODEL_DIR/best_int8.engine}"
INT8_META_PATH="${INT8_META_PATH:-$MODEL_DIR/best_int8_cache_build_meta.json}"
INT8_LOG_PATH="${INT8_LOG_PATH:-$MODEL_DIR/best_int8_build.log}"
CPP_SOURCE="$PKG_ROOT/jetson/build_int8_calibration_cache.cpp"
CPP_BINARY_DIR="$PKG_ROOT/.build"
CPP_BINARY="$CPP_BINARY_DIR/build_int8_calibration_cache"

resolve_dataset_and_manifest() {
  "$PYBIN" - "$PKG_ROOT" "$DATASET_ROOT_ARG" "$MANIFEST_ARG" <<'PY'
from pathlib import Path
import sys

pkg_root = Path(sys.argv[1]).expanduser().resolve()
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from jetson.local_paths import resolve_dataset_root, resolve_inference_bundle_root, resolve_manifest_path

dataset_hint = sys.argv[2]
manifest_hint = sys.argv[3] or None
bundle_root = resolve_inference_bundle_root(dataset_hint)
dataset_root = resolve_dataset_root(dataset_hint, bundle_root=bundle_root)
manifest_path = resolve_manifest_path(dataset_root, manifest_hint, bundle_root=bundle_root)
print(dataset_root)
print(manifest_path or "")
PY
}

compile_cpp_helper() {
  mkdir -p "$CPP_BINARY_DIR"
  local needs_build="0"
  if [[ ! -x "$CPP_BINARY" ]]; then
    needs_build="1"
  elif [[ "$CPP_SOURCE" -nt "$CPP_BINARY" ]]; then
    needs_build="1"
  fi

  if [[ "$needs_build" != "1" ]]; then
    return 0
  fi

  local opencv_flags
  local opencv_cflags
  local opencv_libs
  local cuda_include=""
  local cuda_libs=""
  if pkg-config --exists opencv4; then
    opencv_cflags="$(pkg-config --cflags opencv4)"
    opencv_libs="$(pkg-config --libs opencv4)"
  elif pkg-config --exists opencv; then
    opencv_cflags="$(pkg-config --cflags opencv)"
    opencv_libs="$(pkg-config --libs opencv)"
  else
    echo "OpenCV development files not found via pkg-config." >&2
    exit 1
  fi

  if [[ -d /usr/local/cuda/include ]]; then
    cuda_include="-I/usr/local/cuda/include"
  elif [[ -d /usr/local/cuda-12.6/include ]]; then
    cuda_include="-I/usr/local/cuda-12.6/include"
  else
    echo "CUDA headers not found under /usr/local/cuda/include or /usr/local/cuda-12.6/include." >&2
    exit 1
  fi

  if [[ -d /usr/local/cuda/lib64 ]]; then
    cuda_libs="-L/usr/local/cuda/lib64"
  elif [[ -d /usr/local/cuda-12.6/lib64 ]]; then
    cuda_libs="-L/usr/local/cuda-12.6/lib64"
  else
    echo "CUDA runtime libs not found under /usr/local/cuda/lib64 or /usr/local/cuda-12.6/lib64." >&2
    exit 1
  fi

  g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include/aarch64-linux-gnu \
    -I/usr/include/opencv4 \
    -I/usr/local/include/opencv4 \
    $cuda_include \
    $opencv_cflags \
    "$CPP_SOURCE" \
    -o "$CPP_BINARY" \
    $cuda_libs \
    $opencv_libs \
    -lnvinfer -lnvonnxparser -lnvinfer_plugin -lcudart
}

mapfile -t RESOLVED <<<"$(resolve_dataset_and_manifest)"
DATASET_ROOT="${RESOLVED[0]}"
RESOLVED_MANIFEST="${RESOLVED[1]:-}"

if [[ "${INT8_BUILD_IMPL,,}" == "cpp" ]]; then
  compile_cpp_helper
  CMD=(
    "$CPP_BINARY"
    --onnx "$ONNX_PATH"
    --output-dir "$OUT_DIR"
    --dataset-root "$DATASET_ROOT"
    --cache-path "$INT8_CACHE_PATH"
    --engine-path "$INT8_ENGINE_PATH"
    --meta-path "$INT8_META_PATH"
    --imgsz "$IMGSZ"
    --workspace-mb "$WORKSPACE_MB"
    --max-images "$MAX_IMAGES"
  )
  if [[ -n "$RESOLVED_MANIFEST" ]]; then
    CMD+=(--manifest "$RESOLVED_MANIFEST")
  fi
else
  CMD=(
    python "$PKG_ROOT/jetson/build_int8_calibration_cache.py"
    --onnx "$ONNX_PATH"
    --output-dir "$OUT_DIR"
    --dataset-root "$DATASET_ROOT"
    --cache-path "$INT8_CACHE_PATH"
    --engine-path "$INT8_ENGINE_PATH"
    --imgsz "$IMGSZ"
    --workspace-mb "$WORKSPACE_MB"
    --max-images "$MAX_IMAGES"
  )
  if [[ -n "$RESOLVED_MANIFEST" ]]; then
    CMD+=(--manifest "$RESOLVED_MANIFEST")
  fi
fi

mkdir -p "$(dirname "$INT8_LOG_PATH")"
"${CMD[@]}" 2>&1 | tee "$INT8_LOG_PATH"
