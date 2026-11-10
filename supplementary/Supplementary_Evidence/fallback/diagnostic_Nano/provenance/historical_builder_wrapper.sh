#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_build_conda_env
activate_local_tensorrt_stack
PYBIN="${PYBIN:-python}"

ONNX_PATH="${1:-$MODEL_DIR/best.onnx}"
OUT_DIR="${2:-$MODEL_DIR}"
DATASET_ROOT_ARG="${3:-${INT8_CALIB_DATASET_ROOT:-$DATASET_SOURCE}}"
MANIFEST_ARG="${4:-${INT8_CALIB_MANIFEST:-}}"
IMGSZ="${5:-640}"
WORKSPACE_MB="${6:-2048}"
MAX_IMAGES="${7:-256}"
INT8_BUILD_IMPL="${INT8_BUILD_IMPL:-cpp}"
INT8_CACHE_PATH="${INT8_CALIB_CACHE_PATH:-$MODEL_DIR/best_int8_calibration.cache}"
INT8_ENGINE_PATH="${INT8_MODEL_PATH:-$MODEL_DIR/best_int8.engine}"
INT8_META_PATH="${INT8_META_PATH:-$MODEL_DIR/best_int8_cache_build_meta.json}"
INT8_BUILD_LOG_PATH="${INT8_BUILD_LOG_PATH:-$OUT_DIR/$(basename "${ONNX_PATH%.*}")_int8_build.log}"
CPP_SOURCE="$PKG_ROOT/jetson/build_int8_calibration_cache.cpp"
CPP_BINARY_DIR="$PKG_ROOT/.build"
CPP_BINARY="$CPP_BINARY_DIR/build_int8_calibration_cache"
PATCH_ONNX_TOOL="$PKG_ROOT/scripts/prepare_tensorrt_build_onnx.py"
PATCHED_ONNX_PATH="${INT8_PATCHED_ONNX_PATH:-$OUT_DIR/$(basename "${ONNX_PATH%.*}")_int8_trt.onnx}"
INT8_BUILD_ONNX_PATH="$ONNX_PATH"
INT8_ALLOW_FP16_FALLBACK="${INT8_ALLOW_FP16_FALLBACK:-1}"

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
  local trt_include="-I/usr/include/aarch64-linux-gnu"
  local trt_libs=""
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

  if [[ -d "${LOCAL_TENSORRT_INCLUDE_DIR:-}" ]]; then
    trt_include="-I$LOCAL_TENSORRT_INCLUDE_DIR"
  fi
  if [[ -d "${LOCAL_TENSORRT_LIB_DIR:-}" ]]; then
    trt_libs="-L$LOCAL_TENSORRT_LIB_DIR"
  fi

  g++ -std=c++17 -O2 -Wall -Wextra \
    $trt_include \
    -I/usr/include/opencv4 \
    -I/usr/local/include/opencv4 \
    $cuda_include \
    $opencv_cflags \
    "$CPP_SOURCE" \
    -o "$CPP_BINARY" \
    $cuda_libs \
    $trt_libs \
    $opencv_libs \
    -lnvinfer -lnvonnxparser -lnvinfer_plugin -lcudart
}

mapfile -t RESOLVED <<<"$(resolve_dataset_and_manifest)"
DATASET_ROOT="${RESOLVED[0]}"
RESOLVED_MANIFEST="${RESOLVED[1]:-}"

if [[ -f "$PATCH_ONNX_TOOL" ]]; then
  "$PYBIN" "$PATCH_ONNX_TOOL" --onnx "$ONNX_PATH" --out "$PATCHED_ONNX_PATH" --imgsz "$IMGSZ"
  if [[ -f "$PATCHED_ONNX_PATH" ]]; then
    INT8_BUILD_ONNX_PATH="$PATCHED_ONNX_PATH"
  fi
fi

if [[ "${INT8_BUILD_IMPL,,}" == "cpp" ]]; then
  compile_cpp_helper
  CMD=(
    "$CPP_BINARY"
    --onnx "$INT8_BUILD_ONNX_PATH"
    --output-dir "$OUT_DIR"
    --dataset-root "$DATASET_ROOT"
    --cache-path "$INT8_CACHE_PATH"
    --engine-path "$INT8_ENGINE_PATH"
    --meta-path "$INT8_META_PATH"
    --imgsz "$IMGSZ"
    --workspace-mb "$WORKSPACE_MB"
    --max-images "$MAX_IMAGES"
  )
  if [[ "${INT8_ALLOW_FP16_FALLBACK,,}" == "0" || "${INT8_ALLOW_FP16_FALLBACK,,}" == "false" || "${INT8_ALLOW_FP16_FALLBACK,,}" == "no" ]]; then
    CMD+=(--no-fp16-fallback)
  else
    CMD+=(--allow-fp16-fallback)
  fi
  if [[ -n "$RESOLVED_MANIFEST" ]]; then
    CMD+=(--manifest "$RESOLVED_MANIFEST")
  fi
else
  CMD=(
    python "$PKG_ROOT/jetson/build_int8_calibration_cache.py"
    --onnx "$INT8_BUILD_ONNX_PATH"
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

mkdir -p "$(dirname "$INT8_BUILD_LOG_PATH")"
set +e
"${CMD[@]}" 2>&1 | tee "$INT8_BUILD_LOG_PATH"
cmd_rc=${PIPESTATUS[0]}
set -e
if [[ $cmd_rc -ne 0 ]]; then
  cat >&2 <<EOF
INT8 build failed for this model on the current TensorRT stack.
Saved build log: $INT8_BUILD_LOG_PATH
Most likely cause: TensorRT calibrated INT8 builder cannot handle the current SegFormer-B0 ONNX graph on this JetPack/TensorRT version.
FP16/FP32 build and benchmark can still proceed normally.
EOF
  exit "$cmd_rc"
fi

"$PYBIN" - "$INT8_META_PATH" "$INT8_ENGINE_PATH" "$INT8_CACHE_PATH" "$ONNX_PATH" "$INT8_BUILD_ONNX_PATH" "$INT8_BUILD_LOG_PATH" "$INT8_ALLOW_FP16_FALLBACK" "$RESOLVED_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1]).expanduser().resolve()
engine_path = Path(sys.argv[2]).expanduser().resolve()
cache_path = Path(sys.argv[3]).expanduser().resolve()
source_onnx_path = Path(sys.argv[4]).expanduser().resolve()
build_onnx_path = Path(sys.argv[5]).expanduser().resolve()
build_log_path = Path(sys.argv[6]).expanduser().resolve()
allow_fp16_raw = sys.argv[7].strip().lower()
manifest_raw = sys.argv[8].strip()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
allow_fp16 = allow_fp16_raw not in {"0", "false", "no"}
data.update(
    {
        "onnx_path": str(source_onnx_path),
        "source_onnx_path": str(source_onnx_path),
        "build_onnx_path": str(build_onnx_path),
        "engine_path": str(engine_path),
        "cache_path": str(cache_path),
        "build_log_path": str(build_log_path),
        "sha256_engine": sha256_file(engine_path) if engine_path.exists() else None,
        "sha256_cache": sha256_file(cache_path) if cache_path.exists() else None,
        "sha256_onnx": sha256_file(source_onnx_path) if source_onnx_path.exists() else None,
        "sha256_build_onnx": sha256_file(build_onnx_path) if build_onnx_path.exists() else None,
        "sha256_build_log": sha256_file(build_log_path) if build_log_path.exists() else None,
        "precision_mode": "mixed_int8_fp16_fallback" if allow_fp16 else "strict_int8",
        "allow_fp16_fallback": allow_fp16,
        "calibration_signature_version": "segformerb0_rgb_letterbox114_scale_v1",
        "calibration_preprocess": {
            "mode": "rgb_letterbox_114_normalized",
            "resize": "preserve_aspect_ratio_640x640_letterbox",
            "image_interpolation": "bilinear",
            "pad_value_rgb": [114, 114, 114],
            "scale": "1/255",
            "mean": None,
            "std": None,
            "layout": "NCHW",
            "dtype": "float32",
            "batch_size": 1,
        },
    }
)
if manifest_raw:
    data["manifest_path"] = str(Path(manifest_raw).expanduser().resolve())
meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(data, indent=2))
PY
