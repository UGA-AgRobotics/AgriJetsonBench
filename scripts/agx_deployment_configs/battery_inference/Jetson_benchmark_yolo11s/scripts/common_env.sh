#!/usr/bin/env bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG_ROOT="${PKG_ROOT:-$(cd "$COMMON_DIR/.." && pwd)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$PKG_ROOT/.." && pwd)}"
if [[ -z "${YOLO11S_ROOT:-}" ]]; then
  if [[ -d "$BENCHMARK_ROOT/yolo11s/inference_bundle" ]]; then
    export YOLO11S_ROOT="$BENCHMARK_ROOT/yolo11s"
  elif [[ -d "$BENCHMARK_ROOT/yolo11s_baseline_v1_jetson_inference_bundle/inference_bundle" ]]; then
    export YOLO11S_ROOT="$BENCHMARK_ROOT/yolo11s_baseline_v1_jetson_inference_bundle"
  else
    export YOLO11S_ROOT="$BENCHMARK_ROOT/yolo11s"
  fi
fi
export INFERENCE_BUNDLE_ROOT="${INFERENCE_BUNDLE_ROOT:-$YOLO11S_ROOT/inference_bundle}"
export MODEL_DIR="${MODEL_DIR:-$INFERENCE_BUNDLE_ROOT/model}"
export ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-$MODEL_DIR/best.onnx}"
export YOLO11S_SOURCE_CHECKPOINT_PATH="${YOLO11S_SOURCE_CHECKPOINT_PATH:-$MODEL_DIR/best.pt}"
export FP16_MODEL_PATH="${FP16_MODEL_PATH:-$MODEL_DIR/best_fp16.engine}"
export FP32_MODEL_PATH="${FP32_MODEL_PATH:-$MODEL_DIR/best_fp32.engine}"
export INT8_MODEL_PATH="${INT8_MODEL_PATH:-$MODEL_DIR/best_int8.engine}"
export INT8_CALIB_CACHE_PATH="${INT8_CALIB_CACHE_PATH:-$MODEL_DIR/best_int8_calibration.cache}"
export DATASET_ZIP_PATH="${DATASET_ZIP_PATH:-$INFERENCE_BUNDLE_ROOT/dataset/CottonWeedDet12_testset_for_jetson.zip}"
export DATA_ROOT="${DATA_ROOT:-$INFERENCE_BUNDLE_ROOT/dataset/CottonWeedDet12_testset_for_jetson}"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$BENCHMARK_ROOT/runs}"
export BENCHMARK_OUT_ROOT="${BENCHMARK_OUT_ROOT:-$OUTPUT_BASE_DIR/jetson_benchmark_runs}"
export CALIBRATION_OUT_ROOT="${CALIBRATION_OUT_ROOT:-$OUTPUT_BASE_DIR/jetson_calibration_runs}"
export PI_LOGGER_OUTPUT_ROOT="${PI_LOGGER_OUTPUT_ROOT:-$OUTPUT_BASE_DIR/jetson_external_logs}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-SAM}"
export MINICONDA_ROOT="${MINICONDA_ROOT:-$HOME/miniconda3}"
export PI_ENV_MODE="${PI_ENV_MODE:-auto}"
export PYTHON_VENV_PATH="${PYTHON_VENV_PATH:-$HOME/envs/jetson_pi_logger}"
export PYTHON_SITE_POLICY="${PYTHON_SITE_POLICY:-auto}"
export LOCAL_TENSORRT_ROOT="${LOCAL_TENSORRT_ROOT:-$BENCHMARK_ROOT/.vendor/tensorrt-${PRIVATE_LAN_IP}}"
export LOCAL_TENSORRT_LIB_DIR="${LOCAL_TENSORRT_LIB_DIR:-$LOCAL_TENSORRT_ROOT/usr/lib/aarch64-linux-gnu}"
export LOCAL_TENSORRT_INCLUDE_DIR="${LOCAL_TENSORRT_INCLUDE_DIR:-$LOCAL_TENSORRT_ROOT/usr/include/aarch64-linux-gnu}"
export LOCAL_TRTEXEC_PATH="${LOCAL_TRTEXEC_PATH:-$LOCAL_TENSORRT_ROOT/usr/src/tensorrt/bin/trtexec}"
export LOCAL_TENSORRT_SETUP_SCRIPT="${LOCAL_TENSORRT_SETUP_SCRIPT:-$PKG_ROOT/scripts/setup_local_tensorrt_104.sh}"
export USE_LOCAL_TENSORRT_STACK="${USE_LOCAL_TENSORRT_STACK:-auto}"
export TENSORRT_STACK_TAG="${TENSORRT_STACK_TAG:-system-trt}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -f "$MODEL_DIR/best_fp16.engine" ]]; then
    export MODEL_PATH="$MODEL_DIR/best_fp16.engine"
  elif [[ -f "$MODEL_DIR/best.engine" ]]; then
    export MODEL_PATH="$MODEL_DIR/best.engine"
  elif [[ -f "$MODEL_DIR/best.onnx" ]]; then
    export MODEL_PATH="$MODEL_DIR/best.onnx"
  else
    export MODEL_PATH="$MODEL_DIR/best.pt"
  fi
fi

if [[ -z "${ACCURACY_MODEL:-}" ]]; then
  export ACCURACY_MODEL="$MODEL_PATH"
fi

if [[ -f "$DATASET_ZIP_PATH" && ! -d "$DATA_ROOT" ]]; then
  export DATASET_SOURCE="${DATASET_SOURCE:-$DATASET_ZIP_PATH}"
else
  export DATASET_SOURCE="${DATASET_SOURCE:-$DATA_ROOT}"
fi

load_conda() {
  if declare -F conda >/dev/null 2>&1; then
    return 0
  fi

  local conda_base=""
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
  fi
  if [[ -z "$conda_base" && -f "$MINICONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    conda_base="$MINICONDA_ROOT"
  fi
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found. Set MINICONDA_ROOT or install Miniconda first." >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
}

ensure_conda_env() {
  local env_name="${1:-$CONDA_ENV_NAME}"
  local python_version="${2:-3.11}"
  load_conda
  if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"; then
    conda create -y -n "$env_name" "python=$python_version"
  fi
}

activate_conda_env() {
  local env_name="${1:-$CONDA_ENV_NAME}"
  load_conda
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
    conda activate "$env_name"
  fi
  configure_python_site_policy
}

conda_env_exists() {
  local env_name="$1"
  load_conda
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"
}

activate_build_conda_env() {
  activate_conda_env "${1:-$CONDA_ENV_NAME}"
}

local_tensorrt_available() {
  [[ -x "$LOCAL_TRTEXEC_PATH" && -d "$LOCAL_TENSORRT_LIB_DIR" ]]
}

ensure_local_tensorrt_stack() {
  case "${USE_LOCAL_TENSORRT_STACK,,}" in
    0|false|no)
      return 0
      ;;
  esac
  if local_tensorrt_available; then
    return 0
  fi
  if [[ -x "$LOCAL_TENSORRT_SETUP_SCRIPT" ]]; then
    "$LOCAL_TENSORRT_SETUP_SCRIPT"
  fi
}

activate_local_tensorrt_stack() {
  case "${USE_LOCAL_TENSORRT_STACK,,}" in
    0|false|no)
      return 0
      ;;
  esac
  ensure_local_tensorrt_stack
  if ! local_tensorrt_available; then
    return 0
  fi
  export TRTEXEC_PATH="$LOCAL_TRTEXEC_PATH"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$LOCAL_TENSORRT_LIB_DIR:"*) ;;
    *) export LD_LIBRARY_PATH="$LOCAL_TENSORRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  export TENSORRT_STACK_TAG="local-trt-${PRIVATE_LAN_IP}"
}

have_python_venv() {
  local venv_path="${1:-$PYTHON_VENV_PATH}"
  [[ -f "$venv_path/bin/activate" ]]
}

activate_python_venv() {
  local venv_path="${1:-$PYTHON_VENV_PATH}"
  if [[ ! -f "$venv_path/bin/activate" ]]; then
    echo "Python venv not found at $venv_path. Set PYTHON_VENV_PATH correctly." >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$venv_path/bin/activate"
  configure_python_site_policy
}

detect_pi_env_mode() {
  local requested_mode="${1:-$PI_ENV_MODE}"
  case "$requested_mode" in
    auto)
      if have_python_venv "$PYTHON_VENV_PATH"; then
        echo "venv"
      else
        echo "conda"
      fi
      ;;
    venv|conda)
      echo "$requested_mode"
      ;;
    *)
      echo "Unsupported PI_ENV_MODE: $requested_mode (expected auto, venv, or conda)" >&2
      return 1
      ;;
  esac
}

activate_pi_env() {
  local mode
  mode="$(detect_pi_env_mode "${1:-$PI_ENV_MODE}")"
  if [[ "$mode" == "venv" ]]; then
    activate_python_venv "$PYTHON_VENV_PATH"
  else
    activate_conda_env "${2:-$CONDA_ENV_NAME}"
  fi
}

configure_python_site_policy() {
  local policy="${PYTHON_SITE_POLICY:-auto}"
  case "$policy" in
    isolated)
      export PYTHONNOUSERSITE=1
      return 0
      ;;
    usersite)
      unset PYTHONNOUSERSITE
      return 0
      ;;
    auto)
      ;;
    *)
      echo "Unsupported PYTHON_SITE_POLICY: $policy (expected auto, isolated, or usersite)" >&2
      return 1
      ;;
  esac

  local python_bin="${PYTHON_BIN:-python}"
  local py_ver
  py_ver="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  local active_prefix="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
  local active_site=""
  local user_site="$HOME/.local/lib/python${py_ver}/site-packages"
  if [[ -n "$active_prefix" ]]; then
    active_site="$active_prefix/lib/python${py_ver}/site-packages"
  fi

  if [[ -n "$active_site" ]] \
    && [[ ! -e "$active_site/torch" ]] \
    && [[ ! -e "$active_site/torchvision" ]] \
    && { [[ -e "$user_site/torch" ]] || [[ -e "$user_site/torchvision" ]]; }; then
    unset PYTHONNOUSERSITE
  else
    export PYTHONNOUSERSITE=1
  fi
}
