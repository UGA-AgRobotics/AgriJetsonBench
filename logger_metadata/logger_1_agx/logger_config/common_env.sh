#!/usr/bin/env bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG_ROOT="${PKG_ROOT:-$(cd "$COMMON_DIR/.." && pwd)}"
export BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$PKG_ROOT/.." && pwd)}"
export YOLO11N_ROOT="${YOLO11N_ROOT:-$BENCHMARK_ROOT/yolo11n}"
export INFERENCE_BUNDLE_ROOT="${INFERENCE_BUNDLE_ROOT:-$YOLO11N_ROOT/inference_bundle}"
export MODEL_DIR="${MODEL_DIR:-$INFERENCE_BUNDLE_ROOT/model}"
export ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-$MODEL_DIR/best.onnx}"
export FP16_MODEL_PATH="${FP16_MODEL_PATH:-$MODEL_DIR/best_fp16.engine}"
export FP32_MODEL_PATH="${FP32_MODEL_PATH:-$MODEL_DIR/best_fp32.engine}"
export DATASET_ZIP_PATH="${DATASET_ZIP_PATH:-$INFERENCE_BUNDLE_ROOT/dataset/CottonWeedDet12_testset_for_jetson.zip}"
export DATA_ROOT="${DATA_ROOT:-$INFERENCE_BUNDLE_ROOT/dataset/CottonWeedDet12_testset_for_jetson}"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$BENCHMARK_ROOT/runs}"
export BENCHMARK_OUT_ROOT="${BENCHMARK_OUT_ROOT:-$OUTPUT_BASE_DIR/jetson_benchmark_runs}"
export CALIBRATION_OUT_ROOT="${CALIBRATION_OUT_ROOT:-$OUTPUT_BASE_DIR/jetson_calibration_runs}"
export PC_LOGGER_OUTPUT_ROOT="${PC_LOGGER_OUTPUT_ROOT:-${PI_LOGGER_OUTPUT_ROOT:-$OUTPUT_BASE_DIR/pc_external_logs}}"
export PI_LOGGER_OUTPUT_ROOT="${PI_LOGGER_OUTPUT_ROOT:-$PC_LOGGER_OUTPUT_ROOT}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-SAM}"
export MINICONDA_ROOT="${MINICONDA_ROOT:-$HOME/miniconda3}"
export PC_ENV_MODE="${PC_ENV_MODE:-${PI_ENV_MODE:-venv}}"
export PI_ENV_MODE="${PI_ENV_MODE:-$PC_ENV_MODE}"
export PYTHON_VENV_PATH="${PYTHON_VENV_PATH:-$HOME/envs/pc_logger}"
export PC_SENSOR_PORT="${PC_SENSOR_PORT:-/dev/ttyACM0}"
export PC_SENSOR_PORTS="${PC_SENSOR_PORTS:-}"
export PC_SENSOR_PORT_NANO="${PC_SENSOR_PORT_NANO:-${NANO_SENSOR_PORT:-}}"
export PC_SENSOR_PORT_AGX="${PC_SENSOR_PORT_AGX:-${AGX_SENSOR_PORT:-}}"
# Keep INA260 current sign from QT Py. Set PC_CURRENT_SIGN=-1 if your wiring reports current negative.
export PC_CURRENT_SIGN="${PC_CURRENT_SIGN:-1}"

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
  if [[ -f "$ONNX_MODEL_PATH" ]]; then
    export ACCURACY_MODEL="$ONNX_MODEL_PATH"
  elif [[ -f "$MODEL_DIR/best.pt" ]]; then
    export ACCURACY_MODEL="$MODEL_DIR/best.pt"
  else
    export ACCURACY_MODEL="$MODEL_PATH"
  fi
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
}

detect_pi_env_mode() {
  local requested_mode="${1:-$PI_ENV_MODE}"
  case "$requested_mode" in
    auto)
      if have_python_venv "$PYTHON_VENV_PATH"; then
        echo "venv"
      elif command -v conda >/dev/null 2>&1 || [[ -f "$MINICONDA_ROOT/etc/profile.d/conda.sh" ]]; then
        echo "conda"
      else
        echo "user"
      fi
      ;;
    venv|conda|user)
      echo "$requested_mode"
      ;;
    *)
      echo "Unsupported PC_ENV_MODE/PI_ENV_MODE: $requested_mode (expected auto, venv, conda, or user)" >&2
      return 1
      ;;
  esac
}

activate_pi_env() {
  local mode
  mode="$(detect_pi_env_mode "${1:-$PI_ENV_MODE}")"
  if [[ "$mode" == "venv" ]]; then
    activate_python_venv "$PYTHON_VENV_PATH"
  elif [[ "$mode" == "conda" ]]; then
    activate_conda_env "${2:-$CONDA_ENV_NAME}"
  else
    return 0
  fi
}
