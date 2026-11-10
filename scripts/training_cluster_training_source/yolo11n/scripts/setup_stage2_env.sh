#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${1:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster}"
VENV_PATH="${2:-/home/$USER/envs/SAM}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-0}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"

echo "[INFO] Package dir : $PACKAGE_DIR"
echo "[INFO] Venv path   : $VENV_PATH"
echo "[INFO] Miniforge   : $MINIFORGE_MODULE"
echo "[INFO] PyTorch mod : $PYTORCH_MODULE"
echo "[INFO] Torchvision : $TORCHVISION_MODULE"

if ! command -v module >/dev/null 2>&1 && ! command -v ml >/dev/null 2>&1; then
  echo "[ERROR] Environment Modules are not available in this shell."
  exit 1
fi

module purge >/dev/null 2>&1 || ml purge >/dev/null 2>&1 || true
module load "$MINIFORGE_MODULE" >/dev/null 2>&1 || ml "$MINIFORGE_MODULE"
module load "$PYTORCH_MODULE"  >/dev/null 2>&1 || ml "$PYTORCH_MODULE"
module load "$TORCHVISION_MODULE"  >/dev/null 2>&1 || ml "$TORCHVISION_MODULE"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda was not found after loading $MINIFORGE_MODULE."
  exit 1
fi

# Avoid stale CONDA_* values from the caller confusing prefix activation.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

if [ ! -d "$VENV_PATH/conda-meta" ]; then
  conda create -y -p "$VENV_PATH" python=3.11
fi

if [ -n "${EBROOTMINIFORGE3:-}" ] && [ -f "$EBROOTMINIFORGE3/etc/profile.d/conda.sh" ]; then
  source "$EBROOTMINIFORGE3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

conda activate "$VENV_PATH"
hash -r
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "[INFO] Active Python after activation: $(command -v python)"

ENV_SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [ -n "$ENV_SITE_PACKAGES" ]; then
  export PYTHONPATH="$ENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

conda install -y -p "$VENV_PATH" -c conda-forge packaging pip setuptools wheel
python -m pip install --ignore-installed \
  "numpy>=1.26,<2.0" \
  "sympy>=1.13,<1.15" \
  "pandas>=2.2,<3.0" \
  "typing_extensions>=4.8" \
  "tqdm>=4.66" \
  "protobuf>=4.25,<6" \
  "ml_dtypes>=0.5,<0.6"
python -m pip install -r "$PACKAGE_DIR/requirements.txt"

python - <<'PY'
import sys
print("[INFO] Active Python  :", sys.executable)
try:
    import torch
    print("[INFO] Torch version  :", torch.__version__)
    print("[INFO] CUDA available :", torch.cuda.is_available())
except Exception as e:
    print("[WARN] Torch import failed:", e)

for name in ["ultralytics", "pandas", "onnx", "optuna"]:
    try:
        mod = __import__(name)
        print(f"[INFO] {name:<14}:", getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(f"[WARN] {name:<14}: {e}")
PY

echo "[INFO] Environment setup complete."
echo "[INFO] Use later in jobs via scripts/activate_stage2_env.sh"
