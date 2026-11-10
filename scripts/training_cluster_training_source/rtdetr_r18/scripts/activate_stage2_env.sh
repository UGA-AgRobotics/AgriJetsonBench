#!/bin/bash
set -euo pipefail

VENV_PATH="${1:-/home/$USER/envs/SAM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-0}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"

die() {
  echo "[ERROR] $*" >&2
  return 1 2>/dev/null || exit 1
}

if ! command -v module >/dev/null 2>&1 && ! command -v ml >/dev/null 2>&1; then
  die "Environment Modules are not available in this shell."
fi

module purge >/dev/null 2>&1 || ml purge >/dev/null 2>&1 || true
module load "$MINIFORGE_MODULE" >/dev/null 2>&1 || ml "$MINIFORGE_MODULE"
module load "$PYTORCH_MODULE"  >/dev/null 2>&1 || ml "$PYTORCH_MODULE"
module load "$TORCHVISION_MODULE"  >/dev/null 2>&1 || ml "$TORCHVISION_MODULE"

if ! command -v conda >/dev/null 2>&1; then
  die "conda was not found after loading $MINIFORGE_MODULE."
fi

if [ ! -d "$VENV_PATH/conda-meta" ]; then
  die "Conda environment not found at $VENV_PATH. Run scripts/setup_stage2_env.sh first."
fi

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

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
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES="${TORCHAO_FORCE_SKIP_LOADING_SO_FILES:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

ENV_SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [ -n "$ENV_SITE_PACKAGES" ]; then
  export PYTHONPATH="$ENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

COMPAT_SITE_DIR="$PACKAGE_DIR/scripts/python_compat"
if [ -d "$COMPAT_SITE_DIR" ]; then
  export PYTHONPATH="$COMPAT_SITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

SCRIPTS_SITE_DIR="$PACKAGE_DIR/scripts"
if [ -d "$SCRIPTS_SITE_DIR" ]; then
  export PYTHONPATH="$SCRIPTS_SITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[INFO] Active Python : $(command -v python)"
echo "[INFO] CONDA_PREFIX  : ${CONDA_PREFIX:-unset}"
