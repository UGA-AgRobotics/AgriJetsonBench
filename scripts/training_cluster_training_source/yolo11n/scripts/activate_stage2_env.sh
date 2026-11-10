#!/bin/bash
set -euo pipefail

VENV_PATH="${1:-/home/$USER/envs/SAM}"
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

# Slurm jobs submitted with --export=ALL can inherit stale CONDA_* values from
# the submit shell. Reset them so the requested prefix is placed first on PATH.
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

ENV_SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [ -n "$ENV_SITE_PACKAGES" ]; then
  export PYTHONPATH="$ENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[INFO] Active Python : $(command -v python)"
echo "[INFO] CONDA_PREFIX  : ${CONDA_PREFIX:-unset}"
