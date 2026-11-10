#!/bin/bash
set -euo pipefail

VENV_PATH="${1:-/home/$USER/envs/yolo11}"
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
export PYTHONNOUSERSITE=1

PYTHON_VERSION_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
ENV_SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
MODULE_PYTHONPATH_PARTS=()
for root_var in EBROOTPYTORCH EBROOTTORCHVISION; do
  root_dir="${!root_var:-}"
  if [ -n "$root_dir" ]; then
    candidate="$root_dir/lib/python${PYTHON_VERSION_MM}/site-packages"
    if [ -d "$candidate" ]; then
      MODULE_PYTHONPATH_PARTS+=("$candidate")
    fi
  fi
done

unset PYTHONPATH EBPYTHONPREFIXES PYTHONHOME
PYTHONPATH_PARTS=()
if [ -n "$ENV_SITE_PACKAGES" ]; then
  PYTHONPATH_PARTS+=("$ENV_SITE_PACKAGES")
fi
for part in "${MODULE_PYTHONPATH_PARTS[@]}"; do
  PYTHONPATH_PARTS+=("$part")
done
if [ ${#PYTHONPATH_PARTS[@]} -gt 0 ]; then
  export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"
fi

echo "[INFO] Active Python : $(command -v python)"
echo "[INFO] CONDA_PREFIX  : ${CONDA_PREFIX:-unset}"
echo "[INFO] PYTHONPATH    : ${PYTHONPATH:-unset}"
