#!/usr/bin/env bash
#SBATCH --job-name=prep_cropweed_seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Example usage:
#   sbatch --export=ROOT=${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark,REPO_ZIP=/home/$USER/cropandweed-dataset-main.zip,ENV_PATH=/home/$USER/envs/jetson_cropandweed_prep prepare_cropandweed_segmentation.slurm
#
# If ROOT is not provided, the script exits.

ROOT="${ROOT:-}"
REPO_ZIP="${REPO_ZIP:-}"
ENV_PATH="${ENV_PATH:-/home/$USER/envs/jetson_cropandweed_prep}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$ROOT" ]]; then
  echo "ROOT is required, e.g. /work/MYGROUP/jetson_edge_ai_benchmark"
  exit 1
fi

if [[ ! -d "$ENV_PATH" ]]; then
  echo "Environment not found: $ENV_PATH"
  echo "Create it first with setup_env.sh in an interactive session."
  exit 1
fi

source "$ENV_PATH/bin/activate"

CMD=(python "$SCRIPT_DIR/prepare_cropandweed_segmentation.py" --root "$ROOT")
if [[ -n "$REPO_ZIP" ]]; then
  CMD+=(--repo-zip "$REPO_ZIP")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${EXTRA_ARGS} )
  CMD+=("${EXTRA[@]}")
fi

printf '[slurm] cmd='
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
