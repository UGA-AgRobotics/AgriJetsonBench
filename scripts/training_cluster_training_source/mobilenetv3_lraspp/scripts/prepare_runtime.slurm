#!/bin/bash
#SBATCH --job-name=mobilenetv3_prepare
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG="${CONFIG:-$PACKAGE_DIR/config/segmentation_mobilenetv3_train_lock.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/segment}"

cd "$PACKAGE_DIR"
source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
python "$PACKAGE_DIR/scripts/stage2_mobilenetv3_pipeline.py" --config "$CONFIG" --mode prepare
