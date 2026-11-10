#!/bin/bash
#SBATCH --job-name=y11s_prepare
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster}"
CONFIG="${CONFIG:-$PACKAGE_DIR/config/detection_yolo11s_train_lock.yaml}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/yolo11}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"

cd "$PACKAGE_DIR"
source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"

python "$PACKAGE_DIR/scripts/stage2_yolo11s_pipeline.py" \
  --config "$CONFIG" \
  --mode prepare
