#!/bin/bash
#SBATCH --job-name=y11n_cwd12
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster}"
CONFIG="${CONFIG:-$PACKAGE_DIR/config/detection_yolo11n_train_lock.yaml}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/SAM}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[INFO] Hostname     : $(hostname)"
echo "[INFO] Job ID       : ${SLURM_JOB_ID:-none}"
echo "[INFO] Package      : $PACKAGE_DIR"
echo "[INFO] Config       : $CONFIG"
echo "[INFO] Mode         : $PIPELINE_MODE"
echo "[INFO] Venv         : $VENV_PATH"
echo "[INFO] Torch module : $PYTORCH_MODULE"
echo "[INFO] TV module    : $TORCHVISION_MODULE"
echo "[INFO] OMP threads  : $OMP_NUM_THREADS"

cd "$PACKAGE_DIR"
source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"

python "$PACKAGE_DIR/scripts/stage2_yolo11n_pipeline.py" \
  --config "$CONFIG" \
  --mode "$PIPELINE_MODE"
