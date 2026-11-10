#!/bin/bash
#SBATCH --job-name=rtd18_cwd12
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster}"
CONFIG="${CONFIG:-$PACKAGE_DIR/config/detection_rtdetr_r18_train_lock.yaml}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
VENV_PATH="${VENV_PATH:-/home/$USER/envs/SAM}"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"
RTDETR_REQUESTED_GPUS="${RTDETR_REQUESTED_GPUS:-1}"

if [[ "$RTDETR_REQUESTED_GPUS" =~ ^[0-9]+$ ]] && (( RTDETR_REQUESTED_GPUS > 1 )); then
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / RTDETR_REQUESTED_GPUS ))
  if (( OMP_NUM_THREADS < 1 )); then
    export OMP_NUM_THREADS=1
  fi
else
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi

echo "[INFO] Hostname     : $(hostname)"
echo "[INFO] Job ID       : ${SLURM_JOB_ID:-none}"
echo "[INFO] Package      : $PACKAGE_DIR"
echo "[INFO] Config       : $CONFIG"
echo "[INFO] Mode         : $PIPELINE_MODE"
echo "[INFO] Venv         : $VENV_PATH"
echo "[INFO] Torch module : $PYTORCH_MODULE"
echo "[INFO] TV module    : $TORCHVISION_MODULE"
echo "[INFO] Req GPUs     : $RTDETR_REQUESTED_GPUS"
echo "[INFO] OMP threads  : $OMP_NUM_THREADS"

cd "$PACKAGE_DIR"
source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"
python "$PACKAGE_DIR/scripts/ensure_rtdetr_repo.py" --package-dir "$PACKAGE_DIR" >/tmp/rtd18_repo_path.txt
REPO_DIR="$(cat /tmp/rtd18_repo_path.txt)"
echo "[INFO] Repo path    : $REPO_DIR"
python "$PACKAGE_DIR/scripts/ensure_model_weights.py" --package-dir "$PACKAGE_DIR" >/tmp/rtd18_weights_path.txt
WEIGHTS_PATH="$(cat /tmp/rtd18_weights_path.txt)"
echo "[INFO] Weights path : $WEIGHTS_PATH"

python "$PACKAGE_DIR/scripts/stage2_rtdetr_r18_pipeline.py" \
  --config "$CONFIG" \
  --mode "$PIPELINE_MODE"
