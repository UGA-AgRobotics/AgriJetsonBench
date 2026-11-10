#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$HOME/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster}"
OPTUNA_SLURM="$PACKAGE_DIR/scripts/rtdetr_r18_optuna.slurm"

PARTITION="gpu_p"
GRES="gpu:H100:1"
TIME_LIMIT="72:00:00"
CPUS="32"
MEMORY="96G"
VENV_PATH="/home/$USER/envs/SAM"
PYTORCH_MODULE="${PYTORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"
TORCHVISION_MODULE="${TORCHVISION_MODULE:-torchvision/0.16.2-foss-2023a-CUDA-12.1.1}"
CONFIG="$PACKAGE_DIR/config/detection_rtdetr_r18_train_lock.yaml"

AUTO_FULL="yes"
FULL_PARTITION=""
FULL_GRES=""
FULL_TIME=""
FULL_CPUS=""
FULL_MEMORY=""

extract_gpu_count() {
  local spec="$1"
  local tail="${spec##*:}"
  if [[ "$tail" =~ ^[0-9]+$ ]]; then
    echo "$tail"
    return
  fi
  if [[ "$spec" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo "1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --pytorch-module) PYTORCH_MODULE="$2"; shift 2 ;;
    --torchvision-module) TORCHVISION_MODULE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --no-auto-full) AUTO_FULL="no"; shift 1 ;;
    --full-partition) FULL_PARTITION="$2"; shift 2 ;;
    --full-gres) FULL_GRES="$2"; shift 2 ;;
    --full-time) FULL_TIME="$2"; shift 2 ;;
    --full-cpus) FULL_CPUS="$2"; shift 2 ;;
    --full-mem) FULL_MEMORY="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -n "$FULL_PARTITION" ]] || FULL_PARTITION="$PARTITION"
[[ -n "$FULL_GRES" ]] || FULL_GRES="$GRES"
[[ -n "$FULL_TIME" ]] || FULL_TIME="$TIME_LIMIT"
[[ -n "$FULL_CPUS" ]] || FULL_CPUS="$CPUS"
[[ -n "$FULL_MEMORY" ]] || FULL_MEMORY="$MEMORY"

REQUESTED_GPUS="$(extract_gpu_count "$GRES")"

if [[ "$AUTO_FULL" == "yes" ]] && [[ "$FULL_PARTITION" != "$PARTITION" || "$FULL_GRES" != "$GRES" || "$FULL_TIME" != "$TIME_LIMIT" || "$FULL_CPUS" != "$CPUS" || "$FULL_MEMORY" != "$MEMORY" ]]; then
  echo "[WARN] --full-* overrides are ignored in single-job mode."
  echo "[WARN] Optuna and full training now run sequentially inside one SLURM allocation and one job ID."
fi

if [[ "$AUTO_FULL" == "yes" ]]; then
  echo "[INFO] Submitting one RT-DETR-R18 job for Optuna -> full training partition=$PARTITION gres=$GRES gpus=$REQUESTED_GPUS time=$TIME_LIMIT cpus=$CPUS mem=$MEMORY"
else
  echo "[INFO] Submitting one RT-DETR-R18 Optuna-only job partition=$PARTITION gres=$GRES gpus=$REQUESTED_GPUS time=$TIME_LIMIT cpus=$CPUS mem=$MEMORY"
fi

SUBMIT_OUTPUT=$(sbatch \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --export=ALL,PACKAGE_DIR="$PACKAGE_DIR",CONFIG="$CONFIG",VENV_PATH="$VENV_PATH",PYTORCH_MODULE="$PYTORCH_MODULE",TORCHVISION_MODULE="$TORCHVISION_MODULE",CHAIN_FULL="$AUTO_FULL",FULL_PIPELINE_MODE="full",RTDETR_REQUESTED_GPUS="$REQUESTED_GPUS" \
  "$OPTUNA_SLURM")

echo "$SUBMIT_OUTPUT"
JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
if [[ -z "$JOB_ID" ]]; then
  echo "[ERROR] Could not parse job ID from sbatch output."
  exit 1
fi

if [[ "$AUTO_FULL" == "yes" ]]; then
  echo "[INFO] Single job ID submitted: $JOB_ID"
  echo "[INFO] The same job will run Optuna first, then start full training automatically in the same allocation."
else
  echo "[INFO] Optuna-only job ID: $JOB_ID"
fi
