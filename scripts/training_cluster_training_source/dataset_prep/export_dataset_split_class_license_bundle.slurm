#!/bin/bash
#SBATCH --job-name=dataset_notes_bundle
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail

OUT_DIR="${TRAINING_CLUSTER_SHARED_WORK}/dataset split files + class names + license notes"
STAGE_DIR="${TRAINING_CLUSTER_SHARED_WORK}/.dataset_split_class_license_notes_stage_${SLURM_JOB_ID}"
STATUS_DIR="${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/slurm_transfer_status"
SUMMARY_LOG="${STATUS_DIR}/dataset_split_class_license_notes_${SLURM_JOB_ID}.summary"
SCRIPT="${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/export_dataset_split_class_license_bundle.py"

mkdir -p "${STATUS_DIR}"
mkdir -p "${OUT_DIR}"
umask 007

{
  echo "job_id=${SLURM_JOB_ID}"
  date -Is
  echo "out_dir=${OUT_DIR}"
  echo "stage_dir=${STAGE_DIR}"
  echo "script=${SCRIPT}"
} > "${SUMMARY_LOG}"

if [[ -e "${STAGE_DIR}" ]]; then
  rm -rf "${STAGE_DIR}"
fi
mkdir -p "${STAGE_DIR}"

echo "Building staged dataset notes bundle..." | tee -a "${SUMMARY_LOG}"
python3 "${SCRIPT}" --out-root "${STAGE_DIR}"

echo "Syncing staged bundle into final output..." | tee -a "${SUMMARY_LOG}"
rsync -a --delete "${STAGE_DIR}/" "${OUT_DIR}/"
chgrp -R gcrlab "${OUT_DIR}"
chmod -R g+rwX "${OUT_DIR}"

echo "Cleaning stage dir..." | tee -a "${SUMMARY_LOG}"
rm -rf "${STAGE_DIR}"

echo "Completed successfully." | tee -a "${SUMMARY_LOG}"
date -Is | tee -a "${SUMMARY_LOG}"
