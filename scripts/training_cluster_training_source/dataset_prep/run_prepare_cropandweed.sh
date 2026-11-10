#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the revised CropAndWeed segmentation prepare script with logging.
#
# Example:
#   ROOT=${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark #   REPO_ZIP=/home/$USER/cropandweed-dataset-main.zip #   ENV_PATH=/home/$USER/envs/jetson_cropandweed_prep #   bash run_prepare_cropandweed.sh --download-only
#
#   ROOT=${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark #   REPO_ZIP=/home/$USER/cropandweed-dataset-main.zip #   ENV_PATH=/home/$USER/envs/jetson_cropandweed_prep #   bash run_prepare_cropandweed.sh --prepare-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-}"
REPO_ZIP="${REPO_ZIP:-}"
ENV_PATH="${ENV_PATH:-}"

if [[ -z "${ROOT}" ]]; then
  echo "Please set ROOT, e.g. ROOT=/work/MYGROUP/jetson_edge_ai_benchmark"
  exit 1
fi

if [[ -n "${ENV_PATH}" ]]; then
  source "${ENV_PATH}/bin/activate"
fi

LOG_ROOT="$HOME/jetson_edge_ai_benchmark/logs/cropandweed"
mkdir -p "$LOG_ROOT/runs" "$LOG_ROOT/state" "$LOG_ROOT/slurm"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/runs/prepare_cropandweed_${STAMP}.log"
STATE_FILE="$LOG_ROOT/state/prepare_state.json"

CMD=(python "$SCRIPT_DIR/prepare_cropandweed_segmentation.py" --root "$ROOT")
if [[ -n "${REPO_ZIP}" ]]; then
  CMD+=(--repo-zip "$REPO_ZIP")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "[run_prepare_cropandweed] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[run_prepare_cropandweed] root=$ROOT"
  echo "[run_prepare_cropandweed] repo_zip=$REPO_ZIP"
  printf '[run_prepare_cropandweed] cmd='
  printf '%q ' "${CMD[@]}"
  echo
} | tee "$LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

cat > "$STATE_FILE" <<EOF
{
  "last_run_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "log_file": "$LOG_FILE",
  "root": "$ROOT",
  "repo_zip": "$REPO_ZIP"
}
EOF

echo "[run_prepare_cropandweed] done"
