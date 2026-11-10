#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <job_id> <runtime_root> <monitor_log> [poll_seconds]" >&2
  exit 2
fi

JOB_ID="$1"
RUNTIME_ROOT="$2"
MONITOR_LOG="$3"
POLL_SECONDS="${4:-120}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_OUT="$PACKAGE_DIR/segformer_b0_optuna_${JOB_ID}.out"
SLURM_ERR="$PACKAGE_DIR/segformer_b0_optuna_${JOB_ID}.err"

mkdir -p "$(dirname "$MONITOR_LOG")"

{
  echo "[INFO] watch_submitted_job started at $(date -Is)"
  echo "[INFO] job_id=$JOB_ID"
  echo "[INFO] runtime_root=$RUNTIME_ROOT"
  echo "[INFO] poll_seconds=$POLL_SECONDS"
} >>"$MONITOR_LOG"

while true; do
  now="$(date -Is)"
  squeue_state="$(squeue -h -j "$JOB_ID" -o "%T %M %N" || true)"
  if [[ -z "$squeue_state" ]]; then
    {
      echo "[$now] job no longer in squeue"
      sacct -j "$JOB_ID" --format=JobID,JobName%40,State,ExitCode,Elapsed -P || true
    } >>"$MONITOR_LOG"
    exit 0
  fi

  {
    echo "[$now] squeue: $squeue_state"
    status_file="$RUNTIME_ROOT/optuna/trials/trial_0000/current_trial_status.json"
    if [[ -f "$status_file" ]]; then
      python - "$status_file" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(
        "trial0 "
        f"status={payload.get('status')} "
        f"epoch={payload.get('last_reported_epoch')} "
        f"value={payload.get('last_reported_value')}"
    )
except Exception as exc:  # pragma: no cover - operational helper
    print(f"trial0 status parse error: {exc}")
PY
    fi
  } >>"$MONITOR_LOG"

  if rg -n --no-messages '"status": "failed"' "$RUNTIME_ROOT/optuna" >/dev/null; then
    {
      echo "[$now] detected failed status JSON; canceling job $JOB_ID"
      scancel "$JOB_ID" || true
    } >>"$MONITOR_LOG"
    exit 1
  fi

  if rg -n --no-messages 'Traceback \(most recent call last\):|ChildFailedError|CUDA out of memory|NCCL ERROR' \
    "$SLURM_OUT" "$SLURM_ERR" "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/optuna" >/dev/null; then
    {
      echo "[$now] detected fatal log pattern; canceling job $JOB_ID"
      scancel "$JOB_ID" || true
    } >>"$MONITOR_LOG"
    exit 1
  fi

  sleep "$POLL_SECONDS"
done
