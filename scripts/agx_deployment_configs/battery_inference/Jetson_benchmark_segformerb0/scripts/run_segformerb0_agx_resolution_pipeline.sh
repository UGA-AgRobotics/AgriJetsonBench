#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

BOARD_LABEL="${BOARD_LABEL:-agx_orin_50w}"
SPLIT_NAME="${SPLIT_NAME:-test}"
RUN_PRECISION_CONFIRM="${RUN_PRECISION_CONFIRM:-1}"
ALLOW_INCOMPLETE_SELECTION="${ALLOW_INCOMPLETE_SELECTION:-0}"
DISCOVERY_OUTPUT_ROOT="${DISCOVERY_OUTPUT_ROOT:-$BENCHMARK_ROOT/resolution_ablation/segformerb0/${BOARD_LABEL}/fp16_discovery}"
PRECISION_OUTPUT_ROOT="${PRECISION_OUTPUT_ROOT:-$BENCHMARK_ROOT/resolution_ablation/segformerb0/${BOARD_LABEL}/precision_confirmation}"
DISCOVERY_SUMMARY="$DISCOVERY_OUTPUT_ROOT/selection_summary.json"

unset PI_LOGGER_URL || true
unset STATUS_UPDATE_URL || true
unset EXPECT_EXTERNAL_LOGGER || true

echo "[PIPELINE] SegFormer-B0 AGX resolution pipeline"
echo "[PIPELINE] external logger disabled for discovery/confirmation"

OUTPUT_ROOT="$DISCOVERY_OUTPUT_ROOT" \
BOARD_LABEL="$BOARD_LABEL" \
SPLIT_NAME="$SPLIT_NAME" \
bash "$SCRIPT_DIR/run_segformerb0_resolution_discovery.sh"

if [[ "${RUN_PRECISION_CONFIRM,,}" =~ ^(0|false|no)$ ]]; then
  echo "[DONE] Precision confirmation skipped."
  exit 0
fi

if [[ ! -f "$DISCOVERY_SUMMARY" ]]; then
  echo "[ERROR] Missing discovery summary: $DISCOVERY_SUMMARY" >&2
  exit 1
fi

SELECTED_SIZES="$(
  python - "$DISCOVERY_SUMMARY" "$ALLOW_INCOMPLETE_SELECTION" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allow_incomplete = str(sys.argv[2]).lower() in {"1", "true", "yes"}
status = data.get("status", "ok")
if status != "ok" and not allow_incomplete:
    raise SystemExit(f"selection_summary status is {status!r}; not running precision confirmation")
sizes = []
for key in ("S_min", "reference_size", "S_max_board"):
    value = data.get(key)
    if value is not None:
        sizes.append(int(value))
seen = []
for size in sizes:
    if size not in seen:
        seen.append(size)
print(" ".join(str(x) for x in seen))
PY
)"

if [[ -z "$SELECTED_SIZES" ]]; then
  echo "[ERROR] Could not derive selected sizes from $DISCOVERY_SUMMARY" >&2
  exit 1
fi

SELECTED_SIZES="$SELECTED_SIZES" \
OUTPUT_ROOT="$PRECISION_OUTPUT_ROOT" \
BOARD_LABEL="$BOARD_LABEL" \
SPLIT_NAME="$SPLIT_NAME" \
bash "$SCRIPT_DIR/run_segformerb0_precision_confirm_selected.sh"
