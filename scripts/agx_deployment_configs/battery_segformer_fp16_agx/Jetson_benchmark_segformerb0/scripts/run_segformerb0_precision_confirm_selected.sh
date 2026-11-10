#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_build_conda_env
activate_local_tensorrt_stack

PYBIN="${PYBIN:-python}"
BOARD_LABEL="${BOARD_LABEL:-agx_orin_50w}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SELECTED_SIZES="${SELECTED_SIZES:-}"
SOURCE_ONNX="${SOURCE_ONNX:-$ONNX_MODEL_PATH}"
DATASET_INPUT="${DATASET_INPUT:-$DATASET_SOURCE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BENCHMARK_ROOT/resolution_ablation/segformerb0/$BOARD_LABEL/precision_confirmation}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
MAX_CALIB_IMAGES="${MAX_CALIB_IMAGES:-256}"
MANIFEST="${MANIFEST:-}"

if [[ -z "$SELECTED_SIZES" ]]; then
  echo "SELECTED_SIZES is required, e.g. SELECTED_SIZES='448 640 1024' bash scripts/run_segformerb0_precision_confirm_selected.sh" >&2
  exit 2
fi

unset PI_LOGGER_URL || true
unset STATUS_UPDATE_URL || true
unset EXPECT_EXTERNAL_LOGGER || true

mkdir -p "$OUTPUT_ROOT"

CMD=(
  "$PYBIN" "$PKG_ROOT/scripts/run_segformerb0_precision_confirm_selected.py"
  --source-onnx "$SOURCE_ONNX"
  --dataset-root "$DATASET_INPUT"
  --split-name "$SPLIT_NAME"
  --board-label "$BOARD_LABEL"
  --output-root "$OUTPUT_ROOT"
  --workspace-mb "$WORKSPACE_MB"
  --max-calib-images "$MAX_CALIB_IMAGES"
)
if [[ -n "$MANIFEST" ]]; then
  CMD+=(--manifest "$MANIFEST")
fi
# shellcheck disable=SC2206
SIZE_ARGS=($SELECTED_SIZES)
CMD+=(--sizes "${SIZE_ARGS[@]}")
"${CMD[@]}"
