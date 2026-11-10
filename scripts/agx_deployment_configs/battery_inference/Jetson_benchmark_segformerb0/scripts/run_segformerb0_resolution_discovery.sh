#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
activate_build_conda_env
activate_local_tensorrt_stack

PYBIN="${PYBIN:-python}"
BOARD_LABEL="${BOARD_LABEL:-agx_orin_50w}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SIZES="${SIZES:-256 320 384 448 512 576 640 704 768 896 1024 1152}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
SOURCE_ONNX="${SOURCE_ONNX:-$ONNX_MODEL_PATH}"
DATASET_INPUT="${DATASET_INPUT:-$DATASET_SOURCE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BENCHMARK_ROOT/resolution_ablation/segformerb0/$BOARD_LABEL/fp16_discovery}"
MANIFEST="${MANIFEST:-}"

unset PI_LOGGER_URL || true
unset STATUS_UPDATE_URL || true
unset EXPECT_EXTERNAL_LOGGER || true

mkdir -p "$OUTPUT_ROOT"

echo "[INFO] SegFormer-B0 FP16 resolution discovery"
echo "[INFO] BOARD_LABEL=$BOARD_LABEL"
echo "[INFO] SPLIT_NAME=$SPLIT_NAME"
echo "[INFO] SIZES=$SIZES"
echo "[INFO] SOURCE_ONNX=$SOURCE_ONNX"
echo "[INFO] DATASET_INPUT=$DATASET_INPUT"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"

CMD=(
  "$PYBIN" "$PKG_ROOT/scripts/run_segformerb0_fp16_resolution_discovery.py"
  --source-onnx "$SOURCE_ONNX"
  --dataset-root "$DATASET_INPUT"
  --split-name "$SPLIT_NAME"
  --board-label "$BOARD_LABEL"
  --output-root "$OUTPUT_ROOT"
  --workspace-mb "$WORKSPACE_MB"
)
if [[ -n "$MANIFEST" ]]; then
  CMD+=(--manifest "$MANIFEST")
fi
# shellcheck disable=SC2206
SIZE_ARGS=($SIZES)
CMD+=(--sizes "${SIZE_ARGS[@]}")
"${CMD[@]}"

echo "[INFO] Done:"
echo "  $OUTPUT_ROOT/size_probe_results.csv"
echo "  $OUTPUT_ROOT/selection_summary.json"
