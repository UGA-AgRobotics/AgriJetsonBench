#!/bin/bash
set -euo pipefail

RUNTIME_ROOT="${1:?Usage: bundle_outputs.sh /work/.../yolo11n_baseline_v3 [output_tar.gz]}"
OUTPUT_TAR="${2:-$RUNTIME_ROOT/bundles/yolo11n_baseline_v3_outputs.tar.gz}"

mkdir -p "$(dirname "$OUTPUT_TAR")"
tar --ignore-failed-read -czhf "$OUTPUT_TAR" \
  -C "$RUNTIME_ROOT" \
  FINAL_SUMMARY.json \
  FINAL_SUMMARY.md \
  OUTPUT_FILE_MANIFEST.csv \
  reports \
  environment \
  source_snapshot \
  paper_assets \
  exports \
  artifacts \
  runtime_dataset \
  state

echo "[INFO] Bundle created at: $OUTPUT_TAR"
