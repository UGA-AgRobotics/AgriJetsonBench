#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="Benchmark/battery/inference_pure_engine/Jetson_benchmark_segformerb0_fp16_25W"
cd "$ROOT_DIR"

python3 scripts/run_battery_inference_pure_engine_nano_super.py \
  --config config/battery_inference_pure_engine_segformer_b0_fp16_25w.yaml \
  --config-name segformer_b0_fp16_640_native25 \
  --mode endurance \
  --status-interval 60 \
  --set-nvpmodel \
  --no-wait-for-enter \
  --no-sudo-password-once
