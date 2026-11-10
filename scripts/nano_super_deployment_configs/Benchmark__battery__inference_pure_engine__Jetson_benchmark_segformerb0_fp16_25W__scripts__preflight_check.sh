#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="Benchmark/battery/inference_pure_engine/Jetson_benchmark_segformerb0_fp16_25W"
cd "$ROOT_DIR"

# Privileged device-control steps must be reviewed and run manually on the target system.

python3 scripts/run_battery_inference_pure_engine_nano_super.py \
  --config config/battery_inference_pure_engine_segformer_b0_fp16_25w.yaml \
  --config-name segformer_b0_fp16_640_native25 \
  --set-nvpmodel \
  --dry-run \
  --logger-control off \
  --disable-noise-guard \
  --no-wait-for-enter \
  --no-sudo-password-once
