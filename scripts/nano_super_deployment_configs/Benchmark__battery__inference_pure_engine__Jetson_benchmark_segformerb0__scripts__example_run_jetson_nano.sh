#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BOARD_NAME="${BOARD_NAME:-Jetson Orin Nano Super Developer Kit}"
export RUN_VARIANTS="${RUN_VARIANTS:-standard}"
export INT8_ENABLE="${INT8_ENABLE:-1}"
export RUN_PRECISIONS="${RUN_PRECISIONS:-fp16,fp32,int8}"
export EXPECT_EXTERNAL_LOGGER="${EXPECT_EXTERNAL_LOGGER:-1}"
export AUTO_BUILD_MISSING_ENGINES="${AUTO_BUILD_MISSING_ENGINES:-1}"
export USE_LOCAL_TENSORRT_STACK="${USE_LOCAL_TENSORRT_STACK:-0}"
export REQUIRE_WIRED_LOGGER="${REQUIRE_WIRED_LOGGER:-1}"

exec bash "$SCRIPT_DIR/run_current_mode_fp16_fp32.sh" "$@"
