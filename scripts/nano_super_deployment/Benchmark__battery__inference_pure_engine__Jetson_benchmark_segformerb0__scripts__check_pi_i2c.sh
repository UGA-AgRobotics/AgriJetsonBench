#!/usr/bin/env bash
set -euo pipefail
ls /dev/i2c* || true
i2cdetect -l || true
for BUS in /dev/i2c-*; do
  if [[ -e "$BUS" ]]; then
    NUM="${BUS##*-}"
    echo "==== SCAN BUS $NUM ===="
    sudo i2cdetect -y "$NUM" || true
  fi
done
