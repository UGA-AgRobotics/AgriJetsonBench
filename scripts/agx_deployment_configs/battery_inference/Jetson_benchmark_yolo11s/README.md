# Proposal-oriented Jetson benchmark package with live dashboard + OLED + quiet mode

This version extends the earlier AGX Orin + Raspberry Pi 5 proposal package with three live-operation features:

- a **live browser dashboard / installable PWA-like view** served directly by the Raspberry Pi logger
- an **SSD1306 128x32 I2C OLED status display** on the Jetson DUT
- an automatic **quiet mode / noise guard** on the Jetson DUT during calibration and benchmark runs

Quiet mode is designed to reduce non-benchmark noise while preserving the OLED and the logger connection. In its default safe profile it will, on a best-effort basis:

- turn **Wi-Fi off** if the logger route does not use Wi-Fi
- **block Bluetooth**
- try to force the display **DPMS off**
- stop selected background **services/timers** such as apt/packagekit and a few common desktop helpers
- **restore** everything it changed when the run finishes or errors out

The helper scripts now call `scripts/run_with_sudo_keepalive.sh` so the package can still restore the original state after long runs.

Default local assumptions for the helper scripts in this workspace:

- package root: `~/Benchmark/Jetson_benchmark_yolo11s`
- Pi logger Python venv: `~/envs/jetson_pi_logger` (preferred when present)

To export only the Raspberry Pi 5 side into a portable transfer folder:

```bash
cd ~/Benchmark/Jetson_benchmark_yolo11s
bash scripts/prepare_pi5_package.sh ~/Benchmark/Pi5_package
```
- local model bundle: `~/Benchmark/yolo11s/inference_bundle`
- Miniconda root: `~/miniconda3`
- conda env: `SAM`
- outputs: `~/Benchmark/runs/...`

If your paths differ, override `MINICONDA_ROOT`, `CONDA_ENV_NAME`, `INFERENCE_BUNDLE_ROOT`, `MODEL_PATH`, `ACCURACY_MODEL`, `DATASET_INPUT`, or `OUT_ROOT` when calling the scripts.

The package still keeps the proposal-oriented methodology:

- external Raspberry Pi 5 logger for INA260 + TMP117/TMP119-compatible temperature sensor
- internal Jetson telemetry: INA3221 hwmon, thermal zones, tegrastats
- synced start across external logger, internal telemetry, and inference
- proposal defaults for warm-up and measured window
  - 5 minute warm-up
  - at least 500 images **and** at least 180 s per repeat
- calibration suite (idle / cpu / gpu / mixed)
- native-mode and matched-budget runs
- 30 minute sustained run helper
- aligned merged traces + agreement statistics + comparison tooling
- helper to prepare the user’s `inference_bundle/` tree

Current default benchmark behavior for `scripts/run_current_mode_int8_fp16_fp32.sh`:

- it does not change the Jetson power mode; it benchmarks the current detected mode only
- if required TensorRT engines are missing, it auto-builds the missing ones from `best.onnx`
- it runs all available precisions for that mode (`fp16`, `fp32`, and optionally `int8`)
- it produces two separate output variants by default:
  - `standard`: report-oriented summary outputs, without per-image artifacts
  - `per_image`: the same benchmark family plus per-image outputs such as `per_image_latency.csv` and `per_image_metrics.csv`

If you want to prebuild the engines only, use one command:

```bash
cd ~/Benchmark/Jetson_benchmark_yolo11s
bash scripts/build_engines_int8_fp16_fp32.sh
```

Typical command:

```bash
cd ~/Benchmark/Jetson_benchmark_yolo11s
INT8_ENABLE=1 NOISE_GUARD_ENABLE=1 OLED_BUS=7 OLED_ADDR=0x3C PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 bash scripts/run_current_mode_int8_fp16_fp32.sh
```

For the full step-by-step guide, use `README_FA.md`.
