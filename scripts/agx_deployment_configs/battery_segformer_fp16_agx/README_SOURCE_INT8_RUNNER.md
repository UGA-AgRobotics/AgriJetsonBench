# AGX Battery Inference

This folder contains inference-side runners for UGREEN power-bank endurance validation on the NVIDIA Jetson AGX Orin 64GB Developer Kit.

The runner starts and stops Pc_package recording automatically by sending its generated `run_id` to the logger `/start` and `/stop` endpoints. Keep the Pc_package server/dashboard running, but you do not need to type the run ID in the logger UI. Stop the inference runner with `Ctrl-C` when the power bank reaches 80 percent used and 20 percent remaining; partial outputs are finalized automatically.

After you press ENTER, the runner starts Pc_package recording automatically, checks `/health`, and refuses to start inference if the dashboard is still idle/waiting. It also pushes a live Jetson status heartbeat to `/api/jetson_status`, so the dashboard shows the parent `run_id`, benchmark stage/progress, NoiseGuard state, and AGX internal telemetry while inference is running. Use `--logger-control manual` for the old manual behavior or `--allow-inactive-logger` only for debugging without external CSV logging.

The default target is AGX 15W matched-budget mode, INT8 precision, input size 640, batch size 1. If the current `nvpmodel` output does not look like 15W, the runner stops before inference unless `--allow-power-mode-mismatch` is passed intentionally.

Before a real endurance run, verify the logger sensor is in the AGX current path. `read_once` should report a meaningful AGX bus voltage and non-zero load power; a near-zero voltage or zero power warning means the external CSV will not be usable for battery-energy analysis.

## One-Time Noninteractive Sudo Setup

Run this once before endurance testing. It installs a limited privileged-command policy file only for the AGX battery inference runner and NoiseGuard safe-profile commands.

```bash
# Privileged device-control steps must be reviewed and run manually on the target system.
# Privileged device-control steps must be reviewed and run manually on the target system.
```

If you do not want to install a privileged-command policy file, the runner prompts for sudo once at the start and keeps it alive until the run exits. You can also use the wrapper form:

```bash
python3 scripts/run_battery_inference_agx.py --config config/battery_inference_agx.yaml --config-name yolo11s_int8_640 --mode endurance --status-interval 60
```

or:

```bash
scripts/run_battery_inference_agx_with_sudo.sh --config config/battery_inference_agx.yaml --config-name yolo11s_int8_640 --mode endurance --status-interval 60
```

## Smoke Test YOLO11s INT8

```bash
python3 scripts/run_battery_inference_agx.py \
  --config config/battery_inference_agx.yaml \
  --config-name yolo11s_int8_640 \
  --mode smoke
```

## Smoke Test SegFormer-B0 INT8

```bash
python3 scripts/run_battery_inference_agx.py \
  --config config/battery_inference_agx.yaml \
  --config-name segformer_b0_int8_640 \
  --mode smoke
```

## Endurance YOLO11s INT8

```bash
python3 scripts/run_battery_inference_agx.py \
  --config config/battery_inference_agx.yaml \
  --config-name yolo11s_int8_640 \
  --mode endurance \
  --status-interval 60
```

## Endurance SegFormer-B0 INT8

```bash
python3 scripts/run_battery_inference_agx.py \
  --config config/battery_inference_agx.yaml \
  --config-name segformer_b0_int8_640 \
  --mode endurance \
  --status-interval 60
```

## Summary

```bash
python3 scripts/summarize_battery_inference_agx.py \
  --runs-dir ./battery_inference_runs/agx \
  --out-dir ./battery_inference_summary/agx
```

## Interrupted Run Recovery

If the Jetson reboots or loses power before you can press `Ctrl-C`, recover the partial run instead of treating it as a clean stop:

```bash
python3 scripts/finalize_interrupted_battery_run.py \
  --run-dir ./battery_inference_runs/agx/<run_id> \
  --stopped-reason power_loss_or_unplanned_reboot \
  --try-download-external
```

This writes `summary.json`, `recovery_report.json`, `inference_summary.csv`, `supplementary_battery_endurance_agx.csv`, and `supplementary_battery_endurance_agx.md`. The supplementary files label the run as interrupted, report the last saved event time, recovered runtime, saved image count, latency/FPS, internal telemetry, and whether the external logger archive was available.

## Notes

- Run exactly one configuration per battery-discharge experiment.
- The wrapper records `nvpmodel -q`, `nvpmodel -q --verbose`, device model, L4T, CUDA/TensorRT info, events, latency rows, and summaries.
- External power logs are merged later using `run_id`.
- Live dashboard status payloads are saved in each run directory as `dashboard_status_payloads.jsonl`; internal snapshots used for the dashboard are saved as `dashboard_internal_live.jsonl`.
- Per-prediction outputs are disabled by default; use `--save-predictions` only if you explicitly need them.
