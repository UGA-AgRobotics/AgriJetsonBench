# AgriJetsonBench Nano Super Idle Baseline Code

This code implements the PDF protocol in this folder. It runs idle-only external-power baselines for Nano Super 15W and 25W.

Main commands:

```bash
cd ${NANO_HOME}/project_progress/AgriJetsonBench_NanoSuper_Idle_Baseline_Protocol
# Privileged device-control steps must be reviewed and run manually on the target system.
# Privileged device-control steps must be reviewed and run manually on the target system.
bash code/run_preflight.sh
bash code/run_all_idle_baselines.sh --mode 15W
# after the reboot, run the same command again to collect 15W
bash code/run_all_idle_baselines.sh --mode 15W

bash code/run_all_idle_baselines.sh --mode 25W
# after the reboot, run the same command again to collect 25W
bash code/run_all_idle_baselines.sh --mode 25W
```

Quick smoke test:

```bash
cd ${NANO_HOME}/project_progress/AgriJetsonBench_NanoSuper_Idle_Baseline_Protocol
bash code/run_quick_smoke.sh
```

Outputs are written under:

- `idle_baselines/`
- `paper_tables/idle_baseline_power_summary.csv`
- `reports/`
- `logs/`
- `agrijetsonbench_nano_super_idle_baselines_*.zip`

The full protocol performs each mode as:

- run preflight
- if `--mode 15W` or `--mode 25W` is used through the wrapper, set that power mode and reboot before measurement
- activate NoiseGuard with logger route preserved
- set/confirm `nvpmodel`
- start Pc_package external logger
- start `tegrastats`
- 120 second settle window
- 300 second official idle window
- stop telemetry/logger
- restore NoiseGuard
- download/extract logger archive
- compute P0 metrics only for the official idle window

For paper-clean runs, run one mode at a time. The first command after a mode change only sets `nvpmodel`, writes `reboot_state/pending_power_mode_*.json`, and reboots. After the Jetson comes back, run the exact same command again; the runner confirms the new boot and then starts the official idle baseline. The summary CSV and final ZIP merge valid 15W and 25W runs side by side.

NoiseGuard is enabled by default in `config_idle_baseline.json`. It uses `preserve_url` from `logger_url`, so if the external logger route needs Wi-Fi, Wi-Fi is left enabled; otherwise Wi-Fi/Bluetooth/background services/display are quieted and restored at the end. The runner uses `sudo -n`, so manual privileged-command access must already cover `nvpmodel`, `jetson_clocks`, `tegrastats`, `kill/pkill`, `systemctl`, `nmcli`, and `rfkill`.

To run without NoiseGuard only for debugging:

```bash
python3 code/run_idle_baseline_protocol.py --mode both --no-noise-guard
```
