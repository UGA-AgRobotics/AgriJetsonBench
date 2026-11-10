# AgriJetsonBench AGX Idle Baseline

This folder contains the runnable implementation for the PDF protocol.

Main command:

```bash
cd ${AGX_HOME}/project_progress/AgriJetsonBench_AGX_Idle_Baseline_Protocol/code
./run_idle_baseline_agx_with_sudo.sh
```

If sudo asks for a password, enter it once at the start. The wrapper keeps
sudo alive until both idle runs finish.

Default behavior:

- runs `15W` then `50W`
- uses nvpmodel IDs discovered from `/etc/nvpmodel.conf`
- enables noise guard before each logger/tegrastats run starts
- preserves the logger URL route while noise guard disables common noise sources
- records 120 seconds settle and 300 seconds official idle per mode
- starts and stops the Pc package logger at `http://${LOGGER_DASHBOARD_HOST}:8000`
- records `tegrastats_idle.log`
- writes run folders under `../idle_baselines`
- writes `../paper_tables/idle_baseline_power_summary.csv`
- writes a final zip under the protocol folder

Useful short smoke command without noise guard:

```bash
./run_idle_baseline_agx_with_sudo.sh --modes 50W --settle-seconds 2 --idle-seconds 5 --no-noise-guard
```

Full explicit 15W command with noise guard:

```bash
./run_idle_baseline_agx_with_sudo.sh --modes 15W --mode-id 15W=1 --settle-seconds 120 --idle-seconds 300 --external-sample-hz 10 --noise-guard --noise-guard-profile safe --noise-guard-preserve-url http://${LOGGER_DASHBOARD_HOST}:8000
```

Full explicit 50W command with noise guard:

```bash
./run_idle_baseline_agx_with_sudo.sh --modes 50W --mode-id 50W=3 --settle-seconds 120 --idle-seconds 300 --external-sample-hz 10 --noise-guard --noise-guard-profile safe --noise-guard-preserve-url http://${LOGGER_DASHBOARD_HOST}:8000
```

Rebuild merged tables, manifest, and zip after both mode runs:

```bash
./run_idle_baseline_agx_with_sudo.sh --merge-only
```

On this AGX, changing nvpmodel modes may require a reboot. If that happens,
run the mode command again after boot.
