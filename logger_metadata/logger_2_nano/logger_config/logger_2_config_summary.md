# Logger 2 Configuration Summary

LOGGER_ID: logger_nano
LOGGER_ROLE: external power and sensor logging; associated primarily with Nano Super or second logger channel if applicable

Main runtime command:

```bash
cd ~/Pc_package
bash scripts/run_pc_logger.sh
```

Current user service:

```bash
systemctl --user restart pc-logger-dashboard.service
systemctl --user status pc-logger-dashboard.service --no-pager
```

Important environment variables:
- PC_SENSOR_PORT: single QT Py serial port.
- PC_SENSOR_PORTS: comma-separated device-to-port map, for example jetson-nano=/dev/serial/by-id/...,jetson-agx=/dev/serial/by-id/...
- PC_CURRENT_SIGN: default 1; use -1 only if current sign needs inversion.
- PC_LOGGER_OUTPUT_ROOT: defaults to ~/runs/pc_external_logs through scripts/common_env.sh.
- PORT: dashboard/API port, default 8000.

Sampling interval:
- Logger API default: sample_hz=10.0, equivalent to about 100 ms between samples.

Run naming:
- The logger server writes one run directory per run_id.
- Each run directory normally contains external_measurements.csv, run_meta.json, logger_summary.json, and an optional tar.gz archive.

Time synchronization and alignment:
- CSV rows include scheduled, read-start, read-end, midpoint wall-clock, monotonic, and UTC timestamps.
- mid_wallclock_ns is the preferred external logger timestamp for alignment.
- LAN/Ethernet was used for control/log transfer only, not for energy integration.
