# Runtime configuration notes

The AGX logger service uses the Pc_package `run_pc_logger.sh` launcher and the `anonymous@example.invalid` systemd unit. The live system override sets the AGX serial path through `PC_SENSOR_PORT_AGX` and clears the Nano path.

Relevant defaults:

- `PC_CURRENT_SIGN=1`
- `PC_SENSOR_BAUDRATE=115200`
- `PC_SENSOR_READ_TIMEOUT_S=0.05`
- `PC_SENSOR_FIRST_SAMPLE_TIMEOUT_S=8.0`
- `PC_SENSOR_STALE_TIMEOUT_S=5.0`
- HTTP dashboard/API port: `8000`
- Output root observed in service status: `${LOGGER_HOME}/runs/pc_external_logs`

The source project is not a git repository at staging time, so there is no git commit hash for this logger package.
