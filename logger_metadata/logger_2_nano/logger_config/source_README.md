# Pc_package

Ubuntu PC logger/dashboard package. The INA260 and TMP119/TMP117 are not on a local PC I2C bus; they are connected to the QT Py RP2040, and the PC reads the QT Py over USB serial, normally `/dev/ttyACM0`.

Quick start:

```bash
cd ~/Pc_package
bash scripts/setup_pc_env.sh
PC_SENSOR_PORT=/dev/ttyACM0 bash scripts/run_pc_logger.sh
```

For separate Nano and AGX QT Py sensors, prefer stable `/dev/serial/by-id/...` paths:

```bash
ls -l /dev/serial/by-id/
PC_SENSOR_PORTS='jetson-nano=/dev/serial/by-id/NANO_QTPY,jetson-agx=/dev/serial/by-id/AGX_QTPY' \
bash scripts/run_pc_logger.sh
```

`TMP119` and `TMP117` temperature labels are both accepted.

Open:

```text
http://127.0.0.1:8000/
```

On the direct Ethernet logger network, use:

```text
http://${LOGGER_DASHBOARD_HOST}:8000/
```

Recommended static addresses with an unmanaged switch:

```text
Logger PC:   ${LOGGER_DASHBOARD_HOST}
Jetson Nano: ${JETSON_HOST_IP}
Jetson AGX:  ${LOGGER_PC_IP}
Netmask:     255.255.255.0
```

The dashboard has separate Jetson Nano and Jetson AGX tabs. Jetson status, power-watch events, external CSVs, summaries, and archives are separated by IP/hostname/board name when each Jetson has its own mapped QT Py sensor.

Useful checks:

```bash
bash scripts/check_pc_serial.sh
source ~/envs/pc_logger/bin/activate
python pi_logger/scan_serial.py
bash scripts/check_logger_health.sh http://127.0.0.1:8000
```

The package defaults to `PC_CURRENT_SIGN=1`, so it records the same current sign that the QT Py reports. Set `PC_CURRENT_SIGN=-1` if the QT Py reports negative current and you want the CSV/API current to be positive.
