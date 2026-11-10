# Pc_package

In folder baraye logger/dashboard rooye Ubuntu PC ast. Sensorha rooye PC mostaghim I2C nistan; `INA260` va `TMP119`/`TMP117` be QT Py RP2040 vaslan va QT Py az USB serial, masalan `/dev/ttyACM0`, data ro mide.

## In package chie?

- logger baraye INA260 + TMP119/TMP117 az serial-e QT Py
- dashboard web rooye port `8000`
- CSV va summary/tar.gz mesle packagehaye Pi5/JetsonNano

## 1) env ro amade kon

```bash
cd ~/Pc_package
bash scripts/setup_pc_env.sh
```

Default venv:

```text
~/envs/pc_logger
```

Age script user ro be group-e `dialout` ezafe kard, logout/login ya reboot lazem mishe.

## 2) serial ro check kon

```bash
cd ~/Pc_package
bash scripts/check_pc_serial.sh
source ~/envs/pc_logger/bin/activate
python pi_logger/scan_serial.py
```

Bayad line-hayi shabih in bebinid:

```text
TMP119: 25.18 C / 77.32 F | INA260: -290.00 mA, 19.230 V, 5620.00 mW
```

Default-e in package `PC_CURRENT_SIGN=1` ast, yani hamoon sign-i ke QT Py report mikone record mishe. Agar line-haye QT Py current ro manfi neshoon dadan vali mikhay CSV/API current mosbat bashe:

```bash
PC_CURRENT_SIGN=-1 bash scripts/run_pc_logger.sh
```

## 3) logger ro run kon

```bash
cd ~/Pc_package
PC_SENSOR_PORT=/dev/ttyACM0 bash scripts/run_pc_logger.sh
```

Agar Nano va AGX har kodoom QT Py + INA260 + temp sensor-e joda daran, behtarin halat in ast ke az path-haye stable-e `/dev/serial/by-id/...` estefade koni:

```bash
ls -l /dev/serial/by-id/
```

Bad logger ro ba map-e serial port bala biar:

```bash
cd ~/Pc_package
PC_SENSOR_PORTS='jetson-nano=/dev/serial/by-id/NANO_QTPY,jetson-agx=/dev/serial/by-id/AGX_QTPY' \
bash scripts/run_pc_logger.sh
```

Ya be jaye yek map:

```bash
PC_SENSOR_PORT_NANO=/dev/ttyACM0 PC_SENSOR_PORT_AGX=/dev/ttyACM1 bash scripts/run_pc_logger.sh
```

Parser-e logger har do temp label-e `TMP119` va `TMP117` ro ghabool mikone, pas AGX ke `TMP117` dare ok ast.

Field-e asli-ye API/CSV baraye dama `external_temp_c` va `tmp119_temp_c` ast. Kelid-e ghadimi-ye `tmp117_temp_c` felan baraye compatibility hanuz ersal mishavad.

Dashboard:

```text
http://127.0.0.1:8000/
```

Baraye Jetson-ha rooye kabel/switch:

```text
http://${LOGGER_DASHBOARD_HOST}:8000/
```

Health check:

```bash
bash scripts/check_logger_health.sh http://127.0.0.1:8000
```

## 3.1) Switch baraye Nano + AGX

TP-Link TL-SG105 unmanaged ast, pas config software nadarad. Logger PC, Jetson Nano, va Jetson AGX ro ba kabel Ethernet be switch bezan. IP-ha ro static va too yek subnet negah dar:

```text
Logger PC:   ${LOGGER_DASHBOARD_HOST}
Jetson Nano: ${JETSON_HOST_IP}
Jetson AGX:  ${LOGGER_PC_IP}
Netmask:     255.255.255.0
Gateway:     khali
```

Command-haye Jetson ham mitoonan hamoon URL ghabli ro estefade konan:

```bash
LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000
```

Dashboard az in version tab-haye `Jetson Nano` va `Jetson AGX` dare. Status, OC/power-watch event-ha, external CSV, summary, va tar.gz bar asas IP/hostname/board_name joda save va neshoon dade mishan. Vaghti baraye har Jetson QT Py + INA260 + temp sensor-e joda map shode bashe, Nano va AGX mitoonan hamzaman be logger status/measurement befrestan.

## 4) optional: systemd service

```bash
sudo cp service/pc_external_logger@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pc_external_logger@$USER
sudo systemctl status pc_external_logger@$USER
```
