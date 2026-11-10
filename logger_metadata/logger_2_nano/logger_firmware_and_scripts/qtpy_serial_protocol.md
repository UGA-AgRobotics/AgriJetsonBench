# QT Py Serial Protocol

The PC logger expects parseable serial lines containing an INA260 block and, when available, a TMP117 or TMP119 block. The current logger package accepts both TMP117 and TMP119 labels for backwards compatibility.

Expected line pattern:

```text
TMP119: 25.1875 C / 77.34 F | INA260: 307.50 mA, 19.227 V, 5890.00 mW
```

Parser outputs:
- bus_voltage_v
- current_a
- power_w
- bus_voltage_mV
- current_mA
- power_mW
- external_temp_c
- tmp117_temp_c, retained as a legacy column name
- temperature_sensor_type

The PC package default keeps the current sign reported by the QT Py. Set PC_CURRENT_SIGN=-1 only if the wiring reports negative current and positive current is desired in CSV/API outputs.
