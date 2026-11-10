from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from smbus2 import SMBus, i2c_msg

INA260_ADDR_DEFAULT = 0x40
TMP117_ADDR_DEFAULT = 0x48

INA260_REG_CONFIG = 0x00
INA260_REG_CURRENT = 0x01
INA260_REG_VOLTAGE = 0x02
INA260_REG_POWER = 0x03

TMP117_REG_TEMP = 0x00
TMP117_REG_CONFIG = 0x01
TMP117_REG_DEVICE_ID = 0x0F


def s16(v: int) -> int:
    return v - 65536 if v & 0x8000 else v


def list_i2c_buses() -> List[int]:
    buses = []
    for raw in glob.glob('/dev/i2c-*'):
        try:
            buses.append(int(raw.split('-')[-1]))
        except ValueError:
            continue
    return sorted(set(buses))


def read_u16_be(bus: SMBus, addr: int, reg: int) -> int:
    write = i2c_msg.write(addr, [reg])
    read = i2c_msg.read(addr, 2)
    bus.i2c_rdwr(write, read)
    data = bytes(read)
    return (data[0] << 8) | data[1]


def write_u16_be(bus: SMBus, addr: int, reg: int, value: int) -> None:
    bus.i2c_rdwr(i2c_msg.write(addr, [reg, (value >> 8) & 0xFF, value & 0xFF]))


def identify_temp_sensor(did: int) -> Optional[str]:
    if (did & 0x0FFF) != 0x0117:
        return None
    if did == 0x0117:
        return 'TMP117'
    if did == 0x2117:
        return 'TMP119'
    return 'TMP117-compatible'


def probe_ina260(bus_id: int, address: int = INA260_ADDR_DEFAULT) -> bool:
    try:
        with SMBus(bus_id) as bus:
            _ = read_u16_be(bus, address, INA260_REG_CONFIG)
        return True
    except Exception:
        return False


def probe_temp_sensor(bus_id: int, address: int = TMP117_ADDR_DEFAULT) -> Dict[str, object]:
    out: Dict[str, object] = {
        'present': False,
        'device_id': None,
        'sensor_type': None,
    }
    try:
        with SMBus(bus_id) as bus:
            did = read_u16_be(bus, address, TMP117_REG_DEVICE_ID)
        sensor_type = identify_temp_sensor(did)
        if sensor_type is not None:
            out['present'] = True
            out['device_id'] = f'0x{did:04X}'
            out['sensor_type'] = sensor_type
    except Exception:
        pass
    return out


def autodetect_bus(
    preferred_bus: Optional[int] = None,
    ina260_address: int = INA260_ADDR_DEFAULT,
    temp_sensor_address: int = TMP117_ADDR_DEFAULT,
    require_temp_sensor: bool = False,
) -> Tuple[int, Dict[str, object]]:
    buses = list_i2c_buses()
    if preferred_bus is not None:
        buses = [preferred_bus] + [b for b in buses if b != preferred_bus]

    for bus_id in buses:
        ina_ok = probe_ina260(bus_id, ina260_address)
        temp_info = probe_temp_sensor(bus_id, temp_sensor_address)
        if ina_ok and ((temp_info.get('present') is True) or not require_temp_sensor):
            return bus_id, temp_info

    raise RuntimeError(
        'Could not find INA260 on any local I2C bus. '
        f'Scanned buses: {buses}. '
        'Run pi_logger/scan_i2c_all.py and verify wiring.'
    )


@dataclass
class INA260Reading:
    voltage_v: float
    current_a: float
    power_w: float

    @property
    def voltage_mV(self) -> float:
        return self.voltage_v * 1000.0

    @property
    def current_mA(self) -> float:
        return self.current_a * 1000.0

    @property
    def power_mW(self) -> float:
        return self.power_w * 1000.0


class INA260Sensor:
    def __init__(self, bus_id: int, address: int = INA260_ADDR_DEFAULT):
        self.bus_id = bus_id
        self.address = address
        self.bus = SMBus(bus_id)

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass

    def read(self) -> INA260Reading:
        raw_i = read_u16_be(self.bus, self.address, INA260_REG_CURRENT)
        raw_v = read_u16_be(self.bus, self.address, INA260_REG_VOLTAGE)
        raw_p = read_u16_be(self.bus, self.address, INA260_REG_POWER)
        current_a = s16(raw_i) * 0.00125
        voltage_v = raw_v * 0.00125
        power_w = raw_p * 0.01
        return INA260Reading(voltage_v=voltage_v, current_a=current_a, power_w=power_w)

    def read_config(self) -> int:
        return read_u16_be(self.bus, self.address, INA260_REG_CONFIG)


class TemperatureSensor:
    def __init__(self, bus_id: int, address: int = TMP117_ADDR_DEFAULT):
        self.bus_id = bus_id
        self.address = address
        self.bus = SMBus(bus_id)
        did = self.read_device_id()
        sensor_type = identify_temp_sensor(did)
        if sensor_type is None:
            raise RuntimeError(
                f'TMP117/TMP119-compatible device ID mismatch on bus {bus_id}, addr {hex(address)}: got 0x{did:04X}'
            )
        self.sensor_type = sensor_type

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass

    def read_device_id(self) -> int:
        return read_u16_be(self.bus, self.address, TMP117_REG_DEVICE_ID)

    def read_config(self) -> int:
        return read_u16_be(self.bus, self.address, TMP117_REG_CONFIG)

    def write_config(self, value: int) -> None:
        write_u16_be(self.bus, self.address, TMP117_REG_CONFIG, value & 0xFFFF)

    def configure_fast_continuous_runtime(self) -> None:
        # Continuous conversion, no averaging, shortest conversion cycle.
        # Works for TMP117 and TMP119-compatible register map devices.
        self.write_config(0x0000)

    def read_celsius(self) -> float:
        raw = read_u16_be(self.bus, self.address, TMP117_REG_TEMP)
        return s16(raw) * 0.0078125


class SensorHub:
    def __init__(
        self,
        i2c_bus: Optional[int] = None,
        ina260_address: int = INA260_ADDR_DEFAULT,
        temp_sensor_address: int = TMP117_ADDR_DEFAULT,
        require_temp_sensor: bool = False,
        fast_temp_sensor_on_init: bool = False,
    ):
        bus_id, temp_info = autodetect_bus(
            preferred_bus=i2c_bus,
            ina260_address=ina260_address,
            temp_sensor_address=temp_sensor_address,
            require_temp_sensor=require_temp_sensor,
        )
        self.bus_id = bus_id
        self.ina260_address = ina260_address
        self.temp_sensor_address = temp_sensor_address
        self.ina = INA260Sensor(bus_id, ina260_address)
        self.temp: Optional[TemperatureSensor] = (
            TemperatureSensor(bus_id, temp_sensor_address) if temp_info.get('present') else None
        )
        if self.temp is not None and fast_temp_sensor_on_init:
            self.temp.configure_fast_continuous_runtime()

    def close(self) -> None:
        self.ina.close()
        if self.temp is not None:
            self.temp.close()

    def health(self) -> Dict[str, object]:
        health: Dict[str, object] = {
            'bus_id': self.bus_id,
            'ina260_address': hex(self.ina260_address),
            'temp_sensor_address': hex(self.temp_sensor_address),
            'temp_sensor_present': self.temp is not None,
            'temp_sensor_type': self.temp.sensor_type if self.temp is not None else None,
        }
        try:
            health['ina260_config'] = hex(self.ina.read_config())
        except Exception as exc:
            health['ina260_config_error'] = str(exc)
        if self.temp is not None:
            try:
                did = self.temp.read_device_id()
                health['temp_sensor_config'] = hex(self.temp.read_config())
                health['temp_sensor_device_id'] = hex(did)
            except Exception as exc:
                health['temp_sensor_error'] = str(exc)
        return health

    def configure_temp_sensor_fast_runtime(self) -> bool:
        if self.temp is None:
            return False
        self.temp.configure_fast_continuous_runtime()
        return True

    def read_once(self) -> Dict[str, Optional[float]]:
        ina = self.ina.read()
        temp_c = self.temp.read_celsius() if self.temp is not None else None
        sensor_type = self.temp.sensor_type if self.temp is not None else None
        return {
            'bus_voltage_v': float(ina.voltage_v),
            'current_a': float(ina.current_a),
            'power_w': float(ina.power_w),
            'bus_voltage_mV': float(ina.voltage_mV),
            'current_mA': float(ina.current_mA),
            'power_mW': float(ina.power_mW),
            'external_temp_c': None if temp_c is None else float(temp_c),
            'tmp117_temp_c': None if temp_c is None else float(temp_c),  # legacy alias
            'temperature_sensor_type': sensor_type,
        }


def scan_all(ina260_address: int = INA260_ADDR_DEFAULT, temp_sensor_address: int = TMP117_ADDR_DEFAULT) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for bus_id in list_i2c_buses():
        temp_info = probe_temp_sensor(bus_id, temp_sensor_address)
        entry: Dict[str, object] = {'bus_id': bus_id}
        entry['ina260_present'] = probe_ina260(bus_id, ina260_address)
        entry['temp_sensor_present'] = temp_info.get('present', False)
        entry['temp_sensor_type'] = temp_info.get('sensor_type')
        entry['temp_sensor_device_id'] = temp_info.get('device_id')
        out.append(entry)
    return out
