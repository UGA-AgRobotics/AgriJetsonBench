from __future__ import annotations

import glob
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import serial
except Exception as exc:  # pragma: no cover - handled at runtime with a clearer error
    serial = None  # type: ignore[assignment]
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


INA260_ADDR_DEFAULT = 0x40
TMP117_ADDR_DEFAULT = 0x48

DEFAULT_BAUDRATE = 115200
DEFAULT_READ_TIMEOUT_S = 0.05
DEFAULT_FIRST_SAMPLE_TIMEOUT_S = 8.0
DEFAULT_STALE_TIMEOUT_S = 5.0

TMP_RE = re.compile(r"\b(TMP(?:117|119)):\s*([-+]?\d+(?:\.\d+)?)\s*C\b")
INA_RE = re.compile(
    r"\bINA260:\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*mA,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*V,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*mW\b"
)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value, 0)


def _serial_error_hint() -> str:
    if _SERIAL_IMPORT_ERROR is not None:
        return f"pyserial is not importable: {_SERIAL_IMPORT_ERROR}. Run scripts/setup_pc_env.sh first."
    return "Serial sensor stream is not available."


def list_serial_ports() -> List[str]:
    candidates: List[str] = []
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        for raw in glob.glob(pattern):
            path = str(Path(raw).resolve()) if pattern.endswith("by-id/*") else raw
            if path not in candidates:
                candidates.append(path)
    return sorted(candidates)


def autodetect_serial_port(preferred_port: Optional[str] = None) -> str:
    env_port = os.environ.get("PC_SENSOR_PORT") or os.environ.get("SENSOR_SERIAL_PORT")
    candidates: List[str] = []
    for port in (preferred_port, env_port):
        if port and port.lower() != "auto":
            candidates.append(port)
    candidates.extend([p for p in list_serial_ports() if p not in candidates])

    if not candidates:
        raise RuntimeError(
            "Could not find a QT Py serial device. Expected /dev/ttyACM* or /dev/ttyUSB*. "
            "Plug in the QT Py RP2040 and verify it appears with: ls -l /dev/ttyACM* /dev/ttyUSB*"
        )

    for port in candidates:
        if os.path.exists(port):
            return port

    raise RuntimeError(f"Configured serial port was not found. Tried: {candidates}")


def parse_qtpy_line(line: str) -> Optional[Dict[str, object]]:
    ina = INA_RE.search(line)
    if not ina:
        return None

    temp = TMP_RE.search(line)
    current_mA = float(ina.group(1))
    voltage_v = float(ina.group(2))
    power_mW = float(ina.group(3))
    return {
        "raw_line": line,
        "raw_current_mA": current_mA,
        "bus_voltage_v": voltage_v,
        "power_mW": power_mW,
        "external_temp_c": float(temp.group(2)) if temp else None,
        "temperature_sensor_type": temp.group(1) if temp else None,
    }


@dataclass
class SerialSample:
    bus_voltage_v: float
    current_a: float
    power_w: float
    external_temp_c: Optional[float]
    temperature_sensor_type: Optional[str]
    raw_current_mA: float
    raw_line: str
    received_monotonic_s: float

    @property
    def voltage_mV(self) -> float:
        return self.bus_voltage_v * 1000.0

    @property
    def current_mA(self) -> float:
        return self.current_a * 1000.0

    @property
    def power_mW(self) -> float:
        return self.power_w * 1000.0


class QtPySerialSensor:
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = DEFAULT_BAUDRATE,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        first_sample_timeout_s: float = DEFAULT_FIRST_SAMPLE_TIMEOUT_S,
        stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
        current_sign: float = 1.0,
    ):
        if serial is None:
            raise RuntimeError(_serial_error_hint())
        self.port = autodetect_serial_port(port)
        self.baudrate = baudrate
        self.read_timeout_s = read_timeout_s
        self.first_sample_timeout_s = first_sample_timeout_s
        self.stale_timeout_s = stale_timeout_s
        self.current_sign = current_sign
        self._serial = None
        self._lock = threading.Lock()
        self._last_sample: Optional[SerialSample] = None
        self._last_line: Optional[str] = None

    def close(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

    def _ensure_open(self):
        if self._serial is None or not self._serial.is_open:
            try:
                self._serial = serial.Serial(
                    self.port,
                    self.baudrate,
                    timeout=self.read_timeout_s,
                    write_timeout=1.0,
                )
            except PermissionError as exc:
                raise RuntimeError(
                    f"Permission denied opening {self.port}. Add the user to dialout and re-login: "
                    f"sudo usermod -aG dialout $USER"
                ) from exc
        return self._serial

    def _parse_sample(self, line: str) -> Optional[SerialSample]:
        parsed = parse_qtpy_line(line)
        if parsed is None:
            return None

        raw_current_mA = float(parsed["raw_current_mA"])
        current_mA = raw_current_mA * self.current_sign
        power_mW = float(parsed["power_mW"])
        return SerialSample(
            bus_voltage_v=float(parsed["bus_voltage_v"]),
            current_a=current_mA / 1000.0,
            power_w=power_mW / 1000.0,
            external_temp_c=parsed["external_temp_c"],  # type: ignore[arg-type]
            temperature_sensor_type=parsed["temperature_sensor_type"],  # type: ignore[arg-type]
            raw_current_mA=raw_current_mA,
            raw_line=str(parsed["raw_line"]),
            received_monotonic_s=time.monotonic(),
        )

    def read(self) -> SerialSample:
        with self._lock:
            ser = self._ensure_open()
            timeout_s = self.first_sample_timeout_s if self._last_sample is None else self.read_timeout_s
            deadline = time.monotonic() + max(timeout_s, 0.0)
            latest: Optional[SerialSample] = None

            while True:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    self._last_line = line
                    sample = self._parse_sample(line)
                    if sample is not None:
                        latest = sample
                        self._last_sample = sample
                        while getattr(ser, "in_waiting", 0):
                            raw2 = ser.readline()
                            if not raw2:
                                break
                            line2 = raw2.decode("utf-8", errors="replace").strip()
                            self._last_line = line2
                            sample2 = self._parse_sample(line2)
                            if sample2 is not None:
                                latest = sample2
                                self._last_sample = sample2
                        return latest

                if self._last_sample is not None:
                    age_s = time.monotonic() - self._last_sample.received_monotonic_s
                    if age_s <= self.stale_timeout_s:
                        return self._last_sample

                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)

            raise RuntimeError(
                f"No parseable INA260/TMP117/TMP119 sample received from {self.port}. "
                "Stop any other reader like cat /dev/ttyACM0 and verify QT Py code.py is printing sensor lines."
            )

    def health(self) -> Dict[str, object]:
        sample = self._last_sample
        return {
            "backend": "qtpy_serial",
            "serial_port": self.port,
            "baudrate": self.baudrate,
            "current_sign": self.current_sign,
            "last_raw_line": self._last_line,
            "last_sample_age_s": None if sample is None else time.monotonic() - sample.received_monotonic_s,
            "raw_current_mA": None if sample is None else sample.raw_current_mA,
        }


class SensorHub:
    def __init__(
        self,
        i2c_bus: Optional[int] = None,
        ina260_address: int = INA260_ADDR_DEFAULT,
        temp_sensor_address: int = TMP117_ADDR_DEFAULT,
        require_temp_sensor: bool = False,
        fast_temp_sensor_on_init: bool = False,
        serial_port: Optional[str] = None,
    ):
        del i2c_bus, require_temp_sensor, fast_temp_sensor_on_init
        self.ina260_address = ina260_address
        self.temp_sensor_address = temp_sensor_address
        self.current_sign = _env_float("PC_CURRENT_SIGN", 1.0)
        self.sensor = QtPySerialSensor(
            port=serial_port or os.environ.get("PC_SENSOR_PORT") or os.environ.get("SENSOR_SERIAL_PORT"),
            baudrate=_env_int("PC_SENSOR_BAUDRATE", DEFAULT_BAUDRATE),
            read_timeout_s=_env_float("PC_SENSOR_READ_TIMEOUT_S", DEFAULT_READ_TIMEOUT_S),
            first_sample_timeout_s=_env_float("PC_SENSOR_FIRST_SAMPLE_TIMEOUT_S", DEFAULT_FIRST_SAMPLE_TIMEOUT_S),
            stale_timeout_s=_env_float("PC_SENSOR_STALE_TIMEOUT_S", DEFAULT_STALE_TIMEOUT_S),
            current_sign=self.current_sign,
        )

    def close(self) -> None:
        self.sensor.close()

    def health(self) -> Dict[str, object]:
        latest = None
        try:
            latest = self.sensor.read()
        except Exception as exc:
            read_error = str(exc)
        else:
            read_error = None

        health: Dict[str, object] = {
            "backend": "qtpy_serial",
            "ina260_address": hex(self.ina260_address),
            "temp_sensor_address": hex(self.temp_sensor_address),
            "temp_sensor_present": latest is not None and latest.external_temp_c is not None,
            "temp_sensor_type": latest.temperature_sensor_type if latest is not None else None,
        }
        health.update(self.sensor.health())
        if read_error is not None:
            health["read_error"] = read_error
        return health

    def configure_temp_sensor_fast_runtime(self) -> bool:
        return False

    def read_once(self) -> Dict[str, Optional[float]]:
        sample = self.sensor.read()
        temp_c = None if sample.external_temp_c is None else float(sample.external_temp_c)
        return {
            "bus_voltage_v": float(sample.bus_voltage_v),
            "current_a": float(sample.current_a),
            "power_w": float(sample.power_w),
            "bus_voltage_mV": float(sample.voltage_mV),
            "current_mA": float(sample.current_mA),
            "power_mW": float(sample.power_mW),
            "external_temp_c": temp_c,
            "tmp119_temp_c": temp_c,
            # Keep the legacy key during migration so older consumers do not break.
            "tmp117_temp_c": temp_c,
            "temperature_sensor_type": sample.temperature_sensor_type,
        }


def scan_all(
    ina260_address: int = INA260_ADDR_DEFAULT,
    temp_sensor_address: int = TMP117_ADDR_DEFAULT,
) -> List[Dict[str, object]]:
    del ina260_address, temp_sensor_address
    out: List[Dict[str, object]] = []
    for port in list_serial_ports():
        out.append(
            {
                "backend": "qtpy_serial",
                "serial_port": port,
                "exists": os.path.exists(port),
                "readable": os.access(port, os.R_OK | os.W_OK),
            }
        )
    return out
