from __future__ import annotations

"""
SSD1306 128x32 I2C OLED helper for Jetson benchmark progress.

Typical 4-pin OLED wiring (same style as the attached tester file):
GND -> Jetson header pin 6
VCC -> Jetson header pin 1 (3.3V)
SCL -> Jetson header pin 5
SDA -> Jetson header pin 3
"""

import glob
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from smbus2 import SMBus


class SSD1306_128x32:
    def __init__(self, bus_num: int, address: int = 0x3C, width: int = 128, height: int = 32):
        self.bus_num = bus_num
        self.address = address
        self.width = width
        self.height = height
        self.pages = height // 8
        self.bus = SMBus(bus_num)
        self.font = ImageFont.load_default()
        self._init_display()
        self.clear()

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass

    def _cmd(self, value: int) -> None:
        self.bus.write_byte_data(self.address, 0x00, value & 0xFF)

    def _cmds(self, values: List[int]) -> None:
        for value in values:
            self._cmd(value)

    def _data(self, payload: List[int]) -> None:
        chunk_size = 16
        for i in range(0, len(payload), chunk_size):
            self.bus.write_i2c_block_data(self.address, 0x40, payload[i : i + chunk_size])

    def _init_display(self) -> None:
        init_sequence = [
            0xAE,
            0xD5, 0x80,
            0xA8, 0x1F,
            0xD3, 0x00,
            0x40,
            0x8D, 0x14,
            0x20, 0x00,
            0xA1,
            0xC8,
            0xDA, 0x02,
            0x81, 0x8F,
            0xD9, 0xF1,
            0xDB, 0x40,
            0xA4,
            0xA6,
            0xAF,
        ]
        self._cmds(init_sequence)

    def display_off(self) -> None:
        self._cmd(0xAE)

    def clear(self) -> None:
        blank = [0x00] * (self.width * self.pages)
        self._cmds([0x21, 0, self.width - 1, 0x22, 0, self.pages - 1])
        self._data(blank)

    def _image_to_buffer(self, image: Image.Image) -> List[int]:
        if image.mode != '1':
            image = image.convert('1')
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height))
        pixels = image.load()
        buf = [0x00] * (self.width * self.pages)
        for page in range(self.pages):
            for x in range(self.width):
                byte = 0
                for bit in range(8):
                    y = page * 8 + bit
                    if pixels[x, y]:
                        byte |= (1 << bit)
                buf[page * self.width + x] = byte
        return buf

    def show_lines(self, lines: List[str]) -> None:
        image = Image.new('1', (self.width, self.height), 0)
        draw = ImageDraw.Draw(image)
        normalized: List[str] = []
        for raw in lines[:3]:
            raw = '' if raw is None else str(raw)
            wrapped = textwrap.wrap(raw, width=20) or ['']
            normalized.extend(wrapped[:1])
        normalized = (normalized + ['', '', ''])[:3]
        for y, line in zip([0, 10, 20], normalized):
            draw.text((0, y), line, font=self.font, fill=255)
        buf = self._image_to_buffer(image)
        self._cmds([0x21, 0, self.width - 1, 0x22, 0, self.pages - 1])
        self._data(buf)


class BenchmarkOLED:
    def __init__(self, bus: Optional[int] = None, addr: Optional[int] = None, enabled: bool = False):
        self.enabled = enabled
        self.bus = bus
        self.addr = addr
        self.oled: Optional[SSD1306_128x32] = None
        self.detect_info = 'disabled'
        if enabled:
            try:
                self.oled, self.detect_info = autodetect_oled(bus, addr)
            except Exception as exc:  # pragma: no cover
                self.oled = None
                self.detect_info = f'failed: {exc}'

    @property
    def active(self) -> bool:
        return self.oled is not None

    def close(self) -> None:
        if self.oled is not None:
            self.oled.close()
            self.oled = None

    def show(self, line1: str = '', line2: str = '', line3: str = '') -> None:
        if self.oled is None:
            return
        try:
            self.oled.show_lines([line1, line2, line3])
        except Exception:
            pass

    def _variant_token(self, run_variant: Optional[str]) -> str:
        value = (run_variant or '').strip().lower().replace('-', '_')
        if value == 'per_image':
            return 'PERIMG'
        if value == 'standard':
            return 'STD'
        return (run_variant or '').strip().upper()[:8]

    def _precision_token(self, precision_label: Optional[str]) -> str:
        value = (precision_label or '').strip().upper()
        return value[:8]

    def _header(self, precision_label: Optional[str] = None, run_variant: Optional[str] = None, fallback: str = 'JETSON BENCH') -> str:
        parts = []
        prec = self._precision_token(precision_label)
        var = self._variant_token(run_variant)
        if prec:
            parts.append(prec)
        if var:
            parts.append(var)
        if not parts:
            return fallback[:20]
        return ' '.join(parts)[:20]

    def boot(self, detail: str = 'Booting', precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), detail, self.detect_info[:20])

    def waiting(self, seconds_left: float, repeat_name: str = '', precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), f'WAIT {repeat_name}'[:20], f'{max(0, int(seconds_left))}s')

    def warmup(self, repeat_name: str, done: int, min_images: int, elapsed_s: float, eta_s: Optional[float], precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), f'{repeat_name} {done}/{min_images}'[:20], f'WARM eta {int(max(0, eta_s or 0))}s'[:20])

    def running(self, repeat_name: str, processed: int, target: int, eta_s: Optional[float], latency_ms: Optional[float], precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        line2 = f'{repeat_name} {processed}/{target}'
        line3 = f'eta {int(max(0, eta_s or 0))}s'
        if latency_ms is not None:
            line3 = f'{line3} {latency_ms:.1f}ms'
        self.show(self._header(precision_label, run_variant), line2[:20], line3[:20])

    def stage(self, title: str, line2: str = '', line3: str = '', precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), title[:20], (line3 or line2)[:20])

    def finished(self, line2: str = 'DONE', line3: str = '', precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), line2[:20], line3[:20])

    def error(self, short: str, detail: str = '', precision_label: Optional[str] = None, run_variant: Optional[str] = None) -> None:
        self.show(self._header(precision_label, run_variant), short[:20], detail[:20])


def list_i2c_buses() -> List[int]:
    buses: List[int] = []
    for path in sorted(glob.glob('/dev/i2c-*')):
        try:
            buses.append(int(path.rsplit('-', 1)[1]))
        except Exception:
            continue
    return sorted(set(buses))


def create_oled(bus: int, addr: int) -> SSD1306_128x32:
    return SSD1306_128x32(bus_num=bus, address=addr)


def autodetect_oled(preferred_bus: Optional[int], preferred_addr: Optional[int]) -> Tuple[Optional[SSD1306_128x32], str]:
    candidate_buses = [preferred_bus] if preferred_bus is not None else list_i2c_buses()
    candidate_addrs = [preferred_addr] if preferred_addr is not None else [0x3C, 0x3D]

    errors: List[str] = []
    for bus in candidate_buses:
        for addr in candidate_addrs:
            try:
                oled = create_oled(bus, addr)
                oled.show_lines(['OLED FOUND', f'bus={bus}', f'addr=0x{addr:02X}'])
                return oled, f'bus={bus} addr=0x{addr:02X}'
            except Exception as exc:
                errors.append(f'bus={bus} addr=0x{addr:02X}: {exc}')
                continue
    return None, '; '.join(errors[-4:]) if errors else 'no /dev/i2c-* devices found'
