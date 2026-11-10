#!/usr/bin/env python3
"""
OLED pin   ->  Jetson AGX Orin J30 pin
GND        ->  6
VCC        ->  1   (3.3V)
SCL        ->  5
SDA        ->  3

jetson_oled_dl_tester.py
python3 jetson_oled_dl_tester.py --bus 7 --addr 0x3C --scenario menu


Connects an SSD1306 128x32 I2C OLED to a Jetson board and shows:
- BOOT
- RUNNING (spinner)
- FINISHED
- ERROR codes for multiple simulated failure modes

It can auto-scan /dev/i2c-* for the OLED at 0x3C or 0x3D,
and provides either:
- an interactive scenario menu, or
- a --scenario CLI switch.

Designed for:
- NVIDIA Jetson Orin Nano Developer Kit
- NVIDIA Jetson AGX Orin Developer Kit

OLED tested target:
- 0.91" SSD1306 128x32 I2C module (e.g. MakerFocus 4-pin OLED)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    print("[FATAL] Pillow is required. Install with: sudo apt install python3-pil", file=sys.stderr)
    raise

try:
    from smbus2 import SMBus
except Exception:  # pragma: no cover
    SMBus = None


# -----------------------------
# Small SSD1306 driver (I2C)
# -----------------------------

class SSD1306_128x32:
    """Minimal SSD1306 128x32 I2C driver using smbus2."""

    def __init__(self, bus_num: int, address: int = 0x3C, width: int = 128, height: int = 32):
        if SMBus is None:
            raise RuntimeError("smbus2 is not installed. Install with: pip3 install smbus2")
        self.bus_num = bus_num
        self.address = address
        self.width = width
        self.height = height
        self.pages = height // 8
        self.bus = SMBus(bus_num)
        self._font = ImageFont.load_default()
        self._init_display()
        self.clear()

    def close(self) -> None:
        try:
            self.display_off()
        except Exception:
            pass
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
        # SMBus block write is limited, so send in short chunks.
        chunk_size = 16
        for i in range(0, len(payload), chunk_size):
            self.bus.write_i2c_block_data(self.address, 0x40, payload[i : i + chunk_size])

    def _init_display(self) -> None:
        # Common init sequence for 128x32 SSD1306 I2C modules.
        init_sequence = [
            0xAE,       # DISPLAYOFF
            0xD5, 0x80, # SETDISPLAYCLOCKDIV
            0xA8, 0x1F, # SETMULTIPLEX for 32px height
            0xD3, 0x00, # SETDISPLAYOFFSET
            0x40,       # SETSTARTLINE
            0x8D, 0x14, # CHARGEPUMP ON
            0x20, 0x00, # MEMORYMODE = horizontal addressing
            0xA1,       # SEGREMAP
            0xC8,       # COMSCANDEC
            0xDA, 0x02, # SETCOMPINS for 128x32
            0x81, 0x8F, # SETCONTRAST
            0xD9, 0xF1, # SETPRECHARGE
            0xDB, 0x40, # SETVCOMDETECT
            0xA4,       # DISPLAYALLON_RESUME
            0xA6,       # NORMALDISPLAY
            0xAF,       # DISPLAYON
        ]
        self._cmds(init_sequence)

    def display_off(self) -> None:
        self._cmd(0xAE)

    def display_on(self) -> None:
        self._cmd(0xAF)

    def clear(self) -> None:
        blank = [0x00] * (self.width * self.pages)
        self._cmds([0x21, 0, self.width - 1, 0x22, 0, self.pages - 1])
        self._data(blank)

    def _image_to_buffer(self, image: Image.Image) -> List[int]:
        if image.mode != "1":
            image = image.convert("1")
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

    def show_image(self, image: Image.Image) -> None:
        buf = self._image_to_buffer(image)
        self._cmds([0x21, 0, self.width - 1, 0x22, 0, self.pages - 1])
        self._data(buf)

    def show_lines(self, lines: List[str]) -> None:
        image = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(image)

        # Keep it readable on 128x32 with the default bitmap font.
        normalized = []
        for raw in lines[:3]:
            raw = "" if raw is None else str(raw)
            wrapped = textwrap.wrap(raw, width=20) or [""]
            normalized.extend(wrapped[:1])
        normalized = (normalized + ["", "", ""])[:3]

        y_positions = [0, 10, 20]
        for y, line in zip(y_positions, normalized):
            draw.text((0, y), line, font=self._font, fill=255)

        self.show_image(image)


# -----------------------------
# Display abstraction
# -----------------------------

class StatusDisplay:
    def __init__(self, oled: Optional[SSD1306_128x32]):
        self.oled = oled
        self._lock = threading.Lock()
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_stop = threading.Event()
        self._spin_title = "RUNNING"
        self._spin_line2 = ""
        self._spin_line3 = ""

    def close(self) -> None:
        self.stop_spinner()
        if self.oled is not None:
            self.oled.close()

    def _render(self, lines: List[str]) -> None:
        with self._lock:
            console = " | ".join([str(x) for x in lines if x is not None])
            print(f"[OLED] {console}")
            if self.oled is not None:
                self.oled.show_lines(lines)

    def show(self, line1: str = "", line2: str = "", line3: str = "") -> None:
        self.stop_spinner()
        self._render([line1, line2, line3])

    def boot(self) -> None:
        self.show("JETSON OLED", "Booting...", time.strftime("%H:%M:%S"))

    def start_spinner(self, title: str = "RUNNING", line2: str = "", line3: str = "") -> None:
        self.stop_spinner()
        self._spin_title = title
        self._spin_line2 = line2
        self._spin_line3 = line3
        self._spinner_stop.clear()
        self._spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def update_spinner(self, line2: Optional[str] = None, line3: Optional[str] = None) -> None:
        with self._lock:
            if line2 is not None:
                self._spin_line2 = line2
            if line3 is not None:
                self._spin_line3 = line3

    def stop_spinner(self) -> None:
        if self._spinner_thread is not None:
            self._spinner_stop.set()
            self._spinner_thread.join(timeout=1.0)
            self._spinner_thread = None
            self._spinner_stop.clear()

    def _spinner_loop(self) -> None:
        frames = ["|", "/", "-", "\\"]
        idx = 0
        while not self._spinner_stop.is_set():
            with self._lock:
                title = f"{self._spin_title} {frames[idx % len(frames)]}"
                lines = [title, self._spin_line2, self._spin_line3]
            if self.oled is not None:
                try:
                    self.oled.show_lines(lines)
                except Exception as exc:
                    print(f"[WARN] OLED update failed while spinning: {exc}", file=sys.stderr)
            print(f"[OLED] {' | '.join(lines)}")
            idx += 1
            time.sleep(0.20)

    def finished(self, line2: str = "DONE", line3: str = "") -> None:
        self.show("FINISHED", line2, line3)

    def error(self, code: str, short: str, detail: str = "") -> None:
        self.show(f"ERROR {code}", short, detail)


# -----------------------------
# Exceptions and scenarios
# -----------------------------

class DemoError(RuntimeError):
    def __init__(self, code: str, short: str, detail: str):
        super().__init__(f"{code}: {short}: {detail}")
        self.code = code
        self.short = short
        self.detail = detail


@dataclass
class Scenario:
    key: str
    title: str
    description: str
    func: Callable[[StatusDisplay, argparse.Namespace], Dict[str, str]]


def require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception:
        raise DemoError("DEP", "Torch missing", "Install NVIDIA PyTorch for your JetPack")


class TinyCNNFactory:
    @staticmethod
    def build(nn):
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16 * 16 * 16, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 4),
                )

            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return TinyCNN()


def scenario_ok(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    torch, nn = require_torch()

    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    status.start_spinner("RUNNING", "Loading model", f"device={device}")
    time.sleep(0.4)

    model = TinyCNNFactory.build(nn).to(device).eval()

    if device == "cuda":
        # Warm-up once
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 64, 64, device=device))
        torch.cuda.synchronize()

    start = time.time()
    last_cls = -1

    for step in range(1, args.steps + 1):
        status.update_spinner(f"step {step}/{args.steps}", f"device={device}")
        with torch.no_grad():
            x = torch.randn(args.batch_size, 3, 64, 64, device=device)
            y = model(x)
            last_cls = int(torch.argmax(y[0]).item())
        if device == "cuda":
            torch.cuda.synchronize()
        time.sleep(args.step_sleep)

    elapsed = time.time() - start
    status.stop_spinner()
    status.finished(f"class={last_cls}", f"{elapsed:.1f}s {device}")
    return {"class": str(last_cls), "device": device, "elapsed": f"{elapsed:.2f}"}


def scenario_missing_torch(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    raise DemoError("DEP", "Torch missing", "Test dependency notification")


def scenario_weights_missing(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    fake_path = os.path.abspath("./models/demo_weights.pt")
    raise DemoError("FILE", "Weights missing", os.path.basename(fake_path))


def scenario_bad_checkpoint(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    raise DemoError("CKPT", "Bad checkpoint", "Corrupt or unreadable file")


def scenario_invalid_input(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    torch, nn = require_torch()
    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    status.start_spinner("RUNNING", "Bad input test", f"device={device}")
    model = TinyCNNFactory.build(nn).to(device).eval()
    bad = torch.randn(1, 1, 64, 64, device=device)  # wrong channel count
    try:
        with torch.no_grad():
            _ = model(bad)
    except Exception as exc:
        status.stop_spinner()
        raise DemoError("SHAPE", "Invalid input", "Expected 3 channels") from exc
    finally:
        status.stop_spinner()
    return {"status": "unexpected-pass"}


def scenario_cuda_oom(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    # Safe simulation; do not actually try to exhaust memory.
    raise DemoError("OOM", "CUDA OOM", "Simulated out-of-memory")


def scenario_camera_missing(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    raise DemoError("CAM", "Camera missing", "/dev/video0 not found")


def scenario_inference_fail(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    raise DemoError("INFER", "Inference failed", "Postprocess/runtime error")


def scenario_permission(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    raise DemoError("PERM", "Permission denied", "Try sudo or add i2c access")


def scenario_interrupt(status: StatusDisplay, args: argparse.Namespace) -> Dict[str, str]:
    status.start_spinner("RUNNING", "Interrupt test", "raising Ctrl+C")
    time.sleep(1.0)
    raise KeyboardInterrupt()


SCENARIOS: List[Scenario] = [
    Scenario("ok", "Normal run", "Run a tiny real PyTorch CNN and show RUNNING -> FINISHED", scenario_ok),
    Scenario("missing_torch", "Dependency missing", "Simulate missing PyTorch / dependency", scenario_missing_torch),
    Scenario("weights_missing", "Weights missing", "Simulate missing model/weights file", scenario_weights_missing),
    Scenario("bad_checkpoint", "Bad checkpoint", "Simulate corrupted checkpoint", scenario_bad_checkpoint),
    Scenario("invalid_input", "Invalid input", "Send wrong tensor shape to the model", scenario_invalid_input),
    Scenario("cuda_oom", "CUDA OOM", "Simulate GPU out-of-memory", scenario_cuda_oom),
    Scenario("camera_missing", "Camera missing", "Simulate camera not found", scenario_camera_missing),
    Scenario("inference_fail", "Inference fail", "Simulate runtime/inference failure", scenario_inference_fail),
    Scenario("permission", "Permission error", "Simulate I2C/GPIO permission failure", scenario_permission),
    Scenario("interrupt", "User interrupt", "Simulate Ctrl+C / stop", scenario_interrupt),
]
SCENARIO_MAP: Dict[str, Scenario] = {s.key: s for s in SCENARIOS}


# -----------------------------
# Bus detection and utilities
# -----------------------------


def list_i2c_buses() -> List[int]:
    buses: List[int] = []
    for path in sorted(glob.glob("/dev/i2c-*")):
        try:
            buses.append(int(path.rsplit("-", 1)[1]))
        except Exception:
            continue
    return sorted(set(buses))


def create_oled(bus: int, addr: int) -> SSD1306_128x32:
    oled = SSD1306_128x32(bus_num=bus, address=addr)
    oled.show_lines(["OLED FOUND", f"bus={bus}", f"addr=0x{addr:02X}"])
    return oled


def autodetect_oled(preferred_bus: Optional[int], preferred_addr: Optional[int]) -> Tuple[Optional[SSD1306_128x32], str]:
    if SMBus is None:
        return None, "smbus2 not installed"

    candidate_buses = [preferred_bus] if preferred_bus is not None else list_i2c_buses()
    candidate_addrs = [preferred_addr] if preferred_addr is not None else [0x3C, 0x3D]

    errors: List[str] = []
    for bus in candidate_buses:
        for addr in candidate_addrs:
            try:
                oled = create_oled(bus, addr)
                return oled, f"bus={bus} addr=0x{addr:02X}"
            except Exception as exc:
                errors.append(f"bus={bus} addr=0x{addr:02X}: {exc}")
                continue
    return None, "; ".join(errors[-4:]) if errors else "no /dev/i2c-* devices found"


def classify_exception(exc: BaseException) -> Tuple[str, str, str]:
    if isinstance(exc, DemoError):
        return exc.code, exc.short, exc.detail
    if isinstance(exc, KeyboardInterrupt):
        return "STOP", "Interrupted", "User stopped test"
    if isinstance(exc, FileNotFoundError):
        return "FILE", "File missing", os.path.basename(str(exc))[:20]
    if isinstance(exc, ImportError):
        return "DEP", "Dependency", "Module not installed"

    text = str(exc)
    lower = text.lower()
    if "out of memory" in lower:
        return "OOM", "Out of memory", "Allocation failed"
    if "permission" in lower:
        return "PERM", "Permission", "Access denied"
    if "camera" in lower or "/dev/video" in lower:
        return "CAM", "Camera missing", "Check camera path"
    if "shape" in lower or "channel" in lower or "input" in lower:
        return "SHAPE", "Invalid input", "Bad tensor/input"
    return "ERR", exc.__class__.__name__[:12], text[:20]


# -----------------------------
# UI and execution
# -----------------------------


def print_header() -> None:
    print("=" * 72)
    print("Jetson OLED Deep-Learning Tester")
    print("SSD1306 128x32 I2C OLED + scenario-based notification test")
    print("=" * 72)


def print_scenarios() -> None:
    print("Available scenarios:")
    for i, sc in enumerate(SCENARIOS, start=1):
        print(f"  {i:>2}. {sc.key:<15} - {sc.description}")


def run_scenario(name: str, status: StatusDisplay, args: argparse.Namespace) -> int:
    if name not in SCENARIO_MAP:
        print(f"[ERROR] Unknown scenario: {name}", file=sys.stderr)
        return 2

    sc = SCENARIO_MAP[name]
    print(f"\n[RUN] {sc.key} - {sc.title}")
    print(f"      {sc.description}")
    status.show("SELECTED", sc.key, "starting...")
    time.sleep(0.6)

    try:
        result = sc.func(status, args)
        print(f"[OK] Result: {result}")
        return 0
    except BaseException as exc:
        status.stop_spinner()
        code, short, detail = classify_exception(exc)
        status.error(code, short, detail)
        print(f"[ERROR] {code} | {short} | {detail}")
        print(f"        Raw exception: {exc.__class__.__name__}: {exc}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if args.hold_final > 0:
            time.sleep(args.hold_final)


def interactive_menu(status: StatusDisplay, args: argparse.Namespace) -> int:
    while True:
        print()
        print_scenarios()
        print("   0. exit")
        choice = input("Choose a scenario number (or key): ").strip()
        if choice in {"0", "q", "quit", "exit"}:
            status.show("BYE", "OLED test done", "")
            return 0

        key: Optional[str] = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(SCENARIOS):
                key = SCENARIOS[idx].key
        elif choice in SCENARIO_MAP:
            key = choice

        if key is None:
            print("[WARN] Invalid selection.")
            continue

        run_scenario(key, status, args)


# -----------------------------
# Argument parsing
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson + SSD1306 OLED deep-learning status/error tester"
    )
    parser.add_argument(
        "--scenario",
        default="menu",
        help="Scenario key to run once. Use 'menu' for interactive mode.",
        choices=["menu"] + [s.key for s in SCENARIOS],
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--bus", type=int, default=None, help="I2C bus number. If omitted, auto-scan /dev/i2c-*")
    parser.add_argument(
        "--addr",
        type=lambda x: int(x, 0),
        default=None,
        help="OLED I2C address, e.g. 0x3C. If omitted, scan 0x3C and 0x3D",
    )
    parser.add_argument("--no-oled", action="store_true", help="Console-only mode, do not open OLED")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--steps", type=int, default=12, help="Number of model steps in the normal run")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the normal run")
    parser.add_argument("--step-sleep", type=float, default=0.20, help="Extra delay per step so the OLED status is visible")
    parser.add_argument("--hold-final", type=float, default=2.0, help="Seconds to keep the final OLED message on screen")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_header()

    if args.list:
        print_scenarios()
        return 0

    status: Optional[StatusDisplay] = None
    try:
        oled = None
        if not args.no_oled:
            oled, how = autodetect_oled(args.bus, args.addr)
            if oled is None:
                print(f"[WARN] OLED not found. Falling back to console-only mode.")
                print(f"       Details: {how}")
            else:
                print(f"[INFO] OLED connected via {how}")
        status = StatusDisplay(oled)
        status.boot()
        time.sleep(1.0)

        if args.scenario == "menu":
            return interactive_menu(status, args)
        return run_scenario(args.scenario, status, args)
    finally:
        if status is not None:
            status.close()


if __name__ == "__main__":
    raise SystemExit(main())
