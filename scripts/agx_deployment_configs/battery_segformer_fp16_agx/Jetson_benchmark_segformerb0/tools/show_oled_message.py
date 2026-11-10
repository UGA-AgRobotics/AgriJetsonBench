#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jetson.oled_status import BenchmarkOLED
else:  # pragma: no cover
    from ..jetson.oled_status import BenchmarkOLED


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Show a short status message on the Jetson OLED.")
    p.add_argument("--line1", default="", help="First OLED line")
    p.add_argument("--line2", default="", help="Second OLED line")
    p.add_argument("--line3", default="", help="Third OLED line")
    p.add_argument("--bus", type=int, default=None, help="OLED I2C bus")
    p.add_argument("--addr", type=lambda x: int(x, 0), default=None, help="OLED I2C address, e.g. 0x3C")
    p.add_argument("--require", action="store_true", help="Exit non-zero when the OLED is unavailable")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    oled = BenchmarkOLED(enabled=True, bus=args.bus, addr=args.addr)
    if not oled.active:
        if args.require:
            print(f"[ERROR] OLED unavailable: {oled.detect_info}", file=sys.stderr)
            return 1
        print(f"[WARN] OLED unavailable: {oled.detect_info}", file=sys.stderr)
        return 0
    try:
        oled.show(args.line1, args.line2, args.line3)
        return 0
    finally:
        oled.close()


if __name__ == "__main__":
    raise SystemExit(main())
