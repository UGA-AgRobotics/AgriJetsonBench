from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jetson.noise_guard import NoiseGuard, NoiseGuardConfig
    from jetson.oled_status import BenchmarkOLED
else:  # pragma: no cover
    from .noise_guard import NoiseGuard, NoiseGuardConfig
    from .oled_status import BenchmarkOLED


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def show_oled_message(enabled: bool, bus: int | None, addr: int | None, line1: str, line2: str = "", line3: str = "") -> None:
    if not enabled:
        return
    oled = BenchmarkOLED(enabled=True, bus=bus, addr=addr)
    try:
        if oled.active:
            oled.show(line1, line2, line3)
    finally:
        oled.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a command while holding a single session-wide NoiseGuard context.")
    p.add_argument("--summary-json", required=True, help="Where to write the session-wide noise-guard summary JSON")
    p.add_argument("--profile", default="safe")
    p.add_argument("--preserve-url", default=None)
    p.add_argument("--display-off", dest="display_off", action="store_true")
    p.add_argument("--no-display-off", dest="display_off", action="store_false")
    p.add_argument("--stop-services", dest="stop_services", action="store_true")
    p.add_argument("--no-stop-services", dest="stop_services", action="store_false")
    p.add_argument("--disable-wifi", dest="disable_wifi", action="store_true")
    p.add_argument("--no-disable-wifi", dest="disable_wifi", action="store_false")
    p.add_argument("--disable-bluetooth", dest="disable_bluetooth", action="store_true")
    p.add_argument("--no-disable-bluetooth", dest="disable_bluetooth", action="store_false")
    p.add_argument("--oled-enable", action="store_true")
    p.add_argument("--oled-bus", type=int, default=None)
    p.add_argument("--oled-addr", type=lambda x: int(x, 0), default=None)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(display_off=True, stop_services=True, disable_wifi=True, disable_bluetooth=True)
    return p


def normalize_command(raw: List[str]) -> List[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def main() -> int:
    args = build_arg_parser().parse_args()
    command = normalize_command(args.command)
    if not command:
        print("No command provided to run_with_noise_guard.py", file=sys.stderr)
        return 2

    summary_path = Path(args.summary_json).expanduser().resolve()
    notes: List[str] = []
    guard = NoiseGuard(
        NoiseGuardConfig(
            enabled=True,
            profile=args.profile,
            preserve_url=args.preserve_url,
            display_off=bool(args.display_off),
            stop_services=bool(args.stop_services),
            disable_wifi=bool(args.disable_wifi),
            disable_bluetooth=bool(args.disable_bluetooth),
        ),
        notes=notes,
    )

    return_code = 0
    show_oled_message(args.oled_enable, args.oled_bus, args.oled_addr, "QUIET MODE", "preparing run", "monitor can rest")
    try:
        dump_json(guard.enter(), summary_path)
        show_oled_message(args.oled_enable, args.oled_bus, args.oled_addr, "RUN STARTING", "leave monitor off", "wifi handled")
        completed = subprocess.run(command, check=False)
        return_code = int(completed.returncode)
        return return_code
    except KeyboardInterrupt:
        return_code = 130
        return return_code
    finally:
        try:
            dump_json(guard.restore(), summary_path)
        except Exception as exc:
            notes.append(f"Noise guard restore failed: {exc}")
            try:
                dump_json(guard.summary(), summary_path)
            except Exception:
                pass
        if return_code == 0:
            show_oled_message(args.oled_enable, args.oled_bus, args.oled_addr, "TURN MONITOR ON", "wifi restored", "run complete")
        else:
            show_oled_message(args.oled_enable, args.oled_bus, args.oled_addr, "TURN MONITOR ON", "check terminal", f"rc {return_code}")


if __name__ == "__main__":
    raise SystemExit(main())
