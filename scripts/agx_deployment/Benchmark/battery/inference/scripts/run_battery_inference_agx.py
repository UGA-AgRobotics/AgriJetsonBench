#!/usr/bin/env python3
"""Inference-only endurance runner for AGX battery validation.

This wrapper intentionally does not control the external Pc_package logger.
It creates inference-side records, applies a session-wide noise guard, and
delegates measured inference chunks to the existing Jetson benchmark entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LATENCY_FIELDS = [
    "timestamp_wall",
    "timestamp_monotonic",
    "run_id",
    "board",
    "task",
    "model",
    "precision",
    "input_size",
    "batch_size",
    "loop_index",
    "image_index",
    "image_id_or_path",
    "latency_ms",
    "success",
    "error_message",
]

EVENT_FIELDS = [
    "timestamp_wall",
    "timestamp_monotonic",
    "run_id",
    "board",
    "event",
    "payload_json",
]

SUMMARY_FIELDS = [
    "run_id",
    "board",
    "task",
    "model",
    "precision",
    "input_size",
    "batch_size",
    "total_images",
    "total_elapsed_seconds",
    "fps",
    "median_latency_ms",
    "p95_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "stopped_reason",
]


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class InternalTelemetrySnapshotter:
    """Small on-demand telemetry reader for the Pc_package live dashboard."""

    def __init__(self, repo_root: Path, output_path: Path):
        self.repo_root = repo_root
        self.output_path = output_path
        self.collect_hwmon_records = None
        self.summarize_hwmon_records = None
        self.collect_thermal_records = None
        self.summarize_thermal_records = None
        self.parse_tegrastats_line = None
        self.import_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        sys.path.insert(0, str(self.repo_root))
        try:
            from jetson.internal_telemetry import (  # type: ignore
                collect_hwmon_records,
                collect_thermal_records,
                parse_tegrastats_line,
                summarize_hwmon_records,
                summarize_thermal_records,
            )

            self.collect_hwmon_records = collect_hwmon_records
            self.summarize_hwmon_records = summarize_hwmon_records
            self.collect_thermal_records = collect_thermal_records
            self.summarize_thermal_records = summarize_thermal_records
            self.parse_tegrastats_line = parse_tegrastats_line
        except Exception as exc:
            self.import_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                sys.path.remove(str(self.repo_root))
            except ValueError:
                pass

    def snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {"hwmon": {}, "thermal": {}, "tegrastats": {}}
        sched_ns = time.time_ns()
        if self.collect_hwmon_records is not None and self.summarize_hwmon_records is not None:
            try:
                hw_records = self.collect_hwmon_records(sched_wallclock_ns=sched_ns)
                snap["hwmon"] = self.summarize_hwmon_records(hw_records)
            except Exception as exc:
                snap["hwmon"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            snap["hwmon"] = self._fallback_hwmon()
        if self.collect_thermal_records is not None and self.summarize_thermal_records is not None:
            try:
                thermal_records = self.collect_thermal_records(sched_wallclock_ns=sched_ns)
                snap["thermal"] = self.summarize_thermal_records(thermal_records)
            except Exception as exc:
                snap["thermal"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            snap["thermal"] = self._fallback_thermal()
        snap["tegrastats"] = self._tegrastats_once()
        hwmon = snap.get("hwmon") if isinstance(snap.get("hwmon"), dict) else {}
        tstats = snap.get("tegrastats") if isinstance(snap.get("tegrastats"), dict) else {}
        if not hwmon.get("total_power_w") and tstats.get("power_rails"):
            rails = tstats.get("power_rails") if isinstance(tstats.get("power_rails"), list) else []
            powers = [float(r["power_w"]) for r in rails if isinstance(r, dict) and r.get("power_w") is not None]
            snap["hwmon"] = {
                "available": bool(rails),
                "total_power_w": sum(powers) if powers else None,
                "rails": rails,
                "source": "tegrastats_power_fallback",
            }
        if self.import_error:
            snap["telemetry_import_warning"] = self.import_error
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp_utc": utc_now(), "internal_live": snap}, sort_keys=True, default=str) + "\n")
        except Exception:
            pass
        return snap

    @staticmethod
    def _read_text(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return None

    @classmethod
    def _read_number(cls, path: Path) -> Optional[float]:
        text = cls._read_text(path)
        if text is None or text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _power_raw_to_w(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if abs(value) >= 1_000_000.0:
            return value / 1_000_000.0
        if abs(value) >= 1_000.0:
            return value / 1_000.0
        return value

    def _fallback_hwmon(self) -> Dict[str, Any]:
        rails: List[Dict[str, Any]] = []
        for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            name = self._read_text(root / "name") or root.name
            for power_path in sorted(root.glob("power*_input")):
                match = re.match(r"power(\d+)_input", power_path.name)
                channel = match.group(1) if match else ""
                label = (
                    self._read_text(root / f"power{channel}_label")
                    or self._read_text(root / f"in{channel}_label")
                    or f"{name}:{power_path.stem}"
                )
                raw_power = self._read_number(power_path)
                voltage_mv = self._read_number(root / f"in{channel}_input")
                current_ma = self._read_number(root / f"curr{channel}_input")
                power_w = self._power_raw_to_w(raw_power)
                rails.append(
                    {
                        "label": label,
                        "sensor_name": name,
                        "power_w": power_w,
                        "power_mw": power_w * 1000.0 if power_w is not None else None,
                        "voltage_mv": voltage_mv,
                        "current_ma": current_ma,
                    }
                )
        power_values = [float(r["power_w"]) for r in rails if r.get("power_w") is not None]
        return {
            "available": bool(rails),
            "total_power_w": sum(power_values) if power_values else None,
            "rails": rails,
            "source": "sysfs_fallback",
        }

    def _fallback_thermal(self) -> Dict[str, Any]:
        zones: List[Dict[str, Any]] = []
        for root in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            zone_type = self._read_text(root / "type") or root.name
            raw_temp = self._read_number(root / "temp")
            temp_c = None
            if raw_temp is not None:
                temp_c = raw_temp / 1000.0 if abs(raw_temp) > 500.0 else raw_temp
            zones.append({"thermal_zone": root.name, "type": zone_type, "temp_c": temp_c})
        temps = [float(z["temp_c"]) for z in zones if z.get("temp_c") is not None]
        return {
            "available": bool(zones),
            "max_temp_c": max(temps) if temps else None,
            "zones": zones,
            "source": "sysfs_fallback",
        }

    @staticmethod
    def _parse_tegrastats_power_rails(line: str) -> List[Dict[str, Any]]:
        rails: List[Dict[str, Any]] = []
        for match in re.finditer(r"\b([A-Z][A-Z0-9_]+)\s+(\d+(?:\.\d+)?)mW(?:/(\d+(?:\.\d+)?)mW)?", line):
            label = match.group(1)
            power_mw = float(match.group(2))
            avg_power_mw = float(match.group(3)) if match.group(3) is not None else None
            rails.append(
                {
                    "label": label,
                    "power_w": power_mw / 1000.0,
                    "power_mw": power_mw,
                    "avg_power_w": avg_power_mw / 1000.0 if avg_power_mw is not None else None,
                    "avg_power_mw": avg_power_mw,
                }
            )
        return rails

    @staticmethod
    def _parse_tegrastats_fallback(line: str) -> Dict[str, Any]:
        def match_float(pattern: str) -> Optional[float]:
            m = re.search(pattern, line)
            return float(m.group(1)) if m else None

        temps = {m.group(1): float(m.group(2)) for m in re.finditer(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C", line)}
        power_rails = InternalTelemetrySnapshotter._parse_tegrastats_power_rails(line)
        return {
            "ram_used_mb": match_float(r"RAM\s+(\d+(?:\.\d+)?)/\d+(?:\.\d+)?MB"),
            "ram_total_mb": match_float(r"RAM\s+\d+(?:\.\d+)?/(\d+(?:\.\d+)?)MB"),
            "swap_used_mb": match_float(r"SWAP\s+(\d+(?:\.\d+)?)/\d+(?:\.\d+)?MB"),
            "swap_total_mb": match_float(r"SWAP\s+\d+(?:\.\d+)?/(\d+(?:\.\d+)?)MB"),
            "gr3d_pct": match_float(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%"),
            "emc_pct": match_float(r"EMC_FREQ\s+(\d+(?:\.\d+)?)%"),
            "temps": temps,
            "max_temp_c": max(temps.values()) if temps else None,
            "power_rails": power_rails,
            "total_power_w": sum(float(r["power_w"]) for r in power_rails) if power_rails else None,
        }

    def _tegrastats_once(self) -> Dict[str, Any]:
        if shutil.which("tegrastats") is None:
            return {"available": False, "error": "tegrastats not found"}
        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            line = ""
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                if proc.stdout is None:
                    break
                ready, _, _ = select.select([proc.stdout], [], [], 0.25)
                if ready:
                    candidate = proc.stdout.readline().strip()
                    if candidate:
                        line = candidate
                        break
                if proc.poll() is not None:
                    break
            if not line:
                return {"available": False, "error": "tegrastats produced no line before timeout"}
            parser = self.parse_tegrastats_line or self._parse_tegrastats_fallback
            parsed = parser(line)
            if not isinstance(parsed, dict):
                return {"available": False, "raw_line": line, "error": "tegrastats parser did not return a dict"}
            parsed["raw_line"] = line
            parsed["available"] = True
            power_rails = self._parse_tegrastats_power_rails(line)
            if power_rails and not parsed.get("power_rails"):
                parsed["power_rails"] = power_rails
                parsed["total_power_w"] = sum(float(r["power_w"]) for r in power_rails)
            if "temps" in parsed and isinstance(parsed["temps"], dict):
                temps = [float(v) for v in parsed["temps"].values() if v is not None]
                parsed["max_temp_c"] = max(temps) if temps else None
            return parsed
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("agx_battery_%Y%m%dT%H%M%SZ")


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." not in value:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(text: str) -> Dict[str, Any]:
    """Small YAML subset parser for this config file when PyYAML is absent."""
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]
    for idx, (indent, stripped) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise RuntimeError("Simple YAML parser expected a list parent.")
            parent.append(parse_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise RuntimeError(f"Simple YAML parser cannot parse line: {stripped}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            if not isinstance(parent, dict):
                raise RuntimeError("Simple YAML parser expected a mapping parent.")
            parent[key] = parse_scalar(value)
            continue
        next_is_list = False
        if idx + 1 < len(lines):
            next_indent, next_stripped = lines[idx + 1]
            next_is_list = next_indent > indent and next_stripped.startswith("- ")
        child: Any = [] if next_is_list else {}
        if not isinstance(parent, dict):
            raise RuntimeError("Simple YAML parser expected a mapping parent.")
        parent[key] = child
        stack.append((indent, child))
    return root


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise RuntimeError(f"Config root must be a mapping: {path}")
        return data
    except ModuleNotFoundError:
        return simple_yaml_load(text)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def run_cmd(cmd: List[str], timeout: int = 20, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    started = utc_now()
    try:
        cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": cmd,
            "started_utc": started,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "ok": cp.returncode == 0,
        }
    except Exception as exc:
        return {"cmd": cmd, "started_utc": started, "returncode": None, "stdout": "", "stderr": str(exc), "ok": False}


def try_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except Exception:
        return None


def find_git_commit(start: Path) -> Optional[str]:
    for candidate in [start, *start.parents]:
        res = run_cmd(["git", "-C", str(candidate), "rev-parse", "HEAD"], timeout=5)
        if res.get("ok"):
            return str(res.get("stdout", "")).strip()
    return None


def resolve_trtexec_env() -> Tuple[Dict[str, str], Optional[str]]:
    env = os.environ.copy()
    base = Path("${AGX_HOME}/Benchmark/.vendor/tensorrt-${PRIVATE_LAN_IP}")
    trtexec = base / "usr/src/tensorrt/bin/trtexec"
    if trtexec.exists():
        env["LOCAL_TRTEXEC_PATH"] = str(trtexec)
        lib_dir = base / "usr/lib/aarch64-linux-gnu"
        if lib_dir.exists():
            env["LOCAL_TENSORRT_LIB_DIR"] = str(lib_dir)
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        return env, str(trtexec)
    return env, None


def http_get_json(url: str, timeout: int = 5) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return {"ok": True, "url": url, "data": parsed}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        if isinstance(parsed, dict):
            return {"ok": bool(parsed.get("ok", True)), "url": url, "data": parsed, "http_status": getattr(resp, "status", None)}
        return {"ok": True, "url": url, "data": parsed, "http_status": getattr(resp, "status", None)}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "url": url, "error": detail or str(exc), "http_status": exc.code}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "url": url, "error": str(exc), "http_status": None}


def http_download(url: str, out_path: Path, timeout: int = 600) -> Dict[str, Any]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            out_path.write_bytes(resp.read())
        return {"ok": True, "url": url, "path": str(out_path), "bytes": out_path.stat().st_size}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "url": url, "path": str(out_path), "error": str(exc)}


def nvpmodel_looks_15w(preflight: Dict[str, Any]) -> bool:
    parts: List[str] = []
    for key, value in preflight.items():
        if key.startswith("nvpmodel") and isinstance(value, dict):
            parts.append(str(value.get("stdout", "")))
            parts.append(str(value.get("stderr", "")))
    combined = " ".join(parts)
    return "15W" in combined.replace(" ", "").upper()


def logger_zero_power_warning(read_once: Dict[str, Any]) -> Optional[str]:
    if not read_once.get("ok"):
        return None
    data = read_once.get("data") or {}
    reading = data.get("reading") if isinstance(data, dict) else None
    if not isinstance(reading, dict):
        return None
    def as_float(key: str) -> Optional[float]:
        try:
            value = reading.get(key)
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None
    voltage = as_float("bus_voltage_v")
    power = as_float("power_w")
    if (voltage is not None and voltage < 1.0) or (power is not None and power <= 0.0):
        return (
            "Pc_package read_once is reachable but the INA260 reading is not a valid AGX load "
            f"(bus_voltage_v={voltage}, power_w={power}); confirm wiring/current path before endurance."
        )
    return None


def logger_health_active_for_run(health: Dict[str, Any], run_id: str) -> Tuple[bool, str]:
    if not health.get("ok"):
        return False, f"logger health request failed: {health.get('error')}"
    data = health.get("data") or {}
    if not isinstance(data, dict):
        return False, "logger health response is not a JSON object"
    current = data.get("current_run") or {}
    devices = data.get("devices") or []
    active = bool(current.get("active"))
    current_run_id = current.get("run_id")
    selected_key = data.get("selected_device_key")
    selected_device = None
    if isinstance(devices, list):
        for dev in devices:
            if isinstance(dev, dict) and dev.get("key") == selected_key:
                selected_device = dev
                break
    selected_active = bool((selected_device or {}).get("external_logger_active"))
    selected_state = (selected_device or {}).get("state")
    selected_run_id = (selected_device or {}).get("run_id")
    if not active and not selected_active:
        return False, f"logger is reachable but idle/waiting (selected_state={selected_state}, current_run.active={active})"
    seen_run_ids = {str(x) for x in [current_run_id, selected_run_id] if x}
    if seen_run_ids and str(run_id) not in seen_run_ids:
        return False, f"logger active run_id mismatch: expected {run_id}, saw {sorted(seen_run_ids)}"
    return True, "logger active"


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats_for(latencies: List[float], elapsed_s: float, total_images: int) -> Dict[str, Any]:
    fps = (float(total_images) / elapsed_s) if elapsed_s > 0 and total_images > 0 else 0.0
    return {
        "total_images": total_images,
        "elapsed_seconds": elapsed_s,
        "fps": fps,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


class BatteryInferenceRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        self.base_dir = self.config_path.parent.parent.resolve()
        self.config = load_config(self.config_path)
        self.board = str(self.config.get("board", "agx"))
        self.run_id = args.run_id or timestamp_run_id()
        output_root = resolve_path(str(args.out_dir or self.config.get("output_root", "./battery_inference_runs/agx")), self.base_dir)
        self.run_dir = output_root / self.run_id
        self.events_path = self.run_dir / "events.csv"
        self.latency_path = self.run_dir / "inference_latency.csv"
        self.summary_csv_path = self.run_dir / "inference_summary.csv"
        self.stdout_path = self.run_dir / "stdout.log"
        self.stderr_path = self.run_dir / "stderr.log"
        self.stop_requested = False
        self.current_child: Optional[subprocess.Popen[str]] = None
        self.latencies: List[float] = []
        self.total_images = 0
        self.inference_start_mono: Optional[float] = None
        self.inference_start_wall: Optional[str] = None
        self.stopped_reason = "not_started"
        self.last_child_command: List[str] = []
        self.preflight: Dict[str, Any] = {}
        self.sudo_keepalive: Optional[subprocess.Popen[str]] = None
        self.external_logger_started = False
        self.external_logger_stop_info: Optional[Dict[str, Any]] = None
        self.external_logger_archive_path: Optional[Path] = None
        self.logger_control = str(args.logger_control or self.config.get("external_logger_control", "auto")).lower()
        self.run_start_mono = time.monotonic()
        self.dashboard_status_url = str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")).rstrip("/") + "/api/jetson_status"
        self.dashboard_status_interval = max(1.0, float(args.dashboard_status_interval))
        self.dashboard_status_stop = threading.Event()
        self.dashboard_status_thread: Optional[threading.Thread] = None
        self.dashboard_status_lock = threading.Lock()
        self.dashboard_status_stage_start_mono = self.run_start_mono
        self.dashboard_status_push_failed = False
        self.dashboard_status_snapshotter: Optional[InternalTelemetrySnapshotter] = None
        self.dashboard_status_state: Dict[str, Any] = {
            "stage": "created",
            "state": "created",
            "message": "Battery inference run created",
            "repeat_name": None,
            "processed_images": 0,
            "min_images_target": None,
            "min_duration_s_target": None,
            "progress_pct": None,
            "current_image_name": None,
            "latency_ms_last": None,
            "loop_index": None,
        }
        self.noise_guard: Optional[Any] = None
        self.selected_name, self.selected_cfg = self.select_config()
        self.precision = str(args.precision or self.selected_cfg.get("precision", "INT8")).upper()
        self.input_size = int(args.input_size or self.selected_cfg.get("input_size", 640))
        self.batch_size = int(args.batch_size or self.selected_cfg.get("batch_size", 1))
        self.engine_path = self.resolve_engine()
        self.dataset_path = resolve_path(str(self.selected_cfg.get("dataset_path")), self.base_dir)
        self.entrypoint = resolve_path(str(self.selected_cfg.get("inference_entrypoint")), self.base_dir)
        self.child_output_root = self.run_dir / "child_benchmark_runs"
        self.env, self.trtexec_path = resolve_trtexec_env()

    def select_config(self) -> Tuple[str, Dict[str, Any]]:
        configs = self.config.get("configs") or {}
        if not isinstance(configs, dict):
            raise RuntimeError("Config file is missing the configs mapping.")
        if self.args.config_name:
            if self.args.config_name not in configs:
                raise RuntimeError(f"Unknown --config-name {self.args.config_name}. Available: {sorted(configs)}")
            return self.args.config_name, dict(configs[self.args.config_name])
        candidates = []
        for name, cfg in configs.items():
            if self.args.model and str(cfg.get("model", "")).lower() != self.args.model.lower():
                continue
            if self.args.task and str(cfg.get("task", "")).lower() != self.args.task.lower():
                continue
            if str(cfg.get("precision", "INT8")).upper() != str(self.args.precision or "INT8").upper():
                continue
            if int(cfg.get("input_size", 640)) != int(self.args.input_size or 640):
                continue
            candidates.append((name, dict(cfg)))
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError("Pass exactly one selected --config-name; this runner never runs both models automatically.")

    def resolve_engine(self) -> Path:
        if self.precision == "FP16":
            key = "fp16_engine_path"
        elif self.precision == "INT8":
            key = "engine_path"
        else:
            raise RuntimeError(f"Unsupported precision: {self.precision}")
        if key not in self.selected_cfg:
            raise RuntimeError(f"Config {self.selected_name} has no {key}.")
        return resolve_path(str(self.selected_cfg[key]), self.base_dir)

    def event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        row = {
            "timestamp_wall": utc_now(),
            "timestamp_monotonic": f"{time.monotonic():.9f}",
            "run_id": self.run_id,
            "board": self.board,
            "event": event,
            "payload_json": json.dumps(payload or {}, sort_keys=True, default=str),
        }
        first = not self.events_path.exists()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
            if first:
                writer.writeheader()
            writer.writerow(row)

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def detected_power_mode_label(self) -> Optional[str]:
        for key in ("nvpmodel_q_after_set", "nvpmodel_q", "nvpmodel_q_no_sudo"):
            value = self.preflight.get(key)
            if not isinstance(value, dict):
                continue
            text = "\n".join([str(value.get("stdout", "")), str(value.get("stderr", ""))])
            for line in text.splitlines():
                if "NV Power Mode:" in line:
                    return line.split(":", 1)[1].strip()
        return None

    def dashboard_targets(self) -> Tuple[Optional[int], Optional[float]]:
        if self.args.mode == "smoke":
            return max(500, int(self.args.max_images or 0)), max(180.0, float(self.args.max_seconds or 0.0))
        target_images = int(self.args.max_images) if self.args.max_images is not None else None
        target_seconds = float(self.args.max_seconds) if self.args.max_seconds is not None else None
        return target_images, target_seconds

    def update_dashboard_status(
        self,
        *,
        stage: Optional[str] = None,
        state: Optional[str] = None,
        message: Optional[str] = None,
        reset_stage_timer: bool = False,
        **extra: Any,
    ) -> None:
        now = time.monotonic()
        with self.dashboard_status_lock:
            if stage is not None and (reset_stage_timer or stage != self.dashboard_status_state.get("stage")):
                self.dashboard_status_stage_start_mono = now
            if stage is not None:
                self.dashboard_status_state["stage"] = stage
            if state is not None:
                self.dashboard_status_state["state"] = state
            if message is not None:
                self.dashboard_status_state["message"] = message
            self.dashboard_status_state.update(extra)

    def dashboard_noise_guard_payload(self) -> Optional[Dict[str, Any]]:
        guard = self.noise_guard
        if guard is None:
            return None
        try:
            payload = guard.dashboard_payload()
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def build_dashboard_status_payload(self) -> Dict[str, Any]:
        now_mono = time.monotonic()
        now_ns = time.time_ns()
        target_images, target_seconds = self.dashboard_targets()
        elapsed = max(0.0, now_mono - self.run_start_mono)
        stage_elapsed = max(0.0, now_mono - self.dashboard_status_stage_start_mono)
        st = stats_for(self.latencies, max(0.001, now_mono - (self.inference_start_mono or self.run_start_mono)), self.total_images)
        progress_pct = 0.0
        if target_images:
            progress_pct = max(progress_pct, min(100.0, 100.0 * float(self.total_images) / float(target_images)))
        if target_seconds:
            progress_pct = max(progress_pct, min(100.0, 100.0 * float(stage_elapsed) / float(target_seconds)))
        if self.dashboard_status_state.get("stage") == "finished":
            progress_pct = 100.0
        internal_live: Dict[str, Any] = {}
        if self.dashboard_status_snapshotter is not None:
            internal_live = self.dashboard_status_snapshotter.snapshot()
        with self.dashboard_status_lock:
            state = dict(self.dashboard_status_state)
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "device_key": "jetson-agx",
            "board_name": self.config.get("board_display_name", "NVIDIA Jetson AGX Orin 64GB Developer Kit"),
            "board": self.board,
            "config_name": self.selected_name,
            "task": self.selected_cfg.get("task"),
            "model": self.selected_cfg.get("model"),
            "precision_label": self.precision.lower(),
            "run_variant": "battery_inference_endurance",
            "stage": state.get("stage"),
            "state": state.get("state"),
            "message": state.get("message"),
            "timestamp_utc": utc_now(),
            "wallclock_ns": now_ns,
            "overall_elapsed_s": elapsed,
            "stage_elapsed_s": stage_elapsed,
            "stage_eta_s": None,
            "progress_pct": float(state.get("progress_pct") if state.get("progress_pct") is not None else progress_pct),
            "repeat_index": state.get("loop_index"),
            "total_repeats": None,
            "repeat_name": state.get("repeat_name"),
            "processed_images": self.total_images,
            "min_images_target": target_images,
            "min_duration_s_target": target_seconds,
            "current_image_name": state.get("current_image_name"),
            "latency_ms_last": state.get("latency_ms_last"),
            "fps": st["fps"],
            "median_latency_ms": st["median_latency_ms"],
            "p95_latency_ms": st["p95_latency_ms"],
            "power_mode": {
                "power_mode_label": self.detected_power_mode_label() or self.config.get("target_power_mode_label", "15W"),
                "target_power_mode_label": self.config.get("target_power_mode_label", "15W"),
                "power_mode_looks_15w": self.preflight.get("power_mode_looks_15w"),
            },
            "internal_live": internal_live,
            "noise_guard": self.dashboard_noise_guard_payload(),
        }
        payload.update({k: v for k, v in state.items() if k not in payload and not k.endswith("_mono")})
        return payload

    def push_dashboard_status_once(self) -> Dict[str, Any]:
        payload = self.build_dashboard_status_payload()
        try:
            path = self.run_dir / "dashboard_status_payloads.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        except Exception:
            pass
        result = http_post_json(self.dashboard_status_url, payload, timeout=4)
        if result.get("ok"):
            if self.dashboard_status_push_failed:
                self.log("[LOGGER] dashboard status push restored.")
            self.dashboard_status_push_failed = False
        else:
            if not self.dashboard_status_push_failed:
                self.log(f"[WARN] dashboard status push failed once: {result}")
            self.dashboard_status_push_failed = True
        return result

    def dashboard_status_loop(self) -> None:
        while not self.dashboard_status_stop.is_set():
            self.push_dashboard_status_once()
            self.dashboard_status_stop.wait(self.dashboard_status_interval)

    def start_dashboard_status_pusher(self) -> None:
        if self.dashboard_status_thread is not None and self.dashboard_status_thread.is_alive():
            return
        self.dashboard_status_snapshotter = InternalTelemetrySnapshotter(
            self.entrypoint.parent.parent,
            self.run_dir / "dashboard_internal_live.jsonl",
        )
        if self.dashboard_status_snapshotter.import_error:
            self.log(f"[WARN] internal dashboard telemetry import failed: {self.dashboard_status_snapshotter.import_error}")
        self.dashboard_status_stop.clear()
        self.dashboard_status_thread = threading.Thread(target=self.dashboard_status_loop, name="dashboard-status-pusher", daemon=True)
        self.dashboard_status_thread.start()
        self.event("DASHBOARD_STATUS_PUSH_START", {"url": self.dashboard_status_url, "interval_s": self.dashboard_status_interval})

    def stop_dashboard_status_pusher(self) -> None:
        thread = self.dashboard_status_thread
        if thread is None:
            return
        self.dashboard_status_stop.set()
        thread.join(timeout=max(2.0, self.dashboard_status_interval + 1.0))
        self.event("DASHBOARD_STATUS_PUSH_STOP", {"thread_alive": thread.is_alive()})
        self.dashboard_status_thread = None

    def setup_files(self) -> Tuple[Any, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stdout_f = self.stdout_path.open("a", encoding="utf-8")
        stderr_f = self.stderr_path.open("a", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, stdout_f)  # type: ignore[assignment]
        sys.stderr = Tee(sys.__stderr__, stderr_f)  # type: ignore[assignment]
        if not self.latency_path.exists():
            with self.latency_path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=LATENCY_FIELDS).writeheader()
        self.event("RUN_CREATED", {"run_dir": str(self.run_dir), "config_name": self.selected_name})
        return stdout_f, stderr_f

    def write_initial_metadata(self) -> None:
        run_config = {
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "config_name": self.selected_name,
            "mode": self.args.mode,
            "board": self.board,
            "board_display_name": self.config.get("board_display_name"),
            "task": self.selected_cfg.get("task"),
            "model": self.selected_cfg.get("model"),
            "precision": self.precision,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "engine_path": str(self.engine_path),
            "dataset_path": str(self.dataset_path),
            "split": self.selected_cfg.get("split", "test"),
            "save_predictions": bool(self.args.save_predictions),
            "external_logger_url": self.config.get("external_logger_url"),
            "external_logger_control": self.config.get("external_logger_control", "manual"),
            "power_source": self.config.get("power_source"),
        }
        json_dump(run_config, self.run_dir / "run_config.json")
        json_dump({"run_id": self.run_id, "files": self.output_files_manifest()}, self.run_dir / "run_manifest.json")
        note = (
            f"External power logging package: {self.config.get('external_logger_package', 'Pc_package')}.\n"
            f"Logger URL expected for read-only checks: {self.config.get('external_logger_url', 'http://${LOGGER_DASHBOARD_HOST}:8000')}.\n"
            f"Run ID: {self.run_id}\n"
            f"Logger control mode: {self.logger_control}\n"
            "In auto mode, this wrapper starts and stops Pc_package recording through /start and /stop.\n"
        )
        (self.run_dir / "external_logger_note.txt").write_text(note, encoding="utf-8")

    def output_files_manifest(self) -> Dict[str, str]:
        names = [
            "run_config.json",
            "run_manifest.json",
            "preflight_report.json",
            "events.csv",
            "inference_latency.csv",
            "inference_summary.csv",
            "stdout.log",
            "stderr.log",
            "summary.json",
            "summary.md",
            "external_logger_note.txt",
        ]
        return {name: str(self.run_dir / name) for name in names}

    def validate_paths(self) -> List[str]:
        errors: List[str] = []
        if not self.engine_path.exists():
            errors.append(f"missing engine_path: {self.engine_path}")
        if not self.dataset_path.exists():
            errors.append(f"missing dataset_path: {self.dataset_path}")
        if not self.entrypoint.exists():
            errors.append(f"missing inference_entrypoint: {self.entrypoint}")
        if self.input_size != int(self.selected_cfg.get("input_size", self.input_size)):
            errors.append(f"input size changed from config: config={self.selected_cfg.get('input_size')} requested={self.input_size}")
        if self.batch_size != int(self.selected_cfg.get("batch_size", self.batch_size)) and not self.args.batch_size:
            errors.append(f"batch size mismatch: config={self.selected_cfg.get('batch_size')} requested={self.batch_size}")
        return errors

    def collect_preflight(self) -> Dict[str, Any]:
        self.event("PREFLIGHT_START")
        logger_url = str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")).rstrip("/")
        preflight = {
            "timestamp_utc": utc_now(),
            "nvpmodel_q": run_cmd(["sudo", "-n", "nvpmodel", "-q"], timeout=20),
            "nvpmodel_q_verbose": run_cmd(["sudo", "-n", "nvpmodel", "-q", "--verbose"], timeout=30),
            "device_tree_model": try_read_text(Path("/proc/device-tree/model")),
            "l4t_release": try_read_text(Path("/etc/nv_tegra_release")),
            "cuda_version": run_cmd(["nvcc", "--version"], timeout=10),
            "tensorrt_version": run_cmd([self.trtexec_path or "trtexec", "--version"], timeout=10, env=self.env),
            "git_commit": find_git_commit(self.base_dir),
            "logger_health": http_get_json(logger_url + "/health", timeout=3),
            "logger_read_once": http_get_json(logger_url + "/read_once", timeout=5),
            "path_validation_errors": self.validate_paths(),
            "target_power_mode_label": self.config.get("target_power_mode_label", "15W"),
            "allow_power_mode_mismatch": bool(self.args.allow_power_mode_mismatch),
            "set_nvpmodel_requested": bool(self.args.set_nvpmodel),
        }
        if not preflight["nvpmodel_q"].get("ok"):
            preflight["nvpmodel_q_no_sudo"] = run_cmd(["nvpmodel", "-q"], timeout=20)
        if not preflight["nvpmodel_q_verbose"].get("ok"):
            preflight["nvpmodel_q_verbose_no_sudo"] = run_cmd(["nvpmodel", "-q", "--verbose"], timeout=30)
        if self.args.set_nvpmodel:
            mode_id = str(self.config.get("nvpmodel_15w_mode_id", "TODO"))
            if mode_id.upper() == "TODO" or mode_id.strip() == "":
                preflight["set_nvpmodel_error"] = "nvpmodel_15w_mode_id is TODO; refusing to hard-code or guess the AGX 15W mode ID."
            else:
                preflight["set_nvpmodel"] = run_cmd(["sudo", "-n", "nvpmodel", "-m", mode_id], timeout=30)
                preflight["nvpmodel_q_after_set"] = run_cmd(["sudo", "-n", "nvpmodel", "-q"], timeout=20)
        preflight["power_mode_looks_15w"] = nvpmodel_looks_15w(preflight)
        preflight["logger_zero_power_warning"] = logger_zero_power_warning(preflight["logger_read_once"])
        self.preflight = preflight
        json_dump(preflight, self.run_dir / "preflight_report.json")
        self.event(
            "PREFLIGHT_END",
            {
                "power_mode_looks_15w": preflight["power_mode_looks_15w"],
                "logger_ok": preflight["logger_read_once"].get("ok"),
                "logger_zero_power_warning": preflight.get("logger_zero_power_warning"),
            },
        )
        return preflight

    def preflight_fatal_reasons(self) -> List[str]:
        reasons = list(self.preflight.get("path_validation_errors") or [])
        if self.preflight.get("set_nvpmodel_error"):
            reasons.append(str(self.preflight["set_nvpmodel_error"]))
        if self.preflight.get("set_nvpmodel", {}).get("ok") is False:
            reasons.append("requested --set-nvpmodel failed")
        if not self.preflight.get("power_mode_looks_15w") and not self.args.allow_power_mode_mismatch:
            reasons.append("current nvpmodel output does not appear to be 15W; rerun in 15W mode or pass --allow-power-mode-mismatch intentionally")
        return reasons

    def check_external_logger_active(self) -> None:
        if self.args.allow_inactive_logger:
            self.event("LOGGER_ACTIVE_CHECK_SKIPPED", {"reason": "--allow-inactive-logger"})
            return
        logger_url = str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")).rstrip("/")
        health = http_get_json(logger_url + "/health", timeout=5)
        ok, reason = logger_health_active_for_run(health, self.run_id)
        self.event("LOGGER_ACTIVE_CHECK", {"ok": ok, "reason": reason, "health": health})
        if not ok:
            raise RuntimeError(
                "external logger is not actively recording this run. "
                f"{reason}. Start Pc_package recording with run_id={self.run_id}, then press ENTER."
            )
        self.log(f"[LOGGER] active recording confirmed for run_id={self.run_id}")

    def start_external_logger(self) -> None:
        if self.logger_control == "none":
            self.event("EXTERNAL_LOGGER_START_SKIPPED", {"logger_control": self.logger_control})
            return
        if self.logger_control == "manual":
            self.log("")
            self.log(f"[LOGGER] Start Pc_package manually with this same run_id: {self.run_id}")
            self.log("[LOGGER] This script will not start or stop the external logger in manual mode.")
            self.check_external_logger_active()
            return
        if self.logger_control != "auto":
            raise RuntimeError(f"Unsupported logger control mode: {self.logger_control}")

        logger_url = str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")).rstrip("/")
        scheduled_start_ns = time.time_ns() + 2_000_000_000
        payload = {
            "run_id": self.run_id,
            "sample_hz": float(self.args.external_sample_hz),
            "start_wallclock_ns": scheduled_start_ns,
            "fast_temp_sensor": True,
            "device_key": "jetson-agx",
            "board_name": self.config.get("board_display_name", "NVIDIA Jetson AGX Orin 64GB Developer Kit"),
            "notes": {
                "device_key": "jetson-agx",
                "board_name": self.config.get("board_display_name", "NVIDIA Jetson AGX Orin 64GB Developer Kit"),
                "run_label": self.run_id,
                "config_name": self.selected_name,
                "task": self.selected_cfg.get("task"),
                "model": self.selected_cfg.get("model"),
                "precision": self.precision,
                "input_size": self.input_size,
                "batch_size": self.batch_size,
                "power_source": self.config.get("power_source"),
                "runner": "battery_inference_agx",
            },
        }
        self.event("EXTERNAL_LOGGER_START_REQUEST", {"url": logger_url + "/start", "payload": payload})
        start_info = http_post_json(logger_url + "/start", payload, timeout=20)
        self.event("EXTERNAL_LOGGER_START_RESPONSE", start_info)
        if not start_info.get("ok"):
            raise RuntimeError(f"could not start Pc_package logger automatically: {start_info}")
        self.external_logger_started = True
        self.log(f"[LOGGER] Pc_package recording started automatically: run_id={self.run_id}")
        wait_s = max(0.0, (scheduled_start_ns - time.time_ns()) / 1e9 + 0.25)
        if wait_s > 0:
            self.log(f"[LOGGER] waiting {wait_s:.2f}s for scheduled external logger sampling start...")
            time.sleep(wait_s)
        self.check_external_logger_active()

    def stop_external_logger(self) -> None:
        if self.logger_control != "auto" or not self.external_logger_started:
            return
        logger_url = str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")).rstrip("/")
        payload = {"run_id": self.run_id, "device_key": "jetson-agx"}
        stop_info = http_post_json(logger_url + "/stop", payload, timeout=60)
        self.external_logger_stop_info = stop_info
        self.event("EXTERNAL_LOGGER_STOP_RESPONSE", stop_info)
        ext_dir = self.run_dir / "external_logger"
        archive_path = ext_dir / f"{self.run_id}.tar.gz"
        download_info = http_download(logger_url + f"/download/{self.run_id}", archive_path, timeout=600)
        self.event("EXTERNAL_LOGGER_DOWNLOAD", download_info)
        if download_info.get("ok"):
            self.external_logger_archive_path = archive_path
            self.log(f"[LOGGER] external archive saved: {archive_path}")
        else:
            self.log(f"[WARN] external logger archive download failed: {download_info}")
        self.external_logger_started = False

    def start_sudo_keepalive(self) -> None:
        if not self.args.sudo_prompt_once:
            return
        if os.geteuid() == 0:
            return
        if shutil.which("sudo") is None:
            self.log("[WARN] sudo not found; skipping sudo keepalive.")
            return
        self.log("[SUDO] Enter your password once now; the runner will keep sudo alive until it exits.")
        cp = subprocess.run(["sudo", "-v"], check=False)
        if cp.returncode != 0:
            raise RuntimeError("sudo authentication failed; cannot guarantee noninteractive NoiseGuard/preflight commands.")
        self.sudo_keepalive = subprocess.Popen(
            ["bash", "-c", "while true; do sudo -n -v >/dev/null 2>&1 || exit 1; sleep 60; done"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        self.event("SUDO_KEEPALIVE_START", {"pid": self.sudo_keepalive.pid})

    def stop_sudo_keepalive(self) -> None:
        proc = self.sudo_keepalive
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.event("SUDO_KEEPALIVE_STOP", {"returncode": proc.poll()})
        self.sudo_keepalive = None

    def build_child_command(self, loop_index: int, chunk_images: int, chunk_seconds: float) -> List[str]:
        child_name = f"{self.run_id}_{self.selected_name}_loop{loop_index:05d}"
        task = str(self.selected_cfg.get("task", "")).lower()
        command = [
            sys.executable,
            str(self.entrypoint),
            "--model",
            str(self.engine_path),
            "--dataset-root",
            str(self.dataset_path),
            "--split-name",
            str(self.selected_cfg.get("split", "test")),
            "--run-name",
            child_name,
            "--board-name",
            str(self.config.get("board_display_name", "NVIDIA Jetson AGX Orin 64GB Developer Kit")),
            "--precision-label",
            self.precision.lower(),
            "--run-variant",
            "battery_inference_chunk",
            "--output-root",
            str(self.child_output_root),
            "--imgsz",
            str(self.input_size),
            "--batch",
            str(self.batch_size),
            "--repeats",
            "1",
            "--warmup-images",
            "0",
            "--warmup-seconds",
            "0",
            "--min-images",
            str(max(1, int(chunk_images))),
            "--min-duration-s",
            str(max(0, int(round(chunk_seconds)))),
            "--skip-accuracy-pass",
            "--save-per-image-outputs",
            "--no-noise-guard",
            "--start-delay-s",
            "0",
        ]
        if not self.args.save_predictions:
            command.append("--no-save-predictions")
        extra = self.selected_cfg.get("extra_args") or {}
        if isinstance(extra, dict):
            if "conf" in extra and task == "detection":
                command += ["--conf", str(extra["conf"])]
            if "iou" in extra and task == "detection":
                command += ["--iou", str(extra["iou"])]
            if "max_det" in extra and task == "detection":
                command += ["--max-det", str(extra["max_det"])]
        return command

    def import_noise_guard(self) -> Tuple[Any, Any]:
        repo_root = self.entrypoint.parent.parent
        sys.path.insert(0, str(repo_root))
        try:
            from jetson.noise_guard import NoiseGuard, NoiseGuardConfig  # type: ignore

            return NoiseGuard, NoiseGuardConfig
        finally:
            try:
                sys.path.remove(str(repo_root))
            except ValueError:
                pass

    def enter_noise_guard(self) -> Optional[Any]:
        NoiseGuard, NoiseGuardConfig = self.import_noise_guard()
        guard = NoiseGuard(
            NoiseGuardConfig(
                enabled=True,
                profile="safe",
                preserve_url=str(self.config.get("external_logger_url", "http://${LOGGER_DASHBOARD_HOST}:8000")),
                display_off=True,
                stop_services=True,
                disable_wifi=True,
                disable_bluetooth=True,
            )
        )
        summary = guard.enter()
        json_dump(summary, self.run_dir / "noise_guard_summary.json")
        return guard

    def append_child_latency(self, loop_index: int, child_root: Path) -> int:
        timing = child_root / "benchmark_runs" / "repeat_01" / "per_image_latency.csv"
        if not timing.exists():
            matches = list(child_root.rglob("per_image_latency.csv"))
            if matches:
                timing = matches[0]
        if not timing.exists():
            raise RuntimeError(f"Child run did not produce per_image_latency.csv under {child_root}")
        rows_written = 0
        last_image = None
        last_latency = None
        with timing.open("r", encoding="utf-8", newline="") as src, self.latency_path.open("a", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=LATENCY_FIELDS)
            for row in reader:
                try:
                    latency_ms = float(row.get("latency_ms", "nan"))
                except ValueError:
                    latency_ms = float("nan")
                if latency_ms != latency_ms:
                    continue
                self.latencies.append(latency_ms)
                self.total_images += 1
                rows_written += 1
                last_image = row.get("image_path") or row.get("image_name") or ""
                last_latency = latency_ms
                monotonic_ns = row.get("monotonic_ns_end") or row.get("monotonic_ns_start")
                timestamp_mono = ""
                if monotonic_ns:
                    try:
                        timestamp_mono = f"{float(monotonic_ns) / 1e9:.9f}"
                    except ValueError:
                        timestamp_mono = f"{time.monotonic():.9f}"
                writer.writerow(
                    {
                        "timestamp_wall": row.get("timestamp_utc") or utc_now(),
                        "timestamp_monotonic": timestamp_mono or f"{time.monotonic():.9f}",
                        "run_id": self.run_id,
                        "board": self.board,
                        "task": self.selected_cfg.get("task"),
                        "model": self.selected_cfg.get("model"),
                        "precision": self.precision,
                        "input_size": self.input_size,
                        "batch_size": self.batch_size,
                        "loop_index": loop_index,
                        "image_index": self.total_images,
                        "image_id_or_path": last_image or "",
                        "latency_ms": f"{latency_ms:.6f}",
                        "success": "true",
                        "error_message": "",
                    }
                )
        if rows_written:
            self.update_dashboard_status(
                stage="benchmark",
                state="running",
                message="Battery inference running",
                repeat_name=f"{self.selected_name}_loop{loop_index:05d}",
                loop_index=loop_index,
                current_image_name=last_image,
                latency_ms_last=last_latency,
            )
        return rows_written

    def find_child_root(self, child_name: str) -> Path:
        direct = self.child_output_root / child_name
        if direct.exists():
            return direct
        if (direct / "FINAL_SUMMARY.json").exists():
            return direct
        matches = [p.parent for p in self.child_output_root.rglob("FINAL_SUMMARY.json") if child_name in str(p.parent)]
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
        raise RuntimeError(f"Could not find child FINAL_SUMMARY.json for {child_name}")

    def emit_status(self, loop_index: int, recent_start_images: int, recent_start_mono: float, child_running: bool = False) -> None:
        elapsed = max(0.0, time.monotonic() - (self.inference_start_mono or time.monotonic()))
        recent_elapsed = max(0.001, time.monotonic() - recent_start_mono)
        recent_images = max(0, self.total_images - recent_start_images)
        recent_fps = recent_images / recent_elapsed if recent_images else 0.0
        st = stats_for(self.latencies, elapsed, self.total_images)
        payload = {
            "elapsed_seconds": elapsed,
            "total_images": self.total_images,
            "overall_fps": st["fps"],
            "recent_fps": recent_fps,
            "median_latency_ms": st["median_latency_ms"],
            "p95_latency_ms": st["p95_latency_ms"],
            "config_name": self.selected_name,
            "run_id": self.run_id,
            "loop_index": loop_index,
            "child_running": child_running,
        }
        self.event("STATUS", payload)
        self.update_dashboard_status(
            stage="benchmark",
            state="running",
            message="Battery inference running",
            repeat_name=f"{self.selected_name}_loop{loop_index:05d}",
            loop_index=loop_index,
            processed_images=self.total_images,
            latency_ms_last=self.latencies[-1] if self.latencies else None,
        )
        self.log(
            "[STATUS] "
            f"run_id={self.run_id} config={self.selected_name} elapsed={elapsed:.1f}s "
            f"images={self.total_images} fps={st['fps']:.3f} recent_fps={recent_fps:.3f} "
            f"median_ms={st['median_latency_ms']} p95_ms={st['p95_latency_ms']}"
        )

    def run_child_chunk(self, loop_index: int, chunk_images: int, chunk_seconds: float) -> int:
        cmd = self.build_child_command(loop_index, chunk_images, chunk_seconds)
        self.last_child_command = cmd
        child_name = f"{self.run_id}_{self.selected_name}_loop{loop_index:05d}"
        self.event("LOOP_START", {"loop_index": loop_index, "child_run_name": child_name, "chunk_images": chunk_images, "chunk_seconds": chunk_seconds})
        self.update_dashboard_status(
            stage="benchmark",
            state="running",
            message="Battery inference running",
            reset_stage_timer=(loop_index == 1),
            repeat_name=f"{self.selected_name}_loop{loop_index:05d}",
            loop_index=loop_index,
            processed_images=self.total_images,
        )
        stdout = self.stdout_path.open("a", encoding="utf-8")
        stderr = self.stderr_path.open("a", encoding="utf-8")
        try:
            self.current_child = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                env=self.env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            last_status = time.monotonic()
            recent_start_images = self.total_images
            recent_start_mono = last_status
            while self.current_child.poll() is None:
                if self.stop_requested:
                    self.terminate_child()
                    break
                now = time.monotonic()
                if now - last_status >= float(self.args.status_interval):
                    self.emit_status(loop_index, recent_start_images, recent_start_mono, child_running=True)
                    last_status = now
                time.sleep(0.5)
            rc = self.current_child.wait()
            if self.stop_requested:
                try:
                    child_root = self.find_child_root(child_name)
                    rows = self.append_child_latency(loop_index, child_root)
                    if rows:
                        self.emit_status(loop_index, recent_start_images, recent_start_mono, child_running=False)
                    return rows
                except Exception as exc:
                    self.event("ERROR", {"partial_child_append_error": str(exc), "child_run_name": child_name})
                    return 0
            if rc != 0:
                raise RuntimeError(f"Child inference command failed with return code {rc}. See {self.stdout_path} and {self.stderr_path}")
            child_root = self.find_child_root(child_name)
            rows = self.append_child_latency(loop_index, child_root)
            self.emit_status(loop_index, recent_start_images, recent_start_mono, child_running=False)
            return rows
        finally:
            self.current_child = None
            stdout.close()
            stderr.close()

    def terminate_child(self) -> None:
        child = self.current_child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGINT)
        except Exception:
            child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except Exception:
                child.kill()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()

    def should_stop_after_chunk(self) -> bool:
        if self.stop_requested:
            return True
        elapsed = time.monotonic() - (self.inference_start_mono or time.monotonic())
        if self.args.mode == "smoke":
            target_images = max(500, int(self.args.max_images or 0))
            target_seconds = max(180.0, float(self.args.max_seconds or 0.0))
            return self.total_images >= target_images and elapsed >= target_seconds
        if self.args.max_images is not None and self.total_images >= int(self.args.max_images):
            self.stopped_reason = "max_images_reached"
            return True
        if self.args.max_seconds is not None and elapsed >= float(self.args.max_seconds):
            self.stopped_reason = "max_seconds_reached"
            return True
        return False

    def next_chunk_limits(self) -> Tuple[int, float]:
        if self.args.mode == "smoke":
            return 1, min(60.0, max(1.0, float(self.args.status_interval)))
        if self.args.max_images is not None:
            remaining = max(1, int(self.args.max_images) - self.total_images)
            return remaining, 0.0
        return 1, min(60.0, max(1.0, float(self.args.status_interval)))

    def signal_ctrl_c(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.stop_requested = True
        self.stopped_reason = "manual_ctrl_c_power_bank_80_percent_used_20_percent_remaining"
        self.event("CTRL_C_RECEIVED")
        self.log("\n[INFO] Ctrl-C received. Stopping inference cleanly and writing partial results.")
        self.terminate_child()

    def write_final_summary(self) -> None:
        end_time = utc_now()
        elapsed = 0.0
        if self.inference_start_mono is not None:
            elapsed = max(0.0, time.monotonic() - self.inference_start_mono)
        st = stats_for(self.latencies, elapsed, self.total_images)
        summary = {
            "run_id": self.run_id,
            "board": self.board,
            "board_display_name": self.config.get("board_display_name"),
            "task": self.selected_cfg.get("task"),
            "model": self.selected_cfg.get("model"),
            "precision": self.precision,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "start_time": self.inference_start_wall,
            "end_time": end_time,
            "stopped_reason": self.stopped_reason,
            "total_images": self.total_images,
            "elapsed_seconds": elapsed,
            "fps": st["fps"],
            "median_latency_ms": st["median_latency_ms"],
            "p95_latency_ms": st["p95_latency_ms"],
            "min_latency_ms": st["min_latency_ms"],
            "max_latency_ms": st["max_latency_ms"],
            "nvpmodel_output": {
                "nvpmodel_q": self.preflight.get("nvpmodel_q", {}),
                "nvpmodel_q_verbose": self.preflight.get("nvpmodel_q_verbose", {}),
            },
            "device_tree_model": self.preflight.get("device_tree_model"),
            "jetpack_l4t_version": self.preflight.get("l4t_release"),
            "cuda_version": self.preflight.get("cuda_version"),
            "tensorrt_version": self.preflight.get("tensorrt_version"),
            "git_commit": self.preflight.get("git_commit"),
            "engine_path": str(self.engine_path),
            "dataset_path": str(self.dataset_path),
            "inference_command_used": self.last_child_command,
            "external_logger_note": "External power logging is controlled according to external_logger_control; default auto starts/stops Pc_package by run_id.",
            "external_logger_control": self.logger_control,
            "external_logger_stop_info": self.external_logger_stop_info,
            "external_logger_archive_path": str(self.external_logger_archive_path) if self.external_logger_archive_path else None,
            "power_source_note": "This run is powered by UGREEN 25,000mAh 145W USB-C PD power bank.",
            "run_dir": str(self.run_dir),
            "config_name": self.selected_name,
        }
        json_dump(summary, self.run_dir / "summary.json")
        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": self.run_id,
                    "board": self.board,
                    "task": self.selected_cfg.get("task"),
                    "model": self.selected_cfg.get("model"),
                    "precision": self.precision,
                    "input_size": self.input_size,
                    "batch_size": self.batch_size,
                    "total_images": self.total_images,
                    "total_elapsed_seconds": elapsed,
                    "fps": st["fps"],
                    "median_latency_ms": st["median_latency_ms"],
                    "p95_latency_ms": st["p95_latency_ms"],
                    "min_latency_ms": st["min_latency_ms"],
                    "max_latency_ms": st["max_latency_ms"],
                    "stopped_reason": self.stopped_reason,
                }
            )
        md = [
            f"# AGX Battery Inference Run: {self.run_id}",
            "",
            f"- Config: `{self.selected_name}`",
            f"- Model: `{self.selected_cfg.get('model')}`",
            f"- Task: `{self.selected_cfg.get('task')}`",
            f"- Precision: `{self.precision}`",
            f"- Input size: `{self.input_size}`",
            f"- Batch size: `{self.batch_size}`",
            f"- Stopped reason: `{self.stopped_reason}`",
            f"- Total images: `{self.total_images}`",
            f"- Elapsed seconds: `{elapsed:.3f}`",
            f"- FPS: `{st['fps']:.6f}`",
            f"- Median latency ms: `{st['median_latency_ms']}`",
            f"- P95 latency ms: `{st['p95_latency_ms']}`",
            "",
            f"External power logging control: `{self.logger_control}`.",
            f"External logger archive: `{self.external_logger_archive_path}`.",
            "No J/image calculation is done here.",
        ]
        (self.run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        self.event("SUMMARY_WRITTEN", {"summary_json": str(self.run_dir / "summary.json")})

    def run(self) -> int:
        stdout_f, stderr_f = self.setup_files()
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.signal_ctrl_c)
        guard = None
        exit_code = 0
        try:
            self.log(f"[RUN] run_id: {self.run_id}")
            self.log(f"[RUN] run_dir: {self.run_dir}")
            self.log(f"[RUN] config: {self.selected_name}")
            self.write_initial_metadata()
            if not self.args.dry_run:
                self.start_sudo_keepalive()
            self.update_dashboard_status(stage="preflight", state="running", message="Battery inference preflight", reset_stage_timer=True)
            preflight = self.collect_preflight()
            self.update_dashboard_status(stage="preflight", state="finished", message="Battery inference preflight complete")
            fatal = self.preflight_fatal_reasons()
            if preflight.get("logger_read_once", {}).get("ok"):
                data = preflight["logger_read_once"].get("data")
                self.log(f"[PREFLIGHT] logger read_once ok: {data}")
            else:
                self.log(f"[WARN] logger read_once failed: {preflight.get('logger_read_once')}")
            if preflight.get("logger_zero_power_warning"):
                self.log(f"[WARN] {preflight['logger_zero_power_warning']}")
            if not preflight.get("power_mode_looks_15w"):
                self.log("[WARN] nvpmodel output does not appear to be 15W.")
            if fatal:
                self.stopped_reason = "preflight_failed: " + "; ".join(fatal)
                self.event("ERROR", {"fatal_reasons": fatal})
                self.write_final_summary()
                return 2
            if self.args.dry_run:
                preview_cmd = self.build_child_command(loop_index=1, chunk_images=1, chunk_seconds=1)
                self.last_child_command = preview_cmd
                self.log("[DRY-RUN] No inference command was executed.")
                self.log("[DRY-RUN] Child command preview:")
                self.log(" ".join(preview_cmd))
                self.stopped_reason = "dry_run_no_inference"
                self.write_final_summary()
                self.event("RUN_COMPLETE", {"stopped_reason": self.stopped_reason})
                return 0

            self.event("WAITING_FOR_EXTERNAL_LOGGER", {"logger_url": self.config.get("external_logger_url"), "run_id": self.run_id, "logger_control": self.logger_control})
            if self.args.wait_for_enter:
                if self.logger_control == "auto":
                    input("[LOGGER] Press ENTER when the logger server/dashboard is ready; this runner will start recording automatically... ")
                elif self.logger_control == "none":
                    input("[LOGGER] Press ENTER to start inference without external logger control... ")
                else:
                    input("[LOGGER] Press ENTER only after the external logger is recording... ")
            self.event("USER_CONFIRMED_LOGGER_STARTED", {"logger_control": self.logger_control})
            self.start_external_logger()
            self.update_dashboard_status(
                stage="logger_starting",
                state="running",
                message="External logger active; preparing Jetson telemetry",
                reset_stage_timer=True,
            )
            self.start_dashboard_status_pusher()
            self.push_dashboard_status_once()

            guard = self.enter_noise_guard()
            self.noise_guard = guard
            self.event("INFERENCE_START")
            self.inference_start_mono = time.monotonic()
            self.inference_start_wall = utc_now()
            self.stopped_reason = "completed"
            self.update_dashboard_status(
                stage="benchmark",
                state="running",
                message="Battery inference running",
                reset_stage_timer=True,
                repeat_name=f"{self.selected_name}_loop00001",
                loop_index=1,
                processed_images=0,
            )
            loop_index = 1
            while True:
                chunk_images, chunk_seconds = self.next_chunk_limits()
                self.run_child_chunk(loop_index, chunk_images, chunk_seconds)
                if self.should_stop_after_chunk():
                    if self.stopped_reason == "completed" and self.args.mode == "smoke":
                        self.stopped_reason = "smoke_target_reached"
                    break
                loop_index += 1
        except KeyboardInterrupt:
            self.stop_requested = True
            self.stopped_reason = "manual_ctrl_c_power_bank_80_percent_used_20_percent_remaining"
            self.event("CTRL_C_RECEIVED")
            exit_code = 0
        except Exception as exc:
            self.stopped_reason = f"fatal_error: {exc}"
            self.event("ERROR", {"error": str(exc)})
            print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
            exit_code = 1
        finally:
            self.terminate_child()
            self.update_dashboard_status(stage="stopping", state="running", message="Stopping battery inference and external logger", reset_stage_timer=True)
            if self.dashboard_status_thread is not None:
                self.push_dashboard_status_once()
            self.stop_external_logger()
            if guard is not None:
                try:
                    json_dump(guard.restore(), self.run_dir / "noise_guard_summary.json")
                    self.noise_guard = None
                except Exception as exc:
                    self.event("ERROR", {"noise_guard_restore_error": str(exc)})
            self.stop_sudo_keepalive()
            self.event("INFERENCE_END", {"stopped_reason": self.stopped_reason, "total_images": self.total_images})
            self.write_final_summary()
            self.update_dashboard_status(
                stage="finished" if exit_code == 0 else "error",
                state="finished" if exit_code == 0 else "error",
                message=f"Battery inference ended: {self.stopped_reason}",
                reset_stage_timer=True,
                processed_images=self.total_images,
            )
            if self.dashboard_status_thread is not None:
                self.push_dashboard_status_once()
                self.stop_dashboard_status_pusher()
            self.event("RUN_COMPLETE", {"stopped_reason": self.stopped_reason, "total_images": self.total_images})
            signal.signal(signal.SIGINT, original_sigint)
            stdout_f.flush()
            stderr_f.flush()
        return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference-only AGX battery endurance runner.")
    parser.add_argument("--config", default="config/battery_inference_agx.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-name", choices=["yolo11s_int8_640", "segformer_b0_int8_640"], default=None)
    parser.add_argument("--model", choices=["yolo11s", "segformer_b0"], default=None)
    parser.add_argument("--task", choices=["detection", "segmentation"], default=None)
    parser.add_argument("--precision", choices=["INT8", "FP16"], default="INT8")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=["endurance", "smoke"], default="endurance")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--status-interval", type=float, default=60.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set-nvpmodel", action="store_true")
    parser.add_argument("--allow-power-mode-mismatch", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--allow-inactive-logger", action="store_true")
    parser.add_argument("--logger-control", choices=["auto", "manual", "none"], default=None)
    parser.add_argument("--external-sample-hz", type=float, default=10.0)
    parser.add_argument("--dashboard-status-interval", type=float, default=5.0)
    parser.add_argument("--wait-for-enter", dest="wait_for_enter", action="store_true", default=True)
    parser.add_argument("--no-wait-for-enter", dest="wait_for_enter", action="store_false")
    parser.add_argument("--sudo-prompt-once", dest="sudo_prompt_once", action="store_true", default=True)
    parser.add_argument("--no-sudo-prompt-once", dest="sudo_prompt_once", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return BatteryInferenceRunner(args).run()
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
