#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parents[1]

EVENT_FIELDS = ["timestamp_wall", "timestamp_monotonic", "run_id", "board", "event", "payload_json"]
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
ALARM_EVENT_FIELDS = [
    "timestamp_wall",
    "timestamp_monotonic",
    "run_id",
    "source",
    "alarm_type",
    "severity",
    "metric",
    "value",
    "threshold",
    "message",
    "details_json",
]
ALARM_HIT_FIELDS = [
    "timestamp_wall",
    "source",
    "alarm_type",
    "severity",
    "metric",
    "value",
    "threshold",
    "details_json",
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

TEMP_PATTERN = re.compile(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
RAM_PATTERN = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_PATTERN = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
GR3D_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%@?(\d+)?")
EMC_PATTERN = re.compile(r"EMC_FREQ\s+(\d+)%@?(\d+)?")
LOGGER_RAW_PATTERN = re.compile(
    r"INA260:\s*([-+]?\d+(?:\.\d+)?)\s*mA,\s*([-+]?\d+(?:\.\d+)?)\s*V,\s*([-+]?\d+(?:\.\d+)?)\s*mW",
    flags=re.IGNORECASE,
)
FAULT_TEXT_PATTERN = re.compile(r"\bOC\b|over[-_ ]?current|over[-_ ]?temp|thermal|throttl|hot", flags=re.IGNORECASE)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def resolve_path(value: str | Path, base: Path = SCRIPT_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse to an object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_power_budget_w(label: Any) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*w", str(label or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def power_mode_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    label = cfg.get("target_power_mode_label")
    budget = cfg.get("observed_power_budget_w", cfg.get("target_power_budget_w"))
    if budget is None:
        budget = parse_power_budget_w(label)
    matched_label = cfg.get("matched_budget_label") or cfg.get("comparison_view")
    if not matched_label:
        if budget == 15:
            matched_label = "matched15w"
        elif budget == 25:
            matched_label = "native25w"
    return {
        "power_mode_label": label,
        "matched_budget_label": matched_label,
        "observed_power_budget_w": budget,
    }


def alarm_thresholds(cfg: Dict[str, Any]) -> Dict[str, float]:
    raw = cfg.get("alarm_thresholds") or {}
    defaults = {
        "internal_overtemperature_warning_c": 80.0,
        "internal_overtemperature_critical_c": 90.0,
        "external_overtemperature_warning_c": 60.0,
        "external_overtemperature_critical_c": 70.0,
        "external_overcurrent_warning_a": 1.25,
        "external_overcurrent_critical_a": 1.50,
        "external_power_warning_w": 25.0,
        "external_power_critical_w": 30.0,
    }
    out: Dict[str, float] = {}
    for key, default in defaults.items():
        try:
            out[key] = float(raw.get(key, default))
        except Exception:
            out[key] = float(default)
    return out


def threshold_severity(value: Any, warning: float, critical: float) -> Tuple[Optional[str], Optional[float]]:
    number = _maybe_float(value)
    if number is None:
        return None, None
    if number >= critical:
        return "critical", critical
    if number >= warning:
        return "warning", warning
    return None, None


def parse_logger_sensor_info(sensor_info: Dict[str, Any]) -> Dict[str, Optional[float]]:
    current_a = _maybe_float(sensor_info.get("current_a"))
    voltage_v = _maybe_float(sensor_info.get("bus_voltage_v") or sensor_info.get("voltage_v"))
    power_w = _maybe_float(sensor_info.get("power_w"))
    raw_current_ma = _maybe_float(sensor_info.get("raw_current_mA"))
    raw_power_mw = _maybe_float(sensor_info.get("raw_power_mW"))
    if current_a is None and raw_current_ma is not None:
        current_a = raw_current_ma / 1000.0
    if power_w is None and raw_power_mw is not None:
        power_w = raw_power_mw / 1000.0

    raw_line = str(sensor_info.get("last_raw_line") or "")
    match = LOGGER_RAW_PATTERN.search(raw_line)
    if match:
        if current_a is None:
            current_a = float(match.group(1)) / 1000.0
        if voltage_v is None:
            voltage_v = float(match.group(2))
        if power_w is None:
            power_w = float(match.group(3)) / 1000.0

    temp_c = _maybe_float(sensor_info.get("last_temp_c") or sensor_info.get("external_temp_c"))
    return {
        "current_a": current_a,
        "voltage_v": voltage_v,
        "power_w": power_w,
        "external_temp_c": temp_c,
    }


def run_capture(cmd: List[str], timeout: float = 20.0) -> Dict[str, Any]:
    started = utc_now_iso()
    try:
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "started": started,
            "returncode": cp.returncode,
            "stdout": cp.stdout or "",
            "stderr": cp.stderr or "",
            "ok": cp.returncode == 0,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "started": started,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }


def wallclock_ns() -> int:
    return time.time_ns()


def post_json(url: str, payload: Dict[str, Any], timeout: float = 4.0) -> Dict[str, Any]:
    data = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def http_download(url: str, dst: Path, timeout: float = 120.0) -> Dict[str, Any]:
    started = utc_now_iso()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            with dst.open("wb") as handle:
                shutil.copyfileobj(resp, handle)
        return {
            "ok": True,
            "url": url,
            "path": str(dst),
            "bytes": dst.stat().st_size,
            "started": started,
            "completed": utc_now_iso(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "path": str(dst),
            "bytes": dst.stat().st_size if dst.exists() else 0,
            "started": started,
            "completed": utc_now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def safe_extract_tar(tar_path: Path, dst: Path) -> Dict[str, Any]:
    started = utc_now_iso()
    try:
        dst.mkdir(parents=True, exist_ok=True)
        dst_resolved = dst.resolve()
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                target = (dst / member.name).resolve()
                if target != dst_resolved and dst_resolved not in target.parents:
                    raise RuntimeError(f"unsafe tar member path: {member.name}")
            try:
                tar.extractall(dst, filter="data")
            except TypeError:
                tar.extractall(dst)
        return {"ok": True, "tar_path": str(tar_path), "extract_dir": str(dst), "started": started, "completed": utc_now_iso()}
    except Exception as exc:
        return {
            "ok": False,
            "tar_path": str(tar_path),
            "extract_dir": str(dst),
            "started": started,
            "completed": utc_now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def ensure_sudo_noninteractive(args: argparse.Namespace) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "required": True,
        "running_as_root": os.geteuid() == 0,
        "sudo_path": shutil.which("sudo"),
        "password_once_requested": bool(args.sudo_password_once),
        "noninteractive_check_before_prompt": None,
        "sudo_v": None,
        "noninteractive_check": None,
    }
    if os.geteuid() == 0:
        status["required"] = False
        status["noninteractive_check"] = {"ok": True, "note": "already running as root"}
        return status
    if not status["sudo_path"]:
        raise RuntimeError("sudo command not found; cannot prepare noninteractive run.")

    status["noninteractive_check_before_prompt"] = run_capture(["sudo", "-n", "true"], timeout=15)
    if status["noninteractive_check_before_prompt"].get("ok"):
        status["noninteractive_check"] = status["noninteractive_check_before_prompt"]
        status["sudo_v"] = {"skipped": True, "reason": "sudo already noninteractive or cached"}
        return status

    if args.dry_run:
        status["required"] = False
        status["noninteractive_check"] = status["noninteractive_check_before_prompt"]
        status["sudo_v"] = {"skipped": True, "reason": "dry-run does not prompt for sudo"}
        return status

    if args.sudo_password_once:
        print("[SUDO] Aval run faghat yekbar password migire: sudo -v", flush=True)
        cp = subprocess.run(["sudo", "-v"], check=False)
        status["sudo_v"] = {"returncode": cp.returncode, "ok": cp.returncode == 0}
        if cp.returncode != 0:
            raise RuntimeError("sudo -v failed; run cannot continue noninteractive.")

    status["noninteractive_check"] = run_capture(["sudo", "-n", "true"], timeout=15)
    if not status["noninteractive_check"].get("ok"):
        raise RuntimeError(
            "sudo noninteractive check failed after setup. "
            "Run `sudo -v` once in this terminal or keep --sudo-password-once enabled."
        )
    return status


def safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except Exception:
        return None


class TeeStream:
    def __init__(self, original: Any, log_handle: Any):
        self.original = original
        self.log_handle = log_handle

    def write(self, text: str) -> int:
        self.original.write(text)
        self.original.flush()
        self.log_handle.write(text)
        self.log_handle.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.log_handle.flush()


class EventLogger:
    def __init__(self, path: Path, run_id: str, board: str):
        self.path = path
        self.run_id = run_id
        self.board = board
        self.lock = threading.Lock()
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=EVENT_FIELDS)
        self.writer.writeheader()
        self.handle.flush()

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            self.writer.writerow(
                {
                    "timestamp_wall": utc_now_iso(),
                    "timestamp_monotonic": f"{time.monotonic():.6f}",
                    "run_id": self.run_id,
                    "board": self.board,
                    "event": event,
                    "payload_json": json.dumps(payload or {}, sort_keys=True, default=str),
                }
            )
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class SudoKeepAlive:
    def __init__(self, interval_s: float):
        self.interval_s = max(float(interval_s), 10.0)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.refresh_count = 0
        self.failure_count = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.started_utc: Optional[str] = None
        self.stopped_utc: Optional[str] = None

    def start(self) -> None:
        if os.geteuid() == 0:
            return
        self.started_utc = utc_now_iso()
        self.thread = threading.Thread(target=self._loop, name="sudo_keepalive", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            result = run_capture(["sudo", "-n", "true"], timeout=15)
            self.last_result = result
            if result.get("ok"):
                self.refresh_count += 1
            else:
                self.failure_count += 1
                print(
                    "[WARN] sudo noninteractive health check failed; future sudo -n calls may fail. "
                    "Run should be stopped and restarted after fixing privileged-command policy coverage.",
                    file=sys.stderr,
                    flush=True,
                )

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.stopped_utc = utc_now_iso()
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.thread is not None,
            "interval_s": self.interval_s,
            "started_utc": self.started_utc,
            "stopped_utc": self.stopped_utc,
            "refresh_count": self.refresh_count,
            "failure_count": self.failure_count,
            "last_result": self.last_result,
        }


class FdTee:
    """Tee a process file descriptor, including native C/C++ library writes."""

    def __init__(self, fd: int, log_path: Path):
        self.fd = fd
        self.log_path = log_path
        self.saved_fd: Optional[int] = None
        self.log_fd: Optional[int] = None
        self.pipe_r: Optional[int] = None
        self.thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        self.saved_fd = os.dup(self.fd)
        self.log_fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        pipe_r, pipe_w = os.pipe()
        self.pipe_r = pipe_r
        os.dup2(pipe_w, self.fd)
        os.close(pipe_w)
        self.thread = threading.Thread(target=self._pump, name=f"fd_tee_{self.fd}", daemon=True)
        self.thread.start()
        self._started = True

    def _pump(self) -> None:
        assert self.pipe_r is not None
        assert self.saved_fd is not None
        assert self.log_fd is not None
        while True:
            try:
                data = os.read(self.pipe_r, 8192)
            except OSError:
                break
            if not data:
                break
            try:
                os.write(self.saved_fd, data)
            except OSError:
                pass
            try:
                os.write(self.log_fd, data)
            except OSError:
                pass

    def stop(self) -> None:
        if not self._started:
            return
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if self.saved_fd is not None:
            try:
                os.dup2(self.saved_fd, self.fd)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=5)
        for attr in ("pipe_r", "saved_fd", "log_fd"):
            value = getattr(self, attr)
            if value is not None:
                try:
                    os.close(value)
                except OSError:
                    pass
                setattr(self, attr, None)
        self._started = False


def _read_sys_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    temps = {match.group(1): float(match.group(2)) for match in TEMP_PATTERN.finditer(line)}
    out: Dict[str, Any] = {"raw": line.strip(), "temps": temps}
    ram = RAM_PATTERN.search(line)
    if ram:
        out["ram_used_mb"] = float(ram.group(1))
        out["ram_total_mb"] = float(ram.group(2))
    swap = SWAP_PATTERN.search(line)
    if swap:
        out["swap_used_mb"] = float(swap.group(1))
        out["swap_total_mb"] = float(swap.group(2))
    gr3d = GR3D_PATTERN.search(line)
    if gr3d:
        out["gr3d_pct"] = float(gr3d.group(1))
        if gr3d.group(2):
            out["gr3d_freq_mhz"] = float(gr3d.group(2))
    emc = EMC_PATTERN.search(line)
    if emc:
        out["emc_pct"] = float(emc.group(1))
        if emc.group(2):
            out["emc_freq_mhz"] = float(emc.group(2))
    return out


def _rail_label_key(label: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(label or "").upper()).strip("_")


def select_internal_total_power(rails: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    # NVIDIA documents VDD_IN as Total Module Power for Jetson Orin Nano/NX.
    for rail in rails:
        if _rail_label_key(rail.get("label")) == "VDD_IN" and rail.get("power_w") is not None:
            return _maybe_float(rail.get("power_w")), "vdd_in_total_module_power"

    ignored = {"", "NC", "SUMMATION", "SUM_OF_SHUNT_VOLTAGES"}
    total = 0.0
    count = 0
    for rail in rails:
        label_key = _rail_label_key(rail.get("label"))
        if label_key in ignored or label_key.startswith("INA3221_"):
            continue
        power_w = _maybe_float(rail.get("power_w"))
        if power_w is None:
            continue
        total += power_w
        count += 1
    return (total if count else None), ("sum_labeled_non_summation_rails" if count else "unavailable")


def collect_hwmon_summary() -> Dict[str, Any]:
    roots: List[Path] = []
    seen: set[str] = set()
    for pattern_root in (Path("/sys/bus/i2c/drivers/ina3221"), Path("/sys/class/hwmon")):
        if pattern_root == Path("/sys/bus/i2c/drivers/ina3221"):
            candidates = pattern_root.glob("*/hwmon/hwmon*")
        else:
            candidates = pattern_root.glob("hwmon*")
        for root in candidates:
            try:
                real = str(root.resolve())
            except Exception:
                continue
            if real not in seen:
                seen.add(real)
                roots.append(root)

    rails: List[Dict[str, Any]] = []
    for root in roots:
        name = _read_sys_text(root / "name") or ""
        labels: Dict[str, str] = {}
        voltages: Dict[str, str] = {}
        currents: Dict[str, str] = {}
        powers: Dict[str, str] = {}
        for file in root.glob("*"):
            fname = file.name
            if fname.startswith("in") and fname.endswith("_label"):
                labels[fname.split("_")[0][2:]] = _read_sys_text(file) or ""
            elif fname.startswith("in") and fname.endswith("_input"):
                voltages[fname.split("_")[0][2:]] = _read_sys_text(file) or ""
            elif fname.startswith("curr") and fname.endswith("_input"):
                currents[fname.split("_")[0][4:]] = _read_sys_text(file) or ""
            elif fname.startswith("power") and fname.endswith("_input"):
                powers[fname.split("_")[0][5:]] = _read_sys_text(file) or ""

        for cid in sorted(set(labels) | set(voltages) | set(currents) | set(powers)):
            voltage_mv = _maybe_float(voltages.get(cid))
            current_ma = _maybe_float(currents.get(cid))
            power_mw = _maybe_float(powers.get(cid))
            if power_mw is None and voltage_mv is not None and current_ma is not None:
                power_mw = (voltage_mv * current_ma) / 1000.0
            power_w = None if power_mw is None else power_mw / 1000.0
            if voltage_mv is None and current_ma is None and power_w is None:
                continue
            rails.append(
                {
                    "label": labels.get(cid) or f"{name}:{cid}",
                    "sensor_name": name,
                    "channel_id": cid,
                    "voltage_mV": voltage_mv,
                    "current_mA": current_ma,
                    "power_w": power_w,
                }
            )
    total_power_w, total_power_method = select_internal_total_power(rails)
    return {
        "timestamp_utc": utc_now_iso(),
        "total_power_w": total_power_w,
        "total_power_method": total_power_method,
        "rails": rails,
    }


def collect_thermal_summary() -> Dict[str, Any]:
    zones: List[Dict[str, Any]] = []
    max_temp: Optional[float] = None
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        raw = _read_sys_text(zone / "temp")
        temp_c = _maybe_float(raw)
        if temp_c is not None and abs(temp_c) > 1000:
            temp_c = temp_c / 1000.0
        name = _read_sys_text(zone / "type") or zone.name
        if temp_c is not None:
            max_temp = temp_c if max_temp is None else max(max_temp, temp_c)
        zones.append({"zone": zone.name, "type": name, "temp_c": temp_c})
    return {"timestamp_utc": utc_now_iso(), "max_temp_c": max_temp, "zones": zones}


class InternalTelemetry:
    def __init__(self, run_dir: Path, interval_s: float = 1.0):
        self.run_dir = run_dir
        self.interval_s = max(float(interval_s), 0.5)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.proc: Optional[subprocess.Popen[str]] = None
        self.lock = threading.Lock()
        self.latest: Dict[str, Any] = {"hwmon": {}, "thermal": {}, "tegrastats": {}, "process_memory": {}}
        self.sample_count = 0

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="internal_telemetry", daemon=True)
        self.thread.start()

    def _start_tegrastats(self) -> None:
        if shutil.which("tegrastats") is None:
            return
        interval_ms = int(self.interval_s * 1000)
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self.proc = None

    def _loop(self) -> None:
        self._start_tegrastats()
        csv_path = self.run_dir / "internal_telemetry.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "timestamp_wall",
                "sample_index",
                "internal_total_power_w",
                "internal_max_temp_c",
                "ram_used_mb",
                "ram_total_mb",
                "gr3d_pct",
                "emc_pct",
                "tegrastats_raw",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set():
                line = ""
                if self.proc is not None and self.proc.stdout is not None:
                    try:
                        line = self.proc.stdout.readline().strip()
                    except Exception:
                        line = ""
                else:
                    time.sleep(self.interval_s)
                tegra = parse_tegrastats_line(line) if line else {}
                hwmon = collect_hwmon_summary()
                thermal = collect_thermal_summary()
                process_memory = {"runner_pid": os.getpid()}
                snap = {"hwmon": hwmon, "thermal": thermal, "tegrastats": tegra, "process_memory": process_memory}
                with self.lock:
                    self.latest = snap
                    self.sample_count += 1
                    sample_idx = self.sample_count
                writer.writerow(
                    {
                        "timestamp_wall": utc_now_iso(),
                        "sample_index": sample_idx,
                        "internal_total_power_w": hwmon.get("total_power_w"),
                        "internal_max_temp_c": thermal.get("max_temp_c"),
                        "ram_used_mb": tegra.get("ram_used_mb"),
                        "ram_total_mb": tegra.get("ram_total_mb"),
                        "gr3d_pct": tegra.get("gr3d_pct"),
                        "emc_pct": tegra.get("emc_pct"),
                        "tegrastats_raw": line,
                    }
                )
                handle.flush()

    def get_latest(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.latest, default=str))

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.proc is not None:
            try:
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        return {"sample_count": self.sample_count, "latest": self.get_latest()}


class AlarmMonitor:
    def __init__(self, run_dir: Path, cfg: Dict[str, Any], run_id: str, telemetry: InternalTelemetry):
        self.run_dir = run_dir
        self.cfg = cfg
        self.run_id = run_id
        self.telemetry = telemetry
        self.enabled = bool(cfg.get("alarm_logging_enabled", True))
        self.thresholds = alarm_thresholds(cfg)
        self.interval_s = max(float(cfg.get("alarm_poll_interval_seconds") or 2.0), 1.0)
        self.repeat_interval_s = max(float(cfg.get("alarm_repeat_interval_seconds") or 60.0), 10.0)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.event_count = 0
        self.warning_count = 0
        self.critical_count = 0
        self.last_emit_by_key: Dict[Tuple[str, str, str], float] = {}
        self.active_states: Dict[Tuple[str, str], str] = {}
        self.started_utc: Optional[str] = None
        self.stopped_utc: Optional[str] = None
        self.csv_path = run_dir / "alarm_events.csv"

    def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=ALARM_EVENT_FIELDS).writeheader()
        if not self.enabled:
            return
        self.started_utc = utc_now_iso()
        self.thread = threading.Thread(target=self._loop, name="alarm_monitor", daemon=True)
        self.thread.start()

    def _emit(
        self,
        *,
        source: str,
        alarm_type: str,
        severity: str,
        metric: str,
        value: Any,
        threshold: Any,
        message: str,
        details: Dict[str, Any],
    ) -> None:
        now_mono = time.monotonic()
        key = (source, alarm_type, metric)
        state_key = (source, metric)
        last_severity = self.active_states.get(state_key)
        repeat_key = (source, alarm_type, severity)
        last_emit = self.last_emit_by_key.get(repeat_key, 0.0)
        if last_severity == severity and (now_mono - last_emit) < self.repeat_interval_s:
            return
        self.active_states[state_key] = severity
        self.last_emit_by_key[repeat_key] = now_mono

        row = {
            "timestamp_wall": utc_now_iso(),
            "timestamp_monotonic": f"{now_mono:.6f}",
            "run_id": self.run_id,
            "source": source,
            "alarm_type": alarm_type,
            "severity": severity,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "message": message,
            "details_json": json.dumps(details, sort_keys=True, default=str),
        }
        with self.lock:
            with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=ALARM_EVENT_FIELDS).writerow(row)
            self.event_count += 1
            if severity == "critical":
                self.critical_count += 1
            else:
                self.warning_count += 1
        print(
            f"[ALARM] severity={severity} source={source} metric={metric} "
            f"value={value} threshold={threshold} type={alarm_type} msg={message}",
            flush=True,
        )

    def _check_metric(
        self,
        *,
        source: str,
        metric: str,
        value: Any,
        warning_key: str,
        critical_key: str,
        alarm_type: str,
        details: Dict[str, Any],
    ) -> None:
        severity, threshold = threshold_severity(
            value,
            self.thresholds[warning_key],
            self.thresholds[critical_key],
        )
        if severity is None:
            self.active_states.pop((source, metric), None)
            return
        self._emit(
            source=source,
            alarm_type=alarm_type,
            severity=severity,
            metric=metric,
            value=value,
            threshold=threshold,
            message=f"{metric} crossed {severity} threshold",
            details=details,
        )

    def _poll_external_logger(self) -> Dict[str, Any]:
        url = str(self.cfg.get("external_logger_url") or "").rstrip("/")
        if not url:
            return {}
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        sensor = data.get("sensor_info") or {}
        parsed = parse_logger_sensor_info(sensor if isinstance(sensor, dict) else {})
        details = {"health_timestamp_utc": data.get("timestamp_utc"), "sensor_info": sensor}
        self._check_metric(
            source="external_logger_live",
            metric="current_a",
            value=parsed.get("current_a"),
            warning_key="external_overcurrent_warning_a",
            critical_key="external_overcurrent_critical_a",
            alarm_type="overcurrent",
            details=details,
        )
        self._check_metric(
            source="external_logger_live",
            metric="power_w",
            value=parsed.get("power_w"),
            warning_key="external_power_warning_w",
            critical_key="external_power_critical_w",
            alarm_type="external_high_power",
            details=details,
        )
        self._check_metric(
            source="external_logger_live",
            metric="external_temp_c",
            value=parsed.get("external_temp_c"),
            warning_key="external_overtemperature_warning_c",
            critical_key="external_overtemperature_critical_c",
            alarm_type="external_overtemperature",
            details=details,
        )
        return {"parsed": parsed, "health_ok": bool(data.get("ok", True))}

    def _poll_internal(self) -> Dict[str, Any]:
        snap = self.telemetry.get_latest()
        thermal = snap.get("thermal") or {}
        tegra = snap.get("tegrastats") or {}
        max_temp = thermal.get("max_temp_c")
        self._check_metric(
            source="jetson_internal",
            metric="internal_max_temp_c",
            value=max_temp,
            warning_key="internal_overtemperature_warning_c",
            critical_key="internal_overtemperature_critical_c",
            alarm_type="overtemperature",
            details=snap,
        )
        raw = str(tegra.get("raw") or "")
        if raw and FAULT_TEXT_PATTERN.search(raw):
            self._emit(
                source="tegrastats",
                alarm_type="tegrastats_fault_text",
                severity="warning",
                metric="tegrastats_raw",
                value=raw,
                threshold="OC/overcurrent/thermal/throttle text",
                message="tegrastats raw line contains fault/thermal/throttle text",
                details=snap,
            )
        return {"internal_max_temp_c": max_temp, "tegrastats_raw": raw}

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._poll_internal()
            self._poll_external_logger()

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.stopped_utc = utc_now_iso()
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "started_utc": self.started_utc,
            "stopped_utc": self.stopped_utc,
            "event_count": self.event_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "thresholds": self.thresholds,
            "events_csv": str(self.csv_path),
        }


def format_repeat_name(config_name: str, loop_index: Optional[int]) -> str:
    try:
        loop_number = int(loop_index or 0)
    except (TypeError, ValueError):
        loop_number = 0
    return f"{config_name}_loop{max(loop_number, 0):05d}"


class DashboardPusher:
    def __init__(
        self,
        cfg: Dict[str, Any],
        item: Dict[str, Any],
        config_name: str,
        run_id: str,
        notes: List[str],
        telemetry: Optional[InternalTelemetry] = None,
    ):
        self.cfg = cfg
        self.item = item
        self.config_name = config_name
        self.run_id = run_id
        self.notes = notes
        self.telemetry = telemetry
        self.enabled = bool(cfg.get("dashboard_status_enabled", True))
        logger_url = str(cfg.get("external_logger_url") or "").strip().rstrip("/")
        endpoint = str(cfg.get("dashboard_status_endpoint") or "/api/jetson_status").strip()
        if endpoint and not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self.url = (logger_url + endpoint) if logger_url else ""
        self.failed_once = False
        self.failure_count = 0
        self.success_count = 0
        self.created_mono = time.monotonic()

    def push(
        self,
        *,
        stage: str,
        state: str = "running",
        message: str = "",
        processed_images: Optional[int] = None,
        current_image_name: Optional[str] = None,
        repeat_name: Optional[str] = None,
        repeat_index: Optional[int] = None,
        latency_ms_last: Optional[float] = None,
        fps: Optional[float] = None,
        median_latency_ms: Optional[float] = None,
        p95_latency_ms: Optional[float] = None,
        stopped_reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not self.url:
            return
        now_ns = wallclock_ns()
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "device_key": self.cfg.get("dashboard_device_key") or "jetson-nano",
            "device_label": self.cfg.get("board_display_name") or "Jetson Orin Nano Super",
            "board_name": self.cfg.get("board") or "nano_super",
            "precision_label": self.item.get("precision"),
            "run_variant": self.cfg.get("run_variant") or "battery_inference_pure_engine",
            "config_name": self.config_name,
            "task": self.item.get("task"),
            "model": self.item.get("model"),
            "input_size": self.item.get("input_size"),
            "batch_size": self.item.get("batch_size"),
            "stage": stage,
            "state": state,
            "message": message,
            "repeat_name": repeat_name if repeat_name is not None else format_repeat_name(self.config_name, repeat_index or 0),
            "repeat_index": repeat_index,
            "timestamp_utc": utc_now_iso(),
            "wallclock_ns": now_ns,
            "overall_elapsed_s": time.monotonic() - self.created_mono,
            "processed_images": processed_images,
            "current_image_name": current_image_name,
            "latency_ms_last": latency_ms_last,
            "min_images_target": None,
            "min_duration_s_target": None,
            "fps": fps,
            "median_latency_ms": median_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "stopped_reason": stopped_reason,
            "power_mode": power_mode_payload(self.cfg),
            "internal_live": self.telemetry.get_latest() if self.telemetry is not None else {},
            "external_logger_control": self.cfg.get("external_logger_control", "manual"),
            "external_logger_url": self.cfg.get("external_logger_url"),
            "power_source": self.cfg.get("power_source"),
            "battery_inference": {
                "manual_stop_note": self.cfg.get("manual_stop_note"),
                "fixed_image_order": self.cfg.get("fixed_image_order"),
                "save_predictions_default": self.cfg.get("save_predictions_default"),
                "timing_boundary": self.cfg.get("timing_boundary") or "pure_engine_only",
                "measurement_backend": self.cfg.get("measurement_backend") or "trtexec_random_input",
            },
        }
        if extra:
            payload.update(extra)
        try:
            post_json(self.url, payload, timeout=4)
            self.success_count += 1
            self.failed_once = False
        except Exception as exc:
            self.failure_count += 1
            if not self.failed_once:
                msg = f"Dashboard status push failed once: {type(exc).__name__}: {exc}"
                self.notes.append(msg)
                print(f"[WARN] {msg}", file=sys.stderr, flush=True)
                self.failed_once = True

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.url,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


class ExternalLoggerController:
    def __init__(self, cfg: Dict[str, Any], item: Dict[str, Any], run_id: str, run_dir: Path, notes: List[str]):
        self.cfg = cfg
        self.item = item
        self.run_id = run_id
        self.run_dir = run_dir
        self.notes = notes
        self.control = str(cfg.get("external_logger_control") or "auto").lower()
        self.logger_url = str(cfg.get("external_logger_url") or "").rstrip("/")
        self.device_key = str(cfg.get("dashboard_device_key") or "jetson-nano")
        self.started = False
        self.start_info: Optional[Dict[str, Any]] = None
        self.stop_info: Optional[Dict[str, Any]] = None
        self.download_info: Optional[Dict[str, Any]] = None
        self.extract_info: Optional[Dict[str, Any]] = None
        self.start_wallclock_ns: Optional[int] = None

    def auto_enabled(self) -> bool:
        return self.control == "auto"

    def manual_enabled(self) -> bool:
        return self.control == "manual"

    def start(self) -> Dict[str, Any]:
        if not self.auto_enabled():
            self.start_info = {"ok": True, "skipped": True, "control": self.control}
            return self.start_info
        if not self.logger_url:
            raise RuntimeError("external_logger_control=auto but external_logger_url is empty")

        delay_s = float(self.cfg.get("external_logger_start_delay_seconds") or 2.0)
        self.start_wallclock_ns = wallclock_ns() + int(delay_s * 1e9)
        payload = {
            "run_id": self.run_id,
            "sample_hz": float(self.cfg.get("external_logger_sample_hz") or 10.0),
            "start_wallclock_ns": self.start_wallclock_ns,
            "fast_temp_sensor": bool(self.cfg.get("external_logger_fast_temp_sensor", True)),
            "notes": {
                "device_key": self.device_key,
                "device_label": self.cfg.get("board_display_name") or "Jetson Orin Nano Super",
                "board_name": self.cfg.get("board") or "nano_super",
                "run_id": self.run_id,
                "task": self.item.get("task"),
                "model": self.item.get("model"),
                "precision": self.item.get("precision"),
                "input_size": self.item.get("input_size"),
                "batch_size": self.item.get("batch_size"),
                "run_variant": self.cfg.get("run_variant") or "battery_inference_pure_engine",
                "timing_boundary": self.cfg.get("timing_boundary") or "pure_engine_only",
                "measurement_backend": self.cfg.get("measurement_backend") or "trtexec_random_input",
                "power_source": self.cfg.get("power_source"),
            },
        }
        self.start_info = post_json(self.logger_url + "/start", payload, timeout=20)
        self.started = bool(self.start_info.get("ok"))
        write_json(self.run_dir / "external_logger_start.json", {"payload": payload, "response": self.start_info})
        if not self.started:
            raise RuntimeError(f"External logger /start did not return ok: {self.start_info}")
        return self.start_info

    def wait_until_scheduled_start(self) -> None:
        if not self.auto_enabled() or not self.start_wallclock_ns:
            return
        while wallclock_ns() < self.start_wallclock_ns:
            time.sleep(0.02)

    def stop(self) -> Dict[str, Any]:
        if not self.auto_enabled():
            self.stop_info = {"ok": True, "skipped": True, "control": self.control}
            return self.stop_info
        if not self.started:
            self.stop_info = {"ok": True, "skipped": True, "reason": "logger_was_not_started"}
            return self.stop_info
        payload = {"run_id": self.run_id, "device_key": self.device_key, "board_name": self.cfg.get("board") or "nano_super"}
        try:
            self.stop_info = post_json(self.logger_url + "/stop", payload, timeout=60)
        except Exception as exc:
            self.stop_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "payload": payload}
            self.notes.append(f"External logger stop failed: {self.stop_info['error']}")
        write_json(self.run_dir / "external_logger_stop.json", {"payload": payload, "response": self.stop_info})
        if self.stop_info.get("ok"):
            archive_path = self.run_dir / "external_logger_archive.tar.gz"
            download_url = self.logger_url + f"/download/{self.run_id}"
            self.download_info = http_download(download_url, archive_path, timeout=120)
            write_json(self.run_dir / "external_logger_download.json", self.download_info)
            if self.download_info.get("ok"):
                self.extract_info = safe_extract_tar(archive_path, self.run_dir / "external_logger_archive")
                write_json(self.run_dir / "external_logger_extract.json", self.extract_info)
                if not self.extract_info.get("ok"):
                    self.notes.append(f"External logger archive extract failed: {self.extract_info.get('error')}")
            else:
                self.notes.append(f"External logger archive download failed: {self.download_info.get('error')}")
        self.started = False
        return self.stop_info

    def status(self) -> Dict[str, Any]:
        return {
            "control": self.control,
            "logger_url": self.logger_url,
            "device_key": self.device_key,
            "started": self.started,
            "start_info": self.start_info,
            "stop_info": self.stop_info,
            "download_info": self.download_info,
            "extract_info": self.extract_info,
        }


class ProgressTracker:
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.lock = threading.Lock()
        self.payload: Dict[str, Any] = {
            "stage": "created",
            "state": "running",
            "message": "Pure engine battery run created",
            "processed_images": 0,
            "repeat_name": format_repeat_name(config_name, 0),
            "repeat_index": 0,
        }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            self.payload.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.payload)


class DashboardHeartbeat:
    def __init__(self, dashboard: DashboardPusher, progress: ProgressTracker, interval_s: float = 2.0):
        self.dashboard = dashboard
        self.progress = progress
        self.interval_s = max(float(interval_s), 1.0)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="dashboard_heartbeat", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            snap = self.progress.snapshot()
            self.dashboard.push(
                stage=str(snap.get("stage") or "heartbeat"),
                state=str(snap.get("state") or "running"),
                message=str(snap.get("message") or "Battery inference heartbeat"),
                processed_images=snap.get("processed_images"),
                current_image_name=snap.get("current_image_name"),
                repeat_name=snap.get("repeat_name"),
                repeat_index=snap.get("repeat_index"),
                latency_ms_last=snap.get("latency_ms_last"),
                fps=snap.get("fps"),
                median_latency_ms=snap.get("median_latency_ms"),
                p95_latency_ms=snap.get("p95_latency_ms"),
                stopped_reason=snap.get("stopped_reason"),
                extra={"heartbeat": True},
            )

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return {"enabled": self.thread is not None, "interval_s": self.interval_s}


def init_latency_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=LATENCY_FIELDS).writeheader()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def latency_stats(path: Path) -> Dict[str, Any]:
    rows = []
    latencies = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                if str(row.get("success", "")).lower() == "true":
                    try:
                        latencies.append(float(row.get("latency_ms") or 0.0))
                    except ValueError:
                        pass
    return {
        "total_rows": len(rows),
        "total_images": len(latencies),
        "median_latency_ms": percentile(latencies, 50.0),
        "p95_latency_ms": percentile(latencies, 95.0),
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }


def select_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    configs = cfg.get("configs") or {}
    selected_names = cfg.get("selected_configs") or list(configs)
    if args.config_name:
        if args.config_name not in configs:
            raise KeyError(f"Unknown config-name: {args.config_name}")
        name = args.config_name
    else:
        matches = []
        for candidate in selected_names:
            item = configs.get(candidate) or {}
            if args.model and item.get("model") != args.model:
                continue
            if args.task and item.get("task") != args.task:
                continue
            if int(item.get("input_size", 0)) != int(args.input_size):
                continue
            matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f"Could not select exactly one config. Matches={matches}; pass --config-name.")
        name = matches[0]

    item = dict(configs[name])
    precision = str(args.precision or item.get("precision") or "INT8").upper()
    engine_paths = item.get("engine_paths") or {}
    if precision in engine_paths:
        item["engine_path"] = engine_paths[precision]
    elif precision != str(item.get("precision", "")).upper():
        raise FileNotFoundError(f"No engine path configured for precision {precision} in {name}")
    item["precision"] = precision
    item["input_size"] = int(args.input_size)
    item["batch_size"] = int(args.batch_size)
    return name, item


def list_detection_images(dataset_path: Path, split: str) -> List[Path]:
    root = dataset_path / "images" / split
    if not root.is_dir():
        root = dataset_path / "images"
    if not root.is_dir():
        root = dataset_path
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    return images


def resolve_trtexec_binary() -> str:
    local_trtexec = os.environ.get("LOCAL_TRTEXEC_PATH") or os.environ.get("TRTEXEC_PATH")
    local_lib_dir = os.environ.get("LOCAL_TENSORRT_LIB_DIR")
    if local_trtexec and Path(local_trtexec).is_file():
        if local_lib_dir:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            ld_parts = [part for part in current_ld.split(":") if part]
            if local_lib_dir not in ld_parts:
                os.environ["LD_LIBRARY_PATH"] = f"{local_lib_dir}:{current_ld}" if current_ld else local_lib_dir
        os.environ.setdefault("TRTEXEC_PATH", str(Path(local_trtexec).resolve()))
        return str(Path(local_trtexec).resolve())

    for candidate in (
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/samples/trtexec",
    ):
        if candidate and Path(candidate).is_file():
            os.environ.setdefault("TRTEXEC_PATH", str(Path(candidate).resolve()))
            return str(Path(candidate).resolve())
    raise FileNotFoundError("trtexec not found. Set TRTEXEC_PATH or install JetPack TensorRT binaries.")


def validate_selection(config_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    engine = resolve_path(item["engine_path"])
    dataset = resolve_path(item["dataset_path"])
    package_root = resolve_path(item["package_root"])
    entrypoint_raw = str(item.get("inference_entrypoint") or "trtexec")
    entrypoint = Path("")
    errors = []
    if not engine.is_file():
        errors.append(f"missing engine_path: {engine}")
    if not dataset.exists():
        errors.append(f"missing dataset_path: {dataset}")
    if not package_root.is_dir():
        errors.append(f"missing package_root: {package_root}")
    if entrypoint_raw == "trtexec":
        try:
            entrypoint = Path(resolve_trtexec_binary())
        except Exception as exc:
            errors.append(f"missing trtexec inference_entrypoint: {type(exc).__name__}: {exc}")
    else:
        entrypoint = resolve_path(entrypoint_raw)
        if not entrypoint.exists():
            errors.append(f"missing inference_entrypoint: {entrypoint}")
    if item.get("task") == "segmentation":
        manifest = resolve_path(item.get("manifest_path") or dataset / "manifests" / f"{item.get('split', 'test')}.csv")
        if not manifest.is_file():
            errors.append(f"missing segmentation manifest: {manifest}")
    else:
        image_count = len(list_detection_images(dataset, str(item.get("split") or "test"))) if dataset.exists() else 0
        if image_count <= 0:
            errors.append(f"no detection images found under: {dataset}")
    return {
        "config_name": config_name,
        "engine_path": str(engine),
        "dataset_path": str(dataset),
        "package_root": str(package_root),
        "inference_entrypoint": str(entrypoint),
        "measurement_backend": "trtexec_random_input",
        "timing_boundary": "pure_engine_only",
        "errors": errors,
        "ok": not errors,
    }


def run_preflight(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    target_label = str(cfg.get("target_power_mode_label") or "15W")
    report: Dict[str, Any] = {
        "target_power_mode_label": target_label,
        "set_nvpmodel_requested": bool(args.set_nvpmodel),
        "nvpmodel_set": None,
        "nvpmodel_q": None,
        "nvpmodel_q_verbose": None,
        "power_mode_ok": False,
        "power_mode_warning": None,
        "device_tree_model": safe_read_text(Path("/proc/device-tree/model")),
        "l4t_release": safe_read_text(Path("/etc/nv_tegra_release")),
        "os_release": safe_read_text(Path("/etc/os-release")),
        "versions": {},
        "git_commit": None,
    }

    nvpmodel_query_prefix = [] if args.dry_run else ["sudo", "-n"]

    if args.set_nvpmodel:
        mode_id_value = cfg.get("nvpmodel_mode_id")
        if mode_id_value is None:
            mode_key = "nvpmodel_25w_mode_id" if target_label.lower() == "25w" else "nvpmodel_15w_mode_id"
            mode_id_value = cfg.get(mode_key, cfg.get("nvpmodel_15w_mode_id"))
        mode_id = str(mode_id_value or "").strip()
        if not mode_id or mode_id.upper() == "TODO":
            report["nvpmodel_set"] = {
                "ok": False,
                "error": "nvpmodel_mode_id is missing/TODO in config; not hard-coding a mode ID.",
            }
        else:
            report["nvpmodel_set"] = run_capture(["sudo", "-n", "nvpmodel", "-m", mode_id], timeout=20)

    report["nvpmodel_q"] = run_capture([*nvpmodel_query_prefix, "nvpmodel", "-q"], timeout=20)
    report["nvpmodel_q_verbose"] = run_capture([*nvpmodel_query_prefix, "nvpmodel", "-q", "--verbose"], timeout=30)
    if args.dry_run:
        report["dry_run_nvpmodel_query_without_sudo"] = True
    combined = "\n".join(
        [
            report["nvpmodel_q"].get("stdout", ""),
            report["nvpmodel_q"].get("stderr", ""),
            report["nvpmodel_q_verbose"].get("stdout", ""),
            report["nvpmodel_q_verbose"].get("stderr", ""),
        ]
    )
    label = target_label.lower()
    report["power_mode_ok"] = label in combined.lower()
    if not report["power_mode_ok"]:
        report["power_mode_warning"] = (
            f"Current nvpmodel output did not clearly contain '{target_label}'. "
            "Pass --allow-power-mode-mismatch only if you intentionally want to continue."
        )

    report["versions"]["nvcc"] = run_capture(["bash", "-lc", "command -v nvcc >/dev/null && nvcc --version || true"], timeout=20)
    try:
        trtexec_path = resolve_trtexec_binary()
        report["versions"]["trtexec"] = run_capture([trtexec_path, "--version"], timeout=20)
        report["versions"]["trtexec"]["resolved_path"] = trtexec_path
    except Exception as exc:
        report["versions"]["trtexec"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["versions"]["python_tensorrt"] = run_capture(
        [sys.executable, "-c", "import tensorrt as trt; print(getattr(trt, '__version__', 'unknown'))"],
        timeout=20,
    )
    report["git_commit"] = run_capture(["git", "-C", str(SCRIPT_ROOT), "rev-parse", "HEAD"], timeout=10)
    return report


def write_external_logger_note(path: Path, run_id: str, cfg: Dict[str, Any]) -> None:
    control = str(cfg.get("external_logger_control") or "auto")
    if control == "auto":
        control_note = (
            "External logger control is automatic. The Jetson runner calls Pc_package /start with this run_id, "
            "then calls /stop during clean shutdown or Ctrl-C.\n"
        )
    else:
        control_note = (
            "External logger control is manual. Start Pc_package manually with this same run_id before pressing ENTER "
            "in the Jetson runner, then stop Pc_package manually after the Jetson process exits.\n"
        )
    text = (
        f"run_id: {run_id}\n"
        f"logger_url: {cfg.get('external_logger_url', 'http://${LOGGER_DASHBOARD_HOST}:8000')}\n"
        f"external_logger_control: {control}\n"
        f"device_key: {cfg.get('dashboard_device_key', 'jetson-nano')}\n"
        f"{control_note}"
    )
    path.write_text(text, encoding="utf-8")


def enter_noise_guard(
    item: Dict[str, Any],
    cfg: Dict[str, Any],
    run_dir: Path,
    *,
    enabled: bool = True,
) -> Tuple[Optional[Any], List[str]]:
    notes: List[str] = []
    if not enabled:
        note = "NoiseGuard disabled for this diagnostic/test run; no Wi-Fi, Bluetooth, display, or service changes were applied."
        notes.append(note)
        write_json(run_dir / "noise_guard_summary.json", {"enabled": False, "reason": note})
        return None, notes

    package_root = resolve_path(item["package_root"])
    sys.path.insert(0, str(package_root))
    try:
        from jetson.noise_guard import NoiseGuard, NoiseGuardConfig  # type: ignore

        guard = NoiseGuard(
            NoiseGuardConfig(
                enabled=True,
                profile="safe",
                preserve_url=str(cfg.get("external_logger_url") or "http://${LOGGER_DASHBOARD_HOST}:8000"),
                display_off=True,
                stop_services=True,
                disable_wifi=True,
                disable_bluetooth=True,
            ),
            notes=notes,
        )
        write_json(run_dir / "noise_guard_summary.json", guard.enter())
        return guard, notes
    except Exception as exc:
        notes.append(f"NoiseGuard unavailable or failed: {type(exc).__name__}: {exc}")
        write_json(run_dir / "noise_guard_summary.json", {"enabled": False, "error": notes[-1]})
        return None, notes


def status_payload(
    *,
    start_mono: float,
    total_images: int,
    latencies: List[float],
    last_status_mono: float,
    last_status_images: int,
    config_name: str,
    run_id: str,
    loop_index: int = 0,
    image_index: Optional[int] = None,
) -> Dict[str, Any]:
    now = time.monotonic()
    elapsed = max(now - start_mono, 1e-9)
    recent_elapsed = max(now - last_status_mono, 1e-9)
    return {
        "elapsed_seconds": elapsed,
        "total_images": total_images,
        "overall_fps": total_images / elapsed,
        "recent_fps": (total_images - last_status_images) / recent_elapsed,
        "median_latency_ms": percentile(latencies, 50.0),
        "p95_latency_ms": percentile(latencies, 95.0),
        "config_name": config_name,
        "repeat_name": format_repeat_name(config_name, loop_index),
        "repeat_index": loop_index,
        "loop_index": loop_index,
        "image_index": image_index,
        "run_id": run_id,
    }


def stop_condition(args: argparse.Namespace, total_images: int, elapsed_s: float) -> Optional[str]:
    if args.mode == "smoke":
        target_images = max(int(args.max_images or 0), 500)
        target_seconds = max(float(args.max_seconds or 0.0), 180.0)
        if total_images >= target_images and elapsed_s >= target_seconds:
            return "smoke_complete"
        return None
    if args.max_images is not None and total_images >= int(args.max_images):
        return "max_images_reached"
    if args.max_seconds is not None and elapsed_s >= float(args.max_seconds):
        return "max_seconds_reached"
    return None


def config_bool(cfg: Dict[str, Any], item: Dict[str, Any], key: str, default: bool) -> bool:
    if key in item:
        return bool(item.get(key))
    extra_args = item.get("extra_args")
    if isinstance(extra_args, dict) and key in extra_args:
        return bool(extra_args.get(key))
    return bool(cfg.get(key, default))


def config_float(cfg: Dict[str, Any], item: Dict[str, Any], key: str, default: float) -> float:
    if key in item:
        try:
            return float(item.get(key))
        except Exception:
            return float(default)
    extra_args = item.get("extra_args")
    if isinstance(extra_args, dict) and key in extra_args:
        try:
            return float(extra_args.get(key))
        except Exception:
            return float(default)
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def trtexec_runtime_flags(cfg: Dict[str, Any], item: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    use_cuda_graph = config_bool(cfg, item, "trtexec_use_cuda_graph", True)
    separate_profile = config_bool(cfg, item, "trtexec_separate_profile_run", True)
    if getattr(args, "disable_trtexec_cuda_graph", False):
        use_cuda_graph = False
    if getattr(args, "disable_trtexec_separate_profile_run", False):
        separate_profile = False

    flags: List[str] = []
    if use_cuda_graph:
        flags.append("--useCudaGraph")
    if separate_profile:
        flags.append("--separateProfileRun")
    return flags


def trtexec_chunk_timeout_seconds(
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    args: argparse.Namespace,
    duration_s: int,
    warmup_ms: int,
) -> float:
    configured = getattr(args, "trtexec_chunk_timeout_seconds", None)
    if configured is None:
        configured = config_float(cfg, item, "trtexec_chunk_timeout_seconds", 90.0)
    minimum = float(duration_s) + (float(warmup_ms) / 1000.0) + 60.0
    return max(float(configured), minimum, 30.0)


def build_trtexec_command(
    *,
    trtexec: str,
    engine: Path,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    args: argparse.Namespace,
    iterations: int,
    duration_s: int,
    warmup_ms: int,
    export_times: Path,
) -> List[str]:
    cmd = [
        trtexec,
        f"--loadEngine={engine}",
        f"--iterations={iterations}",
        f"--duration={duration_s}",
        f"--warmUp={warmup_ms}",
    ]
    cmd.extend(trtexec_runtime_flags(cfg, item, args))
    cmd.append(f"--exportTimes={export_times}")
    return cmd


def build_pure_engine_command_template(
    engine: Path,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    chunk_s = float(args.trtexec_chunk_seconds or cfg.get("trtexec_chunk_seconds") or 30.0)
    warmup_ms = int(cfg.get("trtexec_warmup_ms") or 0)
    cmd = [
        resolve_trtexec_binary(),
        f"--loadEngine={engine}",
        "--iterations=<chunk_iterations>",
        f"--duration=<chunk_duration_s, default {chunk_s:g}>",
        f"--warmUp={warmup_ms}",
    ]
    cmd.extend(trtexec_runtime_flags(cfg, item, args))
    cmd.append("--exportTimes=<run_dir>/trtexec_chunks/<chunk>/trtexec_times.json")
    return shlex.join(cmd)


def parse_trtexec_times(path: Path) -> List[float]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        for key in ("times", "records", "iterations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return []

    latencies: List[float] = []
    for sample in data:
        if not isinstance(sample, dict):
            continue
        value = sample.get("latencyMs")
        if value is None:
            start_ms = sample.get("startComputeMs", sample.get("startEnqMs", sample.get("startMs", 0.0)))
            end_ms = sample.get("endComputeMs", sample.get("endEnqMs", sample.get("endMs", start_ms)))
            try:
                value = max(0.0, float(end_ms) - float(start_ms))
            except Exception:
                value = None
        try:
            latency = float(value)
        except Exception:
            continue
        if latency >= 0.0:
            latencies.append(latency)
    return latencies


def trtexec_chunk_limits(args: argparse.Namespace, cfg: Dict[str, Any], total_images: int, elapsed_s: float) -> Tuple[int, int]:
    chunk_s = max(1.0, float(args.trtexec_chunk_seconds or cfg.get("trtexec_chunk_seconds") or 30.0))
    duration_s = int(round(chunk_s))
    iterations = 1

    if args.mode == "smoke":
        target_images = max(int(args.max_images or 0), 500)
        target_seconds = max(float(args.max_seconds or 0.0), 180.0)
        if elapsed_s >= target_seconds and total_images < target_images:
            return max(1, target_images - total_images), 0
        remaining_s = max(1.0, target_seconds - elapsed_s)
        return iterations, int(round(min(chunk_s, remaining_s)))

    if args.max_images is not None:
        remaining_images = max(1, int(args.max_images) - total_images)
        if args.max_seconds is None:
            return remaining_images, 0
        iterations = remaining_images

    if args.max_seconds is not None:
        remaining_s = max(1.0, float(args.max_seconds) - elapsed_s)
        duration_s = int(round(min(chunk_s, remaining_s)))

    return max(1, iterations), max(0, duration_s)


def _write_alarm_hit(
    writer: csv.DictWriter,
    *,
    timestamp_wall: Any,
    source: str,
    alarm_type: str,
    severity: str,
    metric: str,
    value: Any,
    threshold: Any,
    details: Dict[str, Any],
) -> None:
    writer.writerow(
        {
            "timestamp_wall": timestamp_wall,
            "source": source,
            "alarm_type": alarm_type,
            "severity": severity,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "details_json": json.dumps(details, sort_keys=True, default=str),
        }
    )


def scan_alarm_files(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = alarm_thresholds(cfg)
    hits_path = run_dir / "alarm_threshold_hits.csv"
    summary_path = run_dir / "alarm_summary.json"
    summary_csv_path = run_dir / "alarm_summary.csv"
    stats: Dict[str, Any] = {
        "thresholds": thresholds,
        "hit_count": 0,
        "warning_count": 0,
        "critical_count": 0,
        "internal_overtemperature_hits": 0,
        "external_overcurrent_hits": 0,
        "external_high_power_hits": 0,
        "external_overtemperature_hits": 0,
        "tegrastats_fault_text_hits": 0,
        "max_internal_temp_c": None,
        "max_external_current_a": None,
        "max_external_power_w": None,
        "max_external_temp_c": None,
        "realtime_alarm_events": 0,
        "alarm_events_csv": str(run_dir / "alarm_events.csv"),
        "alarm_threshold_hits_csv": str(hits_path),
    }

    def bump(severity: str, key: str) -> None:
        stats["hit_count"] += 1
        stats[key] += 1
        if severity == "critical":
            stats["critical_count"] += 1
        else:
            stats["warning_count"] += 1

    with hits_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALARM_HIT_FIELDS)
        writer.writeheader()

        internal_csv = run_dir / "internal_telemetry.csv"
        if internal_csv.exists():
            with internal_csv.open("r", encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    temp = _maybe_float(row.get("internal_max_temp_c"))
                    if temp is not None:
                        prev = _maybe_float(stats["max_internal_temp_c"])
                        stats["max_internal_temp_c"] = temp if prev is None else max(prev, temp)
                    severity, threshold = threshold_severity(
                        temp,
                        thresholds["internal_overtemperature_warning_c"],
                        thresholds["internal_overtemperature_critical_c"],
                    )
                    if severity is not None:
                        bump(severity, "internal_overtemperature_hits")
                        _write_alarm_hit(
                            writer,
                            timestamp_wall=row.get("timestamp_wall"),
                            source="jetson_internal",
                            alarm_type="overtemperature",
                            severity=severity,
                            metric="internal_max_temp_c",
                            value=temp,
                            threshold=threshold,
                            details={"row": row},
                        )
                    raw = str(row.get("tegrastats_raw") or "")
                    if raw and FAULT_TEXT_PATTERN.search(raw):
                        bump("warning", "tegrastats_fault_text_hits")
                        _write_alarm_hit(
                            writer,
                            timestamp_wall=row.get("timestamp_wall"),
                            source="tegrastats",
                            alarm_type="tegrastats_fault_text",
                            severity="warning",
                            metric="tegrastats_raw",
                            value=raw,
                            threshold="OC/overcurrent/thermal/throttle text",
                            details={"row": row},
                        )

        external_files = sorted((run_dir / "external_logger_archive").rglob("external_measurements.csv"))
        if external_files:
            stats["external_measurements_csv"] = str(external_files[0])
            with external_files[0].open("r", encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    current = _maybe_float(row.get("current_a"))
                    power = _maybe_float(row.get("power_w"))
                    ext_temp = _maybe_float(row.get("external_temp_c") or row.get("tmp117_temp_c"))
                    if current is not None:
                        prev = _maybe_float(stats["max_external_current_a"])
                        stats["max_external_current_a"] = current if prev is None else max(prev, current)
                    if power is not None:
                        prev = _maybe_float(stats["max_external_power_w"])
                        stats["max_external_power_w"] = power if prev is None else max(prev, power)
                    if ext_temp is not None:
                        prev = _maybe_float(stats["max_external_temp_c"])
                        stats["max_external_temp_c"] = ext_temp if prev is None else max(prev, ext_temp)
                    for metric, value, warning_key, critical_key, alarm_type, counter_key in (
                        (
                            "current_a",
                            current,
                            "external_overcurrent_warning_a",
                            "external_overcurrent_critical_a",
                            "overcurrent",
                            "external_overcurrent_hits",
                        ),
                        (
                            "power_w",
                            power,
                            "external_power_warning_w",
                            "external_power_critical_w",
                            "external_high_power",
                            "external_high_power_hits",
                        ),
                        (
                            "external_temp_c",
                            ext_temp,
                            "external_overtemperature_warning_c",
                            "external_overtemperature_critical_c",
                            "external_overtemperature",
                            "external_overtemperature_hits",
                        ),
                    ):
                        severity, threshold = threshold_severity(value, thresholds[warning_key], thresholds[critical_key])
                        if severity is None:
                            continue
                        bump(severity, counter_key)
                        _write_alarm_hit(
                            writer,
                            timestamp_wall=row.get("timestamp_utc"),
                            source="external_logger_archive",
                            alarm_type=alarm_type,
                            severity=severity,
                            metric=metric,
                            value=value,
                            threshold=threshold,
                            details={"row": row},
                        )
        else:
            stats["external_measurements_csv"] = None

    alarm_events = run_dir / "alarm_events.csv"
    if alarm_events.exists():
        with alarm_events.open("r", encoding="utf-8", newline="") as src:
            stats["realtime_alarm_events"] = sum(1 for _ in csv.DictReader(src))

    stats["has_alarm"] = bool(stats["hit_count"] or stats["realtime_alarm_events"])
    stats["status"] = "ALARMS_FOUND" if stats["has_alarm"] else "NO_ALARMS_DETECTED"
    write_json(summary_path, stats)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "status",
            "hit_count",
            "warning_count",
            "critical_count",
            "realtime_alarm_events",
            "internal_overtemperature_hits",
            "external_overcurrent_hits",
            "external_high_power_hits",
            "external_overtemperature_hits",
            "tegrastats_fault_text_hits",
            "max_internal_temp_c",
            "max_external_current_a",
            "max_external_power_w",
            "max_external_temp_c",
            "alarm_events_csv",
            "alarm_threshold_hits_csv",
            "external_measurements_csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: stats.get(key, "") for key in fieldnames})
    return stats


def terminate_process_tree(proc: subprocess.Popen[Any]) -> int:
    try:
        proc.send_signal(signal.SIGINT)
        return proc.wait(timeout=10)
    except Exception:
        pass
    try:
        proc.terminate()
        return proc.wait(timeout=10)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    return proc.wait(timeout=10)


def run_trtexec_chunk(
    cmd: List[str],
    stdout_path: Path,
    stderr_path: Path,
    events: EventLogger,
    *,
    timeout_s: float,
    watchdog_interval_s: float,
    warn_after_s: float,
    progress: ProgressTracker,
    dashboard: DashboardPusher,
    total_images: int,
    chunk_index: int,
    repeat_name: str,
) -> Dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    timed_out = False
    start_mono = time.monotonic()
    next_watchdog = start_mono + max(float(watchdog_interval_s), 1.0)
    with stdout_path.open("w", encoding="utf-8") as out_handle, stderr_path.open("w", encoding="utf-8") as err_handle:
        proc = subprocess.Popen(cmd, stdout=out_handle, stderr=err_handle, text=True)
        events.emit(
            "TRTEXEC_CHUNK_PROCESS_START",
            {
                "child_pid": proc.pid,
                "chunk_index": chunk_index,
                "repeat_name": repeat_name,
                "timeout_s": timeout_s,
                "watchdog_interval_s": watchdog_interval_s,
                "command": cmd,
            },
        )
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    elapsed_s = time.monotonic() - start_mono
                    return {
                        "returncode": rc,
                        "interrupted": interrupted,
                        "timed_out": timed_out,
                        "elapsed_seconds": elapsed_s,
                    }
                elapsed_s = time.monotonic() - start_mono
                if elapsed_s >= float(timeout_s):
                    timed_out = True
                    msg = (
                        f"trtexec chunk {chunk_index} exceeded watchdog timeout "
                        f"({elapsed_s:.1f}s >= {timeout_s:.1f}s). Killing child pid {proc.pid}."
                    )
                    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
                    err_handle.write(f"\n[WATCHDOG_TIMEOUT] {utc_now_iso()} {msg}\n")
                    err_handle.flush()
                    events.emit(
                        "TRTEXEC_CHUNK_TIMEOUT",
                        {
                            "child_pid": proc.pid,
                            "chunk_index": chunk_index,
                            "repeat_name": repeat_name,
                            "elapsed_seconds": elapsed_s,
                            "timeout_s": timeout_s,
                            "command": cmd,
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                            "samples_before_chunk": total_images,
                        },
                    )
                    progress.update(
                        stage="trtexec_timeout",
                        state="error",
                        message=msg,
                        processed_images=total_images,
                        repeat_name=repeat_name,
                        repeat_index=chunk_index,
                        stopped_reason="fatal_trtexec_timeout",
                    )
                    dashboard.push(
                        stage="trtexec_timeout",
                        state="error",
                        message=msg,
                        processed_images=total_images,
                        repeat_name=repeat_name,
                        repeat_index=chunk_index,
                        stopped_reason="fatal_trtexec_timeout",
                        extra={"timing_boundary": "pure_engine_only", "measurement_backend": "trtexec_random_input"},
                    )
                    rc = terminate_process_tree(proc)
                    return {
                        "returncode": rc,
                        "interrupted": interrupted,
                        "timed_out": timed_out,
                        "elapsed_seconds": time.monotonic() - start_mono,
                    }
                if elapsed_s >= float(warn_after_s) and time.monotonic() >= next_watchdog:
                    remaining_s = max(0.0, float(timeout_s) - elapsed_s)
                    msg = (
                        f"trtexec chunk {chunk_index} still running: elapsed={elapsed_s:.1f}s, "
                        f"timeout_in={remaining_s:.1f}s, samples_so_far={total_images}. "
                        "No latency rows are available until this chunk finishes."
                    )
                    print(f"[WATCHDOG] {msg}", flush=True)
                    events.emit(
                        "TRTEXEC_CHUNK_WATCHDOG",
                        {
                            "child_pid": proc.pid,
                            "chunk_index": chunk_index,
                            "repeat_name": repeat_name,
                            "elapsed_seconds": elapsed_s,
                            "timeout_s": timeout_s,
                            "samples_so_far": total_images,
                        },
                    )
                    progress.update(
                        stage="trtexec_chunk_running",
                        state="running",
                        message=msg,
                        processed_images=total_images,
                        repeat_name=repeat_name,
                        repeat_index=chunk_index,
                    )
                    dashboard.push(
                        stage="trtexec_chunk_running",
                        state="running",
                        message=msg,
                        processed_images=total_images,
                        repeat_name=repeat_name,
                        repeat_index=chunk_index,
                        extra={"trtexec_watchdog": True},
                    )
                    next_watchdog = time.monotonic() + max(float(watchdog_interval_s), 1.0)
                time.sleep(0.2)
        except KeyboardInterrupt:
            interrupted = True
            events.emit("CTRL_C_RECEIVED", {"stage": "trtexec_child", "child_pid": proc.pid})
            rc = terminate_process_tree(proc)
            return {
                "returncode": rc,
                "interrupted": interrupted,
                "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - start_mono,
            }


def append_pure_engine_latency_rows(
    *,
    writer: csv.DictWriter,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    run_id: str,
    chunk_index: int,
    start_index: int,
    latencies: List[float],
    sample_prefix: str,
) -> None:
    for offset, latency_ms in enumerate(latencies):
        global_index = start_index + offset
        writer.writerow(
            {
                "timestamp_wall": utc_now_iso(),
                "timestamp_monotonic": f"{time.monotonic():.6f}",
                "run_id": run_id,
                "board": cfg.get("board", "nano_super"),
                "task": item["task"],
                "model": item["model"],
                "precision": item["precision"],
                "input_size": item["input_size"],
                "batch_size": item["batch_size"],
                "loop_index": chunk_index,
                "image_index": global_index,
                "image_id_or_path": f"{sample_prefix}/sample_{offset + 1:08d}",
                "latency_ms": f"{latency_ms:.6f}",
                "success": "true",
                "error_message": "",
            }
        )


def run_pure_engine_loop(
    *,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    config_name: str,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    events: EventLogger,
    dashboard: DashboardPusher,
    progress: ProgressTracker,
    notes: List[str],
) -> str:
    engine = resolve_path(item["engine_path"])
    trtexec = resolve_trtexec_binary()
    chunks_root = run_dir / "trtexec_chunks"
    latency_path = run_dir / "inference_latency.csv"
    latencies: List[float] = []
    total_images = 0
    chunk_index = 0
    last_status_time = time.monotonic()
    last_status_images = 0
    start_mono = time.monotonic()
    warmup_ms = int(cfg.get("trtexec_warmup_ms") or 0)
    watchdog_interval_s = max(
        1.0,
        float(
            getattr(args, "trtexec_watchdog_interval", None)
            or config_float(cfg, item, "trtexec_watchdog_interval_seconds", 15.0)
        ),
    )

    events.emit(
        "INFERENCE_START",
        {
            "backend": "trtexec_random_input_pure_engine",
            "engine_path": str(engine),
            "trtexec": trtexec,
            "timing_boundary": "pure_engine_only",
            "note": "No dataset images are decoded; trtexec uses TensorRT-managed random input buffers.",
        },
    )
    progress.update(
        stage="inference_start",
        state="running",
        message="Pure engine battery inference started",
        processed_images=0,
        repeat_name=format_repeat_name(config_name, 0),
        repeat_index=0,
    )
    dashboard.push(
        stage="inference_start",
        state="running",
        message="Pure engine battery inference started",
        processed_images=0,
        repeat_name=format_repeat_name(config_name, 0),
        repeat_index=0,
        extra={"backend": "trtexec_random_input_pure_engine", "timing_boundary": "pure_engine_only"},
    )

    with latency_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATENCY_FIELDS)
        while True:
            elapsed = time.monotonic() - start_mono
            reason = stop_condition(args, total_images, elapsed)
            if reason:
                return reason

            chunk_index += 1
            repeat_name = format_repeat_name(config_name, chunk_index)
            progress.update(
                stage="inference_running",
                state="running",
                message="Pure engine inference is running",
                processed_images=total_images,
                current_image_name="trtexec-random-input",
                repeat_name=repeat_name,
                repeat_index=chunk_index,
                loop_index=chunk_index,
            )
            events.emit(
                "LOOP_START",
                {
                    "config_name": config_name,
                    "mode": args.mode,
                    "repeat_name": repeat_name,
                    "repeat_index": chunk_index,
                    "total_images_before_chunk": total_images,
                },
            )

            chunk_dir = chunks_root / f"chunk_{chunk_index:05d}"
            export_times = chunk_dir / "trtexec_times.json"
            stdout_path = chunk_dir / "trtexec_stdout.txt"
            stderr_path = chunk_dir / "trtexec_stderr.txt"
            iterations, duration_s = trtexec_chunk_limits(args, cfg, total_images, elapsed)
            timeout_s = trtexec_chunk_timeout_seconds(cfg, item, args, duration_s, warmup_ms)
            warn_after_s = min(max(float(duration_s) + (float(warmup_ms) / 1000.0) + 15.0, 10.0), timeout_s)
            cmd = build_trtexec_command(
                trtexec=trtexec,
                engine=engine,
                cfg=cfg,
                item=item,
                args=args,
                iterations=iterations,
                duration_s=duration_s,
                warmup_ms=warmup_ms,
                export_times=export_times,
            )
            write_json(
                chunk_dir / "trtexec_command.json",
                {
                    "cmd": cmd,
                    "chunk_index": chunk_index,
                    "repeat_name": repeat_name,
                    "iterations": iterations,
                    "duration_s": duration_s,
                    "timeout_s": timeout_s,
                    "watchdog_interval_s": watchdog_interval_s,
                    "runtime_flags": trtexec_runtime_flags(cfg, item, args),
                },
            )
            chunk_result = run_trtexec_chunk(
                cmd,
                stdout_path,
                stderr_path,
                events,
                timeout_s=timeout_s,
                watchdog_interval_s=watchdog_interval_s,
                warn_after_s=warn_after_s,
                progress=progress,
                dashboard=dashboard,
                total_images=total_images,
                chunk_index=chunk_index,
                repeat_name=repeat_name,
            )
            rc = int(chunk_result.get("returncode") if chunk_result.get("returncode") is not None else -1)
            interrupted = bool(chunk_result.get("interrupted"))
            timed_out = bool(chunk_result.get("timed_out"))
            chunk_latencies = parse_trtexec_times(export_times)
            events.emit(
                "TRTEXEC_CHUNK_END",
                {
                    "chunk_index": chunk_index,
                    "repeat_name": repeat_name,
                    "returncode": rc,
                    "interrupted": interrupted,
                    "timed_out": timed_out,
                    "elapsed_seconds": chunk_result.get("elapsed_seconds"),
                    "samples_exported": len(chunk_latencies),
                    "export_times": str(export_times),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )
            if chunk_latencies:
                append_pure_engine_latency_rows(
                    writer=writer,
                    cfg=cfg,
                    item=item,
                    run_id=run_id,
                    chunk_index=chunk_index,
                    start_index=total_images,
                    latencies=chunk_latencies,
                    sample_prefix=f"trtexec_random_input/{repeat_name}",
                )
                handle.flush()
                latencies.extend(chunk_latencies)
                total_images += len(chunk_latencies)

            if timed_out:
                msg = (
                    f"trtexec chunk {chunk_index} timed out before producing a complete result; "
                    f"samples_exported={len(chunk_latencies)}. See {stderr_path}."
                )
                notes.append(msg)
                events.emit(
                    "ERROR",
                    {
                        "stage": "trtexec_chunk",
                        "error": msg,
                        "returncode": rc,
                        "command": cmd,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "samples_exported": len(chunk_latencies),
                    },
                )
                progress.update(
                    stage="trtexec_timeout",
                    state="error",
                    message=msg,
                    processed_images=total_images,
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    stopped_reason="fatal_trtexec_timeout",
                )
                dashboard.push(
                    stage="trtexec_timeout",
                    state="error",
                    message=msg,
                    processed_images=total_images,
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    stopped_reason="fatal_trtexec_timeout",
                    extra={"timing_boundary": "pure_engine_only", "measurement_backend": "trtexec_random_input"},
                )
                return "fatal_trtexec_timeout"

            if not chunk_latencies and not interrupted and config_bool(cfg, item, "trtexec_fail_on_zero_samples", True):
                msg = (
                    f"trtexec chunk {chunk_index} ended without latency samples; "
                    f"returncode={rc}, export_times_exists={export_times.exists()}. See {stdout_path} and {stderr_path}."
                )
                notes.append(msg)
                events.emit(
                    "ERROR",
                    {
                        "stage": "trtexec_chunk",
                        "error": msg,
                        "returncode": rc,
                        "command": cmd,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "samples_exported": 0,
                    },
                )
                progress.update(
                    stage="trtexec_no_samples",
                    state="error",
                    message=msg,
                    processed_images=total_images,
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    stopped_reason="fatal_no_trtexec_samples",
                )
                dashboard.push(
                    stage="trtexec_no_samples",
                    state="error",
                    message=msg,
                    processed_images=total_images,
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    stopped_reason="fatal_no_trtexec_samples",
                    extra={"timing_boundary": "pure_engine_only", "measurement_backend": "trtexec_random_input"},
                )
                return "fatal_no_trtexec_samples"

            if rc != 0 and not interrupted:
                events.emit(
                    "ERROR",
                    {
                        "stage": "trtexec_chunk",
                        "returncode": rc,
                        "command": cmd,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "samples_exported": len(chunk_latencies),
                    },
                )
                return "fatal_inference_error"

            now = time.monotonic()
            if chunk_latencies and (total_images == len(chunk_latencies) or (now - last_status_time) >= float(args.status_interval) or interrupted):
                payload = status_payload(
                    start_mono=start_mono,
                    total_images=total_images,
                    latencies=latencies,
                    last_status_mono=last_status_time,
                    last_status_images=last_status_images,
                    config_name=config_name,
                    run_id=run_id,
                    loop_index=chunk_index,
                    image_index=total_images,
                )
                print(
                    "[STATUS] "
                    f"elapsed={payload['elapsed_seconds']:.1f}s samples={total_images} "
                    f"fps={payload['overall_fps']:.3f} recent_fps={payload['recent_fps']:.3f} "
                    f"median_ms={payload['median_latency_ms']:.3f} p95_ms={payload['p95_latency_ms']:.3f} "
                    f"config={config_name} run_id={run_id} backend=trtexec_pure_engine",
                    flush=True,
                )
                events.emit("STATUS", payload)
                progress.update(
                    stage="inference_running",
                    state="running",
                    message="Pure engine inference is running",
                    processed_images=total_images,
                    current_image_name="trtexec-random-input",
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    loop_index=chunk_index,
                    image_index=total_images,
                    fps=float(payload["overall_fps"]),
                    median_latency_ms=float(payload["median_latency_ms"]),
                    p95_latency_ms=float(payload["p95_latency_ms"]),
                )
                dashboard.push(
                    stage="inference_running",
                    state="running",
                    message="Pure engine inference is running",
                    processed_images=total_images,
                    current_image_name="trtexec-random-input",
                    repeat_name=repeat_name,
                    repeat_index=chunk_index,
                    fps=float(payload["overall_fps"]),
                    median_latency_ms=float(payload["median_latency_ms"]),
                    p95_latency_ms=float(payload["p95_latency_ms"]),
                    extra={"timing_boundary": "pure_engine_only", "measurement_backend": "trtexec_random_input"},
                )
                last_status_time = now
                last_status_images = total_images

            if interrupted:
                return "manual_ctrl_c_power_bank_about_5_percent"


def build_summary(
    *,
    cfg: Dict[str, Any],
    item: Dict[str, Any],
    config_name: str,
    run_id: str,
    run_dir: Path,
    preflight: Dict[str, Any],
    stopped_reason: str,
    start_time: Optional[str],
    end_time: str,
    elapsed_seconds: float,
    command_used: str,
    notes: List[str],
    sudo_status: Dict[str, Any],
    dashboard_status: Dict[str, Any],
    external_logger_status: Dict[str, Any],
    internal_telemetry_status: Dict[str, Any],
    alarm_status: Dict[str, Any],
) -> Dict[str, Any]:
    stats = latency_stats(run_dir / "inference_latency.csv")
    fps = stats["total_images"] / elapsed_seconds if elapsed_seconds > 0 and stats["total_images"] else 0.0
    run_valid = bool(stats["total_images"] > 0 and not str(stopped_reason).startswith("fatal_"))
    logger_control = str(cfg.get("external_logger_control") or "auto")
    logger_note = (
        "External power logging is controlled automatically by this Jetson runner through Pc_package /start and /stop."
        if logger_control == "auto"
        else "External power logging is handled manually by Pc_package and is not started or stopped by this runner."
    )
    return {
        "run_id": run_id,
        "board": cfg.get("board", "nano_super"),
        "board_display_name": cfg.get("board_display_name"),
        "config_name": config_name,
        "task": item.get("task"),
        "model": item.get("model"),
        "precision": item.get("precision"),
        "input_size": item.get("input_size"),
        "batch_size": item.get("batch_size"),
        "start_time": start_time,
        "end_time": end_time,
        "stopped_reason": stopped_reason,
        "run_valid": run_valid,
        "failure_class": "" if run_valid else stopped_reason,
        "total_images": stats["total_images"],
        "elapsed_seconds": elapsed_seconds,
        "fps": fps,
        "median_latency_ms": stats["median_latency_ms"],
        "p95_latency_ms": stats["p95_latency_ms"],
        "min_latency_ms": stats["min_latency_ms"],
        "max_latency_ms": stats["max_latency_ms"],
        "nvpmodel": {
            "query": preflight.get("nvpmodel_q"),
            "query_verbose": preflight.get("nvpmodel_q_verbose"),
            "power_mode_ok": preflight.get("power_mode_ok"),
            "warning": preflight.get("power_mode_warning"),
        },
        "device_tree_model": preflight.get("device_tree_model"),
        "jetpack_l4t": {
            "l4t_release": preflight.get("l4t_release"),
            "os_release": preflight.get("os_release"),
        },
        "cuda_tensorrt": preflight.get("versions"),
        "git_commit": preflight.get("git_commit"),
        "sudo": sudo_status,
        "dashboard_status_push": dashboard_status,
        "external_logger": external_logger_status,
        "internal_telemetry": internal_telemetry_status,
        "alarms": alarm_status,
        "engine_path": str(resolve_path(item["engine_path"])),
        "dataset_path": str(resolve_path(item["dataset_path"])),
        "inference_command_used": command_used,
        "pure_engine_benchmark": True,
        "timing_boundary": "pure_engine_only",
        "measurement_backend": "trtexec_random_input",
        "pure_engine_note": (
            "This run measures TensorRT engine execution through trtexec with TensorRT-managed random input buffers. "
            "It does not include dataset image loading, decode, preprocessing, postprocessing, prediction saving, or model wrapper overhead."
        ),
        "trtexec_runtime_options": {
            "use_cuda_graph": "--useCudaGraph" in str(command_used),
            "separate_profile_run": "--separateProfileRun" in str(command_used),
            "chunk_timeout_seconds": config_float(cfg, item, "trtexec_chunk_timeout_seconds", 90.0),
            "watchdog_interval_seconds": config_float(cfg, item, "trtexec_watchdog_interval_seconds", 15.0),
            "fail_on_zero_samples": config_bool(cfg, item, "trtexec_fail_on_zero_samples", True),
            "config_note": item.get("trtexec_hang_fix_note", ""),
        },
        "external_power_logging_note": logger_note,
        "power_source_note": "This run is powered by UGREEN 25,000mAh 145W power bank.",
        "notes": notes,
        "run_dir": str(run_dir),
    }


def write_summary_files(run_dir: Path, summary: Dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", summary)
    with (run_dir / "inference_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        row = {field: summary.get(field, "") for field in SUMMARY_FIELDS}
        row["total_elapsed_seconds"] = summary.get("elapsed_seconds", "")
        writer.writerow(row)
    md = [
        "# Nano Super Pure Engine Battery Summary",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- board: `{summary.get('board')}`",
        f"- config: `{summary.get('config_name')}`",
        f"- model: `{summary.get('model')}`",
        f"- task: `{summary.get('task')}`",
        f"- precision: `{summary.get('precision')}`",
        f"- input_size: `{summary.get('input_size')}`",
        f"- batch_size: `{summary.get('batch_size')}`",
        f"- stopped_reason: `{summary.get('stopped_reason')}`",
        f"- run_valid: `{summary.get('run_valid')}`",
        f"- failure_class: `{summary.get('failure_class')}`",
        f"- total_images: `{summary.get('total_images')}`",
        f"- elapsed_seconds: `{summary.get('elapsed_seconds'):.3f}`",
        f"- fps: `{summary.get('fps'):.6f}`",
        f"- median_latency_ms: `{summary.get('median_latency_ms'):.6f}`",
        f"- p95_latency_ms: `{summary.get('p95_latency_ms'):.6f}`",
        f"- timing_boundary: `{summary.get('timing_boundary')}`",
        f"- measurement_backend: `{summary.get('measurement_backend')}`",
        f"- alarm_status: `{(summary.get('alarms') or {}).get('postrun_scan', {}).get('status', 'unknown')}`",
        f"- alarm_hits: `{(summary.get('alarms') or {}).get('postrun_scan', {}).get('hit_count', '')}`",
        f"- realtime_alarm_events: `{(summary.get('alarms') or {}).get('event_count', '')}`",
        "",
        "This is pure TensorRT engine timing from `trtexec` random input buffers, not end-to-end image pipeline timing.",
        "External power logs are matched by `run_id`; in auto logger mode the runner starts/stops Pc_package itself.",
    ]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure TensorRT engine-only battery runner for Jetson Orin Nano Super.")
    parser.add_argument("--config", default="config/battery_inference_pure_engine_nano_super.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--model", choices=["yolo11s", "segformer_b0"], default=None)
    parser.add_argument("--task", choices=["detection", "segmentation"], default=None)
    parser.add_argument("--precision", choices=["INT8", "FP16"], default=None)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=["endurance", "smoke"], default="endurance")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--status-interval", type=float, default=None)
    parser.add_argument("--trtexec-chunk-seconds", type=float, default=None)
    parser.add_argument("--trtexec-chunk-timeout-seconds", type=float, default=None)
    parser.add_argument("--trtexec-watchdog-interval", type=float, default=None)
    parser.add_argument("--disable-trtexec-cuda-graph", action="store_true")
    parser.add_argument("--disable-trtexec-separate-profile-run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set-nvpmodel", action="store_true")
    parser.add_argument("--allow-power-mode-mismatch", action="store_true")
    parser.add_argument("--logger-control", choices=["auto", "manual", "off"], default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--disable-noise-guard", action="store_true")
    parser.add_argument("--enable-noise-guard-in-smoke", action="store_true")
    parser.add_argument("--wait-for-enter", dest="wait_for_enter", action="store_true", default=True)
    parser.add_argument("--no-wait-for-enter", dest="wait_for_enter", action="store_false")
    parser.add_argument("--sudo-password-once", dest="sudo_password_once", action="store_true", default=True)
    parser.add_argument("--no-sudo-password-once", dest="sudo_password_once", action="store_false")
    parser.add_argument("--sudo-keepalive-interval", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    if args.logger_control:
        cfg["external_logger_control"] = args.logger_control
    if args.status_interval is None:
        args.status_interval = float(cfg.get("status_interval_seconds") or 60)

    config_name, item = select_config(cfg, args)
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{cfg.get('board', 'nano_super')}_pure_engine_{config_name}_{timestamp}"
    out_root = resolve_path(args.out_dir or cfg.get("output_root") or "./battery_inference_runs/nano_super")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    stdout_tee = FdTee(1, run_dir / "stdout.log")
    stderr_tee = FdTee(2, run_dir / "stderr.log")
    stdout_tee.start()
    stderr_tee.start()

    events = EventLogger(run_dir / "events.csv", run_id, str(cfg.get("board", "nano_super")))
    init_latency_csv(run_dir / "inference_latency.csv")
    write_external_logger_note(run_dir / "external_logger_note.txt", run_id, cfg)

    preflight: Dict[str, Any] = {}
    notes: List[str] = []
    start_time: Optional[str] = None
    start_mono: Optional[float] = None
    stopped_reason = "not_started"
    command_used = ""
    guard = None
    sudo_keepalive: Optional[SudoKeepAlive] = None
    sudo_status: Dict[str, Any] = {}
    telemetry = InternalTelemetry(run_dir, interval_s=1.0)
    dashboard = DashboardPusher(cfg, item, config_name, run_id, notes, telemetry=telemetry)
    progress = ProgressTracker(config_name)
    heartbeat = DashboardHeartbeat(dashboard, progress, interval_s=2.0)
    external_logger = ExternalLoggerController(cfg, item, run_id, run_dir, notes)
    internal_telemetry_status: Dict[str, Any] = {}
    alarm_monitor = AlarmMonitor(run_dir, cfg, run_id, telemetry)
    alarm_status: Dict[str, Any] = {}

    try:
        events.emit("RUN_CREATED", {"run_dir": str(run_dir), "config_name": config_name})
        telemetry.start()
        alarm_monitor.start()
        heartbeat.start()
        dashboard.push(stage="run_created", state="preflight", message="Pure engine battery run created")
        validation = validate_selection(config_name, item)
        write_json(
            run_dir / "run_config.json",
            {"config_file": str(config_path), "config_name": config_name, "selected_config": item, "validation": validation},
        )
        write_json(
            run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "created_utc": utc_now_iso(),
                "expected_files": [
                "run_config.json",
                "run_manifest.json",
                "sudo_status.json",
                "preflight_report.json",
                "events.csv",
                "inference_latency.csv",
                "inference_summary.csv",
                "internal_telemetry.csv",
                "alarm_events.csv",
                "alarm_threshold_hits.csv",
                "alarm_summary.csv",
                "alarm_summary.json",
                "trtexec_chunks/",
                "stdout.log",
                "stderr.log",
                "summary.json",
                "summary.md",
                "external_logger_note.txt",
                "external_logger_start.json",
                "external_logger_stop.json",
                "external_logger_download.json",
                "external_logger_extract.json",
                "external_logger_archive.tar.gz",
                "external_logger_archive/",
            ],
        },
        )
        if not validation["ok"]:
            stopped_reason = "preflight_validation_failed"
            events.emit("ERROR", {"stage": "validation", "errors": validation["errors"]})
            raise RuntimeError("; ".join(validation["errors"]))

        events.emit("SUDO_PREFLIGHT_START", {"password_once": bool(args.sudo_password_once)})
        sudo_status = ensure_sudo_noninteractive(args)
        sudo_keepalive = SudoKeepAlive(interval_s=float(args.sudo_keepalive_interval))
        sudo_keepalive.start()
        sudo_status["keepalive"] = sudo_keepalive.status()
        write_json(run_dir / "sudo_status.json", sudo_status)
        events.emit(
            "SUDO_PREFLIGHT_END",
            {
                "noninteractive_ok": bool((sudo_status.get("noninteractive_check") or {}).get("ok")),
                "keepalive_enabled": bool((sudo_status.get("keepalive") or {}).get("enabled")),
            },
        )

        events.emit("PREFLIGHT_START", {"target_power_mode": cfg.get("target_power_mode_label")})
        progress.update(stage="preflight", state="running", message="Pure engine preflight is running", processed_images=0)
        dashboard.push(stage="preflight", state="running", message="Pure engine preflight is running")
        preflight = run_preflight(cfg, args)
        write_json(run_dir / "preflight_report.json", preflight)
        events.emit("PREFLIGHT_END", {"power_mode_ok": preflight.get("power_mode_ok"), "warning": preflight.get("power_mode_warning")})
        dashboard.push(
            stage="preflight",
            state=("running" if preflight.get("power_mode_ok") else "warning"),
            message=("Preflight complete" if preflight.get("power_mode_ok") else str(preflight.get("power_mode_warning"))),
            extra={"power_mode_ok": preflight.get("power_mode_ok"), "power_mode_warning": preflight.get("power_mode_warning")},
        )

        if args.set_nvpmodel and not (preflight.get("nvpmodel_set") or {}).get("ok", False):
            stopped_reason = "preflight_set_nvpmodel_failed"
            events.emit("ERROR", {"stage": "preflight", "error": preflight.get("nvpmodel_set")})
            raise RuntimeError("Could not set nvpmodel; see preflight_report.json.")
        if not preflight.get("power_mode_ok") and not args.allow_power_mode_mismatch:
            stopped_reason = "preflight_power_mode_mismatch"
            events.emit("ERROR", {"stage": "preflight", "warning": preflight.get("power_mode_warning")})
            raise RuntimeError(preflight.get("power_mode_warning") or "Power mode mismatch")

        print("")
        print(f"[RUN_ID] {run_id}")
        print(f"[RUN_DIR] {run_dir}")
        print(f"[LOGGER_URL] {cfg.get('external_logger_url', 'http://${LOGGER_DASHBOARD_HOST}:8000')}")
        print("[MODE] pure engine only: trtexec random input buffers, no image decode/preprocess/postprocess.")
        if external_logger.auto_enabled():
            print(f"[LOGGER] auto mode: Pc_package /start khodesh ba run_id={run_id} seda zade mishe.")
        elif external_logger.manual_enabled():
            print(f"[LOGGER] manual mode: Pc_package ro manually start kon ba hamin run_id: {run_id}")
        else:
            print("[LOGGER] off mode: external logger start/stop nemishe.")
        print("[STOP] vaghti power bank resid be hodude 5%, Ctrl-C bezan ta summary clean save beshe.")
        if external_logger.auto_enabled() and not args.dry_run:
            events.emit("EXTERNAL_LOGGER_STARTING", {"run_id": run_id, "logger_url": cfg.get("external_logger_url")})
            progress.update(stage="external_logger_starting", state="running", message="Starting Pc_package logger automatically")
            dashboard.push(stage="external_logger_starting", state="running", message="Starting Pc_package logger automatically")
            start_info = external_logger.start()
            events.emit("EXTERNAL_LOGGER_STARTED", start_info)
            progress.update(stage="external_logger_started", state="running", message="Pc_package logger started automatically")
            dashboard.push(stage="external_logger_started", state="running", message="Pc_package logger started automatically", extra={"external_logger_start": start_info})
            external_logger.wait_until_scheduled_start()
        elif external_logger.manual_enabled():
            events.emit("WAITING_FOR_EXTERNAL_LOGGER", {"run_id": run_id, "logger_url": cfg.get("external_logger_url")})
            progress.update(stage="waiting_for_external_logger", state="waiting", message="Start Pc_package manually with this run_id")
            dashboard.push(
                stage="waiting_for_external_logger",
                state="waiting",
                message="Start Pc_package manually with this run_id, then press ENTER on Jetson",
            )
        if args.wait_for_enter and not args.dry_run and external_logger.manual_enabled():
            input("Logger ro start kardi? ENTER bezan ta inference shoroo beshe: ")
            events.emit("USER_CONFIRMED_LOGGER_STARTED", {"dry_run": bool(args.dry_run)})
            dashboard.push(stage="logger_confirmed", state="running", message="User confirmed external logger is ready")
        elif not external_logger.manual_enabled():
            events.emit("LOGGER_READY", {"control": external_logger.control, "dry_run": bool(args.dry_run)})

        if args.dry_run:
            stopped_reason = "dry_run_complete"
            command_used = "dry-run: no inference executed"
            print("[DRY_RUN] preflight va path validation tamam shod; inference ejra nashod.")
            progress.update(stage="dry_run_complete", state="complete", message="Dry-run complete", stopped_reason=stopped_reason)
            dashboard.push(stage="dry_run_complete", state="complete", message="Dry-run complete", stopped_reason=stopped_reason)
            return 0

        noise_guard_enabled = not bool(args.disable_noise_guard) and (
            args.mode != "smoke" or bool(args.enable_noise_guard_in_smoke)
        )
        guard, guard_notes = enter_noise_guard(item, cfg, run_dir, enabled=noise_guard_enabled)
        notes.extend(guard_notes)

        start_time = utc_now_iso()
        start_mono = time.monotonic()
        if item["task"] not in {"detection", "segmentation"}:
            raise RuntimeError(f"Unsupported task: {item['task']}")
        command_used = build_pure_engine_command_template(resolve_path(item["engine_path"]), cfg, item, args)
        stopped_reason = run_pure_engine_loop(
            cfg=cfg,
            item=item,
            config_name=config_name,
            args=args,
            run_id=run_id,
            run_dir=run_dir,
            events=events,
            dashboard=dashboard,
            progress=progress,
            notes=notes,
        )
        events.emit("INFERENCE_END", {"stopped_reason": stopped_reason})
        stopped_with_error = str(stopped_reason).startswith("fatal_")
        progress.update(stage="inference_end", state=("error" if stopped_with_error else "complete"), message=f"Pure engine inference ended: {stopped_reason}", stopped_reason=stopped_reason)
        dashboard.push(
            stage="inference_end",
            state=("error" if stopped_with_error else "complete"),
            message=f"Pure engine inference ended: {stopped_reason}",
            stopped_reason=stopped_reason,
        )
        return 2 if stopped_with_error else 0
    except KeyboardInterrupt:
        stopped_reason = "manual_ctrl_c_power_bank_about_5_percent"
        events.emit("CTRL_C_RECEIVED", {"stage": "python_runner"})
        events.emit("INFERENCE_END", {"stopped_reason": stopped_reason})
        progress.update(stage="manual_stop", state="complete", message="Ctrl-C received; writing final pure engine summary", stopped_reason=stopped_reason)
        dashboard.push(
            stage="manual_stop",
            state="complete",
            message="Ctrl-C received; writing final pure engine summary",
            stopped_reason=stopped_reason,
        )
        return 0
    except Exception as exc:
        if stopped_reason == "not_started":
            stopped_reason = "fatal_error_before_inference"
        notes.append(f"{type(exc).__name__}: {exc}")
        events.emit("ERROR", {"stage": "runner", "error": notes[-1]})
        progress.update(stage="error", state="error", message=notes[-1], stopped_reason=stopped_reason)
        dashboard.push(stage="error", state="error", message=notes[-1], stopped_reason=stopped_reason)
        print(f"[ERROR] {notes[-1]}", file=sys.stderr, flush=True)
        return 2
    finally:
        if guard is not None:
            try:
                write_json(run_dir / "noise_guard_restore_summary.json", guard.restore())
            except Exception as exc:
                notes.append(f"NoiseGuard restore failed: {type(exc).__name__}: {exc}")
        if external_logger.auto_enabled():
            stop_info = external_logger.stop()
            try:
                events.emit("EXTERNAL_LOGGER_STOPPED", stop_info)
            except Exception:
                pass
        if sudo_keepalive is not None:
            sudo_status["keepalive"] = sudo_keepalive.stop()
            write_json(run_dir / "sudo_status.json", sudo_status)
        alarm_status = alarm_monitor.stop()
        internal_telemetry_status = telemetry.stop()
        heartbeat_status = heartbeat.stop()
        postrun_alarm_scan = scan_alarm_files(run_dir, cfg)
        alarm_status["postrun_scan"] = postrun_alarm_scan
        end_time = utc_now_iso()
        elapsed = (time.monotonic() - start_mono) if start_mono is not None else 0.0
        summary = build_summary(
            cfg=cfg,
            item=item,
            config_name=config_name,
            run_id=run_id,
            run_dir=run_dir,
            preflight=preflight,
            stopped_reason=stopped_reason,
            start_time=start_time,
            end_time=end_time,
            elapsed_seconds=elapsed,
            command_used=command_used,
            notes=notes,
            sudo_status=sudo_status,
            dashboard_status=dashboard.status(),
            external_logger_status=external_logger.status(),
            internal_telemetry_status={**internal_telemetry_status, "dashboard_heartbeat": heartbeat_status},
            alarm_status=alarm_status,
        )
        write_summary_files(run_dir, summary)
        events.emit("SUMMARY_WRITTEN", {"summary_json": str(run_dir / "summary.json")})
        dashboard.push(
            stage="summary_written",
            state=("error" if str(stopped_reason).startswith("fatal_") else "complete"),
            message="Pure engine summary written",
            processed_images=summary.get("total_images"),
            fps=summary.get("fps"),
            median_latency_ms=summary.get("median_latency_ms"),
            p95_latency_ms=summary.get("p95_latency_ms"),
            stopped_reason=stopped_reason,
        )
        events.emit("RUN_COMPLETE", {"stopped_reason": stopped_reason})
        dashboard.push(
            stage="run_complete",
            state=("error" if str(stopped_reason).startswith("fatal_") else "complete"),
            message=f"Run complete: {stopped_reason}",
            processed_images=summary.get("total_images"),
            stopped_reason=stopped_reason,
        )
        events.close()
        stderr_tee.stop()
        stdout_tee.stop()


if __name__ == "__main__":
    raise SystemExit(main())
