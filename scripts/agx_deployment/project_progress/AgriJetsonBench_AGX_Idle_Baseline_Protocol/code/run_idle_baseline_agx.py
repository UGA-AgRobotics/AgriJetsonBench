#!/usr/bin/env python3
"""AGX idle baseline runner for the AgriJetsonBench protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BOARD = "NVIDIA Jetson AGX Orin 64GB Developer Kit"
DEFAULT_ROOT = Path("${AGX_HOME}/project_progress/AgriJetsonBench_AGX_Idle_Baseline_Protocol")
DEFAULT_LOGGER_URL = "http://${LOGGER_DASHBOARD_HOST}:8000"
DEFAULT_NOISE_GUARD_REPO = Path("${AGX_HOME}/Benchmark/Jetson_benchmark_yolo11s")
PROCESS_CHECK_PATTERN = "python|trtexec|tensorrt|benchmark|yolo|segformer|rtdetr|bisenet|deeplab"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_cmd(cmd: Sequence[str], out_path: Optional[Path] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    start = utc_iso()
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        text = cp.stdout + cp.stderr
        if out_path is not None:
            header = "command: " + " ".join(cmd) + "\n" + f"captured_at: {start}\n\n"
            write_text(out_path, header + text)
        return {
            "cmd": list(cmd),
            "started_utc": start,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "ok": cp.returncode == 0,
        }
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}\n"
        if out_path is not None:
            header = "command: " + " ".join(cmd) + "\n" + f"captured_at: {start}\n\n"
            write_text(out_path, header + text)
        return {
            "cmd": list(cmd),
            "started_utc": start,
            "returncode": None,
            "stdout": "",
            "stderr": text,
            "ok": False,
        }


def import_noise_guard(repo_root: Path) -> Tuple[Any, Any]:
    repo_root = repo_root.expanduser().resolve()
    if not (repo_root / "jetson" / "noise_guard.py").exists():
        raise RuntimeError(f"noise guard module not found under {repo_root}")
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        from jetson.noise_guard import NoiseGuard, NoiseGuardConfig  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"could not import jetson.noise_guard from {repo_root}: {exc}") from exc
    return NoiseGuard, NoiseGuardConfig


def noise_guard_summary_fields(run_dir: Path) -> Dict[str, Any]:
    summary = read_json(run_dir / "noise_guard_summary.json", {}) or {}
    if not isinstance(summary, dict):
        summary = {}
    route = summary.get("preserve_route") or {}
    services = summary.get("services") or {}
    errors = summary.get("errors") or []
    service_stop_count = 0
    if isinstance(services, dict):
        service_stop_count = sum(1 for value in services.values() if isinstance(value, dict) and value.get("stopped"))
    return {
        "noise_guard_enabled": bool(summary.get("enabled", False)),
        "noise_guard_profile": summary.get("profile", ""),
        "noise_guard_restored": bool(summary.get("restored_utc")) and not bool(summary.get("active", False)),
        "noise_guard_preserve_interface": route.get("dev", "") if isinstance(route, dict) else "",
        "noise_guard_service_stop_count": service_stop_count,
        "noise_guard_errors": "; ".join(str(x) for x in errors) if isinstance(errors, list) else str(errors),
    }


def sudo_cmd(args: Sequence[str]) -> List[str]:
    return ["sudo", "-n", *args]


def http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", None)
        try:
            body: Any = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        ok = bool(body.get("ok", True)) if isinstance(body, dict) else True
        return {"ok": ok, "url": url, "status": status, "data": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "url": url, "status": exc.code, "error": body or str(exc)}
    except Exception as exc:
        return {"ok": False, "url": url, "status": None, "error": str(exc)}


def http_download(url: str, out_path: Path, timeout: int = 600) -> Dict[str, Any]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            out_path.write_bytes(resp.read())
        return {"ok": True, "url": url, "path": str(out_path), "bytes": out_path.stat().st_size}
    except Exception as exc:
        return {"ok": False, "url": url, "path": str(out_path), "error": str(exc)}


def parse_mode_map(text: str) -> Dict[str, str]:
    modes: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"POWER_MODEL:\s+ID=(\d+)\s+NAME=([A-Za-z0-9_+-]+)", line)
        if m:
            mode_id, name = m.group(1), m.group(2)
            normalized = name.upper().replace("MODE_", "")
            modes[normalized] = mode_id
            modes[name.upper()] = mode_id
    return modes


def discover_mode_ids(run_root: Path) -> Dict[str, str]:
    res = run_cmd(sudo_cmd(["nvpmodel", "-p", "--verbose"]), run_root / "nvpmodel_power_models.txt", timeout=30)
    text = str(res.get("stdout", "")) + str(res.get("stderr", ""))
    modes = parse_mode_map(text)
    if "15W" not in modes:
        modes["15W"] = "1"
    if "50W" not in modes:
        modes["50W"] = "3"
    return modes


def read_process_check(path: Path) -> Dict[str, Any]:
    cmd = ["bash", "-lc", f"ps aux | egrep '{PROCESS_CHECK_PATTERN}' || true"]
    return run_cmd(cmd, path, timeout=20)


def start_tegrastats(run_dir: Path) -> subprocess.Popen[str]:
    log_path = run_dir / "tegrastats_idle.log"
    cmd = sudo_cmd(["tegrastats", "--interval", "1000", "--logfile", str(log_path)])
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)
    except Exception:
        cmd = ["tegrastats", "--interval", "1000", "--logfile", str(log_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)
    write_text(run_dir / "tegrastats_pid.txt", f"{proc.pid}\n")
    write_text(run_dir / "tegrastats_command.txt", " ".join(cmd) + "\n")
    return proc


def stop_process(proc: Optional[subprocess.Popen[str]], timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=timeout)
        except Exception:
            pass


def sleep_with_status(seconds: float, label: str, run_id: str) -> None:
    end = time.monotonic() + seconds
    next_print = time.monotonic()
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        now = time.monotonic()
        if now >= next_print:
            print(f"[STATUS] run_id={run_id} stage={label} remaining_s={remaining:.1f}", flush=True)
            next_print = now + 30.0
        time.sleep(min(1.0, remaining))


def logger_active_for_run(health: Dict[str, Any], run_id: str) -> Tuple[bool, str]:
    if not health.get("ok"):
        return False, str(health.get("error", "health request failed"))
    data = health.get("data") or {}
    if not isinstance(data, dict):
        return False, "health response is not a dict"
    current = data.get("current_run") or {}
    devices = data.get("devices") or []
    selected_key = data.get("selected_device_key")
    selected_device: Dict[str, Any] = {}
    if isinstance(devices, list):
        for dev in devices:
            if isinstance(dev, dict) and dev.get("key") == selected_key:
                selected_device = dev
                break
    active = bool(current.get("active")) or bool(selected_device.get("external_logger_active"))
    seen_ids = {str(x) for x in (current.get("run_id"), selected_device.get("run_id")) if x}
    if not active:
        return False, "logger reachable but not active"
    if seen_ids and run_id not in seen_ids:
        return False, f"run id mismatch: {sorted(seen_ids)}"
    return True, "logger active"


def start_external_logger(run_dir: Path, run_id: str, logger_url: str, sample_hz: float, power_mode_label: str) -> Dict[str, Any]:
    logger_url = logger_url.rstrip("/")
    scheduled_start_ns = time.time_ns() + 2_000_000_000
    payload = {
        "run_id": run_id,
        "sample_hz": sample_hz,
        "start_wallclock_ns": scheduled_start_ns,
        "fast_temp_sensor": True,
        "device_key": "jetson-agx",
        "board_name": BOARD,
        "notes": {
            "runner": "agrijetsonbench_idle_baseline",
            "test_type": "idle_baseline",
            "power_mode_label": power_mode_label,
            "workload": "none",
            "timing_boundary": "settle_then_official_idle_window",
        },
    }
    write_json(run_dir / "external_logger_start_payload.json", payload)
    result = http_json("POST", logger_url + "/start", payload, timeout=20)
    write_json(run_dir / "external_logger_start_response.json", result)
    if not result.get("ok"):
        raise RuntimeError(f"could not start external logger: {result}")
    wait_s = max(0.0, (scheduled_start_ns - time.time_ns()) / 1e9 + 0.25)
    if wait_s:
        time.sleep(wait_s)
    health = http_json("GET", logger_url + "/health", timeout=5)
    write_json(run_dir / "external_logger_health_after_start.json", health)
    ok, reason = logger_active_for_run(health, run_id)
    if not ok:
        raise RuntimeError(f"external logger did not become active: {reason}")
    return result


def stop_external_logger(run_dir: Path, run_id: str, logger_url: str) -> Dict[str, Any]:
    logger_url = logger_url.rstrip("/")
    payload = {"run_id": run_id, "device_key": "jetson-agx"}
    result = http_json("POST", logger_url + "/stop", payload, timeout=60)
    write_json(run_dir / "external_logger_stop_response.json", result)
    archive_path = run_dir / "external_logger" / f"{run_id}.tar.gz"
    download = http_download(logger_url + f"/download/{run_id}", archive_path, timeout=600)
    write_json(run_dir / "external_logger_download_response.json", download)
    if download.get("ok"):
        extract_external_archive(run_dir, archive_path, run_id)
    return {"stop": result, "download": download}


def extract_external_archive(run_dir: Path, archive_path: Path, run_id: str) -> None:
    out_dir = run_dir / "external_logger" / "archive_extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(out_dir)
    except Exception as exc:
        write_text(run_dir / "external_logger" / "archive_extract_error.txt", f"{type(exc).__name__}: {exc}\n")
        return
    candidates = list(out_dir.rglob("external_measurements.csv"))
    if candidates:
        shutil.copy2(candidates[0], run_dir / "external_logger" / f"{run_id}_external_power.csv")
    meta = list(out_dir.rglob("run_meta.json"))
    if meta:
        shutil.copy2(meta[0], run_dir / "external_logger" / f"{run_id}_external_run_meta.json")


def float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def read_external_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted: Dict[str, Any] = dict(row)
            for key in ("mid_wallclock_ns", "read_end_ns", "power_w", "bus_voltage_v", "energy_joules"):
                converted[key] = float_or_none(row.get(key))
            rows.append(converted)
    return rows


def integrate_wh(rows: List[Dict[str, Any]]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for row in rows:
        t = row.get("mid_wallclock_ns")
        p = row.get("power_w")
        if t is not None and p is not None:
            pairs.append((float(t) / 1e9, float(p)))
    if len(pairs) < 2:
        return None
    pairs.sort()
    energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(pairs, pairs[1:]):
        dt = max(0.0, t1 - t0)
        energy_j += 0.5 * (p0 + p1) * dt
    return energy_j / 3600.0


def parse_tegrastats(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"available": False}
    max_temp: Optional[float] = None
    powers: List[float] = []
    lines = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines += 1
            temps = [float(m.group(1)) for m in re.finditer(r"@(-?\d+(?:\.\d+)?)C", line)]
            if temps:
                line_max = max(temps)
                max_temp = line_max if max_temp is None else max(max_temp, line_max)
            rails = [float(m.group(1)) / 1000.0 for m in re.finditer(r"\b[A-Z][A-Z0-9_]+\s+(\d+(?:\.\d+)?)mW", line)]
            if rails:
                powers.append(sum(rails))
    return {
        "available": lines > 0,
        "line_count": lines,
        "internal_max_temp_c": max_temp,
        "internal_avg_power_w_if_available": sum(powers) / len(powers) if powers else None,
    }


def summarize_run(run_dir: Path, run_id: str, power_mode_label: str, mode_id: str) -> Dict[str, Any]:
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    idle_start_ns = float(timing["idle_window_start_wallclock_ns"])
    idle_stop_ns = float(timing["idle_window_stop_wallclock_ns"])
    external_csv = run_dir / "external_logger" / f"{run_id}_external_power.csv"
    all_rows = read_external_rows(external_csv)
    rows = [
        row for row in all_rows
        if row.get("mid_wallclock_ns") is not None
        and idle_start_ns <= float(row["mid_wallclock_ns"]) <= idle_stop_ns
    ]
    powers = [float(row["power_w"]) for row in rows if row.get("power_w") is not None]
    voltages = [float(row["bus_voltage_v"]) for row in rows if row.get("bus_voltage_v") is not None]
    times = sorted(float(row["mid_wallclock_ns"]) / 1e9 for row in rows if row.get("mid_wallclock_ns") is not None)
    gaps = [b - a for a, b in zip(times, times[1:])]
    internal = parse_tegrastats(run_dir / "tegrastats_idle.log")
    duration_s = (idle_stop_ns - idle_start_ns) / 1e9
    voltage_ok = bool(voltages) and min(voltages) >= 18.0 and max(voltages) <= 21.5
    gap_count = sum(1 for gap in gaps if gap > 1.0)
    mode_ok = power_mode_label.upper().replace(" ", "") in (run_dir / "nvpmodel_short.txt").read_text(
        encoding="utf-8", errors="replace"
    ).upper().replace(" ", "")
    run_valid = bool(powers) and duration_s >= 295.0 and voltage_ok and gap_count <= 1 and internal.get("available") and mode_ok
    notes: List[str] = []
    if not powers:
        notes.append("external power rows missing in official idle window")
    if duration_s < 295.0:
        notes.append("official idle duration shorter than 295 seconds")
    if not voltage_ok:
        notes.append("voltage outside expected stable range")
    if gap_count > 1:
        notes.append("external logger gaps greater than 1 second")
    if not internal.get("available"):
        notes.append("tegrastats missing or empty")
    if not mode_ok:
        notes.append("nvpmodel short output did not match requested label")
    if not notes:
        notes.append("valid idle baseline")
    noise_guard = noise_guard_summary_fields(run_dir)
    summary = {
        "run_id": run_id,
        "board": BOARD,
        "power_mode_label": power_mode_label,
        "mode_id": mode_id,
        "duration_seconds": duration_s,
        "external_avg_power_w": sum(powers) / len(powers) if powers else None,
        "external_min_power_w": min(powers) if powers else None,
        "external_peak_power_w": max(powers) if powers else None,
        "external_energy_wh": integrate_wh(rows),
        "voltage_min_v": min(voltages) if voltages else None,
        "voltage_max_v": max(voltages) if voltages else None,
        "logger_gap_gt1s_count": gap_count,
        "external_samples_in_idle_window": len(rows),
        "internal_max_temp_c": internal.get("internal_max_temp_c"),
        "internal_avg_power_w_if_available": internal.get("internal_avg_power_w_if_available"),
        "jetson_clocks_status": first_nonempty_line(run_dir / "jetson_clocks_show.txt"),
        **noise_guard,
        "run_valid": run_valid,
        "notes": "; ".join(notes),
    }
    write_json(run_dir / "idle_baseline_summary.json", summary)
    write_summary_md(run_dir / "idle_baseline_summary.md", summary)
    return summary


def first_nonempty_line(path: Path) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("command:") and not stripped.startswith("captured_at:"):
            return stripped[:180]
    return "empty"


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Idle Baseline Summary",
        "",
        f"run_id: {summary['run_id']}",
        f"power_mode_label: {summary['power_mode_label']}",
        f"mode_id: {summary['mode_id']}",
        f"duration_seconds: {summary['duration_seconds']:.3f}",
        f"external_avg_power_w: {fmt(summary.get('external_avg_power_w'))}",
        f"external_min_power_w: {fmt(summary.get('external_min_power_w'))}",
        f"external_peak_power_w: {fmt(summary.get('external_peak_power_w'))}",
        f"external_energy_wh: {fmt(summary.get('external_energy_wh'))}",
        f"voltage_min_v: {fmt(summary.get('voltage_min_v'))}",
        f"voltage_max_v: {fmt(summary.get('voltage_max_v'))}",
        f"logger_gap_gt1s_count: {summary['logger_gap_gt1s_count']}",
        f"internal_max_temp_c: {fmt(summary.get('internal_max_temp_c'))}",
        f"internal_avg_power_w_if_available: {fmt(summary.get('internal_avg_power_w_if_available'))}",
        f"noise_guard_enabled: {summary.get('noise_guard_enabled')}",
        f"noise_guard_profile: {summary.get('noise_guard_profile')}",
        f"noise_guard_restored: {summary.get('noise_guard_restored')}",
        f"noise_guard_preserve_interface: {summary.get('noise_guard_preserve_interface')}",
        f"noise_guard_service_stop_count: {summary.get('noise_guard_service_stop_count')}",
        f"noise_guard_errors: {summary.get('noise_guard_errors')}",
        f"run_valid: {summary['run_valid']}",
        f"notes: {summary['notes']}",
        "",
    ]
    write_text(path, "\n".join(lines))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def csv_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    columns = summary_columns()
    return {key: summary.get(key, "") for key in columns}


def summary_columns() -> List[str]:
    return [
        "run_id",
        "board",
        "power_mode_label",
        "mode_id",
        "duration_seconds",
        "external_avg_power_w",
        "external_min_power_w",
        "external_peak_power_w",
        "external_energy_wh",
        "voltage_min_v",
        "voltage_max_v",
        "logger_gap_gt1s_count",
        "internal_max_temp_c",
        "internal_avg_power_w_if_available",
        "jetson_clocks_status",
        "noise_guard_enabled",
        "noise_guard_profile",
        "noise_guard_restored",
        "noise_guard_preserve_interface",
        "noise_guard_service_stop_count",
        "noise_guard_errors",
        "run_valid",
        "notes",
    ]


def write_table(root: Path, summaries: List[Dict[str, Any]]) -> None:
    table_dir = root / "paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "idle_baseline_power_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns())
        writer.writeheader()
        for summary in summaries:
            writer.writerow(csv_row(summary))
    write_json(table_dir / "idle_baseline_power_summary.json", summaries)
    md = ["# AGX Idle Baseline Power Summary", ""]
    md.append("| run_id | mode | P0_ext_W | min_W | peak_W | energy_Wh | voltage_min | voltage_max | gap_gt1s | max_temp_C | noise_guard | valid |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for s in summaries:
        guard_state = "on/restored" if s.get("noise_guard_enabled") and s.get("noise_guard_restored") else (
            "on/not_restored" if s.get("noise_guard_enabled") else "off"
        )
        md.append(
            "| {run_id} | {power_mode_label} | {p0} | {pmin} | {pmax} | {ewh} | {vmin} | {vmax} | {gap} | {temp} | {guard} | {valid} |".format(
                run_id=s["run_id"],
                power_mode_label=s["power_mode_label"],
                p0=fmt(s.get("external_avg_power_w")),
                pmin=fmt(s.get("external_min_power_w")),
                pmax=fmt(s.get("external_peak_power_w")),
                ewh=fmt(s.get("external_energy_wh")),
                vmin=fmt(s.get("voltage_min_v")),
                vmax=fmt(s.get("voltage_max_v")),
                gap=s.get("logger_gap_gt1s_count", ""),
                temp=fmt(s.get("internal_max_temp_c")),
                guard=guard_state,
                valid=s.get("run_valid", ""),
            )
        )
    write_text(table_dir / "idle_baseline_power_summary.md", "\n".join(md) + "\n")


def write_final_results(root: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = ["# AgriJetsonBench AGX Idle Baseline Results", ""]
    lines.append("| mode | run_id | P0 external avg W | min W | peak W | energy Wh | voltage min V | voltage max V | gap gt 1s | internal max temp C | noise guard | valid |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for s in summaries:
        guard_state = "on/restored" if s.get("noise_guard_enabled") and s.get("noise_guard_restored") else (
            "on/not_restored" if s.get("noise_guard_enabled") else "off"
        )
        lines.append(
            "| {mode} | {run_id} | {p0} | {pmin} | {pmax} | {ewh} | {vmin} | {vmax} | {gap} | {temp} | {guard} | {valid} |".format(
                mode=s.get("power_mode_label", ""),
                run_id=s.get("run_id", ""),
                p0=fmt(s.get("external_avg_power_w")),
                pmin=fmt(s.get("external_min_power_w")),
                pmax=fmt(s.get("external_peak_power_w")),
                ewh=fmt(s.get("external_energy_wh")),
                vmin=fmt(s.get("voltage_min_v")),
                vmax=fmt(s.get("voltage_max_v")),
                gap=s.get("logger_gap_gt1s_count", ""),
                temp=fmt(s.get("internal_max_temp_c")),
                guard=guard_state,
                valid=str(s.get("run_valid", "")).lower(),
            )
        )
    lines.extend(
        [
            "",
            "Main paper table:",
            "",
            "`paper_tables/idle_baseline_power_summary.csv`",
            "",
            "Final zip:",
            "",
            f"`agrijetsonbench_agx_idle_baselines_{utc_now().strftime('%Y%m%d')}.zip`",
            "",
            "Protocol notes:",
            "",
            "- Each official run uses a separate settle window and an idle-only timed window.",
            "- Noise guard is recorded per run as `noise_guard_summary.json`.",
            "- Existing old runs without `noise_guard_summary.json` are shown as `off` and should be rerun if a noise-guarded official baseline is required.",
            "",
        ]
    )
    write_text(root / "FINAL_IDLE_BASELINE_RESULTS.md", "\n".join(lines))


def write_status(root: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = ["status: complete", "", "completed_runs:"]
    for s in summaries:
        guard_state = "on/restored" if s.get("noise_guard_enabled") and s.get("noise_guard_restored") else (
            "on/not_restored" if s.get("noise_guard_enabled") else "off"
        )
        lines.extend(
            [
                f"{s.get('power_mode_label')} official idle baseline:",
                f"run_id: {s.get('run_id')}",
                f"run_valid: {str(s.get('run_valid')).lower()}",
                f"P0_ext_W: {fmt(s.get('external_avg_power_w'))}",
                f"noise_guard: {guard_state}",
                "",
            ]
        )
    lines.extend(
        [
            "final_outputs:",
            "paper_tables/idle_baseline_power_summary.csv",
            "paper_tables/idle_baseline_power_summary.json",
            "paper_tables/idle_baseline_power_summary.md",
            f"agrijetsonbench_agx_idle_baselines_{utc_now().strftime('%Y%m%d')}.zip",
            "",
            "validity_notes:",
            "Official idle windows target 300 seconds after the settle period.",
            "Valid runs require external logger data, stable voltage, tegrastats, and the requested nvpmodel mode.",
            "Noise-guarded runs additionally save noise_guard_config.json, noise_guard_enter_summary.json, noise_guard_restore_summary.json, and noise_guard_summary.json.",
            "",
        ]
    )
    write_text(root / "OFFICIAL_RUN_STATUS.txt", "\n".join(lines))


def write_artifact_manifest(root: Path) -> None:
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size})
    write_json(root / "artifact_manifest.json", manifest)
    with (root / "artifact_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest)


def write_key_sha256s(root: Path) -> None:
    key_files = [
        "artifact_manifest.csv",
        "artifact_manifest.json",
        "OFFICIAL_RUN_STATUS.txt",
        "FINAL_IDLE_BASELINE_RESULTS.md",
        "RUN_FULL_IDLE_BASELINE_COMMAND.txt",
        "RUN_15W_AFTER_REBOOT_COMMAND.txt",
        "RUN_NOISE_GUARD_IDLE_BASELINE_COMMANDS.txt",
        "code/run_idle_baseline_agx.py",
        "code/run_idle_baseline_agx_with_sudo.sh",
        "code/README_IDLE_BASELINE.md",
        "paper_tables/idle_baseline_power_summary.csv",
        "paper_tables/idle_baseline_power_summary.json",
        "paper_tables/idle_baseline_power_summary.md",
        f"agrijetsonbench_agx_idle_baselines_{utc_now().strftime('%Y%m%d')}.zip",
    ]
    for summary in load_existing_summaries(root):
        run_id = str(summary.get("run_id", ""))
        if run_id:
            key_files.append(f"idle_baselines/{run_id}/idle_baseline_summary.json")
            key_files.append(f"idle_baselines/{run_id}/noise_guard_summary.json")
    lines: List[str] = []
    for rel in key_files:
        path = root / rel
        if not path.exists() or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    write_text(root / "KEY_FILE_SHA256SUMS.txt", "\n".join(lines) + "\n")


def summary_sort_key(summary: Dict[str, Any]) -> Tuple[int, str]:
    label = str(summary.get("power_mode_label", ""))
    match = re.search(r"(\d+)", label)
    mode_watts = int(match.group(1)) if match else 999
    return mode_watts, str(summary.get("run_id", ""))


def load_existing_summaries(root: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in sorted((root / "idle_baselines").glob("*/idle_baseline_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            payload.update(noise_guard_summary_fields(path.parent))
            summaries.append(payload)
    return sorted(summaries, key=summary_sort_key)


def merge_summaries(root: Path, new_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_run_id: Dict[str, Dict[str, Any]] = {}
    for summary in load_existing_summaries(root) + list(new_summaries):
        run_id = str(summary.get("run_id", ""))
        if run_id:
            by_run_id[run_id] = summary
    return sorted(by_run_id.values(), key=summary_sort_key)


def write_metadata(
    run_dir: Path,
    run_id: str,
    power_mode_label: str,
    mode_id: str,
    settle_seconds: int,
    idle_seconds: int,
    noise_guard_enabled: bool,
    noise_guard_profile: str,
) -> None:
    payload = {
        "run_id": run_id,
        "board": BOARD,
        "power_mode_label": power_mode_label,
        "mode_id": mode_id,
        "test_type": "idle_baseline",
        "duration_seconds_target": idle_seconds,
        "settle_seconds": settle_seconds,
        "power_path": "UGREEN power bank -> 20V PD trigger -> external INA/Pc_package logger -> Jetson DC input",
        "external_logger": "ON",
        "internal_telemetry": "tegrastats interval 1000 ms",
        "workload": "none",
        "timing_boundary": "idle_only_no_inference",
        "noise_guard_enabled": noise_guard_enabled,
        "noise_guard_profile": noise_guard_profile,
        "notes": "No TensorRT, no Python inference, no benchmark workload during timed window.",
    }
    write_json(run_dir / "idle_baseline_metadata.json", payload)


class IdleBaselineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).expanduser().resolve()
        self.idle_root = self.root / "idle_baselines"
        self.logger_url = str(args.logger_url).rstrip("/")
        self.settle_seconds = int(args.settle_seconds)
        self.idle_seconds = int(args.idle_seconds)
        self.sample_hz = float(args.external_sample_hz)
        self.summaries: List[Dict[str, Any]] = []
        self.current_run_dir: Optional[Path] = None
        self.current_run_id: Optional[str] = None
        self.current_tegrastats: Optional[subprocess.Popen[str]] = None
        self.current_logger_started = False
        self.current_noise_guard: Optional[Any] = None
        self.stop_requested = False

    def event(self, run_dir: Path, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        append_jsonl(run_dir / "events.jsonl", {"timestamp_utc": utc_iso(), "event": event, "payload": payload or {}})

    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            print(f"[WARN] signal received: {signum}; stopping active logger and telemetry", flush=True)
            self.cleanup_active()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def cleanup_active(self) -> None:
        stop_process(self.current_tegrastats)
        self.current_tegrastats = None
        if self.current_logger_started and self.current_run_dir and self.current_run_id:
            try:
                stop_external_logger(self.current_run_dir, self.current_run_id, self.logger_url)
            except Exception as exc:
                write_text(self.current_run_dir / "cleanup_external_logger_error.txt", f"{type(exc).__name__}: {exc}\n")
        self.current_logger_started = False
        self.restore_noise_guard()

    def enter_noise_guard(self, run_dir: Path) -> None:
        if not self.args.noise_guard:
            write_json(
                run_dir / "noise_guard_summary.json",
                {
                    "enabled": False,
                    "active": False,
                    "profile": "",
                    "notes": ["noise guard disabled by CLI"],
                },
            )
            return
        NoiseGuard, NoiseGuardConfig = import_noise_guard(Path(self.args.noise_guard_repo))
        preserve_url = str(self.args.noise_guard_preserve_url or self.logger_url)
        config = NoiseGuardConfig(
            enabled=True,
            profile=self.args.noise_guard_profile,
            preserve_url=preserve_url,
            display_off=bool(self.args.noise_guard_display_off),
            stop_services=bool(self.args.noise_guard_stop_services),
            disable_wifi=bool(self.args.noise_guard_disable_wifi),
            disable_bluetooth=bool(self.args.noise_guard_disable_bluetooth),
        )
        write_json(
            run_dir / "noise_guard_config.json",
            {
                "enabled": True,
                "profile": config.profile,
                "preserve_url": config.preserve_url,
                "display_off": config.display_off,
                "stop_services": config.stop_services,
                "disable_wifi": config.disable_wifi,
                "disable_bluetooth": config.disable_bluetooth,
                "module_repo": str(Path(self.args.noise_guard_repo).expanduser().resolve()),
            },
        )
        guard = NoiseGuard(config)
        self.current_noise_guard = guard
        summary = guard.enter()
        write_json(run_dir / "noise_guard_enter_summary.json", summary)
        write_json(run_dir / "noise_guard_summary.json", summary)
        self.event(run_dir, "NOISE_GUARD_ENTERED", noise_guard_summary_fields(run_dir))

    def restore_noise_guard(self) -> None:
        if self.current_noise_guard is None or self.current_run_dir is None:
            return
        run_dir = self.current_run_dir
        try:
            summary = self.current_noise_guard.restore()
            write_json(run_dir / "noise_guard_restore_summary.json", summary)
            write_json(run_dir / "noise_guard_summary.json", summary)
            self.event(run_dir, "NOISE_GUARD_RESTORED", noise_guard_summary_fields(run_dir))
        except Exception as exc:
            write_text(run_dir / "noise_guard_restore_error.txt", f"{type(exc).__name__}: {exc}\n")
        finally:
            self.current_noise_guard = None

    def preflight(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        health = http_json("GET", self.logger_url + "/health", timeout=5)
        read_once = http_json("GET", self.logger_url + "/read_once", timeout=5)
        write_json(self.root / "preflight_logger_health.json", health)
        write_json(self.root / "preflight_logger_read_once.json", read_once)
        if not read_once.get("ok"):
            raise RuntimeError(f"logger read_once failed: {read_once}")
        reading = (read_once.get("data") or {}).get("reading") if isinstance(read_once.get("data"), dict) else {}
        voltage = float_or_none((reading or {}).get("bus_voltage_v"))
        power = float_or_none((reading or {}).get("power_w"))
        if voltage is None or voltage < 18.0 or power is None or power <= 0:
            raise RuntimeError(f"logger reading is not valid AGX input power: voltage={voltage} power={power}")

    def run_all(self) -> List[Dict[str, Any]]:
        if self.args.merge_only:
            self.root.mkdir(parents=True, exist_ok=True)
            self.finalize_package()
            return self.summaries
        self.install_signal_handlers()
        self.preflight()
        modes = discover_mode_ids(self.root)
        write_json(self.root / "detected_nvpmodel_modes.json", modes)
        for label in self.args.modes:
            key = label.upper().replace("MODE_", "")
            mode_id = str(self.args.mode_id_override.get(key) or modes.get(key) or modes.get("MODE_" + key) or "")
            if not mode_id:
                raise RuntimeError(f"could not resolve nvpmodel mode id for {label}")
            summary = self.run_one(label, mode_id)
            self.summaries.append(summary)
            write_table(self.root, merge_summaries(self.root, self.summaries))
        self.finalize_package()
        return self.summaries

    def run_one(self, power_mode_label: str, mode_id: str) -> Dict[str, Any]:
        run_id = f"agx_idle_{power_mode_label.lower().replace('w', 'w')}_{stamp()}"
        run_dir = self.idle_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = run_dir
        self.current_run_id = run_id
        print(f"[RUN] run_id={run_id}", flush=True)
        print(f"[RUN] run_dir={run_dir}", flush=True)
        write_metadata(
            run_dir,
            run_id,
            power_mode_label,
            mode_id,
            self.settle_seconds,
            self.idle_seconds,
            bool(self.args.noise_guard),
            str(self.args.noise_guard_profile),
        )
        self.event(run_dir, "RUN_CREATED", {"run_id": run_id, "power_mode_label": power_mode_label, "mode_id": mode_id})

        run_cmd(sudo_cmd(["nvpmodel", "-q", "--verbose"]), run_dir / "nvpmodel_before.txt", timeout=30)
        set_result = run_cmd(sudo_cmd(["nvpmodel", "-m", str(mode_id)]), run_dir / "nvpmodel_set.txt", timeout=40)
        if not set_result.get("ok"):
            raise RuntimeError(f"nvpmodel set failed for {power_mode_label}: {set_result}")
        time.sleep(2.0)
        run_cmd(sudo_cmd(["nvpmodel", "-q", "--verbose"]), run_dir / "nvpmodel_after.txt", timeout=30)
        run_cmd(sudo_cmd(["nvpmodel", "-q"]), run_dir / "nvpmodel_short.txt", timeout=20)
        run_cmd(["which", "jetson_clocks"], run_dir / "which_jetson_clocks.txt", timeout=10)
        run_cmd(sudo_cmd(["jetson_clocks", "--show"]), run_dir / "jetson_clocks_show.txt", timeout=30)
        run_cmd(["cat", "/etc/nv_tegra_release"], run_dir / "nv_tegra_release.txt", timeout=10)
        run_cmd(["free", "-m"], run_dir / "memory_before.txt", timeout=10)
        run_cmd(["df", "-h"], run_dir / "disk_before.txt", timeout=10)
        run_cmd(["uptime"], run_dir / "uptime_before.txt", timeout=10)
        run_cmd(["cat", "/proc/device-tree/model"], run_dir / "device_tree_model.txt", timeout=10)
        read_process_check(run_dir / "process_check_before.txt")

        print(f"[NOISE_GUARD] entering profile={self.args.noise_guard_profile}", flush=True)
        self.enter_noise_guard(run_dir)

        write_text(run_dir / "external_logger_started_at.txt", utc_iso() + "\n")
        start_external_logger(run_dir, run_id, self.logger_url, self.sample_hz, power_mode_label)
        self.current_logger_started = True

        write_text(run_dir / "internal_telemetry_started_at.txt", utc_iso() + "\n")
        self.current_tegrastats = start_tegrastats(run_dir)

        timing: Dict[str, Any] = {"run_id": run_id}
        write_text(run_dir / "settle_start.txt", utc_iso() + "\n")
        timing["settle_start_wallclock_ns"] = time.time_ns()
        sleep_with_status(self.settle_seconds, "settle", run_id)
        timing["settle_end_wallclock_ns"] = time.time_ns()
        write_text(run_dir / "settle_end.txt", utc_iso() + "\n")

        write_text(run_dir / "idle_window_start.txt", utc_iso() + "\n")
        timing["idle_window_start_wallclock_ns"] = time.time_ns()
        self.event(run_dir, "IDLE_WINDOW_START", {"wallclock_ns": timing["idle_window_start_wallclock_ns"]})
        sleep_with_status(self.idle_seconds, "official_idle", run_id)
        timing["idle_window_stop_wallclock_ns"] = time.time_ns()
        write_text(run_dir / "idle_window_stop.txt", utc_iso() + "\n")
        self.event(run_dir, "IDLE_WINDOW_STOP", {"wallclock_ns": timing["idle_window_stop_wallclock_ns"]})
        write_json(run_dir / "timing.json", timing)

        stop_process(self.current_tegrastats)
        self.current_tegrastats = None
        write_text(run_dir / "internal_telemetry_stopped_at.txt", utc_iso() + "\n")
        write_text(run_dir / "external_logger_stopped_at.txt", utc_iso() + "\n")
        stop_external_logger(run_dir, run_id, self.logger_url)
        self.current_logger_started = False
        read_process_check(run_dir / "process_check_after.txt")
        run_cmd(["free", "-m"], run_dir / "memory_after.txt", timeout=10)
        run_cmd(["uptime"], run_dir / "uptime_after.txt", timeout=10)
        self.restore_noise_guard()

        summary = summarize_run(run_dir, run_id, power_mode_label, mode_id)
        self.event(run_dir, "RUN_SUMMARIZED", summary)
        return summary

    def finalize_package(self) -> None:
        self.summaries = merge_summaries(self.root, self.summaries)
        write_table(self.root, self.summaries)
        write_final_results(self.root, self.summaries)
        write_status(self.root, self.summaries)
        zip_path = self.root / f"agrijetsonbench_agx_idle_baselines_{utc_now().strftime('%Y%m%d')}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted((self.root / "idle_baselines").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(self.root))
            for path in sorted((self.root / "paper_tables").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(self.root))
            for path in sorted((self.root / "code").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(self.root))
            for name in [
                "FINAL_IDLE_BASELINE_RESULTS.md",
                "OFFICIAL_RUN_STATUS.txt",
                "RUN_FULL_IDLE_BASELINE_COMMAND.txt",
                "RUN_15W_AFTER_REBOOT_COMMAND.txt",
                "RUN_NOISE_GUARD_IDLE_BASELINE_COMMANDS.txt",
                "AgriJetsonBench_AGX_Idle_Baseline_Protocol.pdf",
            ]:
                path = self.root / name
                if path.is_file():
                    zf.write(path, path.relative_to(self.root))
        write_artifact_manifest(self.root)
        write_key_sha256s(self.root)
        print(f"[ZIP] {zip_path}", flush=True)


def parse_mode_overrides(values: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--mode-id must look like 15W=1")
        key, val = value.split("=", 1)
        out[key.upper().replace("MODE_", "")] = val.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AGX idle baseline protocol.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--logger-url", default=DEFAULT_LOGGER_URL)
    parser.add_argument("--modes", nargs="+", default=["15W", "50W"], help="Power mode labels to run.")
    parser.add_argument("--mode-id", action="append", default=[], help="Override nvpmodel id, example: 15W=1")
    parser.add_argument("--settle-seconds", type=int, default=120)
    parser.add_argument("--idle-seconds", type=int, default=300)
    parser.add_argument("--external-sample-hz", type=float, default=10.0)
    parser.add_argument("--merge-only", action="store_true", help="Rebuild tables, manifest, and zip from existing run folders.")
    parser.add_argument("--noise-guard", dest="noise_guard", action="store_true", default=True, help="Enable quiet mode before each run starts.")
    parser.add_argument("--no-noise-guard", dest="noise_guard", action="store_false", help="Disable quiet mode.")
    parser.add_argument("--noise-guard-repo", default=str(DEFAULT_NOISE_GUARD_REPO), help="Repo root that contains jetson/noise_guard.py.")
    parser.add_argument("--noise-guard-profile", choices=["safe", "aggressive"], default="safe")
    parser.add_argument("--noise-guard-preserve-url", default=None, help="URL or host that must stay reachable. Defaults to --logger-url.")
    parser.add_argument("--no-noise-guard-display-off", dest="noise_guard_display_off", action="store_false", default=True)
    parser.add_argument("--no-noise-guard-stop-services", dest="noise_guard_stop_services", action="store_false", default=True)
    parser.add_argument("--no-noise-guard-disable-wifi", dest="noise_guard_disable_wifi", action="store_false", default=True)
    parser.add_argument("--no-noise-guard-disable-bluetooth", dest="noise_guard_disable_bluetooth", action="store_false", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.modes = [str(x).upper().replace("MODE_", "") for x in args.modes]
    args.mode_id_override = parse_mode_overrides(args.mode_id)
    runner = IdleBaselineRunner(args)
    try:
        summaries = runner.run_all()
    except Exception as exc:
        runner.cleanup_active()
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print("[DONE] idle baseline summaries", flush=True)
    for summary in summaries:
        print(
            f"[RESULT] {summary['run_id']} mode={summary['power_mode_label']} "
            f"P0_ext_W={fmt(summary.get('external_avg_power_w'))} "
            f"noise_guard={bool(summary.get('noise_guard_enabled'))}/{bool(summary.get('noise_guard_restored'))} "
            f"valid={summary.get('run_valid')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
