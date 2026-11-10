#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from noise_guard import NoiseGuard, NoiseGuardConfig


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_idle_baseline.json"
IDLE_DIR = ROOT_DIR / "idle_baselines"
PAPER_TABLES_DIR = ROOT_DIR / "paper_tables"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
DIAGNOSTICS_DIR = ROOT_DIR / "diagnostics"
NOISE_GUARD_DIR = ROOT_DIR / "noise_guard"
REBOOT_STATE_DIR = ROOT_DIR / "reboot_state"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def local_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_capture(cmd: List[str], timeout: int = 30, cwd: Optional[Path] = None) -> Dict[str, Any]:
    started = utc_now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "started_utc": started,
            "ended_utc": utc_now(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": (exc.stdout or "") + "\n[TIMEOUT]\n",
            "started_utc": started,
            "ended_utc": utc_now(),
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "returncode": 1,
            "stdout": f"{type(exc).__name__}: {exc}\n",
            "started_utc": started,
            "ended_utc": utc_now(),
        }


def save_command(path: Path, cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    result = run_capture(cmd, timeout=timeout)
    body = (
        f"[COMMAND] {' '.join(cmd)}\n"
        f"[START_UTC] {result['started_utc']}\n"
        "[OUTPUT]\n"
        f"{result['stdout']}\n"
        f"[EXIT_STATUS] {result['returncode']}\n"
        f"[END_UTC] {result['ended_utc']}\n"
    )
    write_text(path, body)
    return result


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "raw": raw}


def http_download(url: str, dst: Path, timeout: int = 120) -> Dict[str, Any]:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        dst.write_bytes(data)
        return {"ok": True, "url": url, "path": str(dst), "size_bytes": len(data)}
    except Exception as exc:
        return {"ok": False, "url": url, "path": str(dst), "error": f"{type(exc).__name__}: {exc}"}


def safe_extract_tar(tar_path: Path, dst: Path) -> Dict[str, Any]:
    if not tar_path.exists():
        return {"ok": False, "error": "archive_missing", "path": str(tar_path)}
    dst.mkdir(parents=True, exist_ok=True)
    root = dst.resolve()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                target = (dst / member.name).resolve()
                if not str(target).startswith(str(root)):
                    raise RuntimeError(f"unsafe_tar_member={member.name}")
            try:
                tar.extractall(dst, filter="data")
            except TypeError:
                tar.extractall(dst)
        return {"ok": True, "archive": str(tar_path), "extract_dir": str(dst)}
    except Exception as exc:
        return {"ok": False, "archive": str(tar_path), "extract_dir": str(dst), "error": f"{type(exc).__name__}: {exc}"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_nvpmodel_modes(conf_text: str) -> Dict[str, str]:
    modes: Dict[str, str] = {}
    for match in re.finditer(r"<\s*POWER_MODEL\s+ID\s*=\s*(\d+)\s+NAME\s*=\s*([^>]+?)\s*>", conf_text):
        modes[match.group(2).strip()] = match.group(1).strip()
    return modes


def parse_current_nvpmodel(text: str) -> Tuple[str, str]:
    match = re.search(r"NV Power Mode:\s*([^\n]+)\n\s*(\d+)", text)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def read_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def get_current_power_mode() -> Dict[str, Any]:
    result = run_capture(["sudo", "-n", "nvpmodel", "-q"], timeout=10)
    label, mode_id = parse_current_nvpmodel(result.get("stdout", ""))
    return {
        "label": label,
        "mode_id": mode_id,
        "raw": result.get("stdout", ""),
        "returncode": result.get("returncode"),
    }


def reboot_state_path(power_mode_label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", power_mode_label)
    return REBOOT_STATE_DIR / f"pending_power_mode_{safe}.json"


def confirm_or_prepare_reboot_for_power_mode(
    power_mode_label: str,
    mode_id: str,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    if not args.require_reboot_after_mode_set:
        return False, {"required": False}

    REBOOT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    boot_id = read_boot_id()
    state_path = reboot_state_path(power_mode_label)
    current = get_current_power_mode()
    if state_path.exists():
        state = load_json(state_path)
        prepared_boot_id = state.get("prepared_boot_id")
        if (
            state.get("target_power_mode_label") == power_mode_label
            and str(state.get("target_mode_id")) == str(mode_id)
            and prepared_boot_id
            and prepared_boot_id != boot_id
        ):
            if current.get("mode_id") == str(mode_id):
                confirmed = dict(state)
                confirmed.update(
                    {
                        "confirmed_after_reboot": True,
                        "confirmed_boot_id": boot_id,
                        "confirmed_utc": utc_now(),
                        "current_power_mode_after_reboot": current,
                    }
                )
                confirmed_path = REBOOT_STATE_DIR / f"confirmed_power_mode_{power_mode_label.lower()}_{local_stamp()}.json"
                write_json(confirmed_path, confirmed)
                state_path.unlink(missing_ok=True)
                print(
                    "[POWER_MODE_REBOOT] confirmed "
                    f"mode={power_mode_label} mode_id={mode_id} "
                    f"boot_id={boot_id} state={confirmed_path}",
                    flush=True,
                )
                return False, {
                    "required": True,
                    "confirmed_after_reboot": True,
                    "confirmed_state_path": str(confirmed_path),
                    "prepared_state_path": str(state_path),
                }
            print(
                "[ERROR] Power mode reboot marker exists, but current mode does not match target. "
                f"current={current.get('label')}:{current.get('mode_id')} target={power_mode_label}:{mode_id}",
                file=sys.stderr,
                flush=True,
            )
            return True, {"required": True, "error": "post_reboot_power_mode_mismatch", "current": current}

    set_result = run_capture(["sudo", "-n", "nvpmodel", "-m", str(mode_id)], timeout=30)
    verify = get_current_power_mode()
    state = {
        "target_power_mode_label": power_mode_label,
        "target_mode_id": str(mode_id),
        "prepared_utc": utc_now(),
        "prepared_boot_id": boot_id,
        "current_power_mode_before_reboot": current,
        "set_result": set_result,
        "current_power_mode_after_set_before_reboot": verify,
        "resume_command": f"bash code/run_all_idle_baselines.sh --mode {power_mode_label}",
        "note": "Power mode was set. Reboot is required before the official idle baseline run.",
    }
    write_json(state_path, state)
    print(
        "[POWER_MODE_REBOOT] prepared "
        f"mode={power_mode_label} mode_id={mode_id} state={state_path}",
        flush=True,
    )
    if set_result.get("returncode") != 0:
        print("[ERROR] nvpmodel set failed; reboot not issued.", file=sys.stderr, flush=True)
        return True, {"required": True, "error": "nvpmodel_set_failed", "state_path": str(state_path)}
    print("[POWER_MODE_REBOOT] issuing sudo -n reboot now. After reboot, run the same command again:", flush=True)
    print(f"bash code/run_all_idle_baselines.sh --mode {power_mode_label}", flush=True)
    reboot_result = run_capture(["sudo", "-n", "reboot"], timeout=5)
    if reboot_result.get("returncode") != 0:
        print(
            "[ERROR] reboot command failed non-interactively. Check privileged-command policy, then run the same command again.",
            file=sys.stderr,
            flush=True,
        )
        write_json(REBOOT_STATE_DIR / f"reboot_failed_{power_mode_label.lower()}_{local_stamp()}.json", reboot_result)
        return True, {"required": True, "error": "reboot_command_failed", "state_path": str(state_path)}
    return True, {"required": True, "reboot_issued": True, "state_path": str(state_path)}


def read_board_model() -> str:
    path = Path("/proc/device-tree/model")
    try:
        return path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def process_check(run_dir: Path, name: str) -> Dict[str, Any]:
    result = save_command(
        run_dir / name,
        ["bash", "-lc", "ps aux | egrep 'python|trtexec|tensorrt|benchmark|yolo|segformer|rtdetr|bisenet|deeplab' || true"],
        timeout=15,
    )
    hits: List[str] = []
    current_pid = str(os.getpid())
    for line in result.get("stdout", "").splitlines():
        lower = line.lower()
        if (
            "egrep" in lower
            or "grep -e" in lower
            or "grep -e" in lower.replace("'", "")
            or "grep" in lower and "python|trtexec|tensorrt|benchmark|yolo|segformer|rtdetr|bisenet|deeplab" in lower
            or "local_run" in lower
            or "run_idle_baseline_protocol.py" in lower
            or current_pid in line
        ):
            continue
        if any(token in lower for token in ["trtexec", "tensorrt", "benchmark", "yolo", "segformer", "rtdetr", "bisenet", "deeplab"]):
            hits.append(line)
    return {"path": str(run_dir / name), "hits": hits, "hit_count": len(hits)}


def logger_preflight(logger_url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"logger_url": logger_url, "ok": False}
    try:
        out["health"] = http_json("GET", logger_url.rstrip("/") + "/health", timeout=8)
    except Exception as exc:
        out["health"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        out["read_once"] = http_json("GET", logger_url.rstrip("/") + "/read_once", timeout=8)
    except Exception as exc:
        out["read_once"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    current_run = ((out.get("health") or {}).get("current_run") or {})
    out["current_run_active"] = bool(current_run.get("active")) if isinstance(current_run, dict) else bool(current_run)
    reading = ((out.get("read_once") or {}).get("reading") or {})
    voltage = safe_float(reading.get("bus_voltage_v") or reading.get("bus_voltage"))
    power = safe_float(reading.get("power_w"))
    current = safe_float(reading.get("current_a"))
    out["voltage_v"] = voltage
    out["current_a"] = current
    out["power_w"] = power
    out["ok"] = (
        bool((out.get("health") or {}).get("ok"))
        and bool((out.get("read_once") or {}).get("ok"))
        and not out["current_run_active"]
        and voltage is not None
        and 15.0 <= voltage <= 25.0
        and power is not None
        and power > 0
    )
    return out


def start_external_logger(cfg: Dict[str, Any], run_id: str, run_dir: Path, power_mode_label: str) -> Tuple[Dict[str, Any], int]:
    logger_url = str(cfg["logger_url"]).rstrip("/")
    start_wallclock_ns = time.time_ns() + int(float(cfg.get("external_logger_start_delay_seconds", 2.0)) * 1e9)
    payload = {
        "run_id": run_id,
        "sample_hz": float(cfg.get("sample_hz", 10.0)),
        "start_wallclock_ns": start_wallclock_ns,
        "fast_temp_sensor": True,
        "notes": {
            "device_key": cfg.get("device_key", "jetson-nano"),
            "device_label": cfg.get("board_display_name", "NVIDIA Jetson Orin Nano Super Developer Kit"),
            "board_name": cfg.get("board", "nano_super"),
            "run_id": run_id,
            "task": "idle_baseline",
            "model": "none",
            "precision": "none",
            "input_size": "none",
            "batch_size": 0,
            "power_mode_label": power_mode_label,
            "run_variant": "agrijetsonbench_idle_baseline",
            "timing_boundary": "idle_only_no_inference",
            "measurement_backend": "external_ina_pc_package",
            "power_source": cfg.get("power_path"),
        },
    }
    response = http_json("POST", logger_url + "/start", payload, timeout=20)
    write_json(run_dir / "external_logger_start.json", {"payload": payload, "response": response})
    if not response.get("ok"):
        raise RuntimeError(f"External logger /start failed: {response}")
    while time.time_ns() < start_wallclock_ns:
        time.sleep(0.02)
    return {"payload": payload, "response": response}, start_wallclock_ns


def stop_external_logger(cfg: Dict[str, Any], run_id: str, run_dir: Path) -> Dict[str, Any]:
    logger_url = str(cfg["logger_url"]).rstrip("/")
    payload = {"run_id": run_id, "device_key": cfg.get("device_key", "jetson-nano"), "board_name": cfg.get("board", "nano_super")}
    try:
        response = http_json("POST", logger_url + "/stop", payload, timeout=60)
    except Exception as exc:
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    write_json(run_dir / "external_logger_stop.json", {"payload": payload, "response": response})
    archive_path = run_dir / "external_logger_archive.tar.gz"
    download = http_download(logger_url + f"/download/{run_id}", archive_path, timeout=120)
    write_json(run_dir / "external_logger_download.json", download)
    extract = {"ok": False, "skipped": True}
    if download.get("ok"):
        extract = safe_extract_tar(archive_path, run_dir / "external_logger_archive")
    write_json(run_dir / "external_logger_extract.json", extract)
    return {"stop": response, "download": download, "extract": extract}


def find_external_csv(run_dir: Path) -> Optional[Path]:
    candidates = list((run_dir / "external_logger_archive").rglob("external_measurements.csv"))
    return candidates[0] if candidates else None


def copy_external_csv(run_dir: Path, run_id: str) -> Optional[Path]:
    src = find_external_csv(run_dir)
    if not src:
        return None
    dst = run_dir / "external_logger" / f"{run_id}_external_power.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def parse_external_metrics(run_dir: Path, idle_start_ns: int, idle_stop_ns: int) -> Dict[str, Any]:
    csv_path = find_external_csv(run_dir)
    out: Dict[str, Any] = {
        "external_csv_path": str(csv_path) if csv_path else None,
        "samples_total": 0,
        "samples_in_idle_window": 0,
        "external_avg_power_w": None,
        "external_min_power_w": None,
        "external_peak_power_w": None,
        "external_energy_wh": None,
        "voltage_min_v": None,
        "voltage_max_v": None,
        "logger_gap_gt1s_count": None,
        "logger_max_gap_s": None,
    }
    if not csv_path:
        return out
    powers: List[float] = []
    voltages: List[float] = []
    energies: List[float] = []
    wall_ns: List[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out["samples_total"] += 1
            ns = safe_int(row.get("mid_wallclock_ns") or row.get("sched_wallclock_ns"))
            if ns is None or ns < idle_start_ns or ns > idle_stop_ns:
                continue
            power = safe_float(row.get("power_w"))
            voltage = safe_float(row.get("bus_voltage_v") or row.get("bus_voltage"))
            energy = safe_float(row.get("energy_joules"))
            out["samples_in_idle_window"] += 1
            wall_ns.append(ns)
            if power is not None:
                powers.append(power)
            if voltage is not None:
                voltages.append(voltage)
            if energy is not None:
                energies.append(energy)
    if powers:
        out["external_avg_power_w"] = mean(powers)
        out["external_min_power_w"] = min(powers)
        out["external_peak_power_w"] = max(powers)
    if voltages:
        out["voltage_min_v"] = min(voltages)
        out["voltage_max_v"] = max(voltages)
    if len(energies) >= 2:
        out["external_energy_wh"] = max(energies[-1] - energies[0], 0.0) / 3600.0
    elif powers:
        duration_s = max((idle_stop_ns - idle_start_ns) / 1e9, 0.0)
        out["external_energy_wh"] = (mean(powers) or 0.0) * duration_s / 3600.0
    if len(wall_ns) >= 2:
        gaps = [(b - a) / 1e9 for a, b in zip(wall_ns, wall_ns[1:]) if b >= a]
        out["logger_gap_gt1s_count"] = sum(1 for gap in gaps if gap > 1.0)
        out["logger_max_gap_s"] = max(gaps) if gaps else None
    return out


def parse_tegrastats(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "line_count": 0,
        "internal_max_temp_c": None,
        "internal_avg_power_w_if_available": None,
        "internal_peak_power_w_if_available": None,
    }
    if not path.exists():
        return out
    temps: List[float] = []
    powers_w: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        out["line_count"] += 1
        for match in re.finditer(r"@([0-9]+(?:\.[0-9]+)?)C", line):
            val = safe_float(match.group(1))
            if val is not None:
                temps.append(val)
        match = re.search(r"VDD_IN\s+([0-9]+)mW", line)
        if match:
            val = safe_float(match.group(1))
            if val is not None:
                powers_w.append(val / 1000.0)
    if temps:
        out["internal_max_temp_c"] = max(temps)
    if powers_w:
        out["internal_avg_power_w_if_available"] = mean(powers_w)
        out["internal_peak_power_w_if_available"] = max(powers_w)
    return out


def sleep_with_progress(seconds: int, label: str, interval: int = 30) -> None:
    end = time.monotonic() + seconds
    next_print = time.monotonic()
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            return
        if now >= next_print:
            elapsed = seconds - max(remaining, 0)
            print(f"[{label}] elapsed={elapsed:.0f}s remaining={remaining:.0f}s", flush=True)
            next_print = now + interval
        time.sleep(min(1.0, max(remaining, 0.1)))


def start_tegrastats(run_dir: Path, interval_ms: int) -> subprocess.Popen[str]:
    write_text(run_dir / "internal_telemetry_started_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")
    cmd = ["sudo", "-n", "tegrastats", "--interval", str(interval_ms), "--logfile", str(run_dir / "tegrastats_idle.log")]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    write_text(run_dir / "tegrastats_pid.txt", str(proc.pid) + "\n")
    return proc


def stop_tegrastats(proc: Optional[subprocess.Popen[str]], run_dir: Path) -> None:
    logfile = run_dir / "tegrastats_idle.log"
    if proc and proc.poll() is None:
        subprocess.run(["sudo", "-n", "kill", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    # Kill only a leftover tegrastats instance tied to this exact logfile, if sudo left the child alive.
    subprocess.run(["sudo", "-n", "pkill", "-f", str(logfile)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_text(run_dir / "internal_telemetry_stopped_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")


def set_power_mode(run_dir: Path, mode_id: str) -> None:
    save_command(run_dir / "nvpmodel_before.txt", ["sudo", "-n", "nvpmodel", "-q", "--verbose"], timeout=20)
    current = get_current_power_mode()
    if current.get("mode_id") == str(mode_id):
        write_text(
            run_dir / "nvpmodel_set.txt",
            "[SKIPPED]\n"
            f"Current nvpmodel mode already matches requested mode_id={mode_id}.\n"
            "No nvpmodel -m command was issued in this post-reboot measurement phase.\n"
            f"[CURRENT]\n{current.get('raw', '')}\n",
        )
    else:
        save_command(run_dir / "nvpmodel_set.txt", ["sudo", "-n", "nvpmodel", "-m", mode_id], timeout=30)
        time.sleep(1.0)
    save_command(run_dir / "nvpmodel_after.txt", ["sudo", "-n", "nvpmodel", "-q", "--verbose"], timeout=20)
    save_command(run_dir / "nvpmodel_short.txt", ["sudo", "-n", "nvpmodel", "-q"], timeout=10)


def capture_static_state(run_dir: Path) -> None:
    save_command(run_dir / "jetson_clocks_show.txt", ["sudo", "-n", "jetson_clocks", "--show"], timeout=20)
    save_command(run_dir / "nv_tegra_release.txt", ["bash", "-lc", "cat /etc/nv_tegra_release 2>/dev/null || true"], timeout=10)
    save_command(run_dir / "memory_before.txt", ["free", "-m"], timeout=10)
    save_command(run_dir / "disk_before.txt", ["df", "-h"], timeout=10)
    save_command(run_dir / "uptime_before.txt", ["uptime"], timeout=10)
    save_command(run_dir / "board_model.txt", ["bash", "-lc", "tr -d '\\0' < /proc/device-tree/model 2>/dev/null || true"], timeout=10)


def write_metadata(run_dir: Path, run_id: str, cfg: Dict[str, Any], power_mode_label: str, mode_id: str, settle_s: int, idle_s: int) -> None:
    metadata = {
        "run_id": run_id,
        "board": cfg.get("board_display_name", "NVIDIA Jetson Orin Nano Super Developer Kit"),
        "board_label": cfg.get("board", "nano_super"),
        "power_mode_label": power_mode_label,
        "mode_id": mode_id,
        "test_type": "idle_baseline",
        "duration_seconds_target": idle_s,
        "settle_seconds": settle_s,
        "power_path": cfg.get("power_path"),
        "external_logger": "ON",
        "external_logger_url": cfg.get("logger_url"),
        "internal_telemetry": f"tegrastats interval {cfg.get('tegrastats_interval_ms', 1000)} ms",
        "workload": "none",
        "timing_boundary": "idle_only_no_inference",
        "notes": "No TensorRT, no Python inference, no benchmark workload during timed window. Python control runner only sleeps and records telemetry.",
    }
    write_json(run_dir / "idle_baseline_metadata.json", metadata)


def evaluate_validity(
    run_dir: Path,
    power_mode_label: str,
    mode_id: str,
    external: Dict[str, Any],
    tegra: Dict[str, Any],
    proc_before: Dict[str, Any],
    proc_after: Dict[str, Any],
    duration_s: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    board = read_board_model()
    if "Orin Nano" not in board and "Jetson Orin Nano" not in board:
        reasons.append("wrong_or_unknown_board")
    short = (run_dir / "nvpmodel_short.txt").read_text(encoding="utf-8", errors="replace") if (run_dir / "nvpmodel_short.txt").exists() else ""
    current_label, current_id = parse_current_nvpmodel(short)
    if current_label != power_mode_label or current_id != str(mode_id):
        reasons.append(f"nvpmodel_mismatch_current={current_label}:{current_id}_expected={power_mode_label}:{mode_id}")
    if (safe_float(external.get("external_avg_power_w")) or 0.0) <= 0:
        reasons.append("missing_external_avg_power")
    if (external.get("samples_in_idle_window") or 0) <= 0:
        reasons.append("no_external_samples_in_idle_window")
    vmin = safe_float(external.get("voltage_min_v"))
    vmax = safe_float(external.get("voltage_max_v"))
    if vmin is None or vmax is None or vmin < 15.0 or vmax > 25.0:
        reasons.append("external_voltage_not_stable_20v_range")
    gap_count = external.get("logger_gap_gt1s_count")
    if gap_count is None or int(gap_count) > 0:
        reasons.append("logger_gap_gt1s_present")
    if not tegra.get("exists") or int(tegra.get("line_count") or 0) <= 0:
        reasons.append("tegrastats_missing")
    if proc_before.get("hit_count") or proc_after.get("hit_count"):
        reasons.append("possible_inference_or_benchmark_process_seen")
    if duration_s < 295:
        reasons.append("official_idle_window_too_short")
    if safe_float(tegra.get("internal_max_temp_c")) and float(tegra["internal_max_temp_c"]) >= 95.0:
        reasons.append("overtemperature_warning")
    return (len(reasons) == 0), reasons


def create_run_summary(
    run_dir: Path,
    run_id: str,
    cfg: Dict[str, Any],
    power_mode_label: str,
    mode_id: str,
    idle_start_ns: int,
    idle_stop_ns: int,
    proc_before: Dict[str, Any],
    proc_after: Dict[str, Any],
    external_status: Dict[str, Any],
) -> Dict[str, Any]:
    external_csv_copy = copy_external_csv(run_dir, run_id)
    external = parse_external_metrics(run_dir, idle_start_ns, idle_stop_ns)
    if external_csv_copy:
        external["external_csv_copy_path"] = str(external_csv_copy)
    tegra = parse_tegrastats(run_dir / "tegrastats_idle.log")
    idle_duration_s = (idle_stop_ns - idle_start_ns) / 1e9
    valid, reasons = evaluate_validity(run_dir, power_mode_label, mode_id, external, tegra, proc_before, proc_after, idle_duration_s)
    clocks = (run_dir / "jetson_clocks_show.txt").read_text(encoding="utf-8", errors="replace") if (run_dir / "jetson_clocks_show.txt").exists() else ""
    if "FreqOverride=1" in clocks:
        jetson_clocks_status = "enabled_or_freq_override_1"
    elif "FreqOverride=0" in clocks:
        jetson_clocks_status = "not_fully_locked_freq_override_0"
    else:
        jetson_clocks_status = "captured_unknown"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "board": cfg.get("board_display_name", "NVIDIA Jetson Orin Nano Super Developer Kit"),
        "board_label": cfg.get("board", "nano_super"),
        "power_mode_label": power_mode_label,
        "mode_id": str(mode_id),
        "duration_seconds": idle_duration_s,
        "settle_seconds": int(cfg.get("settle_seconds", 120)),
        "external": external,
        "internal_telemetry": tegra,
        "process_check_before": proc_before,
        "process_check_after": proc_after,
        "external_logger_status": external_status,
        "jetson_clocks_status": jetson_clocks_status,
        "run_valid": valid,
        "validity_reasons": reasons,
        "summary_created_utc": utc_now(),
    }
    write_json(run_dir / "idle_baseline_summary.json", summary)
    interpretation = [
        "Process-check interpretation",
        "",
        f"before_hit_count={proc_before.get('hit_count')}",
        f"after_hit_count={proc_after.get('hit_count')}",
        "",
        "The saved raw process_check files intentionally preserve all regex matches.",
        "System Python daemons, nvpmodel_indicator, grep/egrep, this idle-baseline runner, and the interactive terminal process are not inference workloads.",
        "The filtered hit_count fields are the validity criteria used by this runner.",
    ]
    if proc_before.get("hits") or proc_after.get("hits"):
        interpretation += ["", "Filtered inference/benchmark-like hits were detected; see idle_baseline_summary.json."]
    else:
        interpretation += ["", "No filtered TensorRT/trtexec/YOLO/SegFormer/RT-DETR/BiSeNet/benchmark workload was detected."]
    write_text(run_dir / "process_check_interpretation.txt", "\n".join(interpretation) + "\n")
    md = [
        f"# Idle Baseline Summary: {run_id}",
        "",
        f"- board: `{summary['board']}`",
        f"- power_mode_label: `{power_mode_label}`",
        f"- mode_id: `{mode_id}`",
        f"- duration_seconds: `{idle_duration_s:.3f}`",
        f"- external_avg_power_w: `{external.get('external_avg_power_w')}`",
        f"- external_min_power_w: `{external.get('external_min_power_w')}`",
        f"- external_peak_power_w: `{external.get('external_peak_power_w')}`",
        f"- external_energy_wh: `{external.get('external_energy_wh')}`",
        f"- voltage_min_v: `{external.get('voltage_min_v')}`",
        f"- voltage_max_v: `{external.get('voltage_max_v')}`",
        f"- logger_gap_gt1s_count: `{external.get('logger_gap_gt1s_count')}`",
        f"- internal_max_temp_c: `{tegra.get('internal_max_temp_c')}`",
        f"- internal_avg_power_w_if_available: `{tegra.get('internal_avg_power_w_if_available')}`",
        f"- jetson_clocks_status: `{jetson_clocks_status}`",
        f"- run_valid: `{valid}`",
        f"- validity_reasons: `{'; '.join(reasons) if reasons else ''}`",
    ]
    write_text(run_dir / "summary.md", "\n".join(md) + "\n")
    return summary


def run_one_mode(cfg: Dict[str, Any], power_mode_label: str, mode_id: str, settle_s: int, idle_s: int, run_label: str) -> Dict[str, Any]:
    suffix = f"_{run_label}" if run_label else ""
    run_id = f"nano_super_idle_{power_mode_label.lower()}{suffix}_{local_stamp()}"
    base_dir = DIAGNOSTICS_DIR / "smoke_runs" if run_label else IDLE_DIR
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN_ID] {run_id}", flush=True)
    print(f"[RUN_DIR] {run_dir}", flush=True)
    print(f"[TARGET] idle baseline power_mode={power_mode_label} mode_id={mode_id} settle={settle_s}s idle={idle_s}s", flush=True)

    write_text(run_dir / "run_started_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")
    write_metadata(run_dir, run_id, cfg, power_mode_label, mode_id, settle_s, idle_s)

    tegra_proc: Optional[subprocess.Popen[str]] = None
    external_status: Dict[str, Any] = {}
    idle_start_ns = 0
    idle_stop_ns = 0
    proc_before: Dict[str, Any] = {}
    proc_after: Dict[str, Any] = {}

    try:
        set_power_mode(run_dir, mode_id)
        capture_static_state(run_dir)
        proc_before = process_check(run_dir, "process_check_before.txt")
        logger_check = logger_preflight(str(cfg["logger_url"]))
        write_json(run_dir / "external_logger_preflight.json", logger_check)
        if not logger_check.get("ok"):
            raise RuntimeError(f"external_logger_preflight_failed: {logger_check}")

        (run_dir / "external_logger").mkdir(exist_ok=True)
        write_text(run_dir / "external_logger_started_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")
        start_info, _ = start_external_logger(cfg, run_id, run_dir, power_mode_label)
        external_status["start"] = start_info

        tegra_proc = start_tegrastats(run_dir, int(cfg.get("tegrastats_interval_ms", 1000)))

        write_text(run_dir / "settle_start.txt", dt.datetime.now().astimezone().isoformat() + "\n")
        sleep_with_progress(settle_s, f"SETTLE {power_mode_label}")
        write_text(run_dir / "settle_end.txt", dt.datetime.now().astimezone().isoformat() + "\n")

        idle_start_ns = time.time_ns()
        write_text(run_dir / "idle_window_start.txt", dt.datetime.now().astimezone().isoformat() + "\n")
        write_text(run_dir / "idle_window_start_wallclock_ns.txt", str(idle_start_ns) + "\n")
        sleep_with_progress(idle_s, f"IDLE {power_mode_label}")
        idle_stop_ns = time.time_ns()
        write_text(run_dir / "idle_window_stop.txt", dt.datetime.now().astimezone().isoformat() + "\n")
        write_text(run_dir / "idle_window_stop_wallclock_ns.txt", str(idle_stop_ns) + "\n")

    finally:
        stop_tegrastats(tegra_proc, run_dir)
        try:
            write_text(run_dir / "external_logger_stopped_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")
            if "start" in external_status:
                external_status.update(stop_external_logger(cfg, run_id, run_dir))
        except Exception as exc:
            external_status["stop_error"] = f"{type(exc).__name__}: {exc}"
            write_json(run_dir / "external_logger_stop_error.json", external_status)
        proc_after = process_check(run_dir, "process_check_after.txt")
        write_text(run_dir / "run_stopped_at.txt", dt.datetime.now().astimezone().isoformat() + "\n")

    if idle_start_ns <= 0 or idle_stop_ns <= 0:
        raise RuntimeError(f"run did not complete official idle window: {run_id}")
    summary = create_run_summary(
        run_dir=run_dir,
        run_id=run_id,
        cfg=cfg,
        power_mode_label=power_mode_label,
        mode_id=mode_id,
        idle_start_ns=idle_start_ns,
        idle_stop_ns=idle_stop_ns,
        proc_before=proc_before,
        proc_after=proc_after,
        external_status=external_status,
    )
    print(
        "[SUMMARY] "
        f"valid={summary['run_valid']} "
        f"mode={power_mode_label} "
        f"avg_ext_w={summary['external'].get('external_avg_power_w')} "
        f"energy_wh={summary['external'].get('external_energy_wh')} "
        f"max_temp_c={summary['internal_telemetry'].get('internal_max_temp_c')}",
        flush=True,
    )
    return summary


def update_paper_tables(summaries: List[Dict[str, Any]]) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PAPER_TABLES_DIR / "idle_baseline_power_summary.csv"
    json_path = PAPER_TABLES_DIR / "idle_baseline_power_summary.json"
    fields = [
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
        "run_valid",
        "notes",
        "summary_path",
    ]
    rows: Dict[str, Dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["run_id"]] = row
    for summary in summaries:
        ext = summary.get("external") or {}
        tel = summary.get("internal_telemetry") or {}
        run_id = summary["run_id"]
        summary_dir = Path(summary.get("run_dir") or (IDLE_DIR / run_id))
        rows[run_id] = {
            "run_id": run_id,
            "board": summary.get("board"),
            "power_mode_label": summary.get("power_mode_label"),
            "mode_id": summary.get("mode_id"),
            "duration_seconds": summary.get("duration_seconds"),
            "external_avg_power_w": ext.get("external_avg_power_w"),
            "external_min_power_w": ext.get("external_min_power_w"),
            "external_peak_power_w": ext.get("external_peak_power_w"),
            "external_energy_wh": ext.get("external_energy_wh"),
            "voltage_min_v": ext.get("voltage_min_v"),
            "voltage_max_v": ext.get("voltage_max_v"),
            "logger_gap_gt1s_count": ext.get("logger_gap_gt1s_count"),
            "internal_max_temp_c": tel.get("internal_max_temp_c"),
            "internal_avg_power_w_if_available": tel.get("internal_avg_power_w_if_available"),
            "jetson_clocks_status": summary.get("jetson_clocks_status"),
            "run_valid": summary.get("run_valid"),
            "notes": "; ".join(summary.get("validity_reasons") or []),
            "summary_path": str(summary_dir / "idle_baseline_summary.json"),
        }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: str(r.get("run_id"))):
            writer.writerow(row)
    write_json(json_path, list(sorted(rows.values(), key=lambda r: str(r.get("run_id")))))


def load_merged_official_summaries(current_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for summary in current_summaries:
        run_id = str(summary.get("run_id") or "")
        if run_id:
            merged[run_id] = summary
    csv_path = PAPER_TABLES_DIR / "idle_baseline_power_summary.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_id = str(row.get("run_id") or "")
                summary_path = Path(row.get("summary_path") or "")
                if not run_id or "smoke" in run_id:
                    continue
                if run_id in merged:
                    continue
                if summary_path.exists():
                    merged[run_id] = load_json(summary_path)
    return list(sorted(merged.values(), key=lambda s: str(s.get("run_id"))))


def create_report_and_zip(summaries: List[Dict[str, Any]]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"idle_baseline_report_{stamp}.md"
    lines = ["# AgriJetsonBench Nano Super Idle Baseline Report", "", f"Created UTC: `{utc_now()}`", ""]
    lines += [
        "| run_id | mode | valid | P0 external avg W | min W | peak W | energy Wh | V min/max | gap >1s | max temp C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        ext = summary.get("external") or {}
        tel = summary.get("internal_telemetry") or {}
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                summary.get("run_id"),
                summary.get("power_mode_label"),
                summary.get("run_valid"),
                ext.get("external_avg_power_w"),
                ext.get("external_min_power_w"),
                ext.get("external_peak_power_w"),
                ext.get("external_energy_wh"),
                ext.get("voltage_min_v"),
                ext.get("voltage_max_v"),
                ext.get("logger_gap_gt1s_count"),
                tel.get("internal_max_temp_c"),
            )
        )
    lines += ["", "## Output Files", "", "- `idle_baselines/`", "- `paper_tables/idle_baseline_power_summary.csv`"]
    write_text(report_path, "\n".join(lines) + "\n")

    zip_path = ROOT_DIR / f"agrijetsonbench_nano_super_idle_baselines_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for summary in summaries:
            run_dir = Path(summary.get("run_dir") or (IDLE_DIR / str(summary.get("run_id"))))
            if not run_dir.exists():
                continue
            for path in run_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT_DIR))
        for folder in [PAPER_TABLES_DIR, REPORTS_DIR]:
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT_DIR))
    return zip_path


def write_destination_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(ROOT_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(ROOT_DIR)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "path": str(path),
                }
            )
    write_json(ROOT_DIR / "MANIFEST_AgriJetsonBench_Idle_Baseline.json", rows)
    with (ROOT_DIR / "MANIFEST_AgriJetsonBench_Idle_Baseline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256", "path"])
        writer.writeheader()
        writer.writerows(rows)


def noise_guard_config_from_args(cfg: Dict[str, Any], args: argparse.Namespace) -> NoiseGuardConfig:
    ng_cfg = dict(cfg.get("noise_guard") or {})
    enabled = bool(ng_cfg.get("enabled", True)) and not bool(args.no_noise_guard)
    profile = args.noise_guard_profile or str(ng_cfg.get("profile", "safe"))
    display_off = bool(ng_cfg.get("display_off", True)) and not bool(args.no_noise_guard_display_off)
    stop_services = bool(ng_cfg.get("stop_services", True)) and not bool(args.no_noise_guard_stop_services)
    disable_wifi = bool(ng_cfg.get("disable_wifi", True)) and not bool(args.no_noise_guard_disable_wifi)
    disable_bluetooth = bool(ng_cfg.get("disable_bluetooth", True)) and not bool(args.no_noise_guard_disable_bluetooth)
    return NoiseGuardConfig(
        enabled=enabled,
        profile=profile,
        preserve_url=str(cfg.get("logger_url") or ""),
        display_off=display_off,
        stop_services=stop_services,
        disable_wifi=disable_wifi,
        disable_bluetooth=disable_bluetooth,
    )


def enter_noise_guard(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[Optional[NoiseGuard], Optional[Path]]:
    ng = noise_guard_config_from_args(cfg, args)
    if not ng.enabled:
        print("[NOISE_GUARD] disabled", flush=True)
        return None, None
    NOISE_GUARD_DIR.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    guard = NoiseGuard(ng, notes=notes)
    summary = guard.enter()
    path = NOISE_GUARD_DIR / f"noise_guard_enter_{local_stamp()}.json"
    write_json(path, summary)
    print(
        "[NOISE_GUARD] active "
        f"profile={summary.get('profile')} "
        f"preserve_dev={((summary.get('preserve_route') or {}).get('dev') or '-')} "
        f"wifi_action={(summary.get('wifi') or {}).get('action')} "
        f"bluetooth_action={(summary.get('bluetooth') or {}).get('action')} "
        f"summary={path}",
        flush=True,
    )
    return guard, path


def restore_noise_guard(guard: Optional[NoiseGuard]) -> Optional[Path]:
    if guard is None:
        return None
    summary = guard.restore()
    path = NOISE_GUARD_DIR / f"noise_guard_restore_{local_stamp()}.json"
    write_json(path, summary)
    print(
        "[NOISE_GUARD] restored "
        f"wifi_restored={(summary.get('wifi') or {}).get('restored')} "
        f"bluetooth_restored={(summary.get('bluetooth') or {}).get('restored')} "
        f"errors={len(summary.get('errors') or [])} "
        f"summary={path}",
        flush=True,
    )
    return path


def attach_noise_guard_to_runs(
    summaries: List[Dict[str, Any]],
    enter_path: Optional[Path],
    restore_path: Optional[Path],
) -> None:
    if not summaries:
        return
    for summary in summaries:
        run_dir = Path(summary.get("run_dir") or "")
        if not run_dir.exists():
            continue
        noise_info: Dict[str, Any] = {"enabled": bool(enter_path)}
        if enter_path and enter_path.exists():
            dst = run_dir / "noise_guard_summary.json"
            shutil.copy2(enter_path, dst)
            noise_info["summary_path"] = str(dst)
        if restore_path and restore_path.exists():
            dst = run_dir / "noise_guard_restore_summary.json"
            shutil.copy2(restore_path, dst)
            noise_info["restore_summary_path"] = str(dst)
        summary["noise_guard"] = noise_info
        summary_path = run_dir / "idle_baseline_summary.json"
        if summary_path.exists():
            existing = load_json(summary_path)
            existing["noise_guard"] = noise_info
            write_json(summary_path, existing)


def run_preflight(cfg: Dict[str, Any]) -> Dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"created_utc": utc_now(), "checks": {}}
    report["checks"]["sudo_n_true"] = run_capture(["sudo", "-n", "true"], timeout=10)
    report["checks"]["board_model"] = {"value": read_board_model(), "ok": "Orin Nano" in read_board_model()}
    nvp_conf = run_capture(["bash", "-lc", "cat /etc/nvpmodel.conf 2>/dev/null || true"], timeout=10)
    modes = parse_nvpmodel_modes(nvp_conf.get("stdout", ""))
    report["checks"]["nvpmodel_modes"] = {"modes": modes, "ok": "15W" in modes and "25W" in modes}
    report["checks"]["nvpmodel_current"] = run_capture(["sudo", "-n", "nvpmodel", "-q"], timeout=10)
    report["checks"]["jetson_clocks_show"] = run_capture(["sudo", "-n", "jetson_clocks", "--show"], timeout=20)
    report["checks"]["tegrastats_path"] = run_capture(["bash", "-lc", "command -v tegrastats"], timeout=10)
    report["checks"]["logger"] = logger_preflight(str(cfg["logger_url"]))
    report["checks"]["noise_guard_module"] = {
        "ok": True,
        "module_path": str((SCRIPT_DIR / "noise_guard.py").resolve()),
        "enabled_by_config": bool((cfg.get("noise_guard") or {}).get("enabled", True)),
    }
    report["checks"]["noise_guard_preserve_route"] = NoiseGuard(
        NoiseGuardConfig(enabled=False, preserve_url=str(cfg.get("logger_url") or ""))
    ).summary()
    report["checks"]["noise_guard_systemctl_probe"] = run_capture(
        ["sudo", "-n", "systemctl", "is-active", "apt-daily.service"],
        timeout=10,
    )
    if shutil.which("nmcli"):
        report["checks"]["noise_guard_nmcli_probe"] = run_capture(["nmcli", "radio", "wifi"], timeout=10)
    else:
        report["checks"]["noise_guard_nmcli_probe"] = {"returncode": 0, "stdout": "nmcli_not_installed"}
    if shutil.which("rfkill"):
        report["checks"]["noise_guard_rfkill_probe"] = run_capture(["rfkill", "list", "bluetooth"], timeout=10)
    else:
        report["checks"]["noise_guard_rfkill_probe"] = {"returncode": 0, "stdout": "rfkill_not_installed"}
    tmp_dir = LOGS_DIR / f"preflight_process_{local_stamp()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report["checks"]["process_check"] = process_check(tmp_dir, "process_check_preflight.txt")
    report["ok"] = (
        report["checks"]["sudo_n_true"]["returncode"] == 0
        and report["checks"]["board_model"]["ok"]
        and report["checks"]["nvpmodel_modes"]["ok"]
        and report["checks"]["tegrastats_path"]["returncode"] == 0
        and bool(report["checks"]["logger"].get("ok"))
    )
    path = LOGS_DIR / f"idle_baseline_preflight_{local_stamp()}.json"
    write_json(path, report)
    print(f"[PREFLIGHT] ok={report['ok']} logger_ok={report['checks']['logger'].get('ok')} modes={modes}", flush=True)
    print(f"[PREFLIGHT_REPORT] {path}", flush=True)
    return report


def load_config(path: Path) -> Dict[str, Any]:
    cfg = load_json(path)
    if not cfg.get("logger_url"):
        raise RuntimeError("logger_url is required")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AgriJetsonBench Nano Super idle baseline protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["15W", "25W", "both"], default="both")
    parser.add_argument("--settle-seconds", type=int, default=None)
    parser.add_argument("--idle-seconds", type=int, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--no-noise-guard", action="store_true", help="Disable NoiseGuard for this run.")
    parser.add_argument("--noise-guard-profile", choices=["safe", "aggressive"], default=None)
    parser.add_argument("--no-noise-guard-display-off", action="store_true")
    parser.add_argument("--no-noise-guard-stop-services", action="store_true")
    parser.add_argument("--no-noise-guard-disable-wifi", action="store_true")
    parser.add_argument("--no-noise-guard-disable-bluetooth", action="store_true")
    parser.add_argument(
        "--require-reboot-after-mode-set",
        dest="require_reboot_after_mode_set",
        action="store_true",
        help="Set the requested single power mode, reboot, then require rerunning the same command before measurement.",
    )
    parser.add_argument(
        "--no-reboot-after-mode-set",
        dest="require_reboot_after_mode_set",
        action="store_false",
        help="Run without the set-mode/reboot gate.",
    )
    parser.set_defaults(require_reboot_after_mode_set=False)
    args = parser.parse_args()

    for folder in [IDLE_DIR, PAPER_TABLES_DIR, REPORTS_DIR, LOGS_DIR, DIAGNOSTICS_DIR, NOISE_GUARD_DIR, REBOOT_STATE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if args.settle_seconds is not None:
        cfg["settle_seconds"] = args.settle_seconds
    if args.idle_seconds is not None:
        cfg["idle_seconds"] = args.idle_seconds

    if args.preflight:
        run_preflight(cfg)
        write_destination_manifest()
        return 0

    preflight = run_preflight(cfg)
    if not preflight.get("ok"):
        print("[ERROR] Preflight failed; not starting idle baseline runs.", file=sys.stderr, flush=True)
        write_destination_manifest()
        return 2
    if args.require_reboot_after_mode_set and args.mode == "both":
        print(
            "[ERROR] Reboot-gated power-mode runs must be launched one mode at a time: "
            "bash code/run_all_idle_baselines.sh --mode 15W, then after it finishes "
            "bash code/run_all_idle_baselines.sh --mode 25W.",
            file=sys.stderr,
            flush=True,
        )
        write_destination_manifest()
        return 2

    summaries: List[Dict[str, Any]] = []
    guard: Optional[NoiseGuard] = None
    noise_enter_path: Optional[Path] = None
    noise_restore_path: Optional[Path] = None
    reboot_context: Dict[str, Any] = {"required": bool(args.require_reboot_after_mode_set)}
    try:
        nvp_conf = run_capture(["bash", "-lc", "cat /etc/nvpmodel.conf 2>/dev/null || true"], timeout=10)
        modes = parse_nvpmodel_modes(nvp_conf.get("stdout", ""))
        targets = ["15W", "25W"] if args.mode == "both" else [args.mode]
        if args.require_reboot_after_mode_set:
            mode_id = modes.get(targets[0])
            if mode_id is None:
                raise RuntimeError(f"nvpmodel mode {targets[0]} not found in /etc/nvpmodel.conf")
            should_exit, reboot_context = confirm_or_prepare_reboot_for_power_mode(targets[0], mode_id, args)
            if should_exit:
                write_destination_manifest()
                return 0 if reboot_context.get("reboot_issued") else 4

        guard, noise_enter_path = enter_noise_guard(cfg, args)
        for label in targets:
            mode_id = modes.get(label)
            if mode_id is None:
                raise RuntimeError(f"nvpmodel mode {label} not found in /etc/nvpmodel.conf")
            summaries.append(
                run_one_mode(
                    cfg=cfg,
                    power_mode_label=label,
                    mode_id=mode_id,
                    settle_s=int(cfg.get("settle_seconds", 120)),
                    idle_s=int(cfg.get("idle_seconds", 300)),
                    run_label=args.run_label,
                )
            )
            summaries[-1]["power_mode_reboot"] = reboot_context
            run_dir = Path(summaries[-1].get("run_dir") or "")
            if run_dir.exists():
                summary_path = run_dir / "idle_baseline_summary.json"
                if summary_path.exists():
                    existing = load_json(summary_path)
                    existing["power_mode_reboot"] = reboot_context
                    write_json(summary_path, existing)
    finally:
        noise_restore_path = restore_noise_guard(guard)
        attach_noise_guard_to_runs(summaries, noise_enter_path, noise_restore_path)

    if args.run_label:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        diagnostic_summary = DIAGNOSTICS_DIR / f"diagnostic_{args.run_label}_summary_{stamp}.json"
        write_json(diagnostic_summary, summaries)
        zip_path = DIAGNOSTICS_DIR / f"diagnostic_{args.run_label}_{stamp}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for summary in summaries:
                run_dir = Path(summary.get("run_dir") or "")
                if not run_dir.exists():
                    continue
                for path in run_dir.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(ROOT_DIR))
        print(f"[DIAGNOSTIC_SUMMARY] {diagnostic_summary}", flush=True)
    else:
        update_paper_tables(summaries)
        zip_path = create_report_and_zip(load_merged_official_summaries(summaries))
    write_destination_manifest()
    print(f"[OUTPUT_DIR] {ROOT_DIR}", flush=True)
    print(f"[SUMMARY_CSV] {PAPER_TABLES_DIR / 'idle_baseline_power_summary.csv'}", flush=True)
    print(f"[ZIP] {zip_path}", flush=True)
    return 0 if all(s.get("run_valid") for s in summaries) else 3


if __name__ == "__main__":
    raise SystemExit(main())
