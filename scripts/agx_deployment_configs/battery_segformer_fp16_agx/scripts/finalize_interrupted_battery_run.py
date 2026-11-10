#!/usr/bin/env python3
"""Finalize an interrupted AGX battery endurance run from saved files.

This script is for power-loss/reboot cases where the runner could not write its
normal final summary. It does not run inference and does not modify the logger.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

SUPPLEMENT_FIELDS = [
    "run_id",
    "board",
    "model",
    "task",
    "precision",
    "input_size",
    "batch_size",
    "start_time_utc",
    "last_saved_event_utc",
    "runtime_seconds_until_last_saved_status",
    "runtime_hours_until_last_saved_status",
    "total_images_saved",
    "fps_from_saved_latency",
    "median_latency_ms",
    "p95_latency_ms",
    "interruption_classification",
    "clean_ctrl_c_stop",
    "run_complete_event",
    "external_logger_archive_available",
    "external_samples",
    "external_energy_joules",
    "external_avg_power_w",
    "internal_avg_power_w",
    "internal_max_temp_c",
    "internal_avg_gr3d_pct",
    "current_boot_time_utc",
    "gap_last_event_to_current_boot_seconds",
    "notes",
    "run_dir",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def fmt_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def clean_lines(path: Path) -> Iterable[str]:
    with path.open("rb") as f:
        for raw in f:
            yield raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:d}:{m:02d}:{s:02d}"


def current_boot_time_utc() -> Optional[datetime]:
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                return datetime.fromtimestamp(int(line.split()[1]), tz=timezone.utc)
    except Exception:
        return None
    return None


def load_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    nul_count = path.read_bytes().count(b"\x00")
    reader = csv.DictReader(clean_lines(path))
    rows: List[Dict[str, Any]] = []
    for row in reader:
        payload_text = row.get("payload_json") or "{}"
        try:
            payload = json.loads(payload_text)
        except Exception:
            payload = {"raw_payload_json": payload_text}
        row["_payload"] = payload
        rows.append(row)
    return rows, nul_count


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for row in events:
        name = str(row.get("event") or "")
        counts[name] = counts.get(name, 0) + 1
    first = events[0] if events else {}
    last = events[-1] if events else {}
    last_status: Optional[Dict[str, Any]] = None
    for row in reversed(events):
        if row.get("event") == "STATUS":
            last_status = row
            break
    return {
        "event_count": len(events),
        "event_counts": counts,
        "first_event": first.get("event"),
        "first_event_time_utc": first.get("timestamp_wall"),
        "last_event": last.get("event"),
        "last_event_time_utc": last.get("timestamp_wall"),
        "last_status_time_utc": last_status.get("timestamp_wall") if last_status else None,
        "last_status_payload": last_status.get("_payload") if last_status else {},
        "has_run_complete": counts.get("RUN_COMPLETE", 0) > 0,
        "has_ctrl_c": counts.get("CTRL_C_RECEIVED", 0) > 0,
        "error_event_count": counts.get("ERROR", 0),
    }


def summarize_latency(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "row_count": 0,
            "latency_count": 0,
        }
    latencies: List[float] = []
    row_count = 0
    success_count = 0
    first_ts = None
    last_ts = None
    last_image = None
    reader = csv.DictReader(clean_lines(path))
    for row in reader:
        row_count += 1
        if first_ts is None:
            first_ts = row.get("timestamp_wall")
        last_ts = row.get("timestamp_wall") or last_ts
        last_image = row.get("image_id_or_path") or last_image
        latency = safe_float(row.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)
        if str(row.get("success", "")).lower() == "true":
            success_count += 1
    return {
        "available": True,
        "row_count": row_count,
        "success_count": success_count,
        "latency_count": len(latencies),
        "first_timestamp_utc": first_ts,
        "last_timestamp_utc": last_ts,
        "last_image_id_or_path": last_image,
        "median_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


def parse_run_loop_index(path: Path) -> Optional[int]:
    match = re.search(r"_loop(\d+)$", path.name)
    return int(match.group(1)) if match else None


def summarize_child_runs(run_dir: Path) -> Dict[str, Any]:
    root = run_dir / "child_benchmark_runs"
    dirs = sorted([p for p in root.glob("*_loop*") if p.is_dir()], key=lambda p: parse_run_loop_index(p) or -1)
    completed: List[int] = []
    incomplete: List[int] = []
    for path in dirs:
        idx = parse_run_loop_index(path)
        if idx is None:
            continue
        if (path / "FINAL_SUMMARY.json").exists():
            completed.append(idx)
        else:
            incomplete.append(idx)
    return {
        "child_run_dir_count": len(dirs),
        "completed_child_loop_count": len(completed),
        "incomplete_child_loop_count": len(incomplete),
        "highest_completed_loop": max(completed) if completed else None,
        "incomplete_loops": incomplete[-10:],
    }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as f:
        for raw in f:
            text = raw.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def summarize_internal(path: Path) -> Dict[str, Any]:
    total_power_values: List[float] = []
    temp_values: List[float] = []
    ram_values: List[float] = []
    gr3d_values: List[float] = []
    line_count = 0
    first_ts = None
    last_ts = None
    for obj in iter_jsonl(path):
        line_count += 1
        first_ts = first_ts or obj.get("timestamp_utc")
        last_ts = obj.get("timestamp_utc") or last_ts
        live = obj.get("internal_live") or {}
        hwmon = live.get("hwmon") if isinstance(live, dict) else {}
        thermal = live.get("thermal") if isinstance(live, dict) else {}
        tstats = live.get("tegrastats") if isinstance(live, dict) else {}
        if isinstance(hwmon, dict):
            power = safe_float(hwmon.get("total_power_w"))
            if power is not None:
                total_power_values.append(power)
        if isinstance(thermal, dict):
            temp = safe_float(thermal.get("max_temp_c"))
            if temp is not None:
                temp_values.append(temp)
        if isinstance(tstats, dict):
            temp = safe_float(tstats.get("max_temp_c"))
            if temp is not None:
                temp_values.append(temp)
            ram = safe_float(tstats.get("ram_used_mb"))
            if ram is not None:
                ram_values.append(ram)
            gr3d = safe_float(tstats.get("gr3d_pct"))
            if gr3d is not None:
                gr3d_values.append(gr3d)
    return {
        "available": line_count > 0,
        "sample_count": line_count,
        "first_timestamp_utc": first_ts,
        "last_timestamp_utc": last_ts,
        "avg_internal_power_w": sum(total_power_values) / len(total_power_values) if total_power_values else None,
        "max_internal_power_w": max(total_power_values) if total_power_values else None,
        "max_temp_c": max(temp_values) if temp_values else None,
        "avg_ram_used_mb": sum(ram_values) / len(ram_values) if ram_values else None,
        "avg_gr3d_pct": sum(gr3d_values) / len(gr3d_values) if gr3d_values else None,
        "max_gr3d_pct": max(gr3d_values) if gr3d_values else None,
    }


def external_archive_path(run_dir: Path, run_id: str) -> Path:
    return run_dir / "external_logger" / f"{run_id}.tar.gz"


def download_external_archive(run_id: str, logger_url: str, out_path: Path) -> Dict[str, Any]:
    url = logger_url.rstrip("/") + f"/download/{run_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310
            data = resp.read()
        out_path.write_bytes(data)
        if not tarfile.is_tarfile(out_path):
            detail = data.decode("utf-8", errors="replace")[:500]
            out_path.unlink(missing_ok=True)
            return {"ok": False, "url": url, "error": f"download was not a tar archive: {detail}"}
        return {"ok": True, "url": url, "path": str(out_path), "bytes": out_path.stat().st_size}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "url": url, "http_status": exc.code, "error": detail or str(exc)}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def summarize_external_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False, "archive_path": str(path)}
    if not tarfile.is_tarfile(path):
        return {"available": False, "archive_path": str(path), "error": "not a tar archive"}
    try:
        with tarfile.open(path) as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith("external_measurements.csv")), None)
            if member is None:
                return {"available": False, "archive_path": str(path), "error": "external_measurements.csv not found"}
            file_obj = tf.extractfile(member)
            if file_obj is None:
                return {"available": False, "archive_path": str(path), "error": "could not extract external_measurements.csv"}
            text = file_obj.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"available": False, "archive_path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    rows = list(csv.DictReader(io.StringIO(text)))
    powers: List[float] = []
    voltages: List[float] = []
    currents: List[float] = []
    energy_values: List[float] = []
    first_ts = None
    last_ts = None
    low_voltage_count = 0
    for row in rows:
        first_ts = first_ts or row.get("timestamp_utc")
        last_ts = row.get("timestamp_utc") or last_ts
        power = safe_float(row.get("power_w"))
        voltage = safe_float(row.get("bus_voltage_v"))
        current = safe_float(row.get("current_a"))
        energy = safe_float(row.get("energy_joules"))
        if power is not None:
            powers.append(power)
        if voltage is not None:
            voltages.append(voltage)
            if voltage < 10.0:
                low_voltage_count += 1
        if current is not None:
            currents.append(current)
        if energy is not None:
            energy_values.append(energy)
    return {
        "available": True,
        "archive_path": str(path),
        "sample_count": len(rows),
        "first_timestamp_utc": first_ts,
        "last_timestamp_utc": last_ts,
        "avg_power_w": sum(powers) / len(powers) if powers else None,
        "max_power_w": max(powers) if powers else None,
        "min_power_w": min(powers) if powers else None,
        "avg_voltage_v": sum(voltages) / len(voltages) if voltages else None,
        "min_voltage_v": min(voltages) if voltages else None,
        "avg_current_a": sum(currents) / len(currents) if currents else None,
        "energy_joules": max(energy_values) if energy_values else None,
        "low_voltage_sample_count": low_voltage_count,
    }


def build_recovered_summary(
    run_dir: Path,
    stopped_reason: str,
    event_summary: Dict[str, Any],
    latency_summary: Dict[str, Any],
    child_summary: Dict[str, Any],
    internal_summary: Dict[str, Any],
    external_summary: Dict[str, Any],
) -> Dict[str, Any]:
    run_config = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    preflight = read_json(run_dir / "preflight_report.json") if (run_dir / "preflight_report.json").exists() else {}
    run_id = str(run_config.get("run_id") or run_dir.name)
    last_status_payload = event_summary.get("last_status_payload") or {}
    elapsed = safe_float(last_status_payload.get("elapsed_seconds"))
    if elapsed is None:
        first_dt = parse_utc(event_summary.get("first_event_time_utc"))
        last_dt = parse_utc(event_summary.get("last_event_time_utc"))
        elapsed = (last_dt - first_dt).total_seconds() if first_dt and last_dt else None
    total_images = int(latency_summary.get("row_count") or last_status_payload.get("total_images") or 0)
    fps = float(total_images) / elapsed if elapsed and elapsed > 0 and total_images > 0 else None
    boot_dt = current_boot_time_utc()
    last_event_dt = parse_utc(event_summary.get("last_event_time_utc"))
    gap_to_boot = (boot_dt - last_event_dt).total_seconds() if boot_dt and last_event_dt and boot_dt > last_event_dt else None
    clean_ctrl_c = bool(event_summary.get("has_ctrl_c") and event_summary.get("has_run_complete"))
    interruption_classification = "clean_ctrl_c" if clean_ctrl_c else "power_loss_or_unplanned_reboot"
    return {
        "run_id": run_id,
        "board": run_config.get("board", "agx"),
        "board_display_name": run_config.get("board_display_name"),
        "task": run_config.get("task"),
        "model": run_config.get("model"),
        "precision": run_config.get("precision"),
        "input_size": run_config.get("input_size"),
        "batch_size": run_config.get("batch_size"),
        "start_time": event_summary.get("first_event_time_utc"),
        "end_time": event_summary.get("last_event_time_utc"),
        "last_saved_status_time": event_summary.get("last_status_time_utc"),
        "stopped_reason": stopped_reason,
        "interruption_classification": interruption_classification,
        "clean_ctrl_c_stop": clean_ctrl_c,
        "run_complete_event": bool(event_summary.get("has_run_complete")),
        "total_images": total_images,
        "elapsed_seconds": elapsed,
        "runtime_hours": elapsed / 3600.0 if elapsed is not None else None,
        "runtime_hms": hms(elapsed),
        "fps": fps,
        "median_latency_ms": latency_summary.get("median_latency_ms"),
        "p95_latency_ms": latency_summary.get("p95_latency_ms"),
        "min_latency_ms": latency_summary.get("min_latency_ms"),
        "max_latency_ms": latency_summary.get("max_latency_ms"),
        "nvpmodel_output": {
            "nvpmodel_q": preflight.get("nvpmodel_q", {}),
            "nvpmodel_q_verbose": preflight.get("nvpmodel_q_verbose", {}),
        },
        "device_tree_model": preflight.get("device_tree_model"),
        "jetpack_l4t_version": preflight.get("l4t_release"),
        "cuda_version": preflight.get("cuda_version"),
        "tensorrt_version": preflight.get("tensorrt_version"),
        "git_commit": preflight.get("git_commit"),
        "engine_path": run_config.get("engine_path"),
        "dataset_path": run_config.get("dataset_path"),
        "power_source_note": "This run was powered by UGREEN 25,000mAh 145W USB-C PD power bank.",
        "external_logger_note": "External logger archive may be missing if reboot/power loss occurred before the runner could stop and download Pc_package logs.",
        "external_logger_archive_available": bool(external_summary.get("available")),
        "external_logger_summary": external_summary,
        "internal_dashboard_summary": internal_summary,
        "event_summary": event_summary,
        "latency_summary": latency_summary,
        "child_run_summary": child_summary,
        "current_boot_time_utc": fmt_utc(boot_dt),
        "gap_last_event_to_current_boot_seconds": gap_to_boot,
        "recovery_generated_at_utc": utc_now(),
        "recovery_note": "Recovered from saved files after an interrupted run; do not label as clean Ctrl-C stop.",
        "run_dir": str(run_dir),
        "config_name": run_config.get("config_name"),
    }


def write_inference_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": summary.get("run_id"),
                "board": summary.get("board"),
                "task": summary.get("task"),
                "model": summary.get("model"),
                "precision": summary.get("precision"),
                "input_size": summary.get("input_size"),
                "batch_size": summary.get("batch_size"),
                "total_images": summary.get("total_images"),
                "total_elapsed_seconds": summary.get("elapsed_seconds"),
                "fps": summary.get("fps"),
                "median_latency_ms": summary.get("median_latency_ms"),
                "p95_latency_ms": summary.get("p95_latency_ms"),
                "min_latency_ms": summary.get("min_latency_ms"),
                "max_latency_ms": summary.get("max_latency_ms"),
                "stopped_reason": summary.get("stopped_reason"),
            }
        )


def supplementary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    external = summary.get("external_logger_summary") or {}
    internal = summary.get("internal_dashboard_summary") or {}
    return {
        "run_id": summary.get("run_id"),
        "board": summary.get("board"),
        "model": summary.get("model"),
        "task": summary.get("task"),
        "precision": summary.get("precision"),
        "input_size": summary.get("input_size"),
        "batch_size": summary.get("batch_size"),
        "start_time_utc": summary.get("start_time"),
        "last_saved_event_utc": summary.get("end_time"),
        "runtime_seconds_until_last_saved_status": summary.get("elapsed_seconds"),
        "runtime_hours_until_last_saved_status": summary.get("runtime_hours"),
        "total_images_saved": summary.get("total_images"),
        "fps_from_saved_latency": summary.get("fps"),
        "median_latency_ms": summary.get("median_latency_ms"),
        "p95_latency_ms": summary.get("p95_latency_ms"),
        "interruption_classification": summary.get("interruption_classification"),
        "clean_ctrl_c_stop": summary.get("clean_ctrl_c_stop"),
        "run_complete_event": summary.get("run_complete_event"),
        "external_logger_archive_available": external.get("available"),
        "external_samples": external.get("sample_count"),
        "external_energy_joules": external.get("energy_joules"),
        "external_avg_power_w": external.get("avg_power_w"),
        "internal_avg_power_w": internal.get("avg_internal_power_w"),
        "internal_max_temp_c": internal.get("max_temp_c"),
        "internal_avg_gr3d_pct": internal.get("avg_gr3d_pct"),
        "current_boot_time_utc": summary.get("current_boot_time_utc"),
        "gap_last_event_to_current_boot_seconds": summary.get("gap_last_event_to_current_boot_seconds"),
        "notes": summary.get("recovery_note"),
        "run_dir": summary.get("run_dir"),
    }


def write_supplement_outputs(run_dir: Path, summary: Dict[str, Any]) -> None:
    row = supplementary_row(summary)
    csv_path = run_dir / "supplementary_battery_endurance_agx.csv"
    md_path = run_dir / "supplementary_battery_endurance_agx.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUPPLEMENT_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUPPLEMENT_FIELDS})
    lines = [
        f"# Supplementary Battery Endurance Recovery: {summary.get('run_id')}",
        "",
        "This run was recovered from saved inference-side files after an unplanned Jetson reboot/power-loss interruption.",
        "It should be reported as an interrupted endurance observation, not as a clean Ctrl-C finalized benchmark.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Model | `{summary.get('model')}` |",
        f"| Task | `{summary.get('task')}` |",
        f"| Precision | `{summary.get('precision')}` |",
        f"| Input size | `{summary.get('input_size')}` |",
        f"| Batch size | `{summary.get('batch_size')}` |",
        f"| Start time UTC | `{summary.get('start_time')}` |",
        f"| Last saved event UTC | `{summary.get('end_time')}` |",
        f"| Runtime until last saved status | `{summary.get('runtime_hms')}` ({summary.get('runtime_hours'):.4f} h) |",
        f"| Total images saved | `{summary.get('total_images')}` |",
        f"| FPS from saved latency rows | `{summary.get('fps')}` |",
        f"| Median latency ms | `{summary.get('median_latency_ms')}` |",
        f"| P95 latency ms | `{summary.get('p95_latency_ms')}` |",
        f"| Interruption classification | `{summary.get('interruption_classification')}` |",
        f"| Clean Ctrl-C stop | `{summary.get('clean_ctrl_c_stop')}` |",
        f"| RUN_COMPLETE event | `{summary.get('run_complete_event')}` |",
        f"| Current boot time UTC observed during recovery | `{summary.get('current_boot_time_utc')}` |",
        f"| Gap from last saved event to observed boot | `{summary.get('gap_last_event_to_current_boot_seconds')}` s |",
        "",
        "## Power And Telemetry Notes",
        "",
    ]
    external = summary.get("external_logger_summary") or {}
    internal = summary.get("internal_dashboard_summary") or {}
    if external.get("available"):
        lines.extend(
            [
                f"- External logger archive: `{external.get('archive_path')}`",
                f"- External samples: `{external.get('sample_count')}`",
                f"- External energy J: `{external.get('energy_joules')}`",
                f"- External average power W: `{external.get('avg_power_w')}`",
            ]
        )
    else:
        lines.append("- External logger archive is not available for this interrupted run; use inference-side endurance duration/images and internal dashboard telemetry.")
    if internal.get("available"):
        lines.extend(
            [
                f"- Internal telemetry samples: `{internal.get('sample_count')}`",
                f"- Internal average power W: `{internal.get('avg_internal_power_w')}`",
                f"- Internal max temp C: `{internal.get('max_temp_c')}`",
                f"- Internal average GR3D %: `{internal.get('avg_gr3d_pct')}`",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"run dir does not exist: {run_dir}")
    run_id = run_dir.name
    if (run_dir / "run_config.json").exists():
        run_id = str(read_json(run_dir / "run_config.json").get("run_id") or run_id)

    archive = external_archive_path(run_dir, run_id)
    download_info = None
    if args.try_download_external and not archive.exists() and args.logger_url:
        download_info = download_external_archive(run_id, str(args.logger_url), archive)

    events, nul_count = load_events(run_dir / "events.csv")
    event_summary = summarize_events(events)
    event_summary["events_csv_nul_byte_count"] = nul_count
    latency_summary = summarize_latency(run_dir / "inference_latency.csv")
    child_summary = summarize_child_runs(run_dir)
    internal_summary = summarize_internal(run_dir / "dashboard_internal_live.jsonl")
    external_summary = summarize_external_archive(archive)
    if download_info is not None:
        external_summary["download_attempt"] = download_info
    summary = build_recovered_summary(
        run_dir=run_dir,
        stopped_reason=str(args.stopped_reason),
        event_summary=event_summary,
        latency_summary=latency_summary,
        child_summary=child_summary,
        internal_summary=internal_summary,
        external_summary=external_summary,
    )

    if (run_dir / "summary.json").exists() and not args.overwrite:
        raise RuntimeError(f"summary.json already exists; pass --overwrite to replace it: {run_dir / 'summary.json'}")

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "recovery_report.json", summary)
    write_inference_summary_csv(run_dir / "inference_summary.csv", summary)
    write_supplement_outputs(run_dir, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize an interrupted AGX battery inference run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stopped-reason", default="power_loss_or_unplanned_reboot")
    parser.add_argument("--logger-url", default="http://${LOGGER_DASHBOARD_HOST}:8000")
    parser.add_argument("--try-download-external", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = finalize_run(args)
    print(f"[OK] recovered run: {summary.get('run_id')}")
    print(f"[OK] stopped_reason: {summary.get('stopped_reason')}")
    print(f"[OK] total_images: {summary.get('total_images')}")
    print(f"[OK] runtime_hms: {summary.get('runtime_hms')}")
    print(f"[OK] summary: {Path(args.run_dir).expanduser().resolve() / 'summary.json'}")
    print(f"[OK] supplementary: {Path(args.run_dir).expanduser().resolve() / 'supplementary_battery_endurance_agx.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
