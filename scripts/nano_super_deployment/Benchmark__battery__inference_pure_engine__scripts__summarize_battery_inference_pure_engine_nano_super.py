#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "board",
    "model",
    "task",
    "precision",
    "input_size",
    "batch_size",
    "timing_boundary",
    "measurement_backend",
    "total_runtime_seconds",
    "total_images",
    "fps",
    "median_latency_ms",
    "p95_latency_ms",
    "stopped_reason",
    "run_dir",
    "run_id",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = SCRIPT_ROOT / path
    return path.resolve()


def load_summaries(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            data = {
                "run_id": summary_path.parent.name,
                "run_dir": str(summary_path.parent),
                "stopped_reason": f"summary_read_error: {type(exc).__name__}: {exc}",
            }
        if data.get("stopped_reason") == "dry_run_complete":
            continue
        try:
            total_images = int(data.get("total_images") or 0)
        except Exception:
            total_images = 0
        if total_images <= 0:
            continue
        rows.append(
            {
                "board": data.get("board", ""),
                "model": data.get("model", ""),
                "task": data.get("task", ""),
                "precision": data.get("precision", ""),
                "input_size": data.get("input_size", ""),
                "batch_size": data.get("batch_size", ""),
                "timing_boundary": data.get("timing_boundary", ""),
                "measurement_backend": data.get("measurement_backend", ""),
                "total_runtime_seconds": data.get("elapsed_seconds", ""),
                "total_images": data.get("total_images", ""),
                "fps": data.get("fps", ""),
                "median_latency_ms": data.get("median_latency_ms", ""),
                "p95_latency_ms": data.get("p95_latency_ms", ""),
                "stopped_reason": data.get("stopped_reason", ""),
                "run_dir": data.get("run_dir", str(summary_path.parent)),
                "run_id": data.get("run_id", summary_path.parent.name),
            }
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "battery_inference_pure_engine_summary_nano_super.csv"
    json_path = out_dir / "battery_inference_pure_engine_summary_nano_super.json"
    md_path = out_dir / "battery_inference_pure_engine_summary_nano_super.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Nano Super Pure Engine Battery Summary",
        "",
        "In file pure TensorRT engine-only run ha ro summarize mikone. J/image ya energy hesab nemishe ta vaghti Pc_package power log ha later merge beshan.",
        "",
        "| run_id | model | task | precision | input | batch | timing | backend | runtime_s | samples | fps | median_ms | p95_ms | stopped_reason |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {model} | {task} | {precision} | {input_size} | {batch_size} | {timing_boundary} | {measurement_backend} | {total_runtime_seconds} | {total_images} | {fps} | {median_latency_ms} | {p95_latency_ms} | {stopped_reason} |".format(
                **{field: row.get(field, "") for field in FIELDS}
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Nano Super pure engine battery runs.")
    parser.add_argument("--runs-dir", default="./battery_inference_pure_engine_runs/nano_super")
    parser.add_argument("--out-dir", default="./battery_inference_pure_engine_summary/nano_super")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_summaries(resolve_path(args.runs_dir))
    write_outputs(rows, resolve_path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
