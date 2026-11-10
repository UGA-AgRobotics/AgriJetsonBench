#!/usr/bin/env python3
"""Summarize AGX battery inference-only runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


FIELDS = [
    "board",
    "model",
    "task",
    "precision",
    "input_size",
    "batch_size",
    "total_runtime_seconds",
    "total_images",
    "fps",
    "median_latency_ms",
    "p95_latency_ms",
    "stopped_reason",
    "run_dir",
    "run_id",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(runs_dir: Path, include_dry_runs: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        try:
            data = load_json(summary_path)
        except Exception as exc:
            rows.append(
                {
                    "board": "",
                    "model": "",
                    "task": "",
                    "precision": "",
                    "input_size": "",
                    "batch_size": "",
                    "total_runtime_seconds": "",
                    "total_images": "",
                    "fps": "",
                    "median_latency_ms": "",
                    "p95_latency_ms": "",
                    "stopped_reason": f"summary_read_error: {exc}",
                    "run_dir": str(summary_path.parent),
                    "run_id": summary_path.parent.name,
                }
            )
            continue
        if data.get("stopped_reason") == "dry_run_no_inference" and not include_dry_runs:
            continue
        rows.append(
            {
                "board": data.get("board"),
                "model": data.get("model"),
                "task": data.get("task"),
                "precision": data.get("precision"),
                "input_size": data.get("input_size"),
                "batch_size": data.get("batch_size"),
                "total_runtime_seconds": data.get("elapsed_seconds"),
                "total_images": data.get("total_images"),
                "fps": data.get("fps"),
                "median_latency_ms": data.get("median_latency_ms"),
                "p95_latency_ms": data.get("p95_latency_ms"),
                "stopped_reason": data.get("stopped_reason"),
                "run_dir": str(summary_path.parent),
                "run_id": data.get("run_id", summary_path.parent.name),
            }
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "battery_inference_summary_agx.csv"
    json_path = out_dir / "battery_inference_summary_agx.json"
    md_path = out_dir / "battery_inference_summary_agx.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = ["# AGX Battery Inference Summary", ""]
    if not rows:
        lines.append("No completed run summaries found.")
    else:
        lines.append("| run_id | model | task | precision | input | batch | runtime_s | images | fps | median_ms | p95_ms | stopped_reason |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                "| {run_id} | {model} | {task} | {precision} | {input_size} | {batch_size} | "
                "{total_runtime_seconds} | {total_images} | {fps} | {median_latency_ms} | "
                "{p95_latency_ms} | {stopped_reason} |".format(**{k: row.get(k, "") for k in FIELDS})
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize completed AGX inference-only battery runs.")
    parser.add_argument("--runs-dir", default="./battery_inference_runs/agx")
    parser.add_argument("--out-dir", default="./battery_inference_summary/agx")
    parser.add_argument("--include-dry-runs", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = collect(Path(args.runs_dir).expanduser().resolve(), include_dry_runs=bool(args.include_dry_runs))
    write_outputs(rows, Path(args.out_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
