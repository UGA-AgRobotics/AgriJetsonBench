#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_segformerb0_fp16_resolution_discovery import (  # type: ignore
    ProgressReporter,
    asdict,
    extract_metrics,
    load_json,
    metric_float,
    run_guarded,
    write_csv,
)


def accuracy_compare(metrics: Dict[str, Any], ref: Dict[str, Any], miou_drop: float, dice_drop: float, class_iou_drop: float) -> tuple[bool, str]:
    failures = []
    for key, allowed in [("miou", miou_drop), ("mdice", dice_drop), ("crop_iou", class_iou_drop), ("weed_iou", class_iou_drop)]:
        cur = metric_float(metrics.get(key))
        base = metric_float(ref.get(key))
        if cur is None or base is None:
            failures.append(f"missing_{key}")
            continue
        threshold = max(0.01, base - allowed) if key in {"crop_iou", "weed_iou"} else base - allowed
        if cur < threshold:
            failures.append(f"{key}_drop={base-cur:.6f}_gt_{allowed}")
    return (not failures), ("pass" if not failures else ";".join(failures))


def make_onnx(source_onnx: Path, size: int, out_root: Path, progress: ProgressReporter) -> Path:
    onnx_dir = out_root / f"s{size}" / "onnx"
    onnx_path = onnx_dir / f"segformerb0_s{size}.onnx"
    meta_path = onnx_dir / "make_onnx_meta.json"
    cmd = [
        sys.executable,
        str(PKG_ROOT / "scripts" / "make_segformerb0_onnx_for_size.py"),
        "--source-onnx", str(source_onnx),
        "--output-onnx", str(onnx_path),
        "--imgsz", str(size),
        "--patch-static-shape",
        "--out-json", str(meta_path),
    ]
    res = run_guarded(
        cmd,
        out_root / f"s{size}" / "logs" / "make_onnx.log",
        timeout_s=300,
        progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: prepare ONNX still running"),
    )
    if res.returncode != 0 or not onnx_path.exists():
        raise RuntimeError(f"ONNX preparation failed for size {size}. See {res.log_path}")
    return onnx_path


def build_engine(
    precision: str,
    onnx_path: Path,
    size: int,
    out_root: Path,
    dataset_root: Path,
    manifest: Optional[Path],
    workspace_mb: int,
    max_calib_images: int,
    timeout_s: int,
    progress: ProgressReporter,
) -> Path:
    engine_dir = out_root / f"s{size}" / precision / "engine"
    logs_dir = out_root / f"s{size}" / precision / "logs"
    engine_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if precision in {"fp16", "fp32"}:
        cmd = ["bash", str(PKG_ROOT / "scripts" / f"build_engine_{precision}.sh"), str(onnx_path), str(engine_dir), str(size), str(workspace_mb)]
        res = run_guarded(
            cmd,
            logs_dir / f"build_{precision}.log",
            timeout_s=timeout_s,
            tegrastats_log=logs_dir / f"build_{precision}_tegrastats.log",
            cwd=PKG_ROOT,
            progress_heartbeat=lambda s=size, p=precision: progress.heartbeat(f"size {s}: build {p} engine still running"),
        )
        engine_path = engine_dir / f"{onnx_path.stem}_{precision}.engine"
    elif precision == "int8":
        engine_path = engine_dir / f"{onnx_path.stem}_int8.engine"
        cache_path = engine_dir / f"{onnx_path.stem}_int8_calibration.cache"
        meta_path = engine_dir / f"{onnx_path.stem}_int8_cache_build_meta.json"
        env = dict(os.environ)
        env.update({
            "INT8_MODEL_PATH": str(engine_path),
            "INT8_CALIB_CACHE_PATH": str(cache_path),
            "INT8_META_PATH": str(meta_path),
            "INT8_BUILD_LOG_PATH": str(logs_dir / "build_int8.log"),
            # C++ helper is default and has now been patched to match SegFormer-B0 preprocessing.
            "INT8_BUILD_IMPL": env.get("INT8_BUILD_IMPL", "cpp"),
        })
        cmd = [
            "bash", str(PKG_ROOT / "scripts" / "build_int8_cache_and_engine.sh"),
            str(onnx_path), str(engine_dir), str(dataset_root), str(manifest) if manifest else "", str(size), str(workspace_mb), str(max_calib_images),
        ]
        res = run_guarded(
            cmd,
            logs_dir / "build_int8_wrapper.log",
            timeout_s=timeout_s,
            tegrastats_log=logs_dir / "build_int8_tegrastats.log",
            cwd=PKG_ROOT,
            env=env,
            progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: build int8 engine/cache still running"),
        )
    else:
        raise ValueError(precision)
    if res.returncode != 0 or res.oom_in_dmesg or not engine_path.exists():
        raise RuntimeError(f"{precision} build failed for size {size}. See {res.log_path}")
    return engine_path


def run_accuracy(
    precision: str,
    engine_path: Path,
    size: int,
    out_root: Path,
    dataset_root: Path,
    split_name: str,
    manifest: Optional[Path],
    timeout_s: int,
    progress: ProgressReporter,
) -> Dict[str, Any]:
    acc_dir = out_root / f"s{size}" / precision / "accuracy"
    logs_dir = out_root / f"s{size}" / precision / "logs"
    cmd = [
        sys.executable, str(PKG_ROOT / "scripts" / "run_segformerb0_engine_accuracy.py"),
        "--engine", str(engine_path),
        "--dataset-root", str(dataset_root),
        "--split-name", split_name,
        "--imgsz", str(size),
        "--output-dir", str(acc_dir),
        "--out-json", str(acc_dir / "val_metrics.json"),
    ]
    if manifest:
        cmd.extend(["--manifest", str(manifest)])
    res = run_guarded(
        cmd,
        logs_dir / "accuracy.log",
        timeout_s=timeout_s,
        tegrastats_log=logs_dir / "accuracy_tegrastats.log",
        cwd=PKG_ROOT,
        progress_heartbeat=lambda s=size, p=precision: progress.heartbeat(f"size {s}: {p} accuracy still running"),
    )
    if res.returncode != 0 or res.oom_in_dmesg or not (acc_dir / "val_metrics.json").exists():
        raise RuntimeError(f"accuracy failed for {precision} size {size}. See {res.log_path}")
    metrics = extract_metrics(load_json(acc_dir / "val_metrics.json"))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm FP32/FP16/INT8 accuracy for selected SegFormer-B0 input sizes.")
    ap.add_argument("--source-onnx", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--board-label", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--max-calib-images", type=int, default=256)
    ap.add_argument("--build-timeout-s", type=int, default=2400)
    ap.add_argument("--accuracy-timeout-s", type=int, default=1200)
    ap.add_argument("--miou-drop", type=float, default=0.05)
    ap.add_argument("--dice-drop", type=float, default=0.05)
    ap.add_argument("--class-iou-drop", type=float, default=0.10)
    args = ap.parse_args()

    out_root = Path(args.output_root).expanduser().resolve()
    source_onnx = Path(args.source_onnx).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    rows: List[Dict[str, Any]] = []
    sizes = sorted(set(int(s) for s in args.sizes))
    progress = ProgressReporter(
        total_units=len(sizes) * 7,
        label="SegFormer-B0 precision confirmation",
        heartbeat_s=float(os.environ.get("PROGRESS_HEARTBEAT_S", "60")),
    )
    for size in sizes:
        progress.start(f"size {size}: prepare ONNX")
        onnx_path = make_onnx(source_onnx, size, out_root, progress)
        progress.done(f"size {size}: prepare ONNX ok")
        fp16_metrics = None
        for precision in ["fp16", "fp32", "int8"]:
            row: Dict[str, Any] = {"model": "SegFormer-B0", "task": "semantic_segmentation", "board_label": args.board_label, "size": size, "precision": precision}
            units_to_close_on_failure = 2
            try:
                progress.start(f"size {size}: build {precision} engine")
                engine_path = build_engine(
                    precision,
                    onnx_path,
                    size,
                    out_root,
                    dataset_root,
                    manifest,
                    args.workspace_mb,
                    args.max_calib_images,
                    args.build_timeout_s,
                    progress,
                )
                progress.done(f"size {size}: build {precision} engine ok")
                units_to_close_on_failure = 1
                row["engine_path"] = str(engine_path)
                progress.start(f"size {size}: {precision} accuracy")
                metrics = run_accuracy(
                    precision,
                    engine_path,
                    size,
                    out_root,
                    dataset_root,
                    args.split_name,
                    manifest,
                    args.accuracy_timeout_s,
                    progress,
                )
                progress.done(f"size {size}: {precision} accuracy ok")
                units_to_close_on_failure = 0
                row.update(metrics)
                if precision == "fp16":
                    fp16_metrics = metrics
                    row["decision"] = "fp16_reference_for_size"
                    row["decision_reason"] = "reference"
                elif fp16_metrics is not None:
                    ok, reason = accuracy_compare(metrics, fp16_metrics, args.miou_drop, args.dice_drop, args.class_iou_drop)
                    row["decision"] = "accepted" if ok else "rejected"
                    row["decision_reason"] = reason
                else:
                    row["decision"] = "pending_fp16_reference"
                    row["decision_reason"] = "fp16_not_yet_run"
            except Exception as exc:
                row["decision"] = "failed"
                row["decision_reason"] = str(exc)
                if units_to_close_on_failure:
                    progress.done(f"size {size}: {precision} failed", "failed", units_to_close_on_failure)
            rows.append(row)
            write_csv(out_root / "precision_confirmation_results.csv", rows)
    summary = {
        "status": "ok" if all(row.get("decision") != "failed" for row in rows) else "completed_with_failures",
        "model": "SegFormer-B0",
        "board_label": args.board_label,
        "sizes": sorted(set(int(s) for s in args.sizes)),
        "precisions": ["fp32", "fp16", "int8"],
        "results_csv": str(out_root / "precision_confirmation_results.csv"),
        "note": "INT8 uses a separate calibration cache per input size and patched SegFormer-B0 calibration preprocessing.",
    }
    (out_root / "precision_confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress.done("precision confirmation finalized", "complete", progress.total_units - progress.completed_units)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
