#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deeplabv3plus_lib import (
    PACKAGE_ROOT,
    Paths,
    StateManager,
    archive_source_snapshot,
    atomic_json_dump,
    build_runtime_manifests,
    capture_environment,
    collect_known_artifacts,
    create_or_replace_link,
    dump_yaml,
    ensure_dir,
    evaluate_checkpoint_on_manifest,
    export_model_to_onnx,
    generate_qualitative_panels,
    important_file_metadata,
    load_yaml,
    read_json,
    build_training_launch_command,
    resolve_tuned_params_path,
    run_command_streaming,
    setup_logger,
    utc_now_iso,
)



def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)



def build_train_command(
    cfg: Dict[str, Any],
    *,
    run_kind: str,
    run_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    epochs: int,
    train_batch_total: int,
    eval_batch_total: int,
    workers_total: int,
    tuned_params_yaml: Optional[Path],
    resume: Optional[Path],
    report_json: Path,
    history_jsonl: Path,
    class_histogram_json: Path,
) -> List[str]:
    entry = PACKAGE_ROOT / "scripts" / "train_deeplabv3plus_entry.py"
    entry_args = [
        "--config",
        str(_config_path(cfg)),
        "--run-kind",
        run_kind,
        "--run-dir",
        str(run_dir),
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--epochs",
        str(int(epochs)),
        "--train-batch-total",
        str(int(train_batch_total)),
        "--eval-batch-total",
        str(int(eval_batch_total)),
        "--workers-total",
        str(int(workers_total)),
        "--report-json",
        str(report_json),
        "--history-jsonl",
        str(history_jsonl),
        "--class-histogram-json",
        str(class_histogram_json),
    ]
    if tuned_params_yaml is not None:
        entry_args += ["--tuned-params-yaml", str(tuned_params_yaml)]
    if resume is not None:
        entry_args += ["--resume", str(resume)]
    return build_training_launch_command(cfg, entry=entry, entry_args=entry_args)



def _config_path(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("_resolved_config_path")
    if raw:
        return Path(str(raw)).resolve()
    raise RuntimeError("Resolved config path missing from cfg")



def run_smoke_training(cfg: Dict[str, Any], paths: Paths, state: StateManager, log) -> Dict[str, Any]:
    stage_name = "smoke_training"
    report_path = paths.reports_dir / "smoke_training_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        runtime_report = build_runtime_manifests(cfg, paths, state, log)
        train_manifest = Path(runtime_report["smoke_train_manifest"])
        val_manifest = Path(runtime_report["smoke_val_manifest"])
        smoke_dir = paths.runtime_root / "train_runs" / "smoke_check"
        ensure_dir(smoke_dir)
        last_ckpt = smoke_dir / "checkpoints" / "last.pt"
        resume = last_ckpt if last_ckpt.exists() else None
        report_json = smoke_dir / "smoke_summary.json"
        history_jsonl = smoke_dir / "history" / "train_history.jsonl"
        class_hist_json = smoke_dir / "history" / "class_histogram.json"
        cmd = build_train_command(
            cfg,
            run_kind="smoke",
            run_dir=smoke_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            epochs=int(cfg.get("training", {}).get("smoke_epochs", 2)),
            train_batch_total=int(cfg.get("training", {}).get("smoke_batch", 8)),
            eval_batch_total=int(cfg.get("training", {}).get("smoke_batch", 8)),
            workers_total=int(cfg.get("training", {}).get("smoke_workers", min(int(cfg.get("training", {}).get("workers", 8)), 4))),
            tuned_params_yaml=None,
            resume=resume,
            report_json=report_json,
            history_jsonl=history_jsonl,
            class_histogram_json=class_hist_json,
        )
        train_log = paths.logs_dir / "smoke_train.log"
        log.info("Per-rank debug logs for smoke training: %s", smoke_dir / "rank_logs")
        text = run_command_streaming(cmd, log, cwd=PACKAGE_ROOT, tee_path=train_log)
        summary = read_json(report_json)
        artifact_report = collect_known_artifacts(smoke_dir / "plots", paths.paper_assets_dir / "01_smoke_training", log)
        report = {
            "run_dir": str(smoke_dir),
            "train_log": str(train_log),
            "best_checkpoint": summary.get("best_checkpoint", important_file_metadata(smoke_dir / "checkpoints" / "best.pt")),
            "last_checkpoint": summary.get("last_checkpoint", important_file_metadata(smoke_dir / "checkpoints" / "last.pt")),
            "summary_json": str(report_json),
            "artifact_report": artifact_report,
            "output_chars": len(text),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def run_full_training(cfg: Dict[str, Any], paths: Paths, state: StateManager, log) -> Dict[str, Any]:
    stage_name = "full_training"
    report_path = paths.reports_dir / "full_training_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        runtime_report = build_runtime_manifests(cfg, paths, state, log)
        train_manifest = Path(runtime_report["train_manifest"])
        val_manifest = Path(runtime_report["val_manifest"])
        train_dir = paths.runtime_root / "train_runs" / "train"
        ensure_dir(train_dir)
        tuned_yaml = resolve_tuned_params_path(cfg, paths)
        last_ckpt = train_dir / "checkpoints" / "last.pt"
        resume = last_ckpt if last_ckpt.exists() else None
        report_json = train_dir / "full_summary.json"
        history_jsonl = train_dir / "history" / "train_history.jsonl"
        class_hist_json = train_dir / "history" / "class_histogram.json"
        cmd = build_train_command(
            cfg,
            run_kind="full",
            run_dir=train_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            epochs=int(cfg.get("training", {}).get("epochs", 300)),
            train_batch_total=int(cfg.get("training", {}).get("batch", 32)),
            eval_batch_total=int(cfg.get("training", {}).get("eval_batch", 8)),
            workers_total=int(cfg.get("training", {}).get("workers", 16)),
            tuned_params_yaml=tuned_yaml,
            resume=resume,
            report_json=report_json,
            history_jsonl=history_jsonl,
            class_histogram_json=class_hist_json,
        )
        dump_yaml(
            {
                "config_path": str(_config_path(cfg)),
                "train_manifest": str(train_manifest),
                "val_manifest": str(val_manifest),
                "tuned_params_yaml": str(tuned_yaml) if tuned_yaml else None,
                "resume": str(resume) if resume else None,
            },
            paths.reports_dir / "effective_training_args.yaml",
        )
        train_log = paths.logs_dir / "full_train.log"
        log.info("Per-rank debug logs for full training: %s", train_dir / "rank_logs")
        text = run_command_streaming(cmd, log, cwd=PACKAGE_ROOT, tee_path=train_log)
        summary = read_json(report_json)
        best_ckpt = train_dir / "checkpoints" / "best.pt"
        last_ckpt_final = train_dir / "checkpoints" / "last.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint missing after full training: {best_ckpt}")
        create_or_replace_link(best_ckpt, paths.artifacts_dir / "best.pt")
        if last_ckpt_final.exists():
            create_or_replace_link(last_ckpt_final, paths.artifacts_dir / "last.pt")
        plot_dir = train_dir / "plots"
        for src_name, dst_name in [
            ("training_loss_curve.png", "training_loss_curve.png"),
            ("training_miou_curve.png", "training_miou_curve.png"),
            ("training_dice_curve.png", "training_dice_curve.png"),
            ("lr_curve.png", "lr_curve.png"),
            ("class_histogram.png", "class_histogram.png"),
        ]:
            copy_if_exists(plot_dir / src_name, paths.paper_assets_dir / dst_name)
        copy_if_exists(plot_dir / "train_history.csv", paths.reports_dir / "train_history.csv")
        report = {
            "run_dir": str(train_dir),
            "train_log": str(train_log),
            "summary_json": str(report_json),
            "best_checkpoint": important_file_metadata(best_ckpt),
            "last_checkpoint": important_file_metadata(last_ckpt_final),
            "history_jsonl": str(history_jsonl),
            "artifact_report": collect_known_artifacts(plot_dir, paths.paper_assets_dir / "02_full_training_plots", log),
            "output_chars": len(text),
            "summary": summary,
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def run_validation(cfg: Dict[str, Any], paths: Paths, state: StateManager, log, split_name: str) -> Dict[str, Any]:
    if split_name not in {"val", "test"}:
        raise ValueError(split_name)
    stage_name = f"validation_{split_name}"
    report_path = paths.reports_dir / f"validation_{split_name}.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        runtime_report = build_runtime_manifests(cfg, paths, state, log)
        manifest_path = Path(runtime_report[f"{split_name}_manifest"])
        best_ckpt = paths.artifacts_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint missing before validation: {best_ckpt}")
        output_dir = paths.validation_root / f"val_{split_name}"
        report = evaluate_checkpoint_on_manifest(cfg, best_ckpt, manifest_path, output_dir, split_name)
        if split_name == "test":
            copy_if_exists(output_dir / "per_class_iou.png", paths.paper_assets_dir / "per_class_iou_test.png")
            copy_if_exists(output_dir / "per_class_dice.png", paths.paper_assets_dir / "per_class_dice_test.png")
            copy_if_exists(output_dir / "confusion_matrix.png", paths.paper_assets_dir / "confusion_matrix_test.png")
            copy_if_exists(output_dir / "per_class_metrics.csv", paths.reports_dir / "per_class_metrics_test.csv")
            copy_if_exists(output_dir / "confusion_matrix.csv", paths.reports_dir / "confusion_matrix_test.csv")
        else:
            copy_if_exists(output_dir / "per_class_iou.png", paths.paper_assets_dir / "per_class_iou_val.png")
            copy_if_exists(output_dir / "per_class_dice.png", paths.paper_assets_dir / "per_class_dice_val.png")
            copy_if_exists(output_dir / "confusion_matrix.png", paths.paper_assets_dir / "confusion_matrix_val.png")
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def run_export_onnx(cfg: Dict[str, Any], paths: Paths, state: StateManager, log) -> Dict[str, Any]:
    stage_name = "export_onnx"
    report_path = paths.reports_dir / "export_onnx.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        if not bool(cfg.get("export", {}).get("enable_onnx_export", True)):
            report = {"skipped": True, "reason": "export.enable_onnx_export is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report
        best_ckpt = paths.artifacts_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint missing before ONNX export: {best_ckpt}")
        import torch
        onnx_path = paths.exports_dir / "best.onnx"
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        report = export_model_to_onnx(cfg, best_ckpt, onnx_path, device=device)
        copy_if_exists(onnx_path, paths.paper_assets_dir / "best.onnx")
        report["artifact_report"] = collect_known_artifacts(paths.exports_dir, paths.paper_assets_dir / "04_export_onnx", log)
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def run_qualitative_predictions(cfg: Dict[str, Any], paths: Paths, state: StateManager, log) -> Dict[str, Any]:
    stage_name = "qualitative_predictions"
    report_path = paths.reports_dir / "qualitative_predictions.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        if not bool(cfg.get("qualitative", {}).get("enable", True)):
            report = {"skipped": True, "reason": "qualitative.enable is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report
        runtime_report = build_runtime_manifests(cfg, paths, state, log)
        best_ckpt = paths.artifacts_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint missing before qualitative stage: {best_ckpt}")
        val_report = generate_qualitative_panels(
            cfg,
            best_ckpt,
            Path(runtime_report["paper_val_manifest"]),
            paths.qualitative_root / "val",
            limit=int(cfg.get("qualitative", {}).get("val_subset_size", 16)),
        )
        test_report = generate_qualitative_panels(
            cfg,
            best_ckpt,
            Path(runtime_report["paper_test_manifest"]),
            paths.qualitative_root / "test",
            limit=int(cfg.get("qualitative", {}).get("test_subset_size", 32)),
        )
        copy_if_exists(paths.qualitative_root / "test" / "contact_sheet.png", paths.paper_assets_dir / "qualitative_contact_sheet_test.png")
        copy_if_exists(paths.qualitative_root / "val" / "contact_sheet.png", paths.paper_assets_dir / "qualitative_contact_sheet_val.png")
        report = {
            "val": val_report,
            "test": test_report,
            "artifact_report": collect_known_artifacts(paths.qualitative_root, paths.paper_assets_dir / "05_qualitative_predictions", log),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def create_output_manifest(paths: Paths) -> Dict[str, Any]:
    rows = []
    for p in sorted(paths.runtime_root.rglob("*")):
        if p.is_file() or p.is_symlink():
            try:
                st = p.stat()
                rows.append(
                    {
                        "relative_path": str(p.relative_to(paths.runtime_root)),
                        "is_symlink": p.is_symlink(),
                        "size_bytes": st.st_size,
                        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                    }
                )
            except Exception:
                continue
    manifest_path = paths.runtime_root / "OUTPUT_FILE_MANIFEST.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    report = {"manifest_csv": str(manifest_path), "file_count": len(rows)}
    atomic_json_dump(report, paths.reports_dir / "output_manifest.json")
    return report



def write_final_summary(cfg: Dict[str, Any], paths: Paths, state: StateManager) -> None:
    stages = state.state.get("stages", {})
    summary = {
        "created_at_utc": utc_now_iso(),
        "project_name": cfg.get("project", {}).get("name"),
        "runtime_root": str(paths.runtime_root),
        "stages": stages,
        "best_checkpoint": important_file_metadata(paths.artifacts_dir / "best.pt") if (paths.artifacts_dir / "best.pt").exists() else {"exists": False},
        "onnx_export": read_json(paths.reports_dir / "export_onnx.json"),
        "validation_val": read_json(paths.reports_dir / "validation_val.json"),
        "validation_test": read_json(paths.reports_dir / "validation_test.json"),
        "qualitative": read_json(paths.reports_dir / "qualitative_predictions.json"),
        "tuning": read_json(paths.optuna_root / "tuning_report.json"),
    }
    atomic_json_dump(summary, paths.runtime_root / "FINAL_SUMMARY.json")
    lines = [
        "# Final summary — DeepLabV3++",
        "",
        f"- generated_at_utc: {summary['created_at_utc']}",
        f"- runtime_root: {summary['runtime_root']}",
        f"- best_checkpoint_exists: {summary['best_checkpoint'].get('exists', False)}",
    ]
    for stage_name, stage_info in stages.items():
        lines.append(f"- {stage_name}: {stage_info.get('status', 'unknown')}")
    (paths.runtime_root / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def print_status(paths: Paths) -> None:
    print(json.dumps(read_json(paths.state_file), indent=2, ensure_ascii=False))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 DeepLabV3++ pipeline for CropAndWeed on Training_Cluster")
    p.add_argument("--config", required=True, help="Path to segmentation_deeplabv3plus_train_lock.yaml")
    p.add_argument("--mode", required=True, choices=["prepare", "smoke", "full", "status"], help="Pipeline mode")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg["_resolved_config_path"] = str(cfg_path)
    cfg["_config_path_name"] = str(cfg_path)
    paths = Paths.from_config(cfg)
    ensure_dir(paths.runtime_root)
    ensure_dir(paths.logs_dir)
    if args.mode == "status":
        print_status(paths)
        return
    log = setup_logger(paths.logs_dir / f"pipeline_{args.mode}_{int(time.time())}.log")
    capture_environment(paths, log, args.mode)
    archive_source_snapshot(paths, cfg_path, log)
    state = StateManager(paths.state_file)
    try:
        build_runtime_manifests(cfg, paths, state, log)
        if args.mode == "prepare":
            create_output_manifest(paths)
            write_final_summary(cfg, paths, state)
            return
        if args.mode == "smoke":
            run_smoke_training(cfg, paths, state, log)
            create_output_manifest(paths)
            write_final_summary(cfg, paths, state)
            return
        if args.mode == "full":
            if bool(cfg.get("training", {}).get("run_smoke_before_full", True)):
                run_smoke_training(cfg, paths, state, log)
            run_full_training(cfg, paths, state, log)
            run_validation(cfg, paths, state, log, "val")
            run_validation(cfg, paths, state, log, "test")
            run_export_onnx(cfg, paths, state, log)
            run_qualitative_predictions(cfg, paths, state, log)
            create_output_manifest(paths)
            write_final_summary(cfg, paths, state)
            return
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
