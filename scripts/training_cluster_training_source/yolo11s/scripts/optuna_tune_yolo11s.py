#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from tqdm.auto import tqdm

from stage2_yolo11s_pipeline import (
    PACKAGE_ROOT,
    Paths,
    StateManager,
    EpochProgressMonitor,
    atomic_json_dump,
    build_runtime_manifests,
    capture_environment,
    create_or_replace_link,
    dump_yaml,
    ensure_dir,
    import_ultralytics,
    important_file_metadata,
    load_yaml,
    parse_results_csv,
    resolve_tuned_params_path,
    setup_logger,
    utc_now_iso,
    archive_source_snapshot,
    sha256_file,
)


ALLOWED_TUNED_KEYS = {
    "lr0", "lrf", "weight_decay", "warmup_epochs", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "mosaic", "mixup"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for YOLO11s on CottonWeedDet12 with auto-chain-ready artifact output.")
    parser.add_argument("--config", required=True, help="Path to detection_yolo11s_train_lock.yaml")
    return parser.parse_args()


def get_metric_from_csv(csv_report: Dict[str, Any], metric_name: str) -> float:
    if metric_name == "map50_95":
        value = csv_report.get("best_metric_value")
        if value is not None:
            return float(value)
        last_row = csv_report.get("last_row", {})
        for candidate in ["metrics/mAP50-95(B)", "metrics/fitness"]:
            if candidate in last_row and last_row[candidate] not in (None, ""):
                return float(last_row[candidate])
    raise RuntimeError(f"Could not extract metric '{metric_name}' from results.csv report")


def suggest_from_spec(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "float_log":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported search space type for {name}: {kind}")


def mark_stale_running_trials_failed(study: optuna.Study, log: logging.Logger) -> int:
    count = 0
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.RUNNING:
            try:
                study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)  # type: ignore[attr-defined]
                count += 1
            except Exception as exc:
                log.warning("Could not mark stale RUNNING trial %s as FAIL: %s", t.number, exc)
    return count


def save_study_artifacts(study: optuna.Study, optuna_root: Path, log: logging.Logger) -> None:
    ensure_dir(optuna_root)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs", "datetime_start", "datetime_complete"))
    df.to_csv(optuna_root / "trials.csv", index=False)

    best_params = study.best_params if len([t for t in study.trials if t.state == TrialState.COMPLETE]) else {}
    dump_yaml(best_params, optuna_root / "best_hyperparameters.yaml")

    report = {
        "created_at_utc": utc_now_iso(),
        "study_name": study.study_name,
        "best_trial_number": int(study.best_trial.number) if best_params else None,
        "best_value": float(study.best_value) if best_params else None,
        "best_params": best_params,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_trials_failed": len([t for t in study.trials if t.state == TrialState.FAIL]),
        "n_trials_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "sqlite_path": str(optuna_root / "study.sqlite3"),
        "trials_csv": str(optuna_root / "trials.csv"),
        "best_hyperparameters_yaml": str(optuna_root / "best_hyperparameters.yaml"),
    }
    atomic_json_dump(report, optuna_root / "tuning_report.json")

    # Optional plots.
    with contextlib.suppress(Exception):
        import matplotlib.pyplot as plt  # type: ignore
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        fig = plot_optimization_history(study)
        fig.figure.savefig(optuna_root / "optimization_history.png", bbox_inches="tight", dpi=160)
        plt.close(fig.figure)

        fig2 = plot_param_importances(study)
        fig2.figure.savefig(optuna_root / "param_importances.png", bbox_inches="tight", dpi=160)
        plt.close(fig2.figure)


def build_trial_train_args(cfg: Dict[str, Any], paths: Paths, trial_params: Dict[str, Any], trial_dir: Path) -> Dict[str, Any]:
    runtime_full_yaml = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"
    args: Dict[str, Any] = {
        "data": str(runtime_full_yaml),
        "imgsz": int(cfg["training"]["imgsz"]),
        "epochs": int(cfg["tuning"]["max_epochs_per_trial"]),
        "batch": int(cfg["tuning"]["batch"]),
        "device": cfg["tuning"]["device"],
        "workers": int(cfg["tuning"]["workers"]),
        "seed": int(cfg["training"]["seed"]),
        "deterministic": bool(cfg["training"]["deterministic"]),
        "cache": bool(cfg["training"]["cache"]),
        "amp": bool(cfg["training"]["amp"]),
        "project": str(trial_dir.parent),
        "name": trial_dir.name,
        "exist_ok": True,
        "save": True,
        "save_period": int(cfg["tuning"]["save_period"]),
        "plots": bool(cfg["tuning"]["plots"]),
        "verbose": True,
        "pretrained": bool(cfg["training"]["pretrained"]),
        "optimizer": cfg["training"]["optimizer"],
        "close_mosaic": int(cfg["training"]["close_mosaic"]),
        "single_cls": False,
        "patience": int(cfg["tuning"]["patience"]),
        "cos_lr": bool(cfg["training"].get("cos_lr", True)),
    }
    args.update(trial_params)
    return args


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    paths = Paths.from_config(cfg)
    optuna_root = ensure_dir(paths.runtime_root / "optuna")
    logs_dir = ensure_dir(optuna_root / "logs")
    log = setup_logger(logs_dir / f"optuna_tuning_{int(time.time())}.log")

    ensure_dir(paths.runtime_root)
    ensure_dir(paths.logs_dir)
    capture_environment(paths, log, "tune")
    archive_source_snapshot(paths, cfg_path, log)
    state = StateManager(paths.state_file)
    build_runtime_manifests(cfg, paths, state, log)

    if not bool(cfg.get("tuning", {}).get("enabled", False)):
        log.info("Tuning is disabled in config. Nothing to do.")
        return

    storage_url = f"sqlite:///{(optuna_root / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=str(cfg["tuning"]["study_name"]),
        direction=str(cfg["tuning"].get("direction", "maximize")),
        sampler=TPESampler(seed=int(cfg["tuning"].get("sampler_seed", cfg["training"]["seed"]))),
        storage=storage_url,
        load_if_exists=True,
    )

    if bool(cfg["tuning"].get("mark_stale_running_trials_failed", True)):
        stale = mark_stale_running_trials_failed(study, log)
        if stale:
            log.info("Marked %d stale RUNNING trial(s) as FAIL before resuming.", stale)

    target_complete = int(cfg["tuning"]["n_trials"])
    complete_count = len([t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE])
    if complete_count >= target_complete:
        log.info("Study already has %d complete trials (target=%d).", complete_count, target_complete)
        save_study_artifacts(study, optuna_root, log)
        return

    metric_name = str(cfg["tuning"].get("metric", "map50_95"))
    search_space = dict(cfg["tuning"]["search_space"])
    YOLO = import_ultralytics()
    trials_root = ensure_dir(optuna_root / "trials")
    pbar = tqdm(total=target_complete, initial=complete_count, desc="Optuna trials", unit="trial", dynamic_ncols=True)

    attempt_limit = target_complete * 3
    attempts = 0
    while complete_count < target_complete and attempts < attempt_limit:
        attempts += 1
        trial = study.ask()
        trial_dir = trials_root / f"trial_{trial.number:04d}"
        ensure_dir(trial_dir)

        trial_params = {name: suggest_from_spec(trial, name, spec) for name, spec in search_space.items()}
        dump_yaml(trial_params, trial_dir / "suggested_params.yaml")
        atomic_json_dump({"trial_number": trial.number, "params": trial_params, "created_at_utc": utc_now_iso()}, trial_dir / "trial_info.json")

        trial_args = build_trial_train_args(cfg, paths, trial_params, trial_dir)
        dump_yaml(trial_args, trial_dir / "effective_train_args.yaml")
        results_csv = trial_dir / "results.csv"
        monitor = EpochProgressMonitor(results_csv=results_csv, total_epochs=int(cfg["tuning"]["max_epochs_per_trial"]), desc=f"Trial {trial.number}")

        log.info("Starting Optuna trial %s with params=%s", trial.number, trial_params)
        started = time.time()
        try:
            model = YOLO(cfg["training"]["model"])
            monitor.start()
            try:
                model.train(**trial_args)
            finally:
                monitor.stop()

            csv_report = parse_results_csv(results_csv)
            objective_value = get_metric_from_csv(csv_report, metric_name)
            duration = time.time() - started
            study.tell(trial, objective_value)
            trial_summary = {
                "trial_number": trial.number,
                "status": "COMPLETE",
                "objective_metric": metric_name,
                "objective_value": objective_value,
                "duration_seconds": duration,
                "params": trial_params,
                "csv_report": csv_report,
                "best_pt": important_file_metadata(trial_dir / "weights" / "best.pt"),
                "last_pt": important_file_metadata(trial_dir / "weights" / "last.pt"),
            }
            atomic_json_dump(trial_summary, trial_dir / "trial_summary.json")
        except Exception as exc:
            monitor.stop()
            log.exception("Trial %s failed: %s", trial.number, exc)
            study.tell(trial, state=TrialState.FAIL)
            atomic_json_dump({
                "trial_number": trial.number,
                "status": "FAIL",
                "error": str(exc),
                "params": trial_params,
            }, trial_dir / "trial_summary.json")

        save_study_artifacts(study, optuna_root, log)
        complete_count = len([t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE])
        pbar.n = complete_count
        pbar.refresh()

    pbar.close()

    save_study_artifacts(study, optuna_root, log)
    tuned_path = resolve_tuned_params_path(cfg, paths)
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    best_params = load_yaml(optuna_root / "best_hyperparameters.yaml") if (optuna_root / "best_hyperparameters.yaml").exists() else {}
    filtered = {k: v for k, v in dict(best_params).items() if k in ALLOWED_TUNED_KEYS}
    if not filtered:
        raise RuntimeError("Optuna completed without a valid best_hyperparameters.yaml. Full training will not be auto-started.")

    dump_yaml(filtered, tuned_path)
    log.info("Copied best hyperparameters to %s", tuned_path)

    best_trial = study.best_trial
    best_trial_dir = trials_root / f"trial_{best_trial.number:04d}"
    create_or_replace_link(best_trial_dir, optuna_root / "best_trial_dir")
    best_pt = best_trial_dir / "weights" / "best.pt"
    if best_pt.exists():
        create_or_replace_link(best_pt, optuna_root / "best_trial_best.pt")

    summary = {
        "created_at_utc": utc_now_iso(),
        "target_complete_trials": target_complete,
        "complete_trials": len([t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]),
        "failed_trials": len([t for t in study.get_trials(deepcopy=False) if t.state == TrialState.FAIL]),
        "best_trial_number": int(study.best_trial.number) if filtered else None,
        "best_value": float(study.best_value) if filtered else None,
        "best_hyperparameters_yaml": str(tuned_path),
        "trials_csv": str(optuna_root / "trials.csv"),
        "sqlite_path": str(optuna_root / "study.sqlite3"),
    }
    atomic_json_dump(summary, optuna_root / "tuning_stage_summary.json")
    log.info("Optuna tuning complete. Summary written to %s", optuna_root / "tuning_stage_summary.json")


if __name__ == "__main__":
    main()
