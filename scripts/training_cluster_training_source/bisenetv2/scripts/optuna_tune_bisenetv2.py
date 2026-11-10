#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from bisenetv2_lib import (
    PACKAGE_ROOT,
    Paths,
    StateManager,
    archive_source_snapshot,
    atomic_json_dump,
    build_training_launch_command,
    build_runtime_manifests,
    capture_environment,
    create_or_replace_link,
    dump_yaml,
    ensure_dir,
    estimate_remaining_seconds,
    format_duration_compact,
    load_yaml,
    read_history_rows,
    read_json,
    run_command_streaming,
    sanitized_runtime_env,
    setup_logger,
    utc_now_iso,
)

import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for BiSeNetV2 on CropAndWeed with auto-chain-ready artifact output")
    p.add_argument("--config", required=True, help="Path to segmentation_bisenetv2_train_lock.yaml")
    return p.parse_args()



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



def build_pruner(cfg: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruning_cfg = dict(cfg.get("tuning", {}).get("pruning", {}))
    if not bool(pruning_cfg.get("enabled", False)):
        return optuna.pruners.NopPruner()
    name = str(pruning_cfg.get("name", "median")).strip().lower()
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruning_cfg.get("n_startup_trials", 2)),
            n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 10)),
            interval_steps=int(pruning_cfg.get("interval_steps", 1)),
            n_min_trials=int(pruning_cfg.get("n_min_trials", 1)),
        )
    if name in {"successive_halving", "asha"}:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruning_cfg.get("min_resource", 10)),
            reduction_factor=int(pruning_cfg.get("reduction_factor", 3)),
            min_early_stopping_rate=int(pruning_cfg.get("min_early_stopping_rate", 0)),
            bootstrap_count=int(pruning_cfg.get("bootstrap_count", 0)),
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruning_cfg.get("min_resource", 10)),
            max_resource=int(pruning_cfg.get("max_resource", cfg["tuning"]["max_epochs_per_trial"])),
            reduction_factor=int(pruning_cfg.get("reduction_factor", 3)),
            bootstrap_count=int(pruning_cfg.get("bootstrap_count", 0)),
        )
    raise ValueError(f"Unsupported pruning strategy: {name}")



def save_study_artifacts(study: optuna.Study, optuna_root: Path) -> None:
    ensure_dir(optuna_root)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs", "datetime_start", "datetime_complete"))
    df.to_csv(optuna_root / "trials.csv", index=False)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    best_params = study.best_params if completed else {}
    if best_params:
        dump_yaml(best_params, optuna_root / "best_hyperparameters.yaml")
    else:
        with contextlib.suppress(FileNotFoundError):
            (optuna_root / "best_hyperparameters.yaml").unlink()
    report = {
        "created_at_utc": utc_now_iso(),
        "study_name": study.study_name,
        "best_trial_number": int(study.best_trial.number) if completed else None,
        "best_value": float(study.best_value) if completed else None,
        "best_params": best_params,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
        "n_trials_failed": len([t for t in study.trials if t.state == TrialState.FAIL]),
        "n_trials_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "sqlite_path": str(optuna_root / "study.sqlite3"),
        "trials_csv": str(optuna_root / "trials.csv"),
        "best_hyperparameters_yaml": str(optuna_root / "best_hyperparameters.yaml"),
    }
    atomic_json_dump(report, optuna_root / "tuning_report.json")

    plots_dir = ensure_dir(optuna_root / "plots")
    if not df.empty:
        completed_df = df[df["state"] == "COMPLETE"].sort_values("number")
        if not completed_df.empty:
            values = completed_df["value"].astype(float).tolist()
            trials = completed_df["number"].astype(int).tolist()
            best_so_far = []
            running_best = None
            for value in values:
                running_best = value if running_best is None else max(running_best, value)
                best_so_far.append(running_best)
            plt.figure(figsize=(8, 5))
            plt.plot(trials, values, marker="o", label="trial value")
            plt.plot(trials, best_so_far, label="best so far")
            plt.xlabel("trial")
            plt.ylabel(report.get("best_value") is not None and "metric" or "value")
            plt.title("Optuna optimization history")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "optuna_history.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(trials, best_so_far, marker="o")
            plt.xlabel("trial")
            plt.ylabel("best score")
            plt.title("Best-so-far curve")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "optuna_best_so_far.png", dpi=160)
            plt.close()

        with contextlib.suppress(Exception):
            from optuna.visualization.matplotlib import plot_param_importances
            fig = plot_param_importances(study)
            fig.figure.savefig(plots_dir / "optuna_param_importance.png", bbox_inches="tight", dpi=160)
            plt.close(fig.figure)



def extract_intermediate_values(history_jsonl: Path, metric_name: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for row in read_history_rows(history_jsonl):
        epoch = row.get("epoch")
        val_metrics = row.get("val_metrics", {})
        if not isinstance(epoch, int) or not isinstance(val_metrics, dict):
            continue
        metric = val_metrics.get(metric_name)
        if metric is None:
            continue
        try:
            values[int(epoch)] = float(metric)
        except (TypeError, ValueError):
            continue
    return values



def mark_stale_running_trials_failed(study: optuna.Study, log) -> None:
    stale = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    if not stale:
        return
    log.warning("Found %d stale RUNNING Optuna trial(s). Marking them FAILED so tuning can continue.", len(stale))
    for trial in stale:
        with contextlib.suppress(Exception):
            study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)



def build_train_command(
    cfg_path: Path,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    epochs: int,
    train_batch_total: int,
    eval_batch_total: int,
    workers_total: int,
    tuned_params_yaml: Path,
    resume: Optional[Path],
    report_json: Path,
    history_jsonl: Path,
    class_histogram_json: Path,
) -> List[str]:
    entry = PACKAGE_ROOT / "scripts" / "train_bisenetv2_entry.py"
    entry_args = [
        "--config", str(cfg_path),
        "--run-kind", "trial",
        "--run-dir", str(run_dir),
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--epochs", str(int(epochs)),
        "--train-batch-total", str(int(train_batch_total)),
        "--eval-batch-total", str(int(eval_batch_total)),
        "--workers-total", str(int(workers_total)),
        "--tuned-params-yaml", str(tuned_params_yaml),
        "--report-json", str(report_json),
        "--history-jsonl", str(history_jsonl),
        "--class-histogram-json", str(class_histogram_json),
    ]
    if resume is not None:
        entry_args += ["--resume", str(resume)]
    return build_training_launch_command(cfg, entry=entry, entry_args=entry_args)



def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg["_resolved_config_path"] = str(cfg_path)
    paths = Paths.from_config(cfg)
    ensure_dir(paths.runtime_root)
    ensure_dir(paths.logs_dir)
    log = setup_logger(paths.logs_dir / f"optuna_{int(time.time())}.log")
    capture_environment(paths, log, "optuna")
    archive_source_snapshot(paths, cfg_path, log)
    state = StateManager(paths.state_file)
    runtime_report = build_runtime_manifests(cfg, paths, state, log)
    train_manifest = Path(runtime_report["train_manifest"])
    val_manifest = Path(runtime_report["val_manifest"])
    tuning_cfg = dict(cfg.get("tuning", {}))
    optuna_root = ensure_dir(paths.optuna_root)
    trials_root = ensure_dir(optuna_root / "trials")
    sqlite_path = optuna_root / "study.sqlite3"
    sampler = TPESampler(seed=int(tuning_cfg.get("sampler_seed", 0)))
    pruner = build_pruner(cfg)

    storage = f"sqlite:///{sqlite_path}"
    study = optuna.create_study(
        study_name=str(tuning_cfg.get("study_name", "bisenetv2_optuna_cropandweed")),
        direction=str(tuning_cfg.get("direction", "maximize")),
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    if bool(tuning_cfg.get("mark_stale_running_trials_failed", True)):
        mark_stale_running_trials_failed(study, log)

    metric_name = str(tuning_cfg.get("metric", "miou"))
    max_epochs = int(tuning_cfg.get("max_epochs_per_trial", 60))
    target_trials = int(tuning_cfg.get("n_trials", 12))
    completed_already = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining_trials = max(0, target_trials - completed_already)
    log.info("Optuna study=%s | target_complete_trials=%d | already_complete=%d | remaining=%d", study.study_name, target_trials, completed_already, remaining_trials)
    atomic_json_dump({
        "created_at_utc": utc_now_iso(),
        "study_name": study.study_name,
        "target_complete_trials": target_trials,
        "already_complete_trials": completed_already,
        "remaining_trials": remaining_trials,
        "metric": metric_name,
        "max_epochs_per_trial": max_epochs,
        "sqlite_path": str(sqlite_path),
    }, optuna_root / "current_trial_status.json")

    if remaining_trials <= 0:
        save_study_artifacts(study, optuna_root)
        log.info("Study already has %d complete trials. Nothing to do.", completed_already)
        return

    start_tuning = time.time()

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {}
        for name, spec in tuning_cfg.get("search_space", {}).items():
            params[name] = suggest_from_spec(trial, name, spec)

        trial_dir = ensure_dir(trials_root / f"trial_{trial.number:04d}")
        params_yaml = trial_dir / "trial_params.yaml"
        dump_yaml(params, params_yaml)
        report_json = trial_dir / "trial_summary.json"
        history_jsonl = trial_dir / "history" / "train_history.jsonl"
        class_hist_json = trial_dir / "history" / "class_histogram.json"
        trial_status_json = trial_dir / "current_trial_status.json"
        last_ckpt = trial_dir / "checkpoints" / "last.pt"
        resume = last_ckpt if last_ckpt.exists() else None
        train_log = trial_dir / "trial_train.log"
        cmd = build_train_command(
            cfg_path,
            cfg,
            run_dir=trial_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            epochs=max_epochs,
            train_batch_total=int(tuning_cfg.get("batch", cfg.get("training", {}).get("batch", 32))),
            eval_batch_total=int(cfg.get("training", {}).get("eval_batch", 8)),
            workers_total=int(tuning_cfg.get("workers", cfg.get("training", {}).get("workers", 16))),
            tuned_params_yaml=params_yaml,
            resume=resume,
            report_json=report_json,
            history_jsonl=history_jsonl,
            class_histogram_json=class_hist_json,
        )

        env = sanitized_runtime_env()
        process = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        reported_epochs: set[int] = set()
        started = time.time()
        with train_log.open("w", encoding="utf-8") as tee:
            tee.write(f"[COMMAND] {' '.join(str(x) for x in cmd)}\n")
            tee.write(f"[RANK_LOG_DIR] {trial_dir / 'rank_logs'}\n")
            tee.flush()
            assert process.stdout is not None
            while True:
                raw_line = process.stdout.readline()
                line = raw_line.decode("utf-8", errors="replace") if raw_line else ""
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    tee.write(line)
                    tee.flush()
                status = extract_intermediate_values(history_jsonl, metric_name)
                for epoch in sorted(status):
                    if epoch in reported_epochs:
                        continue
                    value = float(status[epoch])
                    trial.report(value, step=epoch)
                    reported_epochs.add(epoch)
                    atomic_json_dump(
                        {
                            "trial_number": int(trial.number),
                            "created_at_utc": utc_now_iso(),
                            "status": "running",
                            "elapsed": format_duration_compact(time.time() - started),
                            "last_reported_epoch": int(epoch),
                            "last_reported_value": value,
                            "params": params,
                        },
                        trial_status_json,
                    )
                    if trial.should_prune():
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=30)
                        atomic_json_dump(
                            {
                                "trial_number": int(trial.number),
                                "created_at_utc": utc_now_iso(),
                                "status": "pruned",
                                "pruned_epoch": int(epoch),
                                "pruned_value": value,
                                "params": params,
                            },
                            trial_status_json,
                        )
                        raise optuna.TrialPruned(f"Pruned at epoch {epoch} with {metric_name}={value:.4f}")
                if process.poll() is not None:
                    break
                time.sleep(2)

        returncode = process.wait()
        if returncode != 0:
            atomic_json_dump(
                {
                    "trial_number": int(trial.number),
                    "created_at_utc": utc_now_iso(),
                    "status": "failed",
                    "returncode": int(returncode),
                    "params": params,
                },
                trial_status_json,
            )
            raise RuntimeError(f"Trial {trial.number} training subprocess failed with return code {returncode}")

        summary = read_json(report_json)
        if not summary:
            raise RuntimeError(f"Trial {trial.number} completed but summary JSON is missing: {report_json}")
        value = float(summary.get("best_metric", summary.get("best_metrics", {}).get("val_metrics", {}).get(metric_name, 0.0)))
        trial.set_user_attr("run_dir", str(trial_dir))
        trial.set_user_attr("report_json", str(report_json))
        trial.set_user_attr("best_checkpoint", str(trial_dir / "checkpoints" / "best.pt"))
        trial.set_user_attr("best_metric", value)
        atomic_json_dump(
            {
                "trial_number": int(trial.number),
                "created_at_utc": utc_now_iso(),
                "status": "complete",
                "best_value": value,
                "elapsed": format_duration_compact(time.time() - started),
                "params": params,
            },
            trial_status_json,
        )
        return value

    try:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True, show_progress_bar=False, catch=(RuntimeError,))
    finally:
        save_study_artifacts(study, optuna_root)
        tuning_report = read_json(optuna_root / "tuning_report.json")
        completed = int(tuning_report.get("n_trials_complete", 0))
        remaining = max(0, target_trials - completed)
        elapsed = time.time() - start_tuning
        atomic_json_dump(
            {
                "created_at_utc": utc_now_iso(),
                "study_name": study.study_name,
                "target_complete_trials": target_trials,
                "n_complete": completed,
                "remaining_to_target": remaining,
                "elapsed_seconds": elapsed,
                "elapsed_human": format_duration_compact(elapsed),
                "estimated_remaining_seconds": estimate_remaining_seconds(elapsed, min(1.0, completed / max(1, target_trials))),
            },
            optuna_root / "progress_report.json",
        )

    if (optuna_root / "best_hyperparameters.yaml").exists():
        create_or_replace_link(optuna_root / "best_hyperparameters.yaml", paths.reports_dir / "best_hyperparameters.yaml")
    log.info("Optuna finished. Best params file: %s", optuna_root / "best_hyperparameters.yaml")


if __name__ == "__main__":
    main()
