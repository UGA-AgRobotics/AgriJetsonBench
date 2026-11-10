#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.trial import Trial as OptunaTrial
from optuna.trial import TrialState
from tqdm.auto import tqdm

from stage2_rtdetr_r18_pipeline import (
    PACKAGE_ROOT,
    Paths,
    StateManager,
    archive_source_snapshot,
    atomic_json_dump,
    build_repo_train_command,
    build_rtdetr_config_dict,
    build_runtime_manifests,
    capture_environment,
    create_or_replace_link,
    dump_yaml,
    ensure_dir,
    estimate_remaining_seconds,
    format_duration_compact,
    find_checkpoint_candidates,
    load_yaml,
    parse_coco_metrics_from_text,
    read_training_progress,
    repo_train_script,
    resolve_distributed_process_count,
    resolve_tuned_params_path,
    run_command_streaming,
    sanitized_runtime_env,
    setup_logger,
    utc_now_iso,
)

ALLOWED_TUNED_KEYS = {"lr", "weight_decay", "warmups", "ema_decay", "photometric_p", "iou_crop_p"}
INTERMEDIATE_METRIC_INDEX = {
    "map50_95": 0,
    "map50": 1,
    "map75": 2,
    "ar100": 8,
}

warnings.filterwarnings("ignore", category=ExperimentalWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for RT-DETR-R18 on CottonWeedDet12 with auto-chain-ready artifact output.")
    p.add_argument("--config", required=True, help="Path to detection_rtdetr_r18_train_lock.yaml")
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


def get_running_trials(study: optuna.Study) -> list[Any]:
    return sorted([t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING], key=lambda t: t.number)


def summarize_trial_counts(study: optuna.Study) -> Dict[str, int]:
    trials = study.get_trials(deepcopy=False)
    return {
        "total": len(trials),
        "complete": len([t for t in trials if t.state == TrialState.COMPLETE]),
        "fail": len([t for t in trials if t.state == TrialState.FAIL]),
        "pruned": len([t for t in trials if t.state == TrialState.PRUNED]),
        "running": len([t for t in trials if t.state == TrialState.RUNNING]),
    }


def compute_optuna_overall_progress(
    *,
    target_complete: int,
    complete_count: int,
    current_trial_fraction: float = 0.0,
) -> Dict[str, Any]:
    safe_target = max(1, int(target_complete))
    equivalent_complete = min(float(safe_target), float(complete_count) + max(0.0, current_trial_fraction))
    progress_fraction = min(1.0, max(0.0, equivalent_complete / float(safe_target)))
    return {
        "equivalent_complete_trials": round(equivalent_complete, 4),
        "progress_fraction": progress_fraction,
        "progress_percent": round(progress_fraction * 100.0, 1),
        "remaining_complete_trials": round(max(0.0, float(safe_target) - equivalent_complete), 4),
    }


def save_study_artifacts(study: optuna.Study, optuna_root: Path, log) -> None:
    ensure_dir(optuna_root)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs", "datetime_start", "datetime_complete"))
    df.to_csv(optuna_root / "trials.csv", index=False)

    best_params = study.best_params if len([t for t in study.trials if t.state == TrialState.COMPLETE]) else {}
    best_params_path = optuna_root / "best_hyperparameters.yaml"
    if best_params:
        dump_yaml(best_params, best_params_path)
    else:
        with contextlib.suppress(FileNotFoundError):
            best_params_path.unlink()

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

    if best_params:
        with contextlib.suppress(Exception):
            import matplotlib.pyplot as plt  # type: ignore
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

            fig = plot_optimization_history(study)
            fig.figure.savefig(optuna_root / "optimization_history.png", bbox_inches="tight", dpi=160)
            plt.close(fig.figure)

            fig2 = plot_param_importances(study)
            fig2.figure.savefig(optuna_root / "param_importances.png", bbox_inches="tight", dpi=160)
            plt.close(fig2.figure)


def build_pruner(cfg: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruning_cfg = dict(cfg.get("tuning", {}).get("pruning", {}))
    if not bool(pruning_cfg.get("enabled", False)):
        return optuna.pruners.NopPruner()

    pruner_name = str(pruning_cfg.get("name", "median")).strip().lower()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruning_cfg.get("n_startup_trials", 2)),
            n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 10)),
            interval_steps=int(pruning_cfg.get("interval_steps", 1)),
            n_min_trials=int(pruning_cfg.get("n_min_trials", 1)),
        )
    if pruner_name in {"successive_halving", "asha"}:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruning_cfg.get("min_resource", 10)),
            reduction_factor=int(pruning_cfg.get("reduction_factor", 3)),
            min_early_stopping_rate=int(pruning_cfg.get("min_early_stopping_rate", 0)),
            bootstrap_count=int(pruning_cfg.get("bootstrap_count", 0)),
        )
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruning_cfg.get("min_resource", 10)),
            max_resource=int(pruning_cfg.get("max_resource", cfg["tuning"]["max_epochs_per_trial"])),
            reduction_factor=int(pruning_cfg.get("reduction_factor", 3)),
            bootstrap_count=int(pruning_cfg.get("bootstrap_count", 0)),
        )
    raise ValueError(f"Unsupported pruning strategy: {pruner_name}")


def describe_pruner(cfg: Dict[str, Any], pruner: optuna.pruners.BasePruner) -> str:
    pruning_cfg = dict(cfg.get("tuning", {}).get("pruning", {}))
    if isinstance(pruner, optuna.pruners.NopPruner):
        return "disabled"
    strategy = str(pruning_cfg.get("name", pruner.__class__.__name__)).lower()
    extras = []
    for key in (
        "n_startup_trials",
        "n_warmup_steps",
        "interval_steps",
        "n_min_trials",
        "min_resource",
        "max_resource",
        "reduction_factor",
        "bootstrap_count",
    ):
        if key in pruning_cfg:
            extras.append(f"{key}={pruning_cfg[key]}")
    return strategy if not extras else f"{strategy} ({', '.join(extras)})"


def get_trial_handle(study: optuna.Study, trial_number: int) -> OptunaTrial:
    trial_id = study._storage.get_trial_id_from_study_id_trial_number(study._study_id, int(trial_number))
    return OptunaTrial(study, trial_id)


def get_frozen_trial(study: optuna.Study, trial_number: int) -> Any:
    for trial in study.get_trials(deepcopy=False):
        if int(trial.number) == int(trial_number):
            return trial
    raise KeyError(f"Trial {trial_number} not found in study {study.study_name}")


def read_intermediate_metric_history(log_path: Path, metric_name: str) -> Dict[int, float]:
    metric_index = INTERMEDIATE_METRIC_INDEX.get(metric_name)
    if metric_index is None:
        raise ValueError(
            f"Unsupported metric for pruning: {metric_name}. Supported values: {sorted(INTERMEDIATE_METRIC_INDEX)}"
        )
    if not log_path.exists():
        return {}

    history: Dict[int, float] = {}
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                epoch = row.get("epoch")
                metrics = row.get("test_coco_eval_bbox")
                if not isinstance(epoch, int) or not isinstance(metrics, list) or metric_index >= len(metrics):
                    continue
                try:
                    history[int(epoch) + 1] = float(metrics[metric_index])
                except (TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return history


def report_intermediate_values(
    *,
    trial_handle: OptunaTrial,
    study: optuna.Study,
    trial_number: int,
    metric_name: str,
    metric_log_path: Path,
    reported_steps: set[int],
    log,
) -> tuple[int | None, float | None]:
    history = read_intermediate_metric_history(metric_log_path, metric_name)
    for step in sorted(history):
        if step in reported_steps:
            continue
        value = float(history[step])
        trial_handle.report(value, step)
        trial_handle.set_user_attr("last_reported_epoch", int(step))
        trial_handle.set_user_attr("last_reported_value", value)
        reported_steps.add(step)
        log.info(
            "Optuna trial %s intermediate | metric=%s | epoch=%s | value=%.4f",
            trial_number,
            metric_name,
            step,
            value,
        )
        if trial_handle.should_prune():
            trial_handle.set_user_attr("pruned_epoch", int(step))
            trial_handle.set_user_attr("pruned_value", value)
            return step, value
    return None, None


def terminate_process(process: subprocess.Popen[str] | None, timeout_seconds: float = 30.0) -> None:
    if process is None or process.poll() is not None:
        return
    with contextlib.suppress(Exception):
        process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(Exception):
        process.kill()
    with contextlib.suppress(Exception):
        process.wait(timeout=5)


def run_command_streaming_with_pruning(
    cmd: list[str],
    *,
    study: optuna.Study,
    trial_handle: OptunaTrial,
    trial_number: int,
    metric_name: str,
    metric_log_path: Path,
    log,
    cwd: Path | None,
    tee_path: Path,
    append_output: bool = False,
    check: bool = True,
    poll_interval_seconds: int = 15,
) -> str:
    frozen_trial = get_frozen_trial(study, trial_number)
    reported_steps = {int(step) for step in frozen_trial.intermediate_values}
    prune_step, prune_value = report_intermediate_values(
        trial_handle=trial_handle,
        study=study,
        trial_number=trial_number,
        metric_name=metric_name,
        metric_log_path=metric_log_path,
        reported_steps=reported_steps,
        log=log,
    )
    if prune_step is not None and prune_value is not None:
        raise optuna.TrialPruned(
            f"Pruned before launch at epoch {prune_step} with {metric_name}={prune_value:.4f}"
        )

    ensure_dir(tee_path.parent)
    effective_env = sanitized_runtime_env()
    log.info("Streaming command: %s", " ".join(cmd))
    lines: list[str] = []
    tee_mode = "a" if append_output and tee_path.exists() else "w"
    process: subprocess.Popen[str] | None = None
    drain_thread: threading.Thread | None = None

    with tee_path.open(tee_mode, encoding="utf-8") as tee:
        def _drain_stdout() -> None:
            assert process is not None and process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tee.write(line)
                tee.flush()
                lines.append(line)

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=effective_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            drain_thread = threading.Thread(
                target=_drain_stdout,
                name=f"optuna-trial-{trial_number}-stdout",
                daemon=True,
            )
            drain_thread.start()

            while True:
                prune_step, prune_value = report_intermediate_values(
                    trial_handle=trial_handle,
                    study=study,
                    trial_number=trial_number,
                    metric_name=metric_name,
                    metric_log_path=metric_log_path,
                    reported_steps=reported_steps,
                    log=log,
                )
                if prune_step is not None and prune_value is not None:
                    terminate_process(process)
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {prune_step} with {metric_name}={prune_value:.4f}"
                    )

                if process.poll() is not None:
                    break
                time.sleep(max(1, int(poll_interval_seconds)))

            report_intermediate_values(
                trial_handle=trial_handle,
                study=study,
                trial_number=trial_number,
                metric_name=metric_name,
                metric_log_path=metric_log_path,
                reported_steps=reported_steps,
                log=log,
            )
        except KeyboardInterrupt:
            terminate_process(process)
            raise
        finally:
            if drain_thread is not None:
                drain_thread.join(timeout=5.0)
            if process is not None and process.stdout is not None:
                process.stdout.close()

        if check and process is not None and process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")
    return "".join(lines)


def write_running_trial_status(
    path: Path,
    *,
    study: optuna.Study,
    target_complete: int,
    metric_name: str,
    trial_number: int,
    params: Dict[str, Any],
    state: str,
    started_at_utc: str,
    started_monotonic: float,
    train_log: Path | None = None,
    validation_log: Path | None = None,
    objective_value: float | None = None,
    error: str | None = None,
    training_progress: Dict[str, Any] | None = None,
    overall_progress: Dict[str, Any] | None = None,
    estimated_remaining_seconds_current_trial: float | None = None,
    lock: threading.Lock | None = None,
) -> None:
    payload = {
        "updated_at_utc": utc_now_iso(),
        "study_name": study.study_name,
        "state": state,
        "target_complete_trials": int(target_complete),
        "objective_metric": metric_name,
        "trial_number": int(trial_number),
        "started_at_utc": started_at_utc,
        "elapsed_seconds": round(max(0.0, time.time() - started_monotonic), 1),
        "trial_counts": summarize_trial_counts(study),
        "params": params,
        "train_log": str(train_log) if train_log else None,
        "validation_log": str(validation_log) if validation_log else None,
        "objective_value": objective_value,
        "error": error,
        "training_progress": training_progress or {},
        "overall_progress": overall_progress or {},
        "estimated_remaining_seconds_current_trial": estimated_remaining_seconds_current_trial,
    }
    if lock is None:
        atomic_json_dump(payload, path)
        return
    with lock:
        atomic_json_dump(payload, path)


def start_trial_heartbeat(
    *,
    status_path: Path,
    status_lock: threading.Lock,
    study: optuna.Study,
    target_complete: int,
    metric_name: str,
    trial_number: int,
    params: Dict[str, Any],
    started_at_utc: str,
    started_monotonic: float,
    train_log: Path,
    target_epochs_per_trial: int,
    interval_seconds: int = 15,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.wait(interval_seconds):
            counts = summarize_trial_counts(study)
            training_progress = read_training_progress(train_log, target_epochs_per_trial)
            overall_progress = compute_optuna_overall_progress(
                target_complete=target_complete,
                complete_count=counts["complete"],
                current_trial_fraction=float(training_progress["progress_fraction"]),
            )
            elapsed_seconds = max(0.0, time.time() - started_monotonic)
            current_eta_seconds = estimate_remaining_seconds(elapsed_seconds, float(training_progress["progress_fraction"]))
            batch_suffix = ""
            if training_progress["current_batch_done"] is not None and training_progress["current_batch_total"] is not None:
                batch_suffix = f" | batch={training_progress['current_batch_done']}/{training_progress['current_batch_total']}"
            print(
                (
                    f"[PROGRESS] Optuna trial {trial_number} "
                    f"| trial={training_progress['progress_percent']:.1f}% "
                    f"| overall={overall_progress['progress_percent']:.1f}% "
                    f"| epoch={training_progress['current_epoch']}/{training_progress['target_epochs']}"
                    f"{batch_suffix} "
                    f"| elapsed={format_duration_compact(elapsed_seconds)} "
                    f"| eta={format_duration_compact(current_eta_seconds)} "
                    f"| remaining_trials={overall_progress['remaining_complete_trials']:.2f} "
                    f"| pruned={counts['pruned']} | fail={counts['fail']} | train_log={train_log}"
                ),
                file=sys.stderr,
                flush=True,
            )
            write_running_trial_status(
                status_path,
                study=study,
                target_complete=target_complete,
                metric_name=metric_name,
                trial_number=trial_number,
                params=params,
                state="RUNNING",
                started_at_utc=started_at_utc,
                started_monotonic=started_monotonic,
                train_log=train_log,
                training_progress=training_progress,
                overall_progress=overall_progress,
                estimated_remaining_seconds_current_trial=current_eta_seconds,
                lock=status_lock,
            )

    thread = threading.Thread(target=_run, name=f"optuna-trial-{trial_number}-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


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
    repo_train_script(paths)

    if not bool(cfg.get("tuning", {}).get("enabled", False)):
        log.info("Tuning is disabled in config. Nothing to do.")
        return

    weight_path = PACKAGE_ROOT / cfg["weights"]["file_name"]
    if not weight_path.exists():
        alt = PACKAGE_ROOT / "assets" / cfg["weights"]["file_name"]
        weight_path = alt if alt.exists() else weight_path
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file required for tuning was not found: {weight_path}")

    pruner = build_pruner(cfg)
    storage_url = f"sqlite:///{(optuna_root / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=str(cfg["tuning"]["study_name"]),
        direction=str(cfg["tuning"].get("direction", "maximize")),
        sampler=TPESampler(seed=int(cfg["tuning"].get("sampler_seed", cfg["training"]["seed"]))),
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )
    log.info("Optuna pruner: %s", describe_pruner(cfg, pruner))
    log.info("Optuna training processes per node: %s", resolve_distributed_process_count(cfg))

    stale_trials = get_running_trials(study)
    if stale_trials:
        log.info(
            "Found %d stale RUNNING trial(s). They will be resumed before creating new trials.",
            len(stale_trials),
        )
    save_study_artifacts(study, optuna_root, log)

    target_complete = int(cfg["tuning"]["n_trials"])
    current_status_path = optuna_root / "current_trial_status.json"
    complete_count = summarize_trial_counts(study)["complete"]
    if complete_count >= target_complete:
        log.info("Study already has %d complete trials (target=%d).", complete_count, target_complete)
        save_study_artifacts(study, optuna_root, log)
        return

    metric_name = str(cfg["tuning"].get("metric", "map50_95"))
    search_space = dict(cfg["tuning"]["search_space"])
    trials_root = ensure_dir(optuna_root / "trials")
    pbar = tqdm(total=target_complete, initial=complete_count, desc="Optuna complete trials", unit="trial", dynamic_ncols=True)

    attempt_limit = target_complete * 3
    attempts = 0
    try:
        while complete_count < target_complete and attempts < attempt_limit:
            attempts += 1
            stale_trial = next(iter(get_running_trials(study)), None)
            new_trial = None
            if stale_trial is None:
                new_trial = study.ask()
                save_study_artifacts(study, optuna_root, log)
                trial_number = int(new_trial.number)
                trial_handle = new_trial
            else:
                trial_number = int(stale_trial.number)
                trial_handle = get_trial_handle(study, trial_number)

            trial_dir = trials_root / f"trial_{trial_number:04d}"
            ensure_dir(trial_dir)

            if new_trial is None:
                trial_params = dict(stale_trial.params)
                if not trial_params and (trial_dir / "suggested_params.yaml").exists():
                    trial_params = dict(load_yaml(trial_dir / "suggested_params.yaml"))
                if not trial_params:
                    log.warning("Could not recover params for stale RUNNING trial %s. Marking it as FAIL.", trial_number)
                    study.tell(trial_number, state=TrialState.FAIL)
                    save_study_artifacts(study, optuna_root, log)
                    continue
                log.info("Resuming stale RUNNING trial %s with params=%s", trial_number, trial_params)
            else:
                trial_params = {name: suggest_from_spec(new_trial, name, spec) for name, spec in search_space.items()}
                dump_yaml(trial_params, trial_dir / "suggested_params.yaml")
                atomic_json_dump({"trial_number": trial_number, "params": trial_params, "created_at_utc": utc_now_iso()}, trial_dir / "trial_info.json")

            config_dir = ensure_dir(trial_dir / "configs")
            train_output_dir = trial_dir / "run"
            train_cfg = build_rtdetr_config_dict(
                cfg, paths, train_output_dir,
                train_img_dir=paths.runtime_dataset_dir / "images" / "train",
                train_ann=paths.runtime_dataset_dir / "annotations" / "instances_train.json",
                val_img_dir=paths.runtime_dataset_dir / "images" / "val",
                val_ann=paths.runtime_dataset_dir / "annotations" / "instances_val.json",
                epochs=int(cfg["tuning"]["max_epochs_per_trial"]),
                train_batch=int(cfg["tuning"]["batch"]),
                val_batch=int(cfg["tuning"]["batch"]),
                workers=int(cfg["tuning"]["workers"]),
                tuned=trial_params,
            )
            train_cfg_path = config_dir / "trial_train.yml"
            dump_yaml(train_cfg, train_cfg_path)
            dump_yaml(train_cfg, trial_dir / "effective_train_config.yaml")

            status_lock = threading.Lock()
            counts_before = summarize_trial_counts(study)
            pbar.set_postfix_str(
                f"running={trial_number} pruned={counts_before['pruned']} fail={counts_before['fail']} total={counts_before['total']}"
            )
            pbar.refresh()
            started_at_utc = utc_now_iso()
            started = time.time()
            train_log = trial_dir / "trial_train.log"
            val_log = trial_dir / "trial_validation.log"
            ckpts_before = find_checkpoint_candidates(train_output_dir) if new_trial is None else {"latest": None}
            resume_ckpt = ckpts_before.get("latest")
            train_metric_log = train_output_dir / "log.txt"
            initial_training_progress = read_training_progress(train_log, int(cfg["tuning"]["max_epochs_per_trial"]))
            initial_overall_progress = compute_optuna_overall_progress(
                target_complete=target_complete,
                complete_count=counts_before["complete"],
                current_trial_fraction=float(initial_training_progress["progress_fraction"]),
            )
            print(
                (
                    f"[INFO] Optuna trial {trial_number} {'resumed' if new_trial is None else 'started'} "
                    f"| target_complete={complete_count + 1}/{target_complete} "
                    f"| overall={initial_overall_progress['progress_percent']:.1f}% "
                    f"| fail_so_far={counts_before['fail']} "
                    f"| train_log={train_log}"
                    f"{f' | resume_ckpt={resume_ckpt}' if resume_ckpt else ''}"
                ),
                file=sys.stderr,
                flush=True,
            )
            write_running_trial_status(
                current_status_path,
                study=study,
                target_complete=target_complete,
                metric_name=metric_name,
                trial_number=trial_number,
                params=trial_params,
                state="RUNNING",
                started_at_utc=started_at_utc,
                started_monotonic=started,
                train_log=train_log,
                training_progress=initial_training_progress,
                overall_progress=initial_overall_progress,
                lock=status_lock,
            )
            heartbeat_stop, heartbeat_thread = start_trial_heartbeat(
                status_path=current_status_path,
                status_lock=status_lock,
                study=study,
                target_complete=target_complete,
                metric_name=metric_name,
                trial_number=trial_number,
                params=trial_params,
                started_at_utc=started_at_utc,
                started_monotonic=started,
                train_log=train_log,
                target_epochs_per_trial=int(cfg["tuning"]["max_epochs_per_trial"]),
            )
            try:
                if resume_ckpt and Path(resume_ckpt).exists():
                    train_cmd = build_repo_train_command(
                        cfg,
                        paths,
                        train_cfg_path,
                        seed=int(cfg["training"]["seed"]),
                        resume=Path(resume_ckpt),
                        use_amp=bool(cfg["training"].get("amp", True)),
                    )
                else:
                    train_cmd = build_repo_train_command(
                        cfg,
                        paths,
                        train_cfg_path,
                        seed=int(cfg["training"]["seed"]),
                        tuning=weight_path,
                        use_amp=bool(cfg["training"].get("amp", True)),
                    )
                run_command_streaming_with_pruning(
                    train_cmd,
                    study=study,
                    trial_handle=trial_handle,
                    trial_number=trial_number,
                    metric_name=metric_name,
                    metric_log_path=train_metric_log,
                    log=log,
                    cwd=paths.repo_work_dir,
                    tee_path=train_log,
                    append_output=resume_ckpt is not None,
                )

                ckpts = find_checkpoint_candidates(train_output_dir)
                best_ckpt = ckpts["best"]
                if best_ckpt is None:
                    raise RuntimeError("Trial finished but no checkpoint file was found.")

                val_cfg = build_rtdetr_config_dict(
                    cfg, paths, trial_dir / "validation",
                    train_img_dir=paths.runtime_dataset_dir / "images" / "train",
                    train_ann=paths.runtime_dataset_dir / "annotations" / "instances_train.json",
                    val_img_dir=paths.runtime_dataset_dir / "images" / "val",
                    val_ann=paths.runtime_dataset_dir / "annotations" / "instances_val.json",
                    epochs=int(cfg["tuning"]["max_epochs_per_trial"]),
                    train_batch=int(cfg["tuning"]["batch"]),
                    val_batch=int(cfg["tuning"]["batch"]),
                    workers=int(cfg["tuning"]["workers"]),
                    tuned=trial_params,
                )
                val_cfg_path = config_dir / "trial_val.yml"
                dump_yaml(val_cfg, val_cfg_path)
                val_cmd = build_repo_train_command(
                    cfg,
                    paths,
                    val_cfg_path,
                    seed=int(cfg["training"]["seed"]),
                    resume=best_ckpt,
                    test_only=True,
                    use_amp=bool(cfg["training"].get("amp", True)),
                )
                val_text = run_command_streaming(val_cmd, log, cwd=paths.repo_work_dir, tee_path=val_log)
                metrics = parse_coco_metrics_from_text(val_text)
                if metric_name not in metrics:
                    raise RuntimeError(f"Could not extract {metric_name} from validation log for trial {trial_number}")
                objective_value = float(metrics[metric_name])
                duration = time.time() - started
                study.tell(trial_number, objective_value)
                completed_training_progress = {
                    "target_epochs": int(cfg["tuning"]["max_epochs_per_trial"]),
                    "completed_epochs": int(cfg["tuning"]["max_epochs_per_trial"]),
                    "fractional_epochs": float(cfg["tuning"]["max_epochs_per_trial"]),
                    "progress_fraction": 1.0,
                    "progress_percent": 100.0,
                    "current_epoch": int(cfg["tuning"]["max_epochs_per_trial"]),
                    "current_epoch_zero_based": int(cfg["tuning"]["max_epochs_per_trial"]) - 1,
                    "current_batch_done": None,
                    "current_batch_total": None,
                    "remaining_epochs": 0.0,
                }
                completed_overall_progress = compute_optuna_overall_progress(
                    target_complete=target_complete,
                    complete_count=summarize_trial_counts(study)["complete"],
                    current_trial_fraction=0.0,
                )
                trial_summary = {
                    "trial_number": trial_number,
                    "status": "COMPLETE",
                    "objective_metric": metric_name,
                    "objective_value": objective_value,
                    "duration_seconds": duration,
                    "params": trial_params,
                    "metrics": metrics,
                    "best_checkpoint": str(best_ckpt),
                    "train_log": str(train_log),
                    "validation_log": str(val_log),
                }
                atomic_json_dump(trial_summary, trial_dir / "trial_summary.json")
                write_running_trial_status(
                    current_status_path,
                    study=study,
                    target_complete=target_complete,
                    metric_name=metric_name,
                    trial_number=trial_number,
                    params=trial_params,
                    state="COMPLETE",
                    started_at_utc=started_at_utc,
                    started_monotonic=started,
                    train_log=train_log,
                    validation_log=val_log,
                    objective_value=objective_value,
                    training_progress=completed_training_progress,
                    overall_progress=completed_overall_progress,
                    lock=status_lock,
                )
                print(
                    f"[INFO] Optuna trial {trial_number} completed | {metric_name}={objective_value:.4f} | overall={completed_overall_progress['progress_percent']:.1f}%",
                    file=sys.stderr,
                    flush=True,
                )
            except optuna.TrialPruned as exc:
                study.tell(trial_number, state=TrialState.PRUNED)
                pruned_training_progress = read_training_progress(train_log, int(cfg["tuning"]["max_epochs_per_trial"]))
                pruned_overall_progress = compute_optuna_overall_progress(
                    target_complete=target_complete,
                    complete_count=summarize_trial_counts(study)["complete"],
                    current_trial_fraction=0.0,
                )
                frozen_after_prune = get_frozen_trial(study, trial_number)
                last_step = max((int(step) for step in frozen_after_prune.intermediate_values), default=None)
                last_value = (
                    float(frozen_after_prune.intermediate_values[last_step])
                    if last_step is not None
                    else None
                )
                atomic_json_dump(
                    {
                        "trial_number": trial_number,
                        "status": "PRUNED",
                        "reason": str(exc),
                        "params": trial_params,
                        "objective_metric": metric_name,
                        "last_intermediate_epoch": last_step,
                        "last_intermediate_value": last_value,
                    },
                    trial_dir / "trial_summary.json",
                )
                write_running_trial_status(
                    current_status_path,
                    study=study,
                    target_complete=target_complete,
                    metric_name=metric_name,
                    trial_number=trial_number,
                    params=trial_params,
                    state="PRUNED",
                    started_at_utc=started_at_utc,
                    started_monotonic=started,
                    train_log=train_log,
                    objective_value=last_value,
                    error=str(exc),
                    training_progress=pruned_training_progress,
                    overall_progress=pruned_overall_progress,
                    lock=status_lock,
                )
                print(
                    f"[INFO] Optuna trial {trial_number} pruned | reason={exc}",
                    file=sys.stderr,
                    flush=True,
                )
            except KeyboardInterrupt:
                interrupted_training_progress = read_training_progress(train_log, int(cfg["tuning"]["max_epochs_per_trial"]))
                interrupted_overall_progress = compute_optuna_overall_progress(
                    target_complete=target_complete,
                    complete_count=summarize_trial_counts(study)["complete"],
                    current_trial_fraction=float(interrupted_training_progress["progress_fraction"]),
                )
                current_eta_seconds = estimate_remaining_seconds(
                    max(0.0, time.time() - started),
                    float(interrupted_training_progress["progress_fraction"]),
                )
                write_running_trial_status(
                    current_status_path,
                    study=study,
                    target_complete=target_complete,
                    metric_name=metric_name,
                    trial_number=trial_number,
                    params=trial_params,
                    state="RUNNING",
                    started_at_utc=started_at_utc,
                    started_monotonic=started,
                    train_log=train_log,
                    training_progress=interrupted_training_progress,
                    overall_progress=interrupted_overall_progress,
                    lock=status_lock,
                )
                log.info(
                    "Optuna interrupted by user while trial %s was active. trial_progress=%.1f%% overall=%.1f%% epoch=%s/%s eta=%s. Re-run the same command to resume from the latest checkpoint.",
                    trial_number,
                    interrupted_training_progress["progress_percent"],
                    interrupted_overall_progress["progress_percent"],
                    interrupted_training_progress["current_epoch"],
                    interrupted_training_progress["target_epochs"],
                    format_duration_compact(current_eta_seconds),
                )
                raise
            except Exception as exc:
                log.exception("Trial %s failed: %s", trial_number, exc)
                study.tell(trial_number, state=TrialState.FAIL)
                failed_training_progress = read_training_progress(train_log, int(cfg["tuning"]["max_epochs_per_trial"]))
                failed_overall_progress = compute_optuna_overall_progress(
                    target_complete=target_complete,
                    complete_count=summarize_trial_counts(study)["complete"],
                    current_trial_fraction=float(failed_training_progress["progress_fraction"]),
                )
                atomic_json_dump({"trial_number": trial_number, "status": "FAIL", "error": str(exc), "params": trial_params}, trial_dir / "trial_summary.json")
                write_running_trial_status(
                    current_status_path,
                    study=study,
                    target_complete=target_complete,
                    metric_name=metric_name,
                    trial_number=trial_number,
                    params=trial_params,
                    state="FAIL",
                    started_at_utc=started_at_utc,
                    started_monotonic=started,
                    train_log=train_log,
                    validation_log=val_log if val_log.exists() else None,
                    error=str(exc),
                    training_progress=failed_training_progress,
                    overall_progress=failed_overall_progress,
                    lock=status_lock,
                )
                print(
                    f"[WARN] Optuna trial {trial_number} failed | error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)

            save_study_artifacts(study, optuna_root, log)
            counts_after = summarize_trial_counts(study)
            complete_count = counts_after["complete"]
            pbar.n = complete_count
            pbar.set_postfix_str(
                f"last={trial_number} pruned={counts_after['pruned']} fail={counts_after['fail']} total={counts_after['total']}"
            )
            pbar.refresh()
    finally:
        pbar.close()
    save_study_artifacts(study, optuna_root, log)
    final_counts = summarize_trial_counts(study)
    final_overall_progress = compute_optuna_overall_progress(
        target_complete=target_complete,
        complete_count=final_counts["complete"],
        current_trial_fraction=0.0,
    )
    atomic_json_dump(
        {
            "updated_at_utc": utc_now_iso(),
            "study_name": study.study_name,
            "state": "IDLE" if complete_count < target_complete else "TUNING_COMPLETE",
            "target_complete_trials": target_complete,
            "trial_counts": final_counts,
            "overall_progress": final_overall_progress,
        },
        current_status_path,
    )
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
    ckpts = find_checkpoint_candidates(best_trial_dir / "run")
    if ckpts["best"]:
        create_or_replace_link(ckpts["best"], optuna_root / "best_trial_best.pth")

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
    try:
        main()
    except KeyboardInterrupt:
        print(
            "[INFO] Interrupted by user. Re-run the same command to resume the stale RUNNING trial.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
