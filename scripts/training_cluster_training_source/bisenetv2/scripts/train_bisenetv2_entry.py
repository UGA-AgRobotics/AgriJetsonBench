#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import faulthandler
import os
import socket
import traceback
from pathlib import Path
from typing import TextIO

from bisenetv2_lib import TrainRunConfig, run_training_job


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training entry point for the stage-2 BiSeNetV2 package")
    p.add_argument("--config", required=True)
    p.add_argument("--run-kind", required=True, choices=["smoke", "trial", "full"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--train-batch-total", required=True, type=int)
    p.add_argument("--eval-batch-total", required=True, type=int)
    p.add_argument("--workers-total", required=True, type=int)
    p.add_argument("--tuned-params-yaml", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument("--history-jsonl", default=None)
    p.add_argument("--class-histogram-json", default=None)
    return p.parse_args()


def _env_int(primary: str, fallback: str, default: int) -> int:
    raw = os.environ.get(primary, os.environ.get(fallback, str(default)))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _open_rank_debug_log(run_dir: Path) -> TextIO:
    rank = _env_int("RANK", "SLURM_PROCID", 0)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", 0)
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", 1)
    log_dir = run_dir / "rank_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"python_rank_{rank:02d}_local_{local_rank:02d}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(
        "\n".join(
            [
                "=== BiSeNetV2 Rank Debug ===",
                f"host={socket.gethostname()}",
                f"rank={rank}",
                f"local_rank={local_rank}",
                f"world_size={world_size}",
                f"master_addr={os.environ.get('MASTER_ADDR', '')}",
                f"master_port={os.environ.get('MASTER_PORT', '')}",
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                f"run_dir={run_dir}",
                f"argv={' '.join(os.sys.argv)}",
                "",
            ]
        )
    )
    handle.flush()
    return handle


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    debug_handle = _open_rank_debug_log(run_dir)
    with contextlib.suppress(Exception):
        faulthandler.enable(debug_handle, all_threads=True)
    try:
        cfg = TrainRunConfig(
            config_path=Path(args.config).expanduser().resolve(),
            run_kind=args.run_kind,
            run_dir=run_dir,
            train_manifest=Path(args.train_manifest).expanduser().resolve(),
            val_manifest=Path(args.val_manifest).expanduser().resolve(),
            epochs=int(args.epochs),
            train_batch_total=int(args.train_batch_total),
            eval_batch_total=int(args.eval_batch_total),
            workers_total=int(args.workers_total),
            tuned_params_yaml=Path(args.tuned_params_yaml).expanduser().resolve() if args.tuned_params_yaml else None,
            resume=Path(args.resume).expanduser().resolve() if args.resume else None,
            report_json=Path(args.report_json).expanduser().resolve() if args.report_json else None,
            history_jsonl=Path(args.history_jsonl).expanduser().resolve() if args.history_jsonl else None,
            class_histogram_json=Path(args.class_histogram_json).expanduser().resolve() if args.class_histogram_json else None,
        )
        run_training_job(cfg)
    except BaseException:
        debug_handle.write("\n=== Uncaught Exception ===\n")
        traceback.print_exc(file=debug_handle)
        debug_handle.flush()
        raise
    finally:
        with contextlib.suppress(Exception):
            faulthandler.disable()
        with contextlib.suppress(Exception):
            debug_handle.flush()
        with contextlib.suppress(Exception):
            debug_handle.close()


if __name__ == "__main__":
    main()
