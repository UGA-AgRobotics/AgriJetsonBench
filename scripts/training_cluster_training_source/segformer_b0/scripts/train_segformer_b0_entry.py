#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from segformer_b0_lib import TrainRunConfig, run_training_job



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training entry point for the stage-2 SegFormer-B0 package")
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



def main() -> None:
    args = parse_args()
    cfg = TrainRunConfig(
        config_path=Path(args.config).expanduser().resolve(),
        run_kind=args.run_kind,
        run_dir=Path(args.run_dir).expanduser().resolve(),
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


if __name__ == "__main__":
    main()
