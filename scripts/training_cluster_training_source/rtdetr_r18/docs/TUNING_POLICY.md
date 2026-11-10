# Tuning Policy

- Optuna tuning uses only the locked `train` and `val` splits.
- The frozen `test` split is not used for hyperparameter search.
- In this RT-DETR-R18 package, Optuna targets `12` complete trials with up to `60` epochs per trial.
- Median pruning is enabled after `2` startup trials and `10` warmup epochs, with pruning checks every `2` epochs.
- When more than one GPU is allocated, the package launches RT-DETR with PyTorch distributed (`torch.distributed.run`).
- The configured train/val batch sizes and worker counts are treated as node-level totals and are split evenly across the local processes.
- After Optuna completes, the best hyperparameters are copied into the runtime area and reused for full training automatically in the same Slurm job.
