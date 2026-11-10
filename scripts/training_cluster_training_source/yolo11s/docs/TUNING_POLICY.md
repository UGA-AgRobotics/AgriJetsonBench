# Tuning policy for YOLO11s

- Tune on `train + val` only.
- Never use the frozen `test` split during tuning.
- This package is configured so `submit_optuna.sh` runs Optuna first and then full training inside the same Slurm job.
- If Optuna does not produce a valid `best_hyperparameters.yaml`, the job exits nonzero and the full-training stage does not start.
- Final Jetson benchmarking later should still use the frozen `20% test split`.
