# Package validation note

This package was assembled by adapting the attached stage-2 BiSeNetV2 pipeline to DeepLabV3++ while preserving the same overall workflow shape:

- smoke training
- Optuna tuning
- automatic chained full training in the same Slurm job
- validation on val/test
- qualitative prediction panels
- ONNX export stage

Validation performed during package construction:

- Python syntax check passed for the main Python files via `python -m py_compile`.
- A local CPU-safe fake-dataset run of the full pipeline succeeded through:
  - smoke training
  - full training
  - val/test evaluation
  - qualitative prediction generation
- The local debug wrappers were hardened to default to conservative CPU thread counts.

Container limitation during package construction:

- `optuna`, `onnx`, and `onnxsim` were not installed in the current build container, so the Optuna runtime path and ONNX export runtime path were not executed here.
- The package still includes the correct scripts/config wiring for those stages, and `scripts/setup_stage2_env.sh` now verifies `optuna`, `sqlalchemy`, `alembic`, `onnx`, and `onnxsim` in the target environment.
