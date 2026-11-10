# RT-DETR Local Test Package

This folder is a portable local test package for RT-DETR-R18.

What is included:
- `dataset/CottonWeedDet12_subset100`: a 100-image subset of CottonWeedDet12
- `scripts/`: the patched RT-DETR pipeline and Optuna orchestration scripts
- `vendor/RT-DETR`: the vendored upstream RT-DETR repo used by the scripts
- `rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`: the pretrained weights
- `config/detection_rtdetr_r18_local_subset100.yaml`: a local-safe config

Subset layout:
- `train=80`
- `val=10`
- `test=10`

Recommended local setup:
1. Create a Python 3.10/3.11 environment with a matching `torch` and `torchvision` for your local CUDA setup.
2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Quick smoke run:

```bash
./scripts/run_local_smoke.sh
```

Quick Optuna run:

```bash
./scripts/run_local_optuna.sh
```

Full train with the currently selected tuned params:

```bash
./scripts/run_local_full.sh
```

Run the exact local workflow `Optuna -> full train`:

```bash
./scripts/run_local_tune_then_full.sh
```

Training_Cluster package notes:
- `config/detection_rtdetr_r18_train_lock.yaml` is the full-dataset Training_Cluster config.
- `README_STAGE2_TRAINING_CLUSTER_RTDETR_R18.md` documents the transfer and submit flow for the remote package.

Resume test:
1. Start Optuna.
2. Interrupt it after a few minutes.
3. Re-run the same command.

On a correct resume, the log should mention:
- `Found 1 stale RUNNING trial(s). They will be resumed...`
- `Optuna trial <n> resumed`
- `resume_ckpt=...checkpoint....pth`

Files worth checking:
- `runtime_local/optuna/current_trial_status.json`
- `runtime_local/optuna/trials.csv`
- `runtime_local/optuna/trials/trial_XXXX/trial_train.log`

Notes:
- The wrappers export `PYTHONPATH` so the Pillow compatibility patch in `scripts/python_compat/sitecustomize.py` is active during training subprocesses.
- Prefer `./scripts/...` over `bash scripts/...` so your shell does not inherit the Conda `libtinfo` warning path into a new Bash process.
- `export.enable_onnx_export` and qualitative outputs are disabled in the local config to keep the local debugging loop small.
