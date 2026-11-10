# Stage-2 package for Training_Cluster — YOLO11s

YOLO11s baseline on `CottonWeedDet12_v1` with:
- locked train/val/test split usage
- paper-grade artifact capture
- optional smoke training
- Optuna tuning on train+val only
- **automatic full training in the same Slurm job after Optuna completes successfully**
- ONNX export for later Jetson deployment

## Main difference from the YOLO11n package

This package is matched to the locked model matrix where **YOLO11s** is the stronger same-family detector.
The workflow is the same as YOLO11n, but `submit_optuna.sh` is now designed to run Optuna and full training sequentially inside one job ID.

## Default resources in this package

### Prepare
- partition: `batch`
- CPUs: `4`
- RAM: `16G`
- time: `02:00:00`
- GPU: none

### Smoke
- GPU: `1 x A100`
- CPUs: `32`
- RAM: `48G`
- time: `12:00:00`

### Optuna
- GPU: `1 x A100`
- CPUs: `32`
- RAM: `64G`
- time: `72:00:00`

### Full training
- GPU: `1 x A100`
- CPUs: `32`
- RAM: `64G`
- time: `72:00:00`

Note: in Slurm, `64G` is RAM, not GPU count. This package therefore uses `--gres gpu:A100:1` plus `--mem 64G`.

## Weight file behavior

The package does **not** bundle `yolo11s.pt` itself.
Instead, the training and Optuna Slurm jobs call `scripts/ensure_model_weights.py` first.
If `yolo11s.pt` is not already inside the package directory, the helper tries the official Ultralytics asset download path automatically.

## One-command Optuna -> Full chain

This is the main command for YOLO11s:

```bash
bash scripts/submit_optuna.sh
```

What it does:
1. submits the Optuna job
2. captures the single job ID
3. runs Optuna first inside that job allocation
4. starts full training automatically in the same job only if Optuna finishes successfully

If you want tune-only:

```bash
bash scripts/submit_optuna.sh --no-auto-full
```

## Recommended execution order

### 1) unpack the package

```bash
mkdir -p ~/jetson_edge_ai_benchmark
cd ~/jetson_edge_ai_benchmark
unzip stage2_yolo11s_training_cluster_package.zip
```

### 2) create / update the yolo11 environment

```bash
interact -c 4 --mem=16G --time=02:00:00
cd ~/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster
bash scripts/setup_stage2_env.sh       ~/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster       /home/$USER/envs/yolo11
exit
```

### 3) submit prepare

```bash
cd ~/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster
bash scripts/submit_prepare.sh
```

### 4) optional smoke test

```bash
bash scripts/submit_yolo11s.sh --mode smoke
```

### 5) main command: Optuna + full training in one job

```bash
bash scripts/submit_optuna.sh
```

Example with explicit resources (same as package defaults):

```bash
bash scripts/submit_optuna.sh       --gres gpu:A100:1       --cpus 32       --mem 64G       --time 72:00:00
```

If you split the command across multiple shell lines, end each continued line with `\`.
Otherwise bash will treat a later flag such as `--mem` as a new command and print `-bash: --mem: command not found`.

The same allocation is reused for both stages, so `--gres`, `--cpus`, `--mem`, and `--time` now apply to both Optuna and full training together.
The old `--full-*` flags are accepted only for backward compatibility and are ignored in single-job mode.

### 6) manual full-only (if ever needed)

```bash
bash scripts/submit_yolo11s.sh --mode full
```

## Runtime root

All YOLO11s outputs are isolated here:

```text
${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/experiments/detection/CottonWeedDet12_v1/yolo11s_baseline_v1
```

That means you can run YOLO11n and YOLO11s training in parallel without path collisions.

## Important benchmark policy

- tuning uses **train + val only**
- the **20% test split stays frozen**
- later Jetson final detection benchmarking should use the frozen **test split**, not validation

## Monitoring

Queue:

```bash
watch -n 5 "sq --me"
```

Pipeline state:

```bash
VENV_PATH=/home/$USER/envs/yolo11     bash scripts/run_pipeline.sh ~/jetson_edge_ai_benchmark/stage2_yolo11s_training_cluster status
```

Latest logs:

```bash
tail -f $(ls -t y11s_*.out | head -1)
tail -f $(ls -t y11s_*.err | head -1)
```
