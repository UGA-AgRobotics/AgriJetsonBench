# Stage-2 package for Training_Cluster — DeepLabV3++

DeepLabV3++ baseline on `CropAndWeed_v1` with:
- locked train/val/test split usage
- paper-grade artifact capture
- optional smoke training
- Optuna tuning on train+val only
- automatic full training in the same Slurm job after Optuna completes successfully
- ONNX export for later Jetson deployment
- Slurm-aware distributed training with a default single-node 4 x L4 target

## Main behavior

This package is set up to run `Optuna -> full training` sequentially inside one Slurm allocation and one job ID.

## Default resources in this package

### Prepare
- partition: `batch`
- CPUs: `4`
- RAM: `16G`
- time: `02:00:00`
- GPU: none

### Smoke
- GPU: `4 x L4` via `1 node x 4 GPUs/node`
- CPUs: `32`
- RAM: `96G`
- time: `12:00:00`

### Optuna
- GPU: `4 x L4` via `1 node x 4 GPUs/node`
- CPUs: `32`
- RAM: `96G`
- time: `168:00:00`

### Full training
- GPU: `4 x L4` via `1 node x 4 GPUs/node`
- CPUs: `32`
- RAM: `96G`
- time: `168:00:00`

## Policy intentionally matched to the attached reference package

- full training epochs: `300`
- Optuna trials: `12`
- max epochs per trial: `60`
- same single-job chaining policy: Optuna first, then full training with best hyperparameters
- same split policy: tune on train/val only, keep test frozen for final reporting

## Model-specific adjustment

DeepLabV3++ is noticeably heavier than BiSeNetV2, so this package keeps the same pipeline/trial/epoch policy but reduces the default total training batch to `16` on `4 x L4` for safer memory headroom.

## What to transfer to Training_Cluster

If the Training_Cluster dataset already exists at `${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1`, transfer these:
- `config/`
- `docs/`
- `scripts/`
- `vendor/`
- `requirements.txt`
- `README.md`
- `README_STAGE2_TRAINING_CLUSTER_DEEPLABV3PLUS.md`
- `MODEL_ASSET_NOTE.txt`
- `multigpu.txt`
- `assets/`

Do not transfer these for the real Training_Cluster package:
- `runtime_local/`
- `__pycache__/`
- `*.zip`

## Recommended replace command

From the local repo root, use `rsync` so the Training_Cluster folder becomes the package root:

```bash
rsync -av --delete \
  --exclude runtime_local \
  --exclude '*.zip' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  ./ ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4/
```

## Recommended execution order on Training_Cluster

### 1) replace the package folder

```bash
mkdir -p ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4
```

### 2) verify the shared segmentation environment

```bash
interact -c 4 --mem=16G --time=02:00:00
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4
bash scripts/setup_stage2_env.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4 /home/$USER/envs/segment
exit
```

This package reuses the shared segmentation env and does not create a new one.

### 3) submit prepare

```bash
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4
bash scripts/submit_prepare.sh
```

### 4) optional smoke test

```bash
bash scripts/submit_deeplabv3plus.sh --mode smoke --partition gpu_p --gres gpu:L4:4 --cpus 32 --mem 96G
```

### 5) main command: Optuna + full training in one job

```bash
bash scripts/submit_optuna.sh --partition gpu_p --gres gpu:L4:4 --cpus 32 --mem 96G --time 168:00:00
```

This submits one single-node job ID. Inside the same allocation:
1. Optuna runs first.
2. Best hyperparameters are written into the runtime area.
3. Full training starts automatically in the same job.

The package defaults to a simpler single-node distributed launch on `1 node x 4 L4`. If you later target a different GPU family or node count, keep the same command shape and override `--partition`, `--nodes`, and `--gres`.

## Monitoring

Queue:

```bash
watch -n 5 "sq --me"
```

Pipeline state:

```bash
VENV_PATH=/home/$USER/envs/segment \
  bash scripts/run_pipeline.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_deeplabv3plus_training_cluster_4L4 status
```

Latest logs:

```bash
tail -f $(ls -t deeplabv3plus_*.out | head -1)
tail -f $(ls -t deeplabv3plus_*.err | head -1)
```
