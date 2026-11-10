# Stage-2 package for Training_Cluster — SegFormer-B0

SegFormer-B0 baseline on `CropAndWeed_v1` with:
- locked train/val/test split usage
- paper-grade artifact capture
- optional smoke training
- Optuna tuning on train+val only
- automatic full training in the same Slurm job after Optuna completes successfully
- ONNX export for later Jetson deployment
- 4-GPU DDP support through `torch.distributed.run`

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
- GPU: `4 x L4`
- CPUs: `32`
- RAM: `128G`
- time: `12:00:00`

### Optuna
- GPU: `4 x L4`
- CPUs: `32`
- RAM: `128G`
- time: `168:00:00`

### Full training
- GPU: `4 x L4`
- CPUs: `32`
- RAM: `128G`
- time: `168:00:00`

## Training policy matched to the attached stage-2 package style

- full training epochs: `300`
- Optuna trials: `12`
- max epochs per trial: `60`
- same single-job chaining policy: Optuna first, then full training with best hyperparameters

## What to transfer to Training_Cluster

If the Training_Cluster dataset already exists at `${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1`, transfer these:
- `config/`
- `docs/`
- `scripts/`
- `vendor/`
- `requirements.txt`
- `README.md`
- `README_STAGE2_TRAINING_CLUSTER_SEGFORMER_B0.md`
- `MODEL_ASSET_NOTE.txt`
- `multigpu.txt`

Do not transfer these for the real Training_Cluster package:
- `runtime_local/`
- `__pycache__/`
- `*.zip`

## Recommended replace command

From the local repo root, use `rsync` so the Training_Cluster folder becomes the package root:

```bash
rsync -av --delete   --exclude runtime_local   --exclude '*.zip'   --exclude '__pycache__'   --exclude '*.pyc'   ./ ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster/
```

## Recommended execution order on Training_Cluster

### 1) replace the package folder

```bash
mkdir -p ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster
```

### 2) create or update the SAM environment

```bash
interact -c 4 --mem=16G --time=02:00:00
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster
bash scripts/setup_stage2_env.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster /home/$USER/envs/SAM
exit
```

### 3) submit prepare

```bash
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster
bash scripts/submit_prepare.sh
```

### 4) optional smoke test

```bash
bash scripts/submit_segformer_b0.sh --mode smoke --partition gpu_p --gres gpu:L4:4 --cpus 32 --mem 128G
```

### 5) main command: Optuna + full training in one job

```bash
bash scripts/submit_optuna.sh --partition gpu_p --gres gpu:L4:4 --cpus 32 --mem 128G --time 168:00:00
```

This submits one job ID. Inside the same allocation:
1. Optuna runs first
2. best hyperparameters are copied into the runtime area
3. full training starts automatically

## Monitoring

Queue:

```bash
watch -n 5 "sq --me"
```

Pipeline state:

```bash
VENV_PATH=/home/$USER/envs/SAM   bash scripts/run_pipeline.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_segformer_b0_training_cluster status
```

Latest logs:

```bash
tail -f $(ls -t segformer_b0_*.out | head -1)
tail -f $(ls -t segformer_b0_*.err | head -1)
```
