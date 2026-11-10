# Stage-2 package for Training_Cluster — RT-DETR-R18

RT-DETR-R18 baseline on `CottonWeedDet12_v1` with:
- locked train/val/test split usage
- paper-grade artifact capture
- optional smoke training
- Optuna tuning on train+val only
- automatic full training in the same Slurm job after Optuna completes successfully
- ONNX export for later Jetson deployment

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
- GPU: `1 x H100`
- CPUs: `32`
- RAM: `96G`
- time: `12:00:00`

### Optuna
- GPU: `1 x H100`
- CPUs: `32`
- RAM: `96G`
- time: `72:00:00`

### Full training
- GPU: `1 x H100`
- CPUs: `32`
- RAM: `96G`
- time: `72:00:00`

## Training policy matched to the corrected YOLO11s package

- full training epochs: `300`
- Optuna trials: `12`
- max epochs per trial: `60`
- same single-job chaining policy: Optuna first, then full training with best hyperparameters

## What to transfer to Training_Cluster

If the Training_Cluster dataset already exists at `${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1`, transfer these:
- `config/`
- `scripts/`
- `vendor/`
- `requirements.txt`
- `README.md`
- `README_STAGE2_TRAINING_CLUSTER_RTDETR_R18.md`
- `MODEL_ASSET_NOTE.txt`
- `rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`

Do not transfer these for the real Training_Cluster package:
- `runtime_local/`
- `dataset/`
- `notebook/`

Reason:
- `runtime_local/` is local-debug state and uses local paths
- `dataset/` is only the local subset package
- `notebook/` is not required at runtime

## Recommended replace command

From the local repo root, use `rsync` so the Training_Cluster folder becomes the package root:

```bash
rsync -av --delete \
  --exclude runtime_local \
  --exclude dataset \
  --exclude notebook \
  --exclude '*.zip' \
  --exclude tree_training_cluster.txt \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  ./ ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster/
```

If you also want to carry the pretrained checkpoint explicitly:

```bash
rsync -av rtdetr_r18vd_5x_coco_objects365_from_paddle.pth \
  ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster/
```

Note: the scripts can auto-download the checkpoint, but transferring it is safer.

## Recommended execution order on Training_Cluster

### 1) replace the package folder

```bash
mkdir -p ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster
```

### 2) create or update the SAM environment

```bash
interact -c 4 --mem=16G --time=02:00:00
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster
bash scripts/setup_stage2_env.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster /home/$USER/envs/SAM
exit
```

### 3) submit prepare

```bash
cd ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster
bash scripts/submit_prepare.sh
```

### 4) optional smoke test

```bash
bash scripts/submit_rtdetr_r18.sh --mode smoke --partition gpu_p --gres gpu:H100:1 --cpus 32 --mem 96G
```

### 5) main command: Optuna + full training in one job

```bash
bash scripts/submit_optuna.sh --partition gpu_p --gres gpu:H100:1 --cpus 32 --mem 96G
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
VENV_PATH=/home/$USER/envs/SAM \
  bash scripts/run_pipeline.sh ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/stage2_rtdetr_r18_training_cluster status
```

Latest logs:

```bash
tail -f $(ls -t rtd18_*.out | head -1)
tail -f $(ls -t rtd18_*.err | head -1)
```
