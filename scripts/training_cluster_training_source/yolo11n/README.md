# Stage-2 v3 package for Training_Cluster
YOLO11n baseline on `CottonWeedDet12_v1` with **early stopping**, **optional Optuna tuning**, and **paper-grade artifact capture**

This package is the clean next step after dataset lock.

It supports this workflow:

1. `prepare` runtime dataset layer and paper metadata
2. `smoke` training to verify the whole stack
3. optional `Optuna` tuning on **train + val only**
4. `full` training with a large max-epoch budget plus early stopping
5. `val` and `test` evaluation
6. ONNX export for later Jetson deployment

## Important methodological decision

For **this benchmark project**, heavy per-model tuning is **not mandatory**.

A good paper-safe policy is:
- keep a locked base recipe for fairness
- optionally do a **small controlled tuning stage only for YOLO11n**
- do **not** touch the test split during tuning
- later, report Jetson results on the frozen **20% test split**

So this package gives you both options:
- skip tuning and go straight to full training
- or run a limited Optuna stage first and then let full training reuse the best hyperparameters automatically

## What changed compared with the previous package

- full training now uses `epochs=300` with `patience=40`
- checkpoint saving is no longer every epoch; `save_period=10` is used instead
- optional Optuna tuning is added
- tuning artifacts are saved for later paper writing
- Slurm defaults are now much more realistic for long training
- the README now clearly separates **setup resources** from **training resources**

## Resource meaning in this package

### Setup / environment creation
This is **not training**.

Interactive setup uses:
- `4` CPU cores
- `16G` RAM
- `2` hours
- **no GPU**

This stage is only for environment setup.

### Prepare stage
This is a small CPU batch job.

Default:
- `4` CPU cores
- `16G` RAM
- `2` hours
- **no GPU**

### Smoke training
Default batch submission:
- `1` GPU
- default type: `L4`
- `8` CPU cores
- `40G` RAM
- `4` hours

### Full training
Default batch submission:
- `1` GPU
- default type: `L4`
- `8` CPU cores
- `48G` RAM
- `48` hours

### Optuna tuning
Default batch submission:
- `1` GPU
- default type: `L4`
- `8` CPU cores
- `48G` RAM
- `72` hours

## GPU meaning

If you submit with:

```bash
--gres gpu:L4:1
```

that means:
- exactly **1 GPU**
- specifically **L4**

If you submit with:

```bash
--gres gpu:1
```

Training_Cluster can assign different GPU types, so keep that only if you are okay with mixed hardware.

## Is 2 hours enough?

- for **setup**: usually yes
- for **prepare**: usually yes
- for **full training**: no, 2 hours is not a good limit if you want many epochs
- for **Optuna**: definitely no

That is why this package now uses long walltimes for real training jobs.

## Why early stopping is included

You said you want the max epoch budget to be high, but you do not want waste.
That is exactly why this package now uses:
- `epochs=300`
- `patience=40`

So training can continue when improvement still exists, but it can stop before 300 if validation performance plateaus.

## Jetson split policy later

### Final reported inference / benchmarking on Jetson
Use the frozen **20% test split**.

### Quick debug on Jetson
Use the locked validation pilot subset:

```text
runtime_dataset/manifests/val_jetson_pilot.txt
```

So:
- **final Jetson results later -> test split**
- **debug / parser / first TensorRT checks -> validation pilot subset**

## Step-by-step on Training_Cluster

### 1) unpack the package on the login node

```bash
mkdir -p ~/jetson_edge_ai_benchmark
cd ~/jetson_edge_ai_benchmark
unzip stage2_yolo11n_training_cluster_package.zip
```

Expected folder:

```bash
~/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster
```

### 2) open a small interactive CPU session for environment setup

```bash
interact -c 4 --mem=16G --time=02:00:00
```

Then:

```bash
cd ~/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster
bash scripts/setup_stage2_env.sh \
  ~/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster \
  /home/$USER/envs/SAM
```

If the PyTorch module name is different:

```bash
module spider PyTorch
module spider torchvision
PYTORCH_MODULE='YOUR_MODULE_HERE' bash scripts/setup_stage2_env.sh \
  ~/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster \
  /home/$USER/envs/SAM
```

If the torchvision module also differs, set `TORCHVISION_MODULE='YOUR_MODULE_HERE'` in the same command.

For manual checks in an interactive shell, activate the same environment with:

```bash
source scripts/activate_stage2_env.sh /home/$USER/envs/SAM
```

Exit the interactive session after setup:

```bash
exit
```

### 3) submit prepare from the login node

```bash
cd ~/jetson_edge_ai_benchmark/stage2_yolo11n_training_cluster
bash scripts/submit_prepare.sh
```

### 4) submit smoke training

```bash
bash scripts/submit_yolo11n.sh --mode smoke
```

### 5) optional: run Optuna tuning

```bash
bash scripts/submit_optuna.sh
```

If you want a faster GPU and queue allows it:

```bash
bash scripts/submit_optuna.sh --gres gpu:A100:1
```

### 6) run full training

This stage automatically loads tuned hyperparameters if this file exists:

```text
${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/experiments/detection/CottonWeedDet12_v1/yolo11n_baseline_v3/optuna/best_hyperparameters.yaml
```

Submit full training:

```bash
bash scripts/submit_yolo11n.sh --mode full
```

Example with A100:

```bash
bash scripts/submit_yolo11n.sh \
  --mode full \
  --gres gpu:A100:1 \
  --time 48:00:00 \
  --cpus 8 \
  --mem 48G
```

## Resume behavior

### Prepare / smoke / full pipeline state
Saved in:

```text
state/pipeline_state.json
```

### Full training resume
Uses Ultralytics `last.pt`.
If a job stops, submit `full` again.

### Optuna tuning resume
Uses:

```text
optuna/study.sqlite3
```

If tuning stops, submit `submit_optuna.sh` again.
Stale running trials are marked failed and the study continues.

## Progress visibility

This package uses:
- `tqdm` for prepare progress
- `tqdm` epoch monitoring during smoke, tuning trials, and full training
- persistent JSON reports
- Slurm logs
- CSV results from Ultralytics

## Main saved outputs for the paper

Runtime root:

```text
${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/experiments/detection/CottonWeedDet12_v1/yolo11n_baseline_v3
```

Key outputs:
- `FINAL_SUMMARY.json`
- `FINAL_SUMMARY.md`
- `OUTPUT_FILE_MANIFEST.csv`
- `reports/effective_training_args.yaml`
- `optuna/study.sqlite3`
- `optuna/trials.csv`
- `optuna/best_hyperparameters.yaml`
- `ultralytics/train/results.csv`
- `artifacts/best.pt`
- `exports/best.onnx`

## Recommended order for your project right now

The best order is:

1. `prepare`
2. `smoke`
3. decide whether tuning is needed
4. if yes -> `Optuna`
5. `full`

If you want a cleaner benchmark paper with tighter scope, tuning can stay optional.
If you want a stronger accuracy baseline for YOLO11n, run the Optuna stage.
