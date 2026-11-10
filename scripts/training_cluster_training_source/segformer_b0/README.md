# SegFormer-B0 Local Test Package

This folder is a portable local test package for SegFormer-B0 semantic segmentation on CropAndWeed.

What is included:
- `config/`: the local-debug and Training_Cluster locked configs
- `scripts/`: the training entry, Optuna tuner, pipeline, and wrappers
- `vendor/segformer_b0`: vendored SegFormer-B0 architecture adapted for offline Training_Cluster runs
- `docs/`: tuning and split policy notes

Local quick start:
1. Reuse the shared segmentation environment that was already created:

```bash
source ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/segment_env/activate_env.sh /home/$USER/envs/segment
```

2. Optional environment verification for this package:

```bash
bash scripts/setup_stage2_env.sh
```

3. Smoke run on a fake local dataset:

```bash
./scripts/run_local_smoke.sh
```

4. Local Optuna run on the fake dataset:

```bash
./scripts/run_local_optuna.sh
```

5. Local full run using the same package flow:

```bash
./scripts/run_local_full.sh
```

6. Local chained workflow `Optuna -> full`:

```bash
./scripts/run_local_tune_then_full.sh
```

Notes:
- The local scripts auto-create `runtime_local/fake_cropandweed` if it does not exist.
- `config/segmentation_segformer_b0_local_subset100.yaml` is CPU-safe and small enough for debugging.
- The real Training_Cluster package should use `config/segmentation_segformer_b0_train_lock.yaml`.
- The wrappers now activate the shared `segment` environment themselves, so they do not create a new env.
- The Training_Cluster submit helpers now default to `1 node x 4 L4` and can also be overridden with `--partition`, `--nodes`, `--gpu-type`, `--gpus`, or explicit `--gres`.
