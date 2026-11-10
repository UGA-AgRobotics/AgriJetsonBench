# MobileNetV3 Local Test Package

This folder is a portable local test package for MobileNetV3 semantic segmentation on CropAndWeed.

Default model choice in this package:
- `torchvision` segmentation with MobileNetV3 backbone
- default variant: `LRASPP MobileNetV3-Large`
- optional alternate variant in config: `DeepLabV3 MobileNetV3-Large`

What is included:
- `config/`: the local-debug and Training_Cluster locked configs
- `scripts/`: the training entry, Optuna tuner, pipeline, and wrappers
- `vendor/`: package notes for offline Training_Cluster usage
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
- `config/segmentation_mobilenetv3_local_subset100.yaml` is CPU-safe and small enough for debugging.
- The real Training_Cluster package should use `config/segmentation_mobilenetv3_train_lock.yaml`.
- The wrappers reuse the shared `segment` environment and do not create a new env.
- The Training_Cluster submit helpers default to `1 node x 4 L4` and can be overridden with `--partition`, `--nodes`, `--gpu-type`, `--gpus`, or explicit `--gres`.
