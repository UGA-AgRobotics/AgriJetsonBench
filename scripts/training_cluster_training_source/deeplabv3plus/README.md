# DeepLabV3++ Local Test Package

This folder is a portable local/Training_Cluster-ready package for DeepLabV3++ semantic segmentation on CropAndWeed.

What is included:
- `config/`: local-debug and Training_Cluster locked configs
- `scripts/`: training entry, Optuna tuner, pipeline, Slurm wrappers, and helpers
- `vendor/deeplabv3plus`: self-contained DeepLabV3++ implementation with ResNet backbones and no remote downloads
- `docs/`: tuning and split policy notes
- `runtime_local/fake_cropandweed`: tiny fake dataset for CPU-safe debug runs

Local quick start:
1. Reuse the shared segmentation environment that was already created:

```bash
source ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/segment_env/activate_env.sh /home/$USER/envs/segment
```

2. Optional environment verification for this package:

```bash
bash scripts/setup_stage2_env.sh
```

3. Smoke run on the fake local dataset:

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
- `config/segmentation_deeplabv3plus_local_subset100.yaml` is CPU-safe and intentionally tiny for package validation.
- The real Training_Cluster package should use `config/segmentation_deeplabv3plus_train_lock.yaml`.
- The wrappers activate the shared `segment` environment themselves, so they do not create a new env.
- The Training_Cluster submit helpers default to `1 node x 4 L4`.
- The Optuna/full-training chaining policy, trial count, and epoch counts are matched to the attached reference package. The default total train batch is reduced to `16` because DeepLabV3++ is materially heavier than BiSeNetV2.
