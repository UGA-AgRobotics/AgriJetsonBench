# CottonWeedDet12 locked dataset package

This directory was prepared with `prepare_cottonweeddet12.py`.

## Locked decisions
- Dataset: CottonWeedDet12 v1
- DOI: 10.5281/zenodo.7535814
- Archive MD5: f02e4c32d27b042a4f599a3305c99414
- Canonical detection format: YOLO txt labels + symlinked original images
- Split policy: 70/10/20 train/val/test
- Split method: image-level multilabel iterative stratification
- Seed: 20260414
- Number of generated splits: 1
- Resume behavior: automatic; rerun the same command after interruptions

## Key paths
- Raw archive: ${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1/raw/CottonWeedDet12.7z
- Extracted original data: ${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1/raw/extracted_original
- Canonical YOLO data: ${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1/canonical_yolo
- Inventory CSV: ${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1/metadata/inventory_with_split.csv
- Ultralytics YAML: ${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1/ultralytics/CottonWeedDet12_locked.yaml
- Resume state file: ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/logs/cottonweeddet12/state/prepare_state.json
- Central log directory: ${TRAINING_CLUSTER_HOME}/jetson_edge_ai_benchmark/logs/cottonweeddet12

## Notes
- Images stay at original resolution. Resize/letterbox should be applied later in the training/inference pipeline, not baked into the dataset.
- The raw archive and extracted original data are preserved.
- Split manifests are text files that list absolute image paths.
- If a run stops, inspect the state JSON and the latest log file under the central log directory.
