# Actual Model Artifacts + Artifact Manifest + SHA256

Created UTC: `2026-05-12T13:34:32.710367+00:00`

Destination: `project_progress/actual model artifacts + artifact manifest + sha256`

Scope:
- Copied actual local model artifacts: TensorRT engines, ONNX exports, PyTorch checkpoints/weights, and INT8 calibration caches.
- Included artifacts from `Benchmark` and Nano paper battery model assets.
- Excluded calibration/build helper code, `.build` helper binaries, `__pycache__`, and old code-backup snapshots.

Files:
- `artifacts/`: copied model artifacts preserving source-relative layout.
- `MODEL_ARTIFACTS_MANIFEST.csv` and `.json`: full artifact manifest with source path, copied path, model, precision, input size, size, and SHA256.
- `SHA256SUMS.txt`: sha256 for every copied artifact.
- `SUMMARY_BY_MODEL.csv`: compact per-model count/size/type summary.
- `DUPLICATES_BY_SHA256.csv`: artifacts with identical SHA256.
- `ARTIFACT_COLLECTION_SUMMARY.json`: machine-readable collection summary.

Model Counts:
- `bisenetv2`: 47 artifacts, 442.5 MiB
- `deeplab`: 20 artifacts, 2562.9 MiB
- `mobilenet`: 20 artifacts, 212.9 MiB
- `rtdetr_r18`: 19 artifacts, 1368.2 MiB
- `segformerb0`: 22 artifacts, 292.3 MiB
- `yolo11n`: 49 artifacts, 382.4 MiB
- `yolo11s`: 50 artifacts, 1247.8 MiB

Total artifacts: `227`
Total copied/verified size: `6.356 GiB`
Unique SHA256 hashes: `169`
