# CropAndWeed locked segmentation package

This directory was prepared with `prepare_cropandweed_segmentation.py`.

## Locked benchmark decisions
- Source dataset: CropAndWeed
- Locked benchmark task: semantic segmentation
- Locked class mapping: background / crop / weed
- Canonical class IDs: 0=background, 1=crop, 2=weed
- Split policy: 70/10/20 train/val/test
- Split grouping: recording session (session-aware)
- Tiling: disabled for the main benchmark branch
- Locked recommended validation/test/Jetson runtime input: 640×640 single-pass resize
- Seed: 20260417

## Important notes
- Original images remain at original resolution and are preserved as symlink targets.
- Canonical masks are stored as PNG files and are the main locked semantic-segmentation labels.
- This dataset lock is intentionally full-image and no-tiling. Training-time random resized crop/scale should happen later in the model training package, not here.
- Please do not redistribute the downloaded dataset or modified copies if the dataset license does not allow that.
