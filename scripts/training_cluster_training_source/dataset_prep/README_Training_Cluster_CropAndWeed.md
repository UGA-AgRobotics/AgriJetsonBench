  Man pishnahad midam in flow ra ejra koni:

  Rooye xfer:

  ssh anonymous@example.invalid
  export GROUP=research_group
  cd ~/cropandweed_segmentation_prep

  bash setup_env.sh /home/$USER/envs/jetson_cropandweed_prep
  source /home/$USER/envs/jetson_cropandweed_prep/bin/activate

  python prepare_cropandweed_segmentation.py \
    --root /work/$GROUP/jetson_edge_ai_benchmark \
    --download-only

  Bad rooye compute:

  ssh anonymous@example.invalid
  interact -c 8 --mem=64G --time=08:00:00

  export GROUP=research_group
  cd ~/cropandweed_segmentation_prep
  source /home/$USER/envs/jetson_cropandweed_prep/bin/activate

  python prepare_cropandweed_segmentation.py \
    --root /work/$GROUP/jetson_edge_ai_benchmark \
    --repo-zip /home/$USER/cropandweed-dataset-main.zip \
    --skip-download

  Agar bekhay, man mitoonam alan yek version tamiz az command-ha ro barat bedam
  ke faghat copy/paste koni ba GROUP khodet.






# CropAndWeed segmentation dataset prep on Training_Cluster (revised no-tiling version)

This bundle prepares and **locks** the CropAndWeed dataset for the segmentation branch of the Jetson edge-AI benchmark project.

## What changed in this revised version

The old bundle generated deterministic 512×512 tiles after the split.
This revised bundle **does not tile the dataset**.

The locked policy is now:
- full-image canonical semantic segmentation dataset
- class mapping = `background / crop / weed`
- one deterministic, session-aware `70/10/20` split
- **no tiling for the main benchmark path**
- recommended later runtime policy:
  - training: random resized crop / scale augmentation
  - validation/test/Jetson inference: **single-pass fixed-size resize to 640×640**

## Why this version is better for this project

Your main benchmark is edge deployment on Jetson boards, where FPS, latency, and practical real-time behavior matter.
A no-tiling, single-pass full-image path is a cleaner main benchmark policy for that goal.
This bundle therefore stores full-resolution canonical images and masks, and leaves all resizing/cropping to later training and inference packages.

## What this bundle does

- downloads the official CropAndWeed source archives
- optionally snapshots the exact GitHub repo zip you already downloaded
- extracts and normalizes the dataset layout
- creates a **canonical semantic-segmentation version** with:
  - `0 = background`
  - `1 = crop`
  - `2 = weed`
- creates **one deterministic 70/10/20 train/val/test split**
- makes the split **session-aware**
- writes lock files, manifests, inventories, and a generic segmentation dataset YAML
- writes a separate `runtime_input_policy.yaml` that documents the later no-tiling 640×640 runtime policy

## Bundle contents

- `prepare_cropandweed_segmentation.py` – main Python script
- `prepare_cropandweed_segmentation.slurm` – sample Slurm job
- `run_prepare_cropandweed.sh` – convenient wrapper with logs
- `requirements.txt`
- `setup_env.sh`
- `CropAndWeed_LOCK.yaml`

## Expected dataset root after preparation

```text
/work/<GROUP>/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1
├── DATASET_LOCK.yaml
├── README_LOCKED.md
├── segmentation_dataset_generic.yaml
├── runtime_input_policy.yaml
├── canonical_semseg
│   ├── images
│   ├── masks
│   └── params
├── metadata
│   ├── image_inventory_with_split.csv
│   ├── session_inventory.csv
│   ├── pixel_counts_by_split.csv
│   ├── image_presence_by_split.csv
│   ├── resolution_summary.json
│   └── split_summary.json
├── raw
│   ├── source_archives
│   └── extracted_original
├── splits
│   ├── split_policy.yaml
│   └── manifests
│       ├── images_train.txt
│       ├── masks_train.txt
│       ├── images_val.txt
│       ├── masks_val.txt
│       ├── images_test.txt
│       ├── masks_test.txt
│       ├── pairs_train.csv
│       ├── pairs_val.csv
│       ├── pairs_test.csv
│       └── benchmark_order_test_fullimages.txt
└── tools
    └── cropandweed_repo_snapshot   # only if --repo-zip is provided
```

## Suggested Training_Cluster workflow

### 1) Transfer this bundle and your repo zip to Training_Cluster

From your local machine:

```bash
scp -r cropandweed_segmentation_prep anonymous@example.invalid:${TRAINING_CLUSTER_HOME}/
scp cropandweed-dataset-main.zip anonymous@example.invalid:${TRAINING_CLUSTER_HOME}/
```

### 2) Put the scripts in your project code area

```bash
ssh anonymous@example.invalid

mkdir -p ~/jetson_edge_ai_benchmark/scripts
cp -r ~/cropandweed_segmentation_prep/* ~/jetson_edge_ai_benchmark/scripts/
```

### 3) Create the dataset root on `/work`

```bash
export GROUP=MYGROUP
mkdir -p /work/$GROUP/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1/raw/source_archives
mkdir -p /home/$USER/envs
```

### 4) Download the official source archives

On the xfer node:

```bash
cd ~/jetson_edge_ai_benchmark/scripts

python3 prepare_cropandweed_segmentation.py   --root /work/$GROUP/jetson_edge_ai_benchmark   --repo-zip /home/$USER/cropandweed-dataset-main.zip   --download-only
```

### 5) Start an interactive compute session for the prepare stage

```bash
ssh anonymous@example.invalid
interact -c 8 --mem=64G --time=08:00:00
```

### 6) Create a Python environment

Inside the interactive session:

```bash
cd ~/jetson_edge_ai_benchmark/scripts
bash setup_env.sh /home/$USER/envs/jetson_cropandweed_prep
source /home/$USER/envs/jetson_cropandweed_prep/bin/activate
```

### 7) Run the full prepare stage

```bash
python ~/jetson_edge_ai_benchmark/scripts/prepare_cropandweed_segmentation.py   --root /work/$GROUP/jetson_edge_ai_benchmark   --repo-zip /home/$USER/cropandweed-dataset-main.zip   --skip-download
```

### 8) Or use the wrapper with logs

```bash
cd ~/jetson_edge_ai_benchmark/scripts
source /home/$USER/envs/jetson_cropandweed_prep/bin/activate

ROOT=/work/$GROUP/jetson_edge_ai_benchmark REPO_ZIP=/home/$USER/cropandweed-dataset-main.zip ENV_PATH=/home/$USER/envs/jetson_cropandweed_prep bash run_prepare_cropandweed.sh --prepare-only
```

### 9) Or use the sample Slurm job

```bash
cd ~/jetson_edge_ai_benchmark/scripts

sbatch   --export=ROOT=/work/$GROUP/jetson_edge_ai_benchmark,REPO_ZIP=/home/$USER/cropandweed-dataset-main.zip,ENV_PATH=/home/$USER/envs/jetson_cropandweed_prep   prepare_cropandweed_segmentation.slurm
```

## Important locked decisions

- benchmark dataset root: `CropAndWeed_v1`
- benchmark task: semantic segmentation
- class mapping: `background / crop / weed`
- split policy: `70/10/20`
- split grouping: `recording session`
- **tiling: disabled**
- later validation/test/Jetson runtime policy: **single-pass 640×640 fixed-size resize**

## Good checks after the run

```bash
export GROUP=MYGROUP
export DATA_ROOT=/work/$GROUP/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1

ls $DATA_ROOT
ls $DATA_ROOT/metadata
ls $DATA_ROOT/splits/manifests
cat $DATA_ROOT/DATASET_LOCK.yaml
cat $DATA_ROOT/runtime_input_policy.yaml
cat $DATA_ROOT/metadata/split_summary.json
```

## Notes

- The dataset license says the dataset is for non-commercial use and should not be redistributed as data.
- If later you want to revise the segmentation benchmark, create a new dataset version directory instead of editing this locked one in place.
