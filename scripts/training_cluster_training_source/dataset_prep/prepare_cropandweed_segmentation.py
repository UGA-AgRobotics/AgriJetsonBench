#!/usr/bin/env python3
"""
Prepare and lock the CropAndWeed dataset for the semantic-segmentation branch of the
Jetson edge-AI benchmark project.

Locked methodological decisions in this revised bundle
-----------------------------------------------------
- Task: semantic segmentation
- Canonical class mapping: 0=background, 1=crop, 2=weed
- Split policy: one deterministic 70/10/20 train/val/test split
- Split grouping: session-aware at the image/session level
- Tiling policy: disabled for the main benchmark branch
- Runtime policy: full-image storage; resizing/cropping happens later in training/inference
  pipelines. The locked recommended runtime target is single-pass fixed-size 640x640 for
  validation, test, and Jetson deployment. Training-time random resized crop/scale is left to
  the later model-training packages and is intentionally NOT applied here during dataset prep.

Design intent
-------------
This script creates a reproducible full-image semantic-segmentation dataset package that matches
an edge-deployment benchmark more closely than a tile-first benchmark. It preserves the original
image resolution in the stored dataset and writes canonical remapped masks and fixed split
manifests for later training and Jetson evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_SLUG = "CropAndWeed_v1"
BENCHMARK_VARIANT = "CropOrWeed2_semseg_fullimage_fixed640_v2"
LICENSE_SUMMARY = "Non-commercial research/teaching/publication use only (see CropAndWeed LICENCE)"
DEFAULT_SEED = 20260417
DEFAULT_EVAL_SIZE = 640
SPLIT_POLICY = {"train": 0.70, "val": 0.10, "test": 0.20}
CANONICAL_CLASSES = ["background", "crop", "weed"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}

ARCHIVE_SPECS = [
    ("cropandweed_annotations.tar", "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_annotations.tar"),
    ("cropandweed_images1of4.tar", "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_images1of4.tar"),
    ("cropandweed_images2of4.tar", "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_images2of4.tar"),
    ("cropandweed_images3of4.tar", "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_images3of4.tar"),
    ("cropandweed_images4of4.tar", "https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_images4of4.tar"),
]

# Reproduced from the official CropAndWeed repo's dataset mappings for a crop/weed variant.
# Raw IDs mapped to crop -> 1, weed -> 2, anything else -> 0 background.
CROP_SOURCE_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 94, 24, 18, 13, 26, 27, 15
}
WEED_SOURCE_IDS = {
    31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86,
    32,
    29, 33, 37, 49,
    30, 44, 66, 87, 89, 91,
    61,
    79,
    34, 41, 52,
    35,
    36, 78,
    38, 39, 71, 72, 88,
    42,
    45, 70,
    47, 51, 54, 58, 60, 80, 83, 96,
    22, 63, 85,
    56, 57, 64, 77,
    50, 59, 67, 76,
}


@dataclass
class ImageRecord:
    image_id: str
    image_stem: str
    set_prefix: str
    session_id: str
    original_image_rel: str
    original_mask_rel: str
    original_params_rel: str
    canonical_image_abs: str
    canonical_mask_abs: str
    width: int
    height: int
    crop_pixels: int
    weed_pixels: int
    background_pixels: int
    image_sha256: str
    mask_sha256: str
    split: str = ""


def log(msg: str) -> None:
    print(f"[prepare_cropandweed_segmentation] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_text_lines(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_yaml(path: Path, data: object) -> None:
    import yaml
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def download_with_progress(url: str, dest: Path, force: bool = False) -> None:
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and not force:
        log(f"download exists, skipping: {dest}")
        return
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"download {dest.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(dest)


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with tarfile.open(archive_path, "r") as tf:
        members = tf.getmembers()
        dest_resolved = dest_dir.resolve()
        for m in members:
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe tar member path detected in {archive_path}: {m.name}")
        tf.extractall(dest_dir)


def extract_repo_snapshot(repo_zip: Optional[Path], dataset_root: Path, force: bool = False) -> Optional[Dict[str, str]]:
    if repo_zip is None:
        return None
    repo_zip = repo_zip.expanduser().resolve()
    if not repo_zip.exists():
        raise FileNotFoundError(f"Repo zip not found: {repo_zip}")
    tools_dir = dataset_root / "tools"
    repo_snapshot_dir = tools_dir / "cropandweed_repo_snapshot"
    if repo_snapshot_dir.exists() and force:
        shutil.rmtree(repo_snapshot_dir)
    if not repo_snapshot_dir.exists():
        ensure_dir(repo_snapshot_dir)
        with zipfile.ZipFile(repo_zip) as zf:
            zf.extractall(repo_snapshot_dir)
    repo_readme = next(repo_snapshot_dir.rglob("README.md"), None)
    repo_datasets_py = next(repo_snapshot_dir.rglob("datasets.py"), None)
    info = {
        "repo_zip": str(repo_zip),
        "repo_zip_sha256": sha256_file(repo_zip),
        "snapshot_dir": str(repo_snapshot_dir.resolve()),
        "repo_readme": str(repo_readme.resolve()) if repo_readme else "",
        "repo_datasets_py": str(repo_datasets_py.resolve()) if repo_datasets_py else "",
    }
    with (tools_dir / "repo_snapshot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def normalize_extracted_structure(extracted_root: Path, force: bool = False) -> None:
    """Normalize extracted archives into a predictable layout."""
    images_dir = extracted_root / "images"
    if not images_dir.exists():
        raise RuntimeError(f"Expected images directory after extraction: {images_dir}")

    bboxes_dir = extracted_root / "bboxes"
    if bboxes_dir.exists():
        direct_csvs = [p for p in bboxes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        crop_dir = bboxes_dir / "CropAndWeed"
        if direct_csvs:
            ensure_dir(crop_dir)
            for p in direct_csvs:
                target = crop_dir / p.name
                if target.exists() and force:
                    target.unlink()
                if not target.exists():
                    shutil.move(str(p), str(target))

    labelids_dir = extracted_root / "labelIds"
    if not labelids_dir.exists():
        raise RuntimeError(f"Expected labelIds directory after extraction: {labelids_dir}")
    direct_pngs = [p for p in labelids_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if direct_pngs:
        crop_dir = labelids_dir / "CropAndWeed"
        ensure_dir(crop_dir)
        for p in direct_pngs:
            target = crop_dir / p.name
            if target.exists() and force:
                target.unlink()
            if not target.exists():
                shutil.move(str(p), str(target))

    params_dir = extracted_root / "params"
    if params_dir.exists():
        direct_files = [p for p in params_dir.iterdir() if p.is_file()]
        if direct_files:
            crop_dir = params_dir / "CropAndWeed"
            ensure_dir(crop_dir)
            for p in direct_files:
                target = crop_dir / p.name
                if target.exists() and force:
                    target.unlink()
                if not target.exists():
                    shutil.move(str(p), str(target))


def find_mask_dir(extracted_root: Path) -> Path:
    candidates = [
        extracted_root / "labelIds" / "CropAndWeed",
        extracted_root / "labelIds",
    ]
    for c in candidates:
        if c.exists():
            pngs = list(c.glob("*.png"))
            if pngs:
                return c
    raise RuntimeError(f"Could not locate CropAndWeed semantic masks under {extracted_root}")


def find_images_dir(extracted_root: Path) -> Path:
    candidates = [extracted_root / "images"]
    for c in candidates:
        if c.exists():
            jpgs = list(c.glob("*.jpg")) + list(c.glob("*.jpeg")) + list(c.glob("*.png"))
            if jpgs:
                return c
    raise RuntimeError(f"Could not locate dataset images under {extracted_root}")


def find_params_dir(extracted_root: Path) -> Optional[Path]:
    candidates = [
        extracted_root / "params" / "CropAndWeed",
        extracted_root / "params",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_session_info(image_stem: str) -> Tuple[str, str]:
    parts = image_stem.split("-")
    if len(parts) >= 3:
        set_prefix = parts[0]
        session_id = f"{parts[0]}-{parts[1]}"
        return set_prefix, session_id
    return "unknown", f"unknown-{image_stem}"


def make_symlink(target: Path, link_path: Path, force: bool = False) -> None:
    ensure_dir(link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        if not force:
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target, link_path)


def map_mask_to_canonical(raw_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw_mask, dtype=np.uint8)
    out[np.isin(raw_mask, list(CROP_SOURCE_IDS))] = 1
    out[np.isin(raw_mask, list(WEED_SOURCE_IDS))] = 2
    return out


def matching_param_file(params_dir: Optional[Path], stem: str) -> str:
    if params_dir is None or not params_dir.exists():
        return ""
    matches = sorted(params_dir.glob(f"{stem}.*"))
    return matches[0].name if matches else ""


def split_indices_multilabel_session_aware(
    session_keys: List[str],
    session_features: np.ndarray,
    session_image_counts: np.ndarray,
    seed: int,
    test_ratio: float,
    val_ratio_of_total: float,
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Split session IDs into train/val/test.

    Primary method:
      - Multilabel stratified split at session level on [has_crop, has_weed, is_ave, is_vwg].

    Fallback:
      - Deterministic greedy split by image count.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        idx = np.arange(len(session_keys))
        if len(session_keys) >= 10:
            stage1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            trainval_idx, test_idx = next(stage1.split(idx, session_features))

            val_ratio_within_trainval = val_ratio_of_total / (1.0 - test_ratio)
            stage2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within_trainval, random_state=seed)
            rel_train_idx, rel_val_idx = next(stage2.split(trainval_idx, session_features[trainval_idx]))

            train_idx = trainval_idx[rel_train_idx]
            val_idx = trainval_idx[rel_val_idx]
            return (
                [session_keys[i] for i in sorted(train_idx.tolist())],
                [session_keys[i] for i in sorted(val_idx.tolist())],
                [session_keys[i] for i in sorted(test_idx.tolist())],
                "session-level multilabel iterative stratification",
            )
    except Exception as exc:
        log(f"warning: iterative session split failed, falling back to greedy split. Reason: {exc}")

    rng = random.Random(seed)
    order = list(range(len(session_keys)))
    rng.shuffle(order)
    order.sort(key=lambda i: session_image_counts[i], reverse=True)

    total_images = int(session_image_counts.sum())
    targets = {
        "train": total_images * SPLIT_POLICY["train"],
        "val": total_images * SPLIT_POLICY["val"],
        "test": total_images * SPLIT_POLICY["test"],
    }
    bins = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}

    for i in order:
        deficits = {k: targets[k] - current[k] for k in current}
        best_split = sorted(deficits.items(), key=lambda kv: (-kv[1], current[kv[0]]))[0][0]
        bins[best_split].append(session_keys[i])
        current[best_split] += int(session_image_counts[i])

    return bins["train"], bins["val"], bins["test"], "session-aware greedy fallback by image count"


def generate_runtime_dataset_and_split(
    dataset_root: Path,
    extracted_root: Path,
    force: bool,
    seed: int,
    eval_size: int,
) -> Tuple[List[ImageRecord], Dict[str, object]]:
    images_dir = find_images_dir(extracted_root)
    masks_dir = find_mask_dir(extracted_root)
    params_dir = find_params_dir(extracted_root)

    canonical_root = dataset_root / "canonical_semseg"
    canonical_images_dir = canonical_root / "images"
    canonical_masks_dir = canonical_root / "masks"
    canonical_params_dir = canonical_root / "params"
    metadata_dir = dataset_root / "metadata"
    manifests_dir = dataset_root / "splits" / "manifests"
    ensure_dir(canonical_images_dir)
    ensure_dir(canonical_masks_dir)
    ensure_dir(canonical_params_dir)
    ensure_dir(metadata_dir)
    ensure_dir(manifests_dir)

    image_paths: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths[p.stem] = p

    mask_paths: Dict[str, Path] = {}
    for p in masks_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            mask_paths[p.stem] = p

    common_stems = sorted(set(image_paths.keys()) & set(mask_paths.keys()))
    if not common_stems:
        raise RuntimeError("No matching image/mask pairs were found after download/extraction")

    records: List[ImageRecord] = []
    crop_total = 0
    weed_total = 0
    widths = []
    heights = []

    for stem in tqdm(common_stems, desc="canonicalize full-image segmentation samples"):
        img_path = image_paths[stem]
        raw_mask_path = mask_paths[stem]

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            width, height = im.size
        with Image.open(raw_mask_path) as m:
            raw_mask = np.array(m, dtype=np.uint8)

        canonical_mask = map_mask_to_canonical(raw_mask)
        crop_pixels = int((canonical_mask == 1).sum())
        weed_pixels = int((canonical_mask == 2).sum())
        bg_pixels = int((canonical_mask == 0).sum())

        if crop_pixels + weed_pixels == 0:
            # Keep the benchmark focused on mapped crop/weed content.
            continue

        set_prefix, session_id = parse_session_info(stem)
        image_id = stem

        canonical_img_link = canonical_images_dir / f"{image_id}{img_path.suffix.lower()}"
        canonical_mask_file = canonical_masks_dir / f"{image_id}.png"

        make_symlink(img_path.resolve(), canonical_img_link, force=force)
        if canonical_mask_file.exists() and force:
            canonical_mask_file.unlink()
        if not canonical_mask_file.exists():
            Image.fromarray(canonical_mask, mode="L").save(canonical_mask_file)

        param_name = matching_param_file(params_dir, stem)
        original_param_rel = ""
        if param_name:
            param_src = (params_dir / param_name).resolve()
            param_link = canonical_params_dir / param_name
            make_symlink(param_src, param_link, force=force)
            original_param_rel = str(param_src.relative_to(extracted_root.resolve()))

        records.append(
            ImageRecord(
                image_id=image_id,
                image_stem=stem,
                set_prefix=set_prefix,
                session_id=session_id,
                original_image_rel=str(img_path.resolve().relative_to(extracted_root.resolve())),
                original_mask_rel=str(raw_mask_path.resolve().relative_to(extracted_root.resolve())),
                original_params_rel=original_param_rel,
                canonical_image_abs=str(canonical_img_link.resolve()),
                canonical_mask_abs=str(canonical_mask_file.resolve()),
                width=width,
                height=height,
                crop_pixels=crop_pixels,
                weed_pixels=weed_pixels,
                background_pixels=bg_pixels,
                image_sha256=sha256_file(img_path.resolve()),
                mask_sha256=sha256_file(canonical_mask_file.resolve()),
            )
        )
        crop_total += crop_pixels
        weed_total += weed_pixels
        widths.append(width)
        heights.append(height)

    if not records:
        raise RuntimeError("No valid segmentation samples remained after canonical remapping")

    session_to_records: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        session_to_records.setdefault(rec.session_id, []).append(rec)

    session_keys = sorted(session_to_records.keys())
    features = []
    image_counts = []
    session_rows = []
    for session_id in session_keys:
        session_records = session_to_records[session_id]
        has_crop = int(any(r.crop_pixels > 0 for r in session_records))
        has_weed = int(any(r.weed_pixels > 0 for r in session_records))
        is_ave = int(any(r.set_prefix == "ave" for r in session_records))
        is_vwg = int(any(r.set_prefix == "vwg" for r in session_records))
        features.append([has_crop, has_weed, is_ave, is_vwg])
        image_counts.append(len(session_records))
        session_rows.append({
            "session_id": session_id,
            "set_prefix": session_records[0].set_prefix,
            "n_images": len(session_records),
            "has_crop": has_crop,
            "has_weed": has_weed,
            "crop_pixels_total": sum(r.crop_pixels for r in session_records),
            "weed_pixels_total": sum(r.weed_pixels for r in session_records),
        })

    features_arr = np.array(features, dtype=int)
    image_counts_arr = np.array(image_counts, dtype=int)
    train_sessions, val_sessions, test_sessions, split_method = split_indices_multilabel_session_aware(
        session_keys=session_keys,
        session_features=features_arr,
        session_image_counts=image_counts_arr,
        seed=seed,
        test_ratio=SPLIT_POLICY["test"],
        val_ratio_of_total=SPLIT_POLICY["val"],
    )
    split_lookup = {s: "train" for s in train_sessions}
    split_lookup.update({s: "val" for s in val_sessions})
    split_lookup.update({s: "test" for s in test_sessions})

    for rec in records:
        rec.split = split_lookup[rec.session_id]

    # Image-level manifests
    for split in ("train", "val", "test"):
        split_records = sorted((r for r in records if r.split == split), key=lambda r: r.canonical_image_abs)
        write_text_lines(manifests_dir / f"images_{split}.txt", [r.canonical_image_abs for r in split_records])
        write_text_lines(manifests_dir / f"masks_{split}.txt", [r.canonical_mask_abs for r in split_records])
        with (manifests_dir / f"pairs_{split}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "image_id", "session_id"])
            w.writeheader()
            for r in split_records:
                w.writerow({
                    "image_path": r.canonical_image_abs,
                    "mask_path": r.canonical_mask_abs,
                    "image_id": r.image_id,
                    "session_id": r.session_id,
                })
        if split == "test":
            write_text_lines(manifests_dir / "benchmark_order_test_fullimages.txt", [r.image_id for r in split_records])

    # Metadata files
    with (metadata_dir / "session_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["session_id", "set_prefix", "n_images", "has_crop", "has_weed", "crop_pixels_total", "weed_pixels_total"],
        )
        w.writeheader()
        w.writerows(session_rows)

    with (metadata_dir / "image_inventory_with_split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        for rec in records:
            w.writerow(asdict(rec))

    pixel_rows = []
    presence_rows = []
    for split in ("train", "val", "test"):
        split_records = [r for r in records if r.split == split]
        pixel_rows.extend([
            {"split": split, "class_name": "background", "pixels": sum(r.background_pixels for r in split_records)},
            {"split": split, "class_name": "crop", "pixels": sum(r.crop_pixels for r in split_records)},
            {"split": split, "class_name": "weed", "pixels": sum(r.weed_pixels for r in split_records)},
        ])
        presence_rows.append({
            "split": split,
            "n_images": len(split_records),
            "images_with_crop": sum(1 for r in split_records if r.crop_pixels > 0),
            "images_with_weed": sum(1 for r in split_records if r.weed_pixels > 0),
            "images_with_both": sum(1 for r in split_records if r.crop_pixels > 0 and r.weed_pixels > 0),
        })
    with (metadata_dir / "pixel_counts_by_split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class_name", "pixels"])
        w.writeheader()
        w.writerows(pixel_rows)
    with (metadata_dir / "image_presence_by_split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "n_images", "images_with_crop", "images_with_weed", "images_with_both"])
        w.writeheader()
        w.writerows(presence_rows)

    resolution_summary = {
        "width_min": int(np.min(widths)),
        "width_median": float(np.median(widths)),
        "width_max": int(np.max(widths)),
        "height_min": int(np.min(heights)),
        "height_median": float(np.median(heights)),
        "height_max": int(np.max(heights)),
        "n_images": len(records),
    }
    with (metadata_dir / "resolution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(resolution_summary, f, indent=2)

    split_summary = {
        "n_images_total": len(records),
        "n_sessions_total": len(session_keys),
        "n_images_train": sum(1 for r in records if r.split == "train"),
        "n_images_val": sum(1 for r in records if r.split == "val"),
        "n_images_test": sum(1 for r in records if r.split == "test"),
        "n_sessions_train": len(train_sessions),
        "n_sessions_val": len(val_sessions),
        "n_sessions_test": len(test_sessions),
        "split_method": split_method,
        "seed": seed,
        "canonical_classes": CANONICAL_CLASSES,
        "crop_pixels_total": crop_total,
        "weed_pixels_total": weed_total,
        "tiling_enabled": False,
        "recommended_eval_size": eval_size,
    }
    with (metadata_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    generic_yaml = {
        "dataset_id": DATASET_SLUG,
        "benchmark_variant": BENCHMARK_VARIANT,
        "task": "semantic_segmentation",
        "classes": {i: name for i, name in enumerate(CANONICAL_CLASSES)},
        "image_level_manifests": {
            "train_images": str((manifests_dir / "images_train.txt").resolve()),
            "train_masks": str((manifests_dir / "masks_train.txt").resolve()),
            "val_images": str((manifests_dir / "images_val.txt").resolve()),
            "val_masks": str((manifests_dir / "masks_val.txt").resolve()),
            "test_images": str((manifests_dir / "images_test.txt").resolve()),
            "test_masks": str((manifests_dir / "masks_test.txt").resolve()),
            "test_benchmark_order": str((manifests_dir / "benchmark_order_test_fullimages.txt").resolve()),
        },
        "canonical_root": str(canonical_root.resolve()),
        "runtime_input_policy_yaml": str((dataset_root / "runtime_input_policy.yaml").resolve()),
    }
    write_yaml(dataset_root / "segmentation_dataset_generic.yaml", generic_yaml)

    runtime_policy = {
        "tiling_enabled": False,
        "train_runtime_policy": {
            "apply_at_training_time": True,
            "recommended_strategy": "random resized crop / scale augmentation",
            "stored_dataset_images": "full-resolution canonical images",
            "note": "No resizing or cropping is baked into the dataset during preparation.",
        },
        "validation_test_runtime_policy": {
            "mode": "single-pass fixed-size resize",
            "fixed_input_size": eval_size,
            "note": "Use the same full-image fixed-size policy later for Jetson inference to keep FPS and latency meaningful.",
        },
        "jetson_deployment_policy": {
            "mode": "single-pass fixed-size resize",
            "fixed_input_size": eval_size,
            "batch_size": 1,
            "tiling": "disabled for the main benchmark path",
        },
    }
    write_yaml(dataset_root / "runtime_input_policy.yaml", runtime_policy)

    log(f"n canonical full-image samples: {len(records)}")
    return records, split_summary


def write_lock_files(
    dataset_root: Path,
    archives_info: List[Dict[str, str]],
    repo_snapshot_info: Optional[Dict[str, str]],
    split_summary: Dict[str, object],
    seed: int,
    eval_size: int,
) -> None:
    metadata_dir = dataset_root / "metadata"
    manifests_dir = dataset_root / "splits" / "manifests"
    lock = {
        "dataset_id": DATASET_SLUG,
        "benchmark_variant": BENCHMARK_VARIANT,
        "source": {
            "name": "CropAndWeed",
            "source_type": "official dataset archives downloaded using the project's published setup URLs",
            "download_archives": archives_info,
            "license": LICENSE_SUMMARY,
        },
        "project_use": {
            "task": "semantic_segmentation",
            "role": "main_segmentation_dataset",
            "benchmark_scope": "Jetson edge-AI benchmark for precision agriculture",
        },
        "canonical_format": {
            "images": "symlinks to original extracted images",
            "masks": "PNG semantic masks with IDs 0=background, 1=crop, 2=weed",
            "mask_mapping_source": "CropAndWeed raw label IDs mapped to crop/weed/background",
            "image_resize_at_prep": "none",
        },
        "split_policy": {
            "train": SPLIT_POLICY["train"],
            "val": SPLIT_POLICY["val"],
            "test": SPLIT_POLICY["test"],
            "method": split_summary["split_method"],
            "seed": seed,
            "group_unit": "recording session",
            "filename_pattern": "prefix-session-imageid, e.g. ave-0357-0008",
        },
        "runtime_input_policy": {
            "tiling_enabled": False,
            "training_runtime_preprocessing": "random resized crop / scale augmentation later during training, not during dataset prep",
            "validation_test_inference": "single-pass fixed-size resize",
            "fixed_input_size": eval_size,
            "note": "The stored dataset remains full-resolution; runtime resizing happens later in model-specific pipelines.",
        },
        "classes": {
            "canonical_order": CANONICAL_CLASSES,
            "crop_source_ids": sorted(CROP_SOURCE_IDS),
            "weed_source_ids": sorted(WEED_SOURCE_IDS),
        },
        "paths": {
            "dataset_root": str(dataset_root.resolve()),
            "canonical_images_dir": str((dataset_root / "canonical_semseg" / "images").resolve()),
            "canonical_masks_dir": str((dataset_root / "canonical_semseg" / "masks").resolve()),
            "image_inventory_csv": str((metadata_dir / "image_inventory_with_split.csv").resolve()),
            "session_inventory_csv": str((metadata_dir / "session_inventory.csv").resolve()),
            "split_summary_json": str((metadata_dir / "split_summary.json").resolve()),
            "resolution_summary_json": str((metadata_dir / "resolution_summary.json").resolve()),
            "generic_yaml": str((dataset_root / "segmentation_dataset_generic.yaml").resolve()),
            "runtime_input_policy_yaml": str((dataset_root / "runtime_input_policy.yaml").resolve()),
            "train_images_manifest": str((manifests_dir / "images_train.txt").resolve()),
            "val_images_manifest": str((manifests_dir / "images_val.txt").resolve()),
            "test_images_manifest": str((manifests_dir / "images_test.txt").resolve()),
            "test_benchmark_order": str((manifests_dir / "benchmark_order_test_fullimages.txt").resolve()),
        },
    }
    if repo_snapshot_info:
        lock["source"]["repo_snapshot"] = repo_snapshot_info

    write_yaml(dataset_root / "DATASET_LOCK.yaml", lock)
    write_yaml(dataset_root / "splits" / "split_policy.yaml", lock["split_policy"])

    readme = f"""# CropAndWeed locked segmentation package

This directory was prepared with `prepare_cropandweed_segmentation.py`.

## Locked benchmark decisions
- Source dataset: CropAndWeed
- Locked benchmark task: semantic segmentation
- Locked class mapping: background / crop / weed
- Canonical class IDs: 0=background, 1=crop, 2=weed
- Split policy: 70/10/20 train/val/test
- Split grouping: recording session (session-aware)
- Tiling: disabled for the main benchmark branch
- Locked recommended validation/test/Jetson runtime input: {eval_size}×{eval_size} single-pass resize
- Seed: {seed}

## Important notes
- Original images remain at original resolution and are preserved as symlink targets.
- Canonical masks are stored as PNG files and are the main locked semantic-segmentation labels.
- This dataset lock is intentionally full-image and no-tiling. Training-time random resized crop/scale should happen later in the model training package, not here.
- Please do not redistribute the downloaded dataset or modified copies if the dataset license does not allow that.
"""
    with (dataset_root / "README_LOCKED.md").open("w", encoding="utf-8") as f:
        f.write(readme)


def prepare_dataset(
    root: Path,
    repo_zip: Optional[Path],
    skip_download: bool,
    force: bool,
    seed: int,
    eval_size: int,
) -> None:
    dataset_root = root / "datasets" / DATASET_SLUG
    raw_root = dataset_root / "raw"
    raw_archives_dir = raw_root / "source_archives"
    extracted_root = raw_root / "extracted_original"
    ensure_dir(dataset_root)
    ensure_dir(raw_archives_dir)
    ensure_dir(extracted_root)

    repo_snapshot_info = extract_repo_snapshot(repo_zip, dataset_root, force=force)

    archives_info = []
    if not skip_download:
        for filename, url in ARCHIVE_SPECS:
            archive_path = raw_archives_dir / filename
            download_with_progress(url, archive_path, force=force)

    for filename, url in ARCHIVE_SPECS:
        archive_path = raw_archives_dir / filename
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Required archive missing: {archive_path}. Either run with --download-only first or provide --skip-download only after the archives exist."
            )
        archives_info.append({
            "filename": filename,
            "url": url,
            "path": str(archive_path.resolve()),
            "sha256": sha256_file(archive_path),
        })

    with (raw_root / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archives": archives_info,
                "repo_snapshot": repo_snapshot_info,
            },
            f,
            indent=2,
        )

    for filename, _ in ARCHIVE_SPECS:
        archive_path = raw_archives_dir / filename
        marker = extracted_root / f".extracted__{filename}"
        if marker.exists() and not force:
            continue
        log(f"extracting {archive_path.name}")
        _safe_extract_tar(archive_path, extracted_root)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")

    normalize_extracted_structure(extracted_root, force=force)
    records, split_summary = generate_runtime_dataset_and_split(
        dataset_root=dataset_root,
        extracted_root=extracted_root,
        force=force,
        seed=seed,
        eval_size=eval_size,
    )
    write_lock_files(
        dataset_root=dataset_root,
        archives_info=archives_info,
        repo_snapshot_info=repo_snapshot_info,
        split_summary=split_summary,
        seed=seed,
        eval_size=eval_size,
    )
    log("dataset preparation finished successfully")


def download_only(root: Path, repo_zip: Optional[Path], force: bool) -> None:
    dataset_root = root / "datasets" / DATASET_SLUG
    raw_archives_dir = dataset_root / "raw" / "source_archives"
    ensure_dir(raw_archives_dir)
    repo_snapshot_info = extract_repo_snapshot(repo_zip, dataset_root, force=force)

    archives_info = []
    for filename, url in ARCHIVE_SPECS:
        archive_path = raw_archives_dir / filename
        download_with_progress(url, archive_path, force=force)
        archives_info.append({
            "filename": filename,
            "url": url,
            "path": str(archive_path.resolve()),
            "sha256": sha256_file(archive_path),
        })
    with (dataset_root / "raw" / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archives": archives_info,
                "repo_snapshot": repo_snapshot_info,
            },
            f,
            indent=2,
        )
    log("download-only stage finished successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/prepare CropAndWeed segmentation dataset with a fixed project split.")
    parser.add_argument("--root", required=True, type=Path, help="Project root, e.g. /work/MYGROUP/jetson_edge_ai_benchmark")
    parser.add_argument(
        "--repo-zip",
        type=Path,
        default=None,
        help="Optional local path to the downloaded cropandweed-dataset GitHub repo zip for source snapshotting.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Do not download the dataset archives; require them to already exist in raw/source_archives.")
    parser.add_argument("--download-only", action="store_true", help="Only download the official source archives and snapshot the repo zip if provided, then exit.")
    parser.add_argument("--prepare-only", action="store_true", help="Skip the download stage and only do extract/remap/split/lock on already-downloaded archives.")
    parser.add_argument("--force", action="store_true", help="Overwrite derived files when possible.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Deterministic split seed (default: {DEFAULT_SEED}).")
    parser.add_argument(
        "--eval-size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help=f"Locked recommended fixed input size for validation/test/Jetson later (default: {DEFAULT_EVAL_SIZE}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    ensure_dir(root)

    if args.download_only and args.prepare_only:
        raise SystemExit("--download-only and --prepare-only cannot be used together")

    if args.download_only:
        download_only(root=root, repo_zip=args.repo_zip, force=args.force)
        return 0

    if args.prepare_only:
        prepare_dataset(
            root=root,
            repo_zip=args.repo_zip,
            skip_download=True,
            force=args.force,
            seed=args.seed,
            eval_size=args.eval_size,
        )
        return 0

    prepare_dataset(
        root=root,
        repo_zip=args.repo_zip,
        skip_download=args.skip_download,
        force=args.force,
        seed=args.seed,
        eval_size=args.eval_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
