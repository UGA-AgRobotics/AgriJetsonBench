#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DETECTION_ROOT = Path("${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CottonWeedDet12_v1")
SEGMENTATION_ROOT = Path("${TRAINING_CLUSTER_WORK}/jetson_edge_ai_benchmark/datasets/CropAndWeed_v1")


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    task: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect split files, class names, and license notes for the two benchmark datasets."
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Destination directory where the compact dataset-documentation bundle will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst_dir: Path, dst_name: str | None = None) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / (dst_name or src.name)
    shutil.copy2(src, dst)
    return dst


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_scalar(text: str, key: str) -> str | None:
    lines = text.splitlines()
    key_pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*(.*?)\s*$")
    for idx, line in enumerate(lines):
        match = key_pattern.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        parts = [match.group(2).strip()]
        next_idx = idx + 1
        while next_idx < len(lines):
            next_line = lines[next_idx]
            if not next_line.strip():
                break
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= indent:
                break
            parts.append(next_line.strip())
            next_idx += 1
        value = " ".join(part for part in parts if part).strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        return value
    return None


def parse_indexed_mapping(path: Path, section_name: str) -> list[str]:
    text = read_text(path)
    lines = text.splitlines()
    in_section = False
    mapping: dict[int, str] = {}
    section_pattern = re.compile(rf"^{re.escape(section_name)}:\s*$")
    item_pattern = re.compile(r"^\s+(\d+):\s*(.+?)\s*$")
    for line in lines:
        if not in_section:
            if section_pattern.match(line):
                in_section = True
            continue
        if line and not line.startswith(" "):
            break
        match = item_pattern.match(line)
        if match:
            idx = int(match.group(1))
            value = match.group(2).strip().strip("'").strip('"')
            mapping[idx] = value
    return [mapping[idx] for idx in sorted(mapping)]


def count_manifest_entries(path: Path) -> int:
    lines = [line.strip() for line in read_text(path).splitlines() if line.strip()]
    if not lines:
        return 0
    header = lines[0].lower()
    if "image_path" in header or "mask_path" in header:
        return max(len(lines) - 1, 0)
    return len(lines)


def relpath_list(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def build_detection_bundle(out_root: Path, generated_at: str, slurm_job_id: str | None) -> dict:
    dataset_id = "CottonWeedDet12_v1"
    dataset_dir = out_root / dataset_id
    split_dir = dataset_dir / "split_files"
    class_dir = dataset_dir / "class_names"
    license_dir = dataset_dir / "license_notes"

    split_manifest_dir = DETECTION_ROOT / "splits" / "manifests"
    split_policy = DETECTION_ROOT / "splits" / "split_policy.yaml"
    split_summary = DETECTION_ROOT / "metadata" / "split_summary.json"
    dataset_lock = DETECTION_ROOT / "DATASET_LOCK.yaml"
    readme_locked = DETECTION_ROOT / "README_LOCKED.md"
    yaml_locked = DETECTION_ROOT / "ultralytics" / "CottonWeedDet12_locked.yaml"

    copied_split_files = [
        copy_file(split_manifest_dir / "train.txt", split_dir),
        copy_file(split_manifest_dir / "val.txt", split_dir),
        copy_file(split_manifest_dir / "test.txt", split_dir),
        copy_file(split_policy, split_dir),
        copy_file(split_summary, split_dir),
    ]

    class_names = parse_indexed_mapping(yaml_locked, "names")
    copy_file(yaml_locked, class_dir)
    write_text(
        class_dir / "class_names.txt",
        "\n".join(f"{idx:02d} {name}" for idx, name in enumerate(class_names)) + "\n",
    )
    write_text(
        class_dir / "class_names.json",
        json.dumps(
            {
                "dataset_id": dataset_id,
                "task": "object_detection",
                "nc": len(class_names),
                "class_names": class_names,
            },
            indent=2,
        )
        + "\n",
    )

    lock_text = read_text(dataset_lock)
    readme_text = read_text(readme_locked)
    license_summary = extract_scalar(lock_text, "license") or "CC BY-NC 4.0"
    doi = extract_scalar(lock_text, "doi") or "10.5281/zenodo.7535814"
    note_text = f"""# License Notes

Dataset: `{dataset_id}`
Task: object_detection
Generated: {generated_at}
Slurm job id: {slurm_job_id or "local"}

## Recorded license summary
- `{license_summary}`

## Recorded source reference
- DOI: `{doi}`
- Dataset lock source: `{dataset_lock}`
- Locked readme source: `{readme_locked}`

## Notes
- No standalone `LICENSE` or `LICENCE` file was found under the locked dataset root during bundle generation.
- The project-local license note for this dataset comes from `DATASET_LOCK.yaml`.
- Split files in this bundle list absolute paths to the locked dataset layout.
- For redistribution or external sharing decisions, re-check the original Zenodo record and the dataset's stated terms.

## Locked class names
{chr(10).join(f"- {name}" for name in class_names)}

## Locked split counts
- train entries: {count_manifest_entries(split_manifest_dir / "train.txt")}
- val entries: {count_manifest_entries(split_manifest_dir / "val.txt")}
- test entries: {count_manifest_entries(split_manifest_dir / "test.txt")}
"""
    copy_file(dataset_lock, license_dir)
    copy_file(readme_locked, license_dir)
    write_text(license_dir / "LICENSE_NOTES.md", note_text)

    split_summary_data = json.loads(read_text(split_summary))
    return {
        "dataset_id": dataset_id,
        "task": "object_detection",
        "source_root": str(DETECTION_ROOT),
        "split_files": [str(path.relative_to(out_root)) for path in copied_split_files],
        "class_names": class_names,
        "license_summary": license_summary,
        "split_counts": {
            "train": count_manifest_entries(split_manifest_dir / "train.txt"),
            "val": count_manifest_entries(split_manifest_dir / "val.txt"),
            "test": count_manifest_entries(split_manifest_dir / "test.txt"),
            "total_from_summary": split_summary_data.get("n_images_total"),
        },
        "bundle_files": relpath_list(dataset_dir),
    }


def build_segmentation_bundle(out_root: Path, generated_at: str, slurm_job_id: str | None) -> dict:
    dataset_id = "CropAndWeed_v1"
    dataset_dir = out_root / dataset_id
    split_dir = dataset_dir / "split_files"
    class_dir = dataset_dir / "class_names"
    license_dir = dataset_dir / "license_notes"

    split_manifest_dir = SEGMENTATION_ROOT / "splits" / "manifests"
    split_policy = SEGMENTATION_ROOT / "splits" / "split_policy.yaml"
    split_summary = SEGMENTATION_ROOT / "metadata" / "split_summary.json"
    dataset_lock = SEGMENTATION_ROOT / "DATASET_LOCK.yaml"
    readme_locked = SEGMENTATION_ROOT / "README_LOCKED.md"
    generic_yaml = SEGMENTATION_ROOT / "segmentation_dataset_generic.yaml"
    runtime_policy = SEGMENTATION_ROOT / "runtime_input_policy.yaml"
    raw_license = SEGMENTATION_ROOT / "raw" / "extracted_original" / "LICENCE.txt"

    copied_split_files = [
        copy_file(split_manifest_dir / "images_train.txt", split_dir),
        copy_file(split_manifest_dir / "images_val.txt", split_dir),
        copy_file(split_manifest_dir / "images_test.txt", split_dir),
        copy_file(split_manifest_dir / "masks_train.txt", split_dir),
        copy_file(split_manifest_dir / "masks_val.txt", split_dir),
        copy_file(split_manifest_dir / "masks_test.txt", split_dir),
        copy_file(split_manifest_dir / "pairs_train.csv", split_dir),
        copy_file(split_manifest_dir / "pairs_val.csv", split_dir),
        copy_file(split_manifest_dir / "pairs_test.csv", split_dir),
        copy_file(split_manifest_dir / "benchmark_order_test_fullimages.txt", split_dir),
        copy_file(split_policy, split_dir),
        copy_file(split_summary, split_dir),
    ]

    class_names = parse_indexed_mapping(generic_yaml, "classes")
    copy_file(generic_yaml, class_dir)
    copy_file(runtime_policy, class_dir)
    write_text(
        class_dir / "class_names.txt",
        "\n".join(f"{idx:02d} {name}" for idx, name in enumerate(class_names)) + "\n",
    )
    write_text(
        class_dir / "class_names.json",
        json.dumps(
            {
                "dataset_id": dataset_id,
                "task": "semantic_segmentation",
                "num_classes": len(class_names),
                "class_names": class_names,
            },
            indent=2,
        )
        + "\n",
    )

    lock_text = read_text(dataset_lock)
    license_summary = extract_scalar(lock_text, "license") or (
        "Non-commercial research/teaching/publication use only"
    )
    note_text = f"""# License Notes

Dataset: `{dataset_id}`
Task: semantic_segmentation
Generated: {generated_at}
Slurm job id: {slurm_job_id or "local"}

## Recorded license summary
- `{license_summary}`

## Source files copied into this bundle
- `{raw_license}`
- `{dataset_lock}`
- `{readme_locked}`

## Notes
- The original dataset `LICENCE.txt` is copied into this folder as the primary local license evidence.
- The project-local lock also records the usage note as non-commercial research, teaching, and publication use only.
- Split files in this bundle list absolute paths to the locked dataset layout.
- Please re-check the original dataset terms before redistribution of raw data or derived copies.

## Locked class names
{chr(10).join(f"- {name}" for name in class_names)}

## Locked split counts
- train image entries: {count_manifest_entries(split_manifest_dir / "images_train.txt")}
- val image entries: {count_manifest_entries(split_manifest_dir / "images_val.txt")}
- test image entries: {count_manifest_entries(split_manifest_dir / "images_test.txt")}
"""
    copy_file(dataset_lock, license_dir)
    copy_file(readme_locked, license_dir)
    copy_file(raw_license, license_dir)
    write_text(license_dir / "LICENSE_NOTES.md", note_text)

    split_summary_data = json.loads(read_text(split_summary))
    return {
        "dataset_id": dataset_id,
        "task": "semantic_segmentation",
        "source_root": str(SEGMENTATION_ROOT),
        "split_files": [str(path.relative_to(out_root)) for path in copied_split_files],
        "class_names": class_names,
        "license_summary": license_summary,
        "split_counts": {
            "train_images": count_manifest_entries(split_manifest_dir / "images_train.txt"),
            "val_images": count_manifest_entries(split_manifest_dir / "images_val.txt"),
            "test_images": count_manifest_entries(split_manifest_dir / "images_test.txt"),
            "train_pairs": count_manifest_entries(split_manifest_dir / "pairs_train.csv"),
            "val_pairs": count_manifest_entries(split_manifest_dir / "pairs_val.csv"),
            "test_pairs": count_manifest_entries(split_manifest_dir / "pairs_test.csv"),
            "total_from_summary": split_summary_data.get("n_images_total"),
            "sessions_total_from_summary": split_summary_data.get("n_sessions_total"),
        },
        "bundle_files": relpath_list(dataset_dir),
    }


def write_root_files(out_root: Path, generated_at: str, slurm_job_id: str | None, summary: dict) -> None:
    readme = f"""# Dataset Split Files + Class Names + License Notes

This bundle contains only the small, documentation-grade files needed for paper writing,
review support, and dataset bookkeeping for the two benchmark datasets:

- `CottonWeedDet12_v1`
- `CropAndWeed_v1`

It intentionally does not include raw images, masks, large archives, or model artifacts.

Generated: `{generated_at}`
Slurm job id: `{slurm_job_id or "local"}`
"""
    write_text(out_root / "README.md", readme)

    det = summary["datasets"]["CottonWeedDet12_v1"]
    seg = summary["datasets"]["CropAndWeed_v1"]
    summary_md = f"""# Summary

Generated: `{generated_at}`
Slurm job id: `{slurm_job_id or "local"}`

## CottonWeedDet12_v1
- task: object_detection
- class count: {len(det["class_names"])}
- class names: {", ".join(det["class_names"])}
- split counts: train={det["split_counts"]["train"]}, val={det["split_counts"]["val"]}, test={det["split_counts"]["test"]}, total={det["split_counts"]["total_from_summary"]}
- license summary: {det["license_summary"]}
- split files dir: `CottonWeedDet12_v1/split_files`
- class names dir: `CottonWeedDet12_v1/class_names`
- license notes dir: `CottonWeedDet12_v1/license_notes`

## CropAndWeed_v1
- task: semantic_segmentation
- class count: {len(seg["class_names"])}
- class names: {", ".join(seg["class_names"])}
- split counts: train_images={seg["split_counts"]["train_images"]}, val_images={seg["split_counts"]["val_images"]}, test_images={seg["split_counts"]["test_images"]}, total={seg["split_counts"]["total_from_summary"]}
- session total: {seg["split_counts"]["sessions_total_from_summary"]}
- license summary: {seg["license_summary"]}
- split files dir: `CropAndWeed_v1/split_files`
- class names dir: `CropAndWeed_v1/class_names`
- license notes dir: `CropAndWeed_v1/license_notes`

## Included top-level files
- `README.md`
- `SUMMARY.md`
- `summary.json`
- `bundle_manifest.json`
"""
    write_text(out_root / "SUMMARY.md", summary_md)
    write_text(out_root / "summary.json", json.dumps(summary, indent=2) + "\n")
    manifest = {
        "generated_at": generated_at,
        "slurm_job_id": slurm_job_id,
        "files": relpath_list(out_root),
    }
    write_text(out_root / "bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    datasets = [
        DatasetBundle("CottonWeedDet12_v1", "object_detection", DETECTION_ROOT),
        DatasetBundle("CropAndWeed_v1", "semantic_segmentation", SEGMENTATION_ROOT),
    ]
    for dataset in datasets:
        if not dataset.root.is_dir():
            raise SystemExit(f"Dataset root not found: {dataset.root}")

    summary = {
        "generated_at": generated_at,
        "slurm_job_id": slurm_job_id,
        "output_root": str(out_root),
        "datasets": {},
    }
    summary["datasets"]["CottonWeedDet12_v1"] = build_detection_bundle(out_root, generated_at, slurm_job_id)
    summary["datasets"]["CropAndWeed_v1"] = build_segmentation_bundle(out_root, generated_at, slurm_job_id)
    write_root_files(out_root, generated_at, slurm_job_id, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
