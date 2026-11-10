#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def infer_precision_label(path: Path) -> Optional[str]:
    name = path.name.lower()
    for label in ('int8', 'fp16', 'fp32'):
        if label in name:
            return label
    return None


def _ensure_pillow_compat() -> None:
    try:
        from PIL import Image
    except Exception:
        return

    if not hasattr(Image, 'Resampling'):
        Image.Resampling = SimpleNamespace(  # type: ignore[attr-defined]
            NEAREST=Image.NEAREST,
            BILINEAR=Image.BILINEAR,
            BICUBIC=Image.BICUBIC,
            LANCZOS=getattr(Image, 'LANCZOS', Image.BICUBIC),
        )
    if not hasattr(Image, 'Transpose'):
        Image.Transpose = SimpleNamespace(  # type: ignore[attr-defined]
            FLIP_LEFT_RIGHT=Image.FLIP_LEFT_RIGHT,
            FLIP_TOP_BOTTOM=Image.FLIP_TOP_BOTTOM,
        )


def _import_segformerb0_lib(source_root: Path):
    source_root = source_root.resolve()
    scripts_dir = source_root / 'scripts'
    if not (scripts_dir / 'segformer_b0_lib.py').exists():
        raise FileNotFoundError(f'Could not find segformer_b0_lib.py under {scripts_dir}')
    _ensure_pillow_compat()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module('segformer_b0_lib')


def _resolve_manifest_path(dataset_root: Path, split_name: str, explicit: Optional[Path], runtime_dir: Path) -> Path:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if explicit.exists():
            return explicit
        raise FileNotFoundError(explicit)

    split = str(split_name or 'test').strip() or 'test'
    candidates = [
        dataset_root / 'manifests' / f'{split}.csv',
        dataset_root / 'manifests' / 'test.csv',
        dataset_root / 'metadata' / f'{split}_manifest.csv',
        dataset_root / 'metadata' / 'test_manifest.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    image_dir = dataset_root / 'images' / split
    mask_dir = dataset_root / 'masks' / split
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f'Could not resolve a segmentation manifest for split "{split}" under {dataset_root}. '
            f'Expected manifests/{split}.csv or images/{split} + masks/{split}.'
        )

    images = {p.stem: p for p in sorted(image_dir.iterdir()) if p.is_file()}
    masks = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.is_file()}
    shared = sorted(set(images).intersection(masks))
    if not shared:
        raise RuntimeError(f'No paired image/mask files were found under {image_dir} and {mask_dir}.')

    manifest_path = ensure_dir(runtime_dir) / f'{split}.csv'
    with manifest_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['split', 'image_path', 'mask_path', 'image_width', 'image_height'])
        writer.writeheader()
        for stem in shared:
            writer.writerow(
                {
                    'split': split,
                    'image_path': str(images[stem].resolve()),
                    'mask_path': str(masks[stem].resolve()),
                    'image_width': '',
                    'image_height': '',
                }
            )
    return manifest_path


def _materialize_absolute_manifest(dataset_root: Path, manifest_path: Path, runtime_dir: Path) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    runtime_manifest = ensure_dir(runtime_dir) / manifest_path.name

    with manifest_path.open('r', encoding='utf-8', newline='') as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = list(reader.fieldnames or ['split', 'image_path', 'mask_path', 'image_width', 'image_height'])
        if 'image_path' not in fieldnames or 'mask_path' not in fieldnames:
            raise RuntimeError(
                f'Segmentation manifest must contain image_path and mask_path columns: {manifest_path}'
            )

        rows: List[Dict[str, str]] = []
        missing: List[str] = []
        for row in reader:
            item = dict(row or {})
            for key in ('image_path', 'mask_path'):
                raw_value = str(item.get(key, '')).strip()
                if not raw_value:
                    continue
                candidate = Path(raw_value).expanduser()
                if not candidate.is_absolute():
                    candidate = (dataset_root / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if not candidate.exists():
                    missing.append(f'{key}: {candidate}')
                item[key] = str(candidate)
            rows.append(item)

    if missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} manifest paths were missing after absolutizing against {dataset_root}. '
            f'First entries:\n{preview}'
        )

    with runtime_manifest.open('w', encoding='utf-8', newline='') as dst_handle:
        writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return runtime_manifest


def _extract_metrics(
    report: Dict[str, Any],
    *,
    model_path: Path,
    runtime_manifest: Path,
    imgsz: int,
    batch: int,
) -> Dict[str, Any]:
    metrics = dict(report.get('metrics', {}) or {})
    return {
        'backend': 'pytorch_checkpoint',
        'backend_detail': 'segformerb0_repo',
        'metrics/mIoU': metrics.get('miou'),
        'metrics/mDice': metrics.get('mdice'),
        'metrics/foreground_mIoU': metrics.get('foreground_miou'),
        'metrics/foreground_mDice': metrics.get('foreground_mdice'),
        'metrics/pixel_accuracy': metrics.get('pixel_accuracy'),
        'per_class': metrics.get('per_class'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'model_path': str(model_path),
        'model_sha256': sha256_file(model_path),
        'precision_label': infer_precision_label(model_path),
        'imgsz': int(imgsz),
        'batch': int(batch),
        'manifest_path': str(runtime_manifest),
    }


def run_accuracy(
    *,
    source_root: Path,
    config_path: Path,
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    workers: int,
    output_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    segformerb0_lib = _import_segformerb0_lib(source_root)
    runtime_manifest = _resolve_manifest_path(dataset_root, split_name, manifest_path, output_dir / 'runtime_manifests')
    runtime_manifest = _materialize_absolute_manifest(dataset_root, runtime_manifest, output_dir / 'runtime_manifests')

    cfg = load_yaml(config_path)
    cfg.setdefault('dataset', {})
    cfg.setdefault('training', {})
    cfg.setdefault('distributed', {})
    cfg.setdefault('runtime', {})

    cfg['dataset']['dataset_root'] = str(dataset_root)
    cfg['training']['input_size'] = int(imgsz)
    cfg['training']['eval_batch'] = max(1, int(batch))
    cfg['training']['workers'] = max(0, int(workers))
    cfg['distributed']['enabled'] = False
    cfg['runtime']['root'] = str(output_dir / 'runtime')

    report = segformerb0_lib.evaluate_checkpoint_on_manifest(
        cfg,
        checkpoint_path=model_path,
        manifest_path=runtime_manifest,
        output_dir=output_dir,
        split_name=split_name,
    )

    results = _extract_metrics(
        report,
        model_path=model_path,
        runtime_manifest=runtime_manifest,
        imgsz=imgsz,
        batch=batch,
    )
    results.update(
        {
            'config_path': str(config_path),
            'dataset_root': str(dataset_root),
            'output_dir': str(output_dir),
            'report_json': str(output_dir / 'report.json'),
            'metrics_json': str(output_dir / 'metrics.json'),
        }
    )
    dump_json(results, output_dir / 'val_metrics.json')
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run SegFormer-B0 semantic-segmentation accuracy validation using either a checkpoint or a TensorRT engine.')
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--checkpoint', default=None, help='Deprecated alias for --model.')
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--split-name', default='test')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--out-json', default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_arg = args.model or args.checkpoint
    if not model_arg:
        raise SystemExit('Either --model or --checkpoint must be provided.')

    model_path = Path(model_arg).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    if model_path.suffix.lower() == '.engine':
        engine_helper = Path(__file__).resolve().with_name('run_segformerb0_engine_accuracy.py')
        cmd = [
            str(Path(sys.executable).resolve()),
            str(engine_helper),
            '--engine',
            str(model_path),
            '--dataset-root',
            str(dataset_root),
            '--split-name',
            str(args.split_name or 'test'),
            '--imgsz',
            str(int(args.imgsz)),
            '--output-dir',
            str(output_dir),
        ]
        if manifest_path is not None:
            cmd.extend(['--manifest', str(manifest_path)])
        if args.out_json:
            cmd.extend(['--out-json', str(Path(args.out_json).expanduser().resolve())])
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = (cp.stdout or '').strip()
        if stdout:
            print(stdout)
        return

    results = run_accuracy(
        source_root=source_root,
        config_path=config_path,
        model_path=model_path,
        dataset_root=dataset_root,
        split_name=str(args.split_name or 'test'),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        output_dir=output_dir,
        manifest_path=manifest_path,
    )
    if args.out_json:
        dump_json(results, Path(args.out_json).expanduser().resolve())
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
