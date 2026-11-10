from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ns() -> int:
    return time.monotonic_ns()


def wallclock_ns() -> int:
    return time.time_ns()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    if is_dataclass(obj):
        obj = asdict(obj)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def dump_yaml(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    if is_dataclass(obj):
        obj = asdict(obj)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding='utf-8')


def load_yaml(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def run_command(
    cmd: List[str],
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str | Path] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _relative_if_possible(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except Exception:
        return str(target.resolve())


def discover_images(dataset_root: Path) -> List[Path]:
    dataset_root = dataset_root.resolve()
    candidate_dirs = [
        dataset_root / 'images' / 'test',
        dataset_root / 'test' / 'images',
        dataset_root / 'test',
        dataset_root / 'images',
        dataset_root,
    ]
    images: List[Path] = []
    seen = set()
    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for path in sorted(cdir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                real = str(path.resolve())
                if real not in seen:
                    seen.add(real)
                    images.append(path.resolve())
        if images:
            break
    return images


def default_test_manifest(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    candidates = [
        dataset_root / 'metadata' / 'benchmark_order_test.txt',
        dataset_root / 'metadata' / 'test_full_manifest.txt',
        dataset_root / 'benchmark_order_test.txt',
        dataset_root / 'test_full_manifest.txt',
        dataset_root / 'metadata' / 'test_manifest.txt',
        dataset_root / 'test_manifest.txt',
    ]
    for p in candidates:
        if p.exists():
            return p
    images = discover_images(dataset_root)
    if not images:
        raise FileNotFoundError(f'Could not find a manifest or discover test images under {dataset_root}')
    out = ensure_dir(dataset_root / 'metadata') / 'benchmark_order_test.txt'
    lines = [_relative_if_possible(dataset_root, p) for p in images]
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def _recover_manifest_path(image_index: Dict[str, List[Path]], original_path: Path) -> Optional[Path]:
    candidates = image_index.get(original_path.name, [])
    if not candidates and original_path.suffix == '':
        candidates = image_index.get(original_path.stem, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def list_images_from_manifest(dataset_root: str | Path, manifest: str | Path) -> List[Path]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    if manifest.suffix.lower() == '.csv':
        images: List[Path] = []
        missing: List[str] = []
        image_index: Optional[Dict[str, List[Path]]] = None
        with manifest.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_path = str((row or {}).get('image_path', '')).strip()
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = dataset_root / path
                path = path.resolve()
                if path.exists():
                    images.append(path)
                    continue
                if image_index is None:
                    image_index = {}
                    for image_path in discover_images(dataset_root):
                        image_index.setdefault(image_path.name, []).append(image_path)
                        image_index.setdefault(image_path.stem, []).append(image_path)
                recovered = _recover_manifest_path(image_index, path)
                if recovered is not None:
                    images.append(recovered)
                else:
                    missing.append(str(path))
        if missing:
            preview = '\n'.join(missing[:10])
            raise FileNotFoundError(f'{len(missing)} images listed in manifest were missing. First entries:\n{preview}')
        return images

    images: List[Path] = []
    missing: List[str] = []
    image_index: Optional[Dict[str, List[Path]]] = None
    for raw in manifest.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if path.exists():
            images.append(path)
        else:
            if image_index is None:
                image_index = {}
                for image_path in discover_images(dataset_root):
                    image_index.setdefault(image_path.name, []).append(image_path)
                    image_index.setdefault(image_path.stem, []).append(image_path)
            recovered = _recover_manifest_path(image_index, path)
            if recovered is not None:
                images.append(recovered)
            else:
                missing.append(str(path))

    if missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError(f'{len(missing)} images listed in manifest were missing. First entries:\n{preview}')
    return images


def overlay_nested(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = overlay_nested(out[key], value)
        else:
            out[key] = value
    return out


def bootstrap_ci(values: List[float], n_boot: int = 5000, seed: int = 12345, alpha: float = 0.05) -> Optional[Dict[str, float]]:
    import numpy as np

    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        val = float(arr[0])
        return {'center': val, 'ci_low': val, 'ci_high': val, 'n': 1}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = float(sample.mean())
    return {
        'center': float(arr.mean()),
        'ci_low': float(np.percentile(stats, 100 * (alpha / 2))),
        'ci_high': float(np.percentile(stats, 100 * (1 - alpha / 2))),
        'n': int(arr.size),
    }
