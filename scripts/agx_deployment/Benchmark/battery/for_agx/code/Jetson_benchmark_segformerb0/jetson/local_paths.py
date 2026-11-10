from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional, Sequence

DATASET_ARCHIVE_NAME = 'CropAndWeed_testset_for_jetson.zip'
DATASET_DIR_NAME = 'CropAndWeed_testset_for_jetson'


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    env_root = os.environ.get('BENCHMARK_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return package_root().parent.resolve()


def default_output_root(kind: str) -> Path:
    return workspace_root() / 'runs' / kind


def _path_from_env(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def is_inference_bundle_root(path: Path) -> bool:
    return path.is_dir() and (path / 'model').is_dir() and (path / 'dataset').is_dir()


def _bundle_root_from_hint(path: Path) -> Optional[Path]:
    if is_inference_bundle_root(path):
        return path
    if path.is_dir():
        direct = path / 'inference_bundle'
        if is_inference_bundle_root(direct):
            return direct
        nested = path / 'segformerb0' / 'inference_bundle'
        if is_inference_bundle_root(nested):
            return nested
    return None


def resolve_inference_bundle_root(explicit: Optional[str | Path] = None) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())

    for env_name in ('INFERENCE_BUNDLE_ROOT', 'SEGFORMERB0_ROOT'):
        env_path = _path_from_env(env_name)
        if env_path is not None:
            candidates.append(env_path)

    work_root = workspace_root()
    candidates.extend(
        [
            work_root / 'segformerb0',
            work_root / 'segformerb0' / 'inference_bundle',
            package_root(),
        ]
    )

    for candidate in _dedupe_paths(candidates):
        bundle_root = _bundle_root_from_hint(candidate)
        if bundle_root is not None:
            return bundle_root.resolve()
    return None


def is_portable_dataset_root(path: Path) -> bool:
    return path.is_dir() and ((path / 'data_test_only.yaml').exists() or (path / 'images' / 'test').is_dir())


def extract_portable_dataset(zip_path: str | Path) -> Path:
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    dataset_root = zip_path.parent / DATASET_DIR_NAME
    if is_portable_dataset_root(dataset_root):
        return dataset_root
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(zip_path.parent)
    if not is_portable_dataset_root(dataset_root):
        raise FileNotFoundError(f'Portable dataset root was not produced after extracting {zip_path}')
    return dataset_root


def resolve_dataset_root(explicit: Optional[str | Path] = None, bundle_root: Optional[Path] = None) -> Path:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser().resolve()
        if is_portable_dataset_root(explicit_path):
            return explicit_path
        if explicit_path.is_file() and explicit_path.suffix.lower() == '.zip':
            return extract_portable_dataset(explicit_path)
        hinted_bundle = _bundle_root_from_hint(explicit_path)
        if hinted_bundle is not None:
            bundle_root = hinted_bundle

    if bundle_root is None:
        bundle_root = resolve_inference_bundle_root(explicit)
    if bundle_root is None:
        raise FileNotFoundError('Could not resolve the local segformerb0/inference_bundle root.')

    extracted_root = bundle_root / 'dataset' / DATASET_DIR_NAME
    if is_portable_dataset_root(extracted_root):
        return extracted_root

    archive_path = bundle_root / 'dataset' / DATASET_ARCHIVE_NAME
    if archive_path.exists():
        return extract_portable_dataset(archive_path)

    raise FileNotFoundError(f'Could not resolve a portable dataset under {bundle_root / "dataset"}')


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _resolve_model_from_dir(model_dir: Path, names: Sequence[str]) -> Optional[Path]:
    return _first_existing([model_dir / name for name in names])


def resolve_model_path(explicit: Optional[str | Path] = None, bundle_root: Optional[Path] = None) -> Path:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.is_file():
            return explicit_path
        if explicit_path.is_dir():
            hinted_bundle = _bundle_root_from_hint(explicit_path)
            if hinted_bundle is not None:
                bundle_root = hinted_bundle
            else:
                direct = _resolve_model_from_dir(explicit_path, ('best_fp16.engine', 'best.engine', 'best.onnx', 'best.pt'))
                if direct is not None:
                    return direct

    if bundle_root is None:
        bundle_root = resolve_inference_bundle_root(explicit)
    if bundle_root is None:
        raise FileNotFoundError('Could not resolve the local segformerb0/inference_bundle root.')

    resolved = _resolve_model_from_dir(bundle_root / 'model', ('best_fp16.engine', 'best.engine', 'best.onnx', 'best.pt'))
    if resolved is None:
        raise FileNotFoundError(f'Could not find a benchmark model under {bundle_root / "model"}')
    return resolved


def resolve_accuracy_model_path(
    explicit: Optional[str | Path] = None,
    benchmark_model_path: Optional[str | Path] = None,
    bundle_root: Optional[Path] = None,
) -> Optional[Path]:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(explicit_path)

    benchmark_path = Path(benchmark_model_path).expanduser().resolve() if benchmark_model_path is not None else None
    if benchmark_path is not None:
        return benchmark_path
    return None


def resolve_data_yaml_path(dataset_root: Path, explicit: Optional[str | Path] = None, bundle_root: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            return explicit_path
        hinted_bundle = _bundle_root_from_hint(explicit_path)
        if hinted_bundle is None and explicit_path.parent.is_dir():
            raise FileNotFoundError(explicit_path)
        bundle_root = bundle_root or hinted_bundle

    if bundle_root is None:
        bundle_root = resolve_inference_bundle_root(dataset_root)

    candidates = [dataset_root / 'data_test_only.yaml']
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / 'metadata' / 'runtime_dataset_patched.yaml',
                bundle_root / 'metadata' / 'segmentation_dataset_generic.yaml',
            ]
        )
    return _first_existing(candidates)


def resolve_manifest_path(dataset_root: Path, explicit: Optional[str | Path] = None, bundle_root: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            return explicit_path
        hinted_bundle = _bundle_root_from_hint(explicit_path)
        if hinted_bundle is None and explicit_path.parent.is_dir():
            raise FileNotFoundError(explicit_path)
        bundle_root = bundle_root or hinted_bundle

    if bundle_root is None:
        bundle_root = resolve_inference_bundle_root(dataset_root)

    candidates = [
        dataset_root / 'metadata' / 'benchmark_order_test.txt',
        dataset_root / 'metadata' / 'test_manifest_relative.txt',
        dataset_root / 'metadata' / 'test_full_manifest.txt',
    ]
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / 'metadata' / 'benchmark_order_test.txt',
                bundle_root / 'metadata' / 'test_full_manifest.txt',
            ]
        )
    return _first_existing(candidates)


def resolve_output_root(explicit: Optional[str | Path], kind: str) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return default_output_root(kind).resolve()
