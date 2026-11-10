from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from jetson.local_paths import extract_portable_dataset, resolve_inference_bundle_root


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare the user\'s inference_bundle tree for Jetson benchmarking.')
    ap.add_argument('--experiment-root', required=True, help='Root that contains inference_bundle/ according to the attached tree.txt')
    ap.add_argument('--out-root', required=True, help='Workspace where dataset/model/metadata will be prepared')
    args = ap.parse_args()

    exp_root = Path(args.experiment_root).expanduser().resolve()
    out_root = ensure_dir(Path(args.out_root).expanduser().resolve())
    bundle_root = resolve_inference_bundle_root(exp_root)
    if bundle_root is None:
        raise FileNotFoundError(f'Could not resolve an inference_bundle under {exp_root}')

    onnx_src = bundle_root / 'model' / 'best.onnx'
    pt_src = first_existing([bundle_root / 'model' / 'best.pt'])
    zip_src = bundle_root / 'dataset' / 'CropAndWeed_testset_for_jetson.zip'
    runtime_yaml_src = first_existing(
        [
            bundle_root / 'metadata' / 'segmentation_dataset_generic.yaml',
        ]
    )
    extracted_dataset_root = extract_portable_dataset(zip_src)
    portable_yaml_src = extracted_dataset_root / 'data_test_only.yaml'
    portable_manifest_src = extracted_dataset_root / 'metadata' / 'benchmark_order_test.txt'
    portable_seg_manifest_src = extracted_dataset_root / 'manifests' / 'test.csv'

    for p in [onnx_src, zip_src, portable_yaml_src, portable_manifest_src]:
        if not p.exists():
            raise FileNotFoundError(p)

    model_dir = ensure_dir(out_root / 'model')
    metadata_dir = ensure_dir(out_root / 'metadata')
    dataset_parent = ensure_dir(out_root / 'dataset')
    dataset_root = dataset_parent / extracted_dataset_root.name

    shutil.copy2(onnx_src, model_dir / onnx_src.name)
    if pt_src and pt_src.exists():
        shutil.copy2(pt_src, model_dir / pt_src.name)
    shutil.copy2(zip_src, dataset_parent / zip_src.name)
    shutil.copytree(extracted_dataset_root, dataset_root, dirs_exist_ok=True)
    if runtime_yaml_src and runtime_yaml_src.exists():
        shutil.copy2(runtime_yaml_src, metadata_dir / runtime_yaml_src.name)
    shutil.copy2(portable_manifest_src, metadata_dir / 'benchmark_order_test.txt')
    shutil.copy2(portable_manifest_src, metadata_dir / 'test_full_manifest.txt')
    if portable_seg_manifest_src.exists():
        shutil.copy2(portable_seg_manifest_src, metadata_dir / 'test_manifest.csv')

    yaml_dst = metadata_dir / 'runtime_dataset_patched.yaml'
    cfg = yaml.safe_load(portable_yaml_src.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        cfg = {}
    cfg['path'] = str(dataset_root)
    cfg.setdefault('test', 'images/test')
    yaml_dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')

    info: Dict[str, str] = {
        'experiment_root': str(exp_root),
        'bundle_root': str(bundle_root),
        'dataset_root': str(dataset_root),
        'onnx_model': str(model_dir / onnx_src.name),
        'pt_model': str(model_dir / pt_src.name) if pt_src and pt_src.exists() else '',
        'runtime_yaml_original': str(metadata_dir / runtime_yaml_src.name) if runtime_yaml_src and runtime_yaml_src.exists() else '',
        'runtime_yaml_patched': str(yaml_dst),
        'test_manifest': str(metadata_dir / 'test_full_manifest.txt'),
        'test_manifest_csv': str(metadata_dir / 'test_manifest.csv') if portable_seg_manifest_src.exists() else '',
        'portable_yaml': str(dataset_root / 'data_test_only.yaml'),
        'portable_manifest': str(dataset_root / 'metadata' / 'benchmark_order_test.txt'),
    }
    (out_root / 'PREPARED_BUNDLE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
