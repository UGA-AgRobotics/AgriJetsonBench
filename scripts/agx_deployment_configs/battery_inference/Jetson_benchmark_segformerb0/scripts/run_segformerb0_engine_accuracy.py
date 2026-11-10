#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PKG_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PKG_ROOT.parent
CPP_SOURCE = PKG_ROOT / "jetson" / "eval_segmentation_engine.cpp"
CPP_BINARY = PKG_ROOT / "jetson" / "eval_segmentation_engine"


if __package__ in {None, ""}:
    sys.path.append(str(PKG_ROOT))
    from jetson.local_paths import resolve_dataset_root, resolve_inference_bundle_root
    from jetson.utils import dump_json, ensure_dir, run_command
else:  # pragma: no cover
    from ..jetson.local_paths import resolve_dataset_root, resolve_inference_bundle_root
    from ..jetson.utils import dump_json, ensure_dir, run_command


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_precision_label(path: Path) -> Optional[str]:
    name = path.name.lower()
    for label in ("int8", "fp16", "fp32"):
        if label in name:
            return label
    return None


def _find_cuda_paths() -> tuple[Path, Path]:
    candidates = [
        (Path("/usr/local/cuda/targets/aarch64-linux/include"), Path("/usr/local/cuda/lib64")),
        (Path("/usr/local/cuda-12.6/targets/aarch64-linux/include"), Path("/usr/local/cuda-12.6/lib64")),
        (Path("/usr/local/cuda-12/targets/aarch64-linux/include"), Path("/usr/local/cuda-12/lib64")),
    ]
    for include_dir, lib_dir in candidates:
        if include_dir.is_dir() and lib_dir.is_dir():
            return include_dir.resolve(), lib_dir.resolve()
    raise FileNotFoundError("CUDA include/lib paths not found under /usr/local/cuda, /usr/local/cuda-12, or /usr/local/cuda-12.6")


def _local_tensorrt_paths() -> tuple[Optional[Path], Optional[Path]]:
    root = Path(os.environ.get("LOCAL_TENSORRT_ROOT", str(BENCHMARK_ROOT / ".vendor" / "tensorrt-${PRIVATE_LAN_IP}"))).expanduser().resolve()
    include_dir = Path(os.environ.get("LOCAL_TENSORRT_INCLUDE_DIR", str(root / "usr" / "include" / "aarch64-linux-gnu"))).expanduser().resolve()
    lib_dir = Path(os.environ.get("LOCAL_TENSORRT_LIB_DIR", str(root / "usr" / "lib" / "aarch64-linux-gnu"))).expanduser().resolve()
    return (include_dir if include_dir.is_dir() else None, lib_dir if lib_dir.is_dir() else None)


def _pkg_config_flags(package: str, mode: str) -> List[str]:
    cp = run_command(["pkg-config", mode, package], check=True, capture_output=True, text=True)
    text = (cp.stdout or "").strip()
    return shlex.split(text) if text else []


def _opencv_include_fallback_flags() -> List[str]:
    flags: List[str] = []
    for candidate in (Path("/usr/include/opencv4"), Path("/usr/local/include/opencv4")):
        if candidate.is_dir():
            flags.append(f"-I{candidate.resolve()}")
    return flags


def _resolve_existing_layout_path(path: Path) -> Path:
    candidates = [path.expanduser().resolve()]
    raw = str(candidates[0])
    replacements = [
        ("/inference_bundle/model/", "/model/"),
        ("/inference_bundle/dataset/", "/dataset/"),
        ("/inference_bundle/metadata/", "/metadata/"),
        ("/inference_bundle/", "/"),
    ]
    for old, new in replacements:
        if old in raw:
            candidates.append(Path(raw.replace(old, new)).expanduser().resolve())

    if path.name:
        candidates.extend(
            [
                (BENCHMARK_ROOT / "segformerb0" / "model" / path.name).resolve(),
                (BENCHMARK_ROOT / "segformerb0" / "dataset" / path.name).resolve(),
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_engine_accuracy_binary() -> Path:
    if CPP_BINARY.exists() and CPP_BINARY.stat().st_mtime >= CPP_SOURCE.stat().st_mtime:
        return CPP_BINARY

    cuda_include_dir, cuda_lib_dir = _find_cuda_paths()
    trt_include_dir, trt_lib_dir = _local_tensorrt_paths()
    opencv_cflags = _pkg_config_flags("opencv4", "--cflags")
    opencv_libs = _pkg_config_flags("opencv4", "--libs")

    cmd: List[str] = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
    ]
    if trt_include_dir is not None:
        cmd.append(f"-I{trt_include_dir}")
    cmd.append(f"-I{cuda_include_dir}")
    cmd.extend(_opencv_include_fallback_flags())
    cmd.extend(opencv_cflags)
    cmd.extend(
        [
            str(CPP_SOURCE),
            "-o",
            str(CPP_BINARY),
            f"-L{cuda_lib_dir}",
            f"-Wl,-rpath,{cuda_lib_dir}",
        ]
    )
    if trt_lib_dir is not None:
        cmd.extend([f"-L{trt_lib_dir}", f"-Wl,-rpath,{trt_lib_dir}"])
    cmd.extend(opencv_libs)
    cmd.extend(["-lnvinfer", "-lcudart"])
    run_command(cmd, check=True, capture_output=True, text=True)
    return CPP_BINARY


def resolve_segmentation_manifest(dataset_root: Path, split_name: str, bundle_root: Optional[Path]) -> Path:
    split = str(split_name or "test").strip() or "test"
    candidates = [
        dataset_root / "manifests" / f"{split}.csv",
        dataset_root / "manifests" / "test.csv",
        dataset_root / "metadata" / f"{split}_manifest.csv",
        dataset_root / "metadata" / "test_manifest.csv",
    ]
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / "metadata" / f"{split}_manifest.csv",
                bundle_root / "metadata" / "test_manifest.csv",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve a segmentation manifest for split '{split}' under {dataset_root}")


def _format_results(raw: Dict[str, Any], output_dir: Path, imgsz: int) -> Dict[str, Any]:
    metrics = dict(raw.get("metrics", {}) or {})
    report_json = output_dir / "report.json"
    metrics_json = output_dir / "metrics.json"
    dump_json(raw, report_json)
    dump_json(metrics, metrics_json)
    model_path = Path(str(raw.get("engine_path", ""))).expanduser().resolve()
    return {
        "backend": "tensorrt_engine",
        "backend_detail": "segformerb0_trt_engine",
        "metrics/mIoU": metrics.get("miou"),
        "metrics/mDice": metrics.get("mdice"),
        "metrics/foreground_mIoU": metrics.get("foreground_miou"),
        "metrics/foreground_mDice": metrics.get("foreground_mdice"),
        "metrics/pixel_accuracy": metrics.get("pixel_accuracy"),
        "per_class": metrics.get("per_class"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "precision_label": infer_precision_label(model_path),
        "imgsz": int(imgsz),
        "batch": 1,
        "engine_path": raw.get("engine_path"),
        "manifest_path": raw.get("manifest_path"),
        "dataset_root": raw.get("dataset_root"),
        "output_dir": str(output_dir),
        "report_json": str(report_json),
        "metrics_json": str(metrics_json),
        "input_name": raw.get("input_name"),
        "output_name": raw.get("output_name"),
        "input_shape": raw.get("input_shape"),
        "output_shape": raw.get("output_shape"),
        "processed_samples": raw.get("processed_samples"),
        "preprocess": raw.get("preprocess"),
    }


def run_engine_accuracy(
    *,
    engine_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    output_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    binary = ensure_engine_accuracy_binary()
    engine_path = _resolve_existing_layout_path(engine_path)
    dataset_root = _resolve_existing_layout_path(dataset_root)
    bundle_root = resolve_inference_bundle_root(dataset_root)
    resolved_dataset_root = resolve_dataset_root(dataset_root, bundle_root=bundle_root) if dataset_root.exists() else dataset_root
    resolved_manifest = manifest_path.expanduser().resolve() if manifest_path else resolve_segmentation_manifest(resolved_dataset_root, split_name, bundle_root)

    ensure_dir(output_dir)
    raw_json = output_dir / "engine_metrics_raw.json"
    env = dict(os.environ)
    _, cuda_lib_dir = _find_cuda_paths()
    _, trt_lib_dir = _local_tensorrt_paths()
    if trt_lib_dir is not None:
        env["LD_LIBRARY_PATH"] = f"{trt_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    cmd = [
        str(binary),
        "--engine",
        str(engine_path),
        "--dataset-root",
        str(resolved_dataset_root),
        "--manifest",
        str(resolved_manifest),
        "--json-out",
        str(raw_json),
        "--imgsz",
        str(int(imgsz)),
    ]
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    combined_output = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
    (output_dir / "engine_accuracy.log").write_text(combined_output + "\n", encoding="utf-8")
    if cp.returncode != 0:
        raise RuntimeError(
            "SegFormer-B0 engine accuracy validation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{combined_output[-2000:]}"
        )
    if not raw_json.exists():
        raise FileNotFoundError(raw_json)
    raw = json.loads(raw_json.read_text(encoding="utf-8"))
    return _format_results(raw, output_dir, imgsz)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SegFormer-B0 precision-specific accuracy validation using a TensorRT engine.")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out-json", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results = run_engine_accuracy(
        engine_path=Path(args.engine).expanduser().resolve(),
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        split_name=str(args.split_name or "test"),
        imgsz=int(args.imgsz),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve() if args.manifest else None,
    )
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else Path(args.output_dir).expanduser().resolve() / "val_metrics.json"
    dump_json(results, out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
