from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SYSTEM_TRT_PATH = Path("/usr/lib/python3.10/dist-packages")
if str(SYSTEM_TRT_PATH) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TRT_PATH))

import tensorrt as trt  # type: ignore

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.local_paths import resolve_dataset_root, resolve_manifest_path, resolve_inference_bundle_root
    from jetson.utils import dump_json, ensure_dir, list_images_from_manifest, sha256_file, utc_now_iso
else:  # pragma: no cover
    from .local_paths import resolve_dataset_root, resolve_manifest_path, resolve_inference_bundle_root
    from .utils import dump_json, ensure_dir, list_images_from_manifest, sha256_file, utc_now_iso


def _resolve_cuda_symbol(lib: ctypes.CDLL, *names: str):
    for name in names:
        if hasattr(lib, name):
            return getattr(lib, name)
    raise AttributeError(f"Could not resolve CUDA runtime symbol from candidates: {names}")


_LIBCUDART = ctypes.CDLL("libcudart.so")
_CUDA_MALLOC = _resolve_cuda_symbol(_LIBCUDART, "cudaMalloc", "cudaMalloc_v2")
_CUDA_FREE = _resolve_cuda_symbol(_LIBCUDART, "cudaFree", "cudaFree_v2")
_CUDA_MEMCPY = _resolve_cuda_symbol(_LIBCUDART, "cudaMemcpy", "cudaMemcpy_v2")

_CUDA_MALLOC.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_CUDA_MALLOC.restype = ctypes.c_int
_CUDA_FREE.argtypes = [ctypes.c_void_p]
_CUDA_FREE.restype = ctypes.c_int
_CUDA_MEMCPY.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_CUDA_MEMCPY.restype = ctypes.c_int

CUDA_MEMCPY_HOST_TO_DEVICE = 1


def _cuda_check(code: int, op: str) -> None:
    if code != 0:
        raise RuntimeError(f"{op} failed with CUDA runtime error code {code}")


def cuda_malloc(size: int) -> int:
    ptr = ctypes.c_void_p()
    _cuda_check(_CUDA_MALLOC(ctypes.byref(ptr), size), "cudaMalloc")
    return int(ptr.value)


def cuda_free(ptr: int) -> None:
    if ptr:
        _cuda_check(_CUDA_FREE(ctypes.c_void_p(ptr)), "cudaFree")


def cuda_memcpy_htod(dst_ptr: int, arr: np.ndarray) -> None:
    host_arr = np.ascontiguousarray(arr)
    src_ptr = host_arr.ctypes.data_as(ctypes.c_void_p)
    _cuda_check(
        _CUDA_MEMCPY(ctypes.c_void_p(dst_ptr), src_ptr, host_arr.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE),
        "cudaMemcpy HtoD",
    )


def resolve_image_list(dataset_root: Path, manifest: Optional[Path]) -> List[Path]:
    if manifest is not None and manifest.exists():
        return list_images_from_manifest(dataset_root, manifest)
    for candidate in [
        dataset_root / "images" / "calibration",
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
    ]:
        if candidate.is_dir():
            return sorted([p for p in candidate.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    raise RuntimeError(f"Could not find calibration images under {dataset_root}")


def letterbox_rgb(image_path: Path, imgsz: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    src_w, src_h = image.size
    scale = min(imgsz / max(src_w, 1), imgsz / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), color=(114, 114, 114))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr


def analyze_onnx_input(onnx_path: Path) -> Tuple[str, Sequence[int]]:
    import onnx

    model = onnx.load(str(onnx_path))
    input_tensor = model.graph.input[0]
    dims = []
    for d in input_tensor.type.tensor_type.shape.dim:
        dims.append(int(d.dim_value) if d.dim_value else -1)
    return input_tensor.name, dims


class ImageBatchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: Sequence[Path],
        cache_path: Path,
        imgsz: int,
        batch_size: int,
        channels: int,
        max_images: int,
    ):
        super().__init__()
        self.image_paths = list(image_paths)[: max(1, int(max_images))]
        self.cache_path = cache_path
        self.imgsz = int(imgsz)
        self.batch_size = int(batch_size)
        self.channels = int(channels)
        self.index = 0
        self._alloc_size = self.batch_size * self.channels * self.imgsz * self.imgsz * np.dtype(np.float32).itemsize
        self.device_input = cuda_malloc(self._alloc_size)

    def __del__(self):  # pragma: no cover
        try:
            cuda_free(self.device_input)
        except Exception:
            pass

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: Sequence[str]):  # type: ignore[override]
        if self.index >= len(self.image_paths):
            return None
        batch = np.zeros((self.batch_size, self.channels, self.imgsz, self.imgsz), dtype=np.float32)
        for batch_idx in range(self.batch_size):
            if self.index >= len(self.image_paths):
                break
            batch[batch_idx] = letterbox_rgb(self.image_paths[self.index], self.imgsz)
            self.index += 1
        cuda_memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):  # type: ignore[override]
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):  # type: ignore[override]
        self.cache_path.write_bytes(cache)


def build_cache_and_engine(
    *,
    onnx_path: Path,
    cache_path: Path,
    engine_path: Path,
    dataset_root: Path,
    manifest_path: Optional[Path],
    imgsz: int,
    batch_size: int,
    workspace_mb: int,
    max_images: int,
) -> dict:
    input_name, dims = analyze_onnx_input(onnx_path)
    channels = dims[1] if len(dims) > 1 and dims[1] > 0 else 3
    image_paths = resolve_image_list(dataset_root, manifest_path)
    if not image_paths:
        raise RuntimeError("No calibration images found")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        messages = []
        for idx in range(parser.num_errors):
            messages.append(str(parser.get_error(idx)))
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(messages))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20))
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = ImageBatchEntropyCalibrator(
        image_paths=image_paths,
        cache_path=cache_path,
        imgsz=imgsz,
        batch_size=batch_size,
        channels=channels,
        max_images=max_images,
    )
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = tuple(int(v) if int(v) > 0 else -1 for v in input_tensor.shape)
    if any(v <= 0 for v in shape):
        fixed_shape = (batch_size, channels, imgsz, imgsz)
        profile.set_shape(input_name, fixed_shape, fixed_shape, fixed_shape)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None while creating INT8 engine")

    ensure_dir(engine_path.parent)
    engine_path.write_bytes(bytes(serialized))
    if not cache_path.exists():
        raise RuntimeError(f"Calibration cache was not written: {cache_path}")

    return {
        "timestamp_utc": utc_now_iso(),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "cache_path": str(cache_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "workspace_mb": workspace_mb,
        "max_images": max_images,
        "input_name": input_name,
        "input_dims": list(dims),
        "calibration_image_count": min(len(image_paths), max_images),
        "sha256_engine": sha256_file(engine_path),
        "sha256_cache": sha256_file(cache_path),
        "sha256_onnx": sha256_file(onnx_path),
        "tensorrt_version": trt.__version__,
        "tensorrt_python_origin": getattr(sys.modules.get("tensorrt"), "__file__", None),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build an INT8 calibration cache and INT8 TensorRT engine using system TensorRT Python.")
    p.add_argument("--onnx", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--cache-path", default=None)
    p.add_argument("--engine-path", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    out_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    bundle_root = resolve_inference_bundle_root(args.dataset_root or args.onnx)
    dataset_root = resolve_dataset_root(args.dataset_root, bundle_root=bundle_root)
    manifest_path = resolve_manifest_path(dataset_root, args.manifest, bundle_root=bundle_root)

    cache_path = Path(args.cache_path).expanduser().resolve() if args.cache_path else out_dir / f"{onnx_path.stem}_int8_calibration.cache"
    engine_path = Path(args.engine_path).expanduser().resolve() if args.engine_path else out_dir / f"{onnx_path.stem}_int8.engine"

    meta = build_cache_and_engine(
        onnx_path=onnx_path,
        cache_path=cache_path,
        engine_path=engine_path,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        imgsz=int(args.imgsz),
        batch_size=int(args.batch),
        workspace_mb=int(args.workspace_mb),
        max_images=int(args.max_images),
    )
    meta_path = out_dir / f"{onnx_path.stem}_int8_cache_build_meta.json"
    dump_json(meta, meta_path)
    print(meta_path)


if __name__ == "__main__":
    main()
