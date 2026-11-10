#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from numbers import Number


def prefer_active_env_site_packages() -> None:
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or sys.prefix
    major = sys.version_info.major
    minor = sys.version_info.minor
    site_path = Path(prefix) / "lib" / f"python{major}.{minor}" / "site-packages"
    if site_path.exists():
        site_str = str(site_path.resolve())
        try:
            sys.path.remove(site_str)
        except ValueError:
            pass
        sys.path.insert(0, site_str)

    for name in list(sys.modules):
        if name == "google" or name.startswith("google."):
            sys.modules.pop(name, None)


prefer_active_env_site_packages()

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required") from exc

try:
    import onnx
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "onnx is not installed in the active environment. Install requirements_jetson.txt first."
    ) from exc


VALUE_KEYS = (
    "params",
    "macs",
    "gmacs",
    "gflops_if_1mac_eq_2flops",
)


def find_pkg_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "jetson").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find package root above {start}. Put this file inside the benchmark repo or pass --pkg-root."
    )


def package_root_from_args(user_arg: str | None) -> Path:
    if user_arg:
        return find_pkg_root(Path(user_arg).expanduser())
    return find_pkg_root(Path(__file__).resolve().parent)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _load_local_paths_module(pkg_root: Path):
    module_path = pkg_root / "jetson" / "local_paths.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_benchmark_local_paths", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_bundle_root_via_local_paths(pkg_root: Path) -> Path | None:
    module = _load_local_paths_module(pkg_root)
    if module is None:
        return None
    resolver = getattr(module, "resolve_inference_bundle_root", None)
    if not callable(resolver):
        return None
    try:
        resolved = resolver()
    except TypeError:
        resolved = resolver(None)
    if not resolved:
        return None
    return Path(resolved).expanduser().resolve()


def resolve_default_onnx(pkg_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    bundle_root = _resolve_bundle_root_via_local_paths(pkg_root)
    env_candidates = []
    for name in ("ONNX_MODEL_PATH", "MODEL_PATH", "MODEL_DIR", "INFERENCE_BUNDLE_ROOT"):
        value = os.environ.get(name)
        if not value:
            continue
        p = Path(value).expanduser().resolve()
        if name == "MODEL_DIR":
            env_candidates.append(p / "best.onnx")
        elif name == "INFERENCE_BUNDLE_ROOT":
            env_candidates.append(p / "model" / "best.onnx")
        elif p.is_dir():
            env_candidates.append(p / "best.onnx")
        else:
            env_candidates.append(p)

    bundle_candidates: list[Path] = []
    if bundle_root is not None:
        bundle_candidates.extend(
            [
                bundle_root / "model" / "best.onnx",
                bundle_root / "model" / "best_trt_deploy.onnx",
            ]
        )

    candidates = env_candidates + bundle_candidates + [
        pkg_root / "inference_bundle" / "model" / "best.onnx",
        pkg_root / "model" / "best.onnx",
        pkg_root.parent / pkg_root.name / "inference_bundle" / "model" / "best.onnx",
    ]
    resolved = first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Could not resolve best.onnx automatically. Pass --onnx explicitly."
        )
    return resolved


def load_onnx_input_names(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path))
    return [value.name for value in model.graph.input]


def patch_onnx_if_needed(pkg_root: Path, onnx_path: Path, imgsz: int, out_dir: Path, force_patch: bool = False) -> Path:
    input_names = load_onnx_input_names(onnx_path)
    needs_patch = force_patch or ("orig_target_sizes" in input_names)
    if not needs_patch:
        return onnx_path

    patch_tool = pkg_root / "scripts" / "prepare_tensorrt_build_onnx.py"
    if not patch_tool.exists():
        raise FileNotFoundError(f"Patch tool not found: {patch_tool}")

    patched_path = out_dir / f"{onnx_path.stem}_profile_{imgsz}.onnx"
    cmd = [
        sys.executable,
        str(patch_tool),
        "--onnx",
        str(onnx_path),
        "--out",
        str(patched_path),
        "--imgsz",
        str(int(imgsz)),
    ]
    subprocess.run(cmd, check=True)
    return patched_path


def resolve_primary_input_name_and_shape(onnx_path: Path, imgsz: int, batch: int) -> tuple[str, tuple[int, ...]]:
    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError(f"No graph inputs found in {onnx_path}")

    selected = model.graph.input[0]
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4:
            selected = value
            break

    dims: list[int] = []
    for d in selected.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)

    if len(dims) != 4:
        raise RuntimeError(
            f"Expected a 4D image input for profiling, got {dims} from input {selected.name!r}"
        )

    c = int(dims[1]) if len(dims) > 1 and int(dims[1]) > 0 else 3
    return selected.name, (int(batch), c, int(imgsz), int(imgsz))


def run_profile(onnx_path: Path, input_name: str, input_shape: tuple[int, ...], node_map_path: Path | None) -> dict[str, Any]:
    try:
        import onnx_tool
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "onnx-tool is not installed in the active environment. Install it once with: pip install onnx-tool"
        ) from exc

    model = onnx_tool.Model(str(onnx_path))
    model.graph.shape_infer({input_name: np.zeros(input_shape, dtype=np.float32)})
    model.graph.profile()
    if node_map_path is not None:
        node_map_path.parent.mkdir(parents=True, exist_ok=True)
        model.graph.print_node_map(str(node_map_path))

    params = getattr(model.graph, "params", None)
    macs = getattr(model.graph, "macs", None)
    if params is None or macs is None:
        raise RuntimeError("onnx-tool did not return params/macs for this graph.")

    def coerce_int_scalar(value: Any, *, field_name: str) -> int:
        if isinstance(value, Number):
            return int(round(float(value)))
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return int(round(float(value.item())))
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if field_name == "macs":
                if not value:
                    raise RuntimeError("onnx-tool returned an empty MAC list.")
                first = value[0]
                if isinstance(first, Number):
                    return int(round(float(first)))
                if isinstance(first, np.ndarray) and first.ndim == 0:
                    return int(round(float(first.item())))
                raise RuntimeError(f"onnx-tool returned an unsupported MAC entry: {type(first)!r}")
            raise RuntimeError(f"onnx-tool returned a non-scalar {field_name}: {type(value)!r}")
        raise RuntimeError(f"onnx-tool returned an unsupported {field_name} type: {type(value)!r}")

    params_value = coerce_int_scalar(params, field_name="params")
    macs_value = coerce_int_scalar(macs, field_name="macs")

    return {
        "params": params_value,
        "macs": macs_value,
        "gmacs": float(macs_value) / 1e9,
        "gflops_if_1mac_eq_2flops": float(macs_value) * 2.0 / 1e9,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute params/MACs/GMACs/GFLOPs(assuming 1 MAC = 2 FLOPs) from best.onnx."
    )
    ap.add_argument("--pkg-root", default=None, help="Benchmark package root. Usually auto-detected.")
    ap.add_argument("--onnx", default=None, help="Path to ONNX model. Default: auto-resolve best.onnx from the attached repo layout.")
    ap.add_argument("--imgsz", type=int, default=640, help="Square input size used for counting, e.g. 512 or 640.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-json", default=None, help="Output JSON path. Default: model/complexity_<imgsz>.json")
    ap.add_argument("--node-map", default=None, help="Optional per-node profiling CSV/TXT from onnx-tool.")
    ap.add_argument("--force-patch", action="store_true", help="Force running scripts/prepare_tensorrt_build_onnx.py before profiling.")
    args = ap.parse_args()

    pkg_root = package_root_from_args(args.pkg_root)
    onnx_path = resolve_default_onnx(pkg_root, args.onnx)
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (onnx_path.parent / f"complexity_{args.imgsz}.json")
    node_map = Path(args.node_map).expanduser().resolve() if args.node_map else None

    profiled_onnx = patch_onnx_if_needed(
        pkg_root=pkg_root,
        onnx_path=onnx_path,
        imgsz=int(args.imgsz),
        out_dir=onnx_path.parent,
        force_patch=bool(args.force_patch),
    )
    input_name, input_shape = resolve_primary_input_name_and_shape(profiled_onnx, int(args.imgsz), int(args.batch))
    values = run_profile(profiled_onnx, input_name, input_shape, node_map)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(values, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
