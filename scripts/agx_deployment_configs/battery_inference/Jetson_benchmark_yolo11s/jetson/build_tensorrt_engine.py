from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import onnx

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.utils import dump_json, ensure_dir, sha256_file, utc_now_iso
else:
    from .utils import dump_json, ensure_dir, sha256_file, utc_now_iso


def analyze_onnx_model(onnx_path: Path) -> dict[str, Any]:
    model = onnx.load(str(onnx_path))
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    dims = []
    for d in input_tensor.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)
    op_types = {node.op_type for node in model.graph.node}
    has_explicit_quantization = (
        "QuantizeLinear" in op_types
        or "DequantizeLinear" in op_types
        or "DynamicQuantizeLinear" in op_types
        or any(op.startswith("QLinear") for op in op_types)
    )
    return {
        "input_name": input_name,
        "dims": dims,
        "has_explicit_quantization": has_explicit_quantization,
    }


def resolve_trtexec() -> str:
    env_override = os.environ.get("TRTEXEC_PATH")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    path_hit = shutil.which("trtexec")
    if path_hit:
        candidates.append(Path(path_hit))
    candidates.extend(
        [
            Path("/usr/src/tensorrt/bin/trtexec"),
            Path("/usr/src/tensorrt/samples/trtexec"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise RuntimeError(
        "trtexec not found. Set TRTEXEC_PATH or install the JetPack TensorRT binaries so trtexec is available."
    )


def workspace_flag(trtexec: str, workspace_mb: int) -> str:
    try:
        cp = subprocess.run([trtexec, "--help"], text=True, capture_output=True, check=False)
        help_text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    except Exception:
        help_text = ""
    if "--memPoolSize" in help_text:
        return f"--memPoolSize=workspace:{workspace_mb}"
    return f"--workspace={workspace_mb}"


def resolve_precision(args: argparse.Namespace) -> str:
    if getattr(args, "precision", None):
        return str(args.precision).lower()
    if getattr(args, "int8", False):
        return "int8"
    if getattr(args, "fp16", False):
        return "fp16"
    return "fp32"


def build_engine(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    out_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    model_info = analyze_onnx_model(onnx_path)
    input_name = str(model_info["input_name"])
    dims = list(model_info["dims"])
    has_explicit_quantization = bool(model_info["has_explicit_quantization"])
    if len(dims) != 4:
        raise RuntimeError(f"Expected 4D input tensor, got {dims}")

    batch = int(args.batch)
    c = dims[1] if dims[1] > 0 else 3
    h = int(args.imgsz if args.imgsz else (dims[2] if dims[2] > 0 else 640))
    w = int(args.imgsz if args.imgsz else (dims[3] if dims[3] > 0 else 640))
    has_dynamic_dims = any(d <= 0 for d in dims)
    precision_tag = resolve_precision(args)
    if precision_tag not in {"fp32", "fp16", "int8"}:
        raise RuntimeError(f"Unsupported precision: {precision_tag}")

    calib_cache_path = Path(args.calib_cache).expanduser().resolve() if getattr(args, "calib_cache", None) else None
    if precision_tag == "int8" and calib_cache_path is not None and not calib_cache_path.exists():
        raise FileNotFoundError(calib_cache_path)
    if precision_tag == "int8" and calib_cache_path is None and not has_explicit_quantization:
        raise RuntimeError(
            "INT8 build requires either a quantized ONNX (Q/DQ nodes) or a calibration cache passed with --calib-cache."
        )

    engine_path = out_dir / f"{onnx_path.stem}_{precision_tag}.engine"
    log_path = out_dir / f"{onnx_path.stem}_{precision_tag}_trtexec.log"

    trtexec = resolve_trtexec()

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        workspace_flag(trtexec, int(args.workspace_mb)),
        "--useCudaGraph",
        "--separateProfileRun",
        "--skipInference",
    ]
    if has_dynamic_dims:
        cmd.append(f"--shapes={input_name}:{batch}x{c}x{h}x{w}")
    if precision_tag == "fp16":
        cmd.append("--fp16")
    elif precision_tag == "int8":
        cmd.append("--int8")
        if calib_cache_path is not None:
            cmd.append(f"--calib={calib_cache_path}")
    if args.no_builder_cache:
        cmd.append("--noBuilderCache")

    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_path.write_text((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else ""), encoding="utf-8")
    if cp.returncode != 0:
        raise RuntimeError(f"trtexec failed. See {log_path}")

    meta = {
        "timestamp_utc": utc_now_iso(),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": precision_tag,
        "workspace_mb": args.workspace_mb,
        "input_name": input_name,
        "shape": [batch, c, h, w],
        "has_dynamic_dims": has_dynamic_dims,
        "has_explicit_quantization": has_explicit_quantization,
        "calib_cache": str(calib_cache_path) if calib_cache_path is not None else None,
        "sha256_engine": sha256_file(engine_path),
        "sha256_onnx": sha256_file(onnx_path),
        "trtexec_log": str(log_path),
        "trtexec_command": cmd,
    }
    dump_json(meta, out_dir / f"{onnx_path.stem}_{precision_tag}_engine_meta.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX with trtexec.")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-cache", default=None, help="Optional TensorRT INT8 calibration cache file")
    ap.add_argument("--no-builder-cache", action="store_true")
    build_engine(ap.parse_args())
