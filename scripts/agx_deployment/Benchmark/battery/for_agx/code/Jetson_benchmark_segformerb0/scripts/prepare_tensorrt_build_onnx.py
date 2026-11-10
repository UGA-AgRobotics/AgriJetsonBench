from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

def _prefer_active_env_site_packages() -> None:
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


_prefer_active_env_site_packages()

import onnx
from onnx import TensorProto, helper
from onnx.utils import extract_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a TensorRT-friendly ONNX copy for benchmarking.")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--keep-labels-output", action="store_true")
    args = ap.parse_args()

    src = Path(args.onnx).expanduser().resolve()
    dst = Path(args.out).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src))
    graph = model.graph

    input_names = [value.name for value in graph.input]
    patched = False
    if "orig_target_sizes" in input_names:
        patched = True
        retained_inputs = [value for value in graph.input if value.name != "orig_target_sizes"]
        del graph.input[:]
        graph.input.extend(retained_inputs)

        retained_initializers = [init for init in graph.initializer if init.name != "orig_target_sizes"]
        del graph.initializer[:]
        graph.initializer.extend(retained_initializers)
        graph.initializer.append(
            helper.make_tensor(
                name="orig_target_sizes",
                data_type=TensorProto.INT64,
                dims=[1, 2],
                vals=[args.imgsz, args.imgsz],
            )
        )

    output_names = [value.name for value in model.graph.output]
    final_output_names = output_names[:]
    if not args.keep_labels_output and "labels" in output_names:
        final_output_names = [name for name in output_names if name != "labels"]

    if final_output_names != output_names:
        with tempfile.TemporaryDirectory(prefix="benchmark_trt_patch_") as tmpdir:
            temp_model_path = Path(tmpdir) / "patched_inputs.onnx"
            onnx.checker.check_model(model)
            onnx.save(model, str(temp_model_path))
            extract_model(
                str(temp_model_path),
                str(dst),
                input_names=[value.name for value in model.graph.input],
                output_names=final_output_names,
                check_model=True,
            )
            model = onnx.load(str(dst))
    else:
        onnx.checker.check_model(model)
        onnx.save(model, str(dst))

    print(
        json.dumps(
            {
                "source_onnx": str(src),
                "patched_onnx": str(dst),
                "patched": patched,
                "original_input_names": input_names,
                "patched_input_names": [value.name for value in model.graph.input],
                "patched_output_names": [value.name for value in model.graph.output],
                "imgsz_constant": args.imgsz,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
