from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export a TensorRT-friendly RT-DETR ONNX without the deploy-time postprocessor.")
    ap.add_argument("--repo-root", required=True, help="Path to vendor/RT-DETR/rtdetr_pytorch")
    ap.add_argument("--config", required=True, help="RT-DETR repo-style config YAML used to build the model graph")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint path such as best.pth")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--imgsz", type=int, default=640, help="Square image size used for export")
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--simplify", action="store_true", help="Run onnxsim after export when available")
    ap.add_argument("--meta-out", default=None, help="Optional JSON metadata path")
    return ap.parse_args()


def prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_repo_model(repo_root: Path, config_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    prepend_sys_path(repo_root)

    source_package_root = repo_root.parent.parent
    compat_dir = source_package_root / "scripts" / "python_compat"
    source_scripts_dir = source_package_root / "scripts"
    if compat_dir.is_dir():
        prepend_sys_path(compat_dir)
    if source_scripts_dir.is_dir():
        prepend_sys_path(source_scripts_dir)

    from src.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"]:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            raise KeyError(f"Unsupported checkpoint keys for RT-DETR export: {list(checkpoint.keys())[:20]}")
    else:
        raise TypeError(f"Unsupported checkpoint type for RT-DETR export: {type(checkpoint)!r}")

    cfg.model.load_state_dict(state)
    model = cfg.model.deploy()
    model.eval()
    return model


class TRTExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        if isinstance(outputs, dict):
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            return logits, boxes
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            return outputs[0], outputs[1]
        raise TypeError(f"Unexpected RT-DETR deploy output type: {type(outputs)!r}")


def maybe_simplify_onnx(onnx_path: Path) -> dict[str, Any]:
    try:
        import onnx
        import onnxsim
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {"simplified": False, "reason": f"onnxsim unavailable: {exc}"}

    simplified_model, ok = onnxsim.simplify(str(onnx_path))
    if not ok:
        return {"simplified": False, "reason": "onnxsim reported check=False"}
    onnx.save(simplified_model, str(onnx_path))
    return {"simplified": True}


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    meta_out = Path(args.meta_out).expanduser().resolve() if args.meta_out else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_out is not None:
        meta_out.parent.mkdir(parents=True, exist_ok=True)

    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with torch.no_grad():
        model = load_repo_model(repo_root, config_path, checkpoint_path)
        wrapper = TRTExportWrapper(model)
        dummy = torch.rand(args.batch, 3, args.imgsz, args.imgsz, dtype=torch.float32)
        sample_logits, sample_boxes = wrapper(dummy)

        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            verbose=False,
        )

    simplify_report: dict[str, Any] | None = None
    if args.simplify:
        simplify_report = maybe_simplify_onnx(out_path)

    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "out_path": str(out_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "opset": args.opset,
        "input_names": ["images"],
        "output_names": ["pred_logits", "pred_boxes"],
        "sample_output_shapes": {
            "pred_logits": list(sample_logits.shape),
            "pred_boxes": list(sample_boxes.shape),
        },
    }
    if simplify_report is not None:
        report["simplify"] = simplify_report

    if meta_out is not None:
        meta_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
