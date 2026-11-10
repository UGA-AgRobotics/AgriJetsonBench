#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2
from PIL import Image, ImageFile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


PKG_ROOT = Path(__file__).resolve().parents[1]
ImageFile.LOAD_TRUNCATED_IMAGES = True
CACHE_SCHEMA_VERSION = "yolo11s_trtexec_engine_accuracy_v1"

if __package__ in {None, ""}:
    sys.path.append(str(PKG_ROOT))
    from jetson.local_paths import (
        resolve_data_yaml_path,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
    )
    from jetson.utils import dump_json, ensure_dir, list_images_from_manifest, sha256_file
else:  # pragma: no cover
    from ..jetson.local_paths import (
        resolve_data_yaml_path,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
    )
    from ..jetson.utils import dump_json, ensure_dir, list_images_from_manifest, sha256_file


def _infer_precision_label(path: Path) -> Optional[str]:
    name = path.name.lower()
    for label in ("int8", "fp16", "fp32"):
        if label in name:
            return label
    return None


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_trtexec_binary(explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env_name in ("TRTEXEC_PATH", "LOCAL_TRTEXEC_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            PKG_ROOT.parent / ".vendor" / "tensorrt-${PRIVATE_LAN_IP}" / "usr" / "src" / "tensorrt" / "bin" / "trtexec",
            Path("/usr/src/tensorrt/bin/trtexec"),
            Path("/usr/src/tensorrt/samples/trtexec"),
        ]
    )
    path_hit = shutil.which("trtexec")
    if path_hit:
        candidates.append(Path(path_hit))

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        path = Path(resolved)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("Could not resolve trtexec. Set TRTEXEC_PATH or LOCAL_TRTEXEC_PATH.")


def _ensure_trtexec_env() -> Dict[str, str]:
    env = os.environ.copy()
    local_lib_dir = os.environ.get("LOCAL_TENSORRT_LIB_DIR")
    if local_lib_dir:
        try:
            local_lib = str(Path(local_lib_dir).expanduser().resolve())
        except Exception:
            local_lib = local_lib_dir
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [part for part in current.split(":") if part]
        if local_lib not in parts:
            parts.insert(0, local_lib)
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _load_rgb_image(image_path: Path) -> np.ndarray:
    try:
        image = Image.open(image_path)
        image.load()
        return np.asarray(image.convert("RGB"))
    except Exception:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            raise
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _letterbox_tensor(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[int, int]]:
    rgb = _load_rgb_image(image_path)
    orig_h, orig_w = rgb.shape[:2]
    scale = min(imgsz / max(orig_w, 1), imgsz / max(orig_h, 1))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    arr = canvas.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr, (orig_h, orig_w)


def _run_trtexec_once(
    *,
    trtexec_path: Path,
    engine_path: Path,
    input_path: Path,
    output_path: Path,
    imgsz: int,
    device: str,
) -> None:
    output_path.unlink(missing_ok=True)

    cmd = [
        str(trtexec_path),
        f"--loadEngine={engine_path}",
        f"--loadInputs=images:{input_path}",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        f"--shapes=images:1x3x{int(imgsz)}x{int(imgsz)}",
        f"--exportOutput={output_path}",
    ]
    device_value = str(device).strip()
    if device_value and device_value.isdigit():
        cmd.append(f"--device={device_value}")

    cp = subprocess.run(cmd, check=False, capture_output=True, text=True, env=_ensure_trtexec_env())
    if cp.returncode != 0:
        combined = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
        raise RuntimeError(
            "trtexec inference failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{combined[-4000:]}"
        )
    if not output_path.exists():
        raise FileNotFoundError(output_path)


def _load_output_tensor(output_path: Path) -> np.ndarray:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected trtexec output payload: {output_path}")

    tensor_entry = None
    for entry in payload:
        if isinstance(entry, dict) and str(entry.get("name")) == "output0":
            tensor_entry = entry
            break
    if tensor_entry is None:
        tensor_entry = payload[0]

    dims = str(tensor_entry.get("dimensions") or "").strip()
    values = tensor_entry.get("values")
    if not dims or not isinstance(values, list):
        raise RuntimeError(f"Unexpected tensor export structure in {output_path}")

    shape = tuple(int(part) for part in dims.split("x") if part)
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(shape)


def _detections_from_tensor(
    *,
    prediction: np.ndarray,
    orig_shape: tuple[int, int],
    imgsz: int,
    nc: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> torch.Tensor:
    pred = torch.from_numpy(prediction.astype(np.float32, copy=False))
    nms_output = non_max_suppression(
        pred,
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        nc=int(nc),
        max_det=int(max_det),
    )
    if not nms_output:
        return torch.zeros((0, 6), dtype=torch.float32)

    det = nms_output[0].detach().cpu()
    if det.numel() == 0:
        return det.reshape(0, 6)
    det[:, :4] = scale_boxes((int(imgsz), int(imgsz)), det[:, :4], orig_shape)
    return det


def _xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _original_ground_truth_json_path(dataset_root: Path) -> Path:
    path = dataset_root / "annotations" / "test_coco_ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _build_image_id_map(coco_gt: COCO) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for image in coco_gt.dataset.get("images", []):
        file_name = str(image.get("file_name") or "")
        image_id = int(image.get("id"))
        if file_name:
            mapping[file_name] = image_id
    return mapping


def _category_id_map(coco_gt: COCO) -> Dict[int, int]:
    categories = sorted(coco_gt.dataset.get("categories", []), key=lambda item: int(item.get("id", 0)))
    mapping: Dict[int, int] = {}
    for class_index, category in enumerate(categories):
        mapping[class_index] = int(category.get("id"))
    return mapping


def _prepare_ground_truth_subset(
    *,
    dataset_root: Path,
    image_paths: List[Path],
    accuracy_dir: Path,
) -> tuple[Path, COCO]:
    source_gt_path = _original_ground_truth_json_path(dataset_root)
    payload = json.loads(source_gt_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise RuntimeError(f"Unexpected COCO ground-truth structure in {source_gt_path}")

    keep_file_names = {
        image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
        for image_path in image_paths
    }
    filtered_images = [image for image in images if str(image.get("file_name") or "") in keep_file_names]
    filtered_image_ids = {int(image.get("id")) for image in filtered_images}
    filtered_annotations = [ann for ann in annotations if int(ann.get("image_id")) in filtered_image_ids]

    if len(filtered_images) == len(images):
        return source_gt_path, COCO(str(source_gt_path))

    subset_payload = dict(payload)
    subset_payload["images"] = filtered_images
    subset_payload["annotations"] = filtered_annotations
    subset_path = accuracy_dir / "runtime_ground_truth_subset.json"
    dump_json(subset_payload, subset_path)
    return subset_path, COCO(str(subset_path))


def _evaluate_coco(coco_gt: COCO, detections: List[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        empty_path = Path(tempfile.mkstemp(prefix="yolo11s_empty_det_", suffix=".json")[1])
        try:
            empty_path.write_text("[]\n", encoding="utf-8")
            coco_dt = coco_gt.loadRes(str(empty_path))
        finally:
            empty_path.unlink(missing_ok=True)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        coco_eval.summarize()
    summary_text = buffer.getvalue().strip()
    stats = list(coco_eval.stats)
    metrics = {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
    }
    return metrics, summary_text


def _accuracy_cache_root(
    *,
    model_path: Path,
    manifest_path: Path,
    data_yaml: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_images: Optional[int],
) -> Path:
    key_payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "data_yaml_path": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "split_name": str(split_name),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "conf_thres": float(conf_thres),
        "iou_thres": float(iou_thres),
        "max_det": int(max_det),
        "max_images": None if max_images is None else int(max_images),
    }
    key = _sha256_text(json.dumps(key_payload, sort_keys=True))
    return model_path.parent / "accuracy_cache" / key


def _write_accuracy_outputs(
    *,
    accuracy_dir: Path,
    split_name: str,
    metrics_payload: Dict[str, Any],
    scalar_metrics: Dict[str, Any],
    report_payload: Dict[str, Any],
    summary_text: str,
    detections: List[Dict[str, Any]],
    coco_gt_path: Path,
    detections_source_path: Optional[Path] = None,
    skipped_images: Optional[List[Dict[str, str]]] = None,
    skipped_images_source_path: Optional[Path] = None,
) -> None:
    val_dir = ensure_dir(accuracy_dir / f"val_{split_name}")
    detections_json_path = val_dir / "detections_coco.json"
    skipped_json_path = val_dir / "skipped_images.json"

    metrics_payload = dict(metrics_payload)
    metrics_payload["ground_truth_json"] = str(coco_gt_path.resolve())
    metrics_payload["detections_json"] = str(detections_json_path.resolve())

    report_payload = dict(report_payload)
    report_payload["processed_samples"] = metrics_payload.get("processed_samples")

    if detections_source_path is not None and detections_source_path.exists():
        shutil.copy2(detections_source_path, detections_json_path)
    else:
        dump_json(detections, detections_json_path)

    if skipped_images_source_path is not None and skipped_images_source_path.exists():
        shutil.copy2(skipped_images_source_path, skipped_json_path)
    else:
        dump_json({"skipped_images": skipped_images or []}, skipped_json_path)

    validation_log = accuracy_dir / f"validation_{split_name}.log"
    validation_lines = [
        f"helper={Path(__file__).name}",
        f"backend={metrics_payload.get('backend')}",
        f"backend_detail={metrics_payload.get('backend_detail')}",
        f"precision_label={metrics_payload.get('precision_label')}",
        f"model_path={metrics_payload.get('model_path')}",
        f"model_sha256={metrics_payload.get('model_sha256')}",
        f"manifest_path={metrics_payload.get('manifest_path')}",
        f"data_yaml_path={metrics_payload.get('data_yaml_path')}",
        f"ground_truth_json={metrics_payload.get('ground_truth_json')}",
        f"detections_json={metrics_payload.get('detections_json')}",
        f"trtexec_path={metrics_payload.get('trtexec_path')}",
        f"split_name={split_name}",
        f"processed_samples={metrics_payload.get('processed_samples')}",
        f"prediction_count={metrics_payload.get('prediction_count')}",
        f"skipped_images_count={metrics_payload.get('skipped_images_count')}",
        f"mAP50_95={scalar_metrics.get('mAP50_95')}",
        f"mAP50={scalar_metrics.get('mAP50')}",
        f"mAP75={scalar_metrics.get('mAP75')}",
        f"AR100={scalar_metrics.get('AR100')}",
    ]
    if summary_text:
        validation_lines.extend(["", summary_text])
    validation_log.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    dump_json(metrics_payload, accuracy_dir / "val_metrics.json")
    dump_json(metrics_payload, val_dir / "val_metrics.json")
    dump_json(scalar_metrics, val_dir / "metrics.json")
    dump_json(report_payload, val_dir / "report.json")


def _restore_accuracy_from_cache(
    *,
    cache_root: Path,
    output_root: Path,
    split_name: str,
) -> Optional[Dict[str, Any]]:
    accuracy_dir = ensure_dir(output_root / "accuracy")
    val_dir = ensure_dir(accuracy_dir / f"val_{split_name}")
    cache_accuracy_dir = cache_root / "accuracy"
    cache_val_dir = cache_accuracy_dir / f"val_{split_name}"
    cache_payload_path = cache_accuracy_dir / "val_metrics.json"
    cache_metrics_path = cache_val_dir / "metrics.json"
    cache_report_path = cache_val_dir / "report.json"
    cache_gt_path = cache_accuracy_dir / "runtime_ground_truth_subset.json"
    cache_det_path = cache_val_dir / "detections_coco.json"
    cache_skipped_path = cache_val_dir / "skipped_images.json"
    cache_summary_path = cache_root / "summary.txt"

    required = [cache_payload_path, cache_metrics_path, cache_report_path, cache_gt_path, cache_det_path, cache_summary_path]
    if not all(path.exists() for path in required):
        return None

    metrics_payload = json.loads(cache_payload_path.read_text(encoding="utf-8"))
    scalar_metrics = json.loads(cache_metrics_path.read_text(encoding="utf-8"))
    report_payload = json.loads(cache_report_path.read_text(encoding="utf-8"))
    summary_text = cache_summary_path.read_text(encoding="utf-8")
    shutil.copy2(cache_gt_path, accuracy_dir / "runtime_ground_truth_subset.json")

    _write_accuracy_outputs(
        accuracy_dir=accuracy_dir,
        split_name=split_name,
        metrics_payload=metrics_payload,
        scalar_metrics=scalar_metrics,
        report_payload=report_payload,
        summary_text=summary_text,
        detections=[],
        coco_gt_path=accuracy_dir / "runtime_ground_truth_subset.json",
        detections_source_path=cache_det_path,
        skipped_images_source_path=cache_skipped_path if cache_skipped_path.exists() else None,
    )
    return json.loads((accuracy_dir / "val_metrics.json").read_text(encoding="utf-8"))


def _save_accuracy_cache(
    *,
    cache_root: Path,
    accuracy_dir: Path,
    split_name: str,
    summary_text: str,
) -> None:
    cache_accuracy_dir = ensure_dir(cache_root / "accuracy")
    cache_val_dir = ensure_dir(cache_accuracy_dir / f"val_{split_name}")
    for src, dst in [
        (accuracy_dir / "val_metrics.json", cache_accuracy_dir / "val_metrics.json"),
        (accuracy_dir / "runtime_ground_truth_subset.json", cache_accuracy_dir / "runtime_ground_truth_subset.json"),
        (accuracy_dir / f"val_{split_name}" / "val_metrics.json", cache_val_dir / "val_metrics.json"),
        (accuracy_dir / f"val_{split_name}" / "metrics.json", cache_val_dir / "metrics.json"),
        (accuracy_dir / f"val_{split_name}" / "report.json", cache_val_dir / "report.json"),
        (accuracy_dir / f"val_{split_name}" / "detections_coco.json", cache_val_dir / "detections_coco.json"),
        (accuracy_dir / f"val_{split_name}" / "skipped_images.json", cache_val_dir / "skipped_images.json"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
    (cache_root / "summary.txt").write_text(summary_text, encoding="utf-8")


def run_accuracy(
    *,
    model_path: Path,
    dataset_root: Path,
    data_yaml: Path,
    manifest_path: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    device: str,
    output_root: Path,
    trtexec_path: Path,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_images: Optional[int],
) -> Dict[str, Any]:
    if int(batch) != 1:
        raise RuntimeError("The trtexec-based YOLO11s engine accuracy helper currently supports only --batch 1.")

    accuracy_dir = ensure_dir(output_root / "accuracy")
    val_dir = ensure_dir(accuracy_dir / f"val_{split_name}")
    detections_json_path = val_dir / "detections_coco.json"
    cache_root = _accuracy_cache_root(
        model_path=model_path,
        manifest_path=manifest_path,
        data_yaml=data_yaml,
        split_name=split_name,
        imgsz=imgsz,
        batch=batch,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        max_images=max_images,
    )
    restored = _restore_accuracy_from_cache(
        cache_root=cache_root,
        output_root=output_root,
        split_name=split_name,
    )
    if restored is not None:
        print(f"[INFO] Reused cached YOLO11s engine accuracy from {cache_root}", flush=True)
        return restored

    image_paths = list_images_from_manifest(dataset_root, manifest_path)
    if max_images is not None:
        image_paths = image_paths[: max(1, int(max_images))]

    source_gt_path = _original_ground_truth_json_path(dataset_root)
    source_coco_gt = COCO(str(source_gt_path))
    image_id_by_file = _build_image_id_map(source_coco_gt)
    class_to_category_id = _category_id_map(source_coco_gt)
    nc = len(class_to_category_id)

    detections: List[Dict[str, Any]] = []
    evaluated_image_paths: List[Path] = []
    skipped_images: List[Dict[str, str]] = []
    trtexec_command_template = [
        str(trtexec_path),
        f"--loadEngine={model_path}",
        "--loadInputs=images:<binary-fp32-input>",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        f"--shapes=images:1x3x{int(imgsz)}x{int(imgsz)}",
        "--exportOutput=<json-output>",
    ]
    if str(device).strip().isdigit():
        trtexec_command_template.append(f"--device={str(device).strip()}")

    with tempfile.TemporaryDirectory(prefix="yolo11s_trtexec_accuracy_") as tmpdir:
        tmp_root = Path(tmpdir)
        input_path = tmp_root / "input.fp32"
        output_path = tmp_root / "output.json"

        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            try:
                input_tensor, orig_shape = _letterbox_tensor(image_path, int(imgsz))
            except Exception as exc:
                skipped_images.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[WARN] Skipping unreadable image: {image_path} ({type(exc).__name__}: {exc})", flush=True)
                continue
            input_tensor.tofile(input_path)
            _run_trtexec_once(
                trtexec_path=trtexec_path,
                engine_path=model_path,
                input_path=input_path,
                output_path=output_path,
                imgsz=int(imgsz),
                device=str(device),
            )
            prediction = _load_output_tensor(output_path)
            det = _detections_from_tensor(
                prediction=prediction,
                orig_shape=orig_shape,
                imgsz=int(imgsz),
                nc=int(nc),
                conf_thres=float(conf_thres),
                iou_thres=float(iou_thres),
                max_det=int(max_det),
            )

            image_key = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
            image_id = image_id_by_file.get(image_key)
            if image_id is None:
                raise KeyError(f"Could not resolve COCO image id for {image_key}")
            evaluated_image_paths.append(image_path)

            for row in det:
                class_index = int(row[5].item())
                category_id = class_to_category_id.get(class_index)
                if category_id is None:
                    continue
                detections.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": _xyxy_to_xywh(row[:4]),
                        "score": float(row[4].item()),
                    }
                )

            if index == 1 or index == total or index % 25 == 0:
                print(f"[INFO] Accuracy inference progress: {index}/{total} images", flush=True)

    coco_gt_path, coco_gt = _prepare_ground_truth_subset(
        dataset_root=dataset_root,
        image_paths=evaluated_image_paths,
        accuracy_dir=accuracy_dir,
    )
    scalar_metrics, summary_text = _evaluate_coco(coco_gt, detections)

    metrics_payload: Dict[str, Any] = {
        "metrics/mAP50-95(B)": scalar_metrics["mAP50_95"],
        "metrics/mAP50(B)": scalar_metrics["mAP50"],
        "metrics/mAP75(B)": scalar_metrics["mAP75"],
        "metrics/AR1": scalar_metrics["AR1"],
        "metrics/AR10": scalar_metrics["AR10"],
        "metrics/AR100": scalar_metrics["AR100"],
        "metrics/recall(B)": scalar_metrics["AR100"],
        "backend": "tensorrt_engine",
        "backend_detail": "trtexec_yolo11s_engine_coco_eval",
        "precision_label": _infer_precision_label(model_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "manifest_path": str(manifest_path.resolve()),
        "data_yaml_path": str(data_yaml.resolve()),
        "ground_truth_json": str(coco_gt_path.resolve()),
        "detections_json": str(detections_json_path.resolve()),
        "trtexec_path": str(trtexec_path.resolve()),
        "trtexec_command_template": trtexec_command_template,
        "split_name": split_name,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "processed_samples": int(len(evaluated_image_paths)),
        "prediction_count": int(len(detections)),
        "skipped_images_count": int(len(skipped_images)),
        "conf_threshold": float(conf_thres),
        "iou_threshold": float(iou_thres),
        "max_det": int(max_det),
    }

    report_payload = {
        "task": "detection",
        "backend": metrics_payload.get("backend"),
        "backend_detail": metrics_payload.get("backend_detail"),
        "precision_label": metrics_payload.get("precision_label"),
        "model_path": metrics_payload.get("model_path"),
        "model_sha256": metrics_payload.get("model_sha256"),
        "manifest_path": metrics_payload.get("manifest_path"),
        "data_yaml_path": metrics_payload.get("data_yaml_path"),
        "split_name": split_name,
        "processed_samples": metrics_payload.get("processed_samples"),
        "metrics": scalar_metrics,
    }

    _write_accuracy_outputs(
        accuracy_dir=accuracy_dir,
        split_name=split_name,
        metrics_payload=metrics_payload,
        scalar_metrics=scalar_metrics,
        report_payload=report_payload,
        summary_text=summary_text,
        detections=detections,
        coco_gt_path=coco_gt_path,
        skipped_images=skipped_images,
    )
    _save_accuracy_cache(
        cache_root=cache_root,
        accuracy_dir=accuracy_dir,
        split_name=split_name,
        summary_text=summary_text,
    )
    return json.loads((accuracy_dir / "val_metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run precision-specific YOLO11s TensorRT engine accuracy validation.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--data-yaml", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--trtexec", default=None)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if model_path.suffix.lower() != ".engine":
        raise RuntimeError(f"Expected a TensorRT engine for YOLO11s precision-specific accuracy: {model_path}")

    bundle_root = resolve_inference_bundle_root(model_path)
    dataset_root = resolve_dataset_root(args.dataset_root, bundle_root=bundle_root)
    data_yaml = resolve_data_yaml_path(dataset_root=dataset_root, explicit=args.data_yaml, bundle_root=bundle_root)
    if data_yaml is None:
        raise FileNotFoundError(f"Could not resolve a data YAML under {dataset_root}")
    manifest_path = resolve_manifest_path(dataset_root=dataset_root, explicit=args.manifest, bundle_root=bundle_root)
    if manifest_path is None:
        raise FileNotFoundError(f"Could not resolve a fixed-order manifest under {dataset_root}")

    payload = run_accuracy(
        model_path=model_path,
        dataset_root=dataset_root,
        data_yaml=data_yaml,
        manifest_path=manifest_path,
        split_name=str(args.split_name or "test"),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        output_root=Path(args.output_root).expanduser().resolve(),
        trtexec_path=_resolve_trtexec_binary(args.trtexec),
        conf_thres=float(args.conf_thres),
        iou_thres=float(args.iou_thres),
        max_det=int(args.max_det),
        max_images=args.max_images,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
