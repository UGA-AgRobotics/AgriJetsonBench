#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PKG_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ACCURACY_HELPER = PKG_ROOT / "scripts" / "run_yolo11s_engine_accuracy.py"
COMPARE_RUNS = PKG_ROOT / "analysis" / "compare_runs.py"
MERGE_AND_SUMMARIZE = PKG_ROOT / "analysis" / "merge_and_summarize.py"


if __package__ in {None, ""}:
    sys.path.append(str(PKG_ROOT))
    from jetson.local_paths import resolve_dataset_root, resolve_inference_bundle_root
    from jetson.utils import dump_json, sha256_file
else:  # pragma: no cover
    from ..jetson.local_paths import resolve_dataset_root, resolve_inference_bundle_root
    from ..jetson.utils import dump_json, sha256_file


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    dump_json(payload, path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')


def _normalize_path_str(value: Optional[str | Path]) -> Optional[str]:
    if value in {None, ''}:
        return None
    return str(Path(value).expanduser().resolve())


def _normalize_complexity_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, Any] = {}
    for key in ("params", "macs", "gmacs", "gflops_if_1mac_eq_2flops"):
        value = payload.get(key)
        if value is None:
            return None
        if key in {"params", "macs"}:
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def _load_complexity_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return _normalize_complexity_payload(raw)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _cleanup_accuracy_output(run_root: Path, split_name: str) -> None:
    accuracy_dir = run_root / "accuracy"
    val_dir = accuracy_dir / f"val_{split_name}"
    if val_dir.exists():
        shutil.rmtree(val_dir)
    _safe_unlink(accuracy_dir / "val_metrics.json")
    _safe_unlink(accuracy_dir / f"validation_{split_name}.log")


def _archive_legacy_backup(run_root: Path) -> None:
    legacy = run_root / "FINAL_SUMMARY.pre_precision_accuracy_backup.json"
    if not legacy.exists():
        return
    archive_dir = run_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(archive_dir / legacy.name))


def _resolve_accuracy_manifest(accuracy_metrics: Dict[str, Any]) -> Optional[str]:
    manifest_value = accuracy_metrics.get("manifest_path") if isinstance(accuracy_metrics, dict) else None
    return _normalize_path_str(manifest_value)


def _resolve_existing_file_path(value: Optional[str | Path]) -> Optional[Path]:
    if value in {None, ""}:
        return None
    return _resolve_existing_layout_path(Path(value))


def _normalize_existing_file_path(value: Optional[str | Path]) -> Optional[str]:
    path = _resolve_existing_file_path(value)
    return str(path) if path is not None else None


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
                (PKG_ROOT.parent / "yolo11s" / "inference_bundle" / "model" / path.name).resolve(),
                (PKG_ROOT.parent / "yolo11s" / "inference_bundle" / "dataset" / path.name).resolve(),
                (PKG_ROOT.parent / "yolo11s" / "model" / path.name).resolve(),
                (PKG_ROOT.parent / "yolo11s" / "dataset" / path.name).resolve(),
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


def _resolve_existing_dataset_root(path: Path, measured_model_path: Path) -> Path:
    candidate = _resolve_existing_layout_path(path)
    if candidate.exists():
        try:
            return resolve_dataset_root(candidate)
        except Exception:
            return candidate

    bundle_root = resolve_inference_bundle_root(measured_model_path)
    if bundle_root is None:
        bundle_root = resolve_inference_bundle_root()
    if bundle_root is not None:
        try:
            return resolve_dataset_root(bundle_root=bundle_root)
        except Exception:
            pass
    return candidate


def _resolve_complexity_artifacts(measured_model_path: Path, imgsz: int) -> tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    bundle_root = resolve_inference_bundle_root(measured_model_path)
    candidates = [
        measured_model_path.with_name("best.onnx"),
        measured_model_path.with_suffix(".onnx"),
    ]
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / "model" / "best.onnx",
                bundle_root / "model" / "best_trt_deploy.onnx",
            ]
        )

    seen: set[str] = set()
    onnx_path: Optional[Path] = None
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            onnx_path = candidate.resolve()
            break

    if onnx_path is None:
        return None, None, None

    complexity_json = onnx_path.parent / f"complexity_{int(imgsz)}.json"
    return _load_complexity_json(complexity_json), complexity_json if complexity_json.exists() else None, onnx_path


def _infer_split_name(run_root: Path, final_payload: Dict[str, Any]) -> str:
    files = final_payload.get("files", {}) if isinstance(final_payload.get("files", {}), dict) else {}
    validation_log = files.get("accuracy_validation_log")
    if validation_log:
        name = Path(str(validation_log)).name
        if name.startswith("validation_") and name.endswith(".log"):
            return name[len("validation_") : -len(".log")] or "test"

    accuracy_dir = run_root / "accuracy"
    for candidate in sorted(accuracy_dir.glob("val_*")):
        if candidate.is_dir():
            return candidate.name[len("val_") :] or "test"
    return "test"


def _discover_variant_roots(session_root: Path, variant: Optional[str]) -> List[Path]:
    if variant:
        root = session_root / variant
        if not root.is_dir():
            raise FileNotFoundError(root)
        return [root.resolve()]

    roots: List[Path] = []
    for name in ("standard", "per_image"):
        candidate = session_root / name
        if candidate.is_dir():
            roots.append(candidate.resolve())
    if roots:
        return roots

    runs_root = session_root / "runs"
    if runs_root.is_dir():
        return [session_root.resolve()]
    raise FileNotFoundError(f"Could not locate variant roots under {session_root}")


def _discover_run_roots(variant_root: Path) -> List[Path]:
    runs_root = variant_root / "runs"
    if not runs_root.is_dir():
        return []
    return sorted(path.parent.resolve() for path in runs_root.glob("*/FINAL_SUMMARY.json"))


def _infer_run_precision(final_payload: Dict[str, Any], measured_model_path: Path) -> str:
    model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
    precision = str(model.get("precision_label") or "").strip().lower()
    if precision in {"fp16", "fp32", "int8"}:
        return precision
    stem = measured_model_path.stem.lower()
    for candidate in ("fp16", "fp32", "int8"):
        if candidate in stem:
            return candidate
    return ""


def _run_engine_accuracy(
    *,
    engine_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    run_root: Path,
) -> Dict[str, Any]:
    accuracy_dir = run_root / "accuracy"
    output_dir = accuracy_dir / f"val_{split_name}"
    out_json = accuracy_dir / "val_metrics.json"
    validation_log = accuracy_dir / f"validation_{split_name}.log"
    cmd = [
        str(Path(sys.executable).resolve()),
        str(ENGINE_ACCURACY_HELPER),
        "--model",
        str(engine_path),
        "--dataset-root",
        str(dataset_root),
        "--split-name",
        str(split_name),
        "--imgsz",
        str(int(imgsz)),
        "--output-root",
        str(run_root),
    ]
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    combined_output = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
    if cp.returncode != 0:
        validation_log.write_text(combined_output + "\n", encoding="utf-8")
        raise RuntimeError(
            "YOLO11s precision-specific backfill accuracy failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{combined_output[-2000:]}"
        )
    if not validation_log.exists() and combined_output:
        validation_log.write_text(combined_output + "\n", encoding="utf-8")
    return _load_json(out_json)


def _refresh_final_summary_clean_metadata(final_payload: Dict[str, Any]) -> None:
    aggregate_mean = final_payload.get("aggregate_mean") or {}
    external_summary = final_payload.get("external_summary") or {}
    final_payload["performance"] = {
        "images_benchmarked": aggregate_mean.get("images_benchmarked"),
        "benchmark_duration_s": aggregate_mean.get("benchmark_duration_s"),
        "warmup_elapsed_s": aggregate_mean.get("warmup_elapsed_s"),
        "latency_median_ms": aggregate_mean.get("latency_median_ms"),
        "latency_p95_ms": aggregate_mean.get("latency_p95_ms"),
        "latency_mean_ms": aggregate_mean.get("latency_mean_ms"),
        "throughput_fps": aggregate_mean.get("throughput_fps"),
    }
    final_payload["power"] = {
        "avg_power_external_w": aggregate_mean.get("avg_power_external_w", external_summary.get("avg_power_w")),
        "peak_power_external_w": aggregate_mean.get("peak_power_external_w", external_summary.get("peak_power_w")),
        "energy_joules_external": aggregate_mean.get("energy_joules_external", external_summary.get("energy_joules")),
        "energy_wh_external": aggregate_mean.get("energy_wh_external", external_summary.get("energy_wh")),
        "joules_per_image_external": aggregate_mean.get("joules_per_image_external", external_summary.get("joules_per_image")),
        "external_temp_mean_c": aggregate_mean.get("external_temp_mean_c", external_summary.get("external_temp_mean_c")),
        "external_temp_max_c": aggregate_mean.get("external_temp_max_c", external_summary.get("external_temp_max_c")),
        "internal_total_power_mean_w": aggregate_mean.get("internal_total_power_mean_w"),
        "internal_total_power_peak_w": aggregate_mean.get("internal_total_power_peak_w"),
        "internal_temp_mean_c": aggregate_mean.get("internal_temp_mean_c"),
        "internal_temp_max_c": aggregate_mean.get("internal_temp_max_c"),
    }


def _copy_accuracy_artifacts(source_run_root: Path, target_run_root: Path, split_name: str) -> None:
    source_accuracy_dir = source_run_root / "accuracy"
    target_accuracy_dir = target_run_root / "accuracy"
    target_accuracy_dir.mkdir(parents=True, exist_ok=True)

    for name in ("val_metrics.json", f"validation_{split_name}.log"):
        src = source_accuracy_dir / name
        dst = target_accuracy_dir / name
        if src.exists():
            shutil.copy2(src, dst)

    src_val_dir = source_accuracy_dir / f"val_{split_name}"
    dst_val_dir = target_accuracy_dir / f"val_{split_name}"
    if src_val_dir.exists():
        if dst_val_dir.exists():
            shutil.rmtree(dst_val_dir)
        shutil.copytree(src_val_dir, dst_val_dir)


def _update_final_summary(run_root: Path, final_payload: Dict[str, Any], accuracy_metrics: Dict[str, Any], split_name: str) -> None:
    final_path = run_root / "FINAL_SUMMARY.json"
    _archive_legacy_backup(run_root)
    backup_dir = run_root / "archive"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "FINAL_SUMMARY.pre_precision_accuracy_backup.json"
    if not backup_path.exists():
        backup_path.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")

    model = final_payload.setdefault("model", {})
    dataset = final_payload.setdefault("dataset", {})
    files = final_payload.setdefault("files", {})
    measured_model_path = _resolve_existing_layout_path(Path(str(model["measured_model_path"])))
    dataset_root = _resolve_existing_dataset_root(Path(str(dataset.get("root", ""))), measured_model_path)
    profile_input = model.get("profile_input") if isinstance(model.get("profile_input"), dict) else {}
    imgsz = int(profile_input.get("imgsz") or accuracy_metrics.get("imgsz") or 640)
    model_complexity, complexity_json_path, complexity_onnx_path = _resolve_complexity_artifacts(measured_model_path, imgsz)
    accuracy_dir = run_root / "accuracy"
    output_dir = accuracy_dir / f"val_{split_name}"
    accuracy_manifest = _resolve_accuracy_manifest(accuracy_metrics)
    benchmark_manifest = (
        _normalize_existing_file_path(dataset.get("benchmark_manifest"))
        or _normalize_existing_file_path(dataset.get("manifest"))
    )

    model["measured_model_path"] = str(measured_model_path)
    model["accuracy_model_path"] = str(measured_model_path)
    model["accuracy_model_sha256"] = sha256_file(measured_model_path)
    model["accuracy_backend"] = accuracy_metrics.get("backend")
    model["accuracy_backend_detail"] = accuracy_metrics.get("backend_detail")
    model["accuracy_precision_label"] = accuracy_metrics.get("precision_label")
    model["accuracy_precision_specific"] = True
    model["complexity"] = model_complexity
    final_payload["accuracy_metrics"] = accuracy_metrics
    dataset["root"] = str(dataset_root)
    data_yaml = _normalize_existing_file_path(dataset.get("data_yaml"))
    if data_yaml is not None:
        dataset["data_yaml"] = data_yaml
    if benchmark_manifest is not None:
        dataset["benchmark_manifest"] = benchmark_manifest
        dataset["ordered_manifest"] = benchmark_manifest
    if accuracy_manifest is not None:
        dataset["accuracy_manifest"] = accuracy_manifest
        dataset["manifest"] = accuracy_manifest
    if accuracy_metrics.get("processed_samples") is not None:
        dataset["accuracy_image_count"] = accuracy_metrics.get("processed_samples")

    metrics_json = accuracy_dir / "val_metrics.json"
    report_json = output_dir / "report.json"
    raw_metrics_json = output_dir / "metrics.json"
    val_test_metrics_json = output_dir / "val_metrics.json"
    per_class_metrics_csv = output_dir / "per_class_metrics.csv"
    confusion_matrix_csv = output_dir / "confusion_matrix.csv"
    confusion_matrix_png = output_dir / "confusion_matrix.png"
    per_class_iou_png = output_dir / "per_class_iou.png"
    per_class_dice_png = output_dir / "per_class_dice.png"
    validation_log = accuracy_dir / f"validation_{split_name}.log"

    files["accuracy_metrics_json"] = str(metrics_json) if metrics_json.exists() else None
    files["accuracy_report_json"] = str(report_json) if report_json.exists() else None
    files["accuracy_raw_metrics_json"] = str(raw_metrics_json) if raw_metrics_json.exists() else None
    files["accuracy_val_test_metrics_json"] = str(val_test_metrics_json) if val_test_metrics_json.exists() else None
    files["accuracy_per_class_metrics_csv"] = str(per_class_metrics_csv) if per_class_metrics_csv.exists() else None
    files["accuracy_confusion_matrix_csv"] = str(confusion_matrix_csv) if confusion_matrix_csv.exists() else None
    files["accuracy_confusion_matrix_png"] = str(confusion_matrix_png) if confusion_matrix_png.exists() else None
    files["accuracy_per_class_iou_png"] = str(per_class_iou_png) if per_class_iou_png.exists() else None
    files["accuracy_per_class_dice_png"] = str(per_class_dice_png) if per_class_dice_png.exists() else None
    files["accuracy_validation_log"] = str(validation_log) if validation_log.exists() else None
    files["complexity_json"] = str(complexity_json_path) if complexity_json_path is not None else None
    files["complexity_profile_onnx"] = str(complexity_onnx_path) if complexity_onnx_path is not None else None

    _refresh_final_summary_clean_metadata(final_payload)
    _write_json(final_path, final_payload)


def _update_config_snapshot(run_root: Path, measured_model_path: Path, accuracy_metrics: Dict[str, Any]) -> None:
    config_path = run_root / "config_snapshot.yaml"
    if not config_path.exists():
        return
    cfg = _load_yaml(config_path)
    model = cfg.setdefault("model", {})
    dataset = cfg.setdefault("dataset", {})
    dataset_root = _resolve_existing_dataset_root(Path(str(dataset.get("root", ""))), measured_model_path)
    accuracy_manifest = _resolve_accuracy_manifest(accuracy_metrics)
    benchmark_manifest = (
        _normalize_existing_file_path(dataset.get("benchmark_manifest"))
        or _normalize_existing_file_path(dataset.get("ordered_manifest"))
        or _normalize_existing_file_path(dataset.get("manifest"))
    )
    model["model_path"] = str(measured_model_path)
    model["accuracy_model_path"] = str(measured_model_path)
    model["accuracy_model_sha256"] = sha256_file(measured_model_path)
    model["accuracy_backend"] = accuracy_metrics.get("backend")
    model["accuracy_backend_detail"] = accuracy_metrics.get("backend_detail")
    model["accuracy_precision_label"] = accuracy_metrics.get("precision_label")
    model["accuracy_precision_specific"] = True
    dataset["root"] = str(dataset_root)
    data_yaml = _normalize_existing_file_path(dataset.get("data_yaml"))
    if data_yaml is not None:
        dataset["data_yaml"] = data_yaml
    if benchmark_manifest is not None:
        dataset["benchmark_manifest"] = benchmark_manifest
        dataset["ordered_manifest"] = benchmark_manifest
    if accuracy_manifest is not None:
        dataset["accuracy_manifest"] = accuracy_manifest
        dataset["manifest"] = accuracy_manifest
    _write_yaml(config_path, cfg)


def _validate_updated_run(run_root: Path, measured_model_path: Path, accuracy_metrics: Dict[str, Any], split_name: str) -> None:
    expected_model_path = _normalize_path_str(measured_model_path)
    expected_model_sha = sha256_file(measured_model_path)
    expected_backend = accuracy_metrics.get("backend")
    expected_precision = accuracy_metrics.get("precision_label")
    expected_manifest = _resolve_accuracy_manifest(accuracy_metrics)
    issues: List[str] = []

    final_payload = _load_json(run_root / "FINAL_SUMMARY.json")
    final_model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
    final_dataset = final_payload.get("dataset", {}) if isinstance(final_payload.get("dataset", {}), dict) else {}
    config_payload = _load_yaml(run_root / "config_snapshot.yaml") if (run_root / "config_snapshot.yaml").exists() else {}
    config_model = config_payload.get("model", {}) if isinstance(config_payload.get("model", {}), dict) else {}
    config_dataset = config_payload.get("dataset", {}) if isinstance(config_payload.get("dataset", {}), dict) else {}
    validation_log = run_root / "accuracy" / f"validation_{split_name}.log"

    for label, actual, expected in (
        ("FINAL_SUMMARY accuracy_model_path", _normalize_path_str(final_model.get("accuracy_model_path")), expected_model_path),
        ("FINAL_SUMMARY accuracy_model_sha256", final_model.get("accuracy_model_sha256"), expected_model_sha),
        ("FINAL_SUMMARY accuracy_backend", final_model.get("accuracy_backend"), expected_backend),
        ("FINAL_SUMMARY accuracy_precision_label", final_model.get("accuracy_precision_label"), expected_precision),
        ("config_snapshot accuracy_model_path", _normalize_path_str(config_model.get("accuracy_model_path")), expected_model_path),
        ("config_snapshot accuracy_model_sha256", config_model.get("accuracy_model_sha256"), expected_model_sha),
        ("config_snapshot accuracy_backend", config_model.get("accuracy_backend"), expected_backend),
        ("config_snapshot accuracy_precision_label", config_model.get("accuracy_precision_label"), expected_precision),
        ("FINAL_SUMMARY dataset.manifest", _normalize_path_str(final_dataset.get("manifest")), expected_manifest),
        ("config_snapshot dataset.manifest", _normalize_path_str(config_dataset.get("manifest")), expected_manifest),
    ):
        if expected is not None and actual != expected:
            issues.append(f"{label}: expected {expected!r}, got {actual!r}")

    if not validation_log.exists():
        issues.append(f"{validation_log.name} is missing")
    else:
        text = validation_log.read_text(encoding="utf-8", errors="replace")
        for label, expected in (
            ("validation log model_path", expected_model_path),
            ("validation log backend", expected_backend),
            ("validation log precision_label", expected_precision),
            ("validation log manifest_path", expected_manifest),
        ):
            if expected and expected not in text:
                issues.append(f"{label}: {expected!r} not found in {validation_log.name}")

    if issues:
        raise RuntimeError("Backfilled YOLO11s run is still inconsistent:\n- " + "\n- ".join(issues))


def _read_per_class_metrics_csv(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = str(row.get("class_name", "")).strip()
            if not class_name:
                continue
            rows[class_name] = {
                "iou": float(row.get("iou") or 0.0),
                "dice": float(row.get("dice") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "precision": float(row.get("precision") or 0.0),
                "support_pixels": float(row.get("support_pixels") or 0.0),
            }
    return rows


def _metrics_close(actual: Optional[float], expected: Optional[float], tol: float = 1e-12) -> bool:
    if actual is None or expected is None:
        return actual == expected
    return abs(float(actual) - float(expected)) <= tol


def _validate_generated_accuracy_artifacts(run_root: Path, split_name: str, accuracy_metrics: Dict[str, Any]) -> None:
    accuracy_dir = run_root / "accuracy"
    val_dir = accuracy_dir / f"val_{split_name}"
    top_metrics_path = accuracy_dir / "val_metrics.json"
    nested_metrics_path = val_dir / "val_metrics.json"
    metrics_json_path = val_dir / "metrics.json"
    report_json_path = val_dir / "report.json"
    issues: List[str] = []

    for required in (top_metrics_path, nested_metrics_path, metrics_json_path, report_json_path):
        if not required.exists():
            issues.append(f"Required accuracy artifact missing: {required}")
    if issues:
        raise RuntimeError("Backfilled YOLO11s run is missing generated accuracy artifacts:\n- " + "\n- ".join(issues))

    top_metrics = _load_json(top_metrics_path)
    nested_metrics = _load_json(nested_metrics_path)
    metrics_json = _load_json(metrics_json_path)
    report_json = _load_json(report_json_path)

    if top_metrics != nested_metrics:
        issues.append("accuracy/val_metrics.json does not exactly match accuracy/val_test/val_metrics.json")

    for label, actual, expected in (
        ("metrics.json mAP50_95", metrics_json.get("mAP50_95"), accuracy_metrics.get("metrics/mAP50-95(B)", accuracy_metrics.get("metrics/mAP50-95"))),
        ("metrics.json mAP50", metrics_json.get("mAP50"), accuracy_metrics.get("metrics/mAP50(B)", accuracy_metrics.get("metrics/mAP50"))),
        ("report.json metrics.mAP50_95", (report_json.get("metrics") or {}).get("mAP50_95"), accuracy_metrics.get("metrics/mAP50-95(B)", accuracy_metrics.get("metrics/mAP50-95"))),
        ("report.json metrics.mAP50", (report_json.get("metrics") or {}).get("mAP50"), accuracy_metrics.get("metrics/mAP50(B)", accuracy_metrics.get("metrics/mAP50"))),
    ):
        if not _metrics_close(actual, expected):
            issues.append(f"{label}: expected {expected!r}, got {actual!r}")

    for artifact in (nested_metrics_path, metrics_json_path, report_json_path, accuracy_dir / f"validation_{split_name}.log"):
        text = artifact.read_text(encoding="utf-8", errors="replace")
        if "source validation" in text:
            issues.append(f"Legacy source-validation note found in current artifact: {artifact}")

    if issues:
        raise RuntimeError("Generated YOLO11s accuracy artifacts are inconsistent:\n- " + "\n- ".join(issues))


def _refresh_repeat_accuracy_note_local(repeat_summary: Dict[str, Any], final_payload: Dict[str, Any]) -> None:
    notes_raw = repeat_summary.get("notes")
    notes = [str(item) for item in notes_raw] if isinstance(notes_raw, list) else []
    notes = [note for note in notes if not note.startswith("Accuracy backend:")]

    final_model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
    accuracy_metrics = final_payload.get("accuracy_metrics", {}) if isinstance(final_payload.get("accuracy_metrics", {}), dict) else {}
    accuracy_backend = str(final_model.get("accuracy_backend") or "").strip()
    accuracy_precision = str(final_model.get("accuracy_precision_label") or final_model.get("precision_label") or "").strip()
    map50_95 = accuracy_metrics.get("metrics/mAP50-95(B)", accuracy_metrics.get("metrics/mAP50-95"))
    if accuracy_backend:
        suffix: List[str] = []
        if accuracy_precision:
            suffix.append(f"precision={accuracy_precision}")
        if map50_95 is not None:
            try:
                suffix.append(f"mAP50-95={float(map50_95):.3f}")
            except Exception:
                pass
        note = f"Accuracy backend: precision-specific {accuracy_backend} validation"
        if suffix:
            note += "; " + "; ".join(suffix)
        note += "."
        notes.insert(0, note)
    repeat_summary["notes"] = notes


def _refresh_run_repeat_artifacts(run_root: Path) -> None:
    final_payload = _load_json(run_root / "FINAL_SUMMARY.json")
    final_model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
    final_dataset = final_payload.get("dataset", {}) if isinstance(final_payload.get("dataset", {}), dict) else {}
    for repeat_root in sorted((run_root / "benchmark_runs").glob("repeat_*")):
        repeat_summary_path = repeat_root / "repeat_summary.json"
        if not repeat_summary_path.exists():
            continue
        repeat_summary = _load_json(repeat_summary_path)
        files = repeat_summary.get("files")
        if not isinstance(files, dict):
            files = {}
            repeat_summary["files"] = files

        merged_trace = repeat_root / "merged_trace.csv"
        per_image_metrics = repeat_root / "per_image_metrics.csv"
        agreement_path = repeat_root / "agreement_summary.json"
        external_csvs = sorted(str(p) for p in (repeat_root / "external").rglob("external_measurements.csv"))
        external_archives = sorted(str(p) for p in (repeat_root / "external").rglob("*.tar.gz"))

        repeat_summary["precision_label"] = str(final_model.get("precision_label") or final_model.get("accuracy_precision_label") or repeat_summary.get("precision_label") or "").lower() or None
        repeat_summary["model_path"] = final_model.get("measured_model_path")
        repeat_summary["dataset_root"] = final_dataset.get("root")
        repeat_summary["manifest"] = final_dataset.get("manifest") or final_dataset.get("accuracy_manifest") or final_dataset.get("benchmark_manifest")
        repeat_summary["measured_model_path"] = final_model.get("measured_model_path")
        repeat_summary["measured_model_sha256"] = final_model.get("measured_model_sha256")
        repeat_summary["accuracy_backend"] = final_model.get("accuracy_backend")
        repeat_summary["accuracy_precision_label"] = final_model.get("accuracy_precision_label")
        repeat_summary["merged_trace_csv"] = str(merged_trace) if merged_trace.exists() else None
        repeat_summary["per_image_metrics_csv"] = str(per_image_metrics) if per_image_metrics.exists() else None
        repeat_summary["agreement_summary_path"] = str(agreement_path) if agreement_path.exists() else None
        repeat_summary["external_artifact_path"] = external_archives[0] if external_archives else None
        repeat_summary["external_artifact_paths"] = external_archives or None
        _refresh_repeat_accuracy_note_local(repeat_summary, final_payload)

        files["merged_trace_csv"] = str(merged_trace) if merged_trace.exists() else None
        files["per_image_metrics_csv"] = str(per_image_metrics) if per_image_metrics.exists() else None
        files["agreement_summary_json"] = str(agreement_path) if agreement_path.exists() else None
        files["external_csv"] = external_csvs[0] if external_csvs else files.get("external_csv")
        files["external_archive_path"] = external_archives[0] if external_archives else files.get("external_archive_path")
        files["external_archive_paths"] = external_archives or files.get("external_archive_paths")
        files["measured_model_path"] = final_model.get("measured_model_path")
        files["measured_model_sha256"] = final_model.get("measured_model_sha256")

        _write_json(repeat_summary_path, repeat_summary)


def _repeat_summaries_look_current(run_root: Path) -> bool:
    repeat_roots = sorted((run_root / "benchmark_runs").glob("repeat_*"))
    if not repeat_roots:
        return False
    is_per_image_run = any(parent.name == "per_image" for parent in run_root.parents)
    for repeat_root in repeat_roots:
        summary_path = repeat_root / "repeat_summary.json"
        if not summary_path.exists():
            return False
        payload = _load_json(summary_path)
        precision_label = str(payload.get("precision_label") or "").strip().lower()
        measured_model_path = str(payload.get("measured_model_path") or "")
        model_path = str(payload.get("model_path") or "")
        dataset_root = str(payload.get("dataset_root") or "")
        manifest = str(payload.get("manifest") or "")
        merged_trace_csv = str(payload.get("merged_trace_csv") or "")
        notes = payload.get("notes")
        if precision_label not in {"fp16", "fp32", "int8"}:
            return False
        if not measured_model_path:
            return False
        if not model_path or not dataset_root or not manifest:
            return False
        if not merged_trace_csv:
            return False
        if is_per_image_run:
            per_image_metrics_csv = str(payload.get("per_image_metrics_csv") or "")
            if not per_image_metrics_csv:
                return False
        if isinstance(notes, list):
            note_text = "\n".join(str(item) for item in notes)
            if "source validation" in note_text:
                return False
    return True


def _validate_repeat_summaries(run_root: Path) -> None:
    if not _repeat_summaries_look_current(run_root):
        raise RuntimeError(f"Repeat summaries remain stale or incomplete after refresh: {run_root}")


def _current_accuracy_artifacts_look_current(run_root: Path, split_name: str, accuracy_metrics: Dict[str, Any]) -> bool:
    try:
        _validate_generated_accuracy_artifacts(run_root, split_name, accuracy_metrics)
        return True
    except Exception:
        return False


def _validate_variant_report(variant_root: Path, run_roots: List[Path]) -> None:
    report_csv = variant_root / "reports" / "proposal_comparison_table.csv"
    report_json = variant_root / "reports" / "proposal_comparison_table.json"
    if not report_csv.exists() or not report_json.exists():
        raise RuntimeError(f"Variant report outputs are missing under {variant_root / 'reports'}")

    rows_by_run: Dict[str, Dict[str, str]] = {}
    with report_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_run[str(row.get("run_root") or "")] = row

    payload = _load_json(report_json)
    if isinstance(payload, dict):
        payload_rows = payload.get("rows")
    elif isinstance(payload, list):
        payload_rows = payload
    else:
        payload_rows = None
    if not isinstance(payload_rows, list):
        raise RuntimeError(f"proposal_comparison_table.json does not contain a rows list: {report_json}")
    if len(payload_rows) != len(run_roots):
        raise RuntimeError(
            f"proposal_comparison_table.json row count mismatch for {variant_root}: "
            f"expected {len(run_roots)}, got {len(payload_rows)}"
        )

    issues: List[str] = []
    for run_root in run_roots:
        final_summary = _load_json(run_root / "FINAL_SUMMARY.json")
        accuracy_metrics = final_summary.get("accuracy_metrics", {}) if isinstance(final_summary.get("accuracy_metrics", {}), dict) else {}
        row = rows_by_run.get(str(run_root))
        if row is None:
            issues.append(f"proposal_comparison_table.csv missing run_root row: {run_root}")
            continue
        for field, expected in (
            ("mAP50_95", accuracy_metrics.get("metrics/mAP50-95(B)", accuracy_metrics.get("metrics/mAP50-95"))),
        ):
            actual = row.get(field)
            if actual in {None, ""} or not _metrics_close(float(actual), expected):
                issues.append(f"{run_root.name} report {field}: expected {expected!r}, got {actual!r}")

    if issues:
        raise RuntimeError("Variant comparison report is inconsistent with FINAL_SUMMARY accuracy:\n- " + "\n- ".join(issues))


def _is_existing_run_consistent(run_root: Path, measured_model_path: Path, model: Dict[str, Any]) -> bool:
    expected_model_path = _normalize_path_str(measured_model_path)
    expected_model_sha = sha256_file(measured_model_path)
    expected_backend = "tensorrt_engine"
    expected_precision = (model.get("precision_label") or "").lower()
    if not expected_precision:
        expected_precision = measured_model_path.stem.lower()

    config_path = run_root / "config_snapshot.yaml"
    validation_logs = sorted((run_root / "accuracy").glob("validation_*.log"))
    if not config_path.is_file() or not validation_logs:
        return False

    cfg = _load_yaml(config_path)
    cfg_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if _normalize_path_str(cfg_model.get("accuracy_model_path")) != expected_model_path:
        return False
    if (cfg_model.get("accuracy_model_sha256") or "").strip() != expected_model_sha:
        return False
    if cfg_model.get("accuracy_backend") != expected_backend:
        return False
    if (cfg_model.get("accuracy_precision_label") or "").lower() != expected_precision:
        return False

    validation_text = validation_logs[0].read_text(encoding="utf-8", errors="replace")
    return (
        expected_model_path in validation_text
        and expected_backend in validation_text
        and expected_precision in validation_text.lower()
    )


def _refresh_variant_reports(variant_root: Path, run_roots: List[Path], dry_run: bool) -> Optional[Dict[str, Any]]:
    if not run_roots:
        return None
    reports_root = variant_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(sys.executable).resolve()), str(COMPARE_RUNS), "--outdir", str(reports_root)]
    for run_root in run_roots:
        cmd.extend(["--run-root", str(run_root)])
    if dry_run:
        return {"cmd": cmd}

    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = (cp.stdout or "").strip()
    payload = json.loads(stdout) if stdout else {}
    (reports_root / "compare_runs_output.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill precision-specific YOLO11s accuracy into existing benchmark sessions.")
    ap.add_argument("--session-root", action="append", required=True, help="Repeat for multiple session roots; cached engine results are reused across them.")
    ap.add_argument("--variant", choices=("standard", "per_image"), default=None)
    ap.add_argument("--precision", action="append", choices=("fp16", "fp32", "int8"), default=None)
    ap.add_argument("--force", action="store_true", help="Re-run accuracy even if FINAL_SUMMARY already looks precision-specific.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    accuracy_cache: Dict[tuple[str, str, str, int], Dict[str, Any]] = {}
    session_roots = [Path(raw).expanduser().resolve() for raw in args.session_root]
    selected_precisions = set(args.precision or [])

    summary: Dict[str, Any] = {
        "session_roots": [str(path) for path in session_roots],
        "selected_precisions": sorted(selected_precisions) if selected_precisions else "all",
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "updated_runs": [],
        "refreshed_reports": [],
    }

    for session_root in session_roots:
        variant_roots = _discover_variant_roots(session_root, args.variant)
        for variant_root in variant_roots:
            run_roots = _discover_run_roots(variant_root)
            refreshed_run_roots: List[Path] = list(run_roots)
            for run_root in run_roots:
                final_path = run_root / "FINAL_SUMMARY.json"
                final_payload = _load_json(final_path)
                model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
                measured_model_path = _resolve_existing_layout_path(Path(str(model.get("measured_model_path", ""))))
                if measured_model_path.suffix.lower() != ".engine":
                    continue
                run_precision = _infer_run_precision(final_payload, measured_model_path)
                if selected_precisions and run_precision not in selected_precisions:
                    continue

                already_precision_specific = bool(model.get("accuracy_precision_specific"))
                already_matches_measured = (
                    _normalize_path_str(model.get("accuracy_model_path")) == _normalize_path_str(measured_model_path)
                    and (model.get("accuracy_model_sha256") or "").strip() == sha256_file(measured_model_path)
                )
                current_accuracy_metrics = (
                    final_payload.get("accuracy_metrics", {})
                    if isinstance(final_payload.get("accuracy_metrics", {}), dict)
                    else {}
                )
                split_name = _infer_split_name(run_root, final_payload)
                artifacts_current = _current_accuracy_artifacts_look_current(run_root, split_name, current_accuracy_metrics)
                repeat_summaries_current = _repeat_summaries_look_current(run_root)
                if (
                    already_precision_specific
                    and already_matches_measured
                    and _is_existing_run_consistent(run_root, measured_model_path, model)
                    and artifacts_current
                    and repeat_summaries_current
                    and not args.force
                ):
                    summary["updated_runs"].append(
                        {
                            "session_root": str(session_root),
                            "run_root": str(run_root),
                            "status": "skipped_already_precision_specific",
                        }
                    )
                    continue

                dataset_root = _resolve_existing_dataset_root(
                    Path(str(final_payload.get("dataset", {}).get("root", ""))),
                    measured_model_path,
                )
                imgsz = int((model.get("profile_input", {}) or {}).get("imgsz", 640))

                entry: Dict[str, Any] = {
                    "session_root": str(session_root),
                    "run_root": str(run_root),
                    "measured_model_path": str(measured_model_path),
                    "dataset_root": str(dataset_root),
                    "split_name": split_name,
                    "imgsz": imgsz,
                }

                if args.dry_run:
                    entry["status"] = "planned"
                    summary["updated_runs"].append(entry)
                    continue

                _cleanup_accuracy_output(run_root, split_name)
                cache_key = (str(measured_model_path), str(dataset_root), split_name, imgsz)
                if cache_key in accuracy_cache:
                    cache_entry = accuracy_cache[cache_key]
                    accuracy_metrics = cache_entry["metrics"]
                    _copy_accuracy_artifacts(Path(cache_entry["source_run_root"]), run_root, split_name)
                    entry["accuracy_eval"] = "reused_cached_result"
                else:
                    accuracy_metrics = _run_engine_accuracy(
                        engine_path=measured_model_path,
                        dataset_root=dataset_root,
                        split_name=split_name,
                        imgsz=imgsz,
                        run_root=run_root,
                    )
                    accuracy_cache[cache_key] = {
                        "metrics": accuracy_metrics,
                        "source_run_root": str(run_root),
                    }
                    entry["accuracy_eval"] = "executed"
                _update_final_summary(run_root, final_payload, accuracy_metrics, split_name)
                _update_config_snapshot(run_root, measured_model_path, accuracy_metrics)
                _refresh_run_repeat_artifacts(run_root)
                _validate_updated_run(run_root, measured_model_path, accuracy_metrics, split_name)
                _validate_generated_accuracy_artifacts(run_root, split_name, accuracy_metrics)
                _validate_repeat_summaries(run_root)

                entry["status"] = "updated"
                entry["accuracy_backend"] = accuracy_metrics.get("backend")
                entry["mAP50_95"] = accuracy_metrics.get("metrics/mAP50-95(B)", accuracy_metrics.get("metrics/mAP50-95"))
                entry["mAP50"] = accuracy_metrics.get("metrics/mAP50(B)", accuracy_metrics.get("metrics/mAP50"))
                summary["updated_runs"].append(entry)

            refresh_payload = _refresh_variant_reports(variant_root, refreshed_run_roots, args.dry_run)
            summary["refreshed_reports"].append(
                {
                    "session_root": str(session_root),
                    "variant_root": str(variant_root),
                    "run_count": len(refreshed_run_roots),
                    "result": refresh_payload,
                }
            )
            if not args.dry_run:
                _validate_variant_report(variant_root, refreshed_run_roots)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
