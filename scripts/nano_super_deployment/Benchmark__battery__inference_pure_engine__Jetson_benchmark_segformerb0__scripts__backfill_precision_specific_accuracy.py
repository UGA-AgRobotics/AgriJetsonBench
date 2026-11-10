#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PKG_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ACCURACY_HELPER = PKG_ROOT / "scripts" / "run_segformerb0_engine_accuracy.py"
COMPARE_RUNS = PKG_ROOT / "analysis" / "compare_runs.py"


if __package__ in {None, ""}:
    sys.path.append(str(PKG_ROOT))
    from jetson.utils import dump_json, sha256_file
else:  # pragma: no cover
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
                (PKG_ROOT.parent / "segformerb0" / "inference_bundle" / "model" / path.name).resolve(),
                (PKG_ROOT.parent / "segformerb0" / "inference_bundle" / "dataset" / path.name).resolve(),
                (PKG_ROOT.parent / "segformerb0" / "model" / path.name).resolve(),
                (PKG_ROOT.parent / "segformerb0" / "dataset" / path.name).resolve(),
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
        "--engine",
        str(engine_path),
        "--dataset-root",
        str(dataset_root),
        "--split-name",
        str(split_name),
        "--imgsz",
        str(int(imgsz)),
        "--output-dir",
        str(output_dir),
        "--out-json",
        str(out_json),
    ]
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    combined_output = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
    validation_log.write_text(combined_output + "\n", encoding="utf-8")
    if cp.returncode != 0:
        raise RuntimeError(
            "SegFormer-B0 precision-specific backfill accuracy failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{combined_output[-2000:]}"
        )
    return _load_json(out_json)


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
    backup_path = run_root / "FINAL_SUMMARY.pre_precision_accuracy_backup.json"
    if not backup_path.exists():
        backup_path.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")

    model = final_payload.setdefault("model", {})
    files = final_payload.setdefault("files", {})
    measured_model_path = _resolve_existing_layout_path(Path(str(model["measured_model_path"])))
    accuracy_dir = run_root / "accuracy"
    output_dir = accuracy_dir / f"val_{split_name}"

    model["accuracy_model_path"] = str(measured_model_path)
    model["accuracy_model_sha256"] = sha256_file(measured_model_path)
    model["accuracy_backend"] = accuracy_metrics.get("backend")
    model["accuracy_backend_detail"] = accuracy_metrics.get("backend_detail")
    model["accuracy_precision_label"] = accuracy_metrics.get("precision_label")
    model["accuracy_precision_specific"] = True
    final_payload["accuracy_metrics"] = accuracy_metrics

    metrics_json = accuracy_dir / "val_metrics.json"
    report_json = output_dir / "report.json"
    raw_metrics_json = output_dir / "metrics.json"
    validation_log = accuracy_dir / f"validation_{split_name}.log"

    files["accuracy_metrics_json"] = str(metrics_json) if metrics_json.exists() else None
    files["accuracy_report_json"] = str(report_json) if report_json.exists() else None
    files["accuracy_raw_metrics_json"] = str(raw_metrics_json) if raw_metrics_json.exists() else None
    files["accuracy_validation_log"] = str(validation_log) if validation_log.exists() else None

    _write_json(final_path, final_payload)


def _update_config_snapshot(run_root: Path, measured_model_path: Path, accuracy_metrics: Dict[str, Any]) -> None:
    config_path = run_root / "config_snapshot.yaml"
    if not config_path.exists():
        return
    cfg = _load_yaml(config_path)
    model = cfg.setdefault("model", {})
    model["accuracy_model_path"] = str(measured_model_path)
    model["accuracy_model_sha256"] = sha256_file(measured_model_path)
    model["accuracy_backend"] = accuracy_metrics.get("backend")
    model["accuracy_backend_detail"] = accuracy_metrics.get("backend_detail")
    model["accuracy_precision_label"] = accuracy_metrics.get("precision_label")
    model["accuracy_precision_specific"] = True
    _write_yaml(config_path, cfg)


def _validate_updated_run(run_root: Path, measured_model_path: Path, accuracy_metrics: Dict[str, Any], split_name: str) -> None:
    expected_model_path = _normalize_path_str(measured_model_path)
    expected_model_sha = sha256_file(measured_model_path)
    expected_backend = accuracy_metrics.get("backend")
    expected_precision = accuracy_metrics.get("precision_label")
    issues: List[str] = []

    final_payload = _load_json(run_root / "FINAL_SUMMARY.json")
    final_model = final_payload.get("model", {}) if isinstance(final_payload.get("model", {}), dict) else {}
    config_payload = _load_yaml(run_root / "config_snapshot.yaml") if (run_root / "config_snapshot.yaml").exists() else {}
    config_model = config_payload.get("model", {}) if isinstance(config_payload.get("model", {}), dict) else {}
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
        ):
            if expected and expected not in text:
                issues.append(f"{label}: {expected!r} not found in {validation_log.name}")

    if issues:
        raise RuntimeError("Backfilled SegFormer-B0 run is still inconsistent:\n- " + "\n- ".join(issues))


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
    ap = argparse.ArgumentParser(description="Backfill precision-specific SegFormer-B0 accuracy into existing benchmark sessions.")
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
                if already_precision_specific and already_matches_measured and _is_existing_run_consistent(run_root, measured_model_path, model) and not args.force:
                    summary["updated_runs"].append(
                        {
                            "session_root": str(session_root),
                            "run_root": str(run_root),
                            "status": "skipped_already_precision_specific",
                        }
                    )
                    continue

                split_name = _infer_split_name(run_root, final_payload)
                dataset_root = _resolve_existing_layout_path(Path(str(final_payload.get("dataset", {}).get("root", ""))))
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
                _validate_updated_run(run_root, measured_model_path, accuracy_metrics, split_name)

                entry["status"] = "updated"
                entry["accuracy_backend"] = accuracy_metrics.get("backend")
                entry["mIoU"] = accuracy_metrics.get("metrics/mIoU")
                entry["mDice"] = accuracy_metrics.get("metrics/mDice")
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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
