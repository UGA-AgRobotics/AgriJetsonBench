#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = load_yaml(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def create_or_replace_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink(dst)
    try:
        dst.symlink_to(src)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def copy_or_link(src: Path, dst: Path, mode: str = "copy_if_small", small_mb: int = 50) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink(dst)
    size_bytes = src.stat().st_size
    if mode == "copy":
        shutil.copy2(src, dst)
        used_mode = "copy"
    elif mode == "link":
        create_or_replace_link(src, dst)
        used_mode = "link"
    else:
        if size_bytes <= small_mb * 1024 * 1024:
            shutil.copy2(src, dst)
            used_mode = "copy"
        else:
            create_or_replace_link(src, dst)
            used_mode = "link"
    return {"src": str(src), "dst": str(dst), "mode": used_mode, "size_bytes": size_bytes}


def line_count_excluding_header(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def normalize_names(raw: Any) -> Dict[int, str]:
    if isinstance(raw, dict):
        normalized: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                normalized[int(k)] = str(v)
            except Exception:
                continue
        return dict(sorted(normalized.items()))
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    raise TypeError(f"Unsupported names object: {type(raw)}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_subprocess(cmd: List[str], log: logging.Logger, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        log.info("stdout:\n%s", result.stdout.strip())
    if result.stderr:
        log.info("stderr:\n%s", result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


@dataclass
class Paths:
    dataset_root: Path
    runtime_root: Path
    state_dir: Path
    state_file: Path
    logs_dir: Path
    env_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    exports_dir: Path
    runtime_dataset_dir: Path
    runtime_manifests_dir: Path
    source_snapshot_dir: Path
    paper_assets_dir: Path
    validation_root: Path
    qualitative_root: Path
    ultralytics_project_dir: Path
    ultralytics_run_dir: Path
    bundle_dir: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Paths":
        dataset_root = Path(cfg["dataset"]["dataset_root"]).expanduser().resolve()
        runtime_root = Path(cfg["runtime"]["root"]).expanduser().resolve()
        state_dir = runtime_root / "state"
        return Paths(
            dataset_root=dataset_root,
            runtime_root=runtime_root,
            state_dir=state_dir,
            state_file=state_dir / "pipeline_state.json",
            logs_dir=runtime_root / "logs",
            env_dir=runtime_root / "environment",
            reports_dir=runtime_root / "reports",
            artifacts_dir=runtime_root / "artifacts",
            exports_dir=runtime_root / "exports",
            runtime_dataset_dir=runtime_root / "runtime_dataset",
            runtime_manifests_dir=runtime_root / "runtime_dataset" / "manifests",
            source_snapshot_dir=runtime_root / "source_snapshot",
            paper_assets_dir=runtime_root / "paper_assets",
            validation_root=runtime_root / "validation_runs",
            qualitative_root=runtime_root / "qualitative_predictions",
            ultralytics_project_dir=runtime_root / "ultralytics",
            ultralytics_run_dir=runtime_root / "ultralytics" / cfg["training"]["run_name"],
            bundle_dir=runtime_root / "bundles",
        )


class StateManager:
    def __init__(self, path: Path):
        self.path = path
        self.state = read_json(path)
        if not self.state:
            self.state = {
                "schema_version": 2,
                "created_at_utc": utc_now_iso(),
                "stages": {},
            }
            self.save()

    def save(self) -> None:
        atomic_json_dump(self.state, self.path)

    def stage(self, name: str) -> Dict[str, Any]:
        return self.state.setdefault("stages", {}).setdefault(name, {})

    def is_done(self, name: str) -> bool:
        return self.stage(name).get("status") == "done"

    def mark_started(self, name: str) -> None:
        stage = self.stage(name)
        stage["status"] = "running"
        stage["started_at_utc"] = utc_now_iso()
        self.save()

    def mark_done(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        stage = self.stage(name)
        stage["status"] = "done"
        stage["finished_at_utc"] = utc_now_iso()
        if extra:
            stage.update(extra)
        self.save()

    def mark_failed(self, name: str, exc: Exception) -> None:
        stage = self.stage(name)
        stage["status"] = "failed"
        stage["finished_at_utc"] = utc_now_iso()
        stage["last_error"] = str(exc)
        self.save()


class EpochProgressMonitor:
    def __init__(self, results_csv: Path, total_epochs: int, desc: str):
        self.results_csv = results_csv
        self.total_epochs = total_epochs
        self.desc = desc
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._pbar: Optional[tqdm] = None
        self._last_count = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._pbar is not None:
            self._pbar.close()

    def _run(self) -> None:
        self._pbar = tqdm(total=self.total_epochs, desc=self.desc, unit="epoch", dynamic_ncols=True)
        while not self._stop.is_set():
            done = line_count_excluding_header(self.results_csv)
            if done > self._last_count:
                self._pbar.update(done - self._last_count)
                self._last_count = done
            time.sleep(10)
        done = line_count_excluding_header(self.results_csv)
        if done > self._last_count:
            self._pbar.update(done - self._last_count)
            self._last_count = done


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(f"stage2_yolo11s_v1_{int(time.time())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def capture_environment(paths: Paths, log: logging.Logger, mode_tag: str) -> None:
    target_dir = ensure_dir(paths.env_dir / f"capture_{mode_tag}")
    env_snapshot: Dict[str, Any] = {
        "captured_at_utc": utc_now_iso(),
        "mode_tag": mode_tag,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
    }

    try:
        import torch  # type: ignore
        env_snapshot["torch_version"] = torch.__version__
        env_snapshot["cuda_available"] = bool(torch.cuda.is_available())
        env_snapshot["torch_cuda_version"] = torch.version.cuda
        env_snapshot["cudnn_version"] = torch.backends.cudnn.version()
        env_snapshot["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            env_snapshot["gpu_name"] = torch.cuda.get_device_name(0)
            env_snapshot["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        env_snapshot["torch_probe_error"] = str(exc)

    for pkg_name in ("ultralytics", "pandas", "onnx", "yaml"):
        try:
            module = __import__(pkg_name if pkg_name != "yaml" else "yaml")
            env_snapshot[f"{pkg_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            env_snapshot[f"{pkg_name}_probe_error"] = str(exc)

    atomic_json_dump(env_snapshot, target_dir / "environment_snapshot.json")

    slurm_env = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    atomic_json_dump(slurm_env, target_dir / "slurm_environment.json")

    commands = {
        "uname.txt": ["bash", "-lc", "uname -a || true"],
        "lscpu.txt": ["bash", "-lc", "lscpu || true"],
        "module_list.txt": ["bash", "-lc", "module list 2>&1 || ml list 2>&1 || true"],
        "nvidia_smi_L.txt": ["bash", "-lc", "nvidia-smi -L || true"],
        "nvidia_smi.txt": ["bash", "-lc", "nvidia-smi || true"],
        "nvidia_smi_q.txt": ["bash", "-lc", "nvidia-smi -q || true"],
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "df_h.txt": ["bash", "-lc", "df -h || true"],
        "env_sorted.txt": ["bash", "-lc", "env | sort || true"],
    }

    for name, cmd in commands.items():
        try:
            result = run_subprocess(cmd, log, check=False)
            write_text(target_dir / name, result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
        except Exception as exc:
            write_text(target_dir / name, f"FAILED: {exc}\n")

    if os.environ.get("SLURM_JOB_ID"):
        try:
            result = run_subprocess(["bash", "-lc", f"scontrol show job {os.environ['SLURM_JOB_ID']} || true"], log, check=False)
            write_text(target_dir / "scontrol_show_job.txt", result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
        except Exception as exc:
            write_text(target_dir / "scontrol_show_job.txt", f"FAILED: {exc}\n")

    py_probe = {
        "sys_path": sys.path,
        "package_root": str(PACKAGE_ROOT),
    }
    atomic_json_dump(py_probe, target_dir / "python_paths.json")


def archive_source_snapshot(paths: Paths, cfg_path: Path, log: logging.Logger) -> Dict[str, Any]:
    target = ensure_dir(paths.source_snapshot_dir)
    src_root = PACKAGE_ROOT
    selected = []
    for rel in [
        Path("requirements.txt"),
        Path("config"),
        Path("docs"),
        Path("scripts"),
    ]:
        src = src_root / rel
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    selected.append(p)
        elif src.is_file():
            selected.append(src)

    copied = []
    for src in sorted(set(selected)):
        rel = src.relative_to(src_root)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({
            "relative_path": str(rel),
            "sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size,
        })

    # also save exact config used
    if cfg_path.exists():
        shutil.copy2(cfg_path, target / "config_used.yaml")

    atomic_json_dump({
        "captured_at_utc": utc_now_iso(),
        "package_root": str(src_root),
        "file_count": len(copied),
        "files": copied,
    }, target / "source_snapshot_manifest.json")

    rows = [{"relative_path": x["relative_path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]} for x in copied]
    pd.DataFrame(rows).to_csv(target / "source_snapshot_manifest.csv", index=False)
    log.info("Saved source snapshot to %s", target)
    return {"source_snapshot_dir": str(target), "file_count": len(copied)}


def deterministic_subset(items: List[str], k: int, seed: int) -> List[str]:
    if k >= len(items):
        return list(items)
    rng = random.Random(seed)
    chosen = set(rng.sample(items, k))
    return [x for x in items if x in chosen]


def parse_yolo_label_file(path: Path) -> Dict[str, Any]:
    class_ids: List[int] = []
    parse_errors = 0
    line_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            line_count += 1
            parts = raw.split()
            if len(parts) < 5:
                parse_errors += 1
                continue
            try:
                class_ids.append(int(float(parts[0])))
            except Exception:
                parse_errors += 1
    return {
        "line_count": line_count,
        "class_ids": class_ids,
        "parse_errors": parse_errors,
    }


def build_runtime_manifests(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "prepare_runtime_dataset"
    report_path = paths.reports_dir / "dataset_runtime_report.json"
    if state.is_done(stage_name) and report_path.exists():
        required_outputs = [
            paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml",
            paths.runtime_dataset_dir / "CottonWeedDet12_runtime_smoke.yaml",
            paths.runtime_manifests_dir / "train_full.txt",
            paths.runtime_manifests_dir / "val_full.txt",
            paths.runtime_manifests_dir / "test_full.txt",
            paths.runtime_manifests_dir / "train_smoke.txt",
            paths.runtime_manifests_dir / "val_smoke.txt",
        ]
        if all(p.exists() for p in required_outputs):
            return read_json(report_path)
        log.warning("Prepare stage was marked done, but one or more runtime dataset files are missing. Rebuilding.")

    state.mark_started(stage_name)
    try:
        ensure_dir(paths.runtime_manifests_dir)
        ensure_dir(paths.runtime_dataset_dir)
        ensure_dir(paths.reports_dir)

        inventory_csv = Path(cfg["dataset"]["inventory_with_split_csv"]).expanduser().resolve()
        dataset_yaml_locked = Path(cfg["dataset"]["locked_ultralytics_yaml"]).expanduser().resolve()
        dataset_lock_yaml = Path(cfg["dataset"]["dataset_lock_yaml"]).expanduser().resolve()
        seed = int(cfg["training"]["seed"])

        if not inventory_csv.exists():
            raise FileNotFoundError(f"inventory_with_split.csv not found: {inventory_csv}")
        if not dataset_yaml_locked.exists():
            raise FileNotFoundError(f"locked dataset yaml not found: {dataset_yaml_locked}")
        if not dataset_lock_yaml.exists():
            raise FileNotFoundError(f"dataset lock yaml not found: {dataset_lock_yaml}")

        df = pd.read_csv(inventory_csv)
        if "image_id" not in df.columns or "split" not in df.columns:
            raise ValueError("inventory_with_split.csv must include at least image_id and split columns.")

        names_locked = normalize_names(load_yaml(dataset_yaml_locked)["names"])
        dataset_lock = load_yaml(dataset_lock_yaml)
        class_order = dataset_lock.get("class_order")
        if not class_order:
            class_order = [names_locked[i] for i in sorted(names_locked)]

        image_root = paths.dataset_root / "canonical_yolo" / "images"
        label_root = paths.dataset_root / "canonical_yolo" / "labels"

        split_to_paths: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        missing_images: List[str] = []
        missing_labels: List[str] = []
        parse_error_files: List[str] = []

        split_box_counts = Counter()
        split_empty_label_files = Counter()
        split_nonempty_label_files = Counter()
        class_box_counts: Dict[str, Counter] = defaultdict(Counter)
        class_image_counts: Dict[str, Counter] = defaultdict(Counter)

        progress = tqdm(df.itertuples(index=False), total=len(df), desc="Preflight dataset scan", unit="img", dynamic_ncols=True)
        for row in progress:
            image_id = getattr(row, "image_id")
            split = getattr(row, "split")
            if split not in split_to_paths:
                continue

            image_path = image_root / f"{image_id}.jpg"
            label_path = label_root / f"{image_id}.txt"

            if not image_path.exists():
                missing_images.append(str(image_path))
                continue
            if not label_path.exists():
                missing_labels.append(str(label_path))
                continue

            parsed = parse_yolo_label_file(label_path)
            if parsed["parse_errors"] > 0:
                parse_error_files.append(str(label_path))

            if parsed["line_count"] == 0:
                split_empty_label_files[split] += 1
            else:
                split_nonempty_label_files[split] += 1

            unique_in_image = set()
            for cls_id in parsed["class_ids"]:
                split_box_counts[split] += 1
                class_box_counts[split][cls_id] += 1
                unique_in_image.add(cls_id)

            for cls_id in unique_in_image:
                class_image_counts[split][cls_id] += 1

            split_to_paths[split].append(str(image_path))
        progress.close()

        if missing_images or missing_labels:
            problem = {
                "missing_images": missing_images[:100],
                "missing_labels": missing_labels[:100],
                "missing_image_count": len(missing_images),
                "missing_label_count": len(missing_labels),
            }
            atomic_json_dump(problem, report_path)
            raise RuntimeError(
                f"Dataset runtime check failed: {len(missing_images)} missing images and {len(missing_labels)} missing labels."
            )

        actual_counts = {split: len(paths_list) for split, paths_list in split_to_paths.items()}
        expected_counts = {k: int(v) for k, v in cfg["dataset"]["expected_split_counts"].items()}
        for split_name, expected in expected_counts.items():
            actual = actual_counts.get(split_name, -1)
            if actual != expected:
                raise RuntimeError(f"Split count mismatch for {split_name}: expected {expected}, got {actual}")

        manifest_paths: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            manifest_path = paths.runtime_manifests_dir / f"{split_name}_full.txt"
            write_text(manifest_path, "\n".join(split_to_paths[split_name]) + "\n")
            manifest_paths[split_name] = str(manifest_path)

        smoke_train = deterministic_subset(split_to_paths["train"], int(cfg["dataset"]["smoke_train_subset_size"]), seed + 11)
        smoke_val = deterministic_subset(split_to_paths["val"], int(cfg["dataset"]["smoke_val_subset_size"]), seed + 17)
        jetson_pilot_val = deterministic_subset(split_to_paths["val"], int(cfg["dataset"]["jetson_pilot_val_subset_size"]), seed + 29)
        paper_val_examples = deterministic_subset(split_to_paths["val"], int(cfg["qualitative"]["val_subset_size"]), seed + 37)
        paper_test_examples = deterministic_subset(split_to_paths["test"], int(cfg["qualitative"]["test_subset_size"]), seed + 41)

        smoke_train_manifest = paths.runtime_manifests_dir / "train_smoke.txt"
        smoke_val_manifest = paths.runtime_manifests_dir / "val_smoke.txt"
        jetson_pilot_manifest = paths.runtime_manifests_dir / "val_jetson_pilot.txt"
        paper_val_manifest = paths.runtime_manifests_dir / "paper_val_examples.txt"
        paper_test_manifest = paths.runtime_manifests_dir / "paper_test_examples.txt"

        write_text(smoke_train_manifest, "\n".join(smoke_train) + "\n")
        write_text(smoke_val_manifest, "\n".join(smoke_val) + "\n")
        write_text(jetson_pilot_manifest, "\n".join(jetson_pilot_val) + "\n")
        write_text(paper_val_manifest, "\n".join(paper_val_examples) + "\n")
        write_text(paper_test_manifest, "\n".join(paper_test_examples) + "\n")

        canonical_yolo_root = paths.dataset_root / "canonical_yolo"
        runtime_full_yaml = {
            "path": str(canonical_yolo_root),
            "train": str(paths.runtime_manifests_dir / "train_full.txt"),
            "val": str(paths.runtime_manifests_dir / "val_full.txt"),
            "test": str(paths.runtime_manifests_dir / "test_full.txt"),
            "names": names_locked,
            "nc": len(names_locked),
        }
        runtime_smoke_yaml = {
            "path": str(canonical_yolo_root),
            "train": str(smoke_train_manifest),
            "val": str(smoke_val_manifest),
            "test": str(smoke_val_manifest),
            "names": names_locked,
            "nc": len(names_locked),
        }

        runtime_full_yaml_path = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"
        runtime_smoke_yaml_path = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_smoke.yaml"
        dump_yaml(runtime_full_yaml, runtime_full_yaml_path)
        dump_yaml(runtime_smoke_yaml, runtime_smoke_yaml_path)

        write_text(paths.runtime_dataset_dir / "labels.txt", "\n".join(class_order) + "\n")

        jetson_policy_md = f"""# Jetson inference split policy

## Final reported Jetson benchmark
- split: `test_full.txt`
- number of images: {len(split_to_paths['test'])}
- use for: final accuracy, latency, power, energy, and temperature reporting on Jetson

## Quick board-side smoke checks
- split: `val_jetson_pilot.txt`
- number of images: {len(jetson_pilot_val)}
- use for: export sanity checks, parser checks, first-device wiring tests

## Important rule
Do **not** tune hyperparameters on the frozen test split.
"""
        write_text(paths.runtime_dataset_dir / "JETSON_INFERENCE_POLICY.md", jetson_policy_md)

        split_rows = []
        for split_name in ("train", "val", "test"):
            split_rows.append({
                "split": split_name,
                "image_count": actual_counts[split_name],
                "expected_image_count": expected_counts[split_name],
                "box_count": int(split_box_counts[split_name]),
                "nonempty_label_files": int(split_nonempty_label_files[split_name]),
                "empty_label_files": int(split_empty_label_files[split_name]),
            })
        pd.DataFrame(split_rows).to_csv(paths.reports_dir / "split_level_stats.csv", index=False)

        class_box_rows = []
        class_img_rows = []
        for split_name in ("train", "val", "test"):
            for cls_id, cls_name in names_locked.items():
                class_box_rows.append({
                    "split": split_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "box_count": int(class_box_counts[split_name][cls_id]),
                })
                class_img_rows.append({
                    "split": split_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "image_presence_count": int(class_image_counts[split_name][cls_id]),
                })
        pd.DataFrame(class_box_rows).to_csv(paths.reports_dir / "class_box_counts_by_split.csv", index=False)
        pd.DataFrame(class_img_rows).to_csv(paths.reports_dir / "class_image_presence_by_split.csv", index=False)

        dataset_summary_md = f"""# Dataset runtime summary

- dataset root: `{paths.dataset_root}`
- runtime full yaml: `{runtime_full_yaml_path}`
- runtime smoke yaml: `{runtime_smoke_yaml_path}`

## Locked split sizes
- train: {actual_counts['train']}
- val: {actual_counts['val']}
- test: {actual_counts['test']}

## Deterministic helper subsets
- smoke train: {len(smoke_train)}
- smoke val: {len(smoke_val)}
- Jetson pilot val: {len(jetson_pilot_val)}
- paper val examples: {len(paper_val_examples)}
- paper test examples: {len(paper_test_examples)}

## Notes
- final Jetson reporting later should use the frozen `test_full.txt`
- quick board-side checks should use `val_jetson_pilot.txt`
- class counts by split were computed directly from YOLO label files
"""
        write_text(paths.runtime_dataset_dir / "DATASET_RUNTIME_SUMMARY.md", dataset_summary_md)

        report = {
            "dataset_root": str(paths.dataset_root),
            "inventory_csv": str(inventory_csv),
            "runtime_full_yaml": str(runtime_full_yaml_path),
            "runtime_smoke_yaml": str(runtime_smoke_yaml_path),
            "full_manifest_paths": manifest_paths,
            "smoke_train_manifest": str(smoke_train_manifest),
            "smoke_val_manifest": str(smoke_val_manifest),
            "jetson_pilot_manifest": str(jetson_pilot_manifest),
            "paper_val_manifest": str(paper_val_manifest),
            "paper_test_manifest": str(paper_test_manifest),
            "actual_counts": actual_counts,
            "expected_counts": expected_counts,
            "split_box_counts": {k: int(v) for k, v in split_box_counts.items()},
            "parse_error_file_count": len(parse_error_files),
            "parse_error_files_preview": parse_error_files[:20],
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics could not be imported. "
            "Run scripts/setup_stage2_env.sh again, then submit the job with the refreshed environment. "
            f"Original error: {exc!r}"
        ) from exc
    return YOLO


def resolve_tuned_params_path(cfg: Dict[str, Any], paths: Paths) -> Path:
    configured = cfg["training"].get("tuned_params_yaml")
    if configured:
        return Path(configured).expanduser().resolve()
    return paths.runtime_root / "optuna" / "best_hyperparameters.yaml"


def load_tuned_params_if_available(cfg: Dict[str, Any], paths: Paths, log: logging.Logger) -> Dict[str, Any]:
    if not bool(cfg["training"].get("use_tuned_params_if_available", False)):
        return {}
    tuned_path = resolve_tuned_params_path(cfg, paths)
    if not tuned_path.exists():
        return {}
    try:
        tuned = load_yaml(tuned_path)
        if not isinstance(tuned, dict):
            log.warning("Tuned hyperparameter file exists but is not a mapping: %s", tuned_path)
            return {}
        log.info("Using tuned hyperparameters from %s", tuned_path)
        return tuned
    except Exception as exc:
        log.warning("Could not load tuned hyperparameters from %s: %s", tuned_path, exc)
        return {}


def write_effective_training_args(args: Dict[str, Any], paths: Paths) -> None:
    dump_yaml(args, paths.reports_dir / "effective_training_args.yaml")


def common_train_args(cfg: Dict[str, Any], runtime_yaml: Path, paths: Paths, log: logging.Logger) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "data": str(runtime_yaml),
        "imgsz": int(cfg["training"]["imgsz"]),
        "epochs": int(cfg["training"]["epochs"]),
        "batch": int(cfg["training"]["batch"]),
        "device": cfg["training"]["device"],
        "workers": int(cfg["training"]["workers"]),
        "seed": int(cfg["training"]["seed"]),
        "deterministic": bool(cfg["training"]["deterministic"]),
        "cache": bool(cfg["training"]["cache"]),
        "amp": bool(cfg["training"]["amp"]),
        "project": str(paths.ultralytics_project_dir),
        "name": str(cfg["training"]["run_name"]),
        "exist_ok": True,
        "save": True,
        "save_period": int(cfg["training"]["save_period"]),
        "plots": True,
        "verbose": True,
        "pretrained": bool(cfg["training"]["pretrained"]),
        "optimizer": cfg["training"]["optimizer"],
        "close_mosaic": int(cfg["training"]["close_mosaic"]),
        "single_cls": False,
    }
    if cfg["training"].get("patience") is not None:
        args["patience"] = int(cfg["training"]["patience"])
    if cfg["training"].get("cos_lr") is not None:
        args["cos_lr"] = bool(cfg["training"]["cos_lr"])

    tuned_overrides = load_tuned_params_if_available(cfg, paths, log)
    if tuned_overrides:
        allowed = {
            "lr0", "lrf", "weight_decay", "warmup_epochs", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "mosaic", "mixup"
        }
        for k, v in tuned_overrides.items():
            if k in allowed:
                args[k] = v

    write_effective_training_args(args, paths)
    return args


def parse_results_csv(results_csv: Path) -> Dict[str, Any]:
    if not results_csv.exists():
        return {}
    df = pd.read_csv(results_csv)
    if df.empty:
        return {}
    map_col = None
    for candidate in ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/fitness"]:
        if candidate in df.columns:
            map_col = candidate
            break

    best_epoch_index = None
    best_value = None
    best_row = None
    if map_col:
        numeric = pd.to_numeric(df[map_col], errors="coerce")
        if numeric.notna().any():
            best_epoch_index = int(numeric.idxmax())
            best_value = float(numeric.iloc[best_epoch_index])
            best_row = df.iloc[best_epoch_index].to_dict()

    last_row = df.iloc[-1].to_dict()
    return {
        "epochs_recorded": int(len(df)),
        "best_metric_column": map_col,
        "best_metric_value": best_value,
        "best_epoch_index": best_epoch_index,
        "best_row": best_row,
        "last_row": last_row,
    }


def important_file_metadata(path: Path) -> Dict[str, Any]:
    meta = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.exists():
        meta["size_bytes"] = path.stat().st_size
        meta["sha256"] = sha256_file(path)
        meta["md5"] = md5_file(path)
    return meta


def collect_known_artifacts(src_dir: Path, dst_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    ensure_dir(dst_dir)
    patterns = [
        "args.yaml",
        "results.csv",
        "results.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "train_batch*.jpg",
        "val_batch*.jpg",
        "weights/best.pt",
        "weights/last.pt",
    ]
    copied: List[Dict[str, Any]] = []
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                rel = src.relative_to(src_dir)
                dst = dst_dir / rel
                copied.append(copy_or_link(src, dst))
    atomic_json_dump({"source_dir": str(src_dir), "dest_dir": str(dst_dir), "files": copied}, dst_dir / "artifact_manifest.json")
    return {"dest_dir": str(dst_dir), "file_count": len(copied)}


def capture_model_info(model_path: Path, out_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    ensure_dir(out_dir)
    YOLO = import_ultralytics()
    model = YOLO(str(model_path))
    text_buffer = io.StringIO()
    with contextlib.redirect_stdout(text_buffer):
        try:
            model.info(detailed=False, verbose=True)
        except TypeError:
            try:
                model.info(verbose=True)
            except Exception as exc:
                print(f"model.info() failed: {exc}")
        except Exception as exc:
            print(f"model.info() failed: {exc}")

    info_txt = out_dir / "model_info.txt"
    write_text(info_txt, text_buffer.getvalue())
    names = {}
    try:
        names = dict(model.names) if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    except Exception:
        names = {}
    info_json = {
        "model_path": str(model_path),
        "names": names,
    }
    atomic_json_dump(info_json, out_dir / "model_info.json")
    return {"model_info_txt": str(info_txt), "model_info_json": str(out_dir / "model_info.json")}


def run_smoke_training(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "smoke_training"
    report_path = paths.reports_dir / "smoke_training_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)

    state.mark_started(stage_name)
    try:
        YOLO = import_ultralytics()
        runtime_smoke_yaml = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_smoke.yaml"
        smoke_root = paths.runtime_root / "smoke_check"
        smoke_project = smoke_root / "ultralytics"
        smoke_name = "yolo11s_smoke"
        smoke_run_dir = smoke_project / smoke_name
        ensure_dir(smoke_project)

        train_args = {
            "data": str(runtime_smoke_yaml),
            "imgsz": int(cfg["training"]["imgsz"]),
            "epochs": int(cfg["training"]["smoke_epochs"]),
            "batch": int(cfg["training"]["smoke_batch"]),
            "device": cfg["training"]["device"],
            "workers": int(cfg["training"]["workers"]),
            "seed": int(cfg["training"]["seed"]),
            "deterministic": True,
            "cache": False,
            "amp": bool(cfg["training"]["amp"]),
            "project": str(smoke_project),
            "name": smoke_name,
            "exist_ok": True,
            "save_period": 1,
            "plots": True,
            "verbose": True,
        }

        results_csv = smoke_run_dir / "results.csv"
        monitor = EpochProgressMonitor(results_csv=results_csv, total_epochs=int(cfg["training"]["smoke_epochs"]), desc="Smoke train")
        model = YOLO(cfg["training"]["model"])
        monitor.start()
        try:
            model.train(**train_args)
        finally:
            monitor.stop()

        artifact_report = collect_known_artifacts(smoke_run_dir, paths.paper_assets_dir / "01_smoke_training", log)
        csv_report = parse_results_csv(results_csv)
        report = {
            "smoke_run_dir": str(smoke_run_dir),
            "results_csv": str(results_csv),
            "csv_report": csv_report,
            "artifact_report": artifact_report,
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def run_full_training(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "full_training"
    report_path = paths.reports_dir / "full_training_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)

    state.mark_started(stage_name)
    try:
        YOLO = import_ultralytics()
        runtime_full_yaml = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"
        ensure_dir(paths.ultralytics_project_dir)

        run_dir = paths.ultralytics_run_dir
        results_csv = run_dir / "results.csv"
        weights_dir = run_dir / "weights"
        last_pt = weights_dir / "last.pt"
        best_pt = weights_dir / "best.pt"

        target_epochs = int(cfg["training"]["epochs"])
        recorded_epochs = line_count_excluding_header(results_csv)
        if best_pt.exists() and recorded_epochs >= target_epochs:
            log.info("A complete training run already exists at %s. Skipping retraining.", run_dir)
        else:
            monitor = EpochProgressMonitor(results_csv=results_csv, total_epochs=target_epochs, desc="Full train")
            monitor.start()
            try:
                if last_pt.exists() and line_count_excluding_header(results_csv) < target_epochs:
                    log.info("Detected incomplete training. Resuming from %s", last_pt)
                    model = YOLO(str(last_pt))
                    model.train(resume=True)
                else:
                    log.info("Starting fresh YOLO11s full training run.")
                    model = YOLO(cfg["training"]["model"])
                    model.train(**common_train_args(cfg, runtime_full_yaml, paths, log))
            finally:
                monitor.stop()

        if not best_pt.exists():
            raise FileNotFoundError(f"best.pt was not found after training: {best_pt}")

        ensure_dir(paths.artifacts_dir)
        create_or_replace_link(best_pt, paths.artifacts_dir / "best.pt")
        if last_pt.exists():
            create_or_replace_link(last_pt, paths.artifacts_dir / "last.pt")

        artifact_report = collect_known_artifacts(run_dir, paths.paper_assets_dir / "02_full_training", log)
        model_info_report = capture_model_info(best_pt, paths.paper_assets_dir / "02_full_training" / "model_info", log)
        csv_report = parse_results_csv(results_csv)

        report = {
            "run_dir": str(run_dir),
            "results_csv": str(results_csv),
            "csv_report": csv_report,
            "best_pt": important_file_metadata(best_pt),
            "last_pt": important_file_metadata(last_pt) if last_pt.exists() else {"path": str(last_pt), "exists": False},
            "artifact_report": artifact_report,
            "model_info_report": model_info_report,
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def extract_val_metrics(metrics_obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["results_dict"] = dict(getattr(metrics_obj, "results_dict", {}))
    except Exception:
        out["results_dict"] = {}

    try:
        out["save_dir"] = str(getattr(metrics_obj, "save_dir"))
    except Exception:
        pass

    try:
        speed = getattr(metrics_obj, "speed", None)
        if speed is not None:
            out["speed_ms_per_image"] = {str(k): float(v) for k, v in dict(speed).items()}
    except Exception:
        pass

    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for key, attr in {
            "map50_95": "map",
            "map50": "map50",
            "map75": "map75",
            "mean_precision": "mp",
            "mean_recall": "mr",
        }.items():
            try:
                out[key] = float(getattr(box, attr))
            except Exception:
                pass
        try:
            out["maps_per_class"] = [float(x) for x in getattr(box, "maps")]
        except Exception:
            pass

    cm = getattr(metrics_obj, "confusion_matrix", None)
    if cm is not None:
        try:
            matrix = getattr(cm, "matrix")
            out["confusion_matrix_shape"] = list(matrix.shape)
            out["confusion_matrix"] = matrix.tolist()
        except Exception:
            pass

    return out


def run_validation(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger, split: str) -> Dict[str, Any]:
    stage_name = f"validate_{split}"
    report_path = paths.reports_dir / f"validation_{split}.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)

    state.mark_started(stage_name)
    try:
        YOLO = import_ultralytics()
        runtime_full_yaml = paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"
        best_pt = paths.artifacts_dir / "best.pt"
        if not best_pt.exists():
            best_pt = paths.ultralytics_run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError("best.pt not found for validation.")

        validation_name = f"val_{split}"
        validation_dir = paths.validation_root / validation_name
        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(runtime_full_yaml),
            split=split,
            imgsz=int(cfg["training"]["imgsz"]),
            batch=int(cfg["evaluation"]["batch"]),
            device=cfg["training"]["device"],
            workers=int(cfg["training"]["workers"]),
            plots=bool(cfg["evaluation"]["plots"]),
            conf=float(cfg["evaluation"]["conf"]),
            iou=float(cfg["evaluation"]["iou"]),
            save_json=bool(cfg["evaluation"]["save_json"]),
            project=str(paths.validation_root),
            name=validation_name,
            exist_ok=True,
            verbose=True,
        )

        metric_report = extract_val_metrics(metrics)
        save_dir = Path(metric_report.get("save_dir", validation_dir)).resolve()
        asset_report = collect_known_artifacts(save_dir, paths.paper_assets_dir / f"03_validation_{split}", log)

        report = {
            "split": split,
            "best_pt": str(best_pt),
            "metrics": metric_report,
            "save_dir": str(save_dir),
            "artifact_report": asset_report,
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def results_to_records(results_iter: Iterable[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for res in results_iter:
        row: Dict[str, Any] = {
            "path": str(getattr(res, "path", "")),
            "orig_shape": list(getattr(res, "orig_shape", [])) if getattr(res, "orig_shape", None) is not None else None,
            "speed": getattr(res, "speed", None),
            "detections": [],
        }
        try:
            boxes = getattr(res, "boxes", None)
            names = getattr(res, "names", {})
            if boxes is not None and boxes.data is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                conf = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
                cls = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []
                for i in range(len(xyxy)):
                    cls_id = int(cls[i]) if i < len(cls) else -1
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    row["detections"].append({
                        "xyxy": xyxy[i],
                        "confidence": float(conf[i]) if i < len(conf) else None,
                        "class_id": cls_id,
                        "class_name": class_name,
                    })
        except Exception:
            pass
        records.append(row)
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_qualitative_predictions(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "qualitative_predictions"
    report_path = paths.reports_dir / "qualitative_predictions.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)

    state.mark_started(stage_name)
    try:
        if not bool(cfg["qualitative"]["enable"]):
            report = {"skipped": True, "reason": "qualitative.enable is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report

        YOLO = import_ultralytics()
        best_pt = paths.artifacts_dir / "best.pt"
        if not best_pt.exists():
            best_pt = paths.ultralytics_run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError("best.pt not found for qualitative predictions.")

        model = YOLO(str(best_pt))

        outputs: Dict[str, Any] = {}
        for split_name, manifest_name in {
            "val": "paper_val_examples.txt",
            "test": "paper_test_examples.txt",
        }.items():
            manifest_path = paths.runtime_manifests_dir / manifest_name
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest for qualitative predictions: {manifest_path}")
            image_list = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            pred_dir = ensure_dir(paths.qualitative_root / split_name)
            predictor = model.predict(
                source=image_list,
                imgsz=int(cfg["training"]["imgsz"]),
                conf=float(cfg["qualitative"]["conf"]),
                max_det=int(cfg["qualitative"]["max_det"]),
                device=cfg["training"]["device"],
                save=True,
                save_txt=True,
                save_conf=True,
                show_labels=True,
                show_conf=True,
                show_boxes=True,
                project=str(paths.qualitative_root),
                name=split_name,
                exist_ok=True,
                stream=True,
                verbose=False,
            )
            records = []
            for res in tqdm(predictor, total=len(image_list), desc=f"Qualitative {split_name}", unit="img", dynamic_ncols=True):
                records.extend(results_to_records([res]))

            jsonl_path = pred_dir / f"{split_name}_predictions.jsonl"
            save_jsonl(records, jsonl_path)

            flat_rows = []
            for record in records:
                for det in record["detections"]:
                    flat_rows.append({
                        "image_path": record["path"],
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "x1": det["xyxy"][0],
                        "y1": det["xyxy"][1],
                        "x2": det["xyxy"][2],
                        "y2": det["xyxy"][3],
                    })
            pd.DataFrame(flat_rows).to_csv(pred_dir / f"{split_name}_predictions.csv", index=False)
            outputs[split_name] = {
                "manifest": str(manifest_path),
                "image_count": len(image_list),
                "prediction_dir": str(pred_dir),
                "jsonl": str(jsonl_path),
            }

        atomic_json_dump(outputs, report_path)
        state.mark_done(stage_name, outputs)
        return outputs
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def run_export_onnx(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "export_onnx"
    report_path = paths.reports_dir / "export_onnx.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)

    state.mark_started(stage_name)
    try:
        if not bool(cfg["export"]["enable_onnx_export"]):
            report = {"skipped": True, "reason": "enable_onnx_export is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report

        YOLO = import_ultralytics()
        best_pt = paths.artifacts_dir / "best.pt"
        if not best_pt.exists():
            best_pt = paths.ultralytics_run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError("best.pt not found for ONNX export.")

        ensure_dir(paths.exports_dir)
        model = YOLO(str(best_pt))
        exported_path = model.export(
            format="onnx",
            imgsz=int(cfg["export"]["imgsz"]),
            batch=int(cfg["export"]["batch"]),
            dynamic=bool(cfg["export"]["dynamic"]),
            simplify=bool(cfg["export"]["simplify"]),
            opset=cfg["export"]["opset"],
        )
        exported_path = Path(str(exported_path)).resolve()
        linked_onnx = paths.exports_dir / "best.onnx"
        create_or_replace_link(exported_path, linked_onnx)

        report: Dict[str, Any] = {
            "best_pt": important_file_metadata(best_pt),
            "exported_path": str(exported_path),
            "linked_path": str(linked_onnx),
            "linked_metadata": important_file_metadata(exported_path),
            "note": "Use this ONNX on Jetson later as the hand-off point before building TensorRT engines on-device.",
        }

        try:
            import onnx  # type: ignore
            model_proto = onnx.load(str(exported_path))
            onnx.checker.check_model(model_proto)
            input_info = []
            for x in model_proto.graph.input:
                dims = []
                tensor_type = getattr(x.type, "tensor_type", None)
                if tensor_type is not None and tensor_type.shape is not None:
                    for dim in tensor_type.shape.dim:
                        if dim.dim_value:
                            dims.append(int(dim.dim_value))
                        elif dim.dim_param:
                            dims.append(str(dim.dim_param))
                        else:
                            dims.append("?")
                input_info.append({"name": x.name, "shape": dims})
            output_info = []
            for x in model_proto.graph.output:
                dims = []
                tensor_type = getattr(x.type, "tensor_type", None)
                if tensor_type is not None and tensor_type.shape is not None:
                    for dim in tensor_type.shape.dim:
                        if dim.dim_value:
                            dims.append(int(dim.dim_value))
                        elif dim.dim_param:
                            dims.append(str(dim.dim_param))
                        else:
                            dims.append("?")
                output_info.append({"name": x.name, "shape": dims})
            report["onnx_validation"] = {
                "checker": "passed",
                "ir_version": getattr(model_proto, "ir_version", None),
                "opset_import": [{"domain": x.domain, "version": x.version} for x in model_proto.opset_import],
                "inputs": input_info,
                "outputs": output_info,
            }
        except Exception as exc:
            report["onnx_validation_error"] = str(exc)

        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def create_output_manifest(paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "create_output_manifest"
    report_path = paths.reports_dir / "output_manifest_report.json"

    state.mark_started(stage_name)
    try:
        rows: List[Dict[str, Any]] = []
        for p in sorted(paths.runtime_root.rglob("*")):
            if p.is_file():
                try:
                    rows.append({
                        "relative_path": str(p.relative_to(paths.runtime_root)),
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                        "is_symlink": p.is_symlink(),
                        "modified_at_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    })
                except Exception as exc:
                    rows.append({
                        "relative_path": str(p.relative_to(paths.runtime_root)),
                        "error": str(exc),
                    })

        manifest_csv = paths.runtime_root / "OUTPUT_FILE_MANIFEST.csv"
        pd.DataFrame(rows).to_csv(manifest_csv, index=False)
        report = {"manifest_csv": str(manifest_csv), "file_count": len(rows)}
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def write_final_summary(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> None:
    tuned_params_path = resolve_tuned_params_path(cfg, paths)
    summary = {
        "created_at_utc": utc_now_iso(),
        "dataset_runtime_report": read_json(paths.reports_dir / "dataset_runtime_report.json"),
        "smoke_training_report": read_json(paths.reports_dir / "smoke_training_report.json"),
        "full_training_report": read_json(paths.reports_dir / "full_training_report.json"),
        "validation_val": read_json(paths.reports_dir / "validation_val.json"),
        "validation_test": read_json(paths.reports_dir / "validation_test.json"),
        "qualitative_predictions": read_json(paths.reports_dir / "qualitative_predictions.json"),
        "export_onnx": read_json(paths.reports_dir / "export_onnx.json"),
        "output_manifest": read_json(paths.reports_dir / "output_manifest_report.json"),
        "effective_training_args": read_yaml_or_empty(paths.reports_dir / "effective_training_args.yaml"),
        "tuning_best_hyperparameters": read_yaml_or_empty(tuned_params_path),
        "tuning_report": read_json(paths.runtime_root / "optuna" / "tuning_report.json"),
        "source_snapshot_dir": str(paths.source_snapshot_dir),
        "runtime_full_yaml": str(paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"),
        "runtime_smoke_yaml": str(paths.runtime_dataset_dir / "CottonWeedDet12_runtime_smoke.yaml"),
        "jetson_policy_md": str(paths.runtime_dataset_dir / "JETSON_INFERENCE_POLICY.md"),
        "state_file": str(paths.state_file),
        "runtime_root": str(paths.runtime_root),
        "note": (
            "Final paper-level Jetson reporting later should use the frozen test split with batch size 1. "
            "Quick Jetson parser/export checks should use the deterministic val_jetson_pilot subset."
        ),
    }
    atomic_json_dump(summary, paths.runtime_root / "FINAL_SUMMARY.json")

    val_map = read_json(paths.reports_dir / "validation_val.json").get("metrics", {}).get("map50_95")
    test_map = read_json(paths.reports_dir / "validation_test.json").get("metrics", {}).get("map50_95")
    full_train = read_json(paths.reports_dir / "full_training_report.json")
    best_metric = full_train.get("csv_report", {}).get("best_metric_value")
    tuned_now = read_yaml_or_empty(tuned_params_path)

    summary_md = f"""# Final summary

- runtime root: `{paths.runtime_root}`
- state file: `{paths.state_file}`
- runtime full yaml: `{paths.runtime_dataset_dir / "CottonWeedDet12_runtime_full.yaml"}`
- final Jetson split later: `test_full.txt`
- quick Jetson debug split later: `val_jetson_pilot.txt`
- tuned params file (if used): `{tuned_params_path}`

## Key checkpoints
- best.pt: `{paths.artifacts_dir / "best.pt"}`
- ONNX: `{paths.exports_dir / "best.onnx"}`

## Main metrics currently saved
- best training metric (from results.csv): {best_metric}
- validation mAP50-95: {val_map}
- test mAP50-95: {test_map}

## Tuning status
- tuned hyperparameters present: {bool(tuned_now)}
- tuning report path: `{paths.runtime_root / "optuna" / "tuning_report.json"}`

## Paper-material folders
- environment captures: `{paths.env_dir}`
- source snapshot: `{paths.source_snapshot_dir}`
- tuning assets: `{paths.runtime_root / "optuna"}`
- training assets: `{paths.paper_assets_dir / "02_full_training"}`
- validation assets: `{paths.paper_assets_dir / "03_validation_val"}`
- qualitative predictions: `{paths.qualitative_root}`
- output manifest: `{paths.runtime_root / "OUTPUT_FILE_MANIFEST.csv"}`

## What is now preserved for the paper
- exact source snapshot of the package
- environment and Slurm snapshots
- dataset split manifests and split-level stats
- class counts by split
- optional Optuna study database, trial table, best hyperparameters, and tuning report
- full training curves and checkpoint hashes
- validation plots and confusion matrices
- fixed qualitative example predictions
- ONNX file metadata and checksums
- global output manifest for later archival
"""
    write_text(paths.runtime_root / "FINAL_SUMMARY.md", summary_md)
    log.info("Final summary written to %s", paths.runtime_root / "FINAL_SUMMARY.json")


def print_status(paths: Paths) -> None:
    state = read_json(paths.state_file)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    results_csv = paths.ultralytics_run_dir / "results.csv"
    if results_csv.exists():
        print(f"\nresults.csv: {results_csv}")
        print(f"epochs recorded: {line_count_excluding_header(results_csv)}")
    else:
        print("\nNo training results.csv found yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 YOLO11s pipeline for Training_Cluster.")
    parser.add_argument("--config", required=True, help="Path to detection_yolo11s_train_lock.yaml")
    parser.add_argument("--mode", choices=["status", "prepare", "smoke", "full"], default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    paths = Paths.from_config(cfg)

    ensure_dir(paths.runtime_root)
    ensure_dir(paths.logs_dir)
    log = setup_logger(paths.logs_dir / f"pipeline_{args.mode}_{int(time.time())}.log")
    log.info("Loaded config from %s", cfg_path)

    state = StateManager(paths.state_file)

    if args.mode == "status":
        print_status(paths)
        return

    # Always capture context and source snapshot once per invocation.
    capture_environment(paths, log, args.mode)
    archive_source_snapshot(paths, cfg_path, log)
    build_runtime_manifests(cfg, paths, state, log)

    if args.mode == "prepare":
        create_output_manifest(paths, state, log)
        write_final_summary(cfg, paths, state, log)
        return

    if args.mode == "smoke":
        run_smoke_training(cfg, paths, state, log)
        create_output_manifest(paths, state, log)
        write_final_summary(cfg, paths, state, log)
        return

    if args.mode == "full":
        if bool(cfg["training"]["run_smoke_before_full"]):
            run_smoke_training(cfg, paths, state, log)
        run_full_training(cfg, paths, state, log)
        run_validation(cfg, paths, state, log, split="val")
        run_validation(cfg, paths, state, log, split="test")
        run_qualitative_predictions(cfg, paths, state, log)
        run_export_onnx(cfg, paths, state, log)
        create_output_manifest(paths, state, log)
        write_final_summary(cfg, paths, state, log)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
