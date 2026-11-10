#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import os
import platform
import random
import re
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def sanitized_runtime_env(
    extra_env: Optional[Dict[str, str]] = None,
    *,
    strip_conda_lib_from_ld: bool = False,
) -> Dict[str, str]:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        drop_entries = set()
        if strip_conda_lib_from_ld:
            conda_prefix = env.get("CONDA_PREFIX")
            if conda_prefix:
                drop_entries.add(str(Path(conda_prefix).expanduser() / "lib"))

        cleaned_entries: List[str] = []
        seen_entries = set()
        for entry in ld_library_path.split(":"):
            entry = entry.strip()
            if not entry or entry in seen_entries or entry in drop_entries:
                continue
            seen_entries.add(entry)
            cleaned_entries.append(entry)

        if cleaned_entries:
            env["LD_LIBRARY_PATH"] = ":".join(cleaned_entries)
        else:
            env.pop("LD_LIBRARY_PATH", None)

    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return env


def _parse_trailing_int(raw: Any) -> Optional[int]:
    text = str(raw).strip()
    if not text:
        return None
    match = re.search(r"(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def _normalize_distributed_mode(raw: Any) -> str:
    if isinstance(raw, bool):
        return "enabled" if raw else "disabled"
    text = str(raw).strip().lower()
    if text in {"", "auto"}:
        return "auto"
    if text in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    if text in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    return "auto"


def visible_cuda_device_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices and cuda_visible_devices not in {"NoDevFiles", "-1"}:
        entries = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        if entries:
            return len(entries)

    for key in ("RTDETR_REQUESTED_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        parsed = _parse_trailing_int(os.environ.get(key, ""))
        if parsed and parsed > 0:
            return parsed

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def resolve_distributed_process_count(cfg: Dict[str, Any]) -> int:
    distributed_cfg = dict(cfg.get("distributed", {}))
    mode = _normalize_distributed_mode(distributed_cfg.get("enabled", "auto"))
    if mode == "disabled":
        return 1

    visible_gpus = visible_cuda_device_count()
    requested_raw = distributed_cfg.get("gpus_per_node", "auto")
    requested: Optional[int]
    if isinstance(requested_raw, int):
        requested = requested_raw
    else:
        requested_text = str(requested_raw).strip().lower()
        requested = None if requested_text in {"", "auto"} else _parse_trailing_int(requested_text)

    nproc = requested if requested is not None else visible_gpus
    if nproc is None or nproc < 1:
        nproc = 1
    if visible_gpus > 0:
        nproc = min(nproc, visible_gpus)
    if mode == "auto" and nproc < 2:
        return 1
    return max(1, int(nproc))


def split_total_across_processes(total: int, nproc: int, label: str) -> int:
    total = int(total)
    if nproc <= 1:
        return total
    if total < nproc:
        raise ValueError(f"{label}={total} is smaller than distributed process count {nproc}.")
    if total % nproc != 0:
        raise ValueError(f"{label}={total} must be divisible by distributed process count {nproc}.")
    return total // nproc


def workers_per_process(total_workers: int, nproc: int) -> int:
    total_workers = max(0, int(total_workers))
    if nproc <= 1:
        return total_workers
    return max(1, total_workers // nproc) if total_workers > 0 else 0


def build_repo_train_command(
    cfg: Dict[str, Any],
    paths: "Paths",
    config_path: Path,
    *,
    seed: int,
    resume: Optional[Path] = None,
    tuning: Optional[Path] = None,
    test_only: bool = False,
    use_amp: bool = True,
) -> List[str]:
    train_script = repo_train_script(paths)
    nproc = resolve_distributed_process_count(cfg)
    if nproc > 1:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            str(train_script),
        ]
    else:
        cmd = [sys.executable, str(train_script)]

    cmd += ["-c", str(config_path), "--seed", str(seed)]
    if use_amp:
        cmd.append("--amp")
    if resume is not None:
        cmd += ["-r", str(resume)]
    elif tuning is not None:
        cmd += ["-t", str(tuning)]
    if test_only:
        cmd.append("--test-only")
    return cmd


MODEL_CHECKPOINT_RE = re.compile(r"^checkpoint(?:\d{4})?\.pth$", re.IGNORECASE)
TRAIN_BATCH_PROGRESS_RE = re.compile(r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\]")
TRAIN_EPOCH_COMPLETE_RE = re.compile(r"Epoch:\s*\[(\d+)\]\s*Total time:")


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


def normalize_names(raw: Any) -> Dict[int, str]:
    if isinstance(raw, dict):
        out: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return dict(sorted(out.items()))
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    raise TypeError(f"Unsupported names object: {type(raw)}")


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(f"stage2_rtdetr_r18_{int(time.time())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_subprocess(
    cmd: List[str],
    log: logging.Logger,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    log_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    effective_env = sanitized_runtime_env(env)
    log.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=effective_env, text=True, capture_output=True, check=False)
    if log_output and result.stdout:
        log.info("stdout:\n%s", result.stdout.strip())
    if log_output and result.stderr:
        log.info("stderr:\n%s", result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def run_command_streaming(cmd: List[str], log: logging.Logger, cwd: Optional[Path], tee_path: Path, env: Optional[Dict[str, str]] = None, check: bool = True) -> str:
    ensure_dir(tee_path.parent)
    effective_env = sanitized_runtime_env(env)
    log.info("Streaming command: %s", " ".join(cmd))
    lines: List[str] = []
    process: subprocess.Popen[str] | None = None
    with tee_path.open("w", encoding="utf-8") as tee:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=effective_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tee.write(line)
                tee.flush()
                lines.append(line)
            process.wait()
        except KeyboardInterrupt:
            if process is not None:
                with contextlib.suppress(Exception):
                    process.terminate()
                with contextlib.suppress(Exception):
                    process.wait(timeout=10)
                if process.poll() is None:
                    with contextlib.suppress(Exception):
                        process.kill()
                    with contextlib.suppress(Exception):
                        process.wait(timeout=5)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
        if check and process is not None and process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")
    return "".join(lines)


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
    repo_vendor_dir: Path
    repo_work_dir: Path
    bundle_dir: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Paths":
        dataset_root = Path(cfg["dataset"]["dataset_root"]).expanduser().resolve()
        runtime_root = Path(cfg["runtime"]["root"]).expanduser().resolve()
        state_dir = runtime_root / "state"
        vendor_dir_cfg = cfg.get("repo", {}).get("vendor_dir")
        repo_vendor_dir = Path(vendor_dir_cfg).expanduser().resolve() if vendor_dir_cfg else (PACKAGE_ROOT / "vendor" / "RT-DETR")
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
            repo_vendor_dir=repo_vendor_dir,
            repo_work_dir=repo_vendor_dir / cfg.get("repo", {}).get("checkout_subdir", "rtdetr_pytorch"),
            bundle_dir=runtime_root / "bundles",
        )


class StateManager:
    def __init__(self, path: Path):
        self.path = path
        self.state = read_json(path)
        if not self.state:
            self.state = {"schema_version": 2, "created_at_utc": utc_now_iso(), "stages": {}}
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
        stage.pop("last_error", None)
        if extra:
            stage.update(extra)
        self.save()

    def mark_failed(self, name: str, exc: Exception) -> None:
        stage = self.stage(name)
        stage["status"] = "failed"
        stage["finished_at_utc"] = utc_now_iso()
        stage["last_error"] = str(exc)
        self.save()


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
    atomic_json_dump(env_snapshot, target_dir / "environment_snapshot.json")
    slurm_env = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    atomic_json_dump(slurm_env, target_dir / "slurm_environment.json")

    shell_probe_env = sanitized_runtime_env(strip_conda_lib_from_ld=True)
    commands = {
        "uname.txt": ["uname", "-a"],
        "lscpu.txt": ["lscpu"],
        "module_list.txt": ["/bin/bash", "-lc", "module list 2>&1 || ml list 2>&1 || true"],
        "nvidia_smi_L.txt": ["nvidia-smi", "-L"],
        "nvidia_smi.txt": ["nvidia-smi"],
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "df_h.txt": ["df", "-h"],
        "env_sorted.txt": ["/bin/bash", "-lc", "env | sort || true"],
    }
    for name, cmd in commands.items():
        try:
            result = run_subprocess(
                cmd,
                log,
                check=False,
                log_output=False,
                env=shell_probe_env if cmd[0].endswith("bash") else None,
            )
            write_text(target_dir / name, result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
        except Exception as exc:
            write_text(target_dir / name, f"FAILED: {exc}\n")

    if os.environ.get("SLURM_JOB_ID"):
        try:
            result = run_subprocess(
                ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]],
                log,
                check=False,
                log_output=False,
            )
            write_text(target_dir / "scontrol_show_job.txt", result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
        except Exception as exc:
            write_text(target_dir / "scontrol_show_job.txt", f"FAILED: {exc}\n")

    atomic_json_dump({"sys_path": sys.path, "package_root": str(PACKAGE_ROOT)}, target_dir / "python_paths.json")


def archive_source_snapshot(paths: Paths, cfg_path: Path, log: logging.Logger) -> Dict[str, Any]:
    target = ensure_dir(paths.source_snapshot_dir)
    src_root = PACKAGE_ROOT
    selected: List[Path] = []
    for rel in [Path("requirements.txt"), Path("config"), Path("docs"), Path("scripts")]:
        src = src_root / rel
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    selected.append(p)
        elif src.is_file():
            selected.append(src)

    copied: List[Dict[str, Any]] = []
    for src in sorted(set(selected)):
        rel = src.relative_to(src_root)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"relative_path": str(rel), "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})

    if cfg_path.exists():
        shutil.copy2(cfg_path, target / "config_used.yaml")

    atomic_json_dump({"captured_at_utc": utc_now_iso(), "package_root": str(src_root), "file_count": len(copied), "files": copied}, target / "source_snapshot_manifest.json")
    pd.DataFrame(copied).to_csv(target / "source_snapshot_manifest.csv", index=False)
    log.info("Saved source snapshot to %s", target)
    return {"source_snapshot_dir": str(target), "file_count": len(copied)}


def deterministic_subset(items: List[str], k: int, seed: int) -> List[str]:
    if k >= len(items):
        return list(items)
    rng = random.Random(seed)
    chosen = set(rng.sample(items, k))
    return [x for x in items if x in chosen]


def runtime_coco_ids_match_expected(annotation_path: Path, expected_category_ids: List[int]) -> bool:
    if not annotation_path.exists():
        return False
    try:
        obj = read_json(annotation_path)
        categories = obj.get("categories", [])
        annotations = obj.get("annotations", [])
        category_ids = sorted(int(cat["id"]) for cat in categories)
        if category_ids != sorted(expected_category_ids):
            return False
        if not annotations:
            return True
        ann_ids = [int(ann["category_id"]) for ann in annotations]
        return min(ann_ids) >= min(expected_category_ids) and max(ann_ids) <= max(expected_category_ids)
    except Exception:
        return False


def parse_yolo_label_file(path: Path, img_w: int, img_h: int) -> Dict[str, Any]:
    class_ids: List[int] = []
    annotations: List[Tuple[int, List[float], float]] = []
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
                cls_id = int(float(parts[0]))
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                bw = w * img_w
                bh = h * img_h
                x = (cx * img_w) - bw / 2.0
                y = (cy * img_h) - bh / 2.0
                bbox = [max(0.0, x), max(0.0, y), max(0.0, bw), max(0.0, bh)]
                area = max(0.0, bbox[2] * bbox[3])
                class_ids.append(cls_id)
                annotations.append((cls_id, bbox, area))
            except Exception:
                parse_errors += 1
    return {"line_count": line_count, "class_ids": class_ids, "parse_errors": parse_errors, "annotations": annotations}


def important_file_metadata(path: Path) -> Dict[str, Any]:
    meta = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        meta.update({"size_bytes": path.stat().st_size, "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": sha256_file(path), "md5": md5_file(path)})
    return meta


def count_files_in_tree(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def collect_known_artifacts(src_dir: Path, dst_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    if not src_dir.exists():
        return {"path": str(src_dir), "exists": False}
    create_or_replace_link(src_dir, dst_dir)
    return {"source": str(src_dir), "linked_at": str(dst_dir), "file_count": count_files_in_tree(src_dir)}


def format_duration_compact(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def estimate_remaining_seconds(elapsed_seconds: float, progress_fraction: float) -> float | None:
    if elapsed_seconds <= 0 or progress_fraction <= 0 or progress_fraction >= 1:
        return None
    return elapsed_seconds * ((1.0 - progress_fraction) / progress_fraction)


def read_text_tail(path: Path, max_bytes: int = 256_000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def read_training_progress(train_log: Path, target_epochs: int) -> Dict[str, Any]:
    safe_target_epochs = max(1, int(target_epochs))
    text = read_text_tail(train_log)

    completed_epoch_matches = list(TRAIN_EPOCH_COMPLETE_RE.finditer(text))
    completed_epochs = max((int(m.group(1)) + 1 for m in completed_epoch_matches), default=0)

    batch_matches = list(TRAIN_BATCH_PROGRESS_RE.finditer(text))
    epoch_zero_based: int | None = None
    epoch_one_based = completed_epochs if completed_epochs > 0 else 0
    batch_done: int | None = None
    batch_total: int | None = None
    fractional_epochs = float(completed_epochs)

    if batch_matches:
        last = batch_matches[-1]
        epoch_zero_based = int(last.group(1))
        batch_done = int(last.group(2)) + 1
        batch_total = max(1, int(last.group(3)))
        epoch_one_based = max(epoch_one_based, epoch_zero_based + 1)
        fractional_epochs = max(fractional_epochs, float(epoch_zero_based) + min(batch_done, batch_total) / float(batch_total))

    if "Training time" in text:
        completed_epochs = safe_target_epochs
        fractional_epochs = float(safe_target_epochs)
        epoch_one_based = safe_target_epochs
        batch_done = batch_total

    progress_fraction = min(1.0, max(0.0, fractional_epochs / float(safe_target_epochs)))
    progress_percent = round(progress_fraction * 100.0, 1)

    return {
        "target_epochs": safe_target_epochs,
        "completed_epochs": min(safe_target_epochs, completed_epochs),
        "fractional_epochs": min(float(safe_target_epochs), fractional_epochs),
        "progress_fraction": progress_fraction,
        "progress_percent": progress_percent,
        "current_epoch": min(safe_target_epochs, epoch_one_based),
        "current_epoch_zero_based": epoch_zero_based,
        "current_batch_done": batch_done,
        "current_batch_total": batch_total,
        "remaining_epochs": round(max(0.0, float(safe_target_epochs) - fractional_epochs), 3),
    }


def start_training_progress_heartbeat(
    *,
    label: str,
    train_log: Path,
    target_epochs: int,
    started_monotonic: float,
    log: logging.Logger,
    interval_seconds: int = 30,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.wait(interval_seconds):
            progress = read_training_progress(train_log, target_epochs)
            elapsed_seconds = max(0.0, time.time() - started_monotonic)
            remaining_seconds = estimate_remaining_seconds(elapsed_seconds, float(progress["progress_fraction"]))
            batch_suffix = ""
            if progress["current_batch_done"] is not None and progress["current_batch_total"] is not None:
                batch_suffix = f" | batch={progress['current_batch_done']}/{progress['current_batch_total']}"
            log.info(
                "[PROGRESS] %s | progress=%s%% | epoch=%s/%s%s | elapsed=%s | eta=%s | left_epochs=%s | train_log=%s",
                label,
                progress["progress_percent"],
                progress["current_epoch"],
                progress["target_epochs"],
                batch_suffix,
                format_duration_compact(elapsed_seconds),
                format_duration_compact(remaining_seconds),
                progress["remaining_epochs"],
                train_log,
            )

    thread = threading.Thread(target=_run, name=f"{label}-progress-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def parse_best_epoch_from_log(log_path: Path) -> Optional[int]:
    if not log_path.exists():
        return None
    best_epoch: Optional[int] = None
    best_score: Optional[float] = None
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores = row.get("test_coco_eval_bbox")
                epoch = row.get("epoch")
                if not isinstance(scores, list) or not scores:
                    continue
                try:
                    score = float(scores[0])
                    epoch = int(epoch)
                except (TypeError, ValueError):
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_epoch = epoch
    except OSError:
        return None
    return best_epoch


def find_checkpoint_candidates(output_dir: Path) -> Dict[str, Optional[Path]]:
    if not output_dir.exists():
        return {"best": None, "latest": None, "all": []}
    model_ckpts = [p for p in output_dir.rglob("*.pth") if p.is_file() and MODEL_CHECKPOINT_RE.fullmatch(p.name)]
    if not model_ckpts:
        return {"best": None, "latest": None, "all": []}
    latest_candidates = sorted(model_ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
    latest_ckpt = output_dir / "checkpoint.pth"
    if not latest_ckpt.exists():
        latest_ckpt = latest_candidates[0]

    best_ckpt: Optional[Path] = None
    best_epoch = parse_best_epoch_from_log(output_dir / "log.txt")
    if best_epoch is not None:
        epoch_ckpt = output_dir / f"checkpoint{best_epoch:04}.pth"
        if epoch_ckpt.exists():
            best_ckpt = epoch_ckpt
        else:
            saved_epochs = [
                int(m.group(1))
                for m in (re.match(r"^checkpoint(\d{4})\.pth$", p.name, re.IGNORECASE) for p in model_ckpts)
                if m
            ]
            if saved_epochs and best_epoch == max(saved_epochs):
                best_ckpt = latest_ckpt

    if best_ckpt is None:
        best_candidates = sorted([p for p in model_ckpts if "best" in p.name.lower()], key=lambda p: p.stat().st_mtime, reverse=True)
        best_ckpt = best_candidates[0] if best_candidates else latest_ckpt

    return {"best": best_ckpt, "latest": latest_ckpt, "all": latest_candidates}


def parse_coco_metrics_from_text(text: str) -> Dict[str, Any]:
    patterns = {
        "map50_95": r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)",
        "map50": r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)",
        "map75": r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)",
        "ar100": r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)",
    }
    out: Dict[str, Any] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def resolve_tuned_params_path(cfg: Dict[str, Any], paths: Paths) -> Path:
    explicit = cfg["training"].get("tuned_params_yaml")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return paths.runtime_root / "optuna" / "best_hyperparameters.yaml"


def load_tuned_params_if_available(cfg: Dict[str, Any], paths: Paths, log: logging.Logger) -> Dict[str, Any]:
    if not bool(cfg["training"].get("use_tuned_params_if_available", False)):
        return {}
    path = resolve_tuned_params_path(cfg, paths)
    if not path.exists():
        log.info("No tuned params file found at %s. Using defaults.", path)
        return {}
    data = load_yaml(path)
    log.info("Loaded tuned params from %s: %s", path, data)
    return data


def load_dataset_names(cfg: Dict[str, Any]) -> Tuple[Dict[int, str], List[str]]:
    names_locked = normalize_names(load_yaml(Path(cfg["dataset"]["locked_ultralytics_yaml"]).expanduser().resolve())["names"])
    dataset_lock = load_yaml(Path(cfg["dataset"]["dataset_lock_yaml"]).expanduser().resolve())
    class_order = dataset_lock.get("class_order") or [names_locked[i] for i in sorted(names_locked)]
    return names_locked, class_order


def build_runtime_manifests(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "prepare_runtime_dataset"
    report_path = paths.reports_dir / "dataset_runtime_report.json"
    names_locked, class_order = load_dataset_names(cfg)
    expected_category_ids = [int(i) for i in sorted(names_locked)]
    runtime_train_json = paths.runtime_dataset_dir / "annotations" / "instances_train.json"
    if state.is_done(stage_name) and report_path.exists():
        if runtime_coco_ids_match_expected(runtime_train_json, expected_category_ids):
            return read_json(report_path)
        log.info("Rebuilding runtime dataset because %s does not use expected category ids %s", runtime_train_json, expected_category_ids)

    state.mark_started(stage_name)
    try:
        ensure_dir(paths.runtime_manifests_dir)
        ensure_dir(paths.runtime_dataset_dir)
        ensure_dir(paths.reports_dir)
        ensure_dir(paths.runtime_dataset_dir / "images")
        ensure_dir(paths.runtime_dataset_dir / "annotations")

        inventory_csv = Path(cfg["dataset"]["inventory_with_split_csv"]).expanduser().resolve()
        if not inventory_csv.exists():
            raise FileNotFoundError(f"inventory_with_split.csv not found: {inventory_csv}")

        df = pd.read_csv(inventory_csv)
        if "image_id" not in df.columns or "split" not in df.columns:
            raise ValueError("inventory_with_split.csv must include at least image_id and split columns.")

        image_root = paths.dataset_root / "canonical_yolo" / "images"
        label_root = paths.dataset_root / "canonical_yolo" / "labels"

        split_to_paths: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        coco_data: Dict[str, Dict[str, Any]] = {s: {"images": [], "annotations": [], "categories": [{"id": i, "name": names_locked[i], "supercategory": "weed"} for i in sorted(names_locked)]} for s in ("train", "val", "test")}
        split_box_counts = Counter()
        split_empty_label_files = Counter()
        split_nonempty_label_files = Counter()
        class_box_counts: Dict[str, Counter] = defaultdict(Counter)
        class_image_counts: Dict[str, Counter] = defaultdict(Counter)
        missing_images: List[str] = []
        missing_labels: List[str] = []
        parse_error_files: List[str] = []
        ann_id = 1
        image_id_counter = 1

        progress = tqdm(df.itertuples(index=False), total=len(df), desc="Build RT-DETR runtime dataset", unit="img", dynamic_ncols=True)
        for row in progress:
            image_id = getattr(row, "image_id")
            split = getattr(row, "split")
            if split not in split_to_paths:
                continue
            src_image = image_root / f"{image_id}.jpg"
            src_label = label_root / f"{image_id}.txt"
            if not src_image.exists():
                missing_images.append(str(src_image))
                continue
            if not src_label.exists():
                missing_labels.append(str(src_label))
                continue

            with Image.open(src_image) as im:
                width, height = im.size

            split_dir = ensure_dir(paths.runtime_dataset_dir / "images" / split)
            dst_image = split_dir / f"{image_id}.jpg"
            if not dst_image.exists():
                create_or_replace_link(src_image, dst_image)
            split_to_paths[split].append(str(dst_image))

            parsed = parse_yolo_label_file(src_label, width, height)
            if parsed["parse_errors"] > 0:
                parse_error_files.append(str(src_label))
            if parsed["line_count"] == 0:
                split_empty_label_files[split] += 1
            else:
                split_nonempty_label_files[split] += 1

            unique_in_image = set()
            coco_data[split]["images"].append({"id": image_id_counter, "file_name": dst_image.name, "width": width, "height": height})
            for cls_id, bbox, area in parsed["annotations"]:
                split_box_counts[split] += 1
                class_box_counts[split][cls_id] += 1
                unique_in_image.add(cls_id)
                coco_data[split]["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id_counter,
                    "category_id": int(cls_id),
                    "bbox": [round(float(x), 4) for x in bbox],
                    "area": round(float(area), 4),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
            for cls_id in unique_in_image:
                class_image_counts[split][cls_id] += 1
            image_id_counter += 1
        progress.close()

        if missing_images or missing_labels:
            atomic_json_dump({
                "missing_images": missing_images[:100],
                "missing_labels": missing_labels[:100],
                "missing_image_count": len(missing_images),
                "missing_label_count": len(missing_labels),
            }, report_path)
            raise RuntimeError(f"Dataset runtime check failed: {len(missing_images)} missing images and {len(missing_labels)} missing labels.")

        actual_counts = {split: len(paths_list) for split, paths_list in split_to_paths.items()}
        expected_counts = {k: int(v) for k, v in cfg["dataset"]["expected_split_counts"].items()}
        for split_name, expected in expected_counts.items():
            actual = actual_counts.get(split_name, -1)
            if actual != expected:
                raise RuntimeError(f"Split count mismatch for {split_name}: expected {expected}, got {actual}")

        # write COCO jsons
        annotations_dir = ensure_dir(paths.runtime_dataset_dir / "annotations")
        json_paths: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            json_path = annotations_dir / f"instances_{split_name}.json"
            atomic_json_dump(coco_data[split_name], json_path)
            json_paths[split_name] = str(json_path)
            manifest_path = paths.runtime_manifests_dir / f"{split_name}_full.txt"
            write_text(manifest_path, "\n".join(split_to_paths[split_name]) + "\n")

        seed = int(cfg["training"]["seed"])
        smoke_train = deterministic_subset(split_to_paths["train"], int(cfg["dataset"]["smoke_train_subset_size"]), seed + 11)
        smoke_val = deterministic_subset(split_to_paths["val"], int(cfg["dataset"]["smoke_val_subset_size"]), seed + 17)
        jetson_pilot_val = deterministic_subset(split_to_paths["val"], int(cfg["dataset"]["jetson_pilot_val_subset_size"]), seed + 29)
        paper_val_examples = deterministic_subset(split_to_paths["val"], int(cfg["qualitative"]["val_subset_size"]), seed + 37)
        paper_test_examples = deterministic_subset(split_to_paths["test"], int(cfg["qualitative"]["test_subset_size"]), seed + 41)

        write_text(paths.runtime_manifests_dir / "train_smoke.txt", "\n".join(smoke_train) + "\n")
        write_text(paths.runtime_manifests_dir / "val_smoke.txt", "\n".join(smoke_val) + "\n")
        write_text(paths.runtime_manifests_dir / "val_jetson_pilot.txt", "\n".join(jetson_pilot_val) + "\n")
        write_text(paths.runtime_manifests_dir / "paper_val_examples.txt", "\n".join(paper_val_examples) + "\n")
        write_text(paths.runtime_manifests_dir / "paper_test_examples.txt", "\n".join(paper_test_examples) + "\n")
        write_text(paths.runtime_dataset_dir / "labels.txt", "\n".join(class_order) + "\n")

        # build smoke COCO jsons by filtering full JSONs
        def build_subset_json(src_split: str, selected_paths: List[str], out_name: str) -> str:
            selected_names = {Path(p).name for p in selected_paths}
            full = coco_data[src_split]
            keep_images = [img for img in full["images"] if img["file_name"] in selected_names]
            keep_ids = {img["id"] for img in keep_images}
            keep_anns = [ann for ann in full["annotations"] if ann["image_id"] in keep_ids]
            obj = {"images": keep_images, "annotations": keep_anns, "categories": full["categories"]}
            out_path = annotations_dir / out_name
            atomic_json_dump(obj, out_path)
            return str(out_path)

        smoke_train_json = build_subset_json("train", smoke_train, "instances_train_smoke.json")
        smoke_val_json = build_subset_json("val", smoke_val, "instances_val_smoke.json")

        write_text(paths.runtime_dataset_dir / "JETSON_INFERENCE_POLICY.md", f"""# Jetson inference split policy\n\n## Final reported Jetson benchmark\n- split: `test_full.txt`\n- number of images: {len(split_to_paths['test'])}\n- use for: final accuracy, latency, power, energy, and temperature reporting on Jetson\n\n## Quick board-side smoke checks\n- split: `val_jetson_pilot.txt`\n- number of images: {len(jetson_pilot_val)}\n- use for: export sanity checks, parser checks, first-device wiring tests\n\n## Important rule\nDo **not** tune hyperparameters on the frozen test split.\n""")

        split_rows = []
        for split_name in ("train", "val", "test"):
            split_rows.append({"split": split_name, "image_count": actual_counts[split_name], "expected_image_count": expected_counts[split_name], "box_count": int(split_box_counts[split_name]), "nonempty_label_files": int(split_nonempty_label_files[split_name]), "empty_label_files": int(split_empty_label_files[split_name])})
        pd.DataFrame(split_rows).to_csv(paths.reports_dir / "split_level_stats.csv", index=False)

        class_box_rows = []
        class_img_rows = []
        for split_name in ("train", "val", "test"):
            for cls_id, cls_name in names_locked.items():
                class_box_rows.append({"split": split_name, "class_id": cls_id, "class_name": cls_name, "box_count": int(class_box_counts[split_name][cls_id])})
                class_img_rows.append({"split": split_name, "class_id": cls_id, "class_name": cls_name, "image_presence_count": int(class_image_counts[split_name][cls_id])})
        pd.DataFrame(class_box_rows).to_csv(paths.reports_dir / "class_box_counts_by_split.csv", index=False)
        pd.DataFrame(class_img_rows).to_csv(paths.reports_dir / "class_image_presence_by_split.csv", index=False)

        report = {
            "dataset_root": str(paths.dataset_root),
            "inventory_csv": str(inventory_csv),
            "runtime_dataset_dir": str(paths.runtime_dataset_dir),
            "manifests": {"train": str(paths.runtime_manifests_dir / 'train_full.txt'), "val": str(paths.runtime_manifests_dir / 'val_full.txt'), "test": str(paths.runtime_manifests_dir / 'test_full.txt')},
            "smoke_manifests": {"train": str(paths.runtime_manifests_dir / 'train_smoke.txt'), "val": str(paths.runtime_manifests_dir / 'val_smoke.txt')},
            "json_paths": json_paths,
            "smoke_json_paths": {"train": smoke_train_json, "val": smoke_val_json},
            "actual_counts": actual_counts,
            "expected_counts": expected_counts,
            "missing_images": len(missing_images),
            "missing_labels": len(missing_labels),
            "parse_error_file_count": len(parse_error_files),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def build_rtdetr_config_dict(cfg: Dict[str, Any], paths: Paths, output_dir: Path, train_img_dir: Path, train_ann: Path, val_img_dir: Path, val_ann: Path, *, epochs: int, train_batch: int, val_batch: int, workers: int, tuned: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tuned = tuned or {}
    names_locked, _ = load_dataset_names(cfg)
    imgsz = int(cfg["training"]["imgsz"])
    nproc = resolve_distributed_process_count(cfg)
    per_process_train_batch = split_total_across_processes(int(train_batch), nproc, "train_batch")
    per_process_val_batch = split_total_across_processes(int(val_batch), nproc, "val_batch")
    per_process_workers = workers_per_process(int(workers), nproc)
    lr = float(tuned.get("lr", 1e-4))
    weight_decay = float(tuned.get("weight_decay", 1e-4))
    warmups = int(tuned.get("warmups", 2000))
    ema_decay = float(tuned.get("ema_decay", 0.9999))
    photo_p = float(tuned.get("photometric_p", 0.5))
    iou_crop_p = float(tuned.get("iou_crop_p", 0.8))
    backbone_lr = lr * 0.1

    cfg_dict: Dict[str, Any] = {
        "task": "detection",
        "num_classes": len(names_locked),
        "remap_mscoco_category": False,
        "output_dir": str(output_dir),
        "sync_bn": nproc > 1,
        "find_unused_parameters": True,
        "use_ema": True,
        "ema": {"type": "ModelEMA", "decay": ema_decay},
        "warmups": warmups,
        "epoches": int(epochs),
        "clip_max_norm": 0.1,
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?=.*norm).*$", "lr": backbone_lr, "weight_decay": 0.0},
                {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": backbone_lr},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
            ],
            "lr": lr,
            "betas": [0.9, 0.999],
            "weight_decay": weight_decay,
        },
        "lr_scheduler": {"type": "MultiStepLR", "milestones": [1000], "gamma": 0.1},
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(train_img_dir),
                "ann_file": str(train_ann),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "RandomPhotometricDistort", "p": photo_p},
                        {"type": "RandomZoomOut", "fill": 0},
                        {"type": "RandomIoUCrop", "p": iou_crop_p},
                        {"type": "SanitizeBoundingBox", "min_size": 1},
                        {"type": "RandomHorizontalFlip"},
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                        {"type": "SanitizeBoundingBox", "min_size": 1},
                        {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True},
                    ],
                },
            },
            "shuffle": True,
            "batch_size": per_process_train_batch,
            "num_workers": per_process_workers,
            "drop_last": True,
            "collate_fn": "default_collate_fn",
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(val_img_dir),
                "ann_file": str(val_ann),
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [imgsz, imgsz]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                    ],
                },
            },
            "shuffle": False,
            "batch_size": per_process_val_batch,
            "num_workers": per_process_workers,
            "drop_last": False,
            "collate_fn": "default_collate_fn",
        },
        "model": "RTDETR",
        "criterion": "SetCriterion",
        "postprocessor": "RTDETRPostProcessor",
        "RTDETR": {
            "backbone": "PResNet",
            "encoder": "HybridEncoder",
            "decoder": "RTDETRTransformer",
            "multi_scale": [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
        },
        "PResNet": {
            "depth": 18,
            "variant": "d",
            "freeze_at": -1,
            "return_idx": [1, 2, 3],
            "num_stages": 4,
            "freeze_norm": False,
            "pretrained": True,
        },
        "HybridEncoder": {
            "in_channels": [128, 256, 512],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "enc_act": "gelu",
            "pe_temperature": 10000,
            "expansion": 0.5,
            "depth_mult": 1.0,
            "act": "silu",
            "eval_spatial_size": [imgsz, imgsz],
        },
        "RTDETRTransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_queries": 300,
            "num_decoder_layers": 3,
            "num_denoising": 100,
            "eval_idx": -1,
            "eval_spatial_size": [imgsz, imgsz],
            "use_focal_loss": True,
        },
        "RTDETRPostProcessor": {"num_top_queries": 300},
        "SetCriterion": {
            "weight_dict": {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
            "losses": ["vfl", "boxes"],
            "alpha": 0.75,
            "gamma": 2.0,
            "matcher": {"type": "HungarianMatcher", "weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, "alpha": 0.25, "gamma": 2.0},
        },
    }
    return cfg_dict


def repo_train_script(paths: Paths) -> Path:
    p = paths.repo_work_dir / "tools" / "train.py"
    if not p.exists():
        raise FileNotFoundError(f"RT-DETR training script not found: {p}. Run ensure_rtdetr_repo.py first.")
    return p


def repo_export_script(paths: Paths) -> Path:
    p = paths.repo_work_dir / "tools" / "export_onnx.py"
    if not p.exists():
        raise FileNotFoundError(f"RT-DETR export script not found: {p}. Run ensure_rtdetr_repo.py first.")
    return p


def weights_path_from_package(cfg: Dict[str, Any], paths: Paths) -> Path:
    p = PACKAGE_ROOT / cfg["weights"]["file_name"]
    if not p.exists():
        alt = PACKAGE_ROOT / "assets" / cfg["weights"]["file_name"]
        if alt.exists():
            return alt
    return p


def run_smoke_training(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "smoke_training"
    report_path = paths.reports_dir / "smoke_training_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        build_runtime_manifests(cfg, paths, state, log)
        repo_train_script(paths)
        weight_path = weights_path_from_package(cfg, paths)
        if not weight_path.exists():
            raise FileNotFoundError(f"RT-DETR weight file not found: {weight_path}")

        output_dir = paths.runtime_root / "rtdetr_runs" / "smoke_check"
        ensure_dir(output_dir)
        config_dir = ensure_dir(paths.runtime_dataset_dir / "configs")
        smoke_config = build_rtdetr_config_dict(
            cfg, paths, output_dir,
            train_img_dir=paths.runtime_dataset_dir / "images" / "train",
            train_ann=paths.runtime_dataset_dir / "annotations" / "instances_train_smoke.json",
            val_img_dir=paths.runtime_dataset_dir / "images" / "val",
            val_ann=paths.runtime_dataset_dir / "annotations" / "instances_val_smoke.json",
            epochs=int(cfg["training"]["smoke_epochs"]),
            train_batch=int(cfg["training"]["smoke_batch"]),
            val_batch=int(cfg["training"]["smoke_batch"]),
            workers=int(cfg["training"].get("smoke_workers", min(int(cfg["training"]["workers"]), 2))),
            tuned={},
        )
        config_path = config_dir / "rtdetr_smoke.yml"
        dump_yaml(smoke_config, config_path)

        ckpts_before = find_checkpoint_candidates(output_dir)
        latest_before = ckpts_before["latest"]
        train_log = paths.logs_dir / "smoke_train.log"
        if latest_before and latest_before.exists():
            log.info("Resuming smoke training from %s", latest_before)
            cmd = build_repo_train_command(
                cfg,
                paths,
                config_path,
                seed=int(cfg["training"]["seed"]),
                resume=latest_before,
                use_amp=bool(cfg["training"].get("amp", True)),
            )
        else:
            cmd = build_repo_train_command(
                cfg,
                paths,
                config_path,
                seed=int(cfg["training"]["seed"]),
                tuning=weight_path,
                use_amp=bool(cfg["training"].get("amp", True)),
            )
        train_started = time.time()
        progress_stop, progress_thread = start_training_progress_heartbeat(
            label="smoke_train",
            train_log=train_log,
            target_epochs=int(cfg["training"]["smoke_epochs"]),
            started_monotonic=train_started,
            log=log,
        )
        try:
            train_text = run_command_streaming(cmd, log, cwd=paths.repo_work_dir, tee_path=train_log)
        finally:
            progress_stop.set()
            progress_thread.join(timeout=1.0)

        ckpts = find_checkpoint_candidates(output_dir)
        best_ckpt = ckpts["best"]
        latest_ckpt = ckpts["latest"]
        if best_ckpt is None:
            raise RuntimeError("Smoke training finished but no checkpoint file was found.")

        val_log = paths.logs_dir / "smoke_validation.log"
        val_cmd = build_repo_train_command(
            cfg,
            paths,
            config_path,
            seed=int(cfg["training"]["seed"]),
            resume=best_ckpt,
            test_only=True,
            use_amp=bool(cfg["training"].get("amp", True)),
        )
        val_text = run_command_streaming(val_cmd, log, cwd=paths.repo_work_dir, tee_path=val_log)
        metrics = parse_coco_metrics_from_text(val_text)

        report = {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "train_log": str(train_log),
            "validation_log": str(val_log),
            "metrics": metrics,
            "best_checkpoint": important_file_metadata(best_ckpt),
            "latest_checkpoint": important_file_metadata(latest_ckpt) if latest_ckpt else {"exists": False},
            "train_output_chars": len(train_text),
            "val_output_chars": len(val_text),
            "artifact_report": collect_known_artifacts(output_dir, paths.paper_assets_dir / "01_smoke_training", log),
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
        build_runtime_manifests(cfg, paths, state, log)
        repo_train_script(paths)
        weight_path = weights_path_from_package(cfg, paths)
        if not weight_path.exists():
            raise FileNotFoundError(f"RT-DETR weight file not found: {weight_path}")

        tuned = load_tuned_params_if_available(cfg, paths, log)
        output_dir = paths.runtime_root / "rtdetr_runs" / "train"
        ensure_dir(output_dir)
        config_dir = ensure_dir(paths.runtime_dataset_dir / "configs")
        full_config = build_rtdetr_config_dict(
            cfg, paths, output_dir,
            train_img_dir=paths.runtime_dataset_dir / "images" / "train",
            train_ann=paths.runtime_dataset_dir / "annotations" / "instances_train.json",
            val_img_dir=paths.runtime_dataset_dir / "images" / "val",
            val_ann=paths.runtime_dataset_dir / "annotations" / "instances_val.json",
            epochs=int(cfg["training"]["epochs"]),
            train_batch=int(cfg["training"]["batch"]),
            val_batch=int(cfg["training"].get("eval_batch", cfg["training"]["batch"])),
            workers=int(cfg["training"]["workers"]),
            tuned=tuned,
        )
        config_path = config_dir / "rtdetr_full.yml"
        dump_yaml(full_config, config_path)
        dump_yaml({"config_path": str(config_path), "tuned_params": tuned, "weights_path": str(weight_path)}, paths.reports_dir / "effective_training_args.yaml")

        ckpts_before = find_checkpoint_candidates(output_dir)
        latest_before = ckpts_before["latest"]
        train_log = paths.logs_dir / "full_train.log"
        if latest_before and latest_before.exists():
            log.info("Resuming full training from %s", latest_before)
            cmd = build_repo_train_command(
                cfg,
                paths,
                config_path,
                seed=int(cfg["training"]["seed"]),
                resume=latest_before,
                use_amp=bool(cfg["training"].get("amp", True)),
            )
        else:
            cmd = build_repo_train_command(
                cfg,
                paths,
                config_path,
                seed=int(cfg["training"]["seed"]),
                tuning=weight_path,
                use_amp=bool(cfg["training"].get("amp", True)),
            )
        train_started = time.time()
        progress_stop, progress_thread = start_training_progress_heartbeat(
            label="full_train",
            train_log=train_log,
            target_epochs=int(cfg["training"]["epochs"]),
            started_monotonic=train_started,
            log=log,
        )
        try:
            train_text = run_command_streaming(cmd, log, cwd=paths.repo_work_dir, tee_path=train_log)
        finally:
            progress_stop.set()
            progress_thread.join(timeout=1.0)

        ckpts = find_checkpoint_candidates(output_dir)
        best_ckpt = ckpts["best"]
        latest_ckpt = ckpts["latest"]
        if best_ckpt is None:
            raise RuntimeError("Full training finished but no checkpoint file was found.")

        ensure_dir(paths.artifacts_dir)
        create_or_replace_link(best_ckpt, paths.artifacts_dir / "best.pth")
        if latest_ckpt:
            create_or_replace_link(latest_ckpt, paths.artifacts_dir / "last.pth")

        report = {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "train_log": str(train_log),
            "best_checkpoint": important_file_metadata(best_ckpt),
            "latest_checkpoint": important_file_metadata(latest_ckpt) if latest_ckpt else {"exists": False},
            "tuned_params": tuned,
            "train_output_chars": len(train_text),
            "artifact_report": collect_known_artifacts(output_dir, paths.paper_assets_dir / "02_full_training", log),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def run_validation(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger, split: str) -> Dict[str, Any]:
    stage_name = f"validate_{split}"
    report_path = paths.reports_dir / f"validation_{split}.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        build_runtime_manifests(cfg, paths, state, log)
        repo_train_script(paths)
        best_ckpt = paths.artifacts_dir / "best.pth"
        if not best_ckpt.exists():
            raise FileNotFoundError("best.pth not found for validation.")
        output_dir = paths.validation_root / f"val_{split}"
        ensure_dir(output_dir)
        config_dir = ensure_dir(paths.runtime_dataset_dir / "configs")
        eval_config = build_rtdetr_config_dict(
            cfg, paths, output_dir,
            train_img_dir=paths.runtime_dataset_dir / "images" / "train",
            train_ann=paths.runtime_dataset_dir / "annotations" / "instances_train.json",
            val_img_dir=paths.runtime_dataset_dir / "images" / split,
            val_ann=paths.runtime_dataset_dir / "annotations" / f"instances_{split}.json",
            epochs=int(cfg["training"]["epochs"]),
            train_batch=int(cfg["training"]["batch"]),
            val_batch=int(cfg["training"].get("eval_batch", cfg["training"]["batch"])),
            workers=int(cfg["training"]["workers"]),
            tuned=load_tuned_params_if_available(cfg, paths, log),
        )
        config_path = config_dir / f"rtdetr_eval_{split}.yml"
        dump_yaml(eval_config, config_path)
        val_log = paths.logs_dir / f"validation_{split}.log"
        cmd = build_repo_train_command(
            cfg,
            paths,
            config_path,
            seed=int(cfg["training"]["seed"]),
            resume=best_ckpt,
            test_only=True,
            use_amp=bool(cfg["training"].get("amp", True)),
        )
        text = run_command_streaming(cmd, log, cwd=paths.repo_work_dir, tee_path=val_log)
        metrics = parse_coco_metrics_from_text(text)
        report = {
            "split": split,
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "log_path": str(val_log),
            "metrics": metrics,
            "best_checkpoint": important_file_metadata(best_ckpt),
            "artifact_report": collect_known_artifacts(output_dir, paths.paper_assets_dir / f"03_validation_{split}", log),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
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
        if not bool(cfg["export"].get("enable_onnx_export", True)):
            report = {"skipped": True, "reason": "export.enable_onnx_export is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report
        repo_export_script(paths)
        best_ckpt = paths.artifacts_dir / "best.pth"
        if not best_ckpt.exists():
            raise FileNotFoundError("best.pth not found for ONNX export.")
        config_path = paths.runtime_dataset_dir / "configs" / "rtdetr_full.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Expected full config for export not found: {config_path}")
        ensure_dir(paths.exports_dir)
        onnx_path = paths.exports_dir / "rtdetr_r18_best.onnx"
        export_log = paths.logs_dir / "export_onnx.log"
        cmd = [sys.executable, str(repo_export_script(paths)), "-c", str(config_path), "-r", str(best_ckpt), "-f", str(onnx_path)]
        if bool(cfg["export"].get("check", True)):
            cmd.append("--check")
        if bool(cfg["export"].get("simplify", True)):
            cmd.append("--simplify")
        text = run_command_streaming(cmd, log, cwd=paths.repo_work_dir, tee_path=export_log)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file was not produced: {onnx_path}")
        report = {
            "onnx_path": str(onnx_path),
            "log_path": str(export_log),
            "best_checkpoint": important_file_metadata(best_ckpt),
            "onnx_metadata": important_file_metadata(onnx_path),
            "artifact_report": collect_known_artifacts(paths.exports_dir, paths.paper_assets_dir / "04_export_onnx", log),
            "output_chars": len(text),
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def draw_boxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill="yellow")
    return img


def results_to_records(image_path: str, detections: List[Dict[str, Any]], orig_shape: Tuple[int, int]) -> Dict[str, Any]:
    return {"path": image_path, "orig_shape": [orig_shape[1], orig_shape[0]], "detections": detections}


def run_qualitative_predictions(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "qualitative_predictions"
    report_path = paths.reports_dir / "qualitative_predictions.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        if not bool(cfg["qualitative"].get("enable", True)):
            report = {"skipped": True, "reason": "qualitative.enable is false"}
            atomic_json_dump(report, report_path)
            state.mark_done(stage_name, report)
            return report
        export_report = read_json(paths.reports_dir / "export_onnx.json")
        onnx_path = Path(export_report.get("onnx_path", ""))
        if not onnx_path.exists():
            raise FileNotFoundError("ONNX export missing for qualitative predictions.")
        import onnxruntime as ort  # type: ignore
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names_locked, _ = load_dataset_names(cfg)
        conf_thr = float(cfg["qualitative"]["conf"])
        max_det = int(cfg["qualitative"]["max_det"])
        imgsz = int(cfg["training"]["imgsz"])

        outputs: Dict[str, Any] = {}
        for split_name, manifest_name in {"val": "paper_val_examples.txt", "test": "paper_test_examples.txt"}.items():
            manifest = paths.runtime_manifests_dir / manifest_name
            image_list = [x.strip() for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
            split_out_dir = ensure_dir(paths.qualitative_root / split_name)
            records: List[Dict[str, Any]] = []
            for image_path in tqdm(image_list, desc=f"Qualitative {split_name}", unit="img", dynamic_ncols=True):
                img_path = Path(image_path)
                with Image.open(img_path).convert("RGB") as im:
                    orig_w, orig_h = im.size
                    resized = im.resize((imgsz, imgsz))
                    arr = np.asarray(resized).astype(np.float32) / 255.0
                    arr = np.transpose(arr, (2, 0, 1))[None, ...]
                    orig_sizes = np.asarray([[orig_h, orig_w]], dtype=np.int64)
                    labels, boxes, scores = session.run(None, {"images": arr, "orig_target_sizes": orig_sizes})
                    dets: List[Dict[str, Any]] = []
                    for cls_id, box, score in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                        if float(score) < conf_thr:
                            continue
                        dets.append({"xyxy": [float(x) for x in box], "confidence": float(score), "class_id": int(cls_id), "class_name": names_locked.get(int(cls_id), str(cls_id))})
                        if len(dets) >= max_det:
                            break
                    drawn = draw_boxes(im, dets)
                    out_path = split_out_dir / img_path.name
                    drawn.save(out_path)
                    records.append(results_to_records(str(img_path), dets, (orig_w, orig_h)))
            jsonl_path = split_out_dir / f"{split_name}_predictions.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            outputs[split_name] = {"count": len(records), "dir": str(split_out_dir), "jsonl": str(jsonl_path)}

        report = {"onnx_path": str(onnx_path), "outputs": outputs, "artifact_report": collect_known_artifacts(paths.qualitative_root, paths.paper_assets_dir / "05_qualitative_predictions", log)}
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise


def create_output_manifest(paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    rows = []
    for p in sorted(paths.runtime_root.rglob("*")):
        if p.is_file() or p.is_symlink():
            try:
                stat = p.stat()
                rows.append({"relative_path": str(p.relative_to(paths.runtime_root)), "is_symlink": p.is_symlink(), "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            except Exception:
                continue
    manifest_path = paths.runtime_root / "OUTPUT_FILE_MANIFEST.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    report = {"manifest_csv": str(manifest_path), "file_count": len(rows)}
    atomic_json_dump(report, paths.reports_dir / "output_manifest.json")
    return report


def write_final_summary(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> None:
    stages = state.state.get("stages", {})
    summary = {
        "created_at_utc": utc_now_iso(),
        "project_name": cfg["project"]["name"],
        "runtime_root": str(paths.runtime_root),
        "stages": stages,
        "best_checkpoint": important_file_metadata(paths.artifacts_dir / "best.pth") if (paths.artifacts_dir / "best.pth").exists() else {"exists": False},
        "onnx_export": read_json(paths.reports_dir / "export_onnx.json"),
        "validation_val": read_json(paths.reports_dir / "validation_val.json"),
        "validation_test": read_json(paths.reports_dir / "validation_test.json"),
    }
    atomic_json_dump(summary, paths.runtime_root / "FINAL_SUMMARY.json")

    lines = [
        f"# Final summary — RT-DETR-R18\n",
        f"- generated_at_utc: {summary['created_at_utc']}",
        f"- runtime_root: {summary['runtime_root']}",
        f"- best_checkpoint_exists: {summary['best_checkpoint'].get('exists', False)}",
    ]
    for stage_name, stage_info in stages.items():
        lines.append(f"- {stage_name}: {stage_info.get('status', 'unknown')}")
    write_text(paths.runtime_root / "FINAL_SUMMARY.md", "\n".join(lines) + "\n")


def print_status(paths: Paths) -> None:
    state = read_json(paths.state_file)
    print(json.dumps(state, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 RT-DETR-R18 pipeline for CottonWeedDet12 on Training_Cluster")
    parser.add_argument("--config", required=True, help="Path to detection_rtdetr_r18_train_lock.yaml")
    parser.add_argument("--mode", required=True, choices=["prepare", "smoke", "full", "status"], help="Pipeline mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    paths = Paths.from_config(cfg)
    ensure_dir(paths.runtime_root)
    ensure_dir(paths.logs_dir)

    if args.mode == "status":
        print_status(paths)
        return

    log = setup_logger(paths.logs_dir / f"pipeline_{args.mode}_{int(time.time())}.log")
    capture_environment(paths, log, args.mode)
    archive_source_snapshot(paths, cfg_path, log)
    state = StateManager(paths.state_file)

    try:
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
            if bool(cfg["training"].get("run_smoke_before_full", True)):
                run_smoke_training(cfg, paths, state, log)
            run_full_training(cfg, paths, state, log)
            run_validation(cfg, paths, state, log, "val")
            run_validation(cfg, paths, state, log, "test")
            run_export_onnx(cfg, paths, state, log)
            run_qualitative_predictions(cfg, paths, state, log)
            create_output_manifest(paths, state, log)
            write_final_summary(cfg, paths, state, log)
            return
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
