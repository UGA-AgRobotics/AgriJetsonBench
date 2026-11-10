#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vendor.bisenetv2 import BiSeNetV2, replace_adaptive_avg_pool  # noqa: E402


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")



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



def safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except FileNotFoundError:
        return



def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()



def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()



def important_file_metadata(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        st = path.stat()
        info.update(
            {
                "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": sha256_file(path),
                "md5": md5_file(path),
            }
        )
    return info



def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(f"bisenetv2_stage2_{int(time.time())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger



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
        cleaned: List[str] = []
        seen: set[str] = set()
        for entry in ld_library_path.split(":"):
            entry = entry.strip()
            if not entry or entry in seen or entry in drop_entries:
                continue
            seen.add(entry)
            cleaned.append(entry)
        if cleaned:
            env["LD_LIBRARY_PATH"] = ":".join(cleaned)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env



def run_subprocess(
    cmd: Sequence[str],
    log: logging.Logger,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    log_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    effective_env = sanitized_runtime_env(env)
    log.info("Running command: %s", " ".join(str(x) for x in cmd))
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=effective_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_output and result.stdout:
        log.info("stdout:\n%s", result.stdout.rstrip())
    if log_output and result.stderr:
        log.info("stderr:\n%s", result.stderr.rstrip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(x) for x in cmd)}")
    return result



def run_command_streaming(
    cmd: Sequence[str],
    log: logging.Logger,
    *,
    cwd: Optional[Path],
    tee_path: Path,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> str:
    ensure_dir(tee_path.parent)
    effective_env = sanitized_runtime_env(env)
    log.info("Streaming command: %s", " ".join(str(x) for x in cmd))
    lines: List[str] = []
    process: Optional[subprocess.Popen[bytes]] = None
    with tee_path.open("w", encoding="utf-8") as tee:
        try:
            process = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=str(cwd) if cwd else None,
                env=effective_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
            )
            assert process.stdout is not None
            for raw_line in iter(process.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
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
            raise
    if check and process is not None and process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(str(x) for x in cmd)}")
    return "".join(lines)



def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)



def create_or_replace_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)



def collect_known_artifacts(src_dir: Path, dst_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    if not src_dir.exists():
        return {"copied": 0, "source_missing": str(src_dir)}
    copied = 0
    for path in src_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src_dir)
            target = dst_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    log.info("Copied %d files from %s to %s", copied, src_dir, dst_dir)
    return {"copied": copied, "source": str(src_dir), "dest": str(dst_dir)}



def format_duration_compact(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"



def estimate_remaining_seconds(elapsed_seconds: float, progress_fraction: float) -> Optional[float]:
    if progress_fraction <= 0:
        return None
    total = elapsed_seconds / progress_fraction
    return max(0.0, total - elapsed_seconds)



def save_history_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



def read_history_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device



def _parse_trailing_int(raw: Any) -> Optional[int]:
    text = str(raw).strip()
    if not text:
        return None
    digits = []
    for ch in reversed(text):
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(reversed(digits)))



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
        entries = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
        if entries:
            return len(entries)
    for key in ("BISENETV2_REQUESTED_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        parsed = _parse_trailing_int(os.environ.get(key, ""))
        if parsed and parsed > 0:
            return parsed
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


def single_node_multi_gpu_mode() -> str:
    return str(os.environ.get("BISENETV2_SINGLE_NODE_MULTI_GPU", "dataparallel")).strip().lower()


def single_node_dataparallel_enabled(cfg: Dict[str, Any]) -> bool:
    mode = single_node_multi_gpu_mode()
    if mode not in {"1", "true", "yes", "on", "dp", "dataparallel"}:
        return False
    return resolve_distributed_node_count(cfg) <= 1 and visible_cuda_device_count() > 1



def resolve_distributed_process_count(cfg: Dict[str, Any]) -> int:
    distributed_cfg = dict(cfg.get("distributed", {}))
    mode = _normalize_distributed_mode(distributed_cfg.get("enabled", "auto"))
    if mode == "disabled":
        return 1
    if single_node_dataparallel_enabled(cfg):
        return 1
    visible = visible_cuda_device_count()
    requested_raw = distributed_cfg.get("gpus_per_node", "auto")
    if isinstance(requested_raw, int):
        requested = requested_raw
    else:
        requested_text = str(requested_raw).strip().lower()
        requested = None if requested_text in {"", "auto"} else _parse_trailing_int(requested_text)
    nproc = requested if requested is not None else visible
    if nproc is None or nproc < 1:
        nproc = 1
    if visible > 0:
        nproc = min(nproc, visible)
    if mode == "auto" and nproc < 2:
        return 1
    return max(1, int(nproc))


def resolve_distributed_node_count(cfg: Dict[str, Any]) -> int:
    distributed_cfg = dict(cfg.get("distributed", {}))
    mode = _normalize_distributed_mode(distributed_cfg.get("enabled", "auto"))
    if mode == "disabled":
        return 1
    for env_key in ("BISENETV2_NODE_COUNT", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        parsed = _parse_trailing_int(os.environ.get(env_key, ""))
        if parsed and parsed > 0:
            return int(parsed)
    requested_raw = distributed_cfg.get("nodes", "auto")
    if isinstance(requested_raw, int):
        requested = requested_raw
    else:
        requested_text = str(requested_raw).strip().lower()
        requested = None if requested_text in {"", "auto"} else _parse_trailing_int(requested_text)
    if requested and requested > 0:
        return int(requested)
    return 1


def _resolve_distributed_launcher(cfg: Dict[str, Any], nnodes: int) -> str:
    distributed_cfg = dict(cfg.get("distributed", {}))
    launcher_raw = os.environ.get("BISENETV2_DDP_LAUNCHER", distributed_cfg.get("launcher", "auto"))
    launcher = str(launcher_raw).strip().lower()
    if launcher in {"", "auto"}:
        if nnodes > 1:
            return "torchrun" if os.environ.get("SLURM_JOB_ID") else "unsupported"
        return "torchrun"
    if launcher in {"slurm", "srun"}:
        return "slurm"
    if launcher in {"torchrun", "standalone"}:
        return "torchrun"
    return launcher


def build_training_launch_command(
    cfg: Dict[str, Any],
    *,
    entry: Path,
    entry_args: Sequence[str],
) -> List[str]:
    nproc = resolve_distributed_process_count(cfg)
    nnodes = resolve_distributed_node_count(cfg)
    rendered_args = [str(x) for x in entry_args]
    if nproc <= 1:
        return [sys.executable, str(entry), *rendered_args]

    launcher = _resolve_distributed_launcher(cfg, nnodes)
    if nnodes <= 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            str(entry),
            *rendered_args,
        ]

    master_addr = str(os.environ.get("MASTER_ADDR", "")).strip()
    master_port = str(os.environ.get("MASTER_PORT", "")).strip()
    if not master_addr or not master_port:
        raise RuntimeError(
            "MASTER_ADDR and MASTER_PORT must be set for multi-node training. "
            "The Slurm wrapper should export them before launching Optuna/full training."
        )
    cpus_per_node = _parse_trailing_int(os.environ.get("SLURM_CPUS_PER_TASK", "")) or 1
    cpus_per_task = max(1, cpus_per_node // max(1, nproc))
    package_dir = str(os.environ.get("PACKAGE_DIR", PACKAGE_ROOT)).strip() or str(PACKAGE_ROOT)
    venv_path = str(os.environ.get("VENV_PATH", os.environ.get("VIRTUAL_ENV", ""))).strip()
    activation_script = Path(package_dir) / "scripts" / "activate_stage2_env.sh"
    if launcher == "slurm":
        shell_parts = [
            "set -euo pipefail;",
            f"source {shlex.quote(str(activation_script))} {shlex.quote(venv_path)} >/dev/null;",
            "exec python -u",
            shlex.quote(str(entry)),
        ]
        shell_parts.extend(shlex.quote(arg) for arg in rendered_args)
        return [
            "srun",
            f"--nodes={nnodes}",
            f"--ntasks={nnodes * nproc}",
            f"--ntasks-per-node={nproc}",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            f"--cpus-per-task={cpus_per_task}",
            "--kill-on-bad-exit=1",
            "--export=ALL",
            "--label",
            "bash",
            "-lc",
            " ".join(shell_parts),
        ]

    if launcher != "torchrun":
        raise RuntimeError(
            f"Unsupported multi-node launcher={launcher!r}. Expected 'torchrun' or 'slurm'."
        )

    shell_parts = [
        "set -euo pipefail;",
        f"source {shlex.quote(str(activation_script))} {shlex.quote(venv_path)} >/dev/null;",
        "exec python -u -m torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc}",
        "--node_rank=${SLURM_NODEID}",
        f"--master_addr={shlex.quote(master_addr)}",
        f"--master_port={shlex.quote(master_port)}",
        "--max_restarts=0",
        shlex.quote(str(entry)),
    ]
    shell_parts.extend(shlex.quote(arg) for arg in rendered_args)
    return [
        "srun",
        f"--nodes={nnodes}",
        f"--ntasks={nnodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus_per_node}",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--label",
        "bash",
        "-lc",
        " ".join(shell_parts),
    ]



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
def init_distributed() -> DistInfo:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "-1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    visible = visible_cuda_device_count()
    if visible == 1 and local_rank > 0:
        local_rank = 0
    elif visible > 1 and local_rank >= visible:
        local_rank = local_rank % visible
    enabled = world_size > 1 and local_rank >= 0
    if enabled:
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
    if torch.cuda.is_available():
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    if enabled and not dist.is_initialized():
        backend_raw = str(os.environ.get("BISENETV2_DIST_BACKEND", "auto")).strip().lower()
        if backend_raw in {"", "auto"}:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        else:
            backend = backend_raw
        timeout_seconds = int(os.environ.get("BISENETV2_DIST_TIMEOUT_SEC", "120"))
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_seconds))
    return DistInfo(enabled=enabled, rank=rank, local_rank=local_rank, world_size=world_size, device=device)



def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            if manual_barriers_enabled():
                dist.barrier()
        finally:
            dist.destroy_process_group()



def is_main_process(dist_info: DistInfo) -> bool:
    return not dist_info.enabled or dist_info.rank == 0



def manual_barriers_enabled() -> bool:
    raw = str(os.environ.get("BISENETV2_DISABLE_MANUAL_BARRIERS", "")).strip().lower()
    return raw not in {"1", "true", "yes", "on"}



def barrier_if_needed(dist_info: DistInfo) -> None:
    if manual_barriers_enabled() and dist_info.enabled and dist.is_initialized():
        dist.barrier()


def emit_rank_event(dist_info: DistInfo, message: str) -> None:
    print(f"[RANK {dist_info.rank}/{dist_info.world_size}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Dataset discovery and manifests
# ---------------------------------------------------------------------------


_SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test", "testing", "holdout"},
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SampleRecord:
    split: str
    image_path: str
    mask_path: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None



def normalize_split_name(raw: str) -> Optional[str]:
    text = str(raw).strip().lower()
    for canonical, aliases in _SPLIT_ALIASES.items():
        if text in aliases:
            return canonical
    return None



def _candidate_columns(columns: Sequence[str], role: str) -> List[str]:
    norm = {c.lower().strip(): c for c in columns}
    if role == "split":
        priority = ["split", "subset", "partition", "fold"]
    elif role == "image":
        priority = ["image_path", "image", "image_file", "rgb_path", "img_path", "img"]
    elif role == "mask":
        priority = ["mask_path", "mask", "label_path", "label", "annotation_path", "semantic_mask_path"]
    else:
        raise ValueError(role)
    picked: List[str] = []
    for key in priority:
        if key in norm:
            picked.append(norm[key])
    for col in columns:
        low = col.lower().strip()
        if role == "split" and "split" in low and col not in picked:
            picked.append(col)
        if role == "image" and any(tok in low for tok in ["image", "img", "rgb"]) and col not in picked:
            picked.append(col)
        if role == "mask" and any(tok in low for tok in ["mask", "label", "annotation"]) and col not in picked:
            picked.append(col)
    return picked



def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return (root / path).resolve()



def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]



def discover_from_inventory_csv(dataset_root: Path, csv_path: Path) -> Optional[Dict[str, List[SampleRecord]]]:
    try:
        rows = read_csv_rows(csv_path)
    except Exception:
        return None
    if not rows:
        return None
    columns = list(rows[0].keys())
    split_cols = _candidate_columns(columns, "split")
    image_cols = _candidate_columns(columns, "image")
    mask_cols = _candidate_columns(columns, "mask")
    if not split_cols or not image_cols or not mask_cols:
        return None
    split_col, image_col, mask_col = split_cols[0], image_cols[0], mask_cols[0]
    discovered: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = normalize_split_name(row.get(split_col, ""))
        if split is None:
            continue
        image_value = row.get(image_col, "")
        mask_value = row.get(mask_col, "")
        if not image_value or not mask_value:
            continue
        image_path = resolve_path(dataset_root, image_value)
        mask_path = resolve_path(dataset_root, mask_value)
        if not image_path.exists() or not mask_path.exists():
            alt_image = resolve_path(csv_path.parent, image_value)
            alt_mask = resolve_path(csv_path.parent, mask_value)
            if alt_image.exists() and alt_mask.exists():
                image_path, mask_path = alt_image, alt_mask
            else:
                continue
        discovered[split].append(SampleRecord(split=split, image_path=str(image_path), mask_path=str(mask_path)))
    if all(discovered[key] for key in ("train", "val", "test")):
        return discovered
    return None



def discover_from_split_csvs(dataset_root: Path, csv_paths: Sequence[Path]) -> Optional[Dict[str, List[SampleRecord]]]:
    discovered: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    found = False
    for csv_path in csv_paths:
        name = csv_path.stem.lower()
        split: Optional[str] = None
        for key in ("train", "val", "test"):
            if key in name or any(alias in name for alias in _SPLIT_ALIASES[key]):
                split = key
                break
        if split is None:
            continue
        try:
            rows = read_csv_rows(csv_path)
        except Exception:
            continue
        if not rows:
            continue
        columns = list(rows[0].keys())
        image_cols = _candidate_columns(columns, "image")
        mask_cols = _candidate_columns(columns, "mask")
        if not image_cols or not mask_cols:
            continue
        image_col, mask_col = image_cols[0], mask_cols[0]
        local: List[SampleRecord] = []
        for row in rows:
            image_value = row.get(image_col, "")
            mask_value = row.get(mask_col, "")
            if not image_value or not mask_value:
                continue
            image_path = resolve_path(dataset_root, image_value)
            mask_path = resolve_path(dataset_root, mask_value)
            if not image_path.exists() or not mask_path.exists():
                alt_image = resolve_path(csv_path.parent, image_value)
                alt_mask = resolve_path(csv_path.parent, mask_value)
                if alt_image.exists() and alt_mask.exists():
                    image_path, mask_path = alt_image, alt_mask
                else:
                    continue
            local.append(SampleRecord(split=split, image_path=str(image_path), mask_path=str(mask_path)))
        if local:
            discovered[split].extend(local)
            found = True
    if found and all(discovered[key] for key in ("train", "val", "test")):
        return discovered
    return None



def _paired_files(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
    shared = sorted(set(images).intersection(masks))
    return [(images[k], masks[k]) for k in shared]



def discover_from_split_directories(dataset_root: Path) -> Optional[Dict[str, List[SampleRecord]]]:
    candidate_roots: List[Tuple[Path, Path]] = []
    for base in [dataset_root, dataset_root / "canonical", dataset_root / "segmentation", dataset_root / "prepared"]:
        for image_name, mask_name in [("images", "masks"), ("rgb", "labels")]:
            image_root = base / image_name
            mask_root = base / mask_name
            if image_root.exists() and mask_root.exists():
                candidate_roots.append((image_root, mask_root))
    for image_root, mask_root in candidate_roots:
        discovered: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
        ok = True
        for split in ("train", "val", "test"):
            img_dir = image_root / split
            mask_dir = mask_root / split
            if not img_dir.exists() or not mask_dir.exists():
                ok = False
                break
            pairs = _paired_files(img_dir, mask_dir)
            if not pairs:
                ok = False
                break
            discovered[split] = [SampleRecord(split=split, image_path=str(img), mask_path=str(mask)) for img, mask in pairs]
        if ok:
            return discovered
    return None



def discover_dataset_records(dataset_root: Path, log: Optional[logging.Logger] = None) -> Dict[str, List[SampleRecord]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    csv_candidates = sorted([p for p in dataset_root.rglob("*.csv") if p.is_file()], key=lambda p: (len(p.parts), str(p)))
    for csv_path in csv_candidates:
        name = csv_path.name.lower()
        if any(token in name for token in ["inventory", "manifest", "split"]):
            discovered = discover_from_inventory_csv(dataset_root, csv_path)
            if discovered is not None:
                if log:
                    log.info("Discovered dataset via inventory CSV: %s", csv_path)
                return discovered
    discovered = discover_from_split_csvs(dataset_root, csv_candidates)
    if discovered is not None:
        if log:
            log.info("Discovered dataset via per-split CSV manifests.")
        return discovered
    discovered = discover_from_split_directories(dataset_root)
    if discovered is not None:
        if log:
            log.info("Discovered dataset via images/<split> + masks/<split> layout.")
        return discovered
    raise RuntimeError(
        "Could not discover the CropAndWeed segmentation layout under "
        f"{dataset_root}. Expected an inventory/manifests CSV with image/mask/split columns "
        "or split directories like images/train + masks/train."
    )



def _probe_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size



def write_manifest_csv(records: Sequence[SampleRecord], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_path", "mask_path", "image_width", "image_height"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "image_path": record.image_path,
                    "mask_path": record.mask_path,
                    "image_width": record.image_width if record.image_width is not None else "",
                    "image_height": record.image_height if record.image_height is not None else "",
                }
            )



def read_manifest_csv(path: Path) -> List[SampleRecord]:
    rows = read_csv_rows(path)
    out: List[SampleRecord] = []
    for row in rows:
        out.append(
            SampleRecord(
                split=str(row.get("split", "")).strip(),
                image_path=str(row.get("image_path", "")).strip(),
                mask_path=str(row.get("mask_path", "")).strip(),
                image_width=int(row["image_width"]) if str(row.get("image_width", "")).strip() else None,
                image_height=int(row["image_height"]) if str(row.get("image_height", "")).strip() else None,
            )
        )
    return out



def deterministic_subset_indices(total: int, k: int, seed: int) -> List[int]:
    items = list(range(total))
    if k >= len(items):
        return items
    rng = random.Random(seed)
    chosen = set(rng.sample(items, k))
    return [i for i in items if i in chosen]



def deterministic_subset(records: Sequence[SampleRecord], k: int, seed: int) -> List[SampleRecord]:
    if k <= 0 or k >= len(records):
        return list(records)
    indices = deterministic_subset_indices(len(records), k, seed)
    return [records[i] for i in indices]



def summarize_records(records_by_split: Dict[str, List[SampleRecord]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}, "total_samples": 0}
    widths: List[int] = []
    heights: List[int] = []
    for split, records in records_by_split.items():
        summary["splits"][split] = {"count": len(records)}
        summary["total_samples"] += len(records)
        for r in records:
            if r.image_width is not None and r.image_height is not None:
                widths.append(r.image_width)
                heights.append(r.image_height)
    if widths and heights:
        summary["resolution_summary"] = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "median_width": float(np.median(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "median_height": float(np.median(heights)),
        }
    return summary


# ---------------------------------------------------------------------------
# Transforms and datasets
# ---------------------------------------------------------------------------


class TrainSegTransform:
    def __init__(
        self,
        *,
        size: int,
        min_scale: float,
        max_scale: float,
        hflip_p: float,
        color_jitter_p: float,
        color_jitter_strength: float,
        ignore_index: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.size = int(size)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.hflip_p = float(hflip_p)
        self.color_jitter_p = float(color_jitter_p)
        self.color_jitter_strength = float(color_jitter_strength)
        self.ignore_index = int(ignore_index)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def _normalize(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr)

    def _jitter(self, image: Image.Image) -> Image.Image:
        if self.color_jitter_strength <= 0 or random.random() >= self.color_jitter_p:
            return image
        strength = self.color_jitter_strength
        for enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
            factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
            image = enhancer(image).enhance(factor)
        return image

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        mask = mask.convert("L")
        orig_w, orig_h = image.size
        short_side = min(orig_w, orig_h)
        target_short = self.size * random.uniform(self.min_scale, self.max_scale)
        scale = target_short / max(1.0, float(short_side))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        if random.random() < self.hflip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = self._jitter(image)
        pad_w = max(0, self.size - new_w)
        pad_h = max(0, self.size - new_h)
        if pad_w > 0 or pad_h > 0:
            left = random.randint(0, pad_w)
            top = random.randint(0, pad_h)
            right = pad_w - left
            bottom = pad_h - top
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
            mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=self.ignore_index)
            new_w, new_h = image.size
        x0 = random.randint(0, max(0, new_w - self.size))
        y0 = random.randint(0, max(0, new_h - self.size))
        image = image.crop((x0, y0, x0 + self.size, y0 + self.size))
        mask = mask.crop((x0, y0, x0 + self.size, y0 + self.size))
        image_tensor = self._normalize(image)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return image_tensor, mask_tensor


class EvalSegTransform:
    def __init__(self, *, size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.size = int(size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def _normalize(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        return self._normalize(image), torch.from_numpy(np.asarray(mask, dtype=np.int64))


class SegmentationManifestDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    def __init__(self, records: Sequence[SampleRecord], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[index]
        image_path = Path(record.image_path)
        mask_path = Path(record.mask_path)
        with Image.open(image_path) as img_handle:
            image = img_handle.convert("RGB")
            orig_size = image.size
        with Image.open(mask_path) as mask_handle:
            mask = mask_handle.convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)
        meta = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "orig_width": int(orig_size[0]),
            "orig_height": int(orig_size[1]),
        }
        return image_tensor, mask_tensor, meta



def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    images = torch.stack([x[0] for x in batch], dim=0)
    masks = torch.stack([x[1] for x in batch], dim=0)
    metas = [x[2] for x in batch]
    return images, masks, metas



def compute_class_histogram(
    records: Sequence[SampleRecord],
    num_classes: int,
    ignore_index: int,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    hist = np.zeros(int(num_classes), dtype=np.int64)
    for record in tqdm(records, desc="Class histogram", unit="mask", dynamic_ncols=True, disable=not show_progress, ascii=True):
        with Image.open(record.mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.int64)
        valid = (mask >= 0) & (mask < num_classes) & (mask != ignore_index)
        bincount = np.bincount(mask[valid].reshape(-1), minlength=num_classes)
        hist += bincount[:num_classes]
    return hist



def compute_class_weights_from_histogram(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float64)
    hist = np.maximum(hist, 1.0)
    freqs = hist / np.maximum(1.0, hist.sum())
    median = np.median(freqs[freqs > 0])
    weights = median / freqs
    weights = np.clip(weights, 0.25, 4.0)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


@dataclass
class LoaderBundle:
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    train_records: List[SampleRecord]
    val_records: List[SampleRecord]
    test_records: List[SampleRecord]
    class_weights: Optional[np.ndarray]
    class_histogram: Optional[np.ndarray]



def build_loaders(
    cfg: Dict[str, Any],
    *,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Optional[Path],
    dist_info: DistInfo,
    train_batch_total: Optional[int] = None,
    eval_batch_total: Optional[int] = None,
    workers_total: Optional[int] = None,
    train_subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
) -> LoaderBundle:
    dataset_cfg = dict(cfg.get("dataset", {}))
    training_cfg = dict(cfg.get("training", {}))
    augmentation_cfg = dict(cfg.get("augmentation", {}))
    normalization_cfg = dict(cfg.get("normalization", {}))
    multi_node = dist_info.enabled and dist_info.world_size > max(1, visible_cuda_device_count())

    num_classes = int(dataset_cfg.get("num_classes", 3))
    ignore_index = int(dataset_cfg.get("ignore_index", 255))
    input_size = int(training_cfg.get("input_size", 640))
    mean = normalization_cfg.get("mean", [0.485, 0.456, 0.406])
    std = normalization_cfg.get("std", [0.229, 0.224, 0.225])

    train_records = read_manifest_csv(train_manifest)
    val_records = read_manifest_csv(val_manifest)
    test_records = read_manifest_csv(test_manifest) if test_manifest and test_manifest.exists() else []

    if train_subset_size is not None and 0 < train_subset_size < len(train_records):
        train_records = deterministic_subset(train_records, train_subset_size, int(training_cfg.get("seed", 0)))
    if val_subset_size is not None and 0 < val_subset_size < len(val_records):
        val_records = deterministic_subset(val_records, val_subset_size, int(training_cfg.get("seed", 0)) + 1)

    train_transform = TrainSegTransform(
        size=input_size,
        min_scale=float(augmentation_cfg.get("train_min_scale", 0.6)),
        max_scale=float(augmentation_cfg.get("train_max_scale", 1.8)),
        hflip_p=float(augmentation_cfg.get("hflip_p", 0.5)),
        color_jitter_p=float(augmentation_cfg.get("color_jitter_p", 0.4)),
        color_jitter_strength=float(augmentation_cfg.get("color_jitter_strength", 0.2)),
        ignore_index=ignore_index,
        mean=mean,
        std=std,
    )
    eval_transform = EvalSegTransform(size=input_size, mean=mean, std=std)

    nproc = dist_info.world_size if dist_info.enabled else 1
    train_batch_total = int(train_batch_total if train_batch_total is not None else training_cfg.get("batch", 32))
    eval_batch_total = int(eval_batch_total if eval_batch_total is not None else training_cfg.get("eval_batch", 8))
    workers_total = int(workers_total if workers_total is not None else training_cfg.get("workers", 16))
    train_batch = split_total_across_processes(train_batch_total, nproc, "train_batch")
    eval_batch = split_total_across_processes(eval_batch_total, nproc, "eval_batch")
    workers_local = workers_per_process(workers_total, nproc)
    safe_dataloader = dist_info.enabled and os.environ.get("BISENETV2_SAFE_DATALOADER", "1") != "0"
    if safe_dataloader:
        workers_local = 0
    use_pin_memory = dist_info.device.type == "cuda" and not safe_dataloader and os.environ.get("BISENETV2_DISABLE_PIN_MEMORY", "0") != "1"
    use_persistent_workers = workers_local > 0 and use_pin_memory

    train_dataset = SegmentationManifestDataset(train_records, train_transform)
    val_dataset = SegmentationManifestDataset(val_records, eval_transform)
    test_dataset = SegmentationManifestDataset(test_records, eval_transform) if test_records else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=True) if dist_info.enabled else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=False) if dist_info.enabled else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=False) if dist_info.enabled and test_dataset is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=workers_local,
        pin_memory=use_pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=workers_local,
        pin_memory=use_pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=use_persistent_workers,
    )
    test_loader: Optional[DataLoader] = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch,
            sampler=test_sampler,
            shuffle=False,
            num_workers=workers_local,
            pin_memory=use_pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=use_persistent_workers,
        )

    class_weights: Optional[np.ndarray] = None
    class_histogram: Optional[np.ndarray] = None
    class_weight_mode = str(training_cfg.get("class_weights", "none")).strip().lower()
    if class_weight_mode.startswith("auto"):
        class_histogram = compute_class_histogram(
            train_records,
            num_classes=num_classes,
            ignore_index=ignore_index,
            show_progress=is_main_process(dist_info) or not dist_info.enabled,
        )
        class_weights = compute_class_weights_from_histogram(class_histogram)

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        class_weights=class_weights,
        class_histogram=class_histogram,
    )


# ---------------------------------------------------------------------------
# Model, loss, optimizer, scheduler, metrics
# ---------------------------------------------------------------------------


class OhemCrossEntropy(nn.Module):
    def __init__(
        self,
        *,
        prob_threshold: float,
        ignore_index: int,
        class_weights: Optional[Sequence[float]] = None,
        min_kept_divisor: int = 16,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.threshold = float(-math.log(max(1e-8, prob_threshold)))
        self.ignore_index = int(ignore_index)
        self.min_kept_divisor = max(1, int(min_kept_divisor))
        self.label_smoothing = float(label_smoothing)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("class_weights", weight_tensor if weight_tensor is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kwargs = {"ignore_index": self.ignore_index, "reduction": "none"}
        if self.class_weights is not None:
            kwargs["weight"] = self.class_weights.to(logits.device)
        if self.label_smoothing > 0:
            kwargs["label_smoothing"] = self.label_smoothing
        loss = F.cross_entropy(logits, target, **kwargs).reshape(-1)
        valid = target.reshape(-1) != self.ignore_index
        loss = loss[valid]
        if loss.numel() == 0:
            return logits.sum() * 0.0
        n_min = max(1, loss.numel() // self.min_kept_divisor)
        hard = loss[loss > self.threshold]
        if hard.numel() < n_min:
            hard, _ = torch.topk(loss, k=min(n_min, loss.numel()))
        return torch.mean(hard)


class SegmentationCriterion(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_index: int,
        class_weights: Optional[Sequence[float]],
        aux_loss_weight: float,
        label_smoothing: float,
        loss_name: str,
        ohem_prob_threshold: float,
        ohem_min_kept_divisor: int,
    ) -> None:
        super().__init__()
        del num_classes
        self.ignore_index = int(ignore_index)
        self.aux_loss_weight = float(aux_loss_weight)
        self.loss_name = str(loss_name).strip().lower()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("class_weights", weight_tensor if weight_tensor is not None else None)
        self.label_smoothing = float(label_smoothing)
        if self.loss_name == "ohem_ce":
            self.main_loss = OhemCrossEntropy(
                prob_threshold=ohem_prob_threshold,
                ignore_index=ignore_index,
                class_weights=class_weights,
                min_kept_divisor=ohem_min_kept_divisor,
                label_smoothing=label_smoothing,
            )
        elif self.loss_name in {"ce", "cross_entropy"}:
            self.main_loss = None
        else:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")

    def _cross_entropy(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kwargs = {"ignore_index": self.ignore_index}
        if self.class_weights is not None:
            kwargs["weight"] = self.class_weights.to(logits.device)
        if self.label_smoothing > 0:
            kwargs["label_smoothing"] = self.label_smoothing
        return F.cross_entropy(logits, target, **kwargs)

    def _loss_one(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.main_loss is not None:
            return self.main_loss(logits, target)
        return self._cross_entropy(logits, target)

    def forward(self, outputs: Any, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(outputs, tuple):
            main_logits, aux_outputs = outputs
        else:
            main_logits, aux_outputs = outputs, []
        loss_main = self._loss_one(main_logits, target)
        total = loss_main
        aux_terms: List[torch.Tensor] = []
        for aux_logits in aux_outputs:
            if aux_logits.shape[-2:] != target.shape[-2:]:
                aux_logits = F.interpolate(aux_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            aux_terms.append(self._loss_one(aux_logits, target))
        aux_loss_value = 0.0
        if aux_terms:
            aux_loss = torch.stack(aux_terms).mean()
            total = total + self.aux_loss_weight * aux_loss
            aux_loss_value = float(aux_loss.detach().cpu().item())
        breakdown = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_main": float(loss_main.detach().cpu().item()),
            "loss_aux": aux_loss_value,
        }
        return total, breakdown



def build_model(cfg: Dict[str, Any], num_classes: Optional[int] = None) -> nn.Module:
    dataset_cfg = dict(cfg.get("dataset", {}))
    model_cfg = dict(cfg.get("model", {}))
    model = BiSeNetV2(
        num_class=int(num_classes if num_classes is not None else dataset_cfg.get("num_classes", 3)),
        n_channel=3,
        act_type=str(model_cfg.get("act_type", "relu")),
        use_aux=bool(model_cfg.get("use_aux_heads", True)),
    )
    init_checkpoint = model_cfg.get("init_checkpoint")
    if init_checkpoint:
        load_model_weights(model, Path(str(init_checkpoint)).expanduser().resolve(), map_location="cpu")
    return model



def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out



def load_model_weights(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu") -> None:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)



def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training_cfg = dict(cfg.get("training", {}))
    opt_name = str(training_cfg.get("optimizer", "sgd")).strip().lower()
    lr = float(training_cfg.get("lr", 0.02))
    wd = float(training_cfg.get("weight_decay", 5e-4))
    multi_node = (_parse_trailing_int(os.environ.get("BISENETV2_NODE_COUNT", "")) or _parse_trailing_int(os.environ.get("SLURM_JOB_NUM_NODES", "")) or 1) > 1
    disable_lr_mul_raw = str(os.environ.get("BISENETV2_DISABLE_LR_MUL", "auto")).strip().lower()
    if disable_lr_mul_raw in {"1", "true", "yes", "on"}:
        disable_lr_mul = True
    elif disable_lr_mul_raw in {"0", "false", "no", "off"}:
        disable_lr_mul = False
    else:
        disable_lr_mul = single_node_dataparallel_enabled(cfg)
    foreach_override = os.environ.get("BISENETV2_OPT_FOREACH", "").strip().lower()
    foreach_enabled: Optional[bool]
    if foreach_override in {"0", "false", "no", "off"}:
        foreach_enabled = False
    elif foreach_override in {"1", "true", "yes", "on"}:
        foreach_enabled = True
    else:
        foreach_enabled = False if multi_node else None

    base_model = unwrap_parallel_model(model)
    if hasattr(base_model, "get_params"):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = base_model.get_params()
        if disable_lr_mul:
            params_list = [
                {"params": list(wd_params) + list(lr_mul_wd_params)},
                {"params": list(nowd_params) + list(lr_mul_nowd_params), "weight_decay": 0.0},
            ]
        else:
            params_list = [
                {"params": wd_params},
                {"params": nowd_params, "weight_decay": 0.0},
                {"params": lr_mul_wd_params, "lr": lr * 10.0},
                {"params": lr_mul_nowd_params, "weight_decay": 0.0, "lr": lr * 10.0},
            ]
    else:
        wd_params: List[nn.Parameter] = []
        non_wd_params: List[nn.Parameter] = []
        for _, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        params_list = [{"params": wd_params}, {"params": non_wd_params, "weight_decay": 0.0}]

    if opt_name == "sgd":
        kwargs: Dict[str, Any] = {
            "lr": lr,
            "momentum": float(training_cfg.get("momentum", 0.9)),
            "weight_decay": wd,
            "nesterov": True,
        }
        if foreach_enabled is not None:
            kwargs["foreach"] = foreach_enabled
        return torch.optim.SGD(params_list, **kwargs)
    if opt_name == "adamw":
        kwargs = {"lr": lr, "weight_decay": wd}
        if foreach_enabled is not None:
            kwargs["foreach"] = foreach_enabled
        return torch.optim.AdamW(params_list, **kwargs)
    raise ValueError(f"Unsupported optimizer: {opt_name}")



def build_scheduler(
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    *,
    total_epochs: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    training_cfg = dict(cfg.get("training", {}))
    power = float(training_cfg.get("power", 0.9))
    warmup_epochs = float(training_cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(training_cfg.get("warmup_start_factor", 0.1))
    epochs = max(1, int(total_epochs if total_epochs is not None else training_cfg.get("epochs", 1)))
    warmup_steps = int(round(total_steps * (warmup_epochs / epochs))) if epochs > 0 else 0
    warmup_steps = max(0, min(total_steps - 1 if total_steps > 1 else 0, warmup_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            alpha = current_step / max(1, warmup_steps)
            return warmup_start_factor + alpha * (1.0 - warmup_start_factor)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return max(0.0, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



def update_confusion_matrix(confusion: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    prediction = prediction.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if valid.any():
        inds = num_classes * target[valid] + prediction[valid]
        confusion += torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion



def compute_metrics_from_confusion(confusion: torch.Tensor, class_names: Sequence[str]) -> Dict[str, Any]:
    confusion_np = confusion.detach().cpu().numpy().astype(np.float64)
    intersection = np.diag(confusion_np)
    gt = confusion_np.sum(axis=1)
    pred = confusion_np.sum(axis=0)
    union = gt + pred - intersection
    iou = np.divide(intersection, np.maximum(union, 1.0), where=union > 0)
    dice = np.divide(2.0 * intersection, np.maximum(gt + pred, 1.0), where=(gt + pred) > 0)
    recall = np.divide(intersection, np.maximum(gt, 1.0), where=gt > 0)
    precision = np.divide(intersection, np.maximum(pred, 1.0), where=pred > 0)
    pixel_acc = float(intersection.sum() / np.maximum(confusion_np.sum(), 1.0))
    valid = gt > 0
    mean_iou = float(iou[valid].mean()) if np.any(valid) else 0.0
    mean_dice = float(dice[valid].mean()) if np.any(valid) else 0.0
    fg_valid = valid[1:] if len(valid) > 1 else valid
    foreground_miou = float(iou[1:][fg_valid].mean()) if len(iou) > 1 and np.any(fg_valid) else mean_iou
    foreground_mdice = float(dice[1:][fg_valid].mean()) if len(dice) > 1 and np.any(fg_valid) else mean_dice
    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "iou": float(iou[idx]),
            "dice": float(dice[idx]),
            "recall": float(recall[idx]),
            "precision": float(precision[idx]),
            "support_pixels": int(gt[idx]),
        }
    return {
        "miou": mean_iou,
        "mdice": mean_dice,
        "foreground_miou": foreground_miou,
        "foreground_mdice": foreground_mdice,
        "pixel_accuracy": pixel_acc,
        "per_class": per_class,
        "confusion_matrix": confusion_np.astype(np.int64).tolist(),
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainRunConfig:
    config_path: Path
    run_kind: str
    run_dir: Path
    train_manifest: Path
    val_manifest: Path
    epochs: int
    train_batch_total: int
    eval_batch_total: int
    workers_total: int
    tuned_params_yaml: Optional[Path]
    resume: Optional[Path]
    report_json: Optional[Path]
    history_jsonl: Optional[Path]
    class_histogram_json: Optional[Path]



def unwrap_parallel_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[Any],
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history_path: Path,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = unwrap_parallel_model(model)
    payload = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "model_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "history_jsonl": str(history_path),
        "config": cfg,
        "created_at_utc": utc_now_iso(),
    }
    torch.save(payload, path)



def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(strip_module_prefix(state), strict=False)
    if optimizer is not None and isinstance(payload, dict) and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and isinstance(payload, dict) and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and isinstance(payload, dict) and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return payload if isinstance(payload, dict) else {}



def apply_tuned_params(cfg: Dict[str, Any], tuned_yaml: Optional[Path]) -> Dict[str, Any]:
    if tuned_yaml is None or not tuned_yaml.exists():
        return cfg
    tuned = load_yaml(tuned_yaml)
    training_cfg = cfg.setdefault("training", {})
    augmentation_cfg = cfg.setdefault("augmentation", {})
    mapping = {
        "lr": (training_cfg, "lr"),
        "weight_decay": (training_cfg, "weight_decay"),
        "aux_loss_weight": (training_cfg, "aux_loss_weight"),
        "train_min_scale": (augmentation_cfg, "train_min_scale"),
        "train_max_scale": (augmentation_cfg, "train_max_scale"),
        "hflip_p": (augmentation_cfg, "hflip_p"),
        "color_jitter_strength": (augmentation_cfg, "color_jitter_strength"),
    }
    for key, value in tuned.items():
        if key in mapping:
            target_dict, target_key = mapping[key]
            target_dict[target_key] = value
    return cfg



def _current_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def make_grad_scaler(device: torch.device, enabled: bool) -> Optional[Any]:
    if device.type != "cuda":
        return None
    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
        return cuda_amp.GradScaler(enabled=enabled)
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(device="cuda", enabled=enabled)
    return None


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        cuda_amp = getattr(torch.cuda, "amp", None)
        if cuda_amp is not None and hasattr(cuda_amp, "autocast"):
            return cuda_amp.autocast(enabled=True)
        amp_module = getattr(torch, "amp", None)
        if amp_module is not None and hasattr(amp_module, "autocast"):
            return amp_module.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]],
    criterion: SegmentationCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[Any],
    dist_info: DistInfo,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_batches = 0
    grad_clip = float(cfg.get("training", {}).get("grad_clip_norm", 0.0))
    amp_enabled = bool(cfg.get("training", {}).get("amp", True)) and dist_info.device.type == "cuda"

    iterator: Iterable[Any] = loader
    if is_main_process(dist_info):
        iterator = tqdm(loader, desc=f"Train epoch {epoch}", unit="batch", dynamic_ncols=True, ascii=True)

    emit_rank_event(dist_info, f"waiting for batch 1 epoch={epoch}")
    for batch_idx, (images, masks, _) in enumerate(iterator, start=1):
        if batch_idx == 1:
            emit_rank_event(dist_info, f"batch 1 fetched epoch={epoch}")
        images = images.to(dist_info.device, non_blocking=True)
        masks = masks.to(dist_info.device, non_blocking=True)
        if batch_idx == 1:
            emit_rank_event(dist_info, f"batch 1 on device epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(dist_info.device, amp_enabled):
            outputs = model(images, is_training=True)
            loss, breakdown = criterion(outputs, masks)
        if not torch.isfinite(loss):
            if isinstance(outputs, tuple):
                main_logits = outputs[0]
                aux_outputs = list(outputs[1]) if len(outputs) > 1 and isinstance(outputs[1], (list, tuple)) else list(outputs[1:])
            else:
                main_logits = outputs
                aux_outputs = []
            main_min = float(torch.nan_to_num(main_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0).amin().item())
            main_max = float(torch.nan_to_num(main_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0).amax().item())
            main_finite = bool(torch.isfinite(main_logits).all().item())
            aux_finite = [bool(torch.isfinite(aux).all().item()) for aux in aux_outputs]
            raise FloatingPointError(
                "Non-finite loss encountered "
                f"at epoch={epoch} batch={batch_idx} "
                f"loss_total={breakdown.get('loss_total')} "
                f"loss_main={breakdown.get('loss_main')} "
                f"loss_aux={breakdown.get('loss_aux')} "
                f"main_logits_finite={main_finite} "
                f"aux_logits_finite={aux_finite} "
                f"main_logits_min={main_min:.6f} "
                f"main_logits_max={main_max:.6f}"
            )
        if batch_idx == 1:
            emit_rank_event(dist_info, f"batch 1 forward done epoch={epoch}")
        if scaler is not None and amp_enabled:
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 backward start epoch={epoch}")
            scaler.scale(loss).backward()
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 backward done epoch={epoch}")
            if grad_clip > 0:
                if batch_idx == 1:
                    emit_rank_event(dist_info, f"batch 1 grad clip start epoch={epoch}")
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if batch_idx == 1:
                    emit_rank_event(dist_info, f"batch 1 grad clip done epoch={epoch}")
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 optimizer step start epoch={epoch}")
            scaler.step(optimizer)
            scaler.update()
        else:
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 backward start epoch={epoch}")
            loss.backward()
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 backward done epoch={epoch}")
            if grad_clip > 0:
                if batch_idx == 1:
                    emit_rank_event(dist_info, f"batch 1 grad clip start epoch={epoch}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if batch_idx == 1:
                    emit_rank_event(dist_info, f"batch 1 grad clip done epoch={epoch}")
            if batch_idx == 1:
                emit_rank_event(dist_info, f"batch 1 optimizer step start epoch={epoch}")
            optimizer.step()
        if batch_idx == 1:
            emit_rank_event(dist_info, f"batch 1 optimizer step done epoch={epoch}")
        scheduler.step()
        total_loss += float(breakdown["loss_total"])
        total_main += float(breakdown["loss_main"])
        total_aux += float(breakdown["loss_aux"])
        total_batches += 1
        if is_main_process(dist_info) and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{breakdown['loss_total']:.4f}", lr=f"{_current_lrs(optimizer)[0]:.3e}")

    avg_loss = total_loss / max(1, total_batches)
    avg_main = total_main / max(1, total_batches)
    avg_aux = total_aux / max(1, total_batches)
    if dist_info.enabled:
        tensor = torch.tensor([avg_loss, avg_main, avg_aux, float(total_batches)], device=dist_info.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_loss = float(tensor[0].item() / dist_info.world_size)
        avg_main = float(tensor[1].item() / dist_info.world_size)
        avg_aux = float(tensor[2].item() / dist_info.world_size)
    return {"train_loss": avg_loss, "loss_main": avg_main, "loss_aux": avg_aux}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]],
    dist_info: DistInfo,
    cfg: Dict[str, Any],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    model.eval()
    dataset_cfg = dict(cfg.get("dataset", {}))
    num_classes = int(dataset_cfg.get("num_classes", len(class_names)))
    ignore_index = int(dataset_cfg.get("ignore_index", 255))
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=dist_info.device)
    iterator: Iterable[Any] = loader
    if is_main_process(dist_info):
        iterator = tqdm(loader, desc="Evaluate", unit="batch", dynamic_ncols=True, ascii=True)
    for images, masks, _ in iterator:
        images = images.to(dist_info.device, non_blocking=True)
        masks = masks.to(dist_info.device, non_blocking=True)
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        update_confusion_matrix(confusion, pred, masks, num_classes, ignore_index)
    if dist_info.enabled:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    return compute_metrics_from_confusion(confusion, class_names)


# ---------------------------------------------------------------------------
# Visualization and reports
# ---------------------------------------------------------------------------


DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ],
    dtype=np.uint8,
)



def palette_from_config(cfg: Dict[str, Any]) -> np.ndarray:
    dataset_cfg = dict(cfg.get("dataset", {}))
    qualitative_cfg = dict(cfg.get("qualitative", {}))
    class_names = list(dataset_cfg.get("class_names", ["background", "crop", "weed"]))
    palette_spec = qualitative_cfg.get("palette")
    if isinstance(palette_spec, dict):
        colors: List[List[int]] = []
        for name in class_names:
            value = palette_spec.get(name)
            if value is None or len(value) != 3:
                break
            colors.append([int(value[0]), int(value[1]), int(value[2])])
        if len(colors) == len(class_names):
            return np.asarray(colors, dtype=np.uint8)
    return DEFAULT_PALETTE[: len(class_names)]



def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> Image.Image:
    mask = mask.astype(np.int64)
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_idx in range(min(len(palette), int(mask.max()) + 1)):
        colored[mask == cls_idx] = palette[cls_idx]
    return Image.fromarray(colored, mode="RGB")



def overlay_mask(image: Image.Image, mask: np.ndarray, palette: np.ndarray, alpha: float) -> Image.Image:
    color = colorize_mask(mask, palette)
    return Image.blend(image.convert("RGB"), color, alpha=alpha)



def save_prediction_panel(
    image_path: Path,
    gt_mask_path: Path,
    pred_mask: np.ndarray,
    out_path: Path,
    *,
    input_size: int,
    palette: np.ndarray,
    alpha: float,
) -> None:
    with Image.open(image_path) as img_handle:
        image = img_handle.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    with Image.open(gt_mask_path) as gt_handle:
        gt_mask = np.asarray(gt_handle.convert("L").resize((input_size, input_size), Image.Resampling.NEAREST), dtype=np.uint8)
    pred_mask = pred_mask.astype(np.uint8)
    gt_overlay = overlay_mask(image, gt_mask, palette, alpha)
    pred_overlay = overlay_mask(image, pred_mask, palette, alpha)
    gt_color = colorize_mask(gt_mask, palette)
    pred_color = colorize_mask(pred_mask, palette)
    panel = Image.new("RGB", (input_size * 3, input_size * 2), (255, 255, 255))
    panel.paste(image, (0, 0))
    panel.paste(gt_overlay, (input_size, 0))
    panel.paste(pred_overlay, (input_size * 2, 0))
    panel.paste(gt_color, (input_size, input_size))
    panel.paste(pred_color, (input_size * 2, input_size))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), "image", fill=(255, 255, 255))
    draw.text((input_size + 10, 10), "gt overlay", fill=(255, 255, 255))
    draw.text((2 * input_size + 10, 10), "pred overlay", fill=(255, 255, 255))
    draw.text((input_size + 10, input_size + 10), "gt mask", fill=(0, 0, 0))
    draw.text((2 * input_size + 10, input_size + 10), "pred mask", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)



def plot_history(history_jsonl: Path, out_dir: Path, class_names: Sequence[str]) -> Dict[str, str]:
    del class_names
    ensure_dir(out_dir)
    rows = read_history_rows(history_jsonl)
    if not rows:
        return {}
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row.get("train_loss", 0.0)) for row in rows]
    loss_main = [float(row.get("loss_main", 0.0)) for row in rows]
    loss_aux = [float(row.get("loss_aux", 0.0)) for row in rows]
    lr = [float(row.get("lr", 0.0)) for row in rows]
    miou = [float(row.get("val_metrics", {}).get("miou", 0.0)) for row in rows]
    mdice = [float(row.get("val_metrics", {}).get("mdice", 0.0)) for row in rows]
    fg_miou = [float(row.get("val_metrics", {}).get("foreground_miou", 0.0)) for row in rows]
    fg_mdice = [float(row.get("val_metrics", {}).get("foreground_mdice", 0.0)) for row in rows]
    pixel_acc = [float(row.get("val_metrics", {}).get("pixel_accuracy", 0.0)) for row in rows]
    seconds = [float(row.get("seconds", 0.0)) for row in rows]

    outputs: Dict[str, str] = {}

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, loss_main, label="loss_main")
    plt.plot(epochs, loss_aux, label="loss_aux")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_path = out_dir / "training_loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()
    outputs["training_loss_curve_png"] = str(loss_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, miou, label="mIoU")
    plt.plot(epochs, fg_miou, label="foreground mIoU")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Validation mIoU curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    miou_path = out_dir / "training_miou_curve.png"
    plt.tight_layout()
    plt.savefig(miou_path, dpi=160)
    plt.close()
    outputs["training_miou_curve_png"] = str(miou_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mdice, label="mDice")
    plt.plot(epochs, fg_mdice, label="foreground mDice")
    plt.plot(epochs, pixel_acc, label="pixel accuracy")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Validation Dice / pixel accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    dice_path = out_dir / "training_dice_curve.png"
    plt.tight_layout()
    plt.savefig(dice_path, dpi=160)
    plt.close()
    outputs["training_dice_curve_png"] = str(dice_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("Learning-rate curve")
    plt.grid(True, alpha=0.3)
    lr_path = out_dir / "lr_curve.png"
    plt.tight_layout()
    plt.savefig(lr_path, dpi=160)
    plt.close()
    outputs["lr_curve_png"] = str(lr_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, seconds)
    plt.xlabel("epoch")
    plt.ylabel("seconds")
    plt.title("Epoch duration")
    plt.grid(True, alpha=0.3)
    sec_path = out_dir / "epoch_duration_curve.png"
    plt.tight_layout()
    plt.savefig(sec_path, dpi=160)
    plt.close()
    outputs["epoch_duration_curve_png"] = str(sec_path)

    pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "loss_main": loss_main,
            "loss_aux": loss_aux,
            "lr": lr,
            "val_miou": miou,
            "val_mdice": mdice,
            "val_foreground_miou": fg_miou,
            "val_foreground_mdice": fg_mdice,
            "val_pixel_accuracy": pixel_acc,
            "seconds": seconds,
        }
    ).to_csv(out_dir / "train_history.csv", index=False)
    outputs["train_history_csv"] = str(out_dir / "train_history.csv")
    return outputs



def plot_confusion_matrix(confusion: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    confusion = np.asarray(confusion, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, f"{int(confusion[i, j])}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def plot_per_class_metric(metrics: Dict[str, Any], class_names: Sequence[str], metric_key: str, out_path: Path, title: str) -> None:
    values = [float(metrics.get("per_class", {}).get(name, {}).get(metric_key, 0.0)) for name in class_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(class_names)), values)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def plot_split_counts(summary: Dict[str, Any], out_path: Path) -> None:
    splits = ["train", "val", "test"]
    counts = [int(summary.get("splits", {}).get(split, {}).get("count", 0)) for split in splits]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, counts)
    ax.set_ylabel("samples")
    ax.set_title("Dataset split counts")
    ax.grid(True, axis="y", alpha=0.3)
    for idx, value in enumerate(counts):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def plot_class_histogram(hist: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    values = hist.astype(np.int64).tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(class_names, values)
    ax.set_ylabel("pixels")
    ax.set_title("Train-set pixel histogram")
    ax.grid(True, axis="y", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def write_per_class_metrics_csv(metrics: Dict[str, Any], out_path: Path) -> None:
    rows = []
    per_class = metrics.get("per_class", {})
    for name, info in per_class.items():
        row = {"class_name": name}
        row.update({k: info.get(k) for k in ["iou", "dice", "recall", "precision", "support_pixels"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)



def make_contact_sheet(image_paths: Sequence[Path], out_path: Path, *, columns: int = 2, background: Tuple[int, int, int] = (255, 255, 255)) -> None:
    if not image_paths:
        return
    loaded = [Image.open(path).convert("RGB") for path in image_paths if Path(path).exists()]
    if not loaded:
        return
    width = max(im.width for im in loaded)
    height = max(im.height for im in loaded)
    cols = max(1, int(columns))
    rows = int(math.ceil(len(loaded) / cols))
    sheet = Image.new("RGB", (width * cols, height * rows), background)
    for idx, im in enumerate(loaded):
        x = (idx % cols) * width
        y = (idx // cols) * height
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.BILINEAR)
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    for im in loaded:
        im.close()


# ---------------------------------------------------------------------------
# Main training job
# ---------------------------------------------------------------------------



def run_training_job(train_cfg: TrainRunConfig) -> Dict[str, Any]:
    cfg = load_yaml(train_cfg.config_path)
    cfg = apply_tuned_params(cfg, train_cfg.tuned_params_yaml)
    cfg.setdefault("training", {})["epochs"] = int(train_cfg.epochs)

    dist_info = init_distributed()
    log = setup_logger(train_cfg.run_dir / f"train_{train_cfg.run_kind}.log") if is_main_process(dist_info) else logging.getLogger("bisenetv2_stage2_worker")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
    emit_rank_event(dist_info, f"initialized device={dist_info.device} local_rank={dist_info.local_rank}")

    dataset_cfg = dict(cfg.get("dataset", {}))
    training_cfg_raw = cfg.setdefault("training", {})
    training_cfg = dict(training_cfg_raw)
    multi_node = dist_info.enabled and dist_info.world_size > max(1, visible_cuda_device_count())
    safe_parallel = (dist_info.enabled or single_node_dataparallel_enabled(cfg)) and os.environ.get("BISENETV2_SAFE_DISTRIBUTED", "1") != "0"
    if safe_parallel:
        if bool(training_cfg.get("amp", True)):
            log.info("Disabling AMP for multi-GPU safety mode.")
        if float(training_cfg.get("grad_clip_norm", 0.0)) > 0:
            log.info("Disabling gradient clipping for multi-GPU safety mode.")
        training_cfg["amp"] = False
        training_cfg["grad_clip_norm"] = 0.0
        training_cfg_raw["amp"] = False
        training_cfg_raw["grad_clip_norm"] = 0.0
    if multi_node:
        if bool(training_cfg.get("amp", True)):
            log.info("Disabling AMP for multi-node distributed run.")
        if float(training_cfg.get("grad_clip_norm", 0.0)) > 0:
            log.info("Disabling gradient clipping for multi-node distributed run.")
        if bool(training_cfg.get("deterministic", True)):
            log.info("Disabling deterministic CuDNN mode for multi-node distributed run.")
        training_cfg["amp"] = False
        training_cfg["grad_clip_norm"] = 0.0
        training_cfg["deterministic"] = False
        training_cfg_raw["amp"] = False
        training_cfg_raw["grad_clip_norm"] = 0.0
        training_cfg_raw["deterministic"] = False
    class_names = list(dataset_cfg.get("class_names", ["background", "crop", "weed"]))
    num_classes = int(dataset_cfg.get("num_classes", len(class_names)))
    seed_everything(int(training_cfg.get("seed", 0)), bool(training_cfg.get("deterministic", True)))
    if multi_node and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    checkpoints_dir = ensure_dir(train_cfg.run_dir / "checkpoints")
    history_dir = ensure_dir(train_cfg.run_dir / "history")
    metrics_dir = ensure_dir(train_cfg.run_dir / "metrics")
    plots_dir = ensure_dir(train_cfg.run_dir / "plots")
    history_jsonl = train_cfg.history_jsonl or (history_dir / "train_history.jsonl")

    loader_bundle = build_loaders(
        cfg,
        train_manifest=train_cfg.train_manifest,
        val_manifest=train_cfg.val_manifest,
        test_manifest=None,
        dist_info=dist_info,
        train_batch_total=train_cfg.train_batch_total,
        eval_batch_total=train_cfg.eval_batch_total,
        workers_total=train_cfg.workers_total,
        train_subset_size=None,
        val_subset_size=None,
    )
    emit_rank_event(
        dist_info,
        (
            "loaders ready "
            f"train_batches={len(loader_bundle.train_loader)} "
            f"val_batches={len(loader_bundle.val_loader)}"
        ),
    )
    if is_main_process(dist_info):
        if loader_bundle.class_histogram is not None and train_cfg.class_histogram_json is not None:
            class_hist_json = {name: int(value) for name, value in zip(class_names, loader_bundle.class_histogram.tolist())}
            atomic_json_dump(class_hist_json, train_cfg.class_histogram_json)
            plot_class_histogram(loader_bundle.class_histogram, class_names, plots_dir / "class_histogram.png")

    emit_rank_event(dist_info, f"building model num_classes={num_classes}")
    model = build_model(cfg, num_classes=num_classes).to(dist_info.device)
    emit_rank_event(dist_info, "model moved to device")
    use_sync_batchnorm = dist_info.enabled and bool(training_cfg.get("sync_batchnorm", True)) and dist_info.device.type == "cuda"
    if use_sync_batchnorm and dist_info.enabled and os.environ.get("BISENETV2_SAFE_SYNCBN", "1") != "0":
        if is_main_process(dist_info):
            log.info("Skipping SyncBatchNorm conversion for distributed safety mode.")
        use_sync_batchnorm = False
    elif use_sync_batchnorm and dist_info.world_size > max(1, visible_cuda_device_count()):
        if is_main_process(dist_info):
            log.info("Skipping SyncBatchNorm conversion for multi-node distributed run.")
        use_sync_batchnorm = False
    if use_sync_batchnorm:
        emit_rank_event(dist_info, "converting SyncBatchNorm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        emit_rank_event(dist_info, "SyncBatchNorm converted")

    optimizer = build_optimizer(cfg, model)
    total_steps = max(1, len(loader_bundle.train_loader) * int(train_cfg.epochs))
    scheduler = build_scheduler(cfg, optimizer, total_steps=total_steps, total_epochs=int(train_cfg.epochs))
    amp_enabled = bool(training_cfg.get("amp", True)) and dist_info.device.type == "cuda"
    scaler: Optional[Any] = make_grad_scaler(dist_info.device, amp_enabled)
    emit_rank_event(dist_info, f"optimizer/scheduler ready total_steps={total_steps} amp_enabled={amp_enabled}")

    criterion = SegmentationCriterion(
        num_classes=num_classes,
        ignore_index=int(dataset_cfg.get("ignore_index", 255)),
        class_weights=loader_bundle.class_weights.tolist() if loader_bundle.class_weights is not None else None,
        aux_loss_weight=float(training_cfg.get("aux_loss_weight", 0.4)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        loss_name=str(training_cfg.get("loss_name", "ohem_ce")),
        ohem_prob_threshold=float(training_cfg.get("ohem_prob_threshold", 0.7)),
        ohem_min_kept_divisor=int(training_cfg.get("ohem_min_kept_divisor", 16)),
    ).to(dist_info.device)
    emit_rank_event(dist_info, "criterion ready")

    start_epoch = 1
    best_metric = -float("inf")
    best_epoch = 0
    if train_cfg.resume is not None and train_cfg.resume.exists():
        payload = load_checkpoint(train_cfg.resume, model, optimizer, scheduler, scaler, map_location=dist_info.device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_metric = float(payload.get("best_metric", -float("inf")))
        best_epoch = int(payload.get("best_epoch", 0))
        if is_main_process(dist_info):
            log.info("Resumed from %s at epoch %d", train_cfg.resume, start_epoch)

    use_dataparallel = (
        not dist_info.enabled
        and dist_info.device.type == "cuda"
        and single_node_dataparallel_enabled(cfg)
    )
    if use_dataparallel:
        device_ids = list(range(visible_cuda_device_count()))
        log.info("Using nn.DataParallel on single-node multi-GPU run with device_ids=%s", device_ids)
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    if dist_info.enabled:
        local_param_count = sum(1 for _ in model.parameters())
        emit_rank_event(dist_info, f"pre-DDP local_param_count={local_param_count}")
        visible_devices = visible_cuda_device_count()
        skip_ddp_verify_raw = str(os.environ.get("BISENETV2_SKIP_DDP_VERIFY", "auto")).strip().lower()
        if skip_ddp_verify_raw in {"1", "true", "yes", "on"}:
            skip_ddp_verify = True
        elif skip_ddp_verify_raw in {"0", "false", "no", "off"}:
            skip_ddp_verify = False
        else:
            # All ranks use the same deterministic seed and load the same checkpoint path when resuming,
            # so the initial DDP shape/state sync is redundant here and has proven unstable on this cluster.
            skip_ddp_verify = True
        ddp_device_ids: Optional[List[int]] = None
        ddp_output_device: Optional[int] = None
        if dist_info.device.type == "cuda":
            if visible_devices <= 1:
                ddp_device_index = 0
            else:
                ddp_device_index = dist_info.local_rank
            ddp_device_ids = [ddp_device_index]
            ddp_output_device = ddp_device_index
        emit_rank_event(
            dist_info,
            (
                "DDP config "
                f"visible_devices={visible_devices} "
                f"skip_verify={skip_ddp_verify} "
                f"device_ids={ddp_device_ids}"
            ),
        )
        if skip_ddp_verify:
            import torch.nn.parallel.distributed as torch_parallel_distributed

            def _skip_verify_param_shape_across_processes(*args: Any, **kwargs: Any) -> None:
                return None

            def _skip_sync_module_states(*args: Any, **kwargs: Any) -> None:
                return None

            torch_parallel_distributed._verify_param_shape_across_processes = _skip_verify_param_shape_across_processes
            torch_parallel_distributed._sync_module_states = _skip_sync_module_states
            if is_main_process(dist_info):
                log.info("Skipping PyTorch DDP parameter verification and initial module-state sync for this distributed run.")
        emit_rank_event(dist_info, "wrapping model with DDP")
        model = DDP(
            model,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            broadcast_buffers=not skip_ddp_verify,
            find_unused_parameters=False,
            static_graph=True,
        )
        emit_rank_event(dist_info, "DDP ready")

    best_ckpt = checkpoints_dir / "best.pt"
    last_ckpt = checkpoints_dir / "last.pt"
    best_metrics_json = metrics_dir / "best_metrics.json"
    latest_metrics_json = metrics_dir / "latest_metrics.json"

    start_time = time.time()
    emit_rank_event(dist_info, f"starting epoch loop at epoch={start_epoch}")
    for epoch in range(start_epoch, int(train_cfg.epochs) + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, loader_bundle.train_loader, criterion, optimizer, scheduler, scaler, dist_info, cfg, epoch)
        val_metrics = evaluate_model(model, loader_bundle.val_loader, dist_info, cfg, class_names)
        epoch_seconds = time.time() - epoch_start
        current_lr = _current_lrs(optimizer)[0]
        row = {
            "created_at_utc": utc_now_iso(),
            "epoch": int(epoch),
            "train_loss": float(train_metrics["train_loss"]),
            "loss_main": float(train_metrics["loss_main"]),
            "loss_aux": float(train_metrics["loss_aux"]),
            "lr": float(current_lr),
            "seconds": float(epoch_seconds),
            "val_metrics": val_metrics,
        }
        if is_main_process(dist_info):
            save_history_row(history_jsonl, row)
            atomic_json_dump(row, latest_metrics_json)

        current_metric = float(val_metrics.get(str(cfg.get("evaluation", {}).get("save_best_metric", "miou")), val_metrics.get("miou", 0.0)))
        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            if is_main_process(dist_info):
                save_checkpoint(best_ckpt, model, optimizer, scheduler, scaler, epoch, best_metric, best_epoch, history_jsonl, cfg)
                best_payload = dict(row)
                best_payload["best_metric_name"] = str(cfg.get("evaluation", {}).get("save_best_metric", "miou"))
                atomic_json_dump(best_payload, best_metrics_json)
        if is_main_process(dist_info):
            save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler, epoch, best_metric, best_epoch, history_jsonl, cfg)
            log.info(
                "Epoch %d/%d | loss=%.4f | mIoU=%.4f | mDice=%.4f | best=%.4f @ epoch %d | lr=%.3e | elapsed=%s",
                epoch,
                train_cfg.epochs,
                train_metrics["train_loss"],
                val_metrics.get("miou", 0.0),
                val_metrics.get("mdice", 0.0),
                best_metric,
                best_epoch,
                current_lr,
                format_duration_compact(time.time() - start_time),
            )
        barrier_if_needed(dist_info)

    summary: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "run_kind": train_cfg.run_kind,
        "run_dir": str(train_cfg.run_dir),
        "epochs_target": int(train_cfg.epochs),
        "epochs_completed": int(train_cfg.epochs),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_metric_name": str(cfg.get("evaluation", {}).get("save_best_metric", "miou")),
        "best_checkpoint": important_file_metadata(best_ckpt),
        "last_checkpoint": important_file_metadata(last_ckpt),
        "history_jsonl": str(history_jsonl),
        "class_histogram_json": str(train_cfg.class_histogram_json) if train_cfg.class_histogram_json is not None else None,
        "class_weights": loader_bundle.class_weights.tolist() if loader_bundle.class_weights is not None else None,
        "class_names": class_names,
    }
    if is_main_process(dist_info):
        plot_outputs = plot_history(history_jsonl, plots_dir, class_names)
        summary["plots"] = plot_outputs
        if best_metrics_json.exists():
            best_payload = read_json(best_metrics_json)
            confusion = np.asarray(best_payload.get("val_metrics", {}).get("confusion_matrix", []), dtype=np.int64)
            if confusion.size:
                plot_confusion_matrix(confusion, class_names, plots_dir / "best_epoch_confusion_matrix.png")
                plot_per_class_metric(best_payload.get("val_metrics", {}), class_names, "iou", plots_dir / "best_epoch_per_class_iou_bar.png", "Per-class IoU at best epoch")
                plot_per_class_metric(best_payload.get("val_metrics", {}), class_names, "dice", plots_dir / "best_epoch_per_class_dice_bar.png", "Per-class Dice at best epoch")
                write_per_class_metrics_csv(best_payload.get("val_metrics", {}), plots_dir / "best_epoch_per_class_metrics.csv")
            summary["best_metrics"] = best_payload
        summary["duration_seconds"] = float(time.time() - start_time)
        if train_cfg.report_json is not None:
            atomic_json_dump(summary, train_cfg.report_json)
    barrier_if_needed(dist_info)
    cleanup_distributed()
    return summary


# ---------------------------------------------------------------------------
# Eval, qualitative, export
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint_on_manifest(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    split_name: str,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    class_names = list(cfg.get("dataset", {}).get("class_names", ["background", "crop", "weed"]))
    model = build_model(cfg)
    load_model_weights(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    dist_info = DistInfo(enabled=False, rank=0, local_rank=0 if device.type == "cuda" else -1, world_size=1, device=device)
    loader_bundle = build_loaders(
        cfg,
        train_manifest=manifest_path,
        val_manifest=manifest_path,
        test_manifest=None,
        dist_info=dist_info,
        train_batch_total=int(cfg.get("training", {}).get("eval_batch", 8)),
        eval_batch_total=int(cfg.get("training", {}).get("eval_batch", 8)),
        workers_total=int(cfg.get("training", {}).get("workers", 16)),
        train_subset_size=None,
        val_subset_size=None,
    )
    metrics = evaluate_model(model, loader_bundle.val_loader, dist_info, cfg, class_names)
    ensure_dir(output_dir)
    atomic_json_dump(metrics, output_dir / "metrics.json")
    confusion = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64)
    if confusion.size:
        plot_confusion_matrix(confusion, class_names, output_dir / "confusion_matrix.png")
    plot_per_class_metric(metrics, class_names, "iou", output_dir / "per_class_iou.png", f"Per-class IoU ({split_name})")
    plot_per_class_metric(metrics, class_names, "dice", output_dir / "per_class_dice.png", f"Per-class Dice ({split_name})")
    write_per_class_metrics_csv(metrics, output_dir / "per_class_metrics.csv")
    pd.DataFrame(np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64), index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    report = {
        "split": split_name,
        "metrics": metrics,
        "metrics_json": str(output_dir / "metrics.json"),
        "confusion_matrix_png": str(output_dir / "confusion_matrix.png"),
        "per_class_iou_png": str(output_dir / "per_class_iou.png"),
        "per_class_dice_png": str(output_dir / "per_class_dice.png"),
        "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
        "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
    }
    atomic_json_dump(report, output_dir / "report.json")
    return report


@torch.no_grad()
def generate_qualitative_panels(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    limit: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    palette = palette_from_config(cfg)
    alpha = float(cfg.get("qualitative", {}).get("overlay_alpha", 0.45))
    input_size = int(cfg.get("training", {}).get("input_size", 640))
    records = read_manifest_csv(manifest_path)
    if 0 < limit < len(records):
        records = deterministic_subset(records, limit, int(cfg.get("training", {}).get("seed", 0)) + 99)
    eval_transform = EvalSegTransform(size=input_size, mean=cfg.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]), std=cfg.get("normalization", {}).get("std", [0.229, 0.224, 0.225]))
    model = build_model(cfg)
    load_model_weights(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    ensure_dir(output_dir)
    rows: List[Dict[str, Any]] = []
    panel_paths: List[Path] = []
    for record in tqdm(records, desc=f"Qualitative {manifest_path.stem}", unit="img", dynamic_ncols=True, ascii=True):
        image_path = Path(record.image_path)
        mask_path = Path(record.mask_path)
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
        with Image.open(mask_path) as mask_handle:
            mask = mask_handle.convert("L")
        image_tensor, _ = eval_transform(image, mask)
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        panel_path = output_dir / f"{image_path.stem}_panel.png"
        pred_mask_path = output_dir / f"{image_path.stem}_pred.png"
        save_prediction_panel(image_path, mask_path, pred, panel_path, input_size=input_size, palette=palette, alpha=alpha)
        Image.fromarray(pred, mode="L").save(pred_mask_path)
        rows.append({"image_path": str(image_path), "mask_path": str(mask_path), "panel_png": str(panel_path), "pred_png": str(pred_mask_path)})
        panel_paths.append(panel_path)
    contact_sheet_path = output_dir / "contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet_path, columns=2)
    report = {"count": len(rows), "items": rows, "output_dir": str(output_dir), "contact_sheet_png": str(contact_sheet_path)}
    atomic_json_dump(report, output_dir / "qualitative_report.json")
    return report


class BiseNetOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



def export_model_to_onnx(cfg: Dict[str, Any], checkpoint_path: Path, output_path: Path, *, device: torch.device) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_cfg = dict(cfg.get("dataset", {}))
    export_cfg = dict(cfg.get("export", {}))
    input_size = int(export_cfg.get("input_size", cfg.get("training", {}).get("input_size", 640)))
    model = build_model(cfg, num_classes=int(dataset_cfg.get("num_classes", 3)))
    load_model_weights(model, checkpoint_path, map_location="cpu")
    model.eval()
    replace_adaptive_avg_pool(model)
    wrapper = BiseNetOnnxWrapper(model).to(device)
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    dynamic_axes = None
    if bool(export_cfg.get("dynamic_batch", False)) or bool(export_cfg.get("dynamic_spatial", False)):
        dynamic_axes = {"images": {}, "logits": {}}
        if bool(export_cfg.get("dynamic_batch", False)):
            dynamic_axes["images"][0] = "batch"
            dynamic_axes["logits"][0] = "batch"
        if bool(export_cfg.get("dynamic_spatial", False)):
            dynamic_axes["images"][2] = "height"
            dynamic_axes["images"][3] = "width"
            dynamic_axes["logits"][2] = "height"
            dynamic_axes["logits"][3] = "width"
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=int(export_cfg.get("opset", 17)),
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    import onnx

    model_onnx = onnx.load(str(output_path))
    if bool(export_cfg.get("check", True)):
        onnx.checker.check_model(model_onnx)
    return {"onnx_path": str(output_path), "onnx_metadata": important_file_metadata(output_path), "input_size": input_size}


# ---------------------------------------------------------------------------
# Runtime manifests, environment, source snapshot
# ---------------------------------------------------------------------------


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
    optuna_root: Path
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
            optuna_root=runtime_root / "optuna",
            bundle_dir=runtime_root / "bundles",
        )


class StateManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = read_json(path)
        if not self.state:
            self.state = {"schema_version": 1, "created_at_utc": utc_now_iso(), "stages": {}}
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
    snapshot: Dict[str, Any] = {
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
    }
    snapshot["torch_version"] = torch.__version__
    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    snapshot["torch_cuda_version"] = torch.version.cuda
    snapshot["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    atomic_json_dump(snapshot, target_dir / "environment_snapshot.json")
    atomic_json_dump({k: v for k, v in os.environ.items() if k.startswith("SLURM_")}, target_dir / "slurm_environment.json")
    shell_probe_env = sanitized_runtime_env(strip_conda_lib_from_ld=True)
    commands = {
        "uname.txt": ["uname", "-a"],
        "lscpu.txt": ["lscpu"],
        "module_list.txt": ["/bin/bash", "-lc", "module list 2>&1 || ml list 2>&1 || true"],
        "nvidia_smi.txt": ["/bin/bash", "-lc", "nvidia-smi || true"],
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "df_h.txt": ["df", "-h"],
        "env_sorted.txt": ["/bin/bash", "-lc", "env | sort || true"],
    }
    for name, cmd in commands.items():
        try:
            result = run_subprocess(cmd, log, check=False, log_output=False, env=shell_probe_env if cmd[0].endswith("bash") else None)
            write_text(target_dir / name, result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
        except Exception as exc:
            write_text(target_dir / name, f"FAILED: {exc}\n")
    atomic_json_dump({"sys_path": sys.path, "package_root": str(PACKAGE_ROOT)}, target_dir / "python_paths.json")



def archive_source_snapshot(paths: Paths, cfg_path: Path, log: logging.Logger) -> Dict[str, Any]:
    target = ensure_dir(paths.source_snapshot_dir)
    src_root = PACKAGE_ROOT
    selected: List[Path] = []
    for rel in [Path("requirements.txt"), Path("config"), Path("docs"), Path("scripts"), Path("vendor"), Path("README.md"), Path("README_STAGE2_TRAINING_CLUSTER_BISENETV2.md"), Path("MODEL_ASSET_NOTE.txt"), Path("multigpu.txt")]:
        src = src_root / rel
        if src.is_dir():
            for path in src.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    selected.append(path)
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
    with (target / "source_snapshot_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(copied)
    log.info("Saved source snapshot to %s", target)
    return {"source_snapshot_dir": str(target), "file_count": len(copied)}



def build_runtime_manifests(cfg: Dict[str, Any], paths: Paths, state: StateManager, log: logging.Logger) -> Dict[str, Any]:
    stage_name = "prepare_runtime_manifests"
    report_path = paths.reports_dir / "runtime_dataset_report.json"
    if state.is_done(stage_name) and report_path.exists():
        return read_json(report_path)
    state.mark_started(stage_name)
    try:
        ensure_dir(paths.runtime_manifests_dir)
        records_by_split = discover_dataset_records(paths.dataset_root, log=log)
        for split, records in records_by_split.items():
            for idx, rec in enumerate(tqdm(records, desc=f"Probe sizes {split}", unit="img", dynamic_ncols=True, ascii=True)):
                if rec.image_width is None or rec.image_height is None:
                    width, height = _probe_image_size(Path(rec.image_path))
                    records[idx] = SampleRecord(split=rec.split, image_path=rec.image_path, mask_path=rec.mask_path, image_width=width, image_height=height)

        total_expected = cfg.get("dataset", {}).get("expected_total_samples")
        actual_total = sum(len(records) for records in records_by_split.values())
        if total_expected is not None and int(total_expected) != actual_total:
            raise ValueError(
                f"Dataset total sample mismatch: expected_total_samples={int(total_expected)} actual_total={actual_total}"
            )

        split_counts_expected = cfg.get("dataset", {}).get("expected_split_counts")
        if isinstance(split_counts_expected, dict):
            for split, expected in split_counts_expected.items():
                actual = len(records_by_split.get(split, []))
                if expected is not None and int(expected) != actual:
                    log.warning("Split count mismatch for %s: expected=%s actual=%s", split, expected, actual)

        train_manifest = paths.runtime_manifests_dir / "train.csv"
        val_manifest = paths.runtime_manifests_dir / "val.csv"
        test_manifest = paths.runtime_manifests_dir / "test.csv"
        write_manifest_csv(records_by_split["train"], train_manifest)
        write_manifest_csv(records_by_split["val"], val_manifest)
        write_manifest_csv(records_by_split["test"], test_manifest)

        dataset_cfg = dict(cfg.get("dataset", {}))
        training_cfg = dict(cfg.get("training", {}))
        smoke_train_k = int(dataset_cfg.get("smoke_train_subset_size", min(256, len(records_by_split["train"]))))
        smoke_val_k = int(dataset_cfg.get("smoke_val_subset_size", min(64, len(records_by_split["val"]))))
        jetson_pilot_k = int(dataset_cfg.get("jetson_pilot_val_subset_size", min(128, len(records_by_split["val"]))))
        qual_val_k = int(cfg.get("qualitative", {}).get("val_subset_size", min(16, len(records_by_split["val"]))))
        qual_test_k = int(cfg.get("qualitative", {}).get("test_subset_size", min(32, len(records_by_split["test"]))))
        seed = int(training_cfg.get("seed", 0))

        smoke_train = deterministic_subset(records_by_split["train"], smoke_train_k, seed)
        smoke_val = deterministic_subset(records_by_split["val"], smoke_val_k, seed + 1)
        pilot_val = deterministic_subset(records_by_split["val"], jetson_pilot_k, seed + 2)
        paper_val = deterministic_subset(records_by_split["val"], qual_val_k, seed + 3)
        paper_test = deterministic_subset(records_by_split["test"], qual_test_k, seed + 4)
        write_manifest_csv(smoke_train, paths.runtime_manifests_dir / "smoke_train.csv")
        write_manifest_csv(smoke_val, paths.runtime_manifests_dir / "smoke_val.csv")
        write_manifest_csv(pilot_val, paths.runtime_manifests_dir / "jetson_pilot_val.csv")
        write_manifest_csv(paper_val, paths.runtime_manifests_dir / "paper_val_examples.csv")
        write_manifest_csv(paper_test, paths.runtime_manifests_dir / "paper_test_examples.csv")

        summary = summarize_records(records_by_split)
        summary["created_at_utc"] = utc_now_iso()
        summary["dataset_root"] = str(paths.dataset_root)
        summary["class_names"] = list(cfg.get("dataset", {}).get("class_names", ["background", "crop", "weed"]))
        atomic_json_dump(summary, paths.reports_dir / "dataset_summary.json")
        plot_split_counts(summary, paths.paper_assets_dir / "dataset_split_counts.png")

        report = {
            "dataset_root": str(paths.dataset_root),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_manifest": str(test_manifest),
            "smoke_train_manifest": str(paths.runtime_manifests_dir / "smoke_train.csv"),
            "smoke_val_manifest": str(paths.runtime_manifests_dir / "smoke_val.csv"),
            "jetson_pilot_val_manifest": str(paths.runtime_manifests_dir / "jetson_pilot_val.csv"),
            "paper_val_manifest": str(paths.runtime_manifests_dir / "paper_val_examples.csv"),
            "paper_test_manifest": str(paths.runtime_manifests_dir / "paper_test_examples.csv"),
            "summary": summary,
        }
        atomic_json_dump(report, report_path)
        state.mark_done(stage_name, report)
        return report
    except Exception as exc:
        state.mark_failed(stage_name, exc)
        raise



def resolve_tuned_params_path(cfg: Dict[str, Any], paths: Paths) -> Optional[Path]:
    training_cfg = dict(cfg.get("training", {}))
    explicit = training_cfg.get("tuned_params_yaml")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if path.exists():
            return path
    if bool(training_cfg.get("use_tuned_params_if_available", True)):
        candidate = paths.optuna_root / "best_hyperparameters.yaml"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Fake dataset helper for local debugging
# ---------------------------------------------------------------------------



def create_fake_dataset(root: Path, *, total_train: int = 8, total_val: int = 4, total_test: int = 4, image_size: int = 128) -> None:
    for split, count in [("train", total_train), ("val", total_val), ("test", total_test)]:
        image_dir = root / "images" / split
        mask_dir = root / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            rng = np.random.default_rng(seed=idx + (0 if split == "train" else 100 if split == "val" else 200))
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            image[..., 1] = rng.integers(60, 180, size=(image_size, image_size), dtype=np.uint8)
            image[..., 0] = rng.integers(20, 120, size=(image_size, image_size), dtype=np.uint8)
            image[..., 2] = rng.integers(10, 80, size=(image_size, image_size), dtype=np.uint8)
            yy, xx = np.mgrid[:image_size, :image_size]
            center_x = rng.integers(image_size // 4, 3 * image_size // 4)
            center_y = rng.integers(image_size // 4, 3 * image_size // 4)
            radius = rng.integers(image_size // 10, image_size // 5)
            crop_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
            weed_mask = ((xx + yy + idx * 7) % max(8, image_size // 12) < 3) & (~crop_mask)
            image[crop_mask] = np.array([60, 180, 60], dtype=np.uint8)
            image[weed_mask] = np.array([180, 60, 60], dtype=np.uint8)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            mask[crop_mask] = 1
            mask[weed_mask] = 2
            stem = f"{split}_{idx:04d}"
            Image.fromarray(image).save(image_dir / f"{stem}.png")
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")
    write_text(root / "runtime_input_policy.yaml", "policy: local_debug\n")
    write_text(root / "DATASET_LOCK.yaml", "dataset: fake_cropandweed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilities for the stage-2 BiSeNetV2 Training_Cluster package")
    parser.add_argument("--make-fake-dataset", type=str, default=None, help="Create a tiny fake segmentation dataset at the given path")
    args = parser.parse_args()
    if args.make_fake_dataset:
        create_fake_dataset(Path(args.make_fake_dataset).expanduser().resolve())
        print(args.make_fake_dataset)
