#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = [256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1152]
CLASS_NAMES = ["background", "crop", "weed"]
PROBE_STAGES = ("make_onnx", "build_fp16", "trtexec_smoke", "benchmark_smoke", "accuracy")
OOM_RE = re.compile(r"out of memory|oom|killed process|cuda.*memory|failed allocation", re.IGNORECASE)
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")


@dataclass
class CmdResult:
    returncode: int
    timed_out: bool
    duration_s: float
    log_path: str
    oom_in_dmesg: bool
    peak_ram_used_mb: Optional[int]
    peak_ram_total_mb: Optional[int]
    peak_swap_used_mb: Optional[int]
    peak_swap_total_mb: Optional[int]


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, msg: str) -> None:
        sys.stdout.write(msg)
        sys.stdout.flush()
        self.handle.write(msg)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ProgressReporter:
    def __init__(self, total_units: int, label: str, heartbeat_s: float = 60.0):
        self.total_units = max(1, int(total_units))
        self.label = label
        self.completed_units = 0
        self.started_at = time.monotonic()
        self.last_heartbeat_at = 0.0
        self.heartbeat_s = max(5.0, float(heartbeat_s))

    def _eta_s(self) -> Optional[float]:
        if self.completed_units <= 0:
            return None
        elapsed = time.monotonic() - self.started_at
        remaining = max(0, self.total_units - self.completed_units)
        return (elapsed / float(self.completed_units)) * float(remaining)

    def _line(self, event: str, detail: str) -> str:
        elapsed = time.monotonic() - self.started_at
        pct = (float(self.completed_units) / float(self.total_units)) * 100.0
        remaining = max(0, self.total_units - self.completed_units)
        return (
            f"[PROGRESS] {self.label} {pct:6.2f}% "
            f"({self.completed_units}/{self.total_units} units done, {remaining} left, "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(self._eta_s())}) "
            f"{event}: {detail}"
        )

    def start(self, detail: str) -> None:
        print(self._line("START", detail), flush=True)

    def heartbeat(self, detail: str) -> None:
        current = time.monotonic()
        if current - self.last_heartbeat_at < self.heartbeat_s:
            return
        self.last_heartbeat_at = current
        print(self._line("RUNNING", detail), flush=True)

    def done(self, detail: str, status: str = "done", units: int = 1) -> None:
        self.completed_units = min(self.total_units, self.completed_units + max(0, int(units)))
        print(self._line(status.upper(), detail), flush=True)


def run_text(cmd: Sequence[str], timeout_s: int = 10) -> str:
    try:
        cp = subprocess.run(list(cmd), text=True, capture_output=True, check=False, timeout=timeout_s)
        return (cp.stdout or "") + (cp.stderr or "")
    except Exception:
        return ""


def dmesg_lines() -> List[str]:
    text = run_text(["dmesg", "-T"], timeout_s=10)
    return text.splitlines()


def new_dmesg_lines(before: List[str], after: List[str]) -> List[str]:
    if len(after) >= len(before) and after[: len(before)] == before:
        return after[len(before) :]
    # Fallback: compare suffix-ish by line content. This avoids treating old OOMs as new when possible.
    before_set = set(before)
    return [line for line in after if line not in before_set]


def parse_tegrastats(path: Path) -> Dict[str, Optional[int]]:
    peak_ram_used = None
    peak_ram_total = None
    peak_swap_used = None
    peak_swap_total = None
    if not path.exists():
        return {
            "peak_ram_used_mb": None,
            "peak_ram_total_mb": None,
            "peak_swap_used_mb": None,
            "peak_swap_total_mb": None,
        }
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RAM_RE.search(line)
        if m:
            used = int(m.group(1))
            total = int(m.group(2))
            if peak_ram_used is None or used > peak_ram_used:
                peak_ram_used = used
                peak_ram_total = total
        m = SWAP_RE.search(line)
        if m:
            used = int(m.group(1))
            total = int(m.group(2))
            if peak_swap_used is None or used > peak_swap_used:
                peak_swap_used = used
                peak_swap_total = total
    return {
        "peak_ram_used_mb": peak_ram_used,
        "peak_ram_total_mb": peak_ram_total,
        "peak_swap_used_mb": peak_swap_used,
        "peak_swap_total_mb": peak_swap_total,
    }


def start_tegrastats(log_path: Path, interval_ms: int = 500) -> Optional[subprocess.Popen]:
    exe = shutil.which("tegrastats")
    if not exe:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.Popen([exe, "--interval", str(interval_ms), "--logfile", str(log_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        return None


def stop_process_tree(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def run_guarded(
    cmd: Sequence[str],
    log_path: Path,
    timeout_s: int,
    tegrastats_log: Optional[Path] = None,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    progress_heartbeat: Optional[Callable[[], None]] = None,
) -> CmdResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = dmesg_lines()
    tg = start_tegrastats(tegrastats_log) if tegrastats_log else None
    timed_out = False
    rc = 0
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[command] {' '.join(str(x) for x in cmd)}\n")
        log.write(f"[start_utc] {now()}\n")
        log.flush()
        try:
            proc = subprocess.Popen(list(cmd), stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(cwd) if cwd else None, env=env, start_new_session=True)
            old_sigint = signal.getsignal(signal.SIGINT)
            old_sigterm = signal.getsignal(signal.SIGTERM)

            def _terminate_child_and_exit(signum: int, _frame: Any) -> None:
                log.write(f"\n[SIGNAL] received {signum}; terminating child process group {proc.pid}\n")
                log.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                stop_process_tree(tg)
                raise SystemExit(128 + int(signum))

            signal.signal(signal.SIGINT, _terminate_child_and_exit)
            signal.signal(signal.SIGTERM, _terminate_child_and_exit)
            deadline = time.monotonic() + float(timeout_s) if timeout_s and timeout_s > 0 else None
            heartbeat_s = max(5.0, float(os.environ.get("PROGRESS_HEARTBEAT_S", "60")))
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    log.write(f"\n[TIMEOUT] timeout_s={timeout_s}; killing process group\n")
                    log.flush()
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            proc.kill()
                        proc.wait(timeout=10)
                    rc = 124
                    break
                wait_s = heartbeat_s
                if deadline is not None:
                    wait_s = min(wait_s, max(0.1, deadline - time.monotonic()))
                try:
                    rc = proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    if progress_heartbeat is not None:
                        progress_heartbeat()
        except Exception as exc:
            log.write(f"\n[EXCEPTION] {exc}\n")
            rc = 125
        finally:
            if "old_sigint" in locals():
                signal.signal(signal.SIGINT, old_sigint)
            if "old_sigterm" in locals():
                signal.signal(signal.SIGTERM, old_sigterm)
            log.write(f"[end_utc] {now()}\n")
    stop_process_tree(tg)
    after = dmesg_lines()
    new_lines = new_dmesg_lines(before, after)
    dmesg_new_path = log_path.with_suffix(log_path.suffix + ".dmesg_new.txt")
    dmesg_new_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    oom = any(OOM_RE.search(line) for line in new_lines)
    mem = parse_tegrastats(tegrastats_log) if tegrastats_log else {
        "peak_ram_used_mb": None,
        "peak_ram_total_mb": None,
        "peak_swap_used_mb": None,
        "peak_swap_total_mb": None,
    }
    duration_s = time.monotonic() - start
    return CmdResult(
        returncode=rc,
        timed_out=timed_out,
        duration_s=float(duration_s),
        log_path=str(log_path),
        oom_in_dmesg=oom,
        peak_ram_used_mb=mem["peak_ram_used_mb"],
        peak_ram_total_mb=mem["peak_ram_total_mb"],
        peak_swap_used_mb=mem["peak_swap_used_mb"],
        peak_swap_total_mb=mem["peak_swap_total_mb"],
    )


def resolve_trtexec() -> str:
    candidates = []
    if os.environ.get("TRTEXEC_PATH"):
        candidates.append(Path(os.environ["TRTEXEC_PATH"]))
    hit = shutil.which("trtexec")
    if hit:
        candidates.append(Path(hit))
    candidates.extend([Path("/usr/src/tensorrt/bin/trtexec"), Path("/usr/src/tensorrt/samples/trtexec")])
    local = os.environ.get("LOCAL_TRTEXEC_PATH")
    if local:
        candidates.insert(0, Path(local))
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c.resolve())
    raise RuntimeError("trtexec not found. Set TRTEXEC_PATH or LOCAL_TRTEXEC_PATH.")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    per_class = data.get("per_class") or (data.get("metrics", {}) or {}).get("per_class") or {}
    return {
        "miou": data.get("metrics/mIoU") or (data.get("metrics", {}) or {}).get("miou"),
        "mdice": data.get("metrics/mDice") or (data.get("metrics", {}) or {}).get("mdice"),
        "pixel_accuracy": data.get("metrics/pixel_accuracy") or (data.get("metrics", {}) or {}).get("pixel_accuracy"),
        "foreground_miou": data.get("metrics/foreground_mIoU") or (data.get("metrics", {}) or {}).get("foreground_miou"),
        "foreground_mdice": data.get("metrics/foreground_mDice") or (data.get("metrics", {}) or {}).get("foreground_mdice"),
        "crop_iou": (per_class.get("crop") or {}).get("iou"),
        "weed_iou": (per_class.get("weed") or {}).get("iou"),
        "background_iou": (per_class.get("background") or {}).get("iou"),
        "crop_dice": (per_class.get("crop") or {}).get("dice"),
        "weed_dice": (per_class.get("weed") or {}).get("dice"),
        "background_dice": (per_class.get("background") or {}).get("dice"),
    }


def metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def accuracy_pass(metrics: Dict[str, Any], ref: Dict[str, Any], miou_drop: float, dice_drop: float, class_drop: float) -> tuple[bool, str]:
    needed = ["miou", "mdice", "crop_iou", "weed_iou"]
    for k in needed:
        if metric_float(metrics.get(k)) is None or metric_float(ref.get(k)) is None:
            return False, f"missing_{k}"
    checks = [
        ("miou", miou_drop),
        ("mdice", dice_drop),
        ("crop_iou", class_drop),
        ("weed_iou", class_drop),
    ]
    failures = []
    for key, allowed_drop in checks:
        cur = metric_float(metrics[key])
        base = metric_float(ref[key])
        assert cur is not None and base is not None
        threshold = base - allowed_drop
        # no foreground collapse guard for crop/weed
        if key in {"crop_iou", "weed_iou"}:
            threshold = max(0.01, threshold)
        if cur < threshold:
            failures.append(f"{key}_drop={base-cur:.6f}_gt_{allowed_drop}")
    if failures:
        return False, ";".join(failures)
    return True, "pass"


def memory_pass(result: CmdResult, max_ram_fraction: float, require_swap_zero: bool) -> tuple[bool, str]:
    if result.returncode != 0:
        return False, f"returncode_{result.returncode}"
    if result.timed_out:
        return False, "timeout"
    if result.oom_in_dmesg:
        return False, "dmesg_oom"
    if require_swap_zero and result.peak_swap_used_mb is not None and result.peak_swap_used_mb > 0:
        return False, f"swap_used_{result.peak_swap_used_mb}MB"
    if result.peak_ram_used_mb is not None and result.peak_ram_total_mb:
        frac = float(result.peak_ram_used_mb) / float(result.peak_ram_total_mb)
        if frac > max_ram_fraction:
            return False, f"ram_fraction_{frac:.3f}_gt_{max_ram_fraction:.3f}"
    return True, "pass"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="SegFormer-B0 FP16 resolution discovery: S_min accuracy and S_max memory-safe size.")
    ap.add_argument("--source-onnx", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--board-label", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--patch-static-onnx", action="store_true", default=True)
    ap.add_argument("--build-timeout-s", type=int, default=1800)
    ap.add_argument("--trtexec-timeout-s", type=int, default=180)
    ap.add_argument("--benchmark-timeout-s", type=int, default=300)
    ap.add_argument("--accuracy-timeout-s", type=int, default=1200)
    ap.add_argument("--max-ram-fraction", type=float, default=0.90)
    ap.add_argument("--allow-swap", action="store_true")
    ap.add_argument("--miou-drop", type=float, default=0.05)
    ap.add_argument("--dice-drop", type=float, default=0.05)
    ap.add_argument("--class-iou-drop", type=float, default=0.10)
    args = ap.parse_args()

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_root / "discovery_console.log")
    try:
        tee.write(f"[INFO] SegFormer-B0 FP16 resolution discovery started at {now()}\n")
        tee.write(f"[INFO] board_label={args.board_label}\n")
        tee.write(f"[INFO] sizes={args.sizes}\n")
        source_onnx = Path(args.source_onnx).expanduser().resolve()
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
        sizes = sorted(set(int(s) for s in args.sizes))
        rows: List[Dict[str, Any]] = []
        trtexec = resolve_trtexec()
        reference_metrics: Optional[Dict[str, Any]] = None
        stopped_large_after_fail = False
        progress = ProgressReporter(
            total_units=len(sizes) * len(PROBE_STAGES),
            label="SegFormer-B0 FP16 discovery",
            heartbeat_s=float(os.environ.get("PROGRESS_HEARTBEAT_S", "60")),
        )

        def skip_remaining(size_value: int, completed_for_size: int, reason: str) -> None:
            remaining = max(0, len(PROBE_STAGES) - int(completed_for_size))
            if remaining:
                progress.done(f"size {size_value}: skipped {remaining} remaining probe stages ({reason})", "skipped", remaining)

        for size in sizes:
            if stopped_large_after_fail:
                rows.append({"size": size, "status": "skipped_after_first_large_fail", "board_label": args.board_label})
                progress.done(f"size {size}: skipped after first large memory/build/smoke failure", "skipped", len(PROBE_STAGES))
                continue
            tee.write(f"\n[INFO] ===== size {size} =====\n")
            size_root = out_root / f"s{size}"
            onnx_dir = size_root / "onnx"
            engine_dir = size_root / "engine"
            logs_dir = size_root / "logs"
            acc_dir = size_root / "accuracy"
            run_dir = size_root / "benchmark_smoke"
            onnx_path = onnx_dir / f"segformerb0_s{size}.onnx"
            make_meta = onnx_dir / "make_onnx_meta.json"
            row: Dict[str, Any] = {
                "board_label": args.board_label,
                "model": "SegFormer-B0",
                "task": "semantic_segmentation",
                "precision": "fp16",
                "size": size,
                "timestamp_utc": now(),
            }
            completed_for_size = 0

            # 1) ONNX preparation
            progress.start(f"size {size}: prepare ONNX")
            make_cmd = [
                sys.executable,
                str(PKG_ROOT / "scripts" / "make_segformerb0_onnx_for_size.py"),
                "--source-onnx", str(source_onnx),
                "--output-onnx", str(onnx_path),
                "--imgsz", str(size),
                "--out-json", str(make_meta),
            ]
            if args.patch_static_onnx:
                make_cmd.append("--patch-static-shape")
            make_res = run_guarded(
                make_cmd,
                logs_dir / "make_onnx.log",
                timeout_s=300,
                progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: prepare ONNX still running"),
            )
            row.update({f"make_onnx_{k}": v for k, v in asdict(make_res).items()})
            row["onnx_path"] = str(onnx_path)
            if make_res.returncode != 0 or not onnx_path.exists():
                row["status"] = "onnx_failed"
                rows.append(row)
                progress.done(f"size {size}: prepare ONNX failed", "failed")
                completed_for_size += 1
                skip_remaining(size, completed_for_size, row["status"])
                if size > 640:
                    stopped_large_after_fail = True
                write_csv(out_root / "size_probe_results.csv", rows)
                continue
            progress.done(f"size {size}: prepare ONNX ok")
            completed_for_size += 1
            try:
                row["onnx_method"] = load_json(make_meta).get("method")
            except Exception:
                row["onnx_method"] = None

            # 2) Build FP16 engine
            progress.start(f"size {size}: build FP16 TensorRT engine")
            build_cmd = ["bash", str(PKG_ROOT / "scripts" / "build_engine_fp16.sh"), str(onnx_path), str(engine_dir), str(size), str(args.workspace_mb)]
            build_res = run_guarded(
                build_cmd,
                logs_dir / "build_fp16.log",
                timeout_s=args.build_timeout_s,
                tegrastats_log=logs_dir / "build_fp16_tegrastats.log",
                cwd=PKG_ROOT,
                progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: FP16 TensorRT build still running"),
            )
            row.update({f"build_{k}": v for k, v in asdict(build_res).items()})
            engine_path = engine_dir / f"{onnx_path.stem}_fp16.engine"
            row["engine_path"] = str(engine_path)
            if build_res.returncode != 0 or build_res.oom_in_dmesg or not engine_path.exists():
                row["status"] = "build_failed"
                row["memory_pass"] = False
                row["memory_reason"] = "build_failed_or_oom"
                rows.append(row)
                progress.done(f"size {size}: build FP16 failed", "failed")
                completed_for_size += 1
                skip_remaining(size, completed_for_size, row["status"])
                if size > 640:
                    stopped_large_after_fail = True
                write_csv(out_root / "size_probe_results.csv", rows)
                continue
            progress.done(f"size {size}: build FP16 ok")
            completed_for_size += 1

            # 3) trtexec smoke
            progress.start(f"size {size}: trtexec smoke")
            trt_cmd = [trtexec, f"--loadEngine={engine_path}", "--duration=5", "--warmUp=250", "--iterations=20", "--useCudaGraph", "--separateProfileRun"]
            trt_res = run_guarded(
                trt_cmd,
                logs_dir / "trtexec_smoke.log",
                timeout_s=args.trtexec_timeout_s,
                tegrastats_log=logs_dir / "trtexec_smoke_tegrastats.log",
                progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: trtexec smoke still running"),
            )
            row.update({f"trtexec_{k}": v for k, v in asdict(trt_res).items()})
            trt_mem_ok, trt_mem_reason = memory_pass(trt_res, args.max_ram_fraction, require_swap_zero=not args.allow_swap)
            row["trtexec_memory_pass"] = trt_mem_ok
            row["trtexec_memory_reason"] = trt_mem_reason
            if not trt_mem_ok:
                row["status"] = "trtexec_smoke_failed"
                row["memory_pass"] = False
                row["memory_reason"] = trt_mem_reason
                rows.append(row)
                progress.done(f"size {size}: trtexec smoke failed", "failed")
                completed_for_size += 1
                skip_remaining(size, completed_for_size, row["status"])
                if size > 640:
                    stopped_large_after_fail = True
                write_csv(out_root / "size_probe_results.csv", rows)
                continue
            progress.done(f"size {size}: trtexec smoke ok")
            completed_for_size += 1

            # 4) benchmark smoke with random TensorRT input
            progress.start(f"size {size}: benchmark smoke")
            bench_cmd = [
                sys.executable, str(PKG_ROOT / "jetson" / "run_benchmark.py"),
                "--model", str(engine_path),
                "--dataset-root", str(dataset_root),
                "--split-name", str(args.split_name),
                "--imgsz", str(size),
                "--precision-label", "fp16",
                "--run-name", f"segformerb0_fp16_s{size}_resolution_smoke",
                "--board-name", args.board_label,
                "--run-variant", "resolution_smoke",
                "--repeats", "1",
                "--warmup-seconds", "0",
                "--warmup-images", "4",
                "--min-images", "20",
                "--min-duration-s", "8",
                "--skip-accuracy-pass",
                "--no-save-predictions",
                "--no-save-per-image-outputs",
                "--no-noise-guard",
                "--output-root", str(run_dir),
            ]
            if manifest is not None:
                bench_cmd.extend(["--manifest", str(manifest)])
            bench_res = run_guarded(
                bench_cmd,
                logs_dir / "benchmark_smoke.log",
                timeout_s=args.benchmark_timeout_s,
                tegrastats_log=logs_dir / "benchmark_smoke_tegrastats.log",
                cwd=PKG_ROOT,
                progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: benchmark smoke still running"),
            )
            row.update({f"benchmark_{k}": v for k, v in asdict(bench_res).items()})
            bench_mem_ok, bench_mem_reason = memory_pass(bench_res, args.max_ram_fraction, require_swap_zero=not args.allow_swap)
            row["benchmark_memory_pass"] = bench_mem_ok
            row["benchmark_memory_reason"] = bench_mem_reason
            if not bench_mem_ok:
                row["status"] = "benchmark_smoke_failed"
                row["memory_pass"] = False
                row["memory_reason"] = bench_mem_reason
                rows.append(row)
                progress.done(f"size {size}: benchmark smoke failed", "failed")
                completed_for_size += 1
                skip_remaining(size, completed_for_size, row["status"])
                if size > 640:
                    stopped_large_after_fail = True
                write_csv(out_root / "size_probe_results.csv", rows)
                continue
            progress.done(f"size {size}: benchmark smoke ok")
            completed_for_size += 1

            # 5) accuracy on validation/test split
            progress.start(f"size {size}: segmentation accuracy")
            acc_cmd = [
                sys.executable, str(PKG_ROOT / "scripts" / "run_segformerb0_engine_accuracy.py"),
                "--engine", str(engine_path),
                "--dataset-root", str(dataset_root),
                "--split-name", str(args.split_name),
                "--imgsz", str(size),
                "--output-dir", str(acc_dir),
                "--out-json", str(acc_dir / "val_metrics.json"),
            ]
            if manifest is not None:
                acc_cmd.extend(["--manifest", str(manifest)])
            acc_res = run_guarded(
                acc_cmd,
                logs_dir / "accuracy.log",
                timeout_s=args.accuracy_timeout_s,
                tegrastats_log=logs_dir / "accuracy_tegrastats.log",
                cwd=PKG_ROOT,
                progress_heartbeat=lambda s=size: progress.heartbeat(f"size {s}: segmentation accuracy still running"),
            )
            row.update({f"accuracy_{k}": v for k, v in asdict(acc_res).items()})
            if acc_res.returncode != 0 or acc_res.oom_in_dmesg or not (acc_dir / "val_metrics.json").exists():
                row["status"] = "accuracy_failed"
                row["memory_pass"] = bool(bench_mem_ok and trt_mem_ok)
                row["memory_reason"] = "accuracy_failed_or_oom"
                row["smin_candidate"] = False
                row["smax_candidate"] = bool(row["memory_pass"])
                row["overall_pass"] = False
                rows.append(row)
                progress.done(f"size {size}: segmentation accuracy failed", "failed")
                completed_for_size += 1
                if size > 640:
                    stopped_large_after_fail = False
                write_csv(out_root / "size_probe_results.csv", rows)
                continue
            metrics = extract_metrics(load_json(acc_dir / "val_metrics.json"))
            row.update(metrics)
            row["memory_pass"] = True
            row["memory_reason"] = "pass"

            if size == 640:
                reference_metrics = metrics
                row["accuracy_pass"] = True
                row["accuracy_reason"] = "reference"
            elif reference_metrics is not None:
                ok, reason = accuracy_pass(metrics, reference_metrics, args.miou_drop, args.dice_drop, args.class_iou_drop)
                row["accuracy_pass"] = ok
                row["accuracy_reason"] = reason
            else:
                row["accuracy_pass"] = None
                row["accuracy_reason"] = "reference_640_not_seen_yet"

            row["smin_candidate"] = bool(row.get("memory_pass")) and (row.get("accuracy_pass") is not False)
            row["smax_candidate"] = bool(row.get("memory_pass"))
            row["overall_pass"] = bool(row.get("smin_candidate"))
            row["status"] = "passed_probe" if row["smax_candidate"] else "memory_reject"
            if row.get("accuracy_pass") is False:
                row["status"] = "accuracy_reject"
            rows.append(row)
            progress.done(f"size {size}: segmentation accuracy ok")
            completed_for_size += 1
            write_csv(out_root / "size_probe_results.csv", rows)

        # If the sorted order means small sizes were evaluated before 640, recompute accuracy pass now that 640 exists.
        ref_row = next((r for r in rows if int(r.get("size", -1)) == 640 and r.get("miou") is not None), None)
        if ref_row:
            ref_metrics = {k: ref_row.get(k) for k in ["miou", "mdice", "crop_iou", "weed_iou", "pixel_accuracy", "foreground_miou", "foreground_mdice"]}
            for r in rows:
                if r.get("miou") is not None and int(r.get("size", 0)) != 640:
                    ok, reason = accuracy_pass(r, ref_metrics, args.miou_drop, args.dice_drop, args.class_iou_drop)
                    r["accuracy_pass"] = ok
                    r["accuracy_reason"] = reason
                    r["smin_candidate"] = bool(r.get("memory_pass")) and ok
                    r["smax_candidate"] = bool(r.get("memory_pass"))
                    r["overall_pass"] = bool(r.get("smin_candidate"))
                    if not ok:
                        r["status"] = "accuracy_reject"
                    elif r.get("memory_pass"):
                        r["status"] = "passed_probe"
        write_csv(out_root / "size_probe_results.csv", rows)

        small_passing = sorted(int(r["size"]) for r in rows if r.get("smin_candidate") is True and int(r.get("size", 0)) <= 640) if ref_row else []
        large_passing = sorted(int(r["size"]) for r in rows if r.get("smax_candidate") is True and int(r.get("size", 0)) >= 640)
        s_min = min(small_passing) if small_passing else None
        s_max = max(large_passing) if large_passing else None
        summary = {
            "status": "ok" if s_min is not None and s_max is not None else "incomplete_selection",
            "model": "SegFormer-B0",
            "task": "semantic_segmentation",
            "board_label": args.board_label,
            "precision_discovery": "fp16",
            "split_name": args.split_name,
            "sizes_tested": args.sizes,
            "reference_size": 640,
            "reference_metrics": {k: ref_row.get(k) for k in ref_row.keys() if k in {"miou", "mdice", "foreground_miou", "foreground_mdice", "pixel_accuracy", "background_iou", "crop_iou", "weed_iou", "background_dice", "crop_dice", "weed_dice"}} if ref_row else None,
            "pass_rules": {
                "miou_drop_max": args.miou_drop,
                "dice_drop_max": args.dice_drop,
                "crop_iou_drop_max": args.class_iou_drop,
                "weed_iou_drop_max": args.class_iou_drop,
                "max_ram_fraction": args.max_ram_fraction,
                "swap_required_zero": not args.allow_swap,
            },
            "S_min": s_min,
            "S_max_board": s_max,
            "S_min_selection_rule": "smallest size <= 640 passing mIoU, mDice, crop IoU, weed IoU, memory/build/smoke, and no foreground collapse checks",
            "S_max_selection_rule": "largest size >= 640 passing build, trtexec smoke, benchmark smoke, OOM, swap, and RAM checks; accuracy is recorded but not used for S_max",
            "results_csv": str(out_root / "size_probe_results.csv"),
            "output_root": str(out_root),
            "started_utc": rows[0].get("timestamp_utc") if rows else None,
            "finished_utc": now(),
        }
        (out_root / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress.done("FP16 discovery finalized", "complete", progress.total_units - progress.completed_units)
        tee.write("\n[INFO] Selection summary:\n")
        tee.write(json.dumps(summary, indent=2) + "\n")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
