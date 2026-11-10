from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

def _prefer_runtime_python_site_packages() -> None:
    major = sys.version_info.major
    minor = sys.version_info.minor
    env_prefix = os.environ.get('CONDA_PREFIX') or os.environ.get('VIRTUAL_ENV') or sys.prefix
    # Preserve the host's working NumPy/OpenCV ABI pairing by keeping user-site
    # ahead of the active environment when both exist.
    user_site = Path.home() / '.local' / 'lib' / f'python{major}.{minor}' / 'site-packages'
    env_site = Path(env_prefix) / 'lib' / f'python{major}.{minor}' / 'site-packages'
    candidate_paths = [env_site, user_site]

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        candidate_str = str(candidate.resolve())
        try:
            sys.path.remove(candidate_str)
        except ValueError:
            pass
        sys.path.insert(0, candidate_str)

    for prefix in ('cv2', 'google', 'numpy', 'sympy', 'torch', 'torchvision', 'ultralytics'):
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                sys.modules.pop(name, None)


_prefer_runtime_python_site_packages()


def _load_ultralytics_yolo():
    from ultralytics import YOLO
    return YOLO

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.internal_telemetry import InternalSampler, ProcessMemorySampler, TegrastatsLogger
    from jetson.local_paths import (
        default_output_root,
        resolve_accuracy_model_path as resolve_local_accuracy_model_path,
        resolve_data_yaml_path,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
        resolve_model_path as resolve_local_model_path,
        resolve_output_root,
    )
    from jetson.oled_status import BenchmarkOLED
    from jetson.noise_guard import NoiseGuard, NoiseGuardConfig
    from jetson.utils import (
        bootstrap_ci,
        default_test_manifest,
        dump_json,
        dump_yaml,
        ensure_dir,
        list_images_from_manifest,
        load_yaml,
        monotonic_ns,
        overlay_nested,
        run_command,
        sha256_file,
        utc_now_iso,
        wallclock_ns,
    )
else:
    from .internal_telemetry import InternalSampler, ProcessMemorySampler, TegrastatsLogger
    from .local_paths import (
        default_output_root,
        resolve_accuracy_model_path as resolve_local_accuracy_model_path,
        resolve_data_yaml_path,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
        resolve_model_path as resolve_local_model_path,
        resolve_output_root,
    )
    from .oled_status import BenchmarkOLED
    from .noise_guard import NoiseGuard, NoiseGuardConfig
    from .utils import (
        bootstrap_ci,
        default_test_manifest,
        dump_json,
        dump_yaml,
        ensure_dir,
        list_images_from_manifest,
        load_yaml,
        monotonic_ns,
        overlay_nested,
        run_command,
        sha256_file,
        utc_now_iso,
        wallclock_ns,
    )


def post_json(url: str, payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode('utf-8'))


def get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode('utf-8'))


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        dest.write_bytes(resp.read())


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    detail = f'HTTP {exc.code} {exc.reason}'
    try:
        body = exc.read().decode('utf-8', errors='replace').strip()
    except Exception:
        body = ''
    if body:
        detail = f'{detail}: {body}'
    return detail


def _active_external_logger_run(health: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(health, dict):
        return None
    current_run = health.get('current_run')
    if not isinstance(current_run, dict):
        return None
    if not bool(current_run.get('active')):
        return None
    return current_run


def ensure_external_logger_idle(
    logger_url: str,
    expected_board_name: Optional[str],
    notes: List[str],
    *,
    initial_health: Optional[Dict[str, Any]] = None,
    idle_timeout_s: float = 30.0,
) -> Dict[str, Any]:
    logger_url = logger_url.rstrip('/')
    result: Dict[str, Any] = {}
    health = initial_health if initial_health is not None else get_json(logger_url + '/health', timeout=10)
    result['initial_health'] = health
    current_run = _active_external_logger_run(health)
    if current_run is None:
        result['recovered'] = False
        result['final_health'] = health
        return result

    active_run_id = str(current_run.get('run_id') or '').strip()
    active_notes = current_run.get('notes') if isinstance(current_run.get('notes'), dict) else {}
    active_board_name = str(active_notes.get('board_name') or '').strip()
    active_run_label = str(active_notes.get('run_label') or '').strip()
    active_elapsed_s = current_run.get('elapsed_s')

    if expected_board_name and active_board_name and active_board_name != expected_board_name:
        raise RuntimeError(
            'External logger is busy with another board and will not be auto-stopped: '
            f'run_id={active_run_id or "<unknown>"} board_name={active_board_name!r} '
            f'expected_board_name={expected_board_name!r}'
        )
    if not active_run_id:
        raise RuntimeError('External logger reported an active run but did not provide a run_id.')

    notes.append(
        'External logger already had an active run; attempting automatic recovery: '
        f'run_id={active_run_id}, run_label={active_run_label or "<unknown>"}, '
        f'board_name={active_board_name or "<unknown>"}, elapsed_s={active_elapsed_s}'
    )
    result.update(
        {
            'recovered': True,
            'recovered_run_id': active_run_id,
            'recovered_run_label': active_run_label or None,
            'recovered_board_name': active_board_name or None,
            'recovered_elapsed_s': active_elapsed_s,
        }
    )

    try:
        result['stop_info'] = post_json(logger_url + '/stop', {'run_id': active_run_id}, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise RuntimeError(f'Could not stop conflicting external logger run {active_run_id}: {_http_error_detail(exc)}') from exc
        result['stop_info'] = {'ok': False, 'warning': _http_error_detail(exc)}
        notes.append(
            f'External logger reported the conflicting run as already stopped while recovering {active_run_id}.'
        )

    deadline = time.time() + idle_timeout_s
    last_health = health
    while time.time() < deadline:
        last_health = get_json(logger_url + '/health', timeout=10)
        if _active_external_logger_run(last_health) is None:
            result['final_health'] = last_health
            notes.append(f'External logger recovery completed; {active_run_id} is no longer active.')
            return result
        time.sleep(0.5)

    active_after = _active_external_logger_run(last_health)
    raise RuntimeError(
        'External logger did not become idle after auto-recovery: '
        f'conflicting_run_id={active_run_id}, remaining_run_id={active_after.get("run_id") if active_after else None}'
    )


def start_external_logger_run(
    logger_url: str,
    start_payload: Dict[str, Any],
    expected_board_name: Optional[str],
    notes: List[str],
) -> Dict[str, Any]:
    logger_url = logger_url.rstrip('/')
    try:
        return post_json(logger_url + '/start', start_payload, timeout=20)
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
        notes.append(
            'External logger /start returned a conflict; attempting automatic recovery before retry: '
            f'run_id={start_payload.get("run_id")}, detail={_http_error_detail(exc)}'
        )
        recovery = ensure_external_logger_idle(
            logger_url,
            expected_board_name,
            notes,
            idle_timeout_s=30.0,
        )
        start_info = post_json(logger_url + '/start', start_payload, timeout=20)
        if recovery.get('recovered'):
            start_info['conflict_recovery'] = recovery
        return start_info


def stop_and_fetch_external_run(logger_url: str, run_id: str, repeat_external_dir: Path, notes: List[str]) -> Dict[str, Any]:
    logger_url = logger_url.rstrip('/')
    archive_url = logger_url + f'/download/{run_id}'
    archive_path = repeat_external_dir / f'{run_id}.tar.gz'
    stop_info: Dict[str, Any] = {}
    last_error: Optional[str] = None

    try:
        stop_info.update(post_json(logger_url + '/stop', {'run_id': run_id}, timeout=60))
    except Exception as exc:
        last_error = f'/stop failed: {exc}'
        notes.append(f'External logger stop returned non-success for {run_id}: {exc}')

    for attempt in range(1, 11):
        try:
            download_file(archive_url, archive_path, timeout=120)
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(repeat_external_dir)
            stop_info['archive_path'] = str(archive_path)
            stop_info['download_attempt'] = attempt
            return stop_info
        except Exception as exc:
            last_error = f'/download attempt {attempt} failed: {exc}'
            try:
                health = get_json(logger_url + '/health', timeout=10)
                current_run = (health or {}).get('current_run') or {}
                last_completed = current_run.get('last_completed_run') or {}
                if last_completed.get('run_id') == run_id:
                    stop_info['last_completed_run'] = last_completed
            except Exception:
                pass
            time.sleep(1.0)

    if last_error is not None:
        raise RuntimeError(last_error)
    raise RuntimeError(f'Could not download external logger archive for {run_id}')


def maybe_set_nvpmodel(mode: Optional[str], notes: List[str]) -> None:
    if mode is None:
        return
    try:
        run_command(['sudo', '/usr/sbin/nvpmodel', '-m', str(mode)], check=True)
        notes.append(f'Applied nvpmodel mode {mode}')
    except Exception as exc:
        notes.append(f'Failed to apply nvpmodel mode {mode}: {exc}')


def maybe_set_jetson_clocks(enable: bool, notes: List[str]) -> None:
    if not enable:
        return
    try:
        run_command(['sudo', '/usr/bin/jetson_clocks'], check=True)
        notes.append('Applied jetson_clocks')
    except Exception as exc:
        notes.append(f'Failed to run jetson_clocks: {exc}')


def snapshot_system(output_dir: Path) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    snap: Dict[str, Any] = {
        'timestamp_utc': utc_now_iso(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'hostname': platform.node(),
    }
    for cmd, out_name in [
        (['uname', '-a'], 'uname.txt'),
        (['bash', '-lc', 'nvpmodel -q || true'], 'nvpmodel.txt'),
        (['bash', '-lc', 'jetson_clocks --show || true'], 'jetson_clocks.txt'),
        (['bash', '-lc', 'cat /etc/nv_tegra_release || true'], 'nv_tegra_release.txt'),
        (['bash', '-lc', 'tegrastats --help 2>&1 | head -50 || true'], 'tegrastats_help.txt'),
        (['bash', '-lc', 'python3 -m pip freeze || true'], 'pip_freeze.txt'),
    ]:
        try:
            cp = subprocess.run(cmd, check=False, text=True, capture_output=True)
            (output_dir / out_name).write_text((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else ''), encoding='utf-8')
        except Exception as exc:
            (output_dir / out_name).write_text(f'ERROR: {exc}\n', encoding='utf-8')
    dump_json(snap, output_dir / 'system_snapshot.json')
    return snap


def discover_dataset_paths(dataset_root: Path, data_yaml: Optional[Path], manifest: Optional[Path]) -> Tuple[Optional[Path], Path]:
    if manifest is None:
        manifest = default_test_manifest(dataset_root)
    if data_yaml is not None and not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    return data_yaml, manifest


def resolve_accuracy_model_path(benchmark_model_path: Path, explicit_accuracy_path: Optional[Path]) -> Optional[Path]:
    if explicit_accuracy_path is not None:
        return explicit_accuracy_path
    return benchmark_model_path


COMPLEXITY_KEYS = (
    'params',
    'macs',
    'gmacs',
    'gflops_if_1mac_eq_2flops',
)


def _canonical_model_stem(path: Path) -> str:
    return path.stem.replace('_fp16', '').replace('_fp32', '').replace('_int8', '')


def _normalize_complexity_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, Any] = {}
    for key in COMPLEXITY_KEYS:
        value = payload.get(key)
        if value is None:
            return None
        try:
            if key in {'params', 'macs'}:
                normalized[key] = int(round(float(value)))
            else:
                normalized[key] = float(value)
        except (TypeError, ValueError):
            return None
    return normalized


def _load_complexity_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return _normalize_complexity_payload(raw)


def _resolve_complexity_onnx_path(
    model_path: Path,
    accuracy_model_path: Optional[Path],
    bundle_root: Optional[Path],
) -> Optional[Path]:
    candidates: List[Path] = []
    if model_path.suffix.lower() == '.onnx':
        candidates.append(model_path)
    if accuracy_model_path is not None and accuracy_model_path.suffix.lower() == '.onnx':
        candidates.append(accuracy_model_path)

    canonical_stem = _canonical_model_stem(model_path)
    candidates.extend(
        [
            model_path.with_name(canonical_stem + '.onnx'),
            model_path.with_suffix('.onnx'),
        ]
    )
    if accuracy_model_path is not None:
        accuracy_canonical_stem = _canonical_model_stem(accuracy_model_path)
        candidates.extend(
            [
                accuracy_model_path.with_name(accuracy_canonical_stem + '.onnx'),
                accuracy_model_path.with_suffix('.onnx'),
            ]
        )
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / 'model' / 'best.onnx',
                bundle_root / 'model' / 'best_trt_deploy.onnx',
            ]
        )

    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def _profile_or_load_model_complexity(
    *,
    model_path: Path,
    accuracy_model_path: Optional[Path],
    bundle_root: Optional[Path],
    imgsz: int,
    batch: int,
    notes: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    onnx_path = _resolve_complexity_onnx_path(model_path, accuracy_model_path, bundle_root)
    if onnx_path is None:
        notes.append('Model complexity skipped: could not locate an ONNX graph next to the measured model.')
        return None, None, None

    complexity_json = onnx_path.parent / f'complexity_{int(imgsz)}.json'
    cached = _load_complexity_json(complexity_json)
    if cached is not None:
        return cached, complexity_json, onnx_path

    profile_script = Path(__file__).resolve().parents[1] / 'scripts' / 'profile_model_complexity.py'
    if not profile_script.exists():
        notes.append(f'Model complexity skipped: profiling script not found at {profile_script}.')
        return None, None, onnx_path

    cmd = [
        sys.executable,
        str(profile_script),
        '--pkg-root',
        str(Path(__file__).resolve().parents[1]),
        '--onnx',
        str(onnx_path),
        '--imgsz',
        str(int(imgsz)),
        '--batch',
        str(int(batch)),
        '--out-json',
        str(complexity_json),
    ]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or '').strip()
        stdout = (exc.stdout or '').strip()
        detail = stderr or stdout or repr(exc)
        notes.append(f'Model complexity profiling failed for {onnx_path.name}: {detail}')
        return None, None, onnx_path

    profiled = _load_complexity_json(complexity_json)
    if profiled is None:
        notes.append(f'Model complexity profiling produced an unreadable payload at {complexity_json}.')
        return None, None, onnx_path
    return profiled, complexity_json, onnx_path


def _fallback_accuracy_model_if_needed(model_path: Path, notes: Optional[List[str]] = None) -> Path:
    if model_path.suffix.lower() != '.onnx':
        return model_path
    if importlib.util.find_spec('onnxruntime') is not None:
        return model_path

    sibling_pth = model_path.with_suffix('.pth')
    alt_pth = model_path.with_name(model_path.stem.replace('_fp16', '').replace('_fp32', '').replace('_int8', '') + '.pth')
    sibling_pt = model_path.with_suffix('.pt')
    alt_pt = model_path.with_name(model_path.stem.replace('_fp16', '').replace('_fp32', '').replace('_int8', '') + '.pt')
    for candidate in (sibling_pth, alt_pth, sibling_pt, alt_pt):
        if candidate.exists():
            if notes is not None:
                notes.append(f'Accuracy pass fallback: onnxruntime not available, using {candidate.name} instead of {model_path.name}')
            return candidate
    return model_path


def _prefer_system_tensorrt_bindings(notes: Optional[List[str]] = None):
    system_paths = [
        Path(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages'),
        Path('/usr/lib/python3/dist-packages'),
    ]
    system_paths = [path for path in system_paths if (path / 'tensorrt' / '__init__.py').exists()]
    if not system_paths:
        return None

    preferred = str(system_paths[0])
    if preferred not in sys.path:
        sys.path.insert(0, preferred)

    # If a pip wheel already loaded a different TensorRT build, evict it before importing the JetPack binding.
    for name, module in list(sys.modules.items()):
        if name != 'tensorrt' and not name.startswith('tensorrt.'):
            continue
        module_file = str(getattr(module, '__file__', '') or '')
        if module_file and not module_file.startswith(preferred):
            sys.modules.pop(name, None)

    try:
        trt = importlib.import_module('tensorrt')
    except Exception:
        return None

    trt_file = str(Path(getattr(trt, '__file__', '')).resolve()) if getattr(trt, '__file__', None) else ''
    if preferred and trt_file.startswith(preferred):
        if notes is not None:
            notes.append(f'Using JetPack TensorRT Python bindings from {preferred} ({trt.__version__}).')
        return trt
    return None


def _prime_tensorrt_backend(model_path: Path, notes: Optional[List[str]] = None) -> None:
    if model_path.suffix.lower() != '.engine':
        return

    os.environ.setdefault('ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS', '1')
    try:
        trt = _prefer_system_tensorrt_bindings(notes)
        if trt is None:
            import tensorrt as trt  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            'TensorRT Python bindings are not importable in the active benchmark process. '
            'Run `python -c "import tensorrt; print(tensorrt.__version__)"` inside the same env, '
            'then rerun `bash scripts/setup_jetson_env.sh` if it fails.'
        ) from exc

    if notes is not None:
        notes.append(
            f'TensorRT backend primed in-process via Python bindings {trt.__version__}; '
            'Ultralytics requirements auto-check disabled for engine loading.'
        )


def ultralytics_accuracy_pass(model_path: Path, data_yaml: Path, split_name: str, imgsz: int, batch: int, device: str, project_dir: Path) -> Dict[str, Any]:
    acc_dir = ensure_dir(project_dir / 'accuracy')
    model = _load_ultralytics_yolo()(str(model_path), task='detect')
    results = model.val(
        data=str(data_yaml),
        split=split_name,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(acc_dir),
        name='val_' + split_name,
        exist_ok=True,
        save_json=False,
        plots=True,
        verbose=True,
    )
    results_dict = getattr(results, 'results_dict', None)
    if results_dict is None:
        try:
            results_dict = dict(results)
        except Exception:
            results_dict = {'repr': repr(results)}
    dump_json(results_dict, acc_dir / 'val_metrics.json')
    return results_dict


def _extract_primary_accuracy_metric(metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    candidates = [
        ('mAP50_95', metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95'))),
        ('mIoU', metrics.get('metrics/mIoU', metrics.get('metrics/miou', metrics.get('miou')))),
        ('mDice', metrics.get('metrics/mDice', metrics.get('metrics/mdice', metrics.get('mdice')))),
        ('pixel_accuracy', metrics.get('metrics/pixel_accuracy', metrics.get('pixel_accuracy'))),
    ]
    for name, value in candidates:
        try:
            if value is not None:
                return name, float(value)
        except (TypeError, ValueError):
            continue
    return None, None


def _infer_precision_label_from_model_path(model_path: Optional[Path]) -> Optional[str]:
    if model_path is None:
        return None
    name = model_path.name.lower()
    for label in ('int8', 'fp16', 'fp32'):
        if label in name:
            return label
    return None


def _infer_accuracy_backend_from_model_path(model_path: Optional[Path]) -> Optional[str]:
    if model_path is None:
        return None
    suffix = model_path.suffix.lower()
    if suffix == '.engine':
        return 'tensorrt_engine'
    if suffix in {'.pt', '.pth'}:
        return 'pytorch_checkpoint'
    if suffix == '.onnx':
        return 'onnx'
    return None


def _normalize_path_str(value: Optional[str | Path]) -> Optional[str]:
    if value in {None, ''}:
        return None
    return str(Path(value).expanduser().resolve())


def _resolve_accuracy_manifest_path(accuracy_metrics: Dict[str, Any]) -> Optional[Path]:
    manifest_value = accuracy_metrics.get('manifest_path') if isinstance(accuracy_metrics, dict) else None
    if not manifest_value:
        return None
    return Path(str(manifest_value)).expanduser().resolve()


def _rewrite_accuracy_config_snapshot(
    cfg: Dict[str, Any],
    run_root: Path,
    *,
    benchmark_manifest_path: Optional[Path],
    accuracy_manifest_path: Optional[Path],
    accuracy_model_path: Optional[Path],
    accuracy_model_sha256: Optional[str],
    accuracy_backend: Optional[str],
    accuracy_backend_detail: Optional[str],
    accuracy_precision_label: Optional[str],
    accuracy_precision_specific: bool,
) -> None:
    cfg.setdefault('model', {})
    cfg.setdefault('dataset', {})
    cfg['model']['accuracy_model_path'] = str(accuracy_model_path) if accuracy_model_path is not None else None
    cfg['model']['accuracy_model_sha256'] = accuracy_model_sha256
    cfg['model']['accuracy_backend'] = accuracy_backend
    cfg['model']['accuracy_backend_detail'] = accuracy_backend_detail
    cfg['model']['accuracy_precision_label'] = accuracy_precision_label
    cfg['model']['accuracy_precision_specific'] = bool(accuracy_precision_specific)
    if benchmark_manifest_path is not None:
        cfg['dataset']['ordered_manifest'] = str(benchmark_manifest_path)
        cfg['dataset']['benchmark_manifest'] = str(benchmark_manifest_path)
    if accuracy_manifest_path is not None:
        cfg['dataset']['accuracy_manifest'] = str(accuracy_manifest_path)
        cfg['dataset']['manifest'] = str(accuracy_manifest_path)
    elif benchmark_manifest_path is not None:
        cfg['dataset'].pop('accuracy_manifest', None)
        cfg['dataset']['manifest'] = str(benchmark_manifest_path)
    dump_yaml(cfg, run_root / 'config_snapshot.yaml')


def _validate_accuracy_provenance(
    *,
    run_root: Path,
    measured_model_path: Path,
    measured_model_sha256: str,
    benchmark_manifest_path: Path,
    resolved_accuracy_model_path: Optional[Path],
    accuracy_model_sha256: Optional[str],
    accuracy_backend: Optional[str],
    accuracy_precision_label: Optional[str],
    accuracy_metrics: Dict[str, Any],
    final_summary: Dict[str, Any],
    split_name: str,
) -> None:
    if not accuracy_metrics:
        return

    expected_model_path = _normalize_path_str(resolved_accuracy_model_path)
    expected_backend = accuracy_backend
    expected_precision_label = accuracy_precision_label
    expected_accuracy_manifest = _normalize_path_str(
        _resolve_accuracy_manifest_path(metrics_payload := accuracy_metrics) or accuracy_metrics.get('manifest_path')
    )
    expected_benchmark_manifest = _normalize_path_str(benchmark_manifest_path)
    issues: List[str] = []

    metrics_json_path = run_root / 'accuracy' / 'val_metrics.json'
    metrics_payload = accuracy_metrics
    if metrics_json_path.exists():
        metrics_payload = json.loads(metrics_json_path.read_text(encoding='utf-8'))
        expected_accuracy_manifest = _normalize_path_str(metrics_payload.get('manifest_path')) or expected_accuracy_manifest

    def _expect_equal(label: str, actual: Optional[str], expected: Optional[str]) -> None:
        if expected is None:
            return
        if actual != expected:
            issues.append(f'{label}: expected {expected!r}, got {actual!r}')

    actual_metric_model_path = _normalize_path_str(
        metrics_payload.get('model_path') or metrics_payload.get('engine_path') or metrics_payload.get('checkpoint_path')
    )
    actual_metric_model_sha = metrics_payload.get('model_sha256')
    actual_metric_backend = metrics_payload.get('backend')
    actual_metric_precision = metrics_payload.get('precision_label')
    actual_metric_manifest = _normalize_path_str(metrics_payload.get('manifest_path'))

    _expect_equal('accuracy/val_metrics.json model_path', actual_metric_model_path, expected_model_path)
    _expect_equal('accuracy/val_metrics.json model_sha256', actual_metric_model_sha, accuracy_model_sha256)
    _expect_equal('accuracy/val_metrics.json backend', actual_metric_backend, expected_backend)
    _expect_equal('accuracy/val_metrics.json precision_label', actual_metric_precision, expected_precision_label)
    _expect_equal('accuracy/val_metrics.json manifest_path', actual_metric_manifest, expected_accuracy_manifest)

    config_snapshot_path = run_root / 'config_snapshot.yaml'
    if not config_snapshot_path.exists():
        issues.append('config_snapshot.yaml is missing')
    else:
        cfg_snapshot = load_yaml(config_snapshot_path)
        cfg_model = cfg_snapshot.get('model', {}) if isinstance(cfg_snapshot.get('model', {}), dict) else {}
        cfg_dataset = cfg_snapshot.get('dataset', {}) if isinstance(cfg_snapshot.get('dataset', {}), dict) else {}
        _expect_equal('config_snapshot.yaml model.accuracy_model_path', _normalize_path_str(cfg_model.get('accuracy_model_path')), expected_model_path)
        _expect_equal('config_snapshot.yaml model.accuracy_model_sha256', cfg_model.get('accuracy_model_sha256'), accuracy_model_sha256)
        _expect_equal('config_snapshot.yaml model.accuracy_backend', cfg_model.get('accuracy_backend'), expected_backend)
        _expect_equal('config_snapshot.yaml model.accuracy_precision_label', cfg_model.get('accuracy_precision_label'), expected_precision_label)
        _expect_equal('config_snapshot.yaml dataset.manifest', _normalize_path_str(cfg_dataset.get('manifest')), expected_accuracy_manifest or expected_benchmark_manifest)
        _expect_equal('config_snapshot.yaml dataset.benchmark_manifest', _normalize_path_str(cfg_dataset.get('benchmark_manifest')), expected_benchmark_manifest)
        _expect_equal('config_snapshot.yaml dataset.ordered_manifest', _normalize_path_str(cfg_dataset.get('ordered_manifest')), expected_benchmark_manifest)
        if expected_accuracy_manifest is not None:
            _expect_equal('config_snapshot.yaml dataset.accuracy_manifest', _normalize_path_str(cfg_dataset.get('accuracy_manifest')), expected_accuracy_manifest)

    final_model = final_summary.get('model', {}) if isinstance(final_summary.get('model', {}), dict) else {}
    final_dataset = final_summary.get('dataset', {}) if isinstance(final_summary.get('dataset', {}), dict) else {}
    _expect_equal('FINAL_SUMMARY.json model.accuracy_model_path', _normalize_path_str(final_model.get('accuracy_model_path')), expected_model_path)
    _expect_equal('FINAL_SUMMARY.json model.accuracy_model_sha256', final_model.get('accuracy_model_sha256'), accuracy_model_sha256)
    _expect_equal('FINAL_SUMMARY.json model.accuracy_backend', final_model.get('accuracy_backend'), expected_backend)
    _expect_equal('FINAL_SUMMARY.json model.accuracy_precision_label', final_model.get('accuracy_precision_label'), expected_precision_label)
    _expect_equal('FINAL_SUMMARY.json dataset.manifest', _normalize_path_str(final_dataset.get('manifest')), expected_accuracy_manifest or expected_benchmark_manifest)
    _expect_equal('FINAL_SUMMARY.json dataset.benchmark_manifest', _normalize_path_str(final_dataset.get('benchmark_manifest')), expected_benchmark_manifest)
    _expect_equal('FINAL_SUMMARY.json dataset.ordered_manifest', _normalize_path_str(final_dataset.get('ordered_manifest')), expected_benchmark_manifest)
    if expected_accuracy_manifest is not None:
        _expect_equal('FINAL_SUMMARY.json dataset.accuracy_manifest', _normalize_path_str(final_dataset.get('accuracy_manifest')), expected_accuracy_manifest)

    validation_log_path = run_root / 'accuracy' / f'validation_{split_name}.log'
    if not validation_log_path.exists():
        issues.append(f'{validation_log_path.name} is missing')
    else:
        validation_text = validation_log_path.read_text(encoding='utf-8', errors='replace')
        for label, expected in (
            ('validation log model_path', expected_model_path),
            ('validation log backend', expected_backend),
            ('validation log precision_label', expected_precision_label),
            ('validation log manifest_path', expected_accuracy_manifest),
        ):
            if expected and expected not in validation_text:
                issues.append(f'{label}: {expected!r} not found in {validation_log_path.name}')

    if expected_backend == 'tensorrt_engine':
        measured_model_path_str = _normalize_path_str(measured_model_path)
        if expected_model_path != measured_model_path_str:
            issues.append(
                'engine accuracy provenance mismatch: accuracy_model_path must equal measured_model_path '
                f'({expected_model_path!r} != {measured_model_path_str!r})'
            )
        if accuracy_model_sha256 != measured_model_sha256:
            issues.append(
                'engine accuracy provenance mismatch: accuracy_model_sha256 must equal measured_model_sha256 '
                f'({accuracy_model_sha256!r} != {measured_model_sha256!r})'
            )

    if issues:
        raise RuntimeError('Accuracy provenance consistency check failed:\n- ' + '\n- '.join(issues))


def _resolve_bisenetv2_source_package_root() -> Optional[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get('BISENETV2_SOURCE_PACKAGE_ROOT')
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    bundle_root = resolve_inference_bundle_root()
    if bundle_root is not None:
        parent = bundle_root.parent
        candidates.extend(
            [
                (parent / 'stage2_bisenetv2_training_cluster_2GPU').resolve(),
            ]
        )

    for candidate in candidates:
        if (candidate / 'scripts' / 'bisenetv2_lib.py').exists() and (candidate / 'config' / 'segmentation_bisenetv2_train_lock.yaml').exists():
            return candidate
    return None


def _resolve_bisenetv2_source_config_path(source_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    env_cfg = os.environ.get('BISENETV2_SOURCE_CONFIG_PATH')
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser().resolve())
    candidates.append((source_root / 'config' / 'segmentation_bisenetv2_train_lock.yaml').resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def bisenetv2_repo_accuracy_pass(
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    project_dir: Path,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    source_root = _resolve_bisenetv2_source_package_root()
    if source_root is None:
        raise RuntimeError('BiSeNetV2 source package root could not be resolved for accuracy validation.')

    config_path = _resolve_bisenetv2_source_config_path(source_root)
    if config_path is None:
        raise RuntimeError('BiSeNetV2 source config could not be resolved for accuracy validation.')

    helper = Path(__file__).resolve().parents[1] / 'scripts' / 'run_bisenetv2_accuracy.py'
    if not helper.exists():
        raise FileNotFoundError(helper)

    acc_dir = ensure_dir(project_dir / 'accuracy')
    output_dir = ensure_dir(acc_dir / f'val_{split_name}')
    log_path = acc_dir / f'validation_{split_name}.log'
    out_json = acc_dir / 'val_metrics.json'

    source_cfg = load_yaml(config_path)
    training_cfg = source_cfg.get('training', {}) if isinstance(source_cfg.get('training', {}), dict) else {}
    default_accuracy_batch = max(1, min(int(training_cfg.get('eval_batch', max(1, batch))), 4))
    default_accuracy_workers = max(0, min(int(training_cfg.get('workers', 4)), 4))
    accuracy_batch = int(os.environ.get('BISENETV2_ACCURACY_BATCH', default_accuracy_batch))
    accuracy_workers = int(os.environ.get('BISENETV2_ACCURACY_WORKERS', default_accuracy_workers))

    cmd = [
        str(Path(sys.executable).resolve()),
        str(helper),
        '--source-root',
        str(source_root),
        '--config',
        str(config_path),
        '--model',
        str(model_path),
        '--dataset-root',
        str(dataset_root),
        '--split-name',
        str(split_name),
        '--imgsz',
        str(int(imgsz)),
        '--batch',
        str(max(1, accuracy_batch)),
        '--workers',
        str(max(0, accuracy_workers)),
        '--output-dir',
        str(output_dir),
        '--out-json',
        str(out_json),
    ]
    cp = run_command(cmd, check=False, capture_output=True, text=True)
    combined_output = ((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else '')).strip()
    log_path.write_text(combined_output + '\n', encoding='utf-8')
    if cp.returncode != 0:
        raise RuntimeError(
            'BiSeNetV2 accuracy validation failed.\n'
            f'Command: {" ".join(cmd)}\n'
            f'Last output:\n{combined_output[-2000:]}'
        )
    if not out_json.exists():
        raise FileNotFoundError(out_json)

    results = json.loads(out_json.read_text(encoding='utf-8'))
    metric_name, metric_value = _extract_primary_accuracy_metric(results)
    if notes is not None and metric_name is not None and metric_value is not None:
        notes.append(
            f'Accuracy backend: BiSeNetV2 source validation via {helper.name}; '
            f'{metric_name}={metric_value:.3f}.'
        )
    return results


def bisenetv2_engine_accuracy_pass(
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    project_dir: Path,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    helper = Path(__file__).resolve().parents[1] / 'scripts' / 'run_bisenetv2_engine_accuracy.py'
    if not helper.exists():
        raise FileNotFoundError(helper)

    acc_dir = ensure_dir(project_dir / 'accuracy')
    output_dir = ensure_dir(acc_dir / f'val_{split_name}')
    log_path = acc_dir / f'validation_{split_name}.log'
    out_json = acc_dir / 'val_metrics.json'

    cmd = [
        str(Path(sys.executable).resolve()),
        str(helper),
        '--engine',
        str(model_path),
        '--dataset-root',
        str(dataset_root),
        '--split-name',
        str(split_name),
        '--imgsz',
        str(int(imgsz)),
        '--output-dir',
        str(output_dir),
        '--out-json',
        str(out_json),
    ]
    cp = run_command(cmd, check=False, capture_output=True, text=True)
    combined_output = ((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else '')).strip()
    log_path.write_text(combined_output + '\n', encoding='utf-8')
    if cp.returncode != 0:
        raise RuntimeError(
            'BiSeNetV2 TensorRT engine accuracy validation failed.\n'
            f'Command: {" ".join(cmd)}\n'
            f'Last output:\n{combined_output[-2000:]}'
        )
    if not out_json.exists():
        raise FileNotFoundError(out_json)

    results = json.loads(out_json.read_text(encoding='utf-8'))
    metric_name, metric_value = _extract_primary_accuracy_metric(results)
    if notes is not None and metric_name is not None and metric_value is not None:
        notes.append(
            f'Accuracy backend: BiSeNetV2 TensorRT engine validation via {helper.name}; '
            f'{metric_name}={metric_value:.3f}.'
        )
    return results


def yolo11s_engine_accuracy_pass(
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    device: str,
    project_dir: Path,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    helper = Path(__file__).resolve().parents[1] / 'scripts' / 'run_yolo11s_engine_accuracy.py'
    if not helper.exists():
        raise FileNotFoundError(helper)

    log_path = ensure_dir(project_dir / 'accuracy') / f'validation_{split_name}.log'
    out_json = project_dir / 'accuracy' / 'val_metrics.json'

    cmd = [
        str(Path(sys.executable).resolve()),
        str(helper),
        '--model',
        str(model_path),
        '--dataset-root',
        str(dataset_root),
        '--split-name',
        str(split_name),
        '--imgsz',
        str(int(imgsz)),
        '--batch',
        str(max(1, batch)),
        '--device',
        str(device),
        '--output-root',
        str(project_dir),
    ]
    cp = run_command(cmd, check=False, capture_output=True, text=True)
    combined_output = ((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else '')).strip()
    log_path.write_text(combined_output + '\n', encoding='utf-8')
    if cp.returncode != 0:
        raise RuntimeError(
            'YOLO11s TensorRT engine accuracy validation failed.\n'
            f'Command: {" ".join(cmd)}\n'
            f'Last output:\n{combined_output[-2000:]}'
        )
    if not out_json.exists():
        raise FileNotFoundError(out_json)

    results = json.loads(out_json.read_text(encoding='utf-8'))
    metric_name, metric_value = _extract_primary_accuracy_metric(results)
    if notes is not None and metric_name is not None and metric_value is not None:
        notes.append(
            f'Accuracy backend: YOLO11s TensorRT engine validation via {helper.name}; '
            f'{metric_name}={metric_value:.3f}.'
        )
    return results


def _parse_coco_metrics_from_text(text: str) -> Dict[str, float]:
    patterns = {
        'map50_95': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)',
        'map50': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)',
        'map75': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)',
        'ar100': r'Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*([0-9.]+)',
    }
    out: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def _resolve_rtdetr_repo_root() -> Optional[Path]:
    candidates: List[Path] = []
    env_repo = os.environ.get('RTDETR_SOURCE_REPO_ROOT')
    if env_repo:
        candidates.append(Path(env_repo).expanduser().resolve())

    bundle_root = resolve_inference_bundle_root()
    if bundle_root is not None:
        candidates.append((bundle_root.parent / 'stage2_rtdetr_r18_training_cluster' / 'vendor' / 'RT-DETR' / 'rtdetr_pytorch').resolve())

    for candidate in candidates:
        if (candidate / 'tools' / 'train.py').exists():
            return candidate
    return None


def _resolve_rtdetr_accuracy_template(bundle_root: Optional[Path], split_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    if bundle_root is not None:
        candidates.extend(
            [
                bundle_root / 'metadata' / f'rtdetr_eval_{split_name}.yml',
                bundle_root / 'metadata' / f'rtdetr_eval_{split_name}.yaml',
                bundle_root / 'metadata' / 'rtdetr_eval_test.yml',
                bundle_root / 'metadata' / 'rtdetr_full.yml',
            ]
        )

    env_cfg = os.environ.get('RTDETR_SOURCE_CONFIG_PATH')
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser().resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_rtdetr_annotation_file(dataset_root: Path, split_name: str) -> Path:
    split = str(split_name or 'test')
    candidates = [
        dataset_root / 'annotations' / f'instances_{split}.json',
        dataset_root / 'annotations' / f'{split}_coco_ground_truth.json',
        dataset_root / 'annotations' / 'test_coco_ground_truth.json',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Could not find a COCO annotation file for split "{split}" under {dataset_root / "annotations"}')


def _prepare_rtdetr_runtime_annotation(source_ann: Path, dest_ann: Path) -> Path:
    obj = json.loads(source_ann.read_text(encoding='utf-8'))
    categories = obj.get('categories', [])
    annotations = obj.get('annotations', [])
    category_ids = sorted(int(cat['id']) for cat in categories)

    if not categories:
        shutil.copy2(source_ann, dest_ann)
        return dest_ann

    zero_based_ids = list(range(len(category_ids)))
    if category_ids == zero_based_ids:
        if source_ann.resolve() != dest_ann.resolve():
            shutil.copy2(source_ann, dest_ann)
        return dest_ann

    if category_ids == list(range(1, len(category_ids) + 1)):
        id_map = {old_id: old_id - 1 for old_id in category_ids}
    else:
        id_map = {old_id: new_id for new_id, old_id in enumerate(category_ids)}

    remapped = dict(obj)
    remapped['categories'] = [
        {**cat, 'id': int(id_map[int(cat['id'])])}
        for cat in categories
    ]
    remapped['annotations'] = [
        {**ann, 'category_id': int(id_map[int(ann['category_id'])])}
        for ann in annotations
    ]
    dump_json(remapped, dest_ann)
    return dest_ann


def _patch_rtdetr_loader_cfg(loader_cfg: Dict[str, Any], dataset_root: Path, ann_file: Path, imgsz: int, batch: int, num_workers: int) -> None:
    dataset_cfg = loader_cfg.setdefault('dataset', {})
    dataset_cfg['img_folder'] = str(dataset_root)
    dataset_cfg['ann_file'] = str(ann_file)

    transforms = dataset_cfg.get('transforms')
    if isinstance(transforms, dict):
        ops = transforms.get('ops')
        if isinstance(ops, list):
            for op in ops:
                if isinstance(op, dict) and op.get('type') == 'Resize':
                    op['size'] = [imgsz, imgsz]

    loader_cfg['batch_size'] = max(1, int(batch))
    loader_cfg['num_workers'] = max(0, int(num_workers))
    loader_cfg['shuffle'] = False
    loader_cfg['drop_last'] = False


def _resolve_accuracy_python_candidates() -> List[Path]:
    candidates: List[Path] = [Path(sys.executable).resolve()]
    conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')
    conda_env_name = os.environ.get('CONDA_ENV_NAME')
    build_env_name = os.environ.get('RTDETR_BUILD_CONDA_ENV_NAME')
    miniconda_root = Path(os.environ.get('MINICONDA_ROOT', str(Path.home() / 'miniconda3'))).expanduser()

    for env_name in (conda_default_env, conda_env_name, build_env_name):
        if not env_name:
            continue
        candidates.append((miniconda_root / 'envs' / env_name / 'bin' / 'python').resolve())

    unique: List[Path] = []
    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def rtdetr_repo_accuracy_pass(
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    imgsz: int,
    batch: int,
    project_dir: Path,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bundle_root = resolve_inference_bundle_root(model_path.parent.parent if model_path.parent.name == 'model' else None)
    repo_root = _resolve_rtdetr_repo_root()
    if repo_root is None:
        raise RuntimeError('RT-DETR source repo root could not be resolved for accuracy validation.')

    template_path = _resolve_rtdetr_accuracy_template(bundle_root, split_name)
    if template_path is None:
        raise RuntimeError('RT-DETR eval config template could not be resolved from inference_bundle metadata.')

    source_ann_file = _resolve_rtdetr_annotation_file(dataset_root, split_name)
    acc_dir = ensure_dir(project_dir / 'accuracy')
    output_dir = ensure_dir(acc_dir / f'val_{split_name}')
    runtime_cfg_path = acc_dir / f'runtime_rtdetr_eval_{split_name}.yml'
    log_path = acc_dir / f'validation_{split_name}.log'
    runtime_ann_file = _prepare_rtdetr_runtime_annotation(
        source_ann_file,
        acc_dir / f'instances_{split_name}_rtdetr_runtime.json',
    )

    runtime_cfg = load_yaml(template_path)
    if not runtime_cfg:
        raise RuntimeError(f'RT-DETR eval config template is empty: {template_path}')

    runtime_cfg['output_dir'] = str(output_dir)
    runtime_cfg['sync_bn'] = False
    runtime_cfg['find_unused_parameters'] = False

    presnet_cfg = runtime_cfg.get('PResNet')
    if isinstance(presnet_cfg, dict):
        presnet_cfg['pretrained'] = False

    for section_name in ('HybridEncoder', 'RTDETRTransformer'):
        section = runtime_cfg.get(section_name)
        if isinstance(section, dict) and 'eval_spatial_size' in section:
            section['eval_spatial_size'] = [int(imgsz), int(imgsz)]

    accuracy_batch = int(os.environ.get('RTDETR_ACCURACY_BATCH', runtime_cfg.get('val_dataloader', {}).get('batch_size', batch)))
    accuracy_workers = int(os.environ.get('RTDETR_ACCURACY_WORKERS', runtime_cfg.get('val_dataloader', {}).get('num_workers', 2)))

    _patch_rtdetr_loader_cfg(
        runtime_cfg.setdefault('train_dataloader', {}),
        dataset_root=dataset_root,
        ann_file=runtime_ann_file,
        imgsz=imgsz,
        batch=accuracy_batch,
        num_workers=accuracy_workers,
    )
    _patch_rtdetr_loader_cfg(
        runtime_cfg.setdefault('val_dataloader', {}),
        dataset_root=dataset_root,
        ann_file=runtime_ann_file,
        imgsz=imgsz,
        batch=accuracy_batch,
        num_workers=accuracy_workers,
    )

    dump_yaml(runtime_cfg, runtime_cfg_path)

    train_script = repo_root / 'tools' / 'train.py'
    env = dict(os.environ)
    pythonpath_entries = [str(repo_root)]
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env['PYTHONPATH'] = ':'.join(pythonpath_entries)

    attempt_errors: List[str] = []
    last_output = ''
    used_python: Optional[Path] = None

    for python_exe in _resolve_accuracy_python_candidates():
        cmd = [
            str(python_exe),
            str(train_script),
            '-c',
            str(runtime_cfg_path),
            '-r',
            str(model_path),
            '--test-only',
            '--seed',
            '123',
        ]
        cp = run_command(cmd, check=False, capture_output=True, text=True, env=env, cwd=repo_root)
        combined_output = ((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else '')).strip()
        last_output = combined_output
        if cp.returncode == 0:
            used_python = python_exe
            break
        attempt_errors.append(
            f'python={python_exe} rc={cp.returncode}\n{combined_output[-4000:]}'
        )
        if 'ModuleNotFoundError' not in combined_output and 'No module named' not in combined_output:
            log_path.write_text(combined_output + '\n', encoding='utf-8')
            raise RuntimeError(
                'RT-DETR accuracy validation failed. '
                f'Command: {" ".join(cmd)}\n'
                f'Last output:\n{combined_output[-2000:]}'
            )

    if used_python is None:
        log_path.write_text('\n\n'.join(attempt_errors) + '\n', encoding='utf-8')
        raise RuntimeError(
            'RT-DETR accuracy validation could not start because no candidate Python environment had the required repo dependencies.\n'
            + '\n\n'.join(attempt_errors)
        )

    log_path.write_text(last_output + '\n', encoding='utf-8')
    metrics = _parse_coco_metrics_from_text(last_output)
    if 'map50_95' not in metrics:
        raise RuntimeError(
            'RT-DETR accuracy validation completed but COCO metrics could not be parsed from the validation log. '
            f'Inspect {log_path}.'
        )

    results_dict: Dict[str, Any] = {
        'backend': 'rtdetr_repo',
        'metrics/mAP50-95(B)': metrics.get('map50_95'),
        'metrics/mAP50(B)': metrics.get('map50'),
        'metrics/mAP75(B)': metrics.get('map75'),
        'metrics/AR100(B)': metrics.get('ar100'),
        'metrics/mAP50-95': metrics.get('map50_95'),
        'metrics/mAP50': metrics.get('map50'),
        'metrics/mAP75': metrics.get('map75'),
        'metrics/AR100': metrics.get('ar100'),
        'config_path': str(runtime_cfg_path),
        'log_path': str(log_path),
        'output_dir': str(output_dir),
        'annotation_file': str(runtime_ann_file),
        'annotation_file_source': str(source_ann_file),
        'checkpoint_path': str(model_path),
        'python_executable': str(used_python),
    }

    eval_artifact = output_dir / 'eval.pth'
    if eval_artifact.exists():
        results_dict['eval_artifact'] = str(eval_artifact)

    dump_json(results_dict, acc_dir / 'val_metrics.json')
    if notes is not None:
        notes.append(
            f'Accuracy backend: RT-DETR repo validation via {train_script} '
            f'using {used_python.name}; mAP50-95={metrics.get("map50_95"):.3f}.'
        )
    return results_dict


def _prepare_accuracy_data_yaml(data_yaml: Path, dataset_root: Path, split_name: str, project_dir: Path, notes: Optional[List[str]] = None) -> Path:
    cfg = load_yaml(data_yaml)
    if cfg.get('train') and cfg.get('val'):
        return data_yaml

    desired_split = str(split_name or 'test')
    split_source = None
    split_source_key = None
    for key in (desired_split, 'val', 'test', 'train'):
        value = cfg.get(key)
        if value:
            split_source = value
            split_source_key = key
            break

    if split_source is None:
        split_source = 'images/test'
        split_source_key = 'implicit_test'

    runtime_cfg = dict(cfg)
    runtime_cfg['path'] = str(dataset_root)
    runtime_cfg.setdefault('train', split_source)
    runtime_cfg.setdefault('val', split_source)
    runtime_cfg.setdefault('test', split_source)

    runtime_yaml = ensure_dir(project_dir / 'accuracy') / 'runtime_accuracy_dataset.yaml'
    dump_yaml(runtime_cfg, runtime_yaml)
    if notes is not None:
        notes.append(
            'Accuracy data YAML fallback: generated runtime_accuracy_dataset.yaml '
            f'with train/val/test mapped from {split_source_key}.'
        )
    return runtime_yaml


def write_yolo_txt(pred_path: Path, boxes) -> None:
    lines: List[str] = []
    if boxes is not None and hasattr(boxes, 'xywhn'):
        xywhn = boxes.xywhn.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for i in range(len(cls)):
            x, y, w, h = xywhn[i].tolist()
            lines.append(f'{cls[i]} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf[i]:.6f}')
    pred_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _per_image_latency_fieldnames(task: str) -> List[str]:
    base = [
        'repeat', 'image_global_index', 'cycle_index', 'manifest_index', 'image_path', 'image_name', 'timestamp_utc',
        'wallclock_ns_start', 'wallclock_ns_end', 'monotonic_ns_start', 'monotonic_ns_end', 'latency_ms'
    ]
    if task == 'semantic_segmentation':
        return base
    return [*base, 'num_detections']


def _time_col_for_df(df: pd.DataFrame) -> pd.Series:
    if 'mid_wallclock_ns' in df.columns:
        return pd.to_numeric(df['mid_wallclock_ns'], errors='coerce')
    if 'wallclock_ns' in df.columns:
        return pd.to_numeric(df['wallclock_ns'], errors='coerce')
    if 'wallclock_ns_end' in df.columns and 'wallclock_ns_start' in df.columns:
        return (pd.to_numeric(df['wallclock_ns_start'], errors='coerce') + pd.to_numeric(df['wallclock_ns_end'], errors='coerce')) / 2.0
    if 'wallclock_ns_end' in df.columns:
        return pd.to_numeric(df['wallclock_ns_end'], errors='coerce')
    if 'sched_wallclock_ns' in df.columns:
        return pd.to_numeric(df['sched_wallclock_ns'], errors='coerce')
    raise RuntimeError('Could not infer a time column from dataframe')


def _crop_df_by_window(df: pd.DataFrame, start_ns: Optional[int], end_ns: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    times = _time_col_for_df(df)
    mask = pd.Series(True, index=df.index)
    if start_ns is not None:
        mask &= times >= int(start_ns)
    if end_ns is not None:
        mask &= times <= int(end_ns)
    return df.loc[mask].copy()


def _parse_external_csv(external_csv: Path, start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> Dict[str, Any]:
    df = pd.read_csv(external_csv)
    df = _crop_df_by_window(df, start_ns, end_ns)
    if df.empty:
        return {}
    t = _time_col_for_df(df).astype('int64').to_numpy(dtype='float64') / 1e9
    p_w = pd.to_numeric(df['power_w'], errors='coerce').to_numpy(dtype='float64')
    integrate_trapezoid = getattr(np, 'trapezoid', np.trapz)
    energy_j = float(integrate_trapezoid(p_w, t)) if len(t) >= 2 else 0.0
    temp_col = 'external_temp_c' if 'external_temp_c' in df.columns else ('tmp117_temp_c' if 'tmp117_temp_c' in df.columns else None)
    temp_series = pd.to_numeric(df.get(temp_col), errors='coerce').dropna() if temp_col else pd.Series(dtype=float)
    return {
        'samples': int(len(df)),
        'avg_power_w': float(np.nanmean(p_w)),
        'peak_power_w': float(np.nanmax(p_w)),
        'mean_voltage_v': float(pd.to_numeric(df['bus_voltage_v'], errors='coerce').mean()),
        'mean_current_a': float(pd.to_numeric(df['current_a'], errors='coerce').mean()),
        'energy_joules': energy_j,
        'energy_wh': energy_j / 3600.0,
        'external_temp_mean_c': float(temp_series.mean()) if len(temp_series) else None,
        'external_temp_max_c': float(temp_series.max()) if len(temp_series) else None,
    }


def _build_final_external_summary(aggregate_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if aggregate_df.empty:
        return None

    def _mean_if_present(column: str) -> Optional[float]:
        return _mean_numeric_column(aggregate_df, column)

    repeat_count_with_external = 0
    if 'avg_power_external_w' in aggregate_df.columns:
        repeat_count_with_external = int(pd.to_numeric(aggregate_df['avg_power_external_w'], errors='coerce').dropna().shape[0])

    summary = {
        'repeat_count_with_external': repeat_count_with_external,
        'avg_power_w': _mean_if_present('avg_power_external_w'),
        'peak_power_w': _mean_if_present('peak_power_external_w'),
        'energy_joules': _mean_if_present('energy_joules_external'),
        'energy_wh': _mean_if_present('energy_wh_external'),
        'joules_per_image': _mean_if_present('joules_per_image_external'),
        'external_temp_mean_c': _mean_if_present('external_temp_mean_c'),
        'external_temp_max_c': _mean_if_present('external_temp_max_c'),
    }
    if all(value is None for key, value in summary.items() if key != 'repeat_count_with_external'):
        return None
    return summary


def _mean_numeric_column(aggregate_df: pd.DataFrame, column: str) -> Optional[float]:
    if aggregate_df.empty or column not in aggregate_df.columns:
        return None
    series = pd.to_numeric(aggregate_df[column], errors='coerce').dropna()
    if series.empty:
        return None
    return float(series.mean())


def _build_final_power_summary(
    aggregate_df: pd.DataFrame,
    external_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    summary = {
        'avg_power_external_w': _mean_numeric_column(aggregate_df, 'avg_power_external_w'),
        'peak_power_external_w': _mean_numeric_column(aggregate_df, 'peak_power_external_w'),
        'energy_joules_external': _mean_numeric_column(aggregate_df, 'energy_joules_external'),
        'energy_wh_external': _mean_numeric_column(aggregate_df, 'energy_wh_external'),
        'joules_per_image_external': _mean_numeric_column(aggregate_df, 'joules_per_image_external'),
        'external_temp_mean_c': _mean_numeric_column(aggregate_df, 'external_temp_mean_c'),
        'external_temp_max_c': _mean_numeric_column(aggregate_df, 'external_temp_max_c'),
        'internal_total_power_mean_w': _mean_numeric_column(aggregate_df, 'internal_total_power_mean_w'),
        'internal_total_power_peak_w': _mean_numeric_column(aggregate_df, 'internal_total_power_peak_w'),
        'internal_temp_mean_c': _mean_numeric_column(aggregate_df, 'internal_temp_mean_c'),
        'internal_temp_max_c': _mean_numeric_column(aggregate_df, 'internal_temp_max_c'),
    }
    if external_summary:
        summary['repeat_count_with_external'] = external_summary.get('repeat_count_with_external')
    if all(value is None for value in summary.values()):
        return None
    return summary


def _build_final_performance_summary(aggregate_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    summary = {
        'images_benchmarked': _mean_numeric_column(aggregate_df, 'images_benchmarked'),
        'benchmark_duration_s': _mean_numeric_column(aggregate_df, 'benchmark_duration_s'),
        'warmup_elapsed_s': _mean_numeric_column(aggregate_df, 'warmup_elapsed_s'),
        'latency_median_ms': _mean_numeric_column(aggregate_df, 'latency_median_ms'),
        'latency_p95_ms': _mean_numeric_column(aggregate_df, 'latency_p95_ms'),
        'latency_mean_ms': _mean_numeric_column(aggregate_df, 'latency_mean_ms'),
        'throughput_fps': _mean_numeric_column(aggregate_df, 'throughput_fps'),
    }
    if all(value is None for value in summary.values()):
        return None
    return summary


def _parse_thermal_csv(thermal_csv: Path, start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> Dict[str, Any]:
    df = pd.read_csv(thermal_csv)
    df = _crop_df_by_window(df, start_ns, end_ns)
    if df.empty:
        return {}
    summary = {}
    for ttype, group in df.groupby('type', dropna=False):
        vals = pd.to_numeric(group['temp_c'], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        key = str(ttype) if ttype else 'unknown'
        summary[key] = {'mean_c': float(vals.mean()), 'max_c': float(vals.max())}
    all_vals = pd.to_numeric(df['temp_c'], errors='coerce').dropna()
    if len(all_vals):
        summary['overall_internal_max_c'] = float(all_vals.max())
        summary['overall_internal_mean_c'] = float(all_vals.mean())
    return summary


def _parse_hwmon_csv(hwmon_csv: Path, start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> Dict[str, Any]:
    df = pd.read_csv(hwmon_csv)
    df = _crop_df_by_window(df, start_ns, end_ns)
    if df.empty:
        return {}

    def _rail_name(row: pd.Series) -> str:
        label = str(row.get('label', '') or '').strip()
        if label:
            return label
        sensor_name = str(row.get('sensor_name', '') or '').strip() or 'hwmon'
        channel = str(row.get('channel_id', '') or '?')
        return f'{sensor_name}_ch{channel}'

    df['rail_name'] = df.apply(_rail_name, axis=1)
    reported = pd.to_numeric(df.get('power_mW_reported'), errors='coerce')
    computed = pd.to_numeric(df.get('power_mW_computed'), errors='coerce')
    df['power_mW'] = reported.fillna(computed)
    df = df.dropna(subset=['power_mW'])
    if df.empty:
        return {}
    rails: Dict[str, Dict[str, float]] = {}
    for rail, group in df.groupby('rail_name'):
        vals = pd.to_numeric(group['power_mW'], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        rails[str(rail)] = {'mean_w': float(vals.mean() / 1000.0), 'max_w': float(vals.max() / 1000.0)}

    time_key = 'sched_wallclock_ns' if 'sched_wallclock_ns' in df.columns else 'wallclock_ns_end'
    totals = df.groupby(time_key)['power_mW'].sum()
    return {
        'samples': int(len(totals)),
        'mean_total_power_w': float(totals.mean() / 1000.0),
        'peak_total_power_w': float(totals.max() / 1000.0),
        'rails': rails,
    }


def _parse_tegrastats_csv(tstats_csv: Path, start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> Dict[str, Any]:
    df = pd.read_csv(tstats_csv)
    df = _crop_df_by_window(df, start_ns, end_ns)
    if df.empty:
        return {}
    out: Dict[str, Any] = {'samples': int(len(df))}
    for col in [
        'ram_used_mb',
        'ram_total_mb',
        'lfb_blocks',
        'lfb_size_mb',
        'lfb_total_mb',
        'swap_used_mb',
        'swap_total_mb',
        'swap_cached_mb',
        'gr3d_pct',
        'gr3d_freq_mhz',
        'emc_pct',
        'emc_freq_mhz',
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals):
                out[f'{col}_mean'] = float(vals.mean())
                out[f'{col}_min'] = float(vals.min())
                out[f'{col}_max'] = float(vals.max())
    return out


def _parse_process_memory_csv(mem_csv: Path, start_ns: Optional[int] = None, end_ns: Optional[int] = None) -> Dict[str, Any]:
    df = pd.read_csv(mem_csv)
    df = _crop_df_by_window(df, start_ns, end_ns)
    if df.empty:
        return {}
    out: Dict[str, Any] = {'samples': int(len(df))}
    for col in ['pid_count', 'rss_mb', 'vms_mb', 'shared_mb', 'text_mb', 'data_mb', 'uss_mb', 'pss_mb', 'swap_mb']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals):
                out[f'{col}_mean'] = float(vals.mean())
                out[f'{col}_min'] = float(vals.min())
                out[f'{col}_max'] = float(vals.max())
    if 'root_label' in df.columns:
        non_null = df['root_label'].dropna().astype(str)
        if len(non_null):
            out['root_label'] = non_null.iloc[-1]
    if 'root_pid' in df.columns:
        pid_vals = pd.to_numeric(df['root_pid'], errors='coerce').dropna()
        if len(pid_vals):
            out['root_pid'] = int(pid_vals.iloc[-1])
    return out


def _aggregate_latency(latencies_ms: List[float]) -> Dict[str, Any]:
    arr = np.array(latencies_ms, dtype=float)
    if arr.size == 0:
        return {'count': 0, 'median_ms': None, 'p95_ms': None, 'mean_ms': None, 'std_ms': None}
    return {
        'count': int(arr.size),
        'median_ms': float(np.median(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _merge_args_into_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {
        'project': {
            'name': 'jetson_edge_ai_benchmark',
            'task': 'detection',
            'model_name': 'yolo11s',
            'board_name': args.board_name or 'Jetson AGX Orin 64GB',
            'run_label': args.run_name or f'jetson_run_{int(time.time())}',
            'precision_label': args.precision_label,
            'run_variant': args.run_variant,
            'notes': '',
        },
        'model': {
            'model_path': args.model,
            'accuracy_model_path': args.accuracy_model,
            'backend': 'auto',
            'imgsz': args.imgsz,
            'conf': args.conf,
            'iou': args.iou,
            'max_det': args.max_det,
            'half': False,
        },
        'dataset': {
            'root': args.dataset_root,
            'data_yaml': args.data_yaml,
            'ordered_manifest': args.manifest,
            'split_name': args.split_name,
        },
        'execution': {
            'repeats': args.repeats,
            'warmup_seconds': args.warmup_seconds,
            'warmup_images_min': args.warmup_images,
            'min_measured_images': args.min_images,
            'min_measured_duration_s': args.min_duration_s,
            'batch': args.batch,
            'save_per_image_outputs': args.save_per_image_outputs,
            'save_predictions_first_repeat': args.save_per_image_outputs and (not args.no_save_predictions),
            'accuracy_pass_before_benchmark': not args.skip_accuracy_pass,
            'deterministic_order': True,
        },
        'telemetry': {
            'tegrastats_interval_ms': args.tegrastats_interval_ms,
            'internal_sample_interval_ms': args.internal_interval_ms,
            'external_logger_url': args.logger_url,
            'external_sample_hz': args.external_sample_hz,
            'expect_external_logger': args.expect_external_logger,
            'fast_temp_sensor': True,
        },
        'sync': {'start_delay_s': args.start_delay_s},
        'power_mode': {
            'nvpmodel_mode': args.nvpmodel_mode,
            'jetson_clocks': args.jetson_clocks,
            'power_mode_label': args.power_mode_label,
            'matched_budget_label': args.matched_budget_label,
            'observed_mode_name': args.observed_power_mode_name,
            'observed_mode_id': args.observed_power_mode_id,
            'observed_mode_class': args.observed_power_mode_class,
            'observed_mode_tag': args.observed_power_mode_tag,
            'observed_power_budget_w': args.observed_power_budget_w,
            'detection_source': args.observed_power_mode_source,
        },
        'oled': {'enabled': args.oled_enable, 'bus': args.oled_bus, 'addr': args.oled_addr},
        'noise_guard': {
            'enabled': args.noise_guard,
            'profile': args.noise_guard_profile,
            'preserve_url': args.noise_guard_preserve_url,
            'display_off': args.noise_guard_display_off,
            'stop_services': args.noise_guard_stop_services,
            'disable_wifi': args.noise_guard_disable_wifi,
            'disable_bluetooth': args.noise_guard_disable_bluetooth,
        },
        'outputs': {'root_dir': args.output_root},
    }
    cfg = overlay_nested(cfg, config)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Proposal-oriented Jetson benchmark with internal telemetry and an external Raspberry Pi logger.')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--model', type=str, required=False, help='Path to ONNX / TensorRT engine / PT file used for measured inference')
    p.add_argument('--accuracy-model', type=str, default=None, help='Optional model path used only for Ultralytics val()')
    p.add_argument('--dataset-root', type=str, default=None, help='Path to an unpacked portable test-set root, the dataset zip, or yolo11s/inference_bundle')
    p.add_argument('--data-yaml', type=str, default=None, help='Optional dataset YAML; auto-detected from the portable dataset when omitted')
    p.add_argument('--manifest', type=str, default=None, help='Optional fixed-order manifest; auto-detected from the portable dataset when omitted')
    p.add_argument('--split-name', type=str, default='test')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--board-name', type=str, default=None)
    p.add_argument('--precision-label', type=str, default=None, help='Short label such as fp16 or fp32 stored in FINAL_SUMMARY.json')
    p.add_argument('--run-variant', type=str, default=None, help='Session variant label such as standard or per_image')
    p.add_argument('--output-root', type=str, default=str(default_output_root('jetson_benchmark_runs')))
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup-images', type=int, default=32, help='Minimum warm-up images run BEFORE measured logging/inference')
    p.add_argument('--warmup-seconds', type=float, default=300.0, help='Proposal default is 300 seconds')
    p.add_argument('--min-images', type=int, default=500, help='Proposal default minimum measured images per repeat')
    p.add_argument('--min-duration-s', type=float, default=180.0, help='Proposal default minimum measured duration per repeat')
    p.add_argument('--skip-accuracy-pass', action='store_true')
    p.add_argument('--no-save-predictions', action='store_true')
    p.add_argument('--save-per-image-outputs', dest='save_per_image_outputs', action='store_true', help='Keep per-image latency/timing artifacts (default: disabled)')
    p.add_argument('--no-save-per-image-outputs', dest='save_per_image_outputs', action='store_false', help='Do not keep per-image benchmark artifacts such as per_image_latency.csv or trtexec_times.json')
    p.add_argument('--logger-url', '--pi-logger-url', dest='logger_url', type=str, default=None, help='Example: http://${LOGGER_DASHBOARD_HOST}:8000')
    p.add_argument('--external-sample-hz', type=float, default=10.0)
    p.add_argument('--expect-external-logger', action='store_true')
    p.add_argument('--tegrastats-interval-ms', type=int, default=200)
    p.add_argument('--internal-interval-ms', type=int, default=200)
    p.add_argument('--start-delay-s', type=float, default=5.0)
    p.add_argument('--nvpmodel-mode', type=str, default=None)
    p.add_argument('--jetson-clocks', action='store_true')
    p.add_argument('--power-mode-label', type=str, default='native')
    p.add_argument('--matched-budget-label', type=str, default=None)
    p.add_argument('--observed-power-mode-name', type=str, default=None)
    p.add_argument('--observed-power-mode-id', type=str, default=None)
    p.add_argument('--observed-power-mode-class', type=str, default=None)
    p.add_argument('--observed-power-mode-tag', type=str, default=None)
    p.add_argument('--observed-power-budget-w', type=float, default=None)
    p.add_argument('--observed-power-mode-source', type=str, default=None)
    p.add_argument('--oled-enable', action='store_true', help='Enable SSD1306 128x32 I2C OLED progress display on Jetson')
    p.add_argument('--oled-bus', type=int, default=None, help='OLED I2C bus. If omitted, auto-detect /dev/i2c-*')
    p.add_argument('--oled-addr', type=lambda x: int(x, 0), default=None, help='OLED I2C address, e.g. 0x3C')
    p.add_argument('--noise-guard', dest='noise_guard', action='store_true', help='Temporarily quiet common non-benchmark subsystems while inference is running (default: enabled)')
    p.add_argument('--no-noise-guard', dest='noise_guard', action='store_false', help='Disable automatic quiet-mode actions')
    p.add_argument('--noise-guard-profile', choices=['safe', 'aggressive'], default='safe')
    p.add_argument('--noise-guard-preserve-url', type=str, default=None, help='URL/host that must stay reachable; defaults to the external logger URL when set')
    p.add_argument('--noise-guard-display-off', dest='noise_guard_display_off', action='store_true', help='Best-effort DPMS display off during the run (default: enabled)')
    p.add_argument('--no-noise-guard-display-off', dest='noise_guard_display_off', action='store_false')
    p.add_argument('--noise-guard-stop-services', dest='noise_guard_stop_services', action='store_true', help='Stop selected background services/timers during the run (default: enabled)')
    p.add_argument('--no-noise-guard-stop-services', dest='noise_guard_stop_services', action='store_false')
    p.add_argument('--noise-guard-disable-wifi', dest='noise_guard_disable_wifi', action='store_true', help='Turn Wi-Fi off during the run when it is not needed for the logger route (default: enabled)')
    p.add_argument('--no-noise-guard-disable-wifi', dest='noise_guard_disable_wifi', action='store_false')
    p.add_argument('--noise-guard-disable-bluetooth', dest='noise_guard_disable_bluetooth', action='store_true', help='Block Bluetooth during the run (default: enabled)')
    p.add_argument('--no-noise-guard-disable-bluetooth', dest='noise_guard_disable_bluetooth', action='store_false')
    p.set_defaults(noise_guard=True, noise_guard_display_off=True, noise_guard_stop_services=True, noise_guard_disable_wifi=True, noise_guard_disable_bluetooth=True, save_per_image_outputs=False)
    return p


def run_inference_once(model: Any, image_path: Path, imgsz: int, device: str, conf: float, iou: float, max_det: int):
    return model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        stream=False,
        save=False,
    )


def resolve_trtexec_binary() -> str:
    local_trtexec = os.environ.get('LOCAL_TRTEXEC_PATH')
    local_lib_dir = os.environ.get('LOCAL_TENSORRT_LIB_DIR')
    if not local_trtexec:
        workspace = Path(__file__).resolve().parents[2]
        bundled_root = workspace / '.vendor' / 'tensorrt-${PRIVATE_LAN_IP}'
        bundled_trtexec = bundled_root / 'usr' / 'src' / 'tensorrt' / 'bin' / 'trtexec'
        bundled_lib_dir = bundled_root / 'usr' / 'lib' / 'aarch64-linux-gnu'
        if bundled_trtexec.is_file():
            local_trtexec = str(bundled_trtexec)
            local_lib_dir = str(bundled_lib_dir)

    if local_trtexec and Path(local_trtexec).is_file():
        if local_lib_dir:
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            ld_parts = [part for part in current_ld.split(':') if part]
            if local_lib_dir not in ld_parts:
                os.environ['LD_LIBRARY_PATH'] = f'{local_lib_dir}:{current_ld}' if current_ld else local_lib_dir
        os.environ.setdefault('TRTEXEC_PATH', str(local_trtexec))
        return str(Path(local_trtexec).resolve())

    env_path = os.environ.get('TRTEXEC_PATH')
    if env_path and Path(env_path).is_file():
        return env_path
    path_hit = shutil.which('trtexec')
    if path_hit:
        return path_hit
    for candidate in (
        Path('/usr/src/tensorrt/bin/trtexec'),
        Path('/usr/src/tensorrt/samples/trtexec'),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError('trtexec not found. Set TRTEXEC_PATH or install JetPack TensorRT binaries.')


def _wallclock_ns_to_utc_iso(ns_value: int) -> str:
    return datetime.fromtimestamp(ns_value / 1e9, tz=timezone.utc).isoformat()


def _run_subprocess_with_log(cmd: List[str], log_path: Path, progress_cb=None, process_memory_sampler: Optional[ProcessMemorySampler] = None, process_label: Optional[str] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        if process_memory_sampler is not None:
            process_memory_sampler.set_root_pid(proc.pid, label=process_label or Path(cmd[0]).name)
        start_ns = monotonic_ns()
        last_cb_ns = 0
        while True:
            rc = proc.poll()
            if rc is not None:
                if process_memory_sampler is not None:
                    process_memory_sampler.set_root_pid(None, label=process_label or Path(cmd[0]).name)
                return rc
            now_ns = monotonic_ns()
            if progress_cb is not None and (now_ns - last_cb_ns) >= 1_000_000_000:
                progress_cb((now_ns - start_ns) / 1e9)
                last_cb_ns = now_ns
            time.sleep(0.2)


def warmup_engine_with_trtexec(
    *,
    engine_path: Path,
    warmup_seconds: float,
    warmup_images_min: int,
    repeat_root: Path,
    progress_cb=None,
    process_memory_sampler: Optional[ProcessMemorySampler] = None,
) -> Dict[str, Any]:
    if warmup_seconds <= 0 and warmup_images_min <= 0:
        return {'warmup_images': 0, 'warmup_cycles': 0, 'warmup_elapsed_s': 0.0, 'warmup_method': 'skipped'}

    trtexec = resolve_trtexec_binary()
    warmup_iterations = max(1, int(warmup_images_min))
    warmup_ms = max(0, int(float(warmup_seconds) * 1000.0))
    log_path = repeat_root / 'trtexec_warmup.log'
    cmd = [
        trtexec,
        f'--loadEngine={engine_path}',
        f'--iterations={warmup_iterations}',
        '--duration=0',
        f'--warmUp={warmup_ms}',
        '--useCudaGraph',
        '--separateProfileRun',
    ]
    start_ns = monotonic_ns()
    rc = _run_subprocess_with_log(cmd, log_path, progress_cb=progress_cb, process_memory_sampler=process_memory_sampler, process_label='trtexec_warmup')
    if rc != 0:
        raise RuntimeError(f'trtexec warmup failed. See {log_path}')
    elapsed_s = (monotonic_ns() - start_ns) / 1e9
    return {
        'warmup_images': warmup_iterations,
        'warmup_cycles': None,
        'warmup_elapsed_s': elapsed_s,
        'warmup_method': 'trtexec_random_input',
        'trtexec_warmup_log': str(log_path),
    }


def benchmark_engine_with_trtexec(
    *,
    engine_path: Path,
    image_paths: List[Path],
    repeat_name: str,
    repeat_root: Path,
    timing_csv: Optional[Path],
    imgsz: int,
    min_images: int,
    min_duration_s: float,
    benchmark_start_wallclock_ns: int,
    benchmark_start_monotonic_ns: int,
    progress_cb=None,
    process_memory_sampler: Optional[ProcessMemorySampler] = None,
    save_per_image_outputs: bool = False,
    task: str = 'detect',
) -> Dict[str, Any]:
    trtexec = resolve_trtexec_binary()
    export_times_path = repeat_root / 'trtexec_times.json'
    log_path = repeat_root / 'trtexec_benchmark.log'
    min_iterations = max(1, int(min_images))
    duration_s = max(0, int(math.ceil(float(min_duration_s))))
    cmd = [
        trtexec,
        f'--loadEngine={engine_path}',
        f'--iterations={min_iterations}',
        f'--duration={duration_s}',
        '--warmUp=0',
        '--useCudaGraph',
        '--separateProfileRun',
        f'--exportTimes={export_times_path}',
    ]
    if imgsz > 0:
        # Keep a record of the expected static input shape; harmless for static engines when omitted from trtexec.
        pass
    rc = _run_subprocess_with_log(cmd, log_path, progress_cb=progress_cb, process_memory_sampler=process_memory_sampler, process_label='trtexec_benchmark')
    if rc != 0:
        raise RuntimeError(f'trtexec benchmark failed. See {log_path}')
    if not export_times_path.exists():
        raise FileNotFoundError(export_times_path)

    times = json.loads(export_times_path.read_text(encoding='utf-8'))
    if not isinstance(times, list) or not times:
        raise RuntimeError(f'trtexec did not export timing samples to {export_times_path}')

    latencies_ms: List[float] = []
    writer = None
    timing_handle = None
    if save_per_image_outputs and timing_csv is not None:
        fieldnames = _per_image_latency_fieldnames(task)
        timing_handle = open(timing_csv, 'w', encoding='utf-8', newline='')
        writer = csv.DictWriter(timing_handle, fieldnames=fieldnames)
        writer.writeheader()
    try:
        for idx, sample in enumerate(times):
            manifest_index = idx % len(image_paths)
            cycle_index = idx // len(image_paths)
            image_path = image_paths[manifest_index]
            start_ms = float(sample.get('startComputeMs', sample.get('startEnqMs', 0.0)))
            end_ms = float(sample.get('endComputeMs', sample.get('endEnqMs', start_ms)))
            latency_ms = float(sample.get('latencyMs', max(0.0, end_ms - start_ms)))
            wall_start = benchmark_start_wallclock_ns + int(round(start_ms * 1e6))
            wall_end = benchmark_start_wallclock_ns + int(round(end_ms * 1e6))
            mono_start = benchmark_start_monotonic_ns + int(round(start_ms * 1e6))
            mono_end = benchmark_start_monotonic_ns + int(round(end_ms * 1e6))
            latencies_ms.append(latency_ms)
            if writer is not None:
                row = {
                    'repeat': repeat_name,
                    'image_global_index': idx + 1,
                    'cycle_index': cycle_index,
                    'manifest_index': manifest_index,
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'timestamp_utc': _wallclock_ns_to_utc_iso(wall_start),
                    'wallclock_ns_start': wall_start,
                    'wallclock_ns_end': wall_end,
                    'monotonic_ns_start': mono_start,
                    'monotonic_ns_end': mono_end,
                    'latency_ms': latency_ms,
                }
                if task != 'semantic_segmentation':
                    row['num_detections'] = 0
                writer.writerow(row)
    finally:
        if timing_handle is not None:
            timing_handle.close()
    if not save_per_image_outputs:
        try:
            export_times_path.unlink(missing_ok=True)
        except Exception:
            pass
    return {
        'latencies_ms': latencies_ms,
        'processed_images': len(times),
        'saved_pred_names': set(),
        'trtexec_log': str(log_path),
        'trtexec_times_json': str(export_times_path) if save_per_image_outputs and export_times_path.exists() else None,
        'measurement_backend': 'trtexec_random_input',
    }


def warmup_model(
    model: Any,
    image_paths: List[Path],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    warmup_seconds: float,
    warmup_images_min: int,
    progress_cb=None,
) -> Dict[str, Any]:
    if warmup_seconds <= 0 and warmup_images_min <= 0:
        return {'warmup_images': 0, 'warmup_cycles': 0, 'warmup_elapsed_s': 0.0}
    start_ns = monotonic_ns()
    count = 0
    total = len(image_paths)
    last_cb_ns = 0
    while count < warmup_images_min or ((monotonic_ns() - start_ns) / 1e9) < warmup_seconds:
        image_path = image_paths[count % total]
        _ = run_inference_once(model, image_path, imgsz, device, conf, iou, max_det)
        count += 1
        elapsed_s = (monotonic_ns() - start_ns) / 1e9
        rate = count / max(elapsed_s, 1e-6)
        remaining_images = max(0, warmup_images_min - count)
        eta_images = remaining_images / max(rate, 1e-6)
        eta_time = max(0.0, warmup_seconds - elapsed_s)
        eta_s = max(eta_images, eta_time)
        now_ns = monotonic_ns()
        if progress_cb is not None and (count == 1 or (now_ns - last_cb_ns) >= 1_000_000_000):
            progress_cb(count, warmup_images_min, elapsed_s, eta_s, image_path.name)
            last_cb_ns = now_ns
    elapsed_s = (monotonic_ns() - start_ns) / 1e9
    return {
        'warmup_images': count,
        'warmup_cycles': count / total if total else 0.0,
        'warmup_elapsed_s': elapsed_s,
    }


def run_inference_once(model: Any, image_path: Path, imgsz: int, device: str, conf: float, iou: float, max_det: int):
    return model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        stream=False,
        save=False,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    file_cfg = load_yaml(args.config) if args.config else {}
    cfg = _merge_args_into_config(file_cfg, args)

    bundle_root = resolve_inference_bundle_root(cfg['dataset'].get('root') or cfg['model'].get('model_path'))
    dataset_root = resolve_dataset_root(cfg['dataset'].get('root'), bundle_root=bundle_root)
    model_path = resolve_local_model_path(cfg['model'].get('model_path'), bundle_root=bundle_root)
    accuracy_model_path = resolve_local_accuracy_model_path(
        explicit=cfg['model'].get('accuracy_model_path'),
        benchmark_model_path=model_path,
        bundle_root=bundle_root,
    )
    output_root = ensure_dir(resolve_output_root(cfg['outputs'].get('root_dir'), 'jetson_benchmark_runs'))
    run_name = cfg['project']['run_label']
    run_root = ensure_dir(output_root / run_name)
    notes: List[str] = []
    total_repeats = int(cfg['execution'].get('repeats', 3))
    session_start_wallclock_ns = wallclock_ns()

    data_yaml = resolve_data_yaml_path(dataset_root=dataset_root, explicit=cfg['dataset'].get('data_yaml'), bundle_root=bundle_root)
    manifest = resolve_manifest_path(dataset_root=dataset_root, explicit=cfg['dataset'].get('ordered_manifest'), bundle_root=bundle_root)
    data_yaml, manifest = discover_dataset_paths(dataset_root=dataset_root, data_yaml=data_yaml, manifest=manifest)
    image_paths = list_images_from_manifest(dataset_root, manifest)
    if not image_paths:
        raise RuntimeError('No benchmark images found.')
    imgsz = int(cfg['model'].get('imgsz', 640))
    batch = int(cfg['execution'].get('batch', 1))
    conf = float(cfg['model'].get('conf', 0.25))
    iou = float(cfg['model'].get('iou', 0.7))
    max_det = int(cfg['model'].get('max_det', 300))
    device = str(args.device)

    cfg['model']['model_path'] = str(model_path)
    cfg['model']['accuracy_model_path'] = str(accuracy_model_path) if accuracy_model_path is not None else None
    cfg['dataset']['root'] = str(dataset_root)
    cfg['dataset']['data_yaml'] = str(data_yaml) if data_yaml is not None else None
    cfg['dataset']['ordered_manifest'] = str(manifest)
    cfg['dataset']['benchmark_manifest'] = str(manifest)
    cfg['dataset']['manifest'] = str(manifest)
    cfg['dataset'].pop('accuracy_manifest', None)
    cfg['outputs']['root_dir'] = str(output_root)

    dump_yaml(cfg, run_root / 'config_snapshot.yaml')
    (run_root / 'manifest_snapshot.txt').write_text(manifest.read_text(encoding='utf-8'), encoding='utf-8')
    if data_yaml is not None and data_yaml.exists():
        (run_root / 'data_yaml_snapshot.yaml').write_text(data_yaml.read_text(encoding='utf-8'), encoding='utf-8')
    snapshot_system(run_root / 'system')
    measured_model_sha256 = sha256_file(model_path)
    resolved_accuracy_model_path = accuracy_model_path
    accuracy_model_sha256 = sha256_file(accuracy_model_path) if accuracy_model_path is not None and accuracy_model_path.exists() else None
    accuracy_backend = _infer_accuracy_backend_from_model_path(resolved_accuracy_model_path)
    accuracy_backend_detail: Optional[str] = None
    accuracy_precision_label = _infer_precision_label_from_model_path(resolved_accuracy_model_path)
    model_complexity, complexity_json_path, complexity_onnx_path = _profile_or_load_model_complexity(
        model_path=model_path,
        accuracy_model_path=accuracy_model_path,
        bundle_root=bundle_root,
        imgsz=imgsz,
        batch=batch,
        notes=notes,
    )

    oled_cfg = cfg.get('oled', {}) or {}
    oled = BenchmarkOLED(enabled=bool(oled_cfg.get('enabled', False)), bus=oled_cfg.get('bus'), addr=oled_cfg.get('addr'))
    if oled.active:
        oled.boot('Preparing', precision_label=cfg['project'].get('precision_label'), run_variant=cfg['project'].get('run_variant'))

    logger_url = cfg['telemetry'].get('external_logger_url')
    dashboard_status_url = logger_url.rstrip('/') + '/api/jetson_status' if logger_url else None
    status_push_failed = False
    noise_guard: Optional[NoiseGuard] = None
    noise_guard_summary_path = run_root / 'noise_guard_summary.json'

    def _current_internal_live_snapshot(
        internal_sampler: Optional[InternalSampler],
        tegrastats_logger: Optional[TegrastatsLogger],
        process_memory_sampler: Optional[ProcessMemorySampler] = None,
    ) -> Dict[str, Any]:
        snap: Dict[str, Any] = {'hwmon': {}, 'thermal': {}, 'tegrastats': {}, 'process_memory': {}}
        if internal_sampler is not None:
            try:
                latest = internal_sampler.get_latest_snapshot()
                if isinstance(latest, dict):
                    snap['hwmon'] = latest.get('hwmon', {}) or {}
                    snap['thermal'] = latest.get('thermal', {}) or {}
            except Exception:
                pass
        if tegrastats_logger is not None:
            try:
                latest_t = tegrastats_logger.get_latest_snapshot()
                if isinstance(latest_t, dict):
                    snap['tegrastats'] = latest_t
            except Exception:
                pass
        if process_memory_sampler is not None:
            try:
                latest_p = process_memory_sampler.get_latest_snapshot()
                if isinstance(latest_p, dict):
                    snap['process_memory'] = latest_p
            except Exception:
                pass
        return snap

    def _push_status(
        *,
        stage: str,
        state: str = 'running',
        message: str = '',
        repeat_idx: Optional[int] = None,
        repeat_name: Optional[str] = None,
        stage_start_ns: Optional[int] = None,
        stage_eta_s: Optional[float] = None,
        processed_images: Optional[int] = None,
        min_images_target: Optional[int] = None,
        min_duration_s_target: Optional[float] = None,
        warmup_images_done: Optional[int] = None,
        warmup_images_target: Optional[int] = None,
        current_image_name: Optional[str] = None,
        latency_ms_last: Optional[float] = None,
        internal_sampler: Optional[InternalSampler] = None,
        tegrastats_logger: Optional[TegrastatsLogger] = None,
        process_memory_sampler: Optional[ProcessMemorySampler] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal status_push_failed
        now_ns = wallclock_ns()
        overall_elapsed_s = (now_ns - session_start_wallclock_ns) / 1e9
        stage_elapsed_s = (now_ns - stage_start_ns) / 1e9 if stage_start_ns is not None else None

        progress_pct = 0.0
        if stage == 'warmup':
            img_pct = (warmup_images_done / warmup_images_target) if (warmup_images_done is not None and warmup_images_target) else 0.0
            time_pct = (stage_elapsed_s / float(cfg['execution'].get('warmup_seconds', 0.0))) if stage_elapsed_s is not None and float(cfg['execution'].get('warmup_seconds', 0.0)) > 0 else 0.0
            progress_pct = 100.0 * min(max(img_pct, 0.0), max(time_pct, 0.0))
        elif stage == 'benchmark':
            img_pct = (processed_images / min_images_target) if (processed_images is not None and min_images_target) else 0.0
            time_pct = (stage_elapsed_s / min_duration_s_target) if (stage_elapsed_s is not None and min_duration_s_target and min_duration_s_target > 0) else 0.0
            progress_pct = 100.0 * min(max(img_pct, 0.0), max(time_pct, 0.0))
        elif stage in {'repeat_complete', 'finished'}:
            progress_pct = 100.0
        progress_pct = float(max(0.0, min(100.0, progress_pct)))
        overall_progress_pct = None
        if repeat_idx is not None and total_repeats > 0:
            overall_progress_pct = 100.0 * (((repeat_idx - 1) + progress_pct / 100.0) / total_repeats)

        payload: Dict[str, Any] = {
            'run_id': run_name,
            'board_name': cfg['project'].get('board_name'),
            'precision_label': cfg['project'].get('precision_label'),
            'run_variant': cfg['project'].get('run_variant'),
            'stage': stage,
            'state': state,
            'message': message,
            'timestamp_utc': utc_now_iso(),
            'wallclock_ns': now_ns,
            'overall_elapsed_s': overall_elapsed_s,
            'stage_elapsed_s': stage_elapsed_s,
            'stage_eta_s': stage_eta_s,
            'progress_pct': progress_pct,
            'overall_progress_pct': overall_progress_pct,
            'repeat_index': repeat_idx,
            'total_repeats': total_repeats,
            'repeat_name': repeat_name,
            'processed_images': processed_images,
            'min_images_target': min_images_target,
            'min_duration_s_target': min_duration_s_target,
            'warmup_images_done': warmup_images_done,
            'warmup_images_target': warmup_images_target,
            'current_image_name': current_image_name,
            'latency_ms_last': latency_ms_last,
            'power_mode': cfg['power_mode'],
            'internal_live': _current_internal_live_snapshot(internal_sampler, tegrastats_logger, process_memory_sampler),
            'noise_guard': noise_guard.dashboard_payload() if noise_guard is not None else None,
        }
        if extra:
            payload.update(extra)

        if oled.active:
            if stage == 'warmup':
                oled.warmup(
                    repeat_name or '-',
                    warmup_images_done or 0,
                    warmup_images_target or 0,
                    stage_elapsed_s or 0.0,
                    stage_eta_s,
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )
            elif stage == 'countdown':
                oled.waiting(
                    stage_eta_s or 0.0,
                    repeat_name or '',
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )
            elif stage == 'benchmark':
                oled.running(
                    repeat_name or '-',
                    processed_images or 0,
                    min_images_target or 0,
                    stage_eta_s,
                    latency_ms_last,
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )
            elif stage == 'finished':
                oled.finished(
                    'benchmark done',
                    f'{int(overall_elapsed_s)}s',
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )
            elif stage == 'error':
                oled.error(
                    'benchmark failed',
                    (message or '')[:20],
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )
            else:
                oled.stage(
                    stage.upper()[:20],
                    (message or '')[:20],
                    current_image_name or '',
                    precision_label=cfg['project'].get('precision_label'),
                    run_variant=cfg['project'].get('run_variant'),
                )

        if dashboard_status_url:
            try:
                post_json(dashboard_status_url, payload, timeout=4)
                status_push_failed = False
            except Exception as exc:
                if not status_push_failed:
                    notes.append(f'Dashboard status push failed once: {exc}')
                    status_push_failed = True

    if not cfg.get('noise_guard', {}).get('preserve_url') and logger_url:
        cfg.setdefault('noise_guard', {})['preserve_url'] = logger_url

    run_error: Optional[BaseException] = None
    try:
        _push_status(stage='boot', message='Preparing benchmark package')

        maybe_set_nvpmodel(cfg['power_mode'].get('nvpmodel_mode'), notes)
        maybe_set_jetson_clocks(bool(cfg['power_mode'].get('jetson_clocks', False)), notes)

        if logger_url:
            try:
                initial_logger_health = get_json(logger_url.rstrip('/') + '/health', timeout=10)
                logger_recovery = ensure_external_logger_idle(
                    logger_url,
                    str(cfg['project'].get('board_name') or '').strip() or None,
                    notes,
                    initial_health=initial_logger_health,
                    idle_timeout_s=30.0,
                )
                logger_health = logger_recovery.get('final_health') or initial_logger_health
                dump_json(logger_health, run_root / 'external_logger_health.json')
                if logger_recovery.get('recovered'):
                    dump_json(logger_recovery, run_root / 'external_logger_recovery.json')
            except Exception as exc:
                msg = f'Could not prepare external logger at {logger_url}: {exc}'
                if cfg['telemetry'].get('expect_external_logger', False):
                    raise RuntimeError(msg)
                notes.append(msg)

        if cfg.get('noise_guard', {}).get('enabled', True):
            noise_guard = NoiseGuard(
                NoiseGuardConfig(
                    enabled=True,
                    profile=str(cfg['noise_guard'].get('profile', 'safe')),
                    preserve_url=cfg['noise_guard'].get('preserve_url'),
                    display_off=bool(cfg['noise_guard'].get('display_off', True)),
                    stop_services=bool(cfg['noise_guard'].get('stop_services', True)),
                    disable_wifi=bool(cfg['noise_guard'].get('disable_wifi', True)),
                    disable_bluetooth=bool(cfg['noise_guard'].get('disable_bluetooth', True)),
                ),
                notes=notes,
            )
            dump_json(noise_guard.enter(), noise_guard_summary_path)
            _push_status(stage='quiet_mode', message='Quiet-mode actions applied before inference')

        accuracy_metrics: Dict[str, Any] = {}
        resolved_accuracy_manifest_path: Optional[Path] = None
        accuracy_precision_specific = False
        if cfg['execution'].get('accuracy_pass_before_benchmark', True):
            accuracy_stage_start_ns = wallclock_ns()
            _push_status(stage='accuracy_pass', message='Running accuracy validation pass before measured benchmark')
            acc_model = resolve_accuracy_model_path(model_path, accuracy_model_path)
            resolved_accuracy_model_path = acc_model
            accuracy_model_sha256 = sha256_file(acc_model) if acc_model is not None and acc_model.exists() else None
            accuracy_backend = _infer_accuracy_backend_from_model_path(acc_model)
            accuracy_backend_detail = None
            accuracy_precision_label = _infer_precision_label_from_model_path(acc_model)
            try:
                if acc_model is None:
                    notes.append('Accuracy pass skipped automatically because benchmark model is an engine and no accuracy model was supplied.')
                elif acc_model.suffix.lower() == '.engine' and cfg['project'].get('task') == 'semantic_segmentation':
                    _push_status(stage='accuracy_pass', message='Running BiSeNetV2 TensorRT engine validation before measured benchmark')
                    accuracy_metrics = bisenetv2_engine_accuracy_pass(
                        model_path=acc_model,
                        dataset_root=dataset_root,
                        split_name=cfg['dataset'].get('split_name', 'test'),
                        imgsz=int(cfg['model'].get('imgsz', 640)),
                        project_dir=run_root,
                        notes=notes,
                    )
                    accuracy_precision_specific = True
                elif acc_model.suffix.lower() == '.engine' and cfg['project'].get('task') == 'detection':
                    _push_status(stage='accuracy_pass', message='Running YOLO11s TensorRT engine validation before measured benchmark')
                    accuracy_metrics = yolo11s_engine_accuracy_pass(
                        model_path=acc_model,
                        dataset_root=dataset_root,
                        split_name=cfg['dataset'].get('split_name', 'test'),
                        imgsz=int(cfg['model'].get('imgsz', 640)),
                        batch=int(cfg['execution'].get('batch', 1)),
                        device=str(args.device),
                        project_dir=run_root,
                        notes=notes,
                    )
                    accuracy_precision_specific = True
                elif acc_model.suffix.lower() in {'.pt', '.pth'} and cfg['project'].get('task') == 'semantic_segmentation':
                    _push_status(stage='accuracy_pass', message='Running BiSeNetV2 segmentation validation before measured benchmark')
                    accuracy_metrics = bisenetv2_repo_accuracy_pass(
                        model_path=acc_model,
                        dataset_root=dataset_root,
                        split_name=cfg['dataset'].get('split_name', 'test'),
                        imgsz=int(cfg['model'].get('imgsz', 640)),
                        batch=int(cfg['execution'].get('batch', 1)),
                        project_dir=run_root,
                        notes=notes,
                    )
                elif data_yaml is None:
                    notes.append('Accuracy pass skipped because data_yaml was not supplied.')
                else:
                    accuracy_data_yaml = _prepare_accuracy_data_yaml(
                        data_yaml=data_yaml,
                        dataset_root=dataset_root,
                        split_name=cfg['dataset'].get('split_name', 'test'),
                        project_dir=run_root,
                        notes=notes,
                    )
                    acc_model = _fallback_accuracy_model_if_needed(acc_model, notes)
                    accuracy_metrics = ultralytics_accuracy_pass(
                        model_path=acc_model,
                        data_yaml=accuracy_data_yaml,
                        split_name=cfg['dataset'].get('split_name', 'test'),
                        imgsz=int(cfg['model'].get('imgsz', 640)),
                        batch=int(cfg['execution'].get('batch', 1)),
                        device=str(args.device),
                        project_dir=run_root,
                    )
            except Exception as exc:
                _push_status(
                    stage='error',
                    state='error',
                    message=f'Accuracy pass failed: {type(exc).__name__}: {exc}',
                    stage_start_ns=accuracy_stage_start_ns,
                )
                raise
            if accuracy_metrics:
                accuracy_backend = str(accuracy_metrics.get('backend') or accuracy_backend or '') or None
                accuracy_backend_detail = accuracy_metrics.get('backend_detail')
                accuracy_precision_label = str(
                    accuracy_metrics.get('precision_label')
                    or accuracy_precision_label
                    or ''
                ) or None
                resolved_accuracy_manifest_path = _resolve_accuracy_manifest_path(accuracy_metrics)
            if accuracy_metrics:
                accuracy_metric_name, accuracy_metric_value = _extract_primary_accuracy_metric(accuracy_metrics)
                _push_status(
                    stage='accuracy_pass',
                    message='Accuracy validation pass complete',
                    stage_start_ns=accuracy_stage_start_ns,
                    extra={
                        'accuracy_metric_name': accuracy_metric_name,
                        'accuracy_metric': accuracy_metric_value,
                        'mAP50_95': accuracy_metrics.get('metrics/mAP50-95(B)', accuracy_metrics.get('metrics/mAP50-95')),
                        'mIoU': accuracy_metrics.get('metrics/mIoU', accuracy_metrics.get('metrics/miou', accuracy_metrics.get('miou'))),
                        'accuracy_metrics_json': str(run_root / 'accuracy' / 'val_metrics.json'),
                    },
                )
        _rewrite_accuracy_config_snapshot(
            cfg,
            run_root,
            benchmark_manifest_path=manifest,
            accuracy_manifest_path=resolved_accuracy_manifest_path,
            accuracy_model_path=resolved_accuracy_model_path,
            accuracy_model_sha256=accuracy_model_sha256,
            accuracy_backend=accuracy_backend,
            accuracy_backend_detail=accuracy_backend_detail,
            accuracy_precision_label=accuracy_precision_label,
            accuracy_precision_specific=accuracy_precision_specific,
        )

        engine_measurement_mode = model_path.suffix.lower() == '.engine'
        task_name = str(cfg['project'].get('task') or 'detect')
        if engine_measurement_mode and task_name == 'semantic_segmentation':
            cfg.setdefault('execution', {})['save_predictions_first_repeat'] = False
        if engine_measurement_mode:
            notes.append(
                'Measured engine benchmark uses trtexec with TensorRT-managed random input buffers because '
                'TensorRT Python execution is unstable in the current JetPack/conda stack on this device.'
            )
            model = None
        else:
            if cfg['project'].get('task') == 'semantic_segmentation':
                raise RuntimeError(
                    'BiSeNetV2 measured benchmarking expects TensorRT engine input. '
                    'Use the provided run_current_mode_int8_fp16_fp32.sh flow so engines are built first.'
                )
            _prime_tensorrt_backend(model_path, notes)
            model = _load_ultralytics_yolo()(str(model_path), task='detect')

        benchmark_runs_root = ensure_dir(run_root / 'benchmark_runs')
        aggregate_rows: List[Dict[str, Any]] = []
        save_per_image_outputs = bool(cfg['execution'].get('save_per_image_outputs', False))

        for repeat_idx in range(total_repeats):
            repeat_name = f'repeat_{repeat_idx + 1:02d}'
            repeat_root = ensure_dir(benchmark_runs_root / repeat_name)
            repeat_external_dir = ensure_dir(repeat_root / 'external')
            repeat_internal_dir = ensure_dir(repeat_root / 'internal')
            pred_dir = None
            if cfg['execution'].get('save_predictions_first_repeat', True) and not engine_measurement_mode:
                pred_dir = ensure_dir(repeat_root / 'predictions')
            timing_csv = (repeat_root / 'per_image_latency.csv') if save_per_image_outputs else None
            markers_json = repeat_root / 'run_markers.json'

            min_images = int(cfg['execution'].get('min_measured_images', 500))
            min_duration_s = float(cfg['execution'].get('min_measured_duration_s', 180.0))
            warmup_seconds = float(cfg['execution'].get('warmup_seconds', 300.0))
            warmup_images_min = int(cfg['execution'].get('warmup_images_min', 32))

            warmup_start_wallclock_ns = wallclock_ns()
            if engine_measurement_mode:
                warmup_summary = warmup_engine_with_trtexec(
                    engine_path=model_path,
                    warmup_seconds=warmup_seconds,
                    warmup_images_min=warmup_images_min,
                    repeat_root=repeat_root,
                    progress_cb=lambda elapsed_s: _push_status(
                        stage='warmup',
                        message='TensorRT engine warm-up before measured logging',
                        repeat_idx=repeat_idx + 1,
                        repeat_name=repeat_name,
                        stage_start_ns=warmup_start_wallclock_ns,
                        stage_eta_s=max(0.0, warmup_seconds - elapsed_s) if warmup_seconds > 0 else None,
                        warmup_images_done=None,
                        warmup_images_target=warmup_images_min,
                        current_image_name='trtexec-random-input',
                    ),
                )
            else:
                warmup_summary = warmup_model(
                    model=model,
                    image_paths=image_paths,
                    imgsz=imgsz,
                    device=device,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    warmup_seconds=warmup_seconds,
                    warmup_images_min=warmup_images_min,
                    progress_cb=lambda done, target, elapsed_s, eta_s, image_name: _push_status(
                        stage='warmup',
                        message='Warm-up before measured logging',
                        repeat_idx=repeat_idx + 1,
                        repeat_name=repeat_name,
                        stage_start_ns=warmup_start_wallclock_ns,
                        stage_eta_s=eta_s,
                        warmup_images_done=done,
                        warmup_images_target=target,
                        current_image_name=image_name,
                    ),
                )

            scheduled_start_ns = wallclock_ns() + int(float(cfg['sync'].get('start_delay_s', 5.0)) * 1e9)
            external_started = False
            external_info: Dict[str, Any] = {}
            external_run_id = f'{run_name}_{repeat_name}'
            internal_sampler: Optional[InternalSampler] = None
            tegrastats_logger: Optional[TegrastatsLogger] = None
            process_memory_sampler: Optional[ProcessMemorySampler] = None
            benchmark_start_wallclock_ns: Optional[int] = None
            benchmark_start_monotonic_ns: Optional[int] = None
            benchmark_end_wallclock_ns: Optional[int] = None
            benchmark_end_monotonic_ns: Optional[int] = None
            latencies_ms: List[float] = []
            saved_pred_names: Set[str] = set()
            processed_images = 0

            try:
                internal_sampler = InternalSampler(repeat_internal_dir, interval_ms=int(cfg['telemetry'].get('internal_sample_interval_ms', 200)), start_wallclock_ns=scheduled_start_ns)
                tegrastats_logger = TegrastatsLogger(repeat_internal_dir, interval_ms=int(cfg['telemetry'].get('tegrastats_interval_ms', 200)), start_wallclock_ns=scheduled_start_ns)
                process_memory_sampler = ProcessMemorySampler(
                    repeat_internal_dir,
                    interval_ms=int(cfg['telemetry'].get('internal_sample_interval_ms', 200)),
                    start_wallclock_ns=scheduled_start_ns,
                    root_pid=(None if engine_measurement_mode else os.getpid()),
                    root_label=('python_benchmark_process' if not engine_measurement_mode else 'trtexec_benchmark'),
                )
                internal_sampler.start()
                tegrastats_logger.start()
                process_memory_sampler.start()

                telemetry_stage_start_ns = wallclock_ns()
                _push_status(
                    stage='telemetry_start',
                    message='Internal telemetry started; reserving external logger slot',
                    repeat_idx=repeat_idx + 1,
                    repeat_name=repeat_name,
                    stage_start_ns=telemetry_stage_start_ns,
                    internal_sampler=internal_sampler,
                    tegrastats_logger=tegrastats_logger,
                    process_memory_sampler=process_memory_sampler,
                )

                if logger_url:
                    start_payload = {
                        'run_id': external_run_id,
                        'sample_hz': float(cfg['telemetry'].get('external_sample_hz', 10.0)),
                        'start_wallclock_ns': scheduled_start_ns,
                        'fast_temp_sensor': bool(cfg['telemetry'].get('fast_temp_sensor', True)),
                        'notes': {'board_name': cfg['project'].get('board_name'), 'run_label': run_name, 'repeat': repeat_name},
                    }
                    external_info.update(
                        start_external_logger_run(
                            logger_url=logger_url,
                            start_payload=start_payload,
                            expected_board_name=str(cfg['project'].get('board_name') or '').strip() or None,
                            notes=notes,
                        )
                    )
                    external_started = True

                countdown_stage_start_ns = wallclock_ns()
                last_countdown_push_ns = 0
                while wallclock_ns() < scheduled_start_ns:
                    now_ns = wallclock_ns()
                    if (now_ns - last_countdown_push_ns) >= 500_000_000:
                        _push_status(
                            stage='countdown',
                            message='Waiting for synchronized measured start',
                            repeat_idx=repeat_idx + 1,
                            repeat_name=repeat_name,
                            stage_start_ns=countdown_stage_start_ns,
                            stage_eta_s=(scheduled_start_ns - now_ns) / 1e9,
                            min_images_target=min_images,
                            min_duration_s_target=min_duration_s,
                            internal_sampler=internal_sampler,
                            tegrastats_logger=tegrastats_logger,
                            process_memory_sampler=process_memory_sampler,
                        )
                        last_countdown_push_ns = now_ns
                    time.sleep(0.05)

                benchmark_start_wallclock_ns = wallclock_ns()
                benchmark_start_monotonic_ns = monotonic_ns()
                if engine_measurement_mode:
                    trtexec_result = benchmark_engine_with_trtexec(
                        engine_path=model_path,
                        image_paths=image_paths,
                        repeat_name=repeat_name,
                        repeat_root=repeat_root,
                        timing_csv=timing_csv,
                        imgsz=imgsz,
                        min_images=min_images,
                        min_duration_s=min_duration_s,
                        benchmark_start_wallclock_ns=benchmark_start_wallclock_ns,
                        benchmark_start_monotonic_ns=benchmark_start_monotonic_ns,
                        process_memory_sampler=process_memory_sampler,
                        save_per_image_outputs=save_per_image_outputs,
                        task=task_name,
                        progress_cb=lambda elapsed_s: _push_status(
                            stage='benchmark',
                            message='Measured TensorRT benchmark is running via trtexec',
                            repeat_idx=repeat_idx + 1,
                            repeat_name=repeat_name,
                            stage_start_ns=benchmark_start_wallclock_ns,
                            stage_eta_s=max(0.0, min_duration_s - elapsed_s) if min_duration_s > 0 else None,
                            processed_images=None,
                            min_images_target=min_images,
                            min_duration_s_target=min_duration_s,
                            current_image_name='trtexec-random-input',
                            latency_ms_last=None,
                            internal_sampler=internal_sampler,
                            tegrastats_logger=tegrastats_logger,
                            process_memory_sampler=process_memory_sampler,
                        ),
                    )
                    latencies_ms.extend(trtexec_result['latencies_ms'])
                    processed_images = int(trtexec_result['processed_images'])
                    saved_pred_names = set(trtexec_result['saved_pred_names'])
                    notes.append(
                        f'Repeat {repeat_name} measured with trtexec: '
                        f'{trtexec_result["processed_images"]} timing samples, log={trtexec_result["trtexec_log"]}'
                    )
                else:
                    last_status_push_ns = 0
                    writer = None
                    timing_handle = None
                    if save_per_image_outputs and timing_csv is not None:
                        fieldnames = _per_image_latency_fieldnames(task_name)
                        timing_handle = open(timing_csv, 'w', encoding='utf-8', newline='')
                        writer = csv.DictWriter(timing_handle, fieldnames=fieldnames)
                        writer.writeheader()
                    try:
                        pbar = tqdm(desc=f'Benchmark {repeat_name}', total=None, leave=False)
                        try:
                            while True:
                                elapsed_s = (wallclock_ns() - benchmark_start_wallclock_ns) / 1e9
                                if processed_images >= min_images and elapsed_s >= min_duration_s:
                                    break
                                manifest_index = processed_images % len(image_paths)
                                cycle_index = processed_images // len(image_paths)
                                image_path = image_paths[manifest_index]

                                ts_iso = utc_now_iso()
                                wall_start = wallclock_ns()
                                mono_start = monotonic_ns()
                                results = run_inference_once(model, image_path, imgsz, device, conf, iou, max_det)
                                mono_end = monotonic_ns()
                                wall_end = wallclock_ns()
                                latency_ms = (mono_end - mono_start) / 1e6
                                latencies_ms.append(latency_ms)
                                result0 = results[0]
                                num_det = len(result0.boxes) if hasattr(result0, 'boxes') and result0.boxes is not None else 0
                                if writer is not None:
                                    row = {
                                        'repeat': repeat_name,
                                        'image_global_index': processed_images + 1,
                                        'cycle_index': cycle_index,
                                        'manifest_index': manifest_index,
                                        'image_path': str(image_path),
                                        'image_name': image_path.name,
                                        'timestamp_utc': ts_iso,
                                        'wallclock_ns_start': wall_start,
                                        'wallclock_ns_end': wall_end,
                                        'monotonic_ns_start': mono_start,
                                        'monotonic_ns_end': mono_end,
                                        'latency_ms': latency_ms,
                                    }
                                    if task_name != 'semantic_segmentation':
                                        row['num_detections'] = num_det
                                    writer.writerow(row)
                                if pred_dir is not None and cfg['execution'].get('save_predictions_first_repeat', True) and repeat_idx == 0 and image_path.name not in saved_pred_names:
                                    write_yolo_txt(pred_dir / f'{image_path.stem}.txt', getattr(result0, 'boxes', None))
                                    saved_pred_names.add(image_path.name)
                                processed_images += 1
                                pbar.update(1)
                                if timing_handle is not None:
                                    timing_handle.flush()

                                now_ns = wallclock_ns()
                                elapsed_s = (now_ns - benchmark_start_wallclock_ns) / 1e9
                                rate = processed_images / max(elapsed_s, 1e-6)
                                images_remaining = max(0, min_images - processed_images)
                                eta_images = images_remaining / max(rate, 1e-6)
                                eta_time = max(0.0, min_duration_s - elapsed_s)
                                stage_eta_s = max(eta_images, eta_time)
                                if processed_images == 1 or (now_ns - last_status_push_ns) >= 1_000_000_000:
                                    _push_status(
                                        stage='benchmark',
                                        message='Measured inference is running',
                                        repeat_idx=repeat_idx + 1,
                                        repeat_name=repeat_name,
                                        stage_start_ns=benchmark_start_wallclock_ns,
                                        stage_eta_s=stage_eta_s,
                                        processed_images=processed_images,
                                        min_images_target=min_images,
                                        min_duration_s_target=min_duration_s,
                                        current_image_name=image_path.name,
                                        latency_ms_last=latency_ms,
                                        internal_sampler=internal_sampler,
                                        tegrastats_logger=tegrastats_logger,
                                        process_memory_sampler=process_memory_sampler,
                                    )
                                    last_status_push_ns = now_ns
                        finally:
                            pbar.close()
                    finally:
                        if timing_handle is not None:
                            timing_handle.close()

                benchmark_end_wallclock_ns = wallclock_ns()
                benchmark_end_monotonic_ns = monotonic_ns()
                _push_status(
                    stage='repeat_complete',
                    message=f'{repeat_name} complete',
                    repeat_idx=repeat_idx + 1,
                    repeat_name=repeat_name,
                    stage_start_ns=benchmark_start_wallclock_ns,
                    stage_eta_s=0.0,
                    processed_images=processed_images,
                    min_images_target=min_images,
                    min_duration_s_target=min_duration_s,
                    internal_sampler=internal_sampler,
                    tegrastats_logger=tegrastats_logger,
                )

                marker_obj = {
                    'repeat': repeat_name,
                    'scheduled_start_wallclock_ns': scheduled_start_ns,
                    'benchmark_start_wallclock_ns': benchmark_start_wallclock_ns,
                    'benchmark_end_wallclock_ns': benchmark_end_wallclock_ns,
                    'benchmark_start_monotonic_ns': benchmark_start_monotonic_ns,
                    'benchmark_end_monotonic_ns': benchmark_end_monotonic_ns,
                    'images_benchmarked': processed_images,
                    'timestamp_utc': utc_now_iso(),
                    'warmup_summary': warmup_summary,
                    'min_images_target': min_images,
                    'min_duration_s_target': min_duration_s,
                }
                dump_json(marker_obj, markers_json)
            except Exception as exc:
                _push_status(
                    stage='error',
                    state='error',
                    message=f'{type(exc).__name__}: {exc}',
                    repeat_idx=repeat_idx + 1,
                    repeat_name=repeat_name,
                    stage_start_ns=benchmark_start_wallclock_ns or warmup_start_wallclock_ns,
                    processed_images=processed_images,
                    min_images_target=min_images,
                    min_duration_s_target=min_duration_s,
                    current_image_name=None,
                    internal_sampler=internal_sampler,
                    tegrastats_logger=tegrastats_logger,
                    process_memory_sampler=process_memory_sampler,
                )
                raise
            finally:
                if internal_sampler is not None:
                    internal_sampler.stop()
                    internal_sampler.join(timeout=10)
                if tegrastats_logger is not None:
                    tegrastats_logger.stop()
                    tegrastats_logger.join(timeout=10)
                if process_memory_sampler is not None:
                    process_memory_sampler.stop()
                    process_memory_sampler.join(timeout=10)
                if external_started:
                    try:
                        external_info.update(
                            stop_and_fetch_external_run(
                                logger_url=logger_url,
                                run_id=external_run_id,
                                repeat_external_dir=repeat_external_dir,
                                notes=notes,
                            )
                        )
                    except Exception as exc:
                        external_info['stop_or_download_error'] = str(exc)
                        notes.append(f'External logger stop/download issue on {repeat_name}: {exc}')

            thermal_path = repeat_internal_dir / 'internal_thermal_zones.csv'
            hwmon_path = repeat_internal_dir / 'internal_hwmon.csv'
            tstats_path = repeat_internal_dir / 'tegrastats_parsed.csv'
            process_mem_path = repeat_internal_dir / 'process_memory.csv'
            external_csv_candidates = list(repeat_external_dir.rglob('external_measurements.csv'))

            thermal_summary = _parse_thermal_csv(thermal_path, benchmark_start_wallclock_ns, benchmark_end_wallclock_ns) if thermal_path.exists() else {}
            internal_power_summary = _parse_hwmon_csv(hwmon_path, benchmark_start_wallclock_ns, benchmark_end_wallclock_ns) if hwmon_path.exists() else {}
            tegrastats_summary = _parse_tegrastats_csv(tstats_path, benchmark_start_wallclock_ns, benchmark_end_wallclock_ns) if tstats_path.exists() else {}
            process_memory_summary = _parse_process_memory_csv(process_mem_path, benchmark_start_wallclock_ns, benchmark_end_wallclock_ns) if process_mem_path.exists() else {}
            external_summary = _parse_external_csv(external_csv_candidates[0], benchmark_start_wallclock_ns, benchmark_end_wallclock_ns) if external_csv_candidates else {}

            latency_summary = _aggregate_latency(latencies_ms)
            measured_duration_s = (benchmark_end_wallclock_ns - benchmark_start_wallclock_ns) / 1e9 if benchmark_end_wallclock_ns and benchmark_start_wallclock_ns else None
            throughput_fps = (processed_images / measured_duration_s) if measured_duration_s and measured_duration_s > 0 else None

            repeat_summary = {
                'project': cfg['project'],
                'repeat': repeat_name,
                'timestamp_utc': utc_now_iso(),
                'precision_label': cfg['project'].get('precision_label'),
                'model_path': str(model_path),
                'model_sha256': measured_model_sha256,
                'dataset_root': str(dataset_root),
                'manifest': str(manifest),
                'images_benchmarked': processed_images,
                'unique_prediction_files_saved': len(saved_pred_names),
                'warmup': warmup_summary,
                'latency': latency_summary,
                'throughput_fps': throughput_fps,
                'external_power': external_summary,
                'internal_power': internal_power_summary,
                'internal_thermal': thermal_summary,
                'tegrastats': tegrastats_summary,
                'process_memory': process_memory_summary,
                'external_logger_metadata': external_info,
                'merged_trace_csv': str(repeat_root / 'merged_trace.csv') if (repeat_root / 'merged_trace.csv').exists() else None,
                'per_image_metrics_csv': str(repeat_root / 'per_image_metrics.csv') if (repeat_root / 'per_image_metrics.csv').exists() else None,
                'agreement_summary_path': str(repeat_root / 'agreement_summary.json') if (repeat_root / 'agreement_summary.json').exists() else None,
                'external_artifact_path': external_info.get('archive_path'),
                'external_artifact_paths': [external_info.get('archive_path')] if external_info.get('archive_path') else None,
                'files': {
                    'timing_csv': str(timing_csv) if timing_csv is not None and timing_csv.exists() else None,
                    'hwmon_csv': str(hwmon_path) if hwmon_path.exists() else None,
                    'thermal_csv': str(thermal_path) if thermal_path.exists() else None,
                    'tegrastats_csv': str(tstats_path) if tstats_path.exists() else None,
                    'process_memory_csv': str(process_mem_path) if process_mem_path.exists() else None,
                    'external_csv': str(external_csv_candidates[0]) if external_csv_candidates else None,
                    'external_archive_path': external_info.get('archive_path'),
                    'external_archive_paths': [external_info.get('archive_path')] if external_info.get('archive_path') else None,
                    'merged_trace_csv': str(repeat_root / 'merged_trace.csv') if (repeat_root / 'merged_trace.csv').exists() else None,
                    'per_image_metrics_csv': str(repeat_root / 'per_image_metrics.csv') if (repeat_root / 'per_image_metrics.csv').exists() else None,
                    'agreement_summary_json': str(repeat_root / 'agreement_summary.json') if (repeat_root / 'agreement_summary.json').exists() else None,
                },
                'run_markers': marker_obj,
                'power_mode': cfg['power_mode'],
                'notes': notes,
            }
            if external_summary and processed_images > 0 and external_summary.get('energy_joules') is not None:
                repeat_summary['external_power']['joules_per_image'] = external_summary['energy_joules'] / processed_images
            repeat_summary['external_summary'] = dict(repeat_summary['external_power']) if repeat_summary.get('external_power') else None
            dump_json(repeat_summary, repeat_root / 'repeat_summary.json')

            aggregate_rows.append(
                {
                    'repeat': repeat_name,
                    'images_benchmarked': processed_images,
                    'benchmark_duration_s': measured_duration_s,
                    'warmup_elapsed_s': warmup_summary.get('warmup_elapsed_s'),
                    'latency_median_ms': latency_summary['median_ms'],
                    'latency_p95_ms': latency_summary['p95_ms'],
                    'latency_mean_ms': latency_summary['mean_ms'],
                    'throughput_fps': throughput_fps,
                    'avg_power_external_w': external_summary.get('avg_power_w'),
                    'peak_power_external_w': external_summary.get('peak_power_w'),
                    'energy_joules_external': external_summary.get('energy_joules'),
                    'energy_wh_external': external_summary.get('energy_wh'),
                    'joules_per_image_external': repeat_summary['external_power'].get('joules_per_image') if 'external_power' in repeat_summary else None,
                    'external_temp_mean_c': external_summary.get('external_temp_mean_c'),
                    'external_temp_max_c': external_summary.get('external_temp_max_c'),
                    'internal_total_power_mean_w': internal_power_summary.get('mean_total_power_w'),
                    'internal_total_power_peak_w': internal_power_summary.get('peak_total_power_w'),
                    'internal_temp_mean_c': thermal_summary.get('overall_internal_mean_c'),
                    'internal_temp_max_c': thermal_summary.get('overall_internal_max_c'),
                    'peak_ram_mb': tegrastats_summary.get('ram_used_mb_max'),
                    'mean_ram_mb': tegrastats_summary.get('ram_used_mb_mean'),
                    'swap_used_mb_peak': tegrastats_summary.get('swap_used_mb_max'),
                    'swap_used_mb_mean': tegrastats_summary.get('swap_used_mb_mean'),
                    'lfb_blocks_min': tegrastats_summary.get('lfb_blocks_min'),
                    'lfb_size_mb_min': tegrastats_summary.get('lfb_size_mb_min'),
                    'lfb_total_mb_min': tegrastats_summary.get('lfb_total_mb_min'),
                    'lfb_total_mb_mean': tegrastats_summary.get('lfb_total_mb_mean'),
                    'peak_process_rss_mb': process_memory_summary.get('rss_mb_max'),
                    'mean_process_rss_mb': process_memory_summary.get('rss_mb_mean'),
                    'peak_process_uss_mb': process_memory_summary.get('uss_mb_max'),
                    'mean_process_uss_mb': process_memory_summary.get('uss_mb_mean'),
                    'peak_process_pss_mb': process_memory_summary.get('pss_mb_max'),
                    'mean_process_pss_mb': process_memory_summary.get('pss_mb_mean'),
                    'peak_process_swap_mb': process_memory_summary.get('swap_mb_max'),
                    'process_tree_pid_count_max': process_memory_summary.get('pid_count_max'),
                }
            )

        aggregate_df = pd.DataFrame(aggregate_rows)
        aggregate_csv = run_root / 'aggregate_repeat_summary.csv'
        aggregate_df.to_csv(aggregate_csv, index=False)
        final_external_summary = _build_final_external_summary(aggregate_df)

        ci_summary = {}
        for col in [
            'latency_median_ms',
            'latency_p95_ms',
            'throughput_fps',
            'avg_power_external_w',
            'joules_per_image_external',
            'peak_ram_mb',
            'swap_used_mb_peak',
            'lfb_total_mb_min',
            'peak_process_rss_mb',
            'peak_process_uss_mb',
            'peak_process_pss_mb',
            'peak_process_swap_mb',
        ]:
            if col in aggregate_df.columns:
                ci = bootstrap_ci([float(v) for v in aggregate_df[col].dropna().tolist()])
                if ci is not None:
                    ci_summary[col] = ci

        accuracy_split_name = str(cfg['dataset'].get('split_name', 'test'))
        accuracy_dir = run_root / 'accuracy'
        accuracy_output_dir = accuracy_dir / f'val_{accuracy_split_name}'

        final_summary = {
            'timestamp_utc': utc_now_iso(),
            'project': cfg['project'],
            'model': {
                'measured_model_path': str(model_path),
                'measured_model_sha256': measured_model_sha256,
                'accuracy_model_path': str(resolved_accuracy_model_path) if resolved_accuracy_model_path is not None else None,
                'accuracy_model_sha256': accuracy_model_sha256,
                'accuracy_backend': accuracy_backend,
                'accuracy_backend_detail': accuracy_backend_detail,
                'accuracy_precision_label': accuracy_precision_label,
                'accuracy_precision_specific': accuracy_precision_specific,
                'precision_label': cfg['project'].get('precision_label'),
                'profile_input': {'imgsz': imgsz, 'batch': batch},
                'complexity': model_complexity,
            },
            'dataset': {
                'root': str(dataset_root),
                'data_yaml': str(data_yaml) if data_yaml is not None else None,
                'manifest': str(resolved_accuracy_manifest_path) if resolved_accuracy_manifest_path is not None else str(manifest),
                'ordered_manifest': str(manifest),
                'benchmark_manifest': str(manifest),
                'accuracy_manifest': str(resolved_accuracy_manifest_path) if resolved_accuracy_manifest_path is not None else None,
                'image_count': len(image_paths),
                'accuracy_image_count': accuracy_metrics.get('processed_samples') if accuracy_metrics else None,
            },
            'power_mode': cfg['power_mode'],
            'accuracy_metrics': accuracy_metrics,
            'external_summary': final_external_summary,
            'power': _build_final_power_summary(aggregate_df, final_external_summary),
            'performance': _build_final_performance_summary(aggregate_df),
            'benchmark_repeats': aggregate_rows,
            'aggregate_mean': aggregate_df.mean(numeric_only=True).to_dict(),
            'aggregate_std': aggregate_df.std(numeric_only=True, ddof=1).to_dict() if len(aggregate_df) > 1 else {},
            'bootstrap_95ci': ci_summary,
            'files': {
                'run_root': str(run_root),
                'aggregate_csv': str(aggregate_csv),
                'config_snapshot': str(run_root / 'config_snapshot.yaml'),
                'manifest_snapshot': str(run_root / 'manifest_snapshot.txt'),
                'data_yaml_snapshot': str(run_root / 'data_yaml_snapshot.yaml') if data_yaml is not None and data_yaml.exists() else None,
                'system_snapshot_dir': str(run_root / 'system'),
                'noise_guard_summary': str(noise_guard_summary_path),
                'benchmark_runs_root': str(benchmark_runs_root),
                'complexity_json': str(complexity_json_path) if complexity_json_path is not None else None,
                'complexity_profile_onnx': str(complexity_onnx_path) if complexity_onnx_path is not None else None,
                'accuracy_metrics_json': str(accuracy_dir / 'val_metrics.json') if (accuracy_dir / 'val_metrics.json').exists() else None,
                'accuracy_report_json': str(accuracy_output_dir / 'report.json') if (accuracy_output_dir / 'report.json').exists() else None,
                'accuracy_raw_metrics_json': str(accuracy_output_dir / 'metrics.json') if (accuracy_output_dir / 'metrics.json').exists() else None,
                'accuracy_validation_log': str(accuracy_dir / f'validation_{accuracy_split_name}.log') if (accuracy_dir / f'validation_{accuracy_split_name}.log').exists() else None,
            },
            'noise_guard': noise_guard.summary() if noise_guard is not None else None,
        }
        _validate_accuracy_provenance(
            run_root=run_root,
            measured_model_path=model_path,
            measured_model_sha256=measured_model_sha256,
            benchmark_manifest_path=manifest,
            resolved_accuracy_model_path=resolved_accuracy_model_path,
            accuracy_model_sha256=accuracy_model_sha256,
            accuracy_backend=accuracy_backend,
            accuracy_precision_label=accuracy_precision_label,
            accuracy_metrics=accuracy_metrics,
            final_summary=final_summary,
            split_name=accuracy_split_name,
        )
        dump_json(final_summary, run_root / 'FINAL_SUMMARY.json')
        (run_root / 'README_NEXT_STEPS.txt').write_text(
            '1) Keep FINAL_SUMMARY.json and aggregate_repeat_summary.csv for reporting.\n'
            '2) Keep benchmark_runs/** raw logs for later revision and agreement analysis.\n'
            '3) Use the same fixed test split on all Jetson runs.\n'
            '4) Use analysis/merge_and_summarize.py to build aligned traces and agreement statistics.\n',
            encoding='utf-8',
        )
        _push_status(stage='finished', state='finished', message='Benchmark finished successfully', extra={'final_summary_json': str(run_root / 'FINAL_SUMMARY.json')})
    except Exception as exc:
        run_error = exc
        raise
    finally:
        if noise_guard is not None:
            try:
                dump_json(noise_guard.restore(), noise_guard_summary_path)
                _push_status(
                    stage='restored',
                    state='restored' if run_error is None else 'error',
                    message='System state restored after benchmark' if run_error is None else f'System state restored after error: {type(run_error).__name__}',
                    extra={'noise_guard_summary_json': str(noise_guard_summary_path)},
                )
            except Exception as exc:
                notes.append(f'Noise guard restore failed: {exc}')
                try:
                    dump_json(noise_guard.summary(), noise_guard_summary_path)
                except Exception:
                    pass
        # Keep the last OLED message visible; just release the bus without blanking the display.
        try:
            oled.close()
        except Exception:
            pass


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark(args)
