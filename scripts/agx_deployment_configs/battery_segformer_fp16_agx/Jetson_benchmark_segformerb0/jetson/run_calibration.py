from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics import YOLO

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.internal_telemetry import InternalSampler, TegrastatsLogger
    from jetson.local_paths import (
        default_output_root,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
        resolve_model_path as resolve_local_model_path,
        resolve_output_root,
    )
    from jetson.noise_guard import NoiseGuard, NoiseGuardConfig
    from jetson.utils import default_test_manifest, dump_json, ensure_dir, list_images_from_manifest, monotonic_ns, sha256_file, utc_now_iso, wallclock_ns
else:
    from .internal_telemetry import InternalSampler, TegrastatsLogger
    from .local_paths import (
        default_output_root,
        resolve_dataset_root,
        resolve_inference_bundle_root,
        resolve_manifest_path,
        resolve_model_path as resolve_local_model_path,
        resolve_output_root,
    )
    from .noise_guard import NoiseGuard, NoiseGuardConfig
    from .utils import default_test_manifest, dump_json, ensure_dir, list_images_from_manifest, monotonic_ns, sha256_file, utc_now_iso, wallclock_ns


def post_json(url: str, payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode('utf-8'))


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        dest.write_bytes(resp.read())


def get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode('utf-8'))


def _current_internal_live_snapshot(internal_sampler: Optional[InternalSampler], tegrastats_logger: Optional[TegrastatsLogger]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {'hwmon': {}, 'thermal': {}, 'tegrastats': {}}
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
    return snap


def cpu_worker(stop_event: mp.Event) -> None:
    x = 0.0
    while not stop_event.is_set():
        x = (x * 1.0000001 + 3.14159) % 12345.678


def run_gpu_loop(
    model: YOLO,
    image_paths: List[Path],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    end_wallclock_ns: int,
    timing_csv: Path,
    progress_cb=None,
) -> Dict[str, Any]:
    latencies_ms: List[float] = []
    count = 0
    last_cb_ns = 0
    with open(timing_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_index', 'image_path', 'timestamp_utc', 'wallclock_ns_start', 'wallclock_ns_end', 'monotonic_ns_start', 'monotonic_ns_end', 'latency_ms'])
        writer.writeheader()
        while wallclock_ns() < end_wallclock_ns:
            image_path = image_paths[count % len(image_paths)]
            ts_iso = utc_now_iso()
            ws = wallclock_ns()
            ms = monotonic_ns()
            _ = model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=conf, iou=iou, max_det=max_det, verbose=False, stream=False, save=False)
            me = monotonic_ns()
            we = wallclock_ns()
            latency_ms = (me - ms) / 1e6
            latencies_ms.append(latency_ms)
            writer.writerow({'image_index': count + 1, 'image_path': str(image_path), 'timestamp_utc': ts_iso, 'wallclock_ns_start': ws, 'wallclock_ns_end': we, 'monotonic_ns_start': ms, 'monotonic_ns_end': me, 'latency_ms': latency_ms})
            f.flush()
            count += 1
            now_ns = wallclock_ns()
            if progress_cb is not None and (count == 1 or (now_ns - last_cb_ns) >= 1_000_000_000):
                remaining_s = max(0.0, (end_wallclock_ns - now_ns) / 1e9)
                progress_cb(count, latency_ms, remaining_s, image_path.name)
                last_cb_ns = now_ns
    return {
        'iterations': count,
        'latency_mean_ms': float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
        'latency_median_ms': float(sorted(latencies_ms)[len(latencies_ms)//2]) if latencies_ms else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Run proposal calibration scenarios: idle, cpu, gpu, mixed.')
    ap.add_argument('--scenario', required=True, choices=['idle', 'cpu', 'gpu', 'mixed'])
    ap.add_argument('--duration-s', type=float, default=180.0)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--board-name', default='Jetson AGX Orin 64GB')
    ap.add_argument('--output-root', default=str(default_output_root('jetson_calibration_runs')))
    ap.add_argument('--logger-url', required=True)
    ap.add_argument('--external-sample-hz', type=float, default=10.0)
    ap.add_argument('--tegrastats-interval-ms', type=int, default=200)
    ap.add_argument('--internal-interval-ms', type=int, default=200)
    ap.add_argument('--start-delay-s', type=float, default=5.0)
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--dataset-root', type=str, default=None)
    ap.add_argument('--manifest', type=str, default=None)
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--cpu-workers', type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument('--noise-guard', dest='noise_guard', action='store_true', help='Apply quiet-mode actions during calibration (default: enabled)')
    ap.add_argument('--no-noise-guard', dest='noise_guard', action='store_false')
    ap.add_argument('--noise-guard-profile', choices=['safe', 'aggressive'], default='safe')
    ap.add_argument('--noise-guard-preserve-url', default=None)
    ap.add_argument('--noise-guard-display-off', dest='noise_guard_display_off', action='store_true')
    ap.add_argument('--no-noise-guard-display-off', dest='noise_guard_display_off', action='store_false')
    ap.add_argument('--noise-guard-stop-services', dest='noise_guard_stop_services', action='store_true')
    ap.add_argument('--no-noise-guard-stop-services', dest='noise_guard_stop_services', action='store_false')
    ap.add_argument('--noise-guard-disable-wifi', dest='noise_guard_disable_wifi', action='store_true')
    ap.add_argument('--no-noise-guard-disable-wifi', dest='noise_guard_disable_wifi', action='store_false')
    ap.add_argument('--noise-guard-disable-bluetooth', dest='noise_guard_disable_bluetooth', action='store_true')
    ap.add_argument('--no-noise-guard-disable-bluetooth', dest='noise_guard_disable_bluetooth', action='store_false')
    ap.set_defaults(noise_guard=True, noise_guard_display_off=True, noise_guard_stop_services=True, noise_guard_disable_wifi=True, noise_guard_disable_bluetooth=True)
    args = ap.parse_args()

    args.output_root = str(resolve_output_root(args.output_root, 'jetson_calibration_runs'))
    bundle_root = resolve_inference_bundle_root(args.dataset_root or args.model)
    if args.scenario in {'gpu', 'mixed'}:
        resolved_dataset_root = resolve_dataset_root(args.dataset_root, bundle_root=bundle_root)
        resolved_manifest = resolve_manifest_path(resolved_dataset_root, args.manifest, bundle_root=bundle_root)
        resolved_model_path = resolve_local_model_path(args.model, bundle_root=bundle_root)
        args.dataset_root = str(resolved_dataset_root)
        args.manifest = str(resolved_manifest) if resolved_manifest is not None else None
        args.model = str(resolved_model_path)

    run_root = ensure_dir(Path(args.output_root).expanduser() / args.run_name / args.scenario)
    ext_root = ensure_dir(run_root / 'external')
    int_root = ensure_dir(run_root / 'internal')
    scheduled_start_ns = wallclock_ns() + int(args.start_delay_s * 1e9)
    external_run_id = f'{args.run_name}_{args.scenario}'
    dashboard_status_url = args.logger_url.rstrip('/') + '/api/jetson_status'
    session_start_ns = wallclock_ns()
    noise_guard: Optional[NoiseGuard] = None
    noise_guard_summary_path = run_root / 'noise_guard_summary.json'

    def push_status(
        stage: str,
        message: str = '',
        state: str = 'running',
        stage_start_ns: Optional[int] = None,
        iterations: Optional[int] = None,
        eta_s: Optional[float] = None,
        current_image_name: Optional[str] = None,
        latency_ms_last: Optional[float] = None,
        internal_sampler: Optional[InternalSampler] = None,
        tegrastats_logger: Optional[TegrastatsLogger] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            'run_id': args.run_name,
            'board_name': args.board_name,
            'stage': stage,
            'state': state,
            'message': message,
            'timestamp_utc': utc_now_iso(),
            'wallclock_ns': wallclock_ns(),
            'overall_elapsed_s': (wallclock_ns() - session_start_ns) / 1e9,
            'stage_elapsed_s': ((wallclock_ns() - stage_start_ns) / 1e9) if stage_start_ns is not None else None,
            'stage_eta_s': eta_s,
            'repeat_index': 1,
            'total_repeats': 1,
            'repeat_name': f'calibration_{args.scenario}',
            'processed_images': iterations,
            'min_images_target': None,
            'min_duration_s_target': args.duration_s,
            'current_image_name': current_image_name,
            'latency_ms_last': latency_ms_last,
            'power_mode': {'power_mode_label': f'calibration_{args.scenario}'},
            'internal_live': _current_internal_live_snapshot(internal_sampler, tegrastats_logger),
            'noise_guard': noise_guard.dashboard_payload() if noise_guard is not None else None,
        }
        if extra:
            payload.update(extra)
        try:
            post_json(dashboard_status_url, payload, timeout=4)
        except Exception:
            pass

    if args.noise_guard:
        noise_guard = NoiseGuard(
            NoiseGuardConfig(
                enabled=True,
                profile=args.noise_guard_profile,
                preserve_url=args.noise_guard_preserve_url or args.logger_url,
                display_off=args.noise_guard_display_off,
                stop_services=args.noise_guard_stop_services,
                disable_wifi=args.noise_guard_disable_wifi,
                disable_bluetooth=args.noise_guard_disable_bluetooth,
            )
        )
        dump_json(noise_guard.enter(), noise_guard_summary_path)

    start_payload = {
        'run_id': external_run_id,
        'sample_hz': args.external_sample_hz,
        'start_wallclock_ns': scheduled_start_ns,
        'fast_temp_sensor': True,
        'notes': {'board_name': args.board_name, 'scenario': args.scenario},
    }

    ext_info: Dict[str, Any] = {}
    stop_info: Dict[str, Any] = {}
    internal_sampler: Optional[InternalSampler] = None
    tegrastats_logger: Optional[TegrastatsLogger] = None
    bench_start_wall_ns: Optional[int] = None
    bench_end_wall_ns: Optional[int] = None
    archive_path: Optional[Path] = None
    external_started = False
    external_stopped = False
    extra_summary: Dict[str, Any] = {}
    cpu_stop = mp.Event()
    cpu_procs: List[mp.Process] = []
    last_status_ns = 0
    run_error: Optional[BaseException] = None

    try:
        try:
            get_json(args.logger_url.rstrip('/') + '/health', timeout=10)
        except Exception:
            pass

        ext_info = post_json(args.logger_url.rstrip('/') + '/start', start_payload, timeout=20)
        external_started = True

        internal_sampler = InternalSampler(int_root, interval_ms=args.internal_interval_ms, start_wallclock_ns=scheduled_start_ns)
        tegrastats_logger = TegrastatsLogger(int_root, interval_ms=args.tegrastats_interval_ms, start_wallclock_ns=scheduled_start_ns)
        internal_sampler.start()
        tegrastats_logger.start()

        countdown_start_ns = wallclock_ns()
        last_countdown_push_ns = 0
        while wallclock_ns() < scheduled_start_ns:
            now_ns = wallclock_ns()
            if (now_ns - last_countdown_push_ns) >= 500_000_000:
                push_status(
                    stage='countdown',
                    message=f'Calibration {args.scenario}: waiting for synchronized start',
                    stage_start_ns=countdown_start_ns,
                    eta_s=(scheduled_start_ns - now_ns) / 1e9,
                    internal_sampler=internal_sampler,
                    tegrastats_logger=tegrastats_logger,
                )
                last_countdown_push_ns = now_ns
            time.sleep(0.05)

        bench_start_wall_ns = wallclock_ns()
        end_wall_ns = bench_start_wall_ns + int(args.duration_s * 1e9)

        if args.scenario in {'cpu', 'mixed'}:
            for _ in range(args.cpu_workers):
                p = mp.Process(target=cpu_worker, args=(cpu_stop,), daemon=True)
                p.start()
                cpu_procs.append(p)

        if args.scenario == 'idle':
            while wallclock_ns() < end_wall_ns:
                now_ns = wallclock_ns()
                if (now_ns - last_status_ns) >= 1_000_000_000:
                    push_status(stage=f'calibration_{args.scenario}', message='Idle calibration running', stage_start_ns=bench_start_wall_ns, eta_s=(end_wall_ns - now_ns) / 1e9, internal_sampler=internal_sampler, tegrastats_logger=tegrastats_logger)
                    last_status_ns = now_ns
                time.sleep(0.1)
        else:
            if args.scenario in {'gpu', 'mixed'}:
                dataset_root = Path(args.dataset_root).expanduser().resolve()
                manifest = Path(args.manifest).expanduser().resolve() if args.manifest else default_test_manifest(dataset_root)
                image_paths = list_images_from_manifest(dataset_root, manifest)
                model_path = Path(args.model).expanduser().resolve()
                model = YOLO(str(model_path))
                extra_summary = run_gpu_loop(
                    model,
                    image_paths,
                    args.imgsz,
                    args.device,
                    args.conf,
                    args.iou,
                    args.max_det,
                    end_wall_ns,
                    run_root / 'gpu_iterations.csv',
                    progress_cb=lambda iterations, latency_ms, remaining_s, image_name: push_status(
                        stage=f'calibration_{args.scenario}',
                        message='GPU calibration running',
                        stage_start_ns=bench_start_wall_ns,
                        iterations=iterations,
                        eta_s=remaining_s,
                        current_image_name=image_name,
                        latency_ms_last=latency_ms,
                        internal_sampler=internal_sampler,
                        tegrastats_logger=tegrastats_logger,
                    ),
                )
                extra_summary['model_path'] = str(model_path)
                extra_summary['model_sha256'] = sha256_file(model_path)
            else:
                while wallclock_ns() < end_wall_ns:
                    now_ns = wallclock_ns()
                    if (now_ns - last_status_ns) >= 1_000_000_000:
                        push_status(stage=f'calibration_{args.scenario}', message='CPU calibration running', stage_start_ns=bench_start_wall_ns, eta_s=(end_wall_ns - now_ns) / 1e9, internal_sampler=internal_sampler, tegrastats_logger=tegrastats_logger)
                        last_status_ns = now_ns
                    time.sleep(0.1)

        bench_end_wall_ns = wallclock_ns()
        if internal_sampler is not None:
            internal_sampler.stop()
            internal_sampler.join(timeout=10)
            internal_sampler = None
        if tegrastats_logger is not None:
            tegrastats_logger.stop()
            tegrastats_logger.join(timeout=10)
            tegrastats_logger = None

        stop_info = post_json(args.logger_url.rstrip('/') + '/stop', {'run_id': external_run_id}, timeout=60)
        external_stopped = True
        archive_path = ext_root / f'{external_run_id}.tar.gz'
        download_file(args.logger_url.rstrip('/') + f'/download/{external_run_id}', archive_path, timeout=600)
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(ext_root)

        summary = {
            'timestamp_utc': utc_now_iso(),
            'scenario': args.scenario,
            'board_name': args.board_name,
            'scheduled_start_wallclock_ns': scheduled_start_ns,
            'benchmark_start_wallclock_ns': bench_start_wall_ns,
            'benchmark_end_wallclock_ns': bench_end_wall_ns,
            'duration_s': ((bench_end_wall_ns - bench_start_wall_ns) / 1e9) if (bench_end_wall_ns is not None and bench_start_wall_ns is not None) else None,
            'external_logger': {**ext_info, **stop_info},
            'cpu_workers': args.cpu_workers if args.scenario in {'cpu', 'mixed'} else 0,
            'extra_summary': extra_summary,
            'noise_guard': noise_guard.summary() if noise_guard is not None else None,
        }
        dump_json(summary, run_root / 'calibration_summary.json')
        push_status(stage='finished', state='finished', message=f'Calibration {args.scenario} finished', stage_start_ns=bench_start_wall_ns, eta_s=0.0, extra={'calibration_summary_json': str(run_root / 'calibration_summary.json')})
    except Exception as exc:
        run_error = exc
        push_status(stage='error', state='error', message=f'{type(exc).__name__}: {exc}', stage_start_ns=bench_start_wall_ns, internal_sampler=internal_sampler, tegrastats_logger=tegrastats_logger)
        raise
    finally:
        cpu_stop.set()
        for p in cpu_procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

        if internal_sampler is not None:
            try:
                internal_sampler.stop()
                internal_sampler.join(timeout=10)
            except Exception:
                pass
        if tegrastats_logger is not None:
            try:
                tegrastats_logger.stop()
                tegrastats_logger.join(timeout=10)
            except Exception:
                pass
        if external_started and not external_stopped:
            try:
                stop_info = post_json(args.logger_url.rstrip('/') + '/stop', {'run_id': external_run_id}, timeout=60)
            except Exception:
                pass
        if noise_guard is not None:
            try:
                dump_json(noise_guard.restore(), noise_guard_summary_path)
                push_status(
                    stage='restored',
                    state='restored' if run_error is None else 'error',
                    message=f'Calibration {args.scenario} restored system state',
                    stage_start_ns=bench_start_wall_ns,
                    eta_s=0.0,
                    extra={'noise_guard_summary_json': str(noise_guard_summary_path)},
                )
            except Exception:
                pass


if __name__ == '__main__':
    main()
