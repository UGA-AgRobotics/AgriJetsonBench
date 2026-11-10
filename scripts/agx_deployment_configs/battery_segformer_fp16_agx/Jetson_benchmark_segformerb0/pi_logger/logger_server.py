from __future__ import annotations

import argparse
import csv
import json
import statistics
import tarfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.local_paths import default_output_root
    from pi_logger.sensors import INA260_ADDR_DEFAULT, TMP117_ADDR_DEFAULT, SensorHub, scan_all
else:
    from jetson.local_paths import default_output_root
    from .sensors import INA260_ADDR_DEFAULT, TMP117_ADDR_DEFAULT, SensorHub, scan_all

WEB_ROOT = Path(__file__).resolve().parent / 'web'
app = Flask(__name__, static_folder=str(WEB_ROOT / 'static'), template_folder=str(WEB_ROOT / 'templates'))


def utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def wallclock_ns() -> int:
    return time.time_ns()


def monotonic_ns() -> int:
    return time.monotonic_ns()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def wait_until_wallclock_ns(target_ns: int) -> None:
    while True:
        now = wallclock_ns()
        diff = target_ns - now
        if diff <= 0:
            return
        if diff > 2_000_000:
            time.sleep((diff - 1_000_000) / 1e9)
        else:
            pass


def _deepcopy_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


@dataclass
class RunState:
    run_id: str
    run_dir: Path
    sample_hz: float
    start_wallclock_ns: int
    fast_temp_sensor: bool
    notes: Optional[Dict[str, object]] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    csv_path: Optional[Path] = None
    summary_json: Optional[Path] = None
    tar_path: Optional[Path] = None
    samples_written: int = 0
    latest_sample: Optional[Dict[str, Any]] = None


OUTPUT_ROOT: Path = default_output_root('jetson_external_logs')
CURRENT_RUN: Optional[RunState] = None
LAST_COMPLETED_RUN: Optional[Dict[str, Any]] = None
HUB: Optional[SensorHub] = None
RUN_LOCK = threading.Lock()
HUB_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

LIVE_SAMPLES: Deque[Dict[str, Any]] = deque(maxlen=2400)
INTERNAL_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=1200)
ALERT_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=100)
JETSON_STATUS: Dict[str, Any] = {}
LAST_EXTERNAL_READING: Dict[str, Any] = {}
BASELINE_VOLTAGE: Deque[float] = deque(maxlen=30)
BASELINE_CURRENT: Deque[float] = deque(maxlen=30)
BASELINE_POWER: Deque[float] = deque(maxlen=30)
LAST_ALERT_TIME_NS: Dict[str, int] = {}


def summarize_csv(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {'samples': 0}
    if 'mid_wallclock_ns' in df.columns:
        t = pd.to_numeric(df['mid_wallclock_ns'], errors='coerce').dropna().to_numpy(dtype='float64') / 1e9
    else:
        t = pd.to_numeric(df['read_end_ns'], errors='coerce').dropna().to_numpy(dtype='float64') / 1e9
    p_w = pd.to_numeric(df['power_w'], errors='coerce').dropna().to_numpy(dtype='float64')
    energy_j = float(np.trapz(p_w, t[:len(p_w)])) if len(t) >= 2 and len(p_w) >= 2 else 0.0
    temp_series = pd.to_numeric(df.get('external_temp_c'), errors='coerce').dropna() if 'external_temp_c' in df.columns else pd.Series(dtype=float)
    sensor_types = sorted({str(x) for x in df.get('temperature_sensor_type', pd.Series(dtype=str)).dropna().astype(str).tolist()}) if 'temperature_sensor_type' in df.columns else []
    return {
        'samples': int(len(df)),
        'avg_power_w': float(pd.to_numeric(df['power_w'], errors='coerce').mean()),
        'peak_power_w': float(pd.to_numeric(df['power_w'], errors='coerce').max()),
        'mean_voltage_v': float(pd.to_numeric(df['bus_voltage_v'], errors='coerce').mean()),
        'mean_current_a': float(pd.to_numeric(df['current_a'], errors='coerce').mean()),
        'energy_joules': energy_j,
        'energy_wh': energy_j / 3600.0,
        'external_temp_mean_c': float(temp_series.mean()) if len(temp_series) else None,
        'external_temp_max_c': float(temp_series.max()) if len(temp_series) else None,
        'temperature_sensor_types_seen': sensor_types,
    }


def _emit_alert(event_type: str, sample: Dict[str, Any], baseline: float, value: float, message: str, cooldown_s: float = 10.0) -> None:
    now_ns = int(sample.get('mid_wallclock_ns') or wallclock_ns())
    last_ns = LAST_ALERT_TIME_NS.get(event_type)
    if last_ns is not None and (now_ns - last_ns) < int(cooldown_s * 1e9):
        return
    LAST_ALERT_TIME_NS[event_type] = now_ns
    alert = {
        'event_type': event_type,
        'timestamp_utc': sample.get('timestamp_utc', utc_now_iso()),
        'sample_idx': sample.get('sample_idx'),
        'run_id': CURRENT_RUN.run_id if CURRENT_RUN is not None else JETSON_STATUS.get('run_id'),
        'baseline': baseline,
        'value': value,
        'delta': value - baseline,
        'message': message,
    }
    ALERT_HISTORY.append(alert)


def _update_live_external_state(sample: Dict[str, Any]) -> None:
    voltage = float(sample['bus_voltage_v'])
    current = float(sample['current_a'])
    power = float(sample['power_w'])
    BASELINE_VOLTAGE.append(voltage)
    BASELINE_CURRENT.append(current)
    BASELINE_POWER.append(power)

    LIVE_SAMPLES.append(sample)
    LAST_EXTERNAL_READING.clear()
    LAST_EXTERNAL_READING.update(sample)

    if len(BASELINE_VOLTAGE) >= 12:
        prev_voltages = list(BASELINE_VOLTAGE)[:-1] or list(BASELINE_VOLTAGE)
        prev_currents = list(BASELINE_CURRENT)[:-1] or list(BASELINE_CURRENT)
        prev_powers = list(BASELINE_POWER)[:-1] or list(BASELINE_POWER)
        base_v = statistics.median(prev_voltages)
        base_i = statistics.median(prev_currents)
        base_p = statistics.median(prev_powers)

        voltage_threshold = max(0.10, 0.02 * max(base_v, 1e-6))
        if voltage < (base_v - voltage_threshold):
            _emit_alert(
                'voltage_drop',
                sample,
                baseline=base_v,
                value=voltage,
                message=f'Clear voltage drop: {voltage:.3f} V vs baseline {base_v:.3f} V',
                cooldown_s=8.0,
            )

        if power < (0.50 * base_p) and current < (0.60 * base_i):
            _emit_alert(
                'load_drop',
                sample,
                baseline=base_p,
                value=power,
                message=(
                    f'Load/power drop: {power:.2f} W vs baseline {base_p:.2f} W '
                    '(possible tail-off or inference completed)'
                ),
                cooldown_s=12.0,
            )


def logger_loop(state: RunState) -> None:
    assert HUB is not None
    csv_path = state.run_dir / 'external_measurements.csv'
    state.csv_path = csv_path
    fieldnames = [
        'sample_idx',
        'sched_wallclock_ns',
        'timestamp_utc',
        'read_start_ns',
        'read_end_ns',
        'mid_wallclock_ns',
        'monotonic_ns',
        'bus_voltage_v',
        'current_a',
        'power_w',
        'bus_voltage_mV',
        'current_mA',
        'power_mW',
        'external_temp_c',
        'tmp117_temp_c',
        'temperature_sensor_type',
        'energy_joules',
    ]
    period_ns = int(round(1e9 / max(state.sample_hz, 1e-6)))
    sample_idx = 0
    prev_mid_ns: Optional[int] = None
    prev_power_w: Optional[float] = None
    energy_j = 0.0

    if state.fast_temp_sensor:
        with HUB_LOCK:
            HUB.configure_temp_sensor_fast_runtime()

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not state.stop_event.is_set():
            sched_ns = state.start_wallclock_ns + sample_idx * period_ns
            wait_until_wallclock_ns(sched_ns)
            read_start_ns = wallclock_ns()
            mono_ns = monotonic_ns()
            with HUB_LOCK:
                reading = HUB.read_once()
            read_end_ns = wallclock_ns()
            mid_ns = (read_start_ns + read_end_ns) // 2
            power_w = float(reading['power_w'])

            if prev_mid_ns is not None and prev_power_w is not None:
                dt_s = (mid_ns - prev_mid_ns) / 1e9
                energy_j += 0.5 * (prev_power_w + power_w) * dt_s

            row = {
                'sample_idx': sample_idx,
                'sched_wallclock_ns': sched_ns,
                'timestamp_utc': utc_now_iso(),
                'read_start_ns': read_start_ns,
                'read_end_ns': read_end_ns,
                'mid_wallclock_ns': mid_ns,
                'monotonic_ns': mono_ns,
                'bus_voltage_v': reading['bus_voltage_v'],
                'current_a': reading['current_a'],
                'power_w': reading['power_w'],
                'bus_voltage_mV': reading['bus_voltage_mV'],
                'current_mA': reading['current_mA'],
                'power_mW': reading['power_mW'],
                'external_temp_c': reading.get('external_temp_c'),
                'tmp117_temp_c': reading.get('tmp117_temp_c'),
                'temperature_sensor_type': reading.get('temperature_sensor_type'),
                'energy_joules': energy_j,
            }
            writer.writerow(row)
            f.flush()

            prev_mid_ns = mid_ns
            prev_power_w = power_w
            sample_idx += 1
            state.samples_written = sample_idx
            state.latest_sample = row
            with STATE_LOCK:
                _update_live_external_state(_deepcopy_jsonable(row))

    summary = summarize_csv(csv_path)
    summary.update(
        {
            'run_id': state.run_id,
            'run_dir': str(state.run_dir),
            'timestamp_utc': utc_now_iso(),
            'scheduled_start_wallclock_ns': state.start_wallclock_ns,
            'sensor_info': HUB.health() if HUB is not None else None,
            'notes': state.notes,
        }
    )
    state.summary_json = state.run_dir / 'logger_summary.json'
    state.summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    state.tar_path = state.run_dir.with_suffix('.tar.gz')
    with tarfile.open(state.tar_path, 'w:gz') as tf:
        tf.add(state.run_dir, arcname=state.run_dir.name)


def finalize_run_artifacts(state: RunState) -> Tuple[Optional[Path], Optional[Path]]:
    csv_path = state.csv_path or (state.run_dir / 'external_measurements.csv')
    if csv_path.exists():
        finalize_errors: List[str] = []
        if state.summary_json is None or not state.summary_json.exists():
            try:
                sensor_info = None
                if HUB is not None:
                    try:
                        with HUB_LOCK:
                            sensor_info = HUB.health()
                    except Exception as exc:
                        sensor_info = {'health_error': str(exc)}
                summary = summarize_csv(csv_path)
                summary.update(
                    {
                        'run_id': state.run_id,
                        'run_dir': str(state.run_dir),
                        'timestamp_utc': utc_now_iso(),
                        'scheduled_start_wallclock_ns': state.start_wallclock_ns,
                        'sensor_info': sensor_info,
                        'notes': state.notes,
                    }
                )
                state.summary_json = state.run_dir / 'logger_summary.json'
                state.summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            except Exception as exc:
                finalize_errors.append(f'summary_error={type(exc).__name__}: {exc}')

        if state.tar_path is None or not state.tar_path.exists():
            try:
                state.tar_path = state.run_dir.with_suffix('.tar.gz')
                with tarfile.open(state.tar_path, 'w:gz') as tf:
                    tf.add(state.run_dir, arcname=state.run_dir.name)
            except Exception as exc:
                finalize_errors.append(f'tar_error={type(exc).__name__}: {exc}')

        if finalize_errors:
            (state.run_dir / 'logger_finalize_error.txt').write_text('\n'.join(finalize_errors) + '\n', encoding='utf-8')
    return state.summary_json, state.tar_path


def _build_current_run_payload() -> Dict[str, Any]:
    now_ns = wallclock_ns()
    if CURRENT_RUN is None:
        return {
            'active': False,
            'last_completed_run': LAST_COMPLETED_RUN,
        }
    start_ns = CURRENT_RUN.start_wallclock_ns
    if now_ns < start_ns:
        status = 'countdown'
        elapsed_s = 0.0
        countdown_s = (start_ns - now_ns) / 1e9
    else:
        status = 'logging'
        elapsed_s = (now_ns - start_ns) / 1e9
        countdown_s = 0.0
    return {
        'active': CURRENT_RUN.thread.is_alive() if CURRENT_RUN.thread is not None else False,
        'status': status,
        'run_id': CURRENT_RUN.run_id,
        'run_dir': str(CURRENT_RUN.run_dir),
        'sample_hz': CURRENT_RUN.sample_hz,
        'scheduled_start_wallclock_ns': start_ns,
        'elapsed_s': elapsed_s,
        'countdown_s': countdown_s,
        'samples_written': CURRENT_RUN.samples_written,
        'notes': CURRENT_RUN.notes,
    }


def _read_external_snapshot_if_idle() -> Optional[Dict[str, Any]]:
    if HUB is None:
        return None
    try:
        with HUB_LOCK:
            row = HUB.read_once()
        row['timestamp_utc'] = utc_now_iso()
        row['mid_wallclock_ns'] = wallclock_ns()
        return row
    except Exception:
        return None


@app.get('/')
def dashboard():
    return render_template('index.html')


@app.get('/manifest.webmanifest')
def manifest_web():
    return send_file(WEB_ROOT / 'static' / 'manifest.webmanifest', mimetype='application/manifest+json')


@app.get('/sw.js')
def service_worker():
    return send_file(WEB_ROOT / 'static' / 'sw.js', mimetype='application/javascript')


@app.get('/health')
def health():
    info = {}
    if HUB is not None:
        with HUB_LOCK:
            info = HUB.health()
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'sensor_info': info, 'current_run': _build_current_run_payload()})


@app.get('/scan')
def scan():
    return jsonify({'ok': True, 'buses': scan_all()})


@app.get('/read_once')
def read_once():
    if HUB is None:
        return jsonify({'ok': False, 'error': 'sensor hub not initialized'}), 500
    with HUB_LOCK:
        row = HUB.read_once()
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'reading': row})


@app.get('/api/live')
def api_live():
    with STATE_LOCK:
        latest_external = _deepcopy_jsonable(LAST_EXTERNAL_READING)
        external_recent = _deepcopy_jsonable(list(LIVE_SAMPLES)[-300:])
        internal_recent = _deepcopy_jsonable(list(INTERNAL_HISTORY)[-300:])
        alerts_recent = _deepcopy_jsonable(list(ALERT_HISTORY)[-20:])
        jetson_status = _deepcopy_jsonable(JETSON_STATUS)
    if not latest_external:
        idle_snap = _read_external_snapshot_if_idle()
        if idle_snap is not None:
            latest_external = idle_snap
    return jsonify(
        {
            'ok': True,
            'timestamp_utc': utc_now_iso(),
            'current_run': _build_current_run_payload(),
            'latest_external': latest_external,
            'external_recent': external_recent,
            'internal_recent': internal_recent,
            'alerts_recent': alerts_recent,
            'jetson_status': jetson_status,
        }
    )


@app.get('/api/jetson_status/current')
def api_jetson_status_current():
    with STATE_LOCK:
        return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'jetson_status': _deepcopy_jsonable(JETSON_STATUS)})


@app.post('/api/jetson_status')
def api_jetson_status():
    payload = request.get_json(force=True)
    payload['server_received_utc'] = utc_now_iso()
    payload['server_received_wallclock_ns'] = wallclock_ns()
    with STATE_LOCK:
        JETSON_STATUS.clear()
        JETSON_STATUS.update(_deepcopy_jsonable(payload))
        internal_live = payload.get('internal_live') or {}
        if internal_live:
            ts_ns = payload.get('wallclock_ns') or payload.get('server_received_wallclock_ns')
            snap = {
                'timestamp_utc': payload.get('timestamp_utc', utc_now_iso()),
                'wallclock_ns': ts_ns,
                'stage': payload.get('stage'),
                'repeat_name': payload.get('repeat_name'),
                'internal_total_power_w': (((internal_live.get('hwmon') or {}).get('total_power_w')) if isinstance(internal_live.get('hwmon'), dict) else None),
                'internal_max_temp_c': (((internal_live.get('thermal') or {}).get('max_temp_c')) if isinstance(internal_live.get('thermal'), dict) else None),
                'ram_used_mb': (((internal_live.get('tegrastats') or {}).get('ram_used_mb')) if isinstance(internal_live.get('tegrastats'), dict) else None),
                'gr3d_pct': (((internal_live.get('tegrastats') or {}).get('gr3d_pct')) if isinstance(internal_live.get('tegrastats'), dict) else None),
                'emc_pct': (((internal_live.get('tegrastats') or {}).get('emc_pct')) if isinstance(internal_live.get('tegrastats'), dict) else None),
            }
            INTERNAL_HISTORY.append(snap)
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso()})


@app.post('/start')
def start():
    global CURRENT_RUN
    payload = request.get_json(force=True)
    run_id = str(payload['run_id'])
    sample_hz = float(payload.get('sample_hz', 10.0))
    start_wallclock_ns = int(payload.get('start_wallclock_ns', wallclock_ns() + 2_000_000_000))
    fast_temp_sensor = bool(payload.get('fast_temp_sensor', True))
    notes = payload.get('notes')

    with RUN_LOCK:
        if CURRENT_RUN is not None and CURRENT_RUN.thread is not None and CURRENT_RUN.thread.is_alive():
            return jsonify({'ok': False, 'error': 'another run is active'}), 409

        run_dir = ensure_dir(OUTPUT_ROOT / run_id)
        meta = {
            'timestamp_utc': utc_now_iso(),
            'run_id': run_id,
            'sample_hz': sample_hz,
            'scheduled_start_wallclock_ns': start_wallclock_ns,
            'notes': notes,
            'sensor_info': HUB.health() if HUB is not None else None,
        }
        (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        state = RunState(
            run_id=run_id,
            run_dir=run_dir,
            sample_hz=sample_hz,
            start_wallclock_ns=start_wallclock_ns,
            fast_temp_sensor=fast_temp_sensor,
            notes=notes if isinstance(notes, dict) else None,
        )
        with STATE_LOCK:
            LIVE_SAMPLES.clear()
            INTERNAL_HISTORY.clear()
            ALERT_HISTORY.clear()
            LAST_EXTERNAL_READING.clear()
            BASELINE_VOLTAGE.clear()
            BASELINE_CURRENT.clear()
            BASELINE_POWER.clear()
            LAST_ALERT_TIME_NS.clear()
        thread = threading.Thread(target=logger_loop, args=(state,), daemon=True)
        state.thread = thread
        CURRENT_RUN = state
        thread.start()

    return jsonify(
        {
            'ok': True,
            'run_id': run_id,
            'run_dir': str(run_dir),
            'scheduled_start_wallclock_ns': start_wallclock_ns,
        }
    )


@app.post('/stop')
def stop():
    global CURRENT_RUN, LAST_COMPLETED_RUN
    payload = request.get_json(force=True)
    run_id = str(payload['run_id'])
    with RUN_LOCK:
        if CURRENT_RUN is None or CURRENT_RUN.run_id != run_id:
            return jsonify({'ok': False, 'error': f'run {run_id} is not active'}), 404

        CURRENT_RUN.stop_event.set()
        if CURRENT_RUN.thread is not None:
            CURRENT_RUN.thread.join(timeout=30)
        finalize_error = None
        try:
            summary_path, tar_path = finalize_run_artifacts(CURRENT_RUN)
        except Exception as exc:
            finalize_error = str(exc)
            summary_path = CURRENT_RUN.summary_json
            tar_path = CURRENT_RUN.tar_path
            (CURRENT_RUN.run_dir / 'logger_finalize_error.txt').write_text(
                f'{type(exc).__name__}: {exc}\n',
                encoding='utf-8',
            )
        LAST_COMPLETED_RUN = {
            'run_id': CURRENT_RUN.run_id,
            'run_dir': str(CURRENT_RUN.run_dir),
            'summary_json': str(summary_path) if summary_path else None,
            'tar_path': str(tar_path) if tar_path else None,
            'timestamp_utc': utc_now_iso(),
            'samples_written': CURRENT_RUN.samples_written,
        }
        response = {
            'ok': True,
            'run_id': run_id,
            'tar_path': str(tar_path) if tar_path else None,
            'summary_json': str(summary_path) if summary_path else None,
        }
        if finalize_error is not None:
            response['finalize_error'] = finalize_error
        CURRENT_RUN = None
    return jsonify(response)


@app.get('/download/<run_id>')
def download(run_id: str):
    tar_path = OUTPUT_ROOT / f'{run_id}.tar.gz'
    if not tar_path.exists():
        return jsonify({'ok': False, 'error': f'{tar_path} not found'}), 404
    return send_file(tar_path, mimetype='application/gzip', as_attachment=True, download_name=tar_path.name)


def main() -> None:
    global OUTPUT_ROOT, HUB

    ap = argparse.ArgumentParser(description='Raspberry Pi 5 external logger for INA260 + TMP117/TMP119, with live dashboard.')
    ap.add_argument('--output-root', default=str(default_output_root('jetson_external_logs')))
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--i2c-bus', default='auto', help='I2C bus number, or "auto"')
    ap.add_argument('--ina260-address', type=lambda x: int(x, 0), default=INA260_ADDR_DEFAULT)
    ap.add_argument('--temp-sensor-address', '--tmp117-address', dest='temp_sensor_address', type=lambda x: int(x, 0), default=TMP117_ADDR_DEFAULT)
    ap.add_argument('--require-temp-sensor', '--require-tmp117', dest='require_temp_sensor', action='store_true')
    ap.add_argument('--fast-temp-sensor-on-init', '--fast-tmp117-on-init', dest='fast_temp_sensor_on_init', action='store_true')
    args = ap.parse_args()

    OUTPUT_ROOT = ensure_dir(Path(args.output_root).expanduser())
    i2c_bus = None if str(args.i2c_bus).lower() == 'auto' else int(args.i2c_bus)
    HUB = SensorHub(
        i2c_bus=i2c_bus,
        ina260_address=args.ina260_address,
        temp_sensor_address=args.temp_sensor_address,
        require_temp_sensor=args.require_temp_sensor,
        fast_temp_sensor_on_init=args.fast_temp_sensor_on_init,
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
