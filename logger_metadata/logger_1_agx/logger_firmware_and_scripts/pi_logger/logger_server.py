from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
    device_key: str
    device_label: str
    sensor_hub: SensorHub
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
SENSOR_HUBS: Dict[str, SensorHub] = {}
CURRENT_RUNS_BY_DEVICE: Dict[str, RunState] = {}
LAST_COMPLETED_RUNS_BY_DEVICE: Dict[str, Dict[str, Any]] = {}
RUN_LOCK = threading.Lock()
HUB_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

LIVE_SAMPLES: Deque[Dict[str, Any]] = deque(maxlen=2400)
LIVE_SAMPLES_BY_DEVICE: Dict[str, Deque[Dict[str, Any]]] = {}
INTERNAL_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=1200)
ALERT_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=100)
ALERT_HISTORY_BY_DEVICE: Dict[str, Deque[Dict[str, Any]]] = {}
JETSON_POWER_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=200)
JETSON_STATUS: Dict[str, Any] = {}
KNOWN_DEVICES: Dict[str, Dict[str, Any]] = {
    'jetson-nano': {'key': 'jetson-nano', 'label': 'Jetson Nano', 'ips': ['${JETSON_HOST_IP}']},
    'jetson-agx': {'key': 'jetson-agx', 'label': 'Jetson AGX', 'ips': ['${LOGGER_PC_IP}']},
}
JETSON_STATUSES: Dict[str, Dict[str, Any]] = {}
INTERNAL_HISTORY_BY_DEVICE: Dict[str, Deque[Dict[str, Any]]] = {
    key: deque(maxlen=1200) for key in KNOWN_DEVICES
}
JETSON_POWER_EVENTS_BY_DEVICE: Dict[str, Deque[Dict[str, Any]]] = {
    key: deque(maxlen=200) for key in KNOWN_DEVICES
}
LAST_EXTERNAL_READING: Dict[str, Any] = {}
LAST_EXTERNAL_READING_BY_DEVICE: Dict[str, Dict[str, Any]] = {}
BASELINE_VOLTAGE: Deque[float] = deque(maxlen=30)
BASELINE_CURRENT: Deque[float] = deque(maxlen=30)
BASELINE_POWER: Deque[float] = deque(maxlen=30)
BASELINE_VOLTAGE_BY_DEVICE: Dict[str, Deque[float]] = {}
BASELINE_CURRENT_BY_DEVICE: Dict[str, Deque[float]] = {}
BASELINE_POWER_BY_DEVICE: Dict[str, Deque[float]] = {}
LAST_ALERT_TIME_NS: Dict[str, int] = {}
LAST_ALERT_TIME_NS_BY_DEVICE: Dict[str, Dict[str, int]] = {}


def _safe_filename_part(value: Any, fallback: str = 'events') -> str:
    text = str(value or fallback)
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('._')
    return text[:96] or fallback


def _device_key_from_text(value: Any) -> Optional[str]:
    text = str(value or '').lower()
    if not text:
        return None
    if 'agx' in text:
        return 'jetson-agx'
    if 'nano' in text:
        return 'jetson-nano'
    return None


def _device_from_ip(source_ip: Optional[str]) -> Optional[str]:
    if not source_ip:
        return None
    for key, info in KNOWN_DEVICES.items():
        if source_ip in info.get('ips', []):
            return key
    return None


def _device_label(key: str, payload: Optional[Dict[str, Any]] = None, source_ip: Optional[str] = None) -> str:
    if key in KNOWN_DEVICES:
        return str(KNOWN_DEVICES[key]['label'])
    payload = payload or {}
    for field in ('device_label', 'device_name', 'hostname', 'host_name', 'board_name'):
        value = str(payload.get(field) or '').strip()
        if value:
            return value
    return f'Jetson {source_ip}' if source_ip else key


def _identify_device(payload: Dict[str, Any], source_ip: Optional[str] = None) -> Tuple[str, str]:
    explicit = payload.get('device_key') or payload.get('device_id') or payload.get('jetson_id')
    if explicit:
        key = _safe_filename_part(str(explicit).lower(), fallback='jetson')
        return key, _device_label(key, payload, source_ip)

    key = _device_from_ip(source_ip)
    if key:
        return key, _device_label(key, payload, source_ip)

    for field in ('hostname', 'host_name', 'device_name', 'board_name', 'run_id', 'tag'):
        key = _device_key_from_text(payload.get(field))
        if key:
            return key, _device_label(key, payload, source_ip)

    if source_ip:
        key = 'jetson-' + _safe_filename_part(source_ip.replace('.', '-'), fallback='unknown')
        return key, _device_label(key, payload, source_ip)
    return 'jetson-unknown', _device_label('jetson-unknown', payload, source_ip)


def _device_deque(
    store: Dict[str, Deque[Dict[str, Any]]],
    key: str,
    maxlen: int,
) -> Deque[Dict[str, Any]]:
    if key not in store:
        store[key] = deque(maxlen=maxlen)
    return store[key]


def _device_float_deque(store: Dict[str, Deque[float]], key: str, maxlen: int) -> Deque[float]:
    if key not in store:
        store[key] = deque(maxlen=maxlen)
    return store[key]


def _hub_for_device(device_key: str) -> Optional[SensorHub]:
    if device_key in SENSOR_HUBS:
        return SENSOR_HUBS[device_key]
    if len(SENSOR_HUBS) == 1:
        return next(iter(SENSOR_HUBS.values()))
    return HUB


def _current_run_for_device(device_key: str) -> Optional[RunState]:
    state = CURRENT_RUNS_BY_DEVICE.get(device_key)
    if state is not None:
        return state
    if CURRENT_RUN is not None and CURRENT_RUN.device_key == device_key:
        return CURRENT_RUN
    return None


def _all_device_keys_unlocked() -> List[str]:
    keys = list(KNOWN_DEVICES.keys())
    for key in (
        list(JETSON_STATUSES.keys())
        + list(INTERNAL_HISTORY_BY_DEVICE.keys())
        + list(JETSON_POWER_EVENTS_BY_DEVICE.keys())
        + list(SENSOR_HUBS.keys())
        + list(CURRENT_RUNS_BY_DEVICE.keys())
    ):
        if key not in keys:
            keys.append(key)
    return keys


def _select_device_key_unlocked(requested: Optional[str] = None) -> str:
    keys = _all_device_keys_unlocked()
    if requested and requested in keys:
        return requested
    now_ns = wallclock_ns()
    active_keys = [
        key for key in keys
        if JETSON_STATUSES.get(key, {}).get('state') not in (None, '', 'idle', 'finished')
        and (now_ns - int(JETSON_STATUSES.get(key, {}).get('server_received_wallclock_ns') or 0)) < 30_000_000_000
    ]
    if active_keys:
        active_keys.sort(
            key=lambda key: int(JETSON_STATUSES.get(key, {}).get('server_received_wallclock_ns') or 0),
            reverse=True,
        )
        return active_keys[0]
    return keys[0] if keys else 'jetson-nano'


def _requested_device_key_from_request() -> Optional[str]:
    return request.args.get('device') or _device_from_ip(request.remote_addr)


def _device_list_unlocked(selected_key: Optional[str] = None) -> List[Dict[str, Any]]:
    now_ns = wallclock_ns()
    out: List[Dict[str, Any]] = []
    for key in _all_device_keys_unlocked():
        status = JETSON_STATUSES.get(key, {})
        current_run = _current_run_for_device(key)
        label = status.get('device_label') or KNOWN_DEVICES.get(key, {}).get('label') or key
        seen_ns = status.get('server_received_wallclock_ns')
        stale_s = ((now_ns - int(seen_ns)) / 1e9) if isinstance(seen_ns, int) else None
        state = status.get('state') or ('running' if current_run is not None else ('idle' if not status else 'running'))
        out.append(
            {
                'key': key,
                'label': label,
                'source_ip': status.get('source_ip') or (KNOWN_DEVICES.get(key, {}).get('ips') or [None])[0],
                'last_seen_utc': status.get('server_received_utc'),
                'stale_s': stale_s,
                'state': state,
                'stage': status.get('stage'),
                'run_id': status.get('run_id'),
                'external_logger_active': current_run is not None and current_run.thread is not None and current_run.thread.is_alive(),
                'sensor_configured': _hub_for_device(key) is not None,
                'selected': key == selected_key,
            }
        )
    return out


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + '\n')


def _power_event_log_path(event: Dict[str, Any]) -> Path:
    device_key = event.get('device_key') or 'unknown'
    active_run = _current_run_for_device(str(device_key))
    if active_run is not None:
        return active_run.run_dir / 'jetson_power_events.jsonl'
    day = time.strftime('%Y%m%d', time.gmtime())
    status = JETSON_STATUSES.get(str(device_key), JETSON_STATUS)
    run_hint = event.get('run_id') or event.get('tag') or status.get('run_id') or 'idle'
    name = f'{day}_{_safe_filename_part(run_hint)}.jsonl'
    return ensure_dir(OUTPUT_ROOT / 'jetson_power_events' / _safe_filename_part(device_key, fallback='unknown')) / name


def _record_jetson_power_event(payload: Dict[str, Any], source_ip: Optional[str] = None) -> Dict[str, Any]:
    event = _deepcopy_jsonable(payload)
    device_key, device_label = _identify_device(event, source_ip)
    event['device_key'] = device_key
    event['device_label'] = device_label
    if source_ip:
        event['source_ip'] = source_ip
    event['server_received_utc'] = utc_now_iso()
    event['server_received_wallclock_ns'] = wallclock_ns()
    active_run = _current_run_for_device(device_key)
    if 'run_id' not in event and active_run is not None:
        event['run_id'] = active_run.run_id
    device_status = JETSON_STATUSES.get(device_key, JETSON_STATUS)
    if 'jetson_status_run_id' not in event and device_status.get('run_id'):
        event['jetson_status_run_id'] = device_status.get('run_id')
    log_path = _power_event_log_path(event)
    event['logger_event_log_path'] = str(log_path)
    _append_jsonl(log_path, event)
    JETSON_POWER_EVENTS.append(event)
    _device_deque(JETSON_POWER_EVENTS_BY_DEVICE, device_key, 200).append(event)
    return event


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


def _emit_alert(
    event_type: str,
    sample: Dict[str, Any],
    baseline: float,
    value: float,
    message: str,
    cooldown_s: float = 10.0,
) -> None:
    device_key = str(sample.get('device_key') or 'jetson-unknown')
    cooldowns = LAST_ALERT_TIME_NS_BY_DEVICE.setdefault(device_key, {})
    now_ns = int(sample.get('mid_wallclock_ns') or wallclock_ns())
    last_ns = cooldowns.get(event_type)
    if last_ns is not None and (now_ns - last_ns) < int(cooldown_s * 1e9):
        return
    cooldowns[event_type] = now_ns
    LAST_ALERT_TIME_NS[event_type] = now_ns
    current_run = _current_run_for_device(device_key)
    alert = {
        'event_type': event_type,
        'device_key': device_key,
        'device_label': sample.get('device_label'),
        'timestamp_utc': sample.get('timestamp_utc', utc_now_iso()),
        'sample_idx': sample.get('sample_idx'),
        'run_id': current_run.run_id if current_run is not None else JETSON_STATUSES.get(device_key, {}).get('run_id'),
        'baseline': baseline,
        'value': value,
        'delta': value - baseline,
        'message': message,
    }
    ALERT_HISTORY.append(alert)
    _device_deque(ALERT_HISTORY_BY_DEVICE, device_key, 100).append(alert)


def _update_live_external_state(sample: Dict[str, Any], device_key: str) -> None:
    sample['device_key'] = device_key
    sample.setdefault('device_label', _device_label(device_key))
    voltage = float(sample['bus_voltage_v'])
    current = float(sample['current_a'])
    power = float(sample['power_w'])
    baseline_voltage = _device_float_deque(BASELINE_VOLTAGE_BY_DEVICE, device_key, 30)
    baseline_current = _device_float_deque(BASELINE_CURRENT_BY_DEVICE, device_key, 30)
    baseline_power = _device_float_deque(BASELINE_POWER_BY_DEVICE, device_key, 30)
    baseline_voltage.append(voltage)
    baseline_current.append(current)
    baseline_power.append(power)
    BASELINE_VOLTAGE.append(voltage)
    BASELINE_CURRENT.append(current)
    BASELINE_POWER.append(power)

    LIVE_SAMPLES.append(sample)
    _device_deque(LIVE_SAMPLES_BY_DEVICE, device_key, 2400).append(sample)
    LAST_EXTERNAL_READING.clear()
    LAST_EXTERNAL_READING.update(sample)
    LAST_EXTERNAL_READING_BY_DEVICE[device_key] = _deepcopy_jsonable(sample)

    if len(baseline_voltage) >= 12:
        prev_voltages = list(baseline_voltage)[:-1] or list(baseline_voltage)
        prev_currents = list(baseline_current)[:-1] or list(baseline_current)
        prev_powers = list(baseline_power)[:-1] or list(baseline_power)
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
    hub = state.sensor_hub
    csv_path = state.run_dir / 'external_measurements.csv'
    state.csv_path = csv_path
    fieldnames = [
        'device_key',
        'device_label',
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
        'tmp119_temp_c',
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
            hub.configure_temp_sensor_fast_runtime()

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not state.stop_event.is_set():
            sched_ns = state.start_wallclock_ns + sample_idx * period_ns
            wait_until_wallclock_ns(sched_ns)
            read_start_ns = wallclock_ns()
            mono_ns = monotonic_ns()
            with HUB_LOCK:
                reading = hub.read_once()
            read_end_ns = wallclock_ns()
            mid_ns = (read_start_ns + read_end_ns) // 2
            power_w = float(reading['power_w'])

            if prev_mid_ns is not None and prev_power_w is not None:
                dt_s = (mid_ns - prev_mid_ns) / 1e9
                energy_j += 0.5 * (prev_power_w + power_w) * dt_s

            row = {
                'device_key': state.device_key,
                'device_label': state.device_label,
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
                'tmp119_temp_c': reading.get('tmp119_temp_c'),
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
                _update_live_external_state(_deepcopy_jsonable(row), state.device_key)

    summary = summarize_csv(csv_path)
    summary.update(
        {
            'run_id': state.run_id,
            'run_dir': str(state.run_dir),
            'device_key': state.device_key,
            'device_label': state.device_label,
            'timestamp_utc': utc_now_iso(),
            'scheduled_start_wallclock_ns': state.start_wallclock_ns,
            'sensor_info': hub.health(),
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
                try:
                    with HUB_LOCK:
                        sensor_info = state.sensor_hub.health()
                except Exception as exc:
                    sensor_info = {'health_error': str(exc)}
                summary = summarize_csv(csv_path)
                summary.update(
                    {
                        'run_id': state.run_id,
                        'run_dir': str(state.run_dir),
                        'device_key': state.device_key,
                        'device_label': state.device_label,
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


def _build_current_run_payload(device_key: Optional[str] = None) -> Dict[str, Any]:
    now_ns = wallclock_ns()
    selected_key = device_key or _select_device_key_unlocked(None)
    state = _current_run_for_device(selected_key)
    if state is None:
        return {
            'active': False,
            'device_key': selected_key,
            'last_completed_run': LAST_COMPLETED_RUNS_BY_DEVICE.get(selected_key) or LAST_COMPLETED_RUN,
        }
    start_ns = state.start_wallclock_ns
    if now_ns < start_ns:
        status = 'countdown'
        elapsed_s = 0.0
        countdown_s = (start_ns - now_ns) / 1e9
    else:
        status = 'logging'
        elapsed_s = (now_ns - start_ns) / 1e9
        countdown_s = 0.0
    return {
        'active': state.thread.is_alive() if state.thread is not None else False,
        'status': status,
        'run_id': state.run_id,
        'run_dir': str(state.run_dir),
        'device_key': state.device_key,
        'device_label': state.device_label,
        'sample_hz': state.sample_hz,
        'scheduled_start_wallclock_ns': start_ns,
        'elapsed_s': elapsed_s,
        'countdown_s': countdown_s,
        'samples_written': state.samples_written,
        'notes': state.notes,
    }


def _read_external_snapshot_if_idle(device_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    selected_key = device_key or _select_device_key_unlocked(None)
    hub = _hub_for_device(selected_key)
    if hub is None:
        return None
    try:
        with HUB_LOCK:
            row = hub.read_once()
        row['timestamp_utc'] = utc_now_iso()
        row['mid_wallclock_ns'] = wallclock_ns()
        row['device_key'] = selected_key
        row['device_label'] = _device_label(selected_key)
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
    with STATE_LOCK:
        selected_key = _select_device_key_unlocked(_requested_device_key_from_request())
        devices = _deepcopy_jsonable(_device_list_unlocked(selected_key))
    info = {}
    hub = _hub_for_device(selected_key)
    if hub is not None:
        with HUB_LOCK:
            info = hub.health()
    return jsonify(
        {
            'ok': True,
            'timestamp_utc': utc_now_iso(),
            'sensor_info': info,
            'current_run': _build_current_run_payload(selected_key),
            'selected_device_key': selected_key,
            'devices': devices,
        }
    )


@app.get('/scan')
def scan():
    return jsonify({'ok': True, 'buses': scan_all()})


@app.get('/read_once')
def read_once():
    with STATE_LOCK:
        selected_key = _select_device_key_unlocked(_requested_device_key_from_request())
    hub = _hub_for_device(selected_key)
    if hub is None:
        return jsonify({'ok': False, 'error': 'sensor hub not initialized'}), 500
    with HUB_LOCK:
        row = hub.read_once()
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'device_key': selected_key, 'reading': row})


@app.get('/api/live')
def api_live():
    requested_device = _requested_device_key_from_request()
    with STATE_LOCK:
        selected_device_key = _select_device_key_unlocked(requested_device)
        devices = _deepcopy_jsonable(_device_list_unlocked(selected_device_key))
        latest_external = _deepcopy_jsonable(LAST_EXTERNAL_READING_BY_DEVICE.get(selected_device_key, {}))
        external_recent = _deepcopy_jsonable(list(LIVE_SAMPLES_BY_DEVICE.get(selected_device_key, deque()))[-300:])
        internal_recent = _deepcopy_jsonable(list(INTERNAL_HISTORY_BY_DEVICE.get(selected_device_key, deque()))[-300:])
        alerts_recent = _deepcopy_jsonable(list(ALERT_HISTORY_BY_DEVICE.get(selected_device_key, deque()))[-20:])
        jetson_power_events = _deepcopy_jsonable(list(JETSON_POWER_EVENTS_BY_DEVICE.get(selected_device_key, deque()))[-50:])
        jetson_status = _deepcopy_jsonable(JETSON_STATUSES.get(selected_device_key, {}))
        device_statuses = _deepcopy_jsonable(JETSON_STATUSES)
        all_power_events = _deepcopy_jsonable(list(JETSON_POWER_EVENTS)[-50:])
    if not latest_external:
        idle_snap = _read_external_snapshot_if_idle(selected_device_key)
        if idle_snap is not None:
            latest_external = idle_snap
    return jsonify(
        {
            'ok': True,
            'timestamp_utc': utc_now_iso(),
            'current_run': _build_current_run_payload(selected_device_key),
            'latest_external': latest_external,
            'external_recent': external_recent,
            'internal_recent': internal_recent,
            'alerts_recent': alerts_recent,
            'jetson_power_events': jetson_power_events,
            'all_jetson_power_events': all_power_events,
            'jetson_status': jetson_status,
            'jetson_statuses': device_statuses,
            'selected_device_key': selected_device_key,
            'devices': devices,
        }
    )


@app.get('/api/jetson_status/current')
def api_jetson_status_current():
    with STATE_LOCK:
        selected_key = _select_device_key_unlocked(_requested_device_key_from_request())
        return jsonify(
            {
                'ok': True,
                'timestamp_utc': utc_now_iso(),
                'selected_device_key': selected_key,
                'jetson_status': _deepcopy_jsonable(JETSON_STATUSES.get(selected_key, {})),
                'jetson_statuses': _deepcopy_jsonable(JETSON_STATUSES),
                'devices': _deepcopy_jsonable(_device_list_unlocked(selected_key)),
            }
        )


@app.get('/api/jetson_power_events')
def api_jetson_power_events_current():
    with STATE_LOCK:
        selected_key = _select_device_key_unlocked(_requested_device_key_from_request())
        events = JETSON_POWER_EVENTS_BY_DEVICE.get(selected_key, deque())
        return jsonify(
            {
                'ok': True,
                'timestamp_utc': utc_now_iso(),
                'selected_device_key': selected_key,
                'events': _deepcopy_jsonable(list(events)),
                'all_events': _deepcopy_jsonable(list(JETSON_POWER_EVENTS)),
                'devices': _deepcopy_jsonable(_device_list_unlocked(selected_key)),
            }
        )


@app.post('/api/jetson_power_event')
@app.post('/api/power_event')
def api_jetson_power_event():
    payload = request.get_json(force=True, silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'ok': False, 'error': 'payload must be a JSON object'}), 400
    with STATE_LOCK:
        event = _record_jetson_power_event(payload, request.remote_addr)
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'event': event})


@app.post('/api/jetson_status')
def api_jetson_status():
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        return jsonify({'ok': False, 'error': 'payload must be a JSON object'}), 400
    device_key, device_label = _identify_device(payload, request.remote_addr)
    payload['device_key'] = device_key
    payload['device_label'] = device_label
    if request.remote_addr:
        payload['source_ip'] = request.remote_addr
    payload['server_received_utc'] = utc_now_iso()
    payload['server_received_wallclock_ns'] = wallclock_ns()
    with STATE_LOCK:
        JETSON_STATUS.clear()
        JETSON_STATUS.update(_deepcopy_jsonable(payload))
        JETSON_STATUSES[device_key] = _deepcopy_jsonable(payload)
        internal_live = payload.get('internal_live') or {}
        if internal_live:
            ts_ns = payload.get('wallclock_ns') or payload.get('server_received_wallclock_ns')
            snap = {
                'timestamp_utc': payload.get('timestamp_utc', utc_now_iso()),
                'wallclock_ns': ts_ns,
                'device_key': device_key,
                'device_label': device_label,
                'stage': payload.get('stage'),
                'repeat_name': payload.get('repeat_name'),
                'internal_total_power_w': (((internal_live.get('hwmon') or {}).get('total_power_w')) if isinstance(internal_live.get('hwmon'), dict) else None),
                'internal_max_temp_c': (((internal_live.get('thermal') or {}).get('max_temp_c')) if isinstance(internal_live.get('thermal'), dict) else None),
                'ram_used_mb': (((internal_live.get('tegrastats') or {}).get('ram_used_mb')) if isinstance(internal_live.get('tegrastats'), dict) else None),
                'gr3d_pct': (((internal_live.get('tegrastats') or {}).get('gr3d_pct')) if isinstance(internal_live.get('tegrastats'), dict) else None),
                'emc_pct': (((internal_live.get('tegrastats') or {}).get('emc_pct')) if isinstance(internal_live.get('tegrastats'), dict) else None),
            }
            INTERNAL_HISTORY.append(snap)
            _device_deque(INTERNAL_HISTORY_BY_DEVICE, device_key, 1200).append(snap)
    return jsonify({'ok': True, 'timestamp_utc': utc_now_iso(), 'device_key': device_key, 'device_label': device_label})


@app.post('/start')
def start():
    global CURRENT_RUN
    payload = request.get_json(force=True)
    run_id = str(payload['run_id'])
    sample_hz = float(payload.get('sample_hz', 10.0))
    start_wallclock_ns = int(payload.get('start_wallclock_ns', wallclock_ns() + 2_000_000_000))
    fast_temp_sensor = bool(payload.get('fast_temp_sensor', True))
    notes = payload.get('notes')
    device_payload = dict(payload)
    if isinstance(notes, dict):
        device_payload.update(notes)
    device_key, device_label = _identify_device(device_payload, request.remote_addr)
    notes_dict = dict(notes) if isinstance(notes, dict) else {}
    notes_dict.setdefault('device_key', device_key)
    notes_dict.setdefault('device_label', device_label)
    if request.remote_addr:
        notes_dict.setdefault('source_ip', request.remote_addr)
    hub = _hub_for_device(device_key)
    if hub is None:
        return jsonify({'ok': False, 'error': f'no sensor hub configured for {device_key}'}), 400

    with RUN_LOCK:
        active_run = CURRENT_RUNS_BY_DEVICE.get(device_key)
        if active_run is not None and active_run.thread is not None and active_run.thread.is_alive():
            return jsonify({'ok': False, 'error': f'another run is active for {device_key}'}), 409

        run_dir = ensure_dir(OUTPUT_ROOT / device_key / run_id)
        with HUB_LOCK:
            sensor_info = hub.health()
        meta = {
            'timestamp_utc': utc_now_iso(),
            'run_id': run_id,
            'device_key': device_key,
            'device_label': device_label,
            'sample_hz': sample_hz,
            'scheduled_start_wallclock_ns': start_wallclock_ns,
            'notes': notes_dict,
            'sensor_info': sensor_info,
        }
        (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        state = RunState(
            run_id=run_id,
            run_dir=run_dir,
            device_key=device_key,
            device_label=device_label,
            sensor_hub=hub,
            sample_hz=sample_hz,
            start_wallclock_ns=start_wallclock_ns,
            fast_temp_sensor=fast_temp_sensor,
            notes=notes_dict,
        )
        with STATE_LOCK:
            _device_deque(LIVE_SAMPLES_BY_DEVICE, device_key, 2400).clear()
            _device_deque(INTERNAL_HISTORY_BY_DEVICE, device_key, 1200).clear()
            _device_deque(ALERT_HISTORY_BY_DEVICE, device_key, 100).clear()
            _device_deque(JETSON_POWER_EVENTS_BY_DEVICE, device_key, 200).clear()
            LAST_EXTERNAL_READING_BY_DEVICE.pop(device_key, None)
            _device_float_deque(BASELINE_VOLTAGE_BY_DEVICE, device_key, 30).clear()
            _device_float_deque(BASELINE_CURRENT_BY_DEVICE, device_key, 30).clear()
            _device_float_deque(BASELINE_POWER_BY_DEVICE, device_key, 30).clear()
            LAST_ALERT_TIME_NS_BY_DEVICE[device_key] = {}
        thread = threading.Thread(target=logger_loop, args=(state,), daemon=True)
        state.thread = thread
        CURRENT_RUNS_BY_DEVICE[device_key] = state
        CURRENT_RUN = state
        thread.start()

    return jsonify(
        {
            'ok': True,
            'run_id': run_id,
            'run_dir': str(run_dir),
            'scheduled_start_wallclock_ns': start_wallclock_ns,
            'device_key': device_key,
            'device_label': device_label,
        }
    )


@app.post('/stop')
def stop():
    global CURRENT_RUN, LAST_COMPLETED_RUN
    payload = request.get_json(force=True)
    run_id = str(payload['run_id'])
    device_payload = dict(payload)
    device_key, _device_label_value = _identify_device(device_payload, request.remote_addr)
    with RUN_LOCK:
        state = CURRENT_RUNS_BY_DEVICE.get(device_key)
        if state is None or state.run_id != run_id:
            for candidate in CURRENT_RUNS_BY_DEVICE.values():
                if candidate.run_id == run_id:
                    state = candidate
                    device_key = candidate.device_key
                    break
        if state is None or state.run_id != run_id:
            return jsonify({'ok': False, 'error': f'run {run_id} is not active'}), 404

        state.stop_event.set()
        if state.thread is not None:
            state.thread.join(timeout=30)
        finalize_error = None
        try:
            summary_path, tar_path = finalize_run_artifacts(state)
        except Exception as exc:
            finalize_error = str(exc)
            summary_path = state.summary_json
            tar_path = state.tar_path
            (state.run_dir / 'logger_finalize_error.txt').write_text(
                f'{type(exc).__name__}: {exc}\n',
                encoding='utf-8',
            )
        completed = {
            'run_id': state.run_id,
            'run_dir': str(state.run_dir),
            'device_key': state.device_key,
            'device_label': state.device_label,
            'summary_json': str(summary_path) if summary_path else None,
            'tar_path': str(tar_path) if tar_path else None,
            'timestamp_utc': utc_now_iso(),
            'samples_written': state.samples_written,
        }
        LAST_COMPLETED_RUNS_BY_DEVICE[state.device_key] = completed
        LAST_COMPLETED_RUN = completed
        response = {
            'ok': True,
            'run_id': run_id,
            'device_key': state.device_key,
            'device_label': state.device_label,
            'tar_path': str(tar_path) if tar_path else None,
            'summary_json': str(summary_path) if summary_path else None,
        }
        if finalize_error is not None:
            response['finalize_error'] = finalize_error
        CURRENT_RUNS_BY_DEVICE.pop(state.device_key, None)
        if CURRENT_RUN is state:
            CURRENT_RUN = next(iter(CURRENT_RUNS_BY_DEVICE.values()), None)
    return jsonify(response)


@app.get('/download/<run_id>')
def download(run_id: str):
    requested_key = _requested_device_key_from_request()
    candidates: List[Path] = []
    if requested_key:
        candidates.append(OUTPUT_ROOT / requested_key / f'{run_id}.tar.gz')
    candidates.append(OUTPUT_ROOT / f'{run_id}.tar.gz')
    candidates.extend(sorted(OUTPUT_ROOT.glob(f'*/{run_id}.tar.gz')))
    tar_path = next((path for path in candidates if path.exists()), candidates[0])
    if not tar_path.exists():
        return jsonify({'ok': False, 'error': f'{tar_path} not found'}), 404
    return send_file(tar_path, mimetype='application/gzip', as_attachment=True, download_name=tar_path.name)


def _parse_sensor_port_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in re.split(r'[,;\s]+', raw.strip()):
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f'invalid sensor map item {item!r}; expected device_key=/dev/ttyACM0')
        key_raw, port_raw = item.split('=', 1)
        key = _device_key_from_text(key_raw) or _safe_filename_part(key_raw.lower(), fallback='jetson')
        port = port_raw.strip()
        if not port:
            raise ValueError(f'empty serial port for {key_raw!r}')
        mapping[key] = port
    return mapping


def _configured_sensor_ports(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    raw_map = args.sensor_port_map or os.environ.get('PC_SENSOR_PORTS') or os.environ.get('SENSOR_PORTS') or ''
    mapping: Dict[str, Optional[str]] = {}
    if raw_map.strip():
        mapping.update(_parse_sensor_port_map(raw_map))

    nano_port = os.environ.get('PC_SENSOR_PORT_NANO') or os.environ.get('NANO_SENSOR_PORT')
    agx_port = os.environ.get('PC_SENSOR_PORT_AGX') or os.environ.get('AGX_SENSOR_PORT')
    if nano_port:
        mapping['jetson-nano'] = nano_port
    if agx_port:
        mapping['jetson-agx'] = agx_port

    if not mapping:
        mapping['jetson-nano'] = args.serial_port or os.environ.get('PC_SENSOR_PORT') or os.environ.get('SENSOR_SERIAL_PORT')
    return mapping


def main() -> None:
    global OUTPUT_ROOT, HUB, SENSOR_HUBS

    ap = argparse.ArgumentParser(description='PC external logger for QT Py INA260 + TMP117/TMP119 serial bridge, with live dashboard.')
    ap.add_argument('--output-root', default=str(default_output_root('jetson_external_logs')))
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--serial-port', default=None, help='QT Py serial port, e.g. /dev/ttyACM0. Defaults to PC_SENSOR_PORT/autodetect.')
    ap.add_argument('--sensor-port-map', default='', help='Comma-separated map, e.g. jetson-nano=/dev/ttyACM0,jetson-agx=/dev/ttyACM1.')
    ap.add_argument('--i2c-bus', default='auto', help='I2C bus number, or "auto"')
    ap.add_argument('--ina260-address', type=lambda x: int(x, 0), default=INA260_ADDR_DEFAULT)
    ap.add_argument('--temp-sensor-address', '--tmp117-address', dest='temp_sensor_address', type=lambda x: int(x, 0), default=TMP117_ADDR_DEFAULT)
    ap.add_argument('--require-temp-sensor', '--require-tmp117', dest='require_temp_sensor', action='store_true')
    ap.add_argument('--fast-temp-sensor-on-init', '--fast-tmp117-on-init', dest='fast_temp_sensor_on_init', action='store_true')
    args = ap.parse_args()

    OUTPUT_ROOT = ensure_dir(Path(args.output_root).expanduser())
    i2c_bus = None if str(args.i2c_bus).lower() == 'auto' else int(args.i2c_bus)
    sensor_ports = _configured_sensor_ports(args)
    SENSOR_HUBS = {}
    for device_key, serial_port in sensor_ports.items():
        SENSOR_HUBS[device_key] = SensorHub(
            i2c_bus=i2c_bus,
            ina260_address=args.ina260_address,
            temp_sensor_address=args.temp_sensor_address,
            require_temp_sensor=args.require_temp_sensor,
            fast_temp_sensor_on_init=args.fast_temp_sensor_on_init,
            serial_port=serial_port,
        )
    HUB = SENSOR_HUBS.get('jetson-nano') or next(iter(SENSOR_HUBS.values()), None)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
