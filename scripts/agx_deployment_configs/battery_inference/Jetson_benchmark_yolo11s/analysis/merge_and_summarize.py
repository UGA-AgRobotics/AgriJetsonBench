from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _dump_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _find_single(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matching {pattern} under {root}')
    return matches[0]


def _rename_external(df: pd.DataFrame) -> pd.DataFrame:
    keep = {}
    for c in df.columns:
        if c in {'time_ns', 'mid_wallclock_ns', 'sched_wallclock_ns', 'timestamp_utc'}:
            keep[c] = c
        else:
            keep[c] = f'ext_{c}'
    return df.rename(columns=keep)


def _rename_process_memory(df: pd.DataFrame) -> pd.DataFrame:
    keep = {}
    for c in df.columns:
        if c in {'time_ns', 'wallclock_ns', 'monotonic_ns', 'timestamp_utc'}:
            keep[c] = c
        else:
            keep[c] = f'proc_{c}'
    return df.rename(columns=keep)


def _pivot_hwmon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    if 'label' in df.columns:
        rail = df['label'].fillna('').astype(str)
    else:
        rail = pd.Series('', index=df.index)
    sensor = df.get('sensor_name', pd.Series('', index=df.index)).fillna('').astype(str)
    ch = df.get('channel_id', pd.Series('', index=df.index)).fillna('').astype(str)
    rail_name = rail.where(rail.str.len() > 0, sensor + '_ch' + ch)
    df = df.copy()
    df['rail_name'] = rail_name
    reported = pd.to_numeric(df.get('power_mW_reported'), errors='coerce')
    computed = pd.to_numeric(df.get('power_mW_computed'), errors='coerce')
    df['power_mW'] = reported.fillna(computed)
    df['time_ns'] = pd.to_numeric(df.get('sched_wallclock_ns'), errors='coerce')
    df = df.dropna(subset=['time_ns', 'power_mW'])
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    pivot = df.pivot_table(index='time_ns', columns='rail_name', values='power_mW', aggfunc='mean')
    pivot.columns = [f'int_power_mW_{str(c).replace(" ", "_")}' for c in pivot.columns]
    pivot = pivot.reset_index().sort_values('time_ns')
    power_cols = [c for c in pivot.columns if c.startswith('int_power_mW_')]
    if power_cols:
        pivot['int_total_power_mW_sum'] = pivot[power_cols].sum(axis=1, skipna=True)
    return pivot


def _pivot_thermal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    df = df.copy()
    df['time_ns'] = pd.to_numeric(df.get('sched_wallclock_ns'), errors='coerce')
    df['type'] = df['type'].fillna('unknown').astype(str)
    df['temp_c'] = pd.to_numeric(df['temp_c'], errors='coerce')
    df = df.dropna(subset=['time_ns', 'temp_c'])
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    pivot = df.pivot_table(index='time_ns', columns='type', values='temp_c', aggfunc='mean')
    pivot.columns = [f'temp_{str(c).replace(" ", "_")}_c' for c in pivot.columns]
    pivot = pivot.reset_index().sort_values('time_ns')
    temp_cols = [c for c in pivot.columns if c.startswith('temp_') and c.endswith('_c')]
    if temp_cols:
        pivot['internal_temp_max_c'] = pivot[temp_cols].max(axis=1, skipna=True)
        pivot['internal_temp_mean_c'] = pivot[temp_cols].mean(axis=1, skipna=True)
    return pivot


def _load_tegrastats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    df = df.copy()
    df['time_ns'] = pd.to_numeric(df['wallclock_ns'], errors='coerce')
    if 'temps_json' in df.columns:
        max_temps: List[Optional[float]] = []
        for raw in df['temps_json'].fillna(''):
            try:
                parsed = json.loads(str(raw))
                vals = [float(v) for v in parsed.values()]
                max_temps.append(max(vals) if vals else None)
            except Exception:
                max_temps.append(None)
        df['tstats_max_temp_c'] = max_temps
    return df.dropna(subset=['time_ns']).sort_values('time_ns')


def _load_process_memory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['time_ns'])
    df = df.copy()
    df['time_ns'] = pd.to_numeric(df.get('wallclock_ns'), errors='coerce')
    df = df.dropna(subset=['time_ns']).sort_values('time_ns')
    return _rename_process_memory(df)


def _agreement_metrics(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if 'ext_power_w' not in df.columns or 'int_total_power_mW_sum' not in df.columns:
        return None
    a = pd.to_numeric(df['ext_power_w'], errors='coerce')
    b = pd.to_numeric(df['int_total_power_mW_sum'], errors='coerce') / 1000.0
    pair = pd.DataFrame({'external_w': a, 'internal_w': b}).dropna()
    if len(pair) < 2:
        return None
    diff = pair['internal_w'] - pair['external_w']
    mean_pair = (pair['internal_w'] + pair['external_w']) / 2.0
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    bias = float(np.mean(diff))
    corr = float(pair['external_w'].corr(pair['internal_w']))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    slope, intercept = np.polyfit(pair['external_w'].to_numpy(dtype=float), pair['internal_w'].to_numpy(dtype=float), 1)
    return {
        'n_pairs': int(len(pair)),
        'pearson_r': corr,
        'mae_w': mae,
        'rmse_w': rmse,
        'mean_bias_w_internal_minus_external': bias,
        'bland_altman_mean_diff_w': bias,
        'bland_altman_lower_w': float(bias - 1.96 * sd),
        'bland_altman_upper_w': float(bias + 1.96 * sd),
        'linear_fit_slope_internal_from_external': float(slope),
        'linear_fit_intercept_internal_from_external': float(intercept),
        'mean_of_pair_w': float(mean_pair.mean()),
    }


def _build_base_timeline(
    *,
    ext: pd.DataFrame,
    hwmon_pivot: pd.DataFrame,
    thermal_pivot: pd.DataFrame,
    tstats_df: pd.DataFrame,
    process_df: pd.DataFrame,
    start_ns: int,
    end_ns: int,
) -> pd.DataFrame:
    for candidate in (ext, hwmon_pivot, thermal_pivot, tstats_df, process_df):
        if not candidate.empty and 'time_ns' in candidate.columns:
            return candidate.copy()
    return pd.DataFrame({'time_ns': [start_ns, end_ns]})


def _numeric_series(df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, np.ndarray]:
    if df.empty or time_col not in df.columns or value_col not in df.columns:
        return {'time_ns': np.array([], dtype=np.float64), 'value': np.array([], dtype=np.float64)}
    out = pd.DataFrame(
        {
            'time_ns': pd.to_numeric(df[time_col], errors='coerce'),
            'value': pd.to_numeric(df[value_col], errors='coerce'),
        }
    ).dropna(subset=['time_ns', 'value'])
    if out.empty:
        return {'time_ns': np.array([], dtype=np.float64), 'value': np.array([], dtype=np.float64)}
    out = out.sort_values('time_ns').drop_duplicates(subset=['time_ns'], keep='last').reset_index(drop=True)
    return {
        'time_ns': out['time_ns'].to_numpy(dtype=np.float64),
        'value': out['value'].to_numpy(dtype=np.float64),
    }


def _interp_value(series_df: Dict[str, np.ndarray], target_ns: int) -> Tuple[Optional[float], bool]:
    times = series_df.get('time_ns')
    values = series_df.get('value')
    if times is None or values is None or len(times) == 0:
        return None, False
    if len(times) == 1:
        return float(values[0]), False
    if target_ns < times[0] or target_ns > times[-1]:
        return None, False
    idx = np.searchsorted(times, float(target_ns))
    if idx < len(times) and times[idx] == float(target_ns):
        return float(values[idx]), False
    prev_idx = max(0, idx - 1)
    next_idx = min(len(times) - 1, idx)
    if prev_idx == next_idx:
        return float(values[prev_idx]), False
    x0 = times[prev_idx]
    x1 = times[next_idx]
    y0 = values[prev_idx]
    y1 = values[next_idx]
    if x1 == x0:
        return float(y0), False
    frac = (float(target_ns) - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0)), True


def _interval_numeric_stats(series_df: Dict[str, np.ndarray], start_ns: int, end_ns: int, *, integrate: bool = False, scale: float = 1.0) -> Dict[str, object]:
    times = series_df.get('time_ns')
    values = series_df.get('value')
    if times is None or values is None or len(times) == 0 or end_ns <= start_ns:
        return {}
    start_idx = int(np.searchsorted(times, float(start_ns), side='left'))
    end_idx = int(np.searchsorted(times, float(end_ns), side='right'))
    inside_times = times[start_idx:end_idx]
    inside_vals = values[start_idx:end_idx]
    start_val, start_interp = _interp_value(series_df, start_ns)
    end_val, end_interp = _interp_value(series_df, end_ns)
    mid_ns = start_ns + int((end_ns - start_ns) / 2)
    mid_val, mid_interp = _interp_value(series_df, mid_ns)

    points: List[Tuple[int, float]] = []
    if start_val is not None:
        points.append((int(start_ns), float(start_val)))
    for idx in range(len(inside_times)):
        points.append((int(inside_times[idx]), float(inside_vals[idx])))
    if end_val is not None:
        points.append((int(end_ns), float(end_val)))
    if not points:
        return {}

    dedup: Dict[int, float] = {}
    for ts, val in points:
        dedup[ts] = val
    ordered = sorted(dedup.items(), key=lambda item: item[0])
    point_times = np.array([ts for ts, _ in ordered], dtype=np.float64)
    point_vals = np.array([val for _, val in ordered], dtype=np.float64)
    duration_s = max((end_ns - start_ns) / 1e9, 1e-12)

    out: Dict[str, object] = {
        'sample_count': int(len(inside_times)),
        'point_count': int(len(point_vals)),
        'start_est': float(start_val) * scale if start_val is not None else None,
        'end_est': float(end_val) * scale if end_val is not None else None,
        'mid_est': float(mid_val) * scale if mid_val is not None else None,
        'min_est': float(np.min(point_vals)) * scale,
        'max_est': float(np.max(point_vals)) * scale,
        'interp_used': bool(start_interp or end_interp or mid_interp),
    }
    if len(point_vals) == 1:
        mean_val = float(point_vals[0])
        integral = mean_val * duration_s
    else:
        integral = float(np.trapz(point_vals, point_times / 1e9))
        mean_val = integral / duration_s
    out['mean_est'] = mean_val * scale
    if integrate:
        out['integral_est'] = integral * scale
    return out


def _expand_interval_stats(row: Dict[str, object], prefix: str, stats: Dict[str, object], *, integral_key: Optional[str] = None) -> None:
    row[f'{prefix}_sample_count'] = stats.get('sample_count')
    row[f'{prefix}_point_count'] = stats.get('point_count')
    row[f'{prefix}_interp_used'] = stats.get('interp_used')
    row[f'{prefix}_start_est'] = stats.get('start_est')
    row[f'{prefix}_mid_est'] = stats.get('mid_est')
    row[f'{prefix}_end_est'] = stats.get('end_est')
    row[f'{prefix}_mean_est'] = stats.get('mean_est')
    row[f'{prefix}_min_est'] = stats.get('min_est')
    row[f'{prefix}_max_est'] = stats.get('max_est')
    if integral_key is not None:
        row[integral_key] = stats.get('integral_est')


def _build_per_image_metrics(
    *,
    repeat_root: Path,
    ext: pd.DataFrame,
    hwmon_pivot: pd.DataFrame,
    thermal_pivot: pd.DataFrame,
    tstats_df: pd.DataFrame,
    process_df: pd.DataFrame,
) -> Optional[Path]:
    timing_csv = repeat_root / 'per_image_latency.csv'
    if not timing_csv.exists():
        return None

    timing_df = pd.read_csv(timing_csv)
    if timing_df.empty:
        out = repeat_root / 'per_image_metrics.csv'
        timing_df.to_csv(out, index=False)
        return out

    ext_power = _numeric_series(ext, 'time_ns', 'ext_power_w')
    ext_temp = _numeric_series(ext, 'time_ns', 'ext_external_temp_c')
    int_total_power = _numeric_series(hwmon_pivot, 'time_ns', 'int_total_power_mW_sum')
    internal_temp_max = _numeric_series(thermal_pivot, 'time_ns', 'internal_temp_max_c')
    internal_temp_mean = _numeric_series(thermal_pivot, 'time_ns', 'internal_temp_mean_c')
    tstats_temp = _numeric_series(tstats_df, 'time_ns', 'tstats_max_temp_c')
    ram_used = _numeric_series(tstats_df, 'time_ns', 'ram_used_mb')
    ram_total = _numeric_series(tstats_df, 'time_ns', 'ram_total_mb')
    swap_used = _numeric_series(tstats_df, 'time_ns', 'swap_used_mb')
    lfb_total = _numeric_series(tstats_df, 'time_ns', 'lfb_total_mb')
    gr3d = _numeric_series(tstats_df, 'time_ns', 'gr3d_pct')
    emc = _numeric_series(tstats_df, 'time_ns', 'emc_pct')
    proc_rss = _numeric_series(process_df, 'time_ns', 'proc_rss_mb')
    proc_uss = _numeric_series(process_df, 'time_ns', 'proc_uss_mb')
    proc_pss = _numeric_series(process_df, 'time_ns', 'proc_pss_mb')
    proc_swap = _numeric_series(process_df, 'time_ns', 'proc_swap_mb')
    proc_pid_count = _numeric_series(process_df, 'time_ns', 'proc_pid_count')

    rows: List[Dict[str, object]] = []
    for item in timing_df.to_dict(orient='records'):
        start_ns = int(item['wallclock_ns_start'])
        end_ns = int(item['wallclock_ns_end'])
        out_row = dict(item)
        out_row['duration_ms'] = (end_ns - start_ns) / 1e6

        ext_power_stats = _interval_numeric_stats(ext_power, start_ns, end_ns, integrate=True)
        int_power_stats = _interval_numeric_stats(int_total_power, start_ns, end_ns, integrate=True, scale=1.0 / 1000.0)
        _expand_interval_stats(out_row, 'ext_power_w', ext_power_stats, integral_key='ext_energy_j_est')
        _expand_interval_stats(out_row, 'int_power_w', int_power_stats, integral_key='int_energy_j_est')
        if out_row.get('int_energy_j_est') is not None and out_row.get('ext_energy_j_est') is not None:
            out_row['int_minus_ext_energy_j_est'] = float(out_row['int_energy_j_est']) - float(out_row['ext_energy_j_est'])
        if out_row.get('int_power_w_mean_est') is not None and out_row.get('ext_power_w_mean_est') is not None:
            out_row['int_minus_ext_power_w_mean_est'] = float(out_row['int_power_w_mean_est']) - float(out_row['ext_power_w_mean_est'])

        _expand_interval_stats(out_row, 'ext_temp_c', _interval_numeric_stats(ext_temp, start_ns, end_ns))
        _expand_interval_stats(out_row, 'internal_temp_max_c', _interval_numeric_stats(internal_temp_max, start_ns, end_ns))
        _expand_interval_stats(out_row, 'internal_temp_mean_c', _interval_numeric_stats(internal_temp_mean, start_ns, end_ns))
        _expand_interval_stats(out_row, 'tstats_temp_max_c', _interval_numeric_stats(tstats_temp, start_ns, end_ns))
        _expand_interval_stats(out_row, 'ram_used_mb', _interval_numeric_stats(ram_used, start_ns, end_ns))
        _expand_interval_stats(out_row, 'ram_total_mb', _interval_numeric_stats(ram_total, start_ns, end_ns))
        _expand_interval_stats(out_row, 'swap_used_mb', _interval_numeric_stats(swap_used, start_ns, end_ns))
        _expand_interval_stats(out_row, 'lfb_total_mb', _interval_numeric_stats(lfb_total, start_ns, end_ns))
        _expand_interval_stats(out_row, 'gr3d_pct', _interval_numeric_stats(gr3d, start_ns, end_ns))
        _expand_interval_stats(out_row, 'emc_pct', _interval_numeric_stats(emc, start_ns, end_ns))
        _expand_interval_stats(out_row, 'proc_rss_mb', _interval_numeric_stats(proc_rss, start_ns, end_ns))
        _expand_interval_stats(out_row, 'proc_uss_mb', _interval_numeric_stats(proc_uss, start_ns, end_ns))
        _expand_interval_stats(out_row, 'proc_pss_mb', _interval_numeric_stats(proc_pss, start_ns, end_ns))
        _expand_interval_stats(out_row, 'proc_swap_mb', _interval_numeric_stats(proc_swap, start_ns, end_ns))
        _expand_interval_stats(out_row, 'proc_pid_count', _interval_numeric_stats(proc_pid_count, start_ns, end_ns))
        rows.append(out_row)

    out = repeat_root / 'per_image_metrics.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def merge_repeat(repeat_root: Path, tolerance_ms: int = 150, build_per_image_metrics: bool = True) -> Dict[str, Optional[str]]:
    markers = _load_json(repeat_root / 'run_markers.json')
    start_ns = int(markers['benchmark_start_wallclock_ns'])
    end_ns = int(markers['benchmark_end_wallclock_ns'])

    hwmon_csv = repeat_root / 'internal' / 'internal_hwmon.csv'
    thermal_csv = repeat_root / 'internal' / 'internal_thermal_zones.csv'
    tstats_csv = repeat_root / 'internal' / 'tegrastats_parsed.csv'
    process_csv = repeat_root / 'internal' / 'process_memory.csv'

    try:
        ext_csv = _find_single(repeat_root / 'external', 'external_measurements.csv')
    except FileNotFoundError:
        ext = pd.DataFrame(columns=['time_ns'])
    else:
        ext = pd.read_csv(ext_csv)
        ext['time_ns'] = pd.to_numeric(ext['mid_wallclock_ns'], errors='coerce')
        ext = ext[(ext['time_ns'] >= start_ns) & (ext['time_ns'] <= end_ns)].copy().sort_values('time_ns')
        ext = _rename_external(ext)

    hwmon = pd.read_csv(hwmon_csv) if hwmon_csv.exists() else pd.DataFrame()
    thermal = pd.read_csv(thermal_csv) if thermal_csv.exists() else pd.DataFrame()
    tstats = pd.read_csv(tstats_csv) if tstats_csv.exists() else pd.DataFrame()
    process = pd.read_csv(process_csv) if process_csv.exists() else pd.DataFrame()

    hwmon_pivot = _pivot_hwmon(hwmon)
    thermal_pivot = _pivot_thermal(thermal)
    tstats_df = _load_tegrastats(tstats)
    process_df = _load_process_memory(process)

    if not hwmon_pivot.empty:
        hwmon_pivot = hwmon_pivot[(hwmon_pivot['time_ns'] >= start_ns) & (hwmon_pivot['time_ns'] <= end_ns)]
    if not thermal_pivot.empty:
        thermal_pivot = thermal_pivot[(thermal_pivot['time_ns'] >= start_ns) & (thermal_pivot['time_ns'] <= end_ns)]
    if not tstats_df.empty:
        tstats_df = tstats_df[(tstats_df['time_ns'] >= start_ns) & (tstats_df['time_ns'] <= end_ns)]
    if not process_df.empty:
        process_df = process_df[(process_df['time_ns'] >= start_ns) & (process_df['time_ns'] <= end_ns)]

    merged = _build_base_timeline(
        ext=ext,
        hwmon_pivot=hwmon_pivot,
        thermal_pivot=thermal_pivot,
        tstats_df=tstats_df,
        process_df=process_df,
        start_ns=start_ns,
        end_ns=end_ns,
    ).sort_values('time_ns')
    tol = int(tolerance_ms * 1_000_000)
    if not ext.empty and 'ext_power_w' not in merged.columns:
        merged = pd.merge_asof(merged, ext, on='time_ns', direction='nearest', tolerance=tol)
    if not hwmon_pivot.empty and not hwmon_pivot.equals(merged):
        merged = pd.merge_asof(merged, hwmon_pivot, on='time_ns', direction='nearest', tolerance=tol)
    if not thermal_pivot.empty and not thermal_pivot.equals(merged):
        merged = pd.merge_asof(merged, thermal_pivot, on='time_ns', direction='nearest', tolerance=tol)
    if not tstats_df.empty and not tstats_df.equals(merged):
        merged = pd.merge_asof(merged, tstats_df, on='time_ns', direction='nearest', tolerance=tol)
    if not process_df.empty and not process_df.equals(merged):
        merged = pd.merge_asof(merged, process_df, on='time_ns', direction='nearest', tolerance=tol)

    out = repeat_root / 'merged_trace.csv'
    merged.to_csv(out, index=False)
    per_image_metrics = None
    if build_per_image_metrics:
        per_image_metrics = _build_per_image_metrics(
            repeat_root=repeat_root,
            ext=ext,
            hwmon_pivot=hwmon_pivot,
            thermal_pivot=thermal_pivot,
            tstats_df=tstats_df,
            process_df=process_df,
        )

    agreement = _agreement_metrics(merged)
    if agreement is not None:
        (repeat_root / 'agreement_summary.json').write_text(json.dumps(agreement, indent=2), encoding='utf-8')
    result = {
        'merged_trace': str(out),
        'per_image_metrics': str(per_image_metrics) if per_image_metrics is not None else None,
    }
    _refresh_repeat_summary_artifacts(repeat_root, result)
    return result


def _infer_repeat_precision_label(repeat_summary: Dict[str, object], repeat_root: Path) -> Optional[str]:
    precision_label = repeat_summary.get('precision_label')
    if isinstance(precision_label, str) and precision_label.strip():
        return precision_label

    run_root = repeat_root.parent.parent
    final_summary_path = run_root / 'FINAL_SUMMARY.json'
    if final_summary_path.exists():
        final_summary = _load_json(final_summary_path)
        candidate = ((final_summary.get('model') or {}).get('precision_label'))
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    run_name = run_root.name.lower()
    for candidate in ('fp16', 'fp32', 'int8'):
        if f'_{candidate}_' in run_name:
            return candidate
    return None


def _load_run_final_summary(repeat_root: Path) -> Dict[str, object]:
    run_root = repeat_root.parent.parent
    final_summary_path = run_root / 'FINAL_SUMMARY.json'
    if not final_summary_path.exists():
        return {}
    return _load_json(final_summary_path)


def _refresh_repeat_accuracy_note(repeat_summary: Dict[str, object], repeat_root: Path) -> None:
    notes_raw = repeat_summary.get('notes')
    notes = [str(item) for item in notes_raw] if isinstance(notes_raw, list) else []
    notes = [note for note in notes if not note.startswith('Accuracy backend:')]

    final_summary = _load_run_final_summary(repeat_root)
    final_model = final_summary.get('model', {}) if isinstance(final_summary.get('model', {}), dict) else {}
    accuracy_metrics = final_summary.get('accuracy_metrics', {}) if isinstance(final_summary.get('accuracy_metrics', {}), dict) else {}
    accuracy_backend = str(final_model.get('accuracy_backend') or '').strip()
    accuracy_precision = str(final_model.get('accuracy_precision_label') or final_model.get('precision_label') or '').strip()
    primary_metric_name = None
    primary_metric_value = None
    for candidate_name, candidate_value in [
        ('mAP50_95', accuracy_metrics.get('metrics/mAP50-95(B)', accuracy_metrics.get('metrics/mAP50-95'))),
        ('mIoU', accuracy_metrics.get('metrics/mIoU')),
        ('mDice', accuracy_metrics.get('metrics/mDice')),
        ('pixel_accuracy', accuracy_metrics.get('metrics/pixel_accuracy')),
    ]:
        try:
            if candidate_value is not None:
                primary_metric_name = candidate_name
                primary_metric_value = float(candidate_value)
                break
        except Exception:
            continue
    if accuracy_backend:
        suffix = []
        if accuracy_precision:
            suffix.append(f'precision={accuracy_precision}')
        if primary_metric_name is not None and primary_metric_value is not None:
            suffix.append(f'{primary_metric_name}={primary_metric_value:.3f}')
        note = f'Accuracy backend: precision-specific {accuracy_backend} validation'
        if suffix:
            note += '; ' + '; '.join(suffix)
        note += '.'
        notes.insert(0, note)
    repeat_summary['notes'] = notes


def _refresh_repeat_summary_artifacts(repeat_root: Path, merge_result: Dict[str, Optional[str]]) -> None:
    repeat_summary_path = repeat_root / 'repeat_summary.json'
    if not repeat_summary_path.exists():
        return

    repeat_summary = _load_json(repeat_summary_path)
    final_summary = _load_run_final_summary(repeat_root)
    final_model = final_summary.get('model', {}) if isinstance(final_summary.get('model', {}), dict) else {}
    final_dataset = final_summary.get('dataset', {}) if isinstance(final_summary.get('dataset', {}), dict) else {}
    files = repeat_summary.get('files')
    if not isinstance(files, dict):
        files = {}
        repeat_summary['files'] = files

    merged_trace = merge_result.get('merged_trace')
    per_image_metrics = merge_result.get('per_image_metrics')
    agreement_path = repeat_root / 'agreement_summary.json'
    external_csvs = sorted(str(p) for p in (repeat_root / 'external').rglob('external_measurements.csv'))
    external_archives = sorted(str(p) for p in (repeat_root / 'external').rglob('*.tar.gz'))

    repeat_summary['precision_label'] = _infer_repeat_precision_label(repeat_summary, repeat_root)
    repeat_summary['model_path'] = final_model.get('measured_model_path')
    repeat_summary['dataset_root'] = final_dataset.get('root')
    repeat_summary['manifest'] = final_dataset.get('manifest') or final_dataset.get('accuracy_manifest') or final_dataset.get('benchmark_manifest')
    repeat_summary['measured_model_path'] = final_model.get('measured_model_path')
    repeat_summary['measured_model_sha256'] = final_model.get('measured_model_sha256')
    repeat_summary['accuracy_backend'] = final_model.get('accuracy_backend')
    repeat_summary['accuracy_precision_label'] = final_model.get('accuracy_precision_label')
    repeat_summary['merged_trace_csv'] = merged_trace
    repeat_summary['per_image_metrics_csv'] = per_image_metrics
    repeat_summary['agreement_summary_path'] = str(agreement_path) if agreement_path.exists() else None
    repeat_summary['external_artifact_path'] = external_archives[0] if external_archives else None
    repeat_summary['external_artifact_paths'] = external_archives or None
    _refresh_repeat_accuracy_note(repeat_summary, repeat_root)

    files['merged_trace_csv'] = merged_trace
    files['per_image_metrics_csv'] = per_image_metrics
    files['agreement_summary_json'] = str(agreement_path) if agreement_path.exists() else None
    files['external_csv'] = external_csvs[0] if external_csvs else files.get('external_csv')
    files['external_archive_path'] = external_archives[0] if external_archives else files.get('external_archive_path')
    files['external_archive_paths'] = external_archives or files.get('external_archive_paths')
    files['measured_model_path'] = final_model.get('measured_model_path')
    files['measured_model_sha256'] = final_model.get('measured_model_sha256')

    _dump_json(repeat_summary_path, repeat_summary)


def _refresh_final_summary_clean_metadata(run_root: Path) -> None:
    final_summary_path = run_root / 'FINAL_SUMMARY.json'
    if not final_summary_path.exists():
        return

    final_summary = _load_json(final_summary_path)
    aggregate_mean = final_summary.get('aggregate_mean') or {}
    external_summary = final_summary.get('external_summary') or {}

    final_summary['performance'] = {
        'images_benchmarked': aggregate_mean.get('images_benchmarked'),
        'benchmark_duration_s': aggregate_mean.get('benchmark_duration_s'),
        'warmup_elapsed_s': aggregate_mean.get('warmup_elapsed_s'),
        'latency_median_ms': aggregate_mean.get('latency_median_ms'),
        'latency_p95_ms': aggregate_mean.get('latency_p95_ms'),
        'latency_mean_ms': aggregate_mean.get('latency_mean_ms'),
        'throughput_fps': aggregate_mean.get('throughput_fps'),
    }
    final_summary['power'] = {
        'avg_power_external_w': aggregate_mean.get('avg_power_external_w', external_summary.get('avg_power_w')),
        'peak_power_external_w': aggregate_mean.get('peak_power_external_w', external_summary.get('peak_power_w')),
        'energy_joules_external': aggregate_mean.get('energy_joules_external', external_summary.get('energy_joules')),
        'energy_wh_external': aggregate_mean.get('energy_wh_external', external_summary.get('energy_wh')),
        'joules_per_image_external': aggregate_mean.get('joules_per_image_external', external_summary.get('joules_per_image')),
        'external_temp_mean_c': aggregate_mean.get('external_temp_mean_c', external_summary.get('external_temp_mean_c')),
        'external_temp_max_c': aggregate_mean.get('external_temp_max_c', external_summary.get('external_temp_max_c')),
        'internal_total_power_mean_w': aggregate_mean.get('internal_total_power_mean_w'),
        'internal_total_power_peak_w': aggregate_mean.get('internal_total_power_peak_w'),
        'internal_temp_mean_c': aggregate_mean.get('internal_temp_mean_c'),
        'internal_temp_max_c': aggregate_mean.get('internal_temp_max_c'),
    }

    files = final_summary.get('files')
    if not isinstance(files, dict):
        files = {}
        final_summary['files'] = files
    analysis_summary = run_root / 'analysis_repeat_summary.csv'
    if analysis_summary.exists():
        files['analysis_repeat_summary_csv'] = str(analysis_summary)
    agreement_summary = run_root / 'analysis_agreement_summary.csv'
    if agreement_summary.exists():
        files['analysis_agreement_summary_csv'] = str(agreement_summary)
    overall_agreement = run_root / 'analysis_agreement_overall.json'
    if overall_agreement.exists():
        files['analysis_agreement_overall_json'] = str(overall_agreement)

    _dump_json(final_summary_path, final_summary)


def summarize_run(run_root: Path) -> Path:
    rows: List[Dict[str, object]] = []
    agreement_rows: List[Dict[str, object]] = []
    for repeat_root in sorted((run_root / 'benchmark_runs').glob('repeat_*')):
        repeat_summary = _load_json(repeat_root / 'repeat_summary.json')
        rows.append(
            {
                'repeat': repeat_summary['repeat'],
                'images_benchmarked': repeat_summary.get('images_benchmarked'),
                'benchmark_duration_s': (repeat_summary.get('run_markers', {}).get('benchmark_end_wallclock_ns', 0) - repeat_summary.get('run_markers', {}).get('benchmark_start_wallclock_ns', 0)) / 1e9,
                'median_latency_ms': repeat_summary.get('latency', {}).get('median_ms'),
                'p95_latency_ms': repeat_summary.get('latency', {}).get('p95_ms'),
                'throughput_fps': repeat_summary.get('throughput_fps'),
                'avg_power_external_w': repeat_summary.get('external_power', {}).get('avg_power_w'),
                'energy_external_j': repeat_summary.get('external_power', {}).get('energy_joules'),
                'joules_per_image_external': repeat_summary.get('external_power', {}).get('joules_per_image'),
                'external_temp_max_c': repeat_summary.get('external_power', {}).get('external_temp_max_c'),
                'max_internal_temp_c': repeat_summary.get('internal_thermal', {}).get('overall_internal_max_c'),
                'mean_internal_power_w': repeat_summary.get('internal_power', {}).get('mean_total_power_w'),
                'peak_ram_mb': repeat_summary.get('tegrastats', {}).get('ram_used_mb_max'),
                'mean_ram_mb': repeat_summary.get('tegrastats', {}).get('ram_used_mb_mean'),
                'peak_swap_used_mb': repeat_summary.get('tegrastats', {}).get('swap_used_mb_max'),
                'mean_swap_used_mb': repeat_summary.get('tegrastats', {}).get('swap_used_mb_mean'),
                'min_lfb_blocks': repeat_summary.get('tegrastats', {}).get('lfb_blocks_min'),
                'min_lfb_size_mb': repeat_summary.get('tegrastats', {}).get('lfb_size_mb_min'),
                'min_lfb_total_mb': repeat_summary.get('tegrastats', {}).get('lfb_total_mb_min'),
                'mean_lfb_total_mb': repeat_summary.get('tegrastats', {}).get('lfb_total_mb_mean'),
                'peak_process_rss_mb': repeat_summary.get('process_memory', {}).get('rss_mb_max'),
                'mean_process_rss_mb': repeat_summary.get('process_memory', {}).get('rss_mb_mean'),
                'peak_process_uss_mb': repeat_summary.get('process_memory', {}).get('uss_mb_max'),
                'mean_process_uss_mb': repeat_summary.get('process_memory', {}).get('uss_mb_mean'),
                'peak_process_pss_mb': repeat_summary.get('process_memory', {}).get('pss_mb_max'),
                'mean_process_pss_mb': repeat_summary.get('process_memory', {}).get('pss_mb_mean'),
                'peak_process_swap_mb': repeat_summary.get('process_memory', {}).get('swap_mb_max'),
                'mean_process_swap_mb': repeat_summary.get('process_memory', {}).get('swap_mb_mean'),
                'process_tree_pid_count_max': repeat_summary.get('process_memory', {}).get('pid_count_max'),
                'process_tree_root_label': repeat_summary.get('process_memory', {}).get('root_label'),
            }
        )
        agree_path = repeat_root / 'agreement_summary.json'
        if agree_path.exists():
            a = _load_json(agree_path)
            a['repeat'] = repeat_summary['repeat']
            agreement_rows.append(a)
    df = pd.DataFrame(rows)
    out = run_root / 'analysis_repeat_summary.csv'
    df.to_csv(out, index=False)
    if agreement_rows:
        adf = pd.DataFrame(agreement_rows)
        adf.to_csv(run_root / 'analysis_agreement_summary.csv', index=False)
        mean_agreement = adf.mean(numeric_only=True).to_dict()
        (run_root / 'analysis_agreement_overall.json').write_text(json.dumps(mean_agreement, indent=2), encoding='utf-8')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Build aligned traces and compact summaries from a benchmark run.')
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--tolerance-ms', type=int, default=150)
    ap.add_argument('--skip-per-image-metrics', action='store_true')
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    produced = []
    for repeat_root in sorted((run_root / 'benchmark_runs').glob('repeat_*')):
        produced.append(
            merge_repeat(
                repeat_root,
                tolerance_ms=args.tolerance_ms,
                build_per_image_metrics=not args.skip_per_image_metrics,
            )
        )
    summary_csv = summarize_run(run_root)
    _refresh_final_summary_clean_metadata(run_root)
    print(json.dumps({'produced': produced, 'merged_traces': produced, 'summary_csv': str(summary_csv)}, indent=2))


if __name__ == '__main__':
    main()
