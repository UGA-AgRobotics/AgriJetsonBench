from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


def _find_single(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matching {pattern} under {root}')
    return matches[0]


def _agreement(ext: pd.Series, internal_mw: pd.Series) -> Dict[str, float]:
    pair = pd.DataFrame({'external_w': pd.to_numeric(ext, errors='coerce'), 'internal_w': pd.to_numeric(internal_mw, errors='coerce') / 1000.0}).dropna()
    if len(pair) < 2:
        return {}
    diff = pair['internal_w'] - pair['external_w']
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    slope, intercept = np.polyfit(pair['external_w'].to_numpy(dtype=float), pair['internal_w'].to_numpy(dtype=float), 1)
    return {
        'n_pairs': int(len(pair)),
        'pearson_r': float(pair['external_w'].corr(pair['internal_w'])),
        'mae_w': float(np.mean(np.abs(diff))),
        'rmse_w': float(np.sqrt(np.mean(np.square(diff)))),
        'mean_bias_w_internal_minus_external': float(np.mean(diff)),
        'bland_altman_lower_w': float(np.mean(diff) - 1.96 * sd),
        'bland_altman_upper_w': float(np.mean(diff) + 1.96 * sd),
        'linear_fit_slope_internal_from_external': float(slope),
        'linear_fit_intercept_internal_from_external': float(intercept),
    }


def analyze_one(calib_root: Path, tolerance_ms: int = 150) -> Dict[str, object]:
    ext_csv = _find_single(calib_root / 'external', 'external_measurements.csv')
    hw_csv = calib_root / 'internal' / 'internal_hwmon.csv'
    if not hw_csv.exists():
        return {'scenario': calib_root.name, 'error': 'internal_hwmon.csv missing'}
    ext = pd.read_csv(ext_csv)
    ext['time_ns'] = pd.to_numeric(ext['mid_wallclock_ns'], errors='coerce')
    hw = pd.read_csv(hw_csv)
    reported = pd.to_numeric(hw.get('power_mW_reported'), errors='coerce')
    computed = pd.to_numeric(hw.get('power_mW_computed'), errors='coerce')
    hw['power_mW'] = reported.fillna(computed)
    hw['time_ns'] = pd.to_numeric(hw.get('sched_wallclock_ns'), errors='coerce')
    hw = hw.dropna(subset=['time_ns', 'power_mW'])
    if hw.empty:
        return {'scenario': calib_root.name, 'error': 'no internal power samples'}
    total = hw.groupby('time_ns')['power_mW'].sum().reset_index().sort_values('time_ns')
    merged = pd.merge_asof(ext.sort_values('time_ns'), total, on='time_ns', direction='nearest', tolerance=int(tolerance_ms * 1_000_000))
    metrics = _agreement(merged['power_w'], merged['power_mW'])
    metrics['scenario'] = calib_root.name
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description='Analyze calibration scenarios for internal-vs-external agreement.')
    ap.add_argument('--suite-root', required=True)
    ap.add_argument('--tolerance-ms', type=int, default=150)
    args = ap.parse_args()

    suite_root = Path(args.suite_root).expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for scenario_root in sorted([p for p in suite_root.iterdir() if p.is_dir()]):
        try:
            rows.append(analyze_one(scenario_root, tolerance_ms=args.tolerance_ms))
        except Exception as exc:
            rows.append({'scenario': scenario_root.name, 'error': str(exc)})
    df = pd.DataFrame(rows)
    out_csv = suite_root / 'calibration_agreement_summary.csv'
    df.to_csv(out_csv, index=False)
    out_json = suite_root / 'calibration_agreement_summary.json'
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps({'summary_csv': str(out_csv), 'summary_json': str(out_json)}, indent=2))


if __name__ == '__main__':
    main()
