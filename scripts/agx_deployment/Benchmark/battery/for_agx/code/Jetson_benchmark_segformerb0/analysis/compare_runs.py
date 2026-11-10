from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


COMPLEXITY_KEYS = (
    'params',
    'macs',
    'gmacs',
    'gflops_if_1mac_eq_2flops',
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _bootstrap(values: List[float], n_boot: int = 5000, seed: int = 12345) -> Dict[str, float]:
    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {}
    if arr.size == 1:
        val = float(arr[0])
        return {'center': val, 'ci_low': val, 'ci_high': val, 'n': 1}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = float(sample.mean())
    return {'center': float(arr.mean()), 'ci_low': float(np.percentile(stats, 2.5)), 'ci_high': float(np.percentile(stats, 97.5)), 'n': int(arr.size)}


def _slugify(value: Optional[str], fallback: str) -> str:
    raw = (value or '').strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '_', raw).strip('_')
    return slug or fallback


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_accuracy_metrics(acc: Dict[str, object]) -> Dict[str, Optional[float]]:
    map50_95 = _to_float(acc.get('metrics/mAP50-95(B)', acc.get('metrics/mAP50-95')))
    miou = _to_float(acc.get('metrics/mIoU', acc.get('metrics/miou', acc.get('miou'))))
    mdice = _to_float(acc.get('metrics/mDice', acc.get('metrics/mdice', acc.get('mdice'))))
    pixel_accuracy = _to_float(acc.get('metrics/pixel_accuracy', acc.get('pixel_accuracy')))
    return {
        'mAP50_95': map50_95,
        'mIoU': miou,
        'mDice': mdice,
        'pixel_accuracy': pixel_accuracy,
        'primary_accuracy': map50_95 if map50_95 is not None else miou,
    }


def _normalize_complexity_payload(raw: object) -> Dict[str, Optional[float]]:
    if not isinstance(raw, dict):
        return {key: None for key in COMPLEXITY_KEYS}

    out: Dict[str, Optional[float]] = {}
    for key in COMPLEXITY_KEYS:
        value = raw.get(key)
        if value is None:
            out[key] = None
            continue
        try:
            if key in {'params', 'macs'}:
                out[key] = int(round(float(value)))
            else:
                out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = None
    return out


def _extract_complexity_metrics(final: Dict[str, object]) -> Dict[str, Optional[float]]:
    model_info = final.get('model', {}) if isinstance(final.get('model', {}), dict) else {}
    files_info = final.get('files', {}) if isinstance(final.get('files', {}), dict) else {}

    direct = _normalize_complexity_payload(model_info.get('complexity'))
    if any(direct.get(key) is not None for key in COMPLEXITY_KEYS):
        return direct

    candidate_paths: List[Path] = []
    complexity_json = files_info.get('complexity_json')
    if complexity_json:
        candidate_paths.append(Path(str(complexity_json)).expanduser().resolve())

    measured_model_path = model_info.get('measured_model_path')
    profile_input = model_info.get('profile_input', {}) if isinstance(model_info.get('profile_input', {}), dict) else {}
    imgsz = profile_input.get('imgsz')
    if measured_model_path:
        model_dir = Path(str(measured_model_path)).expanduser().resolve().parent
        if imgsz is not None:
            try:
                candidate_paths.append(model_dir / f'complexity_{int(imgsz)}.json')
            except (TypeError, ValueError):
                pass
        candidate_paths.extend(sorted(model_dir.glob('complexity_*.json')))

    seen: set[str] = set()
    for path in candidate_paths:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        payload = _normalize_complexity_payload(_load_json(path))
        if any(payload.get(metric) is not None for metric in COMPLEXITY_KEYS):
            return payload

    return {key: None for key in COMPLEXITY_KEYS}


def _pairwise_summary(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, object]:
    left_key = _slugify(str(left.get('precision_label') or left.get('run_label') or ''), 'run_1')
    right_key = _slugify(str(right.get('precision_label') or right.get('run_label') or ''), 'run_2')
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in [
        'primary_accuracy',
        'mAP50_95',
        'mIoU',
        'mDice',
        'pixel_accuracy',
        'median_latency_ms_mean',
        'j_per_image_mean',
        'avg_power_external_w_mean',
        'peak_ram_mb_mean',
        'peak_process_rss_mb_mean',
        'peak_process_uss_mb_mean',
        'peak_process_pss_mb_mean',
        'peak_swap_used_mb_mean',
        'min_lfb_total_mb_mean',
        'max_internal_temp_c_mean',
        'params',
        'macs',
        'gmacs',
        'gflops_if_1mac_eq_2flops',
    ]:
        left_val = _to_float(left.get(metric))
        right_val = _to_float(right.get(metric))
        entry: Dict[str, Optional[float]] = {left_key: left_val, right_key: right_val}
        if left_val is not None and right_val is not None:
            entry[f'{right_key}_minus_{left_key}'] = right_val - left_val
            entry[f'{left_key}_minus_{right_key}'] = left_val - right_val
            entry[f'{right_key}_over_{left_key}'] = (right_val / left_val) if left_val not in (0.0, None) else None
            entry[f'{left_key}_over_{right_key}'] = (left_val / right_val) if right_val not in (0.0, None) else None
        metrics[metric] = entry
    return {
        'pair_label': f'{left_key}_vs_{right_key}',
        'left_run': {
            'run_label': left.get('run_label'),
            'precision_label': left.get('precision_label'),
            'power_mode_label': left.get('power_mode_label'),
            'observed_power_mode_class': left.get('observed_power_mode_class'),
        },
        'right_run': {
            'run_label': right.get('run_label'),
            'precision_label': right.get('precision_label'),
            'power_mode_label': right.get('power_mode_label'),
            'observed_power_mode_class': right.get('observed_power_mode_class'),
        },
        'metrics': metrics,
    }


def _pairwise_summaries(rows: List[Dict[str, object]]) -> Optional[List[Dict[str, object]]]:
    if len(rows) < 2:
        return None
    return [_pairwise_summary(left, right) for left, right in combinations(rows, 2)]


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare multiple benchmark run roots for proposal tables.')
    ap.add_argument('--run-root', action='append', required=True, help='Repeat this flag for each FINAL_SUMMARY run root')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for raw in args.run_root:
        run_root = Path(raw).expanduser().resolve()
        final = _load_json(run_root / 'FINAL_SUMMARY.json')
        agg = pd.read_csv(run_root / 'aggregate_repeat_summary.csv')
        acc = _extract_accuracy_metrics(final.get('accuracy_metrics', {}))
        complexity = _extract_complexity_metrics(final)
        rows.append(
            {
                'run_root': str(run_root),
                'board_name': final.get('project', {}).get('board_name'),
                'run_label': final.get('project', {}).get('run_label'),
                'precision_label': final.get('project', {}).get('precision_label') or final.get('model', {}).get('precision_label'),
                'power_mode_label': final.get('power_mode', {}).get('power_mode_label') if 'power_mode' in final else None,
                'matched_budget_label': final.get('power_mode', {}).get('matched_budget_label') if 'power_mode' in final else None,
                'observed_power_mode_name': final.get('power_mode', {}).get('observed_mode_name') if 'power_mode' in final else None,
                'observed_power_mode_id': final.get('power_mode', {}).get('observed_mode_id') if 'power_mode' in final else None,
                'observed_power_mode_class': final.get('power_mode', {}).get('observed_mode_class') if 'power_mode' in final else None,
                'observed_power_mode_tag': final.get('power_mode', {}).get('observed_mode_tag') if 'power_mode' in final else None,
                'observed_power_budget_w': final.get('power_mode', {}).get('observed_power_budget_w') if 'power_mode' in final else None,
                'measured_model_path': final.get('model', {}).get('measured_model_path'),
                'measured_model_sha256': final.get('model', {}).get('measured_model_sha256'),
                'params': complexity.get('params'),
                'macs': complexity.get('macs'),
                'gmacs': complexity.get('gmacs'),
                'gflops_if_1mac_eq_2flops': complexity.get('gflops_if_1mac_eq_2flops'),
                'primary_accuracy': acc.get('primary_accuracy'),
                'mAP50_95': acc.get('mAP50_95'),
                'mIoU': acc.get('mIoU'),
                'mDice': acc.get('mDice'),
                'pixel_accuracy': acc.get('pixel_accuracy'),
                'median_latency_ms_mean': float(agg['latency_median_ms'].dropna().mean()) if 'latency_median_ms' in agg else None,
                'median_latency_ms_ci_low': _bootstrap(agg['latency_median_ms'].dropna().tolist()).get('ci_low') if 'latency_median_ms' in agg else None,
                'median_latency_ms_ci_high': _bootstrap(agg['latency_median_ms'].dropna().tolist()).get('ci_high') if 'latency_median_ms' in agg else None,
                'j_per_image_mean': float(agg['joules_per_image_external'].dropna().mean()) if 'joules_per_image_external' in agg else None,
                'j_per_image_ci_low': _bootstrap(agg['joules_per_image_external'].dropna().tolist()).get('ci_low') if 'joules_per_image_external' in agg else None,
                'j_per_image_ci_high': _bootstrap(agg['joules_per_image_external'].dropna().tolist()).get('ci_high') if 'joules_per_image_external' in agg else None,
                'avg_power_external_w_mean': float(agg['avg_power_external_w'].dropna().mean()) if 'avg_power_external_w' in agg else None,
                'peak_ram_mb_mean': float(agg['peak_ram_mb'].dropna().mean()) if 'peak_ram_mb' in agg else None,
                'peak_swap_used_mb_mean': float(agg['swap_used_mb_peak'].dropna().mean()) if 'swap_used_mb_peak' in agg else None,
                'min_lfb_total_mb_mean': float(agg['lfb_total_mb_min'].dropna().mean()) if 'lfb_total_mb_min' in agg else None,
                'peak_process_rss_mb_mean': float(agg['peak_process_rss_mb'].dropna().mean()) if 'peak_process_rss_mb' in agg else None,
                'peak_process_uss_mb_mean': float(agg['peak_process_uss_mb'].dropna().mean()) if 'peak_process_uss_mb' in agg else None,
                'peak_process_pss_mb_mean': float(agg['peak_process_pss_mb'].dropna().mean()) if 'peak_process_pss_mb' in agg else None,
                'process_tree_pid_count_max_mean': float(agg['process_tree_pid_count_max'].dropna().mean()) if 'process_tree_pid_count_max' in agg else None,
                'max_internal_temp_c_mean': float(agg['internal_temp_max_c'].dropna().mean()) if 'internal_temp_max_c' in agg else None,
            }
        )
    df = pd.DataFrame(rows)
    out_csv = outdir / 'proposal_comparison_table.csv'
    out_json = outdir / 'proposal_comparison_table.json'
    pairwise_json = outdir / 'proposal_pairwise_summary.json'
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    pairwise = _pairwise_summaries(rows)
    if pairwise is not None:
        pairwise_json.write_text(json.dumps(pairwise, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {
                'comparison_csv': str(out_csv),
                'comparison_json': str(out_json),
                'pairwise_json': str(pairwise_json) if pairwise is not None else None,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
