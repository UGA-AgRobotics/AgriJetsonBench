#!/usr/bin/env python3
"""Build AgriJetsonBench time-aligned agreement analysis and paper figures.

Outputs:
- paper_tables/internal_external_agreement_summary.csv (time-aligned 1 Hz)
- paper_tables/internal_external_agreement_time_aligned_summary.csv
- paper_tables/internal_external_time_aligned_samples_1hz.csv
- paper_tables/deployment_decision_map.csv
- paper_figures/*.png and *.pdf
- supplementary copies
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "paper_tables"
PF = ROOT / "paper_figures"
SUP = ROOT / "supplementary"
PROJECT_ZIPS: Dict[str, Path] = {}
AGX_INTERNAL_ZIP: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-level internal-external agreement tables and "
            "decision-map figures from curated AGX Orin and Orin Nano Super inputs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing paper_tables, paper_figures, and supplementary.",
    )
    parser.add_argument(
        "--agx-project-zip",
        type=Path,
        required=True,
        help="Curated AGX project-progress archive containing external logger traces.",
    )
    parser.add_argument(
        "--nano-project-zip",
        type=Path,
        required=True,
        help="Curated Orin Nano Super project-progress archive containing external and internal traces.",
    )
    parser.add_argument(
        "--agx-internal-zip",
        type=Path,
        required=True,
        help="Curated AGX internal telemetry archive for the main battery-continuation runs.",
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global ROOT, PT, PF, SUP, PROJECT_ZIPS, AGX_INTERNAL_ZIP

    ROOT = args.root.expanduser().resolve()
    PT = ROOT / "paper_tables"
    PF = ROOT / "paper_figures"
    SUP = ROOT / "supplementary"
    PROJECT_ZIPS = {
        "agx": args.agx_project_zip.expanduser().resolve(),
        "nano_super": args.nano_project_zip.expanduser().resolve(),
    }
    AGX_INTERNAL_ZIP = args.agx_internal_zip.expanduser().resolve()

    missing = []
    for label, path in [
        ("--agx-project-zip", PROJECT_ZIPS["agx"]),
        ("--nano-project-zip", PROJECT_ZIPS["nano_super"]),
        ("--agx-internal-zip", AGX_INTERNAL_ZIP),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required input archive(s). Provide valid paths for:\n" + "\n".join(missing)
        )

    PT.mkdir(parents=True, exist_ok=True)
    PF.mkdir(parents=True, exist_ok=True)
    SUP.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def load_main_battery() -> pd.DataFrame:
    df = pd.read_csv(PT / 'main_battery_crossboard_results.csv')
    # normalize possible older columns
    for c in ['board', 'run_id', 'model', 'precision', 'power_mode_label']:
        df[c] = df[c].astype(str)
    return df


def read_external_trace(run_id: str, board: str) -> Optional[pd.Series]:
    """Read external logger trace and return 1Hz mean power series."""
    zpath = PROJECT_ZIPS.get(board)
    if not zpath or not zpath.exists():
        return None
    prefix = f"project_progress_{'agx' if board == 'agx' else 'nano'}/paper_assets/battery/runs/{run_id}/external_logger/"
    try:
        with zipfile.ZipFile(zpath) as zz:
            candidates = [n for n in zz.namelist() if n.startswith(prefix) and n.endswith('.tar.gz')]
            if not candidates:
                # fallback contains run_id and external_logger
                candidates = [n for n in zz.namelist() if run_id in n and 'external_logger' in n and n.endswith('.tar.gz')]
            if not candidates:
                return None
            tar_bytes = zz.read(candidates[0])
    except Exception as e:
        print(f"external read failed for {run_id}: {e}")
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith('external_measurements.csv')), None)
            if member is None:
                return None
            b = tf.extractfile(member).read()
            df = pd.read_csv(io.BytesIO(b))
    except Exception as e:
        print(f"external tar failed for {run_id}: {e}")
        return None
    if 'timestamp_utc' not in df or 'power_w' not in df:
        return None
    t = pd.to_datetime(df['timestamp_utc'], utc=True, errors='coerce')
    s = pd.Series(pd.to_numeric(df['power_w'], errors='coerce').to_numpy(), index=t)
    s = s.dropna().sort_index()
    if s.empty:
        return None
    # Average all samples in each 1 s bin.
    return s.resample("1s").mean().dropna().rename('external_power_w')


def read_nano_internal_trace(run_id: str) -> Optional[pd.Series]:
    zpath = PROJECT_ZIPS['nano_super']
    path = f'project_progress_nano/paper_assets/battery/runs/{run_id}/internal_telemetry.csv'
    try:
        with zipfile.ZipFile(zpath) as zz:
            if path not in zz.namelist():
                # fallback
                cand = [n for n in zz.namelist() if run_id in n and n.endswith('internal_telemetry.csv')]
                if not cand:
                    return None
                path = cand[0]
            b = zz.read(path)
            df = pd.read_csv(io.BytesIO(b))
    except Exception as e:
        print(f"nano internal read failed for {run_id}: {e}")
        return None
    tcol = 'timestamp_wall' if 'timestamp_wall' in df.columns else 'timestamp_utc'
    pcol = 'internal_total_power_w' if 'internal_total_power_w' in df.columns else None
    if pcol is None or tcol not in df.columns:
        return None
    t = pd.to_datetime(df[tcol], utc=True, errors='coerce')
    s = pd.Series(pd.to_numeric(df[pcol], errors='coerce').to_numpy(), index=t)
    s = s.dropna().sort_index()
    if s.empty:
        return None
    return s.resample("1s").mean().dropna().rename('internal_power_w')


def read_agx_internal_trace(run_id: str) -> Optional[pd.Series]:
    """Read AGX internal hwmon traces and sum internal rails per timestamp.

    This uses internal_hwmon.csv files and sums power_mW_computed or reported across
    rail rows at each timestamp. It is a diagnostic rail-level estimate, not a
    board-input reference.
    """
    if AGX_INTERNAL_ZIP is None or not AGX_INTERNAL_ZIP.exists():
        return None
    series_chunks = []
    with zipfile.ZipFile(AGX_INTERNAL_ZIP) as zz:
        names = [n for n in zz.namelist() if run_id in n and n.endswith('internal_hwmon.csv')]
        if not names:
            return None
        # Process incrementally to avoid memory spikes.
        for n in names:
            try:
                df = pd.read_csv(io.BytesIO(zz.read(n)), usecols=lambda c: c in {
                    'timestamp_utc', 'power_mW_reported', 'power_mW_computed'
                })
            except Exception:
                continue
            if 'timestamp_utc' not in df.columns:
                continue
            p = None
            if 'power_mW_computed' in df.columns and 'power_mW_reported' in df.columns:
                p = pd.to_numeric(df['power_mW_computed'], errors='coerce').fillna(
                    pd.to_numeric(df['power_mW_reported'], errors='coerce'))
            elif 'power_mW_computed' in df.columns:
                p = pd.to_numeric(df['power_mW_computed'], errors='coerce')
            elif 'power_mW_reported' in df.columns:
                p = pd.to_numeric(df['power_mW_reported'], errors='coerce')
            else:
                continue
            tmp = pd.DataFrame({'timestamp_utc': df['timestamp_utc'], 'p_mw': p})
            tmp = tmp.dropna(subset=['p_mw'])
            if tmp.empty:
                continue
            # Sum rails at exact timestamp.
            g = tmp.groupby('timestamp_utc', as_index=False)['p_mw'].sum()
            t = pd.to_datetime(g['timestamp_utc'], utc=True, errors='coerce')
            s = pd.Series(g['p_mw'].to_numpy(dtype=float) / 1000.0, index=t)
            s = s.dropna().sort_index()
            if not s.empty:
                # Downsample this chunk first.
                series_chunks.append(s.resample("1s").mean().dropna())
    if not series_chunks:
        return None
    all_s = pd.concat(series_chunks).sort_index()
    # Multiple loop files may share seconds; average within second.
    all_s = all_s.groupby(all_s.index).mean().sort_index().rename('internal_power_w')
    return all_s


def load_internal_trace(run_id: str, board: str) -> Optional[pd.Series]:
    if board == 'nano_super':
        return read_nano_internal_trace(run_id)
    if board == 'agx':
        return read_agx_internal_trace(run_id)
    return None


def align_external_internal(ext: pd.Series, internal: pd.Series) -> pd.DataFrame:
    # Join on 1Hz bins; preserve only overlap.
    df = pd.concat([ext, internal], axis=1).dropna().sort_index()
    # Very light smoothing can distort agreement; keep raw 1Hz means.
    return df


def agreement_metrics(aligned: pd.DataFrame, run_row: pd.Series) -> Dict:
    ext = aligned['external_power_w'].astype(float)
    it = aligned['internal_power_w'].astype(float)
    diff = it - ext
    mean_power = (it + ext) / 2
    n = len(aligned)
    duration_s = float((aligned.index.max() - aligned.index.min()).total_seconds() + 1) if n > 1 else float(n)
    mae = float(diff.abs().mean()) if n else np.nan
    rmse = float(np.sqrt((diff ** 2).mean())) if n else np.nan
    mape = float((diff.abs() / ext.replace(0, np.nan)).mean() * 100) if n else np.nan
    bias_w = float(diff.mean()) if n else np.nan
    bias_pct = float(bias_w / ext.mean() * 100) if n and ext.mean() else np.nan
    sd_diff = float(diff.std(ddof=1)) if n > 1 else np.nan
    pearson = float(ext.corr(it, method='pearson')) if n > 2 else np.nan
    spearman = float(ext.corr(it, method='spearman')) if n > 2 else np.nan
    ext_wh = float(ext.sum() / 3600.0)
    int_wh = float(it.sum() / 3600.0)
    energy_bias_pct = float((int_wh - ext_wh) / ext_wh * 100) if ext_wh else np.nan
    return {
        'run_id': run_row['run_id'],
        'board': run_row['board'],
        'model': run_row['model'],
        'precision': run_row['precision'],
        'power_mode_label': run_row['power_mode_label'],
        'paper_use': run_row.get('paper_use', ''),
        'analysis_type': 'time_aligned_1hz_overlap',
        'aligned_samples_1hz': n,
        'overlap_duration_seconds': duration_s,
        'external_trace_duration_seconds': float(len(ext)),
        'mean_external_power_w_overlap': float(ext.mean()) if n else np.nan,
        'mean_internal_power_w_overlap': float(it.mean()) if n else np.nan,
        'bias_internal_minus_external_w': bias_w,
        'bias_internal_minus_external_percent': bias_pct,
        'mae_w': mae,
        'rmse_w': rmse,
        'mape_percent': mape,
        'pearson_r': pearson,
        'spearman_r': spearman,
        'bland_altman_mean_bias_w': bias_w,
        'bland_altman_sd_diff_w': sd_diff,
        'bland_altman_lower_loa_w': bias_w - 1.96 * sd_diff if np.isfinite(sd_diff) else np.nan,
        'bland_altman_upper_loa_w': bias_w + 1.96 * sd_diff if np.isfinite(sd_diff) else np.nan,
        'external_energy_wh_overlap': ext_wh,
        'internal_energy_wh_overlap': int_wh,
        'energy_bias_percent_overlap': energy_bias_pct,
        'time_start_utc': aligned.index.min().isoformat() if n else '',
        'time_stop_utc': aligned.index.max().isoformat() if n else '',
        'alignment_note': 'External inline logger is the primary board-input reference; internal telemetry is diagnostic and rail-dependent.'
    }


def build_agreement_tables(main: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    sample_rows = []
    for _, row in main.iterrows():
        run_id = row['run_id']
        board = row['board']
        print(f"Aligning {run_id} ...")
        ext = read_external_trace(run_id, board)
        it = load_internal_trace(run_id, board)
        if ext is None or it is None or len(ext) == 0 or len(it) == 0:
            summary_rows.append({
                'run_id': run_id,
                'board': board,
                'model': row['model'],
                'precision': row['precision'],
                'power_mode_label': row['power_mode_label'],
                'paper_use': row.get('paper_use', ''),
                'analysis_type': 'time_aligned_1hz_overlap',
                'aligned_samples_1hz': 0,
                'alignment_note': 'Could not load both external and internal traces for time-aligned analysis.'
            })
            continue
        aligned = align_external_internal(ext, it)
        summary_rows.append(agreement_metrics(aligned, row))
        # Keep all points but annotate. This is ~60k rows, fine.
        tmp = aligned.reset_index().rename(columns={'index': 'timestamp_utc'})
        tmp['run_id'] = run_id
        tmp['board'] = board
        tmp['model'] = row['model']
        tmp['precision'] = row['precision']
        tmp['power_mode_label'] = row['power_mode_label']
        tmp['diff_internal_minus_external_w'] = tmp['internal_power_w'] - tmp['external_power_w']
        tmp['mean_power_w'] = (tmp['internal_power_w'] + tmp['external_power_w']) / 2
        sample_rows.append(tmp[['run_id','board','model','precision','power_mode_label','timestamp_utc','external_power_w','internal_power_w','diff_internal_minus_external_w','mean_power_w']])
    summary = pd.DataFrame(summary_rows)
    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    # Sort for readability
    if not summary.empty:
        summary = summary.sort_values(['paper_use','board','model','precision','power_mode_label']).reset_index(drop=True)
    return summary, samples


def save_figure(fig, name: str):
    for ext in ['png', 'pdf']:
        path = PF / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=220)
    plt.close(fig)


def plot_bland_altman(samples: pd.DataFrame, summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if samples.empty:
        ax.text(0.5, 0.5, 'No aligned samples available', ha='center', va='center')
    else:
        # Downsample deterministically if too many points.
        plot_df = samples.copy()
        max_n = 9000
        if len(plot_df) > max_n:
            plot_df = plot_df.iloc[np.linspace(0, len(plot_df)-1, max_n).astype(int)]
        for (board, mode), g in plot_df.groupby(['board','power_mode_label']):
            ax.scatter(g['mean_power_w'], g['diff_internal_minus_external_w'], s=8, alpha=0.35, label=f'{board} {mode}')
        diff_all = samples['diff_internal_minus_external_w'].dropna()
        if len(diff_all) > 1:
            bias = diff_all.mean()
            sd = diff_all.std(ddof=1)
            ax.axhline(bias, linestyle='-', linewidth=1.4, label=f'overall bias = {bias:.2f} W')
            ax.axhline(bias + 1.96*sd, linestyle='--', linewidth=1.0, label=f'+1.96 SD = {bias + 1.96*sd:.2f} W')
            ax.axhline(bias - 1.96*sd, linestyle='--', linewidth=1.0, label=f'-1.96 SD = {bias - 1.96*sd:.2f} W')
        ax.set_xlabel('Mean of external and internal power (W)')
        ax.set_ylabel('Internal telemetry - external logger (W)')
        ax.set_title('Time-aligned internal-vs-external power agreement (Bland-Altman)')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=True, loc='best')
    fig.tight_layout()
    save_figure(fig, 'Fig_BlandAltman_internal_external_power')


def plot_internal_external_scatter(samples: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    if samples.empty:
        ax.text(0.5, 0.5, 'No aligned samples available', ha='center', va='center')
    else:
        plot_df = samples.copy()
        max_n = 9000
        if len(plot_df) > max_n:
            plot_df = plot_df.iloc[np.linspace(0, len(plot_df)-1, max_n).astype(int)]
        for (board, mode), g in plot_df.groupby(['board','power_mode_label']):
            ax.scatter(g['external_power_w'], g['internal_power_w'], s=8, alpha=0.35, label=f'{board} {mode}')
        low = min(plot_df['external_power_w'].min(), plot_df['internal_power_w'].min())
        high = max(plot_df['external_power_w'].max(), plot_df['internal_power_w'].max())
        ax.plot([low, high], [low, high], linewidth=1.2, linestyle='--', label='1:1 line')
        ax.set_xlabel('External board-input power (W)')
        ax.set_ylabel('Internal telemetry power (W)')
        ax.set_title('Time-aligned external vs internal power')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=True, loc='best')
    fig.tight_layout()
    save_figure(fig, 'Fig_Internal_vs_External_Power_Scatter')


def plot_pareto(main: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 6.1))
    markers = {'INT8': 'o', 'FP16': 's', 'FP32': '^'}
    for _, r in main.iterrows():
        size = 40 + 10 * math.sqrt(max(safe_float(r.get('gmacs_per_inference')), 1))
        label = f"{r['board']} {r['power_mode_label']} {r['precision']}"
        ax.scatter(r['j_per_inference'], r['fps'], marker=markers.get(r['precision'], 'o'), s=size, alpha=0.8)
        ax.annotate(f"{r['model']}\n{r['board']} {r['power_mode_label']}",
                    (r['j_per_inference'], r['fps']), textcoords='offset points', xytext=(5,5), fontsize=7)
    ax.set_xlabel('External energy per inference (J/inference)')
    ax.set_ylabel('Throughput (FPS)')
    ax.set_title('AgriJetsonBench battery Pareto: FPS vs J/inference')
    ax.grid(True, alpha=0.25)
    # custom legend by precision
    handles = [Line2D([0], [0], marker=m, color='black', linestyle='', label=p) for p,m in markers.items() if p in set(main['precision'])]
    ax.legend(handles=handles, title='Precision', fontsize=8, title_fontsize=8, loc='best')
    fig.tight_layout()
    save_figure(fig, 'Fig_FPS_vs_J_per_inference_Pareto')


def plot_jgmac(main: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    markers = {'INT8': 'o', 'FP16': 's', 'FP32': '^'}
    for _, r in main.iterrows():
        size = 50 + 6 * math.sqrt(max(safe_float(r.get('total_inferences')), 1)) / 20
        ax.scatter(r['gmac_per_s'], r['j_per_gmac'], marker=markers.get(r['precision'], 'o'), s=size, alpha=0.8)
        ax.annotate(f"{r['model']}\n{r['board']} {r['power_mode_label']}",
                    (r['gmac_per_s'], r['j_per_gmac']), textcoords='offset points', xytext=(5,5), fontsize=7)
    ax.set_xlabel('Compute throughput (GMAC/s)')
    ax.set_ylabel('Compute-normalized energy (J/GMAC)')
    ax.set_title('Complexity-aware battery efficiency: J/GMAC vs GMAC/s')
    ax.grid(True, alpha=0.25)
    handles = [Line2D([0], [0], marker=m, color='black', linestyle='', label=p) for p,m in markers.items() if p in set(main['precision'])]
    ax.legend(handles=handles, title='Precision', fontsize=8, title_fontsize=8, loc='best')
    fig.tight_layout()
    save_figure(fig, 'Fig_J_per_GMAC_vs_GMAC_per_s')


def plot_battery_runtime(main: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    labels = [f"{r['board']}\n{r['model']}\n{r['precision']} {r['power_mode_label']}" for _, r in main.iterrows()]
    x = np.arange(len(main))
    bars = ax1.bar(x, main['elapsed_hours'])
    ax1.set_ylabel('Runtime from 100% to 20% power-bank remaining (h)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax1.set_title('Battery continuation summary')
    ax1.grid(axis='y', alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, main['total_inferences'], marker='o', linestyle='--')
    ax2.set_ylabel('Total inferences')
    fig.tight_layout()
    save_figure(fig, 'Fig_Battery_Runtime_Total_Inferences')


def build_decision_map_table(main: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            'deployment_goal': 'Lowest energy and highest throughput for lightweight matched-budget detection',
            'recommended_configuration': 'Nano Super 15W - YOLO11s INT8 640 - pure engine',
            'supporting_result': 'Nano Super achieved higher FPS and lower J/inference than AGX in the 15W YOLO11s INT8 battery run.',
            'paper_message': 'Nano Super is the preferred matched-budget INT8 detector platform.'
        },
        {
            'deployment_goal': 'Matched-budget transformer-style segmentation at INT8',
            'recommended_configuration': 'Nano Super 15W - SegFormer-B0 INT8 640 - pure engine',
            'supporting_result': 'Nano Super achieved higher FPS, lower latency, and lower J/inference than AGX at 15W.',
            'paper_message': 'Nano Super matched-budget advantage generalizes beyond YOLO11s.'
        },
        {
            'deployment_goal': 'Native high-load FP16 segmentation when latency/throughput matter',
            'recommended_configuration': 'AGX Orin 50W - SegFormer-B0 FP16 640 - pure engine',
            'supporting_result': 'AGX achieved substantially higher FPS and lower p95 latency than Nano Super 25W while maintaining comparable J/inference.',
            'paper_message': 'AGX converts higher power into much higher throughput without a large normalized-energy penalty.'
        },
        {
            'deployment_goal': 'Native RT-DETR-R18 FP16 energy contrast',
            'recommended_configuration': 'Nano Super 25W for energy, AGX 50W for latency',
            'supporting_result': 'RT-DETR-R18 shows that faster AGX native operation is not always the energy-efficiency winner.',
            'paper_message': 'Board ranking is model-architecture dependent.'
        },
        {
            'deployment_goal': 'Power/energy reporting in the manuscript',
            'recommended_configuration': 'Use external inline logger as primary board-input reference',
            'supporting_result': 'Internal telemetry is useful but has board- and rail-dependent bias relative to the external logger.',
            'paper_message': 'Dual-source measurement is a methodological contribution, not just an instrumentation detail.'
        },
        {
            'deployment_goal': 'Full application image-pipeline reporting',
            'recommended_configuration': 'Report as supplementary only; do not mix with pure-engine J/inference',
            'supporting_result': 'Pure TensorRT engine boundary and end-to-end image path measure different deployment layers.',
            'paper_message': 'Measurement boundary must be explicit for fair agricultural edge-AI benchmarking.'
        },
    ]
    return pd.DataFrame(rows)


def plot_decision_map(decision: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    ax.axis('off')
    ax.set_title('AgriJetsonBench deployment decision map', fontsize=15, pad=18)
    # Draw two main blocks and notes.
    left_x, right_x = 0.05, 0.53
    top_y = 0.86
    block_w = 0.42
    block_h = 0.26
    blocks = [
        (left_x, top_y, 'Matched-budget 15W INT8', 'Recommended: Nano Super 15W\nYOLO11s INT8 and SegFormer-B0 INT8\nHigher FPS, lower latency, lower J/inference'),
        (right_x, top_y, 'Native high-load FP16', 'Recommended: AGX Orin 50W\nSegFormer-B0 FP16\nHigher FPS and lower p95 latency\nEnergy per inference near-tied with Nano 25W'),
        (left_x, 0.48, 'Contrast case', 'RT-DETR-R18 FP16 native:\nAGX is faster, but Nano remains more energy-efficient.\nUse as architecture-dependent contrast.'),
        (right_x, 0.48, 'Measurement rule', 'Main results use pure TensorRT engine-level boundary.\nExternal logger = primary board-input energy.\nInternal telemetry = diagnostic/agreement channel.'),
    ]
    for x, y, title, body in blocks:
        rect = Rectangle((x, y-block_h), block_w, block_h, fill=False, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x+0.015, y-0.035, title, fontsize=12, fontweight='bold', va='top')
        ax.text(x+0.015, y-0.085, body, fontsize=9.3, va='top', wrap=True)
    ax.text(0.5, 0.17,
            'Central finding: the deployment winner is not fixed by board class; it emerges from model complexity, precision, TensorRT boundary, batch size, and power mode.',
            ha='center', va='center', fontsize=11, fontweight='bold', wrap=True)
    ax.text(0.5, 0.08,
            'Paper wording: Nano Super wins matched-budget INT8; AGX Orin wins native high-load SegFormer-B0 FP16 performance, with comparable normalized energy in the long battery run.',
            ha='center', va='center', fontsize=10, wrap=True)
    fig.tight_layout()
    save_figure(fig, 'Fig_Deployment_Decision_Map')


def write_caption_drafts():
    text = """# Figure caption drafts

**Fig. X. Time-aligned internal-vs-external power agreement (Bland-Altman).** Each point represents a 1 Hz overlap sample between the external inline board-input logger and Jetson internal telemetry. The y-axis shows internal telemetry minus external logger power. The external logger remains the primary board-input energy reference; internal telemetry is interpreted as a diagnostic rail-dependent channel.

**Fig. X. Time-aligned external versus internal power scatter.** External board-input power and internal telemetry were aligned to a common 1 Hz timeline. The 1:1 line is shown for visual reference. Deviations indicate rail coverage differences and board/mode-dependent bias.

**Fig. X. AgriJetsonBench battery Pareto plot.** Throughput is plotted against external energy per TensorRT engine-level inference. Marker annotations identify model, board, and power mode. The plot summarizes the main deployment trade-off: Nano Super dominates the matched-budget INT8 regime, while AGX Orin provides higher native SegFormer-B0 FP16 throughput.

**Fig. X. Complexity-normalized battery efficiency.** Compute throughput (GMAC/s) is plotted against energy per GMAC. This figure contextualizes J/inference using model complexity at 640 x 640 and shows that AGX native SegFormer-B0 FP16 reaches higher GMAC/s while remaining close to Nano Super in J/GMAC.

**Fig. X. AgriJetsonBench deployment decision map.** The map translates benchmark evidence into practical configuration choices for precision-agriculture edge inference. Matched-budget INT8 workloads favor Nano Super, whereas native high-load SegFormer-B0 FP16 favors AGX Orin when latency and throughput are priorities.
"""
    (PF / 'figure_caption_drafts.md').write_text(text, encoding='utf-8')
    (SUP / 'figure_caption_drafts.md').write_text(text, encoding='utf-8')


def update_manifest():
    import hashlib
    rows = []
    for path in ROOT.rglob('*'):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({'relative_path': rel, 'size_bytes': path.stat().st_size, 'sha256': h})
    pd.DataFrame(rows).sort_values('relative_path').to_csv(ROOT / 'MANIFEST.csv', index=False)


def main(args: Optional[argparse.Namespace] = None):
    configure_paths(args or parse_args())
    main_df = load_main_battery()
    summary, samples = build_agreement_tables(main_df)

    # Preserve previous aggregate agreement if it exists.
    old_path = PT / 'internal_external_agreement_summary.csv'
    if old_path.exists():
        backup = PT / 'internal_external_agreement_aggregate_summary_previous.csv'
        if not backup.exists():
            old_path.rename(backup)
    summary.to_csv(PT / 'internal_external_agreement_summary.csv', index=False)
    summary.to_csv(PT / 'internal_external_agreement_time_aligned_summary.csv', index=False)
    summary.to_json(PT / 'internal_external_agreement_time_aligned_summary.json', orient='records', indent=2)
    if not samples.empty:
        samples.to_csv(PT / 'internal_external_time_aligned_samples_1hz.csv', index=False)
        # Supplementary copy but not duplicate huge if too large? This is acceptable.
        samples.to_csv(SUP / 'S9_internal_external_time_aligned_samples_1hz.csv', index=False)
    summary.to_csv(SUP / 'S9_internal_external_agreement_time_aligned_summary.csv', index=False)

    # Plots
    plot_bland_altman(samples, summary)
    plot_internal_external_scatter(samples)
    plot_pareto(main_df)
    plot_jgmac(main_df)
    plot_battery_runtime(main_df)
    decision = build_decision_map_table(main_df)
    decision.to_csv(PT / 'deployment_decision_map.csv', index=False)
    decision.to_csv(SUP / 'S13_deployment_decision_map.csv', index=False)
    plot_decision_map(decision)
    write_caption_drafts()

    # Report
    report = {
        'created_outputs': {
            'tables': [
                'paper_tables/internal_external_agreement_summary.csv',
                'paper_tables/internal_external_agreement_time_aligned_summary.csv',
                'paper_tables/internal_external_time_aligned_samples_1hz.csv',
                'paper_tables/deployment_decision_map.csv',
            ],
            'figures': sorted([p.name for p in PF.glob("Fig_*.png")])
        },
        'agreement_rows': int(len(summary)),
        'aligned_sample_rows': int(len(samples)),
        'notes': 'External inline logger remains the primary board-input energy reference; internal telemetry is diagnostic/rail-dependent.',
    }
    (PT / 'agreement_figures_generation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (SUP / 'agreement_figures_generation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    update_manifest()
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
