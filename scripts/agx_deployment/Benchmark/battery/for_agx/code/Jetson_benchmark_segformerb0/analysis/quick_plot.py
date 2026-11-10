from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description='Quick plots for a merged proposal trace.')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'time_ns' not in df.columns:
        raise RuntimeError('merged trace must contain time_ns column')
    t = (pd.to_numeric(df['time_ns'], errors='coerce') - pd.to_numeric(df['time_ns'], errors='coerce').min()) / 1e9

    if 'ext_power_w' in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(t, pd.to_numeric(df['ext_power_w'], errors='coerce'))
        if 'int_total_power_mW_sum' in df.columns:
            plt.plot(t, pd.to_numeric(df['int_total_power_mW_sum'], errors='coerce') / 1000.0)
            plt.legend(['External board-input power', 'Internal summed rail power'])
        else:
            plt.legend(['External board-input power'])
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.tight_layout()
        plt.savefig(outdir / 'power_trace.png', dpi=150)
        plt.close()

    temp_cols = [c for c in df.columns if c.startswith('temp_') and c.endswith('_c')]
    temp_cols += [c for c in ['ext_external_temp_c', 'ext_tmp117_temp_c'] if c in df.columns]
    if temp_cols:
        plt.figure(figsize=(10, 4))
        for col in temp_cols:
            plt.plot(t, pd.to_numeric(df[col], errors='coerce'))
        plt.legend(temp_cols, fontsize=8)
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.tight_layout()
        plt.savefig(outdir / 'temperature_trace.png', dpi=150)
        plt.close()

    if 'ram_used_mb' in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(t, pd.to_numeric(df['ram_used_mb'], errors='coerce'))
        plt.xlabel('Time (s)')
        plt.ylabel('RAM used (MB)')
        plt.tight_layout()
        plt.savefig(outdir / 'ram_trace.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    main()
