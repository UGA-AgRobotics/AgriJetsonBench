from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from jetson.utils import ensure_dir, monotonic_ns, utc_now_iso, wallclock_ns
else:
    from .utils import ensure_dir, monotonic_ns, utc_now_iso, wallclock_ns

TEMP_PATTERN = re.compile(r'([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C')
RAM_PATTERN = re.compile(r'RAM\s+(\d+)/(\d+)MB')
LFB_PATTERN = re.compile(r'RAM\s+\d+/\d+MB\s+\(lfb\s+(\d+)x(\d+)MB\)')
SWAP_PATTERN = re.compile(r'SWAP\s+(\d+)/(\d+)MB\s+\(cached\s+(\d+)MB\)')
GR3D_PATTERN = re.compile(r'GR3D_FREQ\s+(\d+)%@?(\d+)?')
EMC_PATTERN = re.compile(r'EMC_FREQ\s+(\d+)%@?(\d+)?')


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


def discover_hwmon_roots() -> List[Path]:
    paths: List[Path] = []
    patterns = [
        '/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*',
        '/sys/class/hwmon/hwmon*',
    ]
    seen = set()
    for pattern in patterns:
        for raw in glob.glob(pattern):
            p = Path(raw)
            try:
                real = str(p.resolve())
                if real not in seen:
                    seen.add(real)
                    paths.append(p)
            except Exception:
                continue
    return paths


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8').strip()
    except Exception:
        return None


def _maybe_float(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw == '':
        return None
    try:
        return float(raw)
    except Exception:
        return None


def parse_tegrastats_line(line: str) -> Dict[str, object]:
    temps = {m.group(1): float(m.group(2)) for m in TEMP_PATTERN.finditer(line)}
    ram_used = None
    ram_total = None
    lfb_blocks = None
    lfb_size_mb = None
    lfb_total_mb = None
    swap_used = None
    swap_total = None
    swap_cached = None
    gr3d_pct = None
    gr3d_freq_mhz = None
    emc_pct = None
    emc_freq_mhz = None

    m = RAM_PATTERN.search(line)
    if m:
        ram_used = float(m.group(1))
        ram_total = float(m.group(2))
    m = LFB_PATTERN.search(line)
    if m:
        lfb_blocks = float(m.group(1))
        lfb_size_mb = float(m.group(2))
        lfb_total_mb = lfb_blocks * lfb_size_mb
    m = SWAP_PATTERN.search(line)
    if m:
        swap_used = float(m.group(1))
        swap_total = float(m.group(2))
        swap_cached = float(m.group(3))
    m = GR3D_PATTERN.search(line)
    if m:
        gr3d_pct = float(m.group(1))
        if m.group(2):
            gr3d_freq_mhz = float(m.group(2))
    m = EMC_PATTERN.search(line)
    if m:
        emc_pct = float(m.group(1))
        if m.group(2):
            emc_freq_mhz = float(m.group(2))
    return {
        'temps': temps,
        'ram_used_mb': ram_used,
        'ram_total_mb': ram_total,
        'lfb_blocks': lfb_blocks,
        'lfb_size_mb': lfb_size_mb,
        'lfb_total_mb': lfb_total_mb,
        'swap_used_mb': swap_used,
        'swap_total_mb': swap_total,
        'swap_cached_mb': swap_cached,
        'gr3d_pct': gr3d_pct,
        'gr3d_freq_mhz': gr3d_freq_mhz,
        'emc_pct': emc_pct,
        'emc_freq_mhz': emc_freq_mhz,
    }


def collect_hwmon_records(sched_wallclock_ns: Optional[int] = None) -> List[Dict[str, object]]:
    roots = discover_hwmon_roots()
    ts_iso = utc_now_iso()
    ts_wall_start_ns = wallclock_ns()
    ts_mono_start_ns = monotonic_ns()
    records: List[Dict[str, object]] = []

    for root in roots:
        name = _safe_read_text(root / 'name') or ''
        labels: Dict[str, str] = {}
        voltages: Dict[str, str] = {}
        currents: Dict[str, str] = {}
        powers: Dict[str, str] = {}
        current_max: Dict[str, str] = {}
        current_crit: Dict[str, str] = {}

        for file in root.glob('*'):
            if not file.is_file():
                continue
            fname = file.name
            if fname.startswith('in') and fname.endswith('_label'):
                labels[fname.split('_')[0][2:]] = _safe_read_text(file) or ''
            elif fname.startswith('in') and fname.endswith('_input'):
                voltages[fname.split('_')[0][2:]] = _safe_read_text(file) or ''
            elif fname.startswith('curr') and fname.endswith('_input'):
                currents[fname.split('_')[0][4:]] = _safe_read_text(file) or ''
            elif fname.startswith('power') and fname.endswith('_input'):
                powers[fname.split('_')[0][5:]] = _safe_read_text(file) or ''
            elif fname.startswith('curr') and fname.endswith('_max'):
                current_max[fname.split('_')[0][4:]] = _safe_read_text(file) or ''
            elif fname.startswith('curr') and fname.endswith('_crit'):
                current_crit[fname.split('_')[0][4:]] = _safe_read_text(file) or ''

        channel_ids = sorted(
            set(labels) | set(voltages) | set(currents) | set(powers) | set(current_max) | set(current_crit),
            key=lambda x: int(x) if str(x).isdigit() else 999,
        )
        ts_wall_end_ns = wallclock_ns()
        ts_mono_end_ns = monotonic_ns()
        for cid in channel_ids:
            label = labels.get(cid, '')
            v = voltages.get(cid)
            c = currents.get(cid)
            p = powers.get(cid)
            computed_power_mw = None
            try:
                if p is None and v is not None and c is not None:
                    computed_power_mw = (float(v) * float(c)) / 1000.0
            except Exception:
                computed_power_mw = None

            records.append(
                {
                    'sched_wallclock_ns': sched_wallclock_ns,
                    'timestamp_utc': ts_iso,
                    'wallclock_ns_start': ts_wall_start_ns,
                    'wallclock_ns_end': ts_wall_end_ns,
                    'monotonic_ns_start': ts_mono_start_ns,
                    'monotonic_ns_end': ts_mono_end_ns,
                    'hwmon_root': str(root),
                    'sensor_name': name,
                    'channel_id': cid,
                    'label': label,
                    'voltage_mV': v,
                    'current_mA': c,
                    'power_mW_reported': p,
                    'power_mW_computed': computed_power_mw,
                    'current_limit_avg_mA': current_max.get(cid),
                    'current_limit_inst_mA': current_crit.get(cid),
                }
            )
    return records


def summarize_hwmon_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    rails: List[Dict[str, object]] = []
    total_power_w = 0.0
    max_rail_power_w = None
    for rec in records:
        voltage_mV = _maybe_float(rec.get('voltage_mV'))
        current_mA = _maybe_float(rec.get('current_mA'))
        power_mW = _maybe_float(rec.get('power_mW_reported'))
        if power_mW is None:
            power_mW = _maybe_float(rec.get('power_mW_computed'))
        power_w = None if power_mW is None else power_mW / 1000.0
        label = str(rec.get('label') or '').strip()
        if not label:
            label = f"{rec.get('sensor_name', 'hwmon')}:{rec.get('channel_id', '?')}"
        rail = {
            'label': label,
            'sensor_name': rec.get('sensor_name'),
            'channel_id': rec.get('channel_id'),
            'voltage_mV': voltage_mV,
            'current_mA': current_mA,
            'power_w': power_w,
            'current_limit_avg_mA': _maybe_float(rec.get('current_limit_avg_mA')),
            'current_limit_inst_mA': _maybe_float(rec.get('current_limit_inst_mA')),
        }
        if voltage_mV is None and current_mA is None and power_w is None:
            continue
        rails.append(rail)
        if power_w is not None:
            total_power_w += power_w
            max_rail_power_w = power_w if max_rail_power_w is None else max(max_rail_power_w, power_w)
    return {
        'timestamp_utc': records[0]['timestamp_utc'] if records else utc_now_iso(),
        'sched_wallclock_ns': records[0].get('sched_wallclock_ns') if records else None,
        'total_power_w': total_power_w if rails else None,
        'max_rail_power_w': max_rail_power_w,
        'rails': rails,
    }


def collect_thermal_records(sched_wallclock_ns: Optional[int] = None) -> List[Dict[str, object]]:
    ts_iso = utc_now_iso()
    ts_wall_start_ns = wallclock_ns()
    ts_mono_start_ns = monotonic_ns()
    records: List[Dict[str, object]] = []
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        ttype = _safe_read_text(zone / 'type')
        temp_raw = _safe_read_text(zone / 'temp')
        temp_c = None
        try:
            if temp_raw is not None:
                val = float(temp_raw)
                temp_c = val / 1000.0 if abs(val) > 1000 else val
        except Exception:
            temp_c = None
        records.append(
            {
                'sched_wallclock_ns': sched_wallclock_ns,
                'timestamp_utc': ts_iso,
                'wallclock_ns_start': ts_wall_start_ns,
                'wallclock_ns_end': wallclock_ns(),
                'monotonic_ns_start': ts_mono_start_ns,
                'monotonic_ns_end': monotonic_ns(),
                'thermal_zone': zone.name,
                'type': ttype,
                'temp_c': temp_c,
            }
        )
    return records


def summarize_thermal_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    zones: List[Dict[str, object]] = []
    temps: List[float] = []
    for rec in records:
        temp_c = _maybe_float(rec.get('temp_c'))
        if temp_c is None:
            continue
        zone_name = str(rec.get('type') or rec.get('thermal_zone') or 'unknown')
        zones.append({'type': zone_name, 'thermal_zone': rec.get('thermal_zone'), 'temp_c': temp_c})
        temps.append(temp_c)
    max_temp = max(temps) if temps else None
    mean_temp = (sum(temps) / len(temps)) if temps else None
    return {
        'timestamp_utc': records[0]['timestamp_utc'] if records else utc_now_iso(),
        'sched_wallclock_ns': records[0].get('sched_wallclock_ns') if records else None,
        'zones': zones,
        'max_temp_c': max_temp,
        'mean_temp_c': mean_temp,
    }


class InternalSampler(threading.Thread):
    def __init__(self, output_dir: Path, interval_ms: int = 200, start_wallclock_ns: Optional[int] = None):
        super().__init__(daemon=True)
        self.output_dir = ensure_dir(output_dir)
        self.interval_ms = max(interval_ms, 50)
        self.interval_ns = int(self.interval_ms * 1_000_000)
        self.start_wallclock_ns = start_wallclock_ns
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.hwmon_csv = self.output_dir / 'internal_hwmon.csv'
        self.thermal_csv = self.output_dir / 'internal_thermal_zones.csv'
        self.latest_hwmon_snapshot: Dict[str, object] = {}
        self.latest_thermal_snapshot: Dict[str, object] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def get_latest_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                'hwmon': json.loads(json.dumps(self.latest_hwmon_snapshot)),
                'thermal': json.loads(json.dumps(self.latest_thermal_snapshot)),
            }

    def run(self) -> None:
        start_ns = self.start_wallclock_ns if self.start_wallclock_ns is not None else wallclock_ns()
        sample_idx = 0
        with open(self.hwmon_csv, 'a', encoding='utf-8', newline='') as hwf, open(self.thermal_csv, 'a', encoding='utf-8', newline='') as thf:
            hw_writer = csv.DictWriter(
                hwf,
                fieldnames=[
                    'sched_wallclock_ns', 'timestamp_utc', 'wallclock_ns_start', 'wallclock_ns_end',
                    'monotonic_ns_start', 'monotonic_ns_end', 'hwmon_root', 'sensor_name', 'channel_id', 'label',
                    'voltage_mV', 'current_mA', 'power_mW_reported', 'power_mW_computed', 'current_limit_avg_mA', 'current_limit_inst_mA'
                ],
            )
            th_writer = csv.DictWriter(
                thf,
                fieldnames=[
                    'sched_wallclock_ns', 'timestamp_utc', 'wallclock_ns_start', 'wallclock_ns_end',
                    'monotonic_ns_start', 'monotonic_ns_end', 'thermal_zone', 'type', 'temp_c'
                ],
            )
            if self.hwmon_csv.stat().st_size == 0:
                hw_writer.writeheader()
            if self.thermal_csv.stat().st_size == 0:
                th_writer.writeheader()

            while not self._stop_event.is_set():
                sched_ns = start_ns + sample_idx * self.interval_ns
                wait_until_wallclock_ns(sched_ns)

                hw_records = collect_hwmon_records(sched_wallclock_ns=sched_ns)
                thermal_records = collect_thermal_records(sched_wallclock_ns=sched_ns)

                for row in hw_records:
                    hw_writer.writerow(row)
                for row in thermal_records:
                    th_writer.writerow(row)
                hwf.flush()
                thf.flush()

                with self._lock:
                    self.latest_hwmon_snapshot = summarize_hwmon_records(hw_records)
                    self.latest_thermal_snapshot = summarize_thermal_records(thermal_records)
                sample_idx += 1


def _bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _sum_optional(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present))


def _collect_process_tree_snapshot(root_pid: Optional[int], root_label: Optional[str]) -> Dict[str, object]:
    ts_iso = utc_now_iso()
    ts_wall_ns = wallclock_ns()
    ts_mono_ns = monotonic_ns()
    if root_pid is None:
        return {
            'timestamp_utc': ts_iso,
            'wallclock_ns': ts_wall_ns,
            'monotonic_ns': ts_mono_ns,
            'root_pid': None,
            'root_label': root_label,
            'pid_count': 0,
            'available': False,
        }

    try:
        root = psutil.Process(int(root_pid))
        procs = [root] + root.children(recursive=True)
    except Exception as exc:
        return {
            'timestamp_utc': ts_iso,
            'wallclock_ns': ts_wall_ns,
            'monotonic_ns': ts_mono_ns,
            'root_pid': int(root_pid),
            'root_label': root_label,
            'pid_count': 0,
            'available': False,
            'error': f'{type(exc).__name__}: {exc}',
        }

    process_rows: List[Dict[str, object]] = []
    rss_vals: List[Optional[float]] = []
    vms_vals: List[Optional[float]] = []
    shared_vals: List[Optional[float]] = []
    text_vals: List[Optional[float]] = []
    data_vals: List[Optional[float]] = []
    uss_vals: List[Optional[float]] = []
    pss_vals: List[Optional[float]] = []
    swap_vals: List[Optional[float]] = []

    for proc in procs:
        try:
            mem = proc.memory_info()
            try:
                full = proc.memory_full_info()
            except Exception:
                full = None
            row = {
                'pid': proc.pid,
                'name': proc.name(),
                'rss_mb': _bytes_to_mb(getattr(mem, 'rss', None)),
                'vms_mb': _bytes_to_mb(getattr(mem, 'vms', None)),
                'shared_mb': _bytes_to_mb(getattr(mem, 'shared', None)),
                'text_mb': _bytes_to_mb(getattr(mem, 'text', None)),
                'data_mb': _bytes_to_mb(getattr(mem, 'data', None)),
                'uss_mb': _bytes_to_mb(getattr(full, 'uss', None)) if full is not None else None,
                'pss_mb': _bytes_to_mb(getattr(full, 'pss', None)) if full is not None else None,
                'swap_mb': _bytes_to_mb(getattr(full, 'swap', None)) if full is not None else None,
            }
            process_rows.append(row)
            rss_vals.append(row['rss_mb'])
            vms_vals.append(row['vms_mb'])
            shared_vals.append(row['shared_mb'])
            text_vals.append(row['text_mb'])
            data_vals.append(row['data_mb'])
            uss_vals.append(row['uss_mb'])
            pss_vals.append(row['pss_mb'])
            swap_vals.append(row['swap_mb'])
        except Exception:
            continue

    process_rows.sort(key=lambda item: float(item.get('rss_mb') or 0.0), reverse=True)
    top_processes = process_rows[:5]
    return {
        'timestamp_utc': ts_iso,
        'wallclock_ns': ts_wall_ns,
        'monotonic_ns': ts_mono_ns,
        'root_pid': int(root_pid),
        'root_label': root_label,
        'pid_count': len(process_rows),
        'pid_list': [int(row['pid']) for row in process_rows],
        'rss_mb': _sum_optional(rss_vals),
        'vms_mb': _sum_optional(vms_vals),
        'shared_mb': _sum_optional(shared_vals),
        'text_mb': _sum_optional(text_vals),
        'data_mb': _sum_optional(data_vals),
        'uss_mb': _sum_optional(uss_vals),
        'pss_mb': _sum_optional(pss_vals),
        'swap_mb': _sum_optional(swap_vals),
        'top_processes': top_processes,
        'available': bool(process_rows),
    }


class ProcessMemorySampler(threading.Thread):
    def __init__(
        self,
        output_dir: Path,
        interval_ms: int = 200,
        start_wallclock_ns: Optional[int] = None,
        root_pid: Optional[int] = None,
        root_label: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.output_dir = ensure_dir(output_dir)
        self.interval_ms = max(interval_ms, 100)
        self.interval_ns = int(self.interval_ms * 1_000_000)
        self.start_wallclock_ns = start_wallclock_ns
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._root_pid = int(root_pid) if root_pid is not None else None
        self._root_label = root_label or 'process_tree'
        self.csv_log = self.output_dir / 'process_memory.csv'
        self.latest_snapshot: Dict[str, object] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def set_root_pid(self, pid: Optional[int], label: Optional[str] = None) -> None:
        with self._lock:
            self._root_pid = int(pid) if pid is not None else None
            if label:
                self._root_label = label

    def get_latest_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return json.loads(json.dumps(self.latest_snapshot))

    def run(self) -> None:
        start_ns = self.start_wallclock_ns if self.start_wallclock_ns is not None else wallclock_ns()
        sample_idx = 0
        with open(self.csv_log, 'a', encoding='utf-8', newline='') as csvf:
            writer = csv.DictWriter(
                csvf,
                fieldnames=[
                    'timestamp_utc', 'wallclock_ns', 'monotonic_ns', 'root_pid', 'root_label', 'pid_count',
                    'rss_mb', 'vms_mb', 'shared_mb', 'text_mb', 'data_mb', 'uss_mb', 'pss_mb', 'swap_mb',
                    'pid_list_json', 'top_processes_json', 'available', 'error',
                ],
            )
            if self.csv_log.stat().st_size == 0:
                writer.writeheader()

            while not self._stop_event.is_set():
                sched_ns = start_ns + sample_idx * self.interval_ns
                wait_until_wallclock_ns(sched_ns)
                with self._lock:
                    root_pid = self._root_pid
                    root_label = self._root_label
                snapshot = _collect_process_tree_snapshot(root_pid, root_label)
                writer.writerow(
                    {
                        'timestamp_utc': snapshot.get('timestamp_utc'),
                        'wallclock_ns': snapshot.get('wallclock_ns'),
                        'monotonic_ns': snapshot.get('monotonic_ns'),
                        'root_pid': snapshot.get('root_pid'),
                        'root_label': snapshot.get('root_label'),
                        'pid_count': snapshot.get('pid_count'),
                        'rss_mb': snapshot.get('rss_mb'),
                        'vms_mb': snapshot.get('vms_mb'),
                        'shared_mb': snapshot.get('shared_mb'),
                        'text_mb': snapshot.get('text_mb'),
                        'data_mb': snapshot.get('data_mb'),
                        'uss_mb': snapshot.get('uss_mb'),
                        'pss_mb': snapshot.get('pss_mb'),
                        'swap_mb': snapshot.get('swap_mb'),
                        'pid_list_json': json.dumps(snapshot.get('pid_list')),
                        'top_processes_json': json.dumps(snapshot.get('top_processes')),
                        'available': snapshot.get('available'),
                        'error': snapshot.get('error'),
                    }
                )
                csvf.flush()
                with self._lock:
                    self.latest_snapshot = snapshot
                sample_idx += 1


class TegrastatsLogger(threading.Thread):
    def __init__(self, output_dir: Path, interval_ms: int = 200, start_wallclock_ns: Optional[int] = None):
        super().__init__(daemon=True)
        self.output_dir = ensure_dir(output_dir)
        self.interval_ms = max(interval_ms, 100)
        self.start_wallclock_ns = start_wallclock_ns
        self.raw_log = self.output_dir / 'tegrastats_raw.log'
        self.csv_log = self.output_dir / 'tegrastats_parsed.csv'
        self.proc: Optional[subprocess.Popen[str]] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.latest_snapshot: Dict[str, object] = {}

    def stop(self) -> None:
        self._stop_event.set()
        try:
            subprocess.run(['tegrastats', '--stop'], check=False, capture_output=True, text=True)
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def get_latest_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return json.loads(json.dumps(self.latest_snapshot))

    def run(self) -> None:
        if self.start_wallclock_ns is not None:
            wait_until_wallclock_ns(self.start_wallclock_ns)

        cmd = ['tegrastats', '--interval', str(self.interval_ms)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(self.raw_log, 'a', encoding='utf-8') as rawf, open(self.csv_log, 'a', encoding='utf-8', newline='') as csvf:
            writer = csv.DictWriter(
                csvf,
                fieldnames=[
                    'timestamp_utc', 'wallclock_ns', 'monotonic_ns', 'ram_used_mb', 'ram_total_mb',
                    'lfb_blocks', 'lfb_size_mb', 'lfb_total_mb',
                    'swap_used_mb', 'swap_total_mb', 'swap_cached_mb',
                    'gr3d_pct', 'gr3d_freq_mhz', 'emc_pct', 'emc_freq_mhz', 'temps_json', 'raw_line'
                ],
            )
            if self.csv_log.stat().st_size == 0:
                writer.writeheader()

            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if self._stop_event.is_set():
                    break
                ts_iso = utc_now_iso()
                ts_wall_ns = wallclock_ns()
                ts_mono_ns = monotonic_ns()
                line = line.rstrip('\n')
                parsed = parse_tegrastats_line(line)
                rawf.write(f'{ts_iso}\t{ts_wall_ns}\t{ts_mono_ns}\t{line}\n')
                writer.writerow(
                    {
                        'timestamp_utc': ts_iso,
                        'wallclock_ns': ts_wall_ns,
                        'monotonic_ns': ts_mono_ns,
                        'ram_used_mb': parsed['ram_used_mb'],
                        'ram_total_mb': parsed['ram_total_mb'],
                        'lfb_blocks': parsed['lfb_blocks'],
                        'lfb_size_mb': parsed['lfb_size_mb'],
                        'lfb_total_mb': parsed['lfb_total_mb'],
                        'swap_used_mb': parsed['swap_used_mb'],
                        'swap_total_mb': parsed['swap_total_mb'],
                        'swap_cached_mb': parsed['swap_cached_mb'],
                        'gr3d_pct': parsed['gr3d_pct'],
                        'gr3d_freq_mhz': parsed['gr3d_freq_mhz'],
                        'emc_pct': parsed['emc_pct'],
                        'emc_freq_mhz': parsed['emc_freq_mhz'],
                        'temps_json': json.dumps(parsed['temps']),
                        'raw_line': line,
                    }
                )
                csvf.flush()
                rawf.flush()
                snapshot = {
                    'timestamp_utc': ts_iso,
                    'wallclock_ns': ts_wall_ns,
                    'monotonic_ns': ts_mono_ns,
                    'ram_used_mb': parsed['ram_used_mb'],
                    'ram_total_mb': parsed['ram_total_mb'],
                    'lfb_blocks': parsed['lfb_blocks'],
                    'lfb_size_mb': parsed['lfb_size_mb'],
                    'lfb_total_mb': parsed['lfb_total_mb'],
                    'swap_used_mb': parsed['swap_used_mb'],
                    'swap_total_mb': parsed['swap_total_mb'],
                    'swap_cached_mb': parsed['swap_cached_mb'],
                    'gr3d_pct': parsed['gr3d_pct'],
                    'gr3d_freq_mhz': parsed['gr3d_freq_mhz'],
                    'emc_pct': parsed['emc_pct'],
                    'emc_freq_mhz': parsed['emc_freq_mhz'],
                    'temps': parsed['temps'],
                    'max_temp_c': max(parsed['temps'].values()) if parsed['temps'] else None,
                    'raw_line': line,
                }
                with self._lock:
                    self.latest_snapshot = snapshot
