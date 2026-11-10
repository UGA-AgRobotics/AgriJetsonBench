from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_query(commands: Sequence[Sequence[str]]) -> Dict[str, Any]:
    attempts = []
    for cmd in commands:
        try:
            cp = subprocess.run(cmd, check=False, text=True, capture_output=True)
        except FileNotFoundError as exc:
            attempts.append({'command': list(cmd), 'returncode': None, 'stderr': str(exc)})
            continue
        text = ((cp.stdout or '') + ('\n' + cp.stderr if cp.stderr else '')).strip()
        attempts.append({'command': list(cmd), 'returncode': cp.returncode, 'text': text})
        if cp.returncode == 0 and text:
            return {
                'query_ok': True,
                'query_command': list(cmd),
                'raw_text': text,
                'attempts': attempts,
            }
    return {
        'query_ok': False,
        'query_command': None,
        'raw_text': '',
        'attempts': attempts,
    }


def _extract_current_mode(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    mode_name: Optional[str] = None
    mode_id: Optional[str] = None

    for idx, line in enumerate(lines):
        match = re.search(r'(?:NV\s+)?Power\s+Mode\s*:\s*(.+)', line, flags=re.IGNORECASE)
        if match:
            mode_name = match.group(1).strip()
            id_match = re.search(r'\b(?:ID|MODE)\s*[:=]\s*(\d+)\b', line, flags=re.IGNORECASE)
            if id_match:
                mode_id = id_match.group(1)
            elif idx + 1 < len(lines) and re.fullmatch(r'\d+', lines[idx + 1]):
                mode_id = lines[idx + 1]
            break

    if mode_name is None:
        for line in lines:
            match = re.search(r'\bMode\s+Name\s*[:=]\s*(.+)', line, flags=re.IGNORECASE)
            if match:
                mode_name = match.group(1).strip()
                break

    if mode_id is None:
        for line in lines:
            match = re.search(r'\bMode\s+ID\s*[:=]\s*(\d+)\b', line, flags=re.IGNORECASE)
            if match:
                mode_id = match.group(1)
                break

    return mode_name or None, mode_id or None


def _slugify(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
    return slug or 'unknown_mode'


def _classify_mode(mode_name: Optional[str], mode_id: Optional[str]) -> Dict[str, Any]:
    norm = (mode_name or '').strip().lower()
    watts = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*w\b', norm, flags=re.IGNORECASE)]
    power_budget_w = watts[0] if watts else None

    if any(abs(w - 15.0) < 0.25 for w in watts) or '15w' in norm or '15 w' in norm:
        return {
            'power_mode_class': 'matched_budget_15w',
            'power_mode_tag': 'matched15w',
            'power_mode_label': 'matched_budget',
            'matched_budget_label': 'nominal_15W',
            'power_budget_w': 15.0,
        }

    if any(token in norm for token in ('maxn', 'max-q', 'maxq', 'maxp', 'max power', 'maximum')):
        return {
            'power_mode_class': 'native_max',
            'power_mode_tag': 'native_max',
            'power_mode_label': 'native',
            'matched_budget_label': None,
            'power_budget_w': power_budget_w,
        }

    if any(w > 15.0 for w in watts):
        watts_int = int(round(max(watts)))
        return {
            'power_mode_class': 'native_higher_power',
            'power_mode_tag': f'native_{watts_int}w',
            'power_mode_label': 'native',
            'matched_budget_label': None,
            'power_budget_w': max(watts),
        }

    if any(w < 15.0 for w in watts):
        watts_int = int(round(min(watts)))
        return {
            'power_mode_class': 'observed_lower_power',
            'power_mode_tag': f'lowpower_{watts_int}w',
            'power_mode_label': 'observed_current_mode',
            'matched_budget_label': None,
            'power_budget_w': min(watts),
        }

    fallback_tag = _slugify(mode_name) if mode_name else (f'mode_{mode_id}' if mode_id else 'unknown_mode')
    return {
        'power_mode_class': 'observed_current_mode',
        'power_mode_tag': fallback_tag,
        'power_mode_label': 'observed_current_mode',
        'matched_budget_label': None,
        'power_budget_w': power_budget_w,
    }


def detect_current_power_mode() -> Dict[str, Any]:
    query = _run_query(
        [
            ('/usr/sbin/nvpmodel', '-q', '--verbose'),
            ('nvpmodel', '-q', '--verbose'),
            ('/usr/sbin/nvpmodel', '-q'),
            ('nvpmodel', '-q'),
        ]
    )
    mode_name, mode_id = _extract_current_mode(query.get('raw_text', ''))
    classified = _classify_mode(mode_name, mode_id)
    payload: Dict[str, Any] = {
        'timestamp_utc': utc_now_iso(),
        'query_ok': bool(query.get('query_ok')),
        'query_command': query.get('query_command'),
        'current_mode_name': mode_name,
        'current_mode_id': mode_id,
        'raw_text': query.get('raw_text'),
        'attempts': query.get('attempts', []),
    }
    payload.update(classified)
    return payload


def _shell_value(value: Any) -> str:
    if value is None:
        return "''"
    if isinstance(value, bool):
        return shlex.quote('1' if value else '0')
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect the currently active Jetson nvpmodel mode and classify it for proposal runs.')
    parser.add_argument('--json', action='store_true', help='Print JSON (default output format).')
    parser.add_argument('--shell', action='store_true', help='Print shell-safe KEY=VALUE lines for use with eval.')
    args = parser.parse_args()

    detected = detect_current_power_mode()
    if args.shell:
        shell_map = {
            'CURRENT_POWER_MODE_NAME': detected.get('current_mode_name') or '',
            'CURRENT_POWER_MODE_ID': detected.get('current_mode_id') or '',
            'CURRENT_POWER_MODE_CLASS': detected.get('power_mode_class') or '',
            'CURRENT_POWER_MODE_TAG': detected.get('power_mode_tag') or '',
            'CURRENT_POWER_MODE_LABEL': detected.get('power_mode_label') or '',
            'CURRENT_MATCHED_BUDGET_LABEL': detected.get('matched_budget_label') or '',
            'CURRENT_POWER_BUDGET_W': detected.get('power_budget_w') or '',
            'CURRENT_POWER_MODE_QUERY_OK': 1 if detected.get('query_ok') else 0,
        }
        for key, value in shell_map.items():
            print(f'{key}={_shell_value(value)}')
        return

    print(json.dumps(detected, indent=2))


if __name__ == '__main__':
    main()
