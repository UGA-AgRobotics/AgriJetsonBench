from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

SAFE_STOP_UNITS: List[str] = [
    'apt-daily.timer',
    'apt-daily.service',
    'apt-daily-upgrade.timer',
    'apt-daily-upgrade.service',
    'unattended-upgrades.service',
    'packagekit.service',
    'packagekit-offline-update.service',
    'ModemManager.service',
    'avahi-daemon.service',
    'cups.service',
    'whoopsie.service',
    'tracker-miner-fs-3.service',
    'tracker-extract-3.service',
    'tracker-miner-fs.service',
    'tracker-extract.service',
]

AGGRESSIVE_EXTRA_UNITS: List[str] = [
    'display-manager.service',
]


def utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class CommandResult:
    ok: bool
    returncode: int
    stdout: str = ''
    stderr: str = ''
    cmd: List[str] = field(default_factory=list)


@dataclass
class NoiseGuardConfig:
    enabled: bool = True
    profile: str = 'safe'
    preserve_url: Optional[str] = None
    display_off: bool = True
    stop_services: bool = True
    disable_wifi: bool = True
    disable_bluetooth: bool = True
    display_env: Optional[str] = None
    xauthority_env: Optional[str] = None


class NoiseGuard:
    """
    Best-effort system quieting helper for Jetson benchmark runs.

    Goals:
    - keep Ethernet / logger connection intact
    - reduce common non-benchmark background noise
    - restore prior state on normal exit, exceptions, SIGINT, or SIGTERM
    """

    def __init__(self, config: NoiseGuardConfig, notes: Optional[List[str]] = None):
        self.config = config
        self.notes = notes if notes is not None else []
        self._entered = False
        self._restored = False
        self._registered = False
        self._prev_signal_handlers: Dict[int, Any] = {}
        self._can_sudo = self._sudo_available()
        self.state: Dict[str, Any] = {
            'enabled': bool(config.enabled),
            'profile': config.profile,
            'entered_utc': None,
            'restored_utc': None,
            'active': False,
            'can_sudo_noninteractive': self._can_sudo,
            'preserve_url': config.preserve_url,
            'preserve_route': None,
            'wifi': {'initial': None, 'action': 'skipped', 'restored': None, 'reason': None},
            'bluetooth': {'initial_soft_blocked': None, 'action': 'skipped', 'restored': None, 'reason': None},
            'display': {'requested': bool(config.display_off), 'available': False, 'display': None, 'action': 'skipped', 'restored': None, 'reason': None},
            'services': {},
            'errors': [],
            'notes': [],
        }

    # ---------- public API ----------
    def enter(self) -> Dict[str, Any]:
        if not self.config.enabled:
            self._note('Noise guard disabled by configuration.')
            return self.summary()
        if self._entered:
            return self.summary()

        self.state['entered_utc'] = utc_now_iso()
        self.state['active'] = True
        self.state['preserve_route'] = self._detect_preserve_route(self.config.preserve_url)

        self._register_cleanup_hooks()
        self._quiet_wifi()
        self._quiet_bluetooth()
        self._quiet_display()
        self._quiet_services()

        self._entered = True
        return self.summary()

    def restore(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return self.summary()
        if self._restored:
            return self.summary()

        # reverse order of disruption
        self._restore_services()
        self._restore_display()
        self._restore_bluetooth()
        self._restore_wifi()

        self.state['restored_utc'] = utc_now_iso()
        self.state['active'] = False
        self._restored = True
        self._unregister_cleanup_hooks()
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        self.state['notes'] = list(self.notes)
        return json.loads(json.dumps(self.state, default=str))

    def dashboard_payload(self) -> Dict[str, Any]:
        route = self.state.get('preserve_route') or {}
        return {
            'enabled': bool(self.config.enabled),
            'active': bool(self.state.get('active', False)),
            'profile': self.config.profile,
            'can_sudo_noninteractive': bool(self._can_sudo),
            'preserve_interface': route.get('dev'),
            'wifi_action': self.state.get('wifi', {}).get('action'),
            'bluetooth_action': self.state.get('bluetooth', {}).get('action'),
            'display_action': self.state.get('display', {}).get('action'),
            'service_stop_count': sum(1 for v in (self.state.get('services') or {}).values() if v.get('stopped')),
            'errors': list(self.state.get('errors', []))[-5:],
        }

    # ---------- helpers ----------
    def _note(self, msg: str) -> None:
        self.notes.append(msg)

    def _error(self, msg: str) -> None:
        self.state.setdefault('errors', []).append(msg)
        self._note(msg)

    def _sudo_available(self) -> bool:
        if os.geteuid() == 0:
            return True
        if shutil.which('sudo') is None:
            return False
        cp = subprocess.run(['sudo', '-n', 'true'], check=False, capture_output=True, text=True)
        return cp.returncode == 0

    def _run(self, cmd: Sequence[str], *, allow_sudo: bool = False, check: bool = False, env: Optional[Dict[str, str]] = None) -> CommandResult:
        candidates: List[List[str]] = [list(cmd)]
        if allow_sudo and os.geteuid() != 0 and self._can_sudo:
            candidates.append(['sudo', '-n', *cmd])

        last_cp: Optional[subprocess.CompletedProcess[str]] = None
        last_cmd: List[str] = list(cmd)
        for attempt in candidates:
            try:
                cp = subprocess.run(attempt, check=False, capture_output=True, text=True, env=env)
            except Exception as exc:  # pragma: no cover - subprocess launch failure is rare
                return CommandResult(ok=False, returncode=127, stdout='', stderr=str(exc), cmd=list(attempt))
            last_cp = cp
            last_cmd = list(attempt)
            if cp.returncode == 0:
                return CommandResult(ok=True, returncode=0, stdout=cp.stdout or '', stderr=cp.stderr or '', cmd=last_cmd)
        if last_cp is None:
            return CommandResult(ok=False, returncode=127, stdout='', stderr='command not executed', cmd=last_cmd)
        if check:
            raise RuntimeError(f'Command failed: {last_cmd} :: {last_cp.stderr or last_cp.stdout}')
        return CommandResult(ok=False, returncode=last_cp.returncode, stdout=last_cp.stdout or '', stderr=last_cp.stderr or '', cmd=last_cmd)

    def _command_exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def _detect_preserve_route(self, preserve_url: Optional[str]) -> Optional[Dict[str, Any]]:
        if not preserve_url:
            return None
        host = preserve_url
        try:
            if '://' in preserve_url:
                host = urlparse(preserve_url).hostname or preserve_url
        except Exception:
            host = preserve_url

        try:
            # Prefer numeric address so `ip route get` does not depend on shell resolution.
            addrinfos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            if addrinfos:
                host = addrinfos[0][4][0]
        except Exception:
            pass

        if not self._command_exists('ip'):
            return {'host': host, 'dev': None, 'raw': None}

        cp = self._run(['ip', 'route', 'get', host], allow_sudo=False)
        if not cp.ok:
            return {'host': host, 'dev': None, 'raw': cp.stderr or cp.stdout}
        raw = (cp.stdout or '').strip().split()
        dev = None
        src = None
        for i, tok in enumerate(raw):
            if tok == 'dev' and i + 1 < len(raw):
                dev = raw[i + 1]
            if tok == 'src' and i + 1 < len(raw):
                src = raw[i + 1]
        info = {'host': host, 'dev': dev, 'src': src, 'raw': cp.stdout.strip()}
        if dev:
            self._note(f'Noise guard preserve-route: {host} via {dev}')
        return info

    def _preserve_requires_wifi(self) -> bool:
        dev = ((self.state.get('preserve_route') or {}).get('dev') or '').lower()
        return dev.startswith('wl') or dev.startswith('wlan')

    # ---------- wifi ----------
    def _get_wifi_state(self) -> Optional[str]:
        if not self._command_exists('nmcli'):
            return None
        cp = self._run(['nmcli', 'radio', 'wifi'])
        if not cp.ok:
            return None
        value = (cp.stdout or '').strip().lower()
        return value if value in {'enabled', 'disabled'} else value or None

    def _set_wifi(self, enabled: bool) -> bool:
        if not self._command_exists('nmcli'):
            return False
        cmd = ['nmcli', 'radio', 'wifi', 'on' if enabled else 'off']
        cp = self._run(cmd, allow_sudo=True)
        return cp.ok

    def _quiet_wifi(self) -> None:
        state = self.state['wifi']
        state['initial'] = self._get_wifi_state()
        if not self.config.disable_wifi:
            state['reason'] = 'disable_wifi=false'
            return
        if self._preserve_requires_wifi():
            state['reason'] = 'preserve route uses wifi; left wifi enabled'
            self._note('Noise guard kept Wi-Fi enabled because the logger/dashboard route is over wlan.')
            return
        if state['initial'] == 'enabled':
            if self._set_wifi(False):
                state['action'] = 'disabled'
                self._note('Noise guard disabled Wi-Fi.')
            else:
                state['action'] = 'failed'
                state['reason'] = 'nmcli failed'
                self._error('Noise guard could not disable Wi-Fi. Run the example scripts so sudo keepalive is available.')
        elif state['initial'] == 'disabled':
            state['action'] = 'already_disabled'
        else:
            state['reason'] = 'nmcli unavailable or state unknown'

    def _restore_wifi(self) -> None:
        state = self.state['wifi']
        if state.get('action') != 'disabled':
            return
        initial = state.get('initial')
        if initial == 'enabled':
            ok = self._set_wifi(True)
            state['restored'] = ok
            if ok:
                self._note('Noise guard restored Wi-Fi to enabled.')
            else:
                self._error('Noise guard could not restore Wi-Fi to enabled.')

    # ---------- bluetooth ----------
    def _get_bluetooth_soft_blocked(self) -> Optional[bool]:
        if not self._command_exists('rfkill'):
            return None
        cp = self._run(['rfkill', 'list', 'bluetooth'])
        if not cp.ok:
            return None
        text = (cp.stdout or '').lower()
        if 'soft blocked: yes' in text:
            return True
        if 'soft blocked: no' in text:
            return False
        return None

    def _set_bluetooth_blocked(self, blocked: bool) -> bool:
        if not self._command_exists('rfkill'):
            return False
        cmd = ['rfkill', 'block' if blocked else 'unblock', 'bluetooth']
        cp = self._run(cmd, allow_sudo=True)
        return cp.ok

    def _quiet_bluetooth(self) -> None:
        state = self.state['bluetooth']
        state['initial_soft_blocked'] = self._get_bluetooth_soft_blocked()
        if not self.config.disable_bluetooth:
            state['reason'] = 'disable_bluetooth=false'
            return
        if state['initial_soft_blocked'] is False:
            ok = self._set_bluetooth_blocked(True)
            state['action'] = 'blocked' if ok else 'failed'
            if ok:
                self._note('Noise guard blocked Bluetooth.')
            else:
                self._error('Noise guard could not block Bluetooth.')
        elif state['initial_soft_blocked'] is True:
            state['action'] = 'already_blocked'
        else:
            state['reason'] = 'rfkill unavailable or no bluetooth adapter detected'

    def _restore_bluetooth(self) -> None:
        state = self.state['bluetooth']
        if state.get('action') != 'blocked':
            return
        initial = state.get('initial_soft_blocked')
        if initial is False:
            ok = self._set_bluetooth_blocked(False)
            state['restored'] = ok
            if ok:
                self._note('Noise guard restored Bluetooth to unblocked.')
            else:
                self._error('Noise guard could not restore Bluetooth.')

    # ---------- display ----------
    def _display_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        display = self.config.display_env or env.get('DISPLAY')
        if not display:
            # best-effort default for a local graphical session
            display = ':0'
        env['DISPLAY'] = display
        xauth = self.config.xauthority_env or env.get('XAUTHORITY')
        if xauth:
            env['XAUTHORITY'] = xauth
        return env

    def _xset_available(self, env: Dict[str, str]) -> bool:
        if not self._command_exists('xset'):
            return False
        cp = self._run(['xset', '-q'], allow_sudo=False, env=env)
        return cp.ok

    def _quiet_display(self) -> None:
        state = self.state['display']
        if not self.config.display_off:
            state['reason'] = 'display_off=false'
            return
        env = self._display_env()
        state['display'] = env.get('DISPLAY')
        if not self._xset_available(env):
            state['reason'] = 'xset unavailable or DISPLAY/XAUTHORITY not accessible'
            return
        state['available'] = True
        cp = self._run(['xset', 'dpms', 'force', 'off'], env=env)
        if cp.ok:
            state['action'] = 'dpms_off'
            self._note(f'Noise guard forced display DPMS off on {env.get("DISPLAY")}')
        else:
            state['action'] = 'failed'
            state['reason'] = cp.stderr or cp.stdout
            self._error('Noise guard could not blank the display automatically. You can still switch the monitor off manually.')

    def _restore_display(self) -> None:
        state = self.state['display']
        if state.get('action') != 'dpms_off':
            return
        env = self._display_env()
        cp = self._run(['xset', 'dpms', 'force', 'on'], env=env)
        state['restored'] = cp.ok
        if cp.ok:
            self._note('Noise guard restored the display DPMS state to on.')
        else:
            self._error('Noise guard could not restore the display DPMS state.')

    # ---------- services ----------
    def _service_units(self) -> List[str]:
        units = list(SAFE_STOP_UNITS)
        if self.config.profile == 'aggressive':
            units.extend(AGGRESSIVE_EXTRA_UNITS)
        return units

    def _unit_exists(self, unit: str) -> bool:
        if not self._command_exists('systemctl'):
            return False
        cp = self._run(['systemctl', 'status', unit], allow_sudo=True)
        text = (cp.stdout or '') + '\n' + (cp.stderr or '')
        return 'Loaded:' in text or cp.returncode in {0, 3}

    def _unit_active(self, unit: str) -> Optional[bool]:
        if not self._command_exists('systemctl'):
            return None
        cp = self._run(['systemctl', 'is-active', unit], allow_sudo=True)
        if cp.returncode == 0:
            return True
        out = (cp.stdout or '').strip().lower()
        if out in {'inactive', 'failed', 'unknown'}:
            return False
        return None

    def _stop_unit(self, unit: str) -> bool:
        cp = self._run(['systemctl', 'stop', unit], allow_sudo=True)
        return cp.ok

    def _start_unit(self, unit: str) -> bool:
        cp = self._run(['systemctl', 'start', unit], allow_sudo=True)
        return cp.ok

    def _quiet_services(self) -> None:
        if not self.config.stop_services:
            return
        for unit in self._service_units():
            entry = {'exists': False, 'was_active': None, 'stopped': False, 'restored': None}
            self.state['services'][unit] = entry
            if not self._unit_exists(unit):
                continue
            entry['exists'] = True
            was_active = self._unit_active(unit)
            entry['was_active'] = was_active
            if was_active:
                ok = self._stop_unit(unit)
                entry['stopped'] = ok
                if ok:
                    self._note(f'Noise guard stopped {unit}.')
                else:
                    self._error(f'Noise guard could not stop {unit}.')

    def _restore_services(self) -> None:
        for unit, entry in (self.state.get('services') or {}).items():
            if not entry.get('exists') or not entry.get('stopped'):
                continue
            if entry.get('was_active'):
                ok = self._start_unit(unit)
                entry['restored'] = ok
                if ok:
                    self._note(f'Noise guard restored {unit}.')
                else:
                    self._error(f'Noise guard could not restore {unit}.')

    # ---------- signal / atexit ----------
    def _register_cleanup_hooks(self) -> None:
        if self._registered:
            return
        self._registered = True
        atexit.register(self.restore)
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                self._prev_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except Exception:
                pass

    def _unregister_cleanup_hooks(self) -> None:
        if not self._registered:
            return
        self._registered = False
        for sig, handler in self._prev_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        self._prev_signal_handlers.clear()

    def _signal_handler(self, signum: int, frame: Any) -> None:  # pragma: no cover - depends on signal delivery
        try:
            self.restore()
        finally:
            prev = self._prev_signal_handlers.get(signum)
            if callable(prev):
                prev(signum, frame)
            raise KeyboardInterrupt(f'Interrupted by signal {signum}')


# ---------- CLI helper ----------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Best-effort quiet mode helper for Jetson benchmark runs.')
    ap.add_argument('--preserve-url', default=None, help='URL or host that must remain reachable, e.g. http://${LOGGER_DASHBOARD_HOST}:8000')
    ap.add_argument('--profile', default='safe', choices=['safe', 'aggressive'])
    ap.add_argument('--hold-s', type=float, default=15.0, help='How long to stay in quiet mode before auto-restore')
    ap.add_argument('--no-display-off', action='store_true')
    ap.add_argument('--no-stop-services', action='store_true')
    ap.add_argument('--no-disable-wifi', action='store_true')
    ap.add_argument('--no-disable-bluetooth', action='store_true')
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    guard = NoiseGuard(
        NoiseGuardConfig(
            enabled=True,
            profile=args.profile,
            preserve_url=args.preserve_url,
            display_off=not args.no_display_off,
            stop_services=not args.no_stop_services,
            disable_wifi=not args.no_disable_wifi,
            disable_bluetooth=not args.no_disable_bluetooth,
        )
    )
    print(json.dumps({'enter': guard.enter()}, indent=2))
    try:
        time.sleep(max(0.0, args.hold_s))
    finally:
        print(json.dumps({'restore': guard.restore()}, indent=2))


if __name__ == '__main__':
    main()
