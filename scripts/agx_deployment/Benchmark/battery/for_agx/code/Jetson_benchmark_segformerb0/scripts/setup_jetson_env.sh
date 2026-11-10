#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
TARGET_ENV_NAME="${1:-$CONDA_ENV_NAME}"
PKG_ROOT="${2:-$PKG_ROOT}"

sudo apt-get update
sudo apt-get install -y python3-pip chrony curl unzip i2c-tools rfkill x11-xserver-utils

ensure_conda_env "$TARGET_ENV_NAME"
activate_conda_env "$TARGET_ENV_NAME"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PKG_ROOT/requirements_jetson.txt"
python -m pip install "ultralytics>=8.4.37" --no-deps
python - <<'PY'
import site
from pathlib import Path

site_pkgs = None
for candidate in site.getsitepackages():
    if candidate.endswith('site-packages'):
        site_pkgs = Path(candidate)
        break

if site_pkgs is None:
    raise SystemExit('Could not determine env site-packages path')

dist_candidates = [
    Path('/usr/lib/python3.10/dist-packages'),
    Path('/usr/lib/python3/dist-packages'),
]
keep = [str(path) for path in dist_candidates if path.exists()]
pth_path = site_pkgs / 'jetson_system_dist_packages.pth'
pth_path.write_text(''.join(f'{path}\n' for path in keep), encoding='utf-8')
print(f'[INFO] Wrote {pth_path} with: {keep}')
PY

echo "[OK] Jetson DUT environment ready in conda env $TARGET_ENV_NAME"
echo "[NOTE] This script intentionally does NOT install torch."
echo "[NOTE] Make sure your Jetson environment already has a working Jetson-compatible torch + torchvision stack."
echo "[NOTE] OpenCV runtime for ultralytics is installed via requirements_jetson.txt (opencv-python-headless)."
echo "[NOTE] System TensorRT Python bindings are exposed to the env via jetson_system_dist_packages.pth."

echo "[NOTE] Quiet mode uses nmcli, rfkill, systemctl, and best-effort xset/DPMS display control."
