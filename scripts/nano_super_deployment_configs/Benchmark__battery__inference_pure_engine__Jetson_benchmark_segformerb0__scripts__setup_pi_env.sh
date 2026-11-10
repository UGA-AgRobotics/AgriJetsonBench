#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
TARGET_ENV_NAME="${1:-$CONDA_ENV_NAME}"
PKG_ROOT="${2:-$PKG_ROOT}"
PI_RUNTIME_MODE="$(detect_pi_env_mode "${PI_ENV_MODE:-auto}")"

sudo apt-get update
sudo apt-get install -y python3-pip i2c-tools chrony curl unzip

if [[ "$PI_RUNTIME_MODE" == "venv" ]]; then
  activate_python_venv "$PYTHON_VENV_PATH"
else
  ensure_conda_env "$TARGET_ENV_NAME"
  activate_conda_env "$TARGET_ENV_NAME"
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PKG_ROOT/requirements_pi.txt"

if getent group i2c >/dev/null; then
  sudo usermod -aG i2c "$USER" || true
fi

if [[ "$PI_RUNTIME_MODE" == "venv" ]]; then
  echo "[OK] Raspberry Pi logger env ready in python venv $PYTHON_VENV_PATH"
else
  echo "[OK] Raspberry Pi logger env ready in conda env $TARGET_ENV_NAME"
fi
echo "[NOTE] Re-login or reboot if you just added the current user to the i2c group."
