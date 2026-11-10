#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
TARGET_ENV_NAME="${1:-$CONDA_ENV_NAME}"
PKG_ROOT="${2:-$PKG_ROOT}"
PC_RUNTIME_MODE="$(detect_pi_env_mode "${PI_ENV_MODE:-auto}")"

bootstrap_user_pip() {
  if python3 -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  local tmp_get_pip
  tmp_get_pip="$(mktemp)"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp_get_pip"
  python3 "$tmp_get_pip" --user
  rm -f "$tmp_get_pip"
}

create_python_venv() {
  if python3 -m venv "$PYTHON_VENV_PATH" 2>/tmp/pc_logger_venv_error.log; then
    return 0
  fi

  echo "[WARN] python3 -m venv failed; trying virtualenv fallback."
  cat /tmp/pc_logger_venv_error.log >&2 || true
  bootstrap_user_pip
  python3 -m pip install --user --upgrade virtualenv
  python3 -m virtualenv "$PYTHON_VENV_PATH"
}

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-venv curl unzip chrony
else
  echo "[WARN] manual privileged-command access available nist; apt install skip shod."
  echo "[WARN] Motmaen sho python3-pip, python3-venv, curl, unzip, chrony nasb bashan."
fi

if [[ "$PC_RUNTIME_MODE" == "venv" && ! -f "$PYTHON_VENV_PATH/bin/activate" ]]; then
  create_python_venv
fi

if [[ "$PC_RUNTIME_MODE" == "venv" ]]; then
  activate_python_venv "$PYTHON_VENV_PATH"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$PKG_ROOT/requirements_pc.txt"
elif [[ "$PC_RUNTIME_MODE" == "conda" ]]; then
  ensure_conda_env "$TARGET_ENV_NAME"
  activate_conda_env "$TARGET_ENV_NAME"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$PKG_ROOT/requirements_pc.txt"
else
  bootstrap_user_pip
  python3 -m pip install --user --upgrade pip setuptools wheel
  python3 -m pip install --user -r "$PKG_ROOT/requirements_pc.txt"
fi

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && getent group dialout >/dev/null; then
  sudo usermod -aG dialout "$USER" || true
fi

if [[ "$PC_RUNTIME_MODE" == "venv" ]]; then
  echo "[OK] PC logger env ready in python venv $PYTHON_VENV_PATH"
elif [[ "$PC_RUNTIME_MODE" == "conda" ]]; then
  echo "[OK] PC logger env ready in conda env $TARGET_ENV_NAME"
else
  echo "[OK] PC logger env ready in user Python packages under ~/.local"
fi
echo "[NOTE] Re-login or reboot if you were just added to dialout."
