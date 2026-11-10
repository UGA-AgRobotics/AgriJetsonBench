#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_PATH="${2:-/home/$USER/envs/segment}"

echo "[INFO] Package dir : $PACKAGE_DIR"
echo "[INFO] Venv path   : $VENV_PATH"
echo "[INFO] Shared segmentation env is expected. No new env will be created here."

source "$PACKAGE_DIR/scripts/activate_stage2_env.sh" "$VENV_PATH"

python - <<'PY'
import importlib.metadata as metadata
import sys

print('[INFO] Active Python  :', sys.executable)
failed = False
try:
    import torch
    print('[INFO] Torch version  :', torch.__version__)
    print('[INFO] CUDA available :', torch.cuda.is_available())
except Exception as e:
    print('[WARN] Torch import failed:', e)
    failed = True

def ensure_torchvision_importable():
    try:
        import torchvision
        return torchvision
    except RuntimeError as exc:
        if 'torchvision::nms' not in str(exc):
            raise
        import torch
        lib = torch.library.Library('torchvision', 'DEF')
        try:
            lib.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
        except Exception:
            pass
        import torchvision
        return torchvision

for name in ['numpy', 'pandas', 'yaml', 'onnx', 'onnxsim', 'optuna', 'PIL', 'psutil', 'torchvision']:
    try:
        if name in {'onnxsim', 'optuna'}:
            print(f'[INFO] {name:<14}:', metadata.version(name))
            continue
        if name == 'torchvision':
            mod = ensure_torchvision_importable()
        else:
            mod = __import__(name if name != 'PIL' else 'PIL')
        print(f'[INFO] {name:<14}:', getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(f'[WARN] {name:<14}: {e}')
        failed = True

if failed:
    raise SystemExit(1)
PY

echo "[INFO] Shared segmentation environment is ready for MobileNetV3."
echo "[INFO] Use later in jobs via scripts/activate_stage2_env.sh $VENV_PATH"
