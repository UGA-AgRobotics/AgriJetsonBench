from __future__ import annotations

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
