from __future__ import annotations

import json
from pathlib import Path

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pi_logger.sensors import scan_all
else:
    from .sensors import scan_all


if __name__ == '__main__':
    print(json.dumps(scan_all(), indent=2))
