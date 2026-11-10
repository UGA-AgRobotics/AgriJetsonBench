#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensure requested Ultralytics model weights are present locally.")
    p.add_argument('--model', required=True, help='Model filename or path, e.g. yolo11s.pt')
    p.add_argument('--package-dir', default='.', help='Directory where the weights should be available')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    requested = Path(args.model)
    if requested.is_absolute() and requested.exists():
        print(str(requested))
        return

    local_target = package_dir / requested.name
    if local_target.exists():
        print(str(local_target))
        return

    if requested.exists():
        shutil.copy2(requested, local_target)
        print(str(local_target))
        return

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info('Weights %s not found locally. Trying official Ultralytics asset download...', requested.name)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        resolved = Path(attempt_download_asset(requested.name)).expanduser().resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {requested.name} automatically. "
            f"If your cluster blocks outbound downloads, fetch the official weights once on a login node "
            f"and copy them into {package_dir}. Original error: {exc}"
        ) from exc

    if resolved != local_target:
        shutil.copy2(resolved, local_target)
    print(str(local_target))

if __name__ == '__main__':
    main()
