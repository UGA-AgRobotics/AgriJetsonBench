#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

PRIMARY_URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
FALLBACK_URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"


class TqdmUpTo(tqdm):
    def update_to(self, blocks: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensure RT-DETR-R18 weights exist locally.")
    p.add_argument("--file-name", default="rtdetr_r18vd_5x_coco_objects365_from_paddle.pth")
    p.add_argument("--package-dir", default=".")
    p.add_argument("--primary-url", default=PRIMARY_URL)
    p.add_argument("--fallback-url", default=FALLBACK_URL)
    return p.parse_args()


def try_download(url: str, target: Path) -> bool:
    try:
        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {target.name}") as t:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        return True
    except Exception:
        if target.exists():
            target.unlink(missing_ok=True)
        return False


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    target = package_dir / args.file_name
    assets_dir = package_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    alt = assets_dir / args.file_name

    for candidate in (target, alt):
        if candidate.exists():
            print(str(candidate))
            return

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("RT-DETR-R18 weights not found locally. Trying primary URL.")
    if try_download(args.primary_url, target):
        print(str(target))
        return

    logging.warning("Primary weight URL failed. Trying fallback URL.")
    if try_download(args.fallback_url, target):
        print(str(target))
        return

    raise RuntimeError(
        "Could not download RT-DETR-R18 weights automatically. "
        f"Please place the checkpoint manually at {target} or {alt}."
    )


if __name__ == "__main__":
    main()
