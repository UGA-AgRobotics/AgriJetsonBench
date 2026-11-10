#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

REPO_URL = "https://github.com/lyuwenyu/RT-DETR.git"
ARCHIVE_URL = "https://github.com/lyuwenyu/RT-DETR/archive/refs/heads/main.zip"


class TqdmUpTo(tqdm):
    def update_to(self, blocks: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensure the official RT-DETR repo is available locally.")
    p.add_argument("--vendor-dir", default=None, help="Directory that should contain the RT-DETR checkout")
    p.add_argument("--package-dir", default=".", help="Package directory; vendor/RT-DETR is used when --vendor-dir is omitted")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).expanduser().resolve()
    vendor_dir = Path(args.vendor_dir).expanduser().resolve() if args.vendor_dir else package_dir / "vendor" / "RT-DETR"
    work_dir = vendor_dir / "rtdetr_pytorch"
    patch_script = package_dir / "scripts" / "patch_rtdetr_repo_compat.py"

    def patch_repo() -> None:
        if not patch_script.exists():
            raise FileNotFoundError(f"Compatibility patch script not found: {patch_script}")
        subprocess.run(
            [sys.executable, str(patch_script), "--repo-root", str(work_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    if (work_dir / "tools" / "train.py").exists():
        patch_repo()
        print(str(vendor_dir))
        return

    vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("RT-DETR repo not found locally. Target: %s", vendor_dir)

    git_ok = shutil.which("git") is not None
    if git_ok:
        try:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(vendor_dir)], check=True)
            if (work_dir / "tools" / "train.py").exists():
                print(str(vendor_dir))
                return
        except Exception as exc:
            logging.warning("git clone failed: %s", exc)
            if vendor_dir.exists() and not (work_dir / "tools" / "train.py").exists():
                shutil.rmtree(vendor_dir, ignore_errors=True)

    logging.info("Falling back to archive download: %s", ARCHIVE_URL)
    with tempfile.TemporaryDirectory() as td:
        archive_path = Path(td) / "rtdetr_main.zip"
        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading RT-DETR") as t:
            urllib.request.urlretrieve(ARCHIVE_URL, filename=archive_path, reporthook=t.update_to)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(td)
        extracted = Path(td) / "RT-DETR-main"
        if not extracted.exists():
            raise RuntimeError("Downloaded archive did not contain RT-DETR-main directory.")
        if vendor_dir.exists():
            shutil.rmtree(vendor_dir)
        shutil.move(str(extracted), str(vendor_dir))

    if not (work_dir / "tools" / "train.py").exists():
        raise RuntimeError("RT-DETR repo download finished, but rtdetr_pytorch/tools/train.py was not found.")
    patch_repo()
    print(str(vendor_dir))


if __name__ == "__main__":
    main()
