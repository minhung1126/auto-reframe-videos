# -*- coding: utf-8 -*-
"""Verify a built Release archive through the same staging path as the GUI."""

import argparse
import hashlib
from pathlib import Path
import tempfile

from auto_reframe_core.updater import UpdateInfo, stage_update
from scripts.build_release import project_version


ROOT = Path(__file__).resolve().parents[1]


def verify_release(version: str, dist_dir: Path) -> None:
    destination = Path(dist_dir)
    archive = destination / f"auto-reframe-videos-v{version}.zip"
    checksums = destination / f"SHA256SUMS-v{version}.txt"
    if not archive.is_file() or not checksums.is_file():
        raise RuntimeError("Release archive or checksum file is missing.")

    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_line = f"{digest}  {archive.name}\n"
    if checksums.read_text(encoding="ascii") != expected_line:
        raise RuntimeError("Release checksum file does not match the archive.")

    info = UpdateInfo(
        current_version="0.0.0",
        latest_version=version,
        tag_name=f"v{version}",
        asset_name=archive.name,
        download_url=(
            "https://github.com/minhung1126/auto-reframe-videos/releases/download/"
            f"v{version}/{archive.name}"
        ),
        sha256=digest,
        size=len(payload),
        release_url=(
            "https://github.com/minhung1126/auto-reframe-videos/releases/tag/"
            f"v{version}"
        ),
        notes="",
        published_at="",
        immutable=True,
    )
    temporary = tempfile.TemporaryDirectory(prefix="verify-release-")
    try:
        staged = stage_update(archive, info, Path(temporary.name))
        if not (
            staged.staged_root / "auto_reframe_core" / "__main__.py"
        ).is_file():
            raise RuntimeError("Release archive does not contain the unified entry point.")
        if not (staged.staged_root / "auto_reframe_core" / "gui.py").is_file():
            raise RuntimeError("Release archive does not contain the GUI module.")
        if not (staged.staged_root / "auto_reframe_core" / "updater.py").is_file():
            raise RuntimeError("Release archive does not contain the updater.")
        if not (staged.staged_root / "fonts" / "LICENSE").is_file():
            raise RuntimeError("Release archive does not contain the font license.")
        if (staged.staged_root / "watermark").exists():
            raise RuntimeError("Release archive unexpectedly contains personal watermarks.")
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    version = args.version or project_version()
    verify_release(version, args.dist_dir)
    print(f"Verified Release v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
