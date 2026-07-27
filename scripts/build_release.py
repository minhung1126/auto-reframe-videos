# -*- coding: utf-8 -*-
"""Build the deterministic source application archive used by the updater."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_ROOT_FILES = (
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "config.json.example",
    "run.bat",
    "run.command",
)
INCLUDED_GLOBS = (
    "auto_reframe_core/*.py",
    "fonts/*",
)
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
MANIFEST_NAME = ".release-manifest.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def project_version(root: Path = ROOT) -> str:
    source = (root / "auto_reframe_core" / "version.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "VERSION"
                for target in node.targets
            )
        ):
            value = ast.literal_eval(node.value)
            if (
                isinstance(value, tuple)
                and len(value) == 3
                and all(isinstance(part, int) and part >= 0 for part in value)
            ):
                return ".".join(str(part) for part in value)
    raise RuntimeError("auto_reframe_core/version.py has no valid VERSION tuple")


def application_files(root: Path = ROOT) -> list:
    paths = [root / name for name in INCLUDED_ROOT_FILES]
    license_path = root / "LICENSE"
    if license_path.is_file():
        paths.append(license_path)
    for pattern in INCLUDED_GLOBS:
        paths.extend(root.glob(pattern))
    unique = sorted(set(path.resolve() for path in paths))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise RuntimeError(f"Release input is missing: {missing[0]}")
    return unique


def manifest_for(version: str, paths: list, root: Path = ROOT) -> tuple:
    entries = []
    payloads = {}
    for path in paths:
        relative = path.relative_to(root.resolve()).as_posix()
        payload = path.read_bytes()
        if relative.endswith(".command"):
            mode = 0o755
        else:
            mode = 0o644
        entries.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size": len(payload),
                "mode": mode,
            }
        )
        payloads[relative] = payload
    manifest = {
        "format_version": 1,
        "version": version,
        "files": entries,
    }
    return manifest, payloads


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def build_release(version: str, output_dir: Path) -> tuple:
    if not VERSION_PATTERN.fullmatch(version):
        raise RuntimeError(f"Invalid release version: {version!r}")
    code_version = project_version()
    if code_version != version:
        raise RuntimeError(
            f"Requested version {version} does not match "
            f"auto_reframe_core.version {code_version}"
        )

    manifest, payloads = manifest_for(version, application_files())
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    prefix = f"auto-reframe-videos-v{version}/"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"auto-reframe-videos-v{version}.zip"
    with zipfile.ZipFile(archive, "w", compresslevel=9) as bundle:
        for item in manifest["files"]:
            bundle.writestr(
                _zip_info(prefix + item["path"], item["mode"]),
                payloads[item["path"]],
            )
        bundle.writestr(
            _zip_info(prefix + MANIFEST_NAME, 0o644),
            manifest_payload,
        )

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksums = destination / f"SHA256SUMS-v{version}.txt"
    checksums.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, checksums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    archive, checksums = build_release(
        args.version or project_version(),
        args.output_dir,
    )
    print(archive)
    print(checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
