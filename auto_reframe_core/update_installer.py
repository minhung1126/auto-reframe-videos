# -*- coding: utf-8 -*-
"""Standalone transactional installer copied to a temporary directory."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
import unicodedata


MANIFEST_NAME = ".release-manifest.json"
PROTECTED_TOP_LEVEL = frozenset(
    {
        ".git",
        ".update-backups",
        "config.json",
        "input",
        "output",
        "top_text.txt",
        "bottom_text.txt",
        "watermark",
    }
)


class InstallError(RuntimeError):
    pass


def _safe_relative_path(value: str) -> PurePosixPath:
    if (
        not value
        or len(value) > 1024
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(char in '<>:"|?*' for char in value)
    ):
        raise InstallError(f"Unsafe update path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise InstallError(f"Unsafe update path: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{number}" for number in range(1, 10))
    reserved.update(f"LPT{number}" for number in range(1, 10))
    for part in path.parts:
        if (
            len(part) > 255
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in reserved
        ):
            raise InstallError(f"Unsafe update path: {value!r}")
    if path.parts[0].casefold() in {item.casefold() for item in PROTECTED_TOP_LEVEL}:
        raise InstallError(f"Protected update path: {value}")
    return path


def _path_key(path: PurePosixPath) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _load_manifest(path: Path, expected_version: Optional[str] = None) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallError(f"Cannot read release manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise InstallError("Unsupported release manifest.")
    if expected_version is not None and document.get("version") != expected_version:
        raise InstallError("Release manifest version mismatch.")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise InstallError("Release manifest has no files.")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise InstallError("Invalid release manifest entry.")
        relative = _safe_relative_path(str(item.get("path", "")))
        key = _path_key(relative)
        if key in seen:
            raise InstallError("Duplicate release manifest path.")
        seen.add(key)
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise InstallError(f"Invalid release manifest digest: {relative}")
        try:
            size = int(item.get("size", -1))
            mode = int(item.get("mode", 0))
        except (TypeError, ValueError) as exc:
            raise InstallError(f"Invalid release manifest data: {relative}") from exc
        if size < 0 or mode & ~0o777:
            raise InstallError(f"Invalid release manifest data: {relative}")
    return document


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                return hasher.hexdigest()
            hasher.update(block)


def _confined_path(root: Path, relative: PurePosixPath) -> Path:
    """Resolve a child path without following any install-tree symlink."""
    base = Path(root).resolve()
    candidate = base.joinpath(*relative.parts)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise InstallError(f"Symlink in managed update path: {relative}")
    try:
        candidate.parent.resolve().relative_to(base)
    except ValueError as exc:
        raise InstallError(f"Managed update path escapes install root: {relative}") from exc
    return candidate


def verify_staged_files(staged_root: Path, manifest: dict) -> None:
    for item in manifest["files"]:
        relative = _safe_relative_path(item["path"])
        source = _confined_path(staged_root, relative)
        if not source.is_file():
            raise InstallError(f"Staged update file is missing: {relative}")
        if source.stat().st_size != item["size"] or _hash_file(source) != item["sha256"]:
            raise InstallError(f"Staged update file verification failed: {relative}")


def find_local_modifications(install_root: Path, old_manifest: Optional[dict]) -> list:
    """Return managed files changed since the currently installed release."""
    if old_manifest is None:
        return []
    changed = []
    for item in old_manifest["files"]:
        relative = _safe_relative_path(item["path"])
        target = _confined_path(install_root, relative)
        if not target.is_file():
            changed.append(relative.as_posix())
            continue
        if target.stat().st_size != item["size"] or _hash_file(target) != item["sha256"]:
            changed.append(relative.as_posix())
    return changed


def _copy_into_place(source: Path, target: Path, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".update-tmp")
    try:
        shutil.copyfile(source, temporary)
        try:
            temporary.chmod(mode & 0o777)
        except OSError:
            pass
        os.replace(str(temporary), str(target))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def apply_update(
    install_root: Path,
    staged_root: Path,
    manifest_path: Path,
    version: str,
) -> None:
    """Apply one manifest transaction without leaving install-tree artifacts."""
    root = Path(install_root).resolve()
    staged = Path(staged_root).resolve()
    if (root / ".git").exists():
        raise InstallError("Refusing to update a Git working tree.")
    entrypoints = (
        PurePosixPath("auto_reframe_core/__main__.py"),
        PurePosixPath("auto_reframe_gui.py"),
    )
    if not any(_confined_path(root, path).is_file() for path in entrypoints):
        raise InstallError("Install root does not contain an application entry point.")
    new_manifest = _load_manifest(Path(manifest_path), version)
    verify_staged_files(staged, new_manifest)

    current_manifest_path = _confined_path(root, PurePosixPath(MANIFEST_NAME))
    old_manifest = (
        _load_manifest(current_manifest_path)
        if current_manifest_path.is_file()
        else None
    )
    local_changes = find_local_modifications(root, old_manifest)
    if local_changes:
        preview = ", ".join(local_changes[:8])
        if len(local_changes) > 8:
            preview += ", ..."
        raise InstallError(f"Locally modified managed files: {preview}")

    new_items = {item["path"]: item for item in new_manifest["files"]}
    old_items = (
        {item["path"]: item for item in old_manifest["files"]}
        if old_manifest is not None
        else {}
    )
    touched = sorted(set(new_items) | set(old_items))
    rollback_root = Path(tempfile.mkdtemp(prefix=".update-rollback-", dir=root))
    existed = {}
    try:
        try:
            for name in touched:
                relative = _safe_relative_path(name)
                target = _confined_path(root, relative)
                existed[name] = target.is_file()
                if existed[name]:
                    backup = rollback_root.joinpath(*relative.parts)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)

            for name, item in new_items.items():
                relative = _safe_relative_path(name)
                _copy_into_place(
                    _confined_path(staged, relative),
                    _confined_path(root, relative),
                    item["mode"],
                )
            for name in sorted(set(old_items) - set(new_items)):
                relative = _safe_relative_path(name)
                _confined_path(root, relative).unlink(missing_ok=True)
            current_manifest_path.unlink(missing_ok=True)
        except BaseException:
            for name in reversed(touched):
                relative = _safe_relative_path(name)
                target = _confined_path(root, relative)
                backup = rollback_root.joinpath(*relative.parts)
                try:
                    if existed.get(name) and backup.is_file():
                        _copy_into_place(backup, target, backup.stat().st_mode & 0o777)
                    elif not existed.get(name):
                        target.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    finally:
        shutil.rmtree(rollback_root, ignore_errors=True)


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            wait_timeout = 0x00000102
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_exit(pid: int, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while _process_running(pid):
        if time.monotonic() >= deadline:
            raise InstallError("Timed out waiting for the application to exit.")
        time.sleep(0.25)


def _restart(command: list, cwd: Path) -> None:
    kwargs = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _cleanup_work_dir(plan_path: Path) -> None:
    """Remove only updater-owned temporary work directories."""
    work_dir = Path(plan_path).resolve().parent
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        work_dir.relative_to(temp_root)
    except ValueError:
        return
    if not work_dir.name.startswith("auto-reframe-update-"):
        return
    shutil.rmtree(work_dir, ignore_errors=True)


def run_plan(plan_path: Path) -> None:
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallError(f"Cannot read install plan: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("format_version") != 1:
        raise InstallError("Unsupported install plan.")
    root = Path(str(plan.get("install_root", ""))).resolve()
    staged = Path(str(plan.get("staged_root", ""))).resolve()
    manifest = Path(str(plan.get("manifest_path", ""))).resolve()
    version = str(plan.get("version", ""))
    command = plan.get("restart_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part and "\x00" not in part for part in command
    ):
        raise InstallError("Invalid restart command.")
    parent_pid = int(plan.get("parent_pid", 0))
    _wait_for_exit(parent_pid)
    apply_update(root, staged, manifest, version)
    try:
        (root / "update-error.log").unlink(missing_ok=True)
    except OSError:
        pass
    _restart(command, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    restart_command = None
    restart_root = None
    exit_code = 1
    try:
        raw_plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if isinstance(raw_plan, dict):
            candidate = raw_plan.get("restart_command")
            if isinstance(candidate, list) and candidate:
                restart_command = candidate
            root_value = raw_plan.get("install_root")
            if root_value:
                restart_root = Path(str(root_value)).resolve()
        run_plan(args.plan)
        exit_code = 0
    except BaseException as exc:
        try:
            error_path = (
                restart_root / "update-error.log"
                if restart_root is not None
                else args.plan.parent / "update-error.log"
            )
            if error_path.is_symlink():
                error_path.unlink()
            error_path.write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        if restart_command and restart_root:
            try:
                _restart(restart_command, restart_root)
            except OSError:
                pass
    finally:
        _cleanup_work_dir(args.plan)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
