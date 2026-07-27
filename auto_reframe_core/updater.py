# -*- coding: utf-8 -*-
"""Secure GitHub Release update discovery, download, and staging."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Sequence, Tuple
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile

from auto_reframe_core.version import __version__


GITHUB_OWNER = "minhung1126"
GITHUB_REPOSITORY = "auto-reframe-videos"
GITHUB_API_VERSION = "2026-03-10"
LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/latest"
)
USER_AGENT = f"auto-reframe-videos/{__version__}"
MAX_API_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 500
MAX_COMPRESSION_RATIO = 250
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MANIFEST_NAME = ".release-manifest.json"
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
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


class UpdateError(RuntimeError):
    """Raised when update metadata or update content is unsafe or invalid."""


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    tag_name: str
    asset_name: str
    download_url: str
    sha256: str
    size: int
    release_url: str
    notes: str
    published_at: str
    immutable: bool

    @property
    def available(self) -> bool:
        return parse_version(self.latest_version) > parse_version(self.current_version)


@dataclass(frozen=True)
class StagedUpdate:
    info: UpdateInfo
    work_dir: Path
    staged_root: Path
    manifest_path: Path


def parse_version(value: str) -> Tuple[int, int, int]:
    """Parse the stable semantic version format used by releases."""
    text = str(value).strip()
    if text.startswith("v"):
        text = text[1:]
    parts = text.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise UpdateError(f"不支援的版本格式: {value!r}")
    numbers = tuple(int(part) for part in parts)
    if any(number < 0 for number in numbers):
        raise UpdateError(f"不支援的版本格式: {value!r}")
    return numbers  # type: ignore[return-value]


def _read_limited(response, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise UpdateError("GitHub 回應超過允許大小。")
        except ValueError as exc:
            raise UpdateError("GitHub 回應的 Content-Length 無效。") from exc
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise UpdateError("GitHub 回應超過允許大小。")
    return payload


def _validate_release_url(value: str, tag_name: str) -> str:
    expected = (
        f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/tag/"
        f"{quote(tag_name, safe='')}"
    )
    if value != expected:
        raise UpdateError("Release 頁面不是此專案的 GitHub 官方網址。")
    return value


def _validate_asset_url(value: str, tag_name: str, asset_name: str) -> str:
    expected = (
        f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/download/"
        f"{quote(tag_name, safe='')}/{quote(asset_name, safe='')}"
    )
    if value != expected:
        raise UpdateError("更新檔不是此專案 Release 的官方下載網址。")
    return value


def _request(url: str) -> Request:
    return Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )


def check_for_update(
    current_version: str = __version__,
    opener: Callable = urlopen,
    timeout: int = 15,
) -> UpdateInfo:
    """Return validated metadata for GitHub's latest stable release."""
    parse_version(current_version)
    try:
        with opener(_request(LATEST_RELEASE_API), timeout=timeout) as response:
            final_url = response.geturl()
            if final_url != LATEST_RELEASE_API:
                raise UpdateError("GitHub API 回應發生非預期重新導向。")
            payload = _read_limited(response, MAX_API_BYTES)
    except HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("目前尚未發布任何正式 Release。") from exc
        raise UpdateError(f"GitHub 更新檢查失敗（HTTP {exc.code}）。") from exc
    except (OSError, URLError) as exc:
        raise UpdateError(f"無法連線 GitHub 檢查更新: {exc}") from exc

    try:
        release = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdateError("GitHub Release 回應不是有效 JSON。") from exc
    if not isinstance(release, dict):
        raise UpdateError("GitHub Release 回應格式錯誤。")
    if release.get("draft") or release.get("prerelease"):
        raise UpdateError("GitHub latest Release 不是正式版本。")

    tag_name = str(release.get("tag_name", ""))
    latest_version = ".".join(str(part) for part in parse_version(tag_name))
    expected_asset = f"auto-reframe-videos-v{latest_version}.zip"
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise UpdateError("Release 缺少更新檔清單。")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == expected_asset
    ]
    if len(matches) != 1:
        raise UpdateError(f"Release 必須包含唯一的 {expected_asset}。")
    asset = matches[0]

    digest = str(asset.get("digest", ""))
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise UpdateError("Release 更新檔缺少 GitHub SHA-256 digest。")
    sha256 = digest[7:].lower()
    if any(char not in "0123456789abcdef" for char in sha256):
        raise UpdateError("Release 更新檔的 SHA-256 digest 無效。")
    try:
        size = int(asset.get("size", 0))
    except (TypeError, ValueError) as exc:
        raise UpdateError("Release 更新檔大小無效。") from exc
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise UpdateError("Release 更新檔大小超過安全限制。")

    return UpdateInfo(
        current_version=current_version,
        latest_version=latest_version,
        tag_name=tag_name,
        asset_name=expected_asset,
        download_url=_validate_asset_url(
            str(asset.get("browser_download_url", "")),
            tag_name,
            expected_asset,
        ),
        sha256=sha256,
        size=size,
        release_url=_validate_release_url(str(release.get("html_url", "")), tag_name),
        notes=str(release.get("body") or ""),
        published_at=str(release.get("published_at") or ""),
        immutable=bool(release.get("immutable", False)),
    )


def _validate_final_download_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateError("更新檔被重新導向至不受信任的網址。")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise UpdateError("更新檔下載網址包含不允許的連線資訊。")


def download_update(
    info: UpdateInfo,
    destination_dir: Path,
    opener: Callable = urlopen,
    timeout: int = 30,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download and verify the exact Release asset selected by check_for_update."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / info.asset_name
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    hasher = hashlib.sha256()
    received = 0
    try:
        with opener(_request(info.download_url), timeout=timeout) as response:
            _validate_final_download_url(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) != info.size:
                        raise UpdateError("下載大小與 GitHub Release metadata 不符。")
                except ValueError as exc:
                    raise UpdateError("更新檔 Content-Length 無效。") from exc
            with partial_path.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > info.size or received > MAX_ARCHIVE_BYTES:
                        raise UpdateError("下載內容超過 Release 宣告大小。")
                    output.write(block)
                    hasher.update(block)
                    if progress:
                        progress(received, info.size)
                output.flush()
                os.fsync(output.fileno())
        if received != info.size:
            raise UpdateError("更新檔下載不完整。")
        if hasher.hexdigest() != info.sha256:
            raise UpdateError("更新檔 SHA-256 驗證失敗，已取消安裝。")
        partial_path.replace(archive_path)
        return archive_path
    except (HTTPError, OSError, URLError) as exc:
        raise UpdateError(f"無法下載更新檔: {exc}") from exc
    finally:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_relative_path(value: str) -> PurePosixPath:
    if (
        not value
        or len(value) > 1024
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(char in '<>:"|?*' for char in value)
    ):
        raise UpdateError(f"更新檔包含不安全路徑: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise UpdateError(f"更新檔包含不安全路徑: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{number}" for number in range(1, 10))
    reserved.update(f"LPT{number}" for number in range(1, 10))
    for part in path.parts:
        if (
            len(part) > 255
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in reserved
        ):
            raise UpdateError(f"更新檔包含不安全路徑: {value!r}")
    if path.parts[0].casefold() in {item.casefold() for item in PROTECTED_TOP_LEVEL}:
        raise UpdateError(f"更新檔試圖覆寫使用者資料: {value}")
    return path


def _path_key(path: PurePosixPath) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _load_manifest_bytes(payload: bytes, expected_version: str) -> dict:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdateError("Release manifest 不是有效 JSON。") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise UpdateError("Release manifest 格式版本不支援。")
    if manifest.get("version") != expected_version:
        raise UpdateError("Release manifest 版本與 Release tag 不符。")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise UpdateError("Release manifest 沒有可安裝檔案。")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise UpdateError("Release manifest 檔案項目格式錯誤。")
        path = _safe_relative_path(str(item.get("path", "")))
        key = _path_key(path)
        if key in seen:
            raise UpdateError("Release manifest 包含重複檔案路徑。")
        seen.add(key)
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise UpdateError(f"Release manifest SHA-256 無效: {path}")
        try:
            size = int(item.get("size", -1))
            mode = int(item.get("mode", 0))
        except (TypeError, ValueError) as exc:
            raise UpdateError(f"Release manifest 檔案資料無效: {path}") from exc
        if size < 0 or size > MAX_EXTRACTED_BYTES:
            raise UpdateError(f"Release manifest 檔案大小無效: {path}")
        if mode & ~0o777:
            raise UpdateError(f"Release manifest 權限無效: {path}")
    return manifest


def stage_update(
    archive_path: Path,
    info: UpdateInfo,
    work_dir: Path,
) -> StagedUpdate:
    """Securely validate and extract a verified update archive."""
    archive = Path(archive_path)
    workspace = Path(work_dir)
    staged_root = workspace / "staged"
    staged_root.mkdir(parents=True, exist_ok=False)
    expected_prefix = f"auto-reframe-videos-v{info.latest_version}/"

    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if not entries or len(entries) > MAX_ARCHIVE_ENTRIES:
                raise UpdateError("更新壓縮檔的項目數量不合理。")
            names = set()
            total_size = 0
            files_by_relative = {}
            manifest_entry = None
            for entry in entries:
                name = entry.filename
                if not name.startswith(expected_prefix):
                    raise UpdateError("更新壓縮檔的根目錄名稱不正確。")
                relative_name = name[len(expected_prefix):]
                if not relative_name:
                    continue
                if entry.is_dir():
                    _safe_relative_path(relative_name.rstrip("/"))
                    continue
                relative = _safe_relative_path(relative_name)
                key = _path_key(relative)
                if key in names:
                    raise UpdateError("更新壓縮檔包含重複檔案路徑。")
                names.add(key)
                unix_mode = (entry.external_attr >> 16) & 0o177777
                if stat.S_ISLNK(unix_mode):
                    raise UpdateError("更新壓縮檔不可包含符號連結。")
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in (0, stat.S_IFREG):
                    raise UpdateError("更新壓縮檔不可包含特殊檔案。")
                if entry.flag_bits & 0x1:
                    raise UpdateError("更新壓縮檔不可包含加密檔案。")
                total_size += entry.file_size
                if total_size > MAX_EXTRACTED_BYTES:
                    raise UpdateError("更新壓縮檔解壓後超過安全限制。")
                if (
                    entry.compress_size == 0
                    and entry.file_size > 0
                    or entry.compress_size > 0
                    and entry.file_size / entry.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise UpdateError("更新壓縮檔壓縮比異常。")
                if relative.as_posix() == MANIFEST_NAME:
                    if entry.file_size > MAX_MANIFEST_BYTES:
                        raise UpdateError("Release manifest 超過安全限制。")
                    manifest_entry = entry
                else:
                    files_by_relative[relative.as_posix()] = entry

            if manifest_entry is None:
                raise UpdateError(f"更新壓縮檔缺少 {MANIFEST_NAME}。")
            manifest_payload = bundle.read(manifest_entry)
            manifest = _load_manifest_bytes(manifest_payload, info.latest_version)
            manifest_files = {item["path"]: item for item in manifest["files"]}
            if set(manifest_files) != set(files_by_relative):
                raise UpdateError("更新壓縮檔內容與 Release manifest 不一致。")

            for relative_name, item in manifest_files.items():
                entry = files_by_relative[relative_name]
                if entry.file_size != item["size"]:
                    raise UpdateError(f"更新檔案大小不符: {relative_name}")
                target = staged_root.joinpath(*PurePosixPath(relative_name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                with bundle.open(entry) as source, target.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        hasher.update(block)
                if hasher.hexdigest() != item["sha256"]:
                    raise UpdateError(f"更新檔案 SHA-256 不符: {relative_name}")
                try:
                    target.chmod(item["mode"] & 0o777)
                except OSError:
                    pass

            manifest_path = staged_root / MANIFEST_NAME
            manifest_path.write_bytes(manifest_payload)
            return StagedUpdate(
                info=info,
                work_dir=workspace,
                staged_root=staged_root,
                manifest_path=manifest_path,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError(f"無法驗證更新壓縮檔: {exc}") from exc


def can_self_update(install_root: Path) -> Tuple[bool, str]:
    """Refuse automatic overwrite of developer checkouts or unwritable installs."""
    root = Path(install_root).resolve()
    if (root / ".git").exists():
        return False, "偵測到 Git 工作目錄；請使用 git pull 更新開發版本。"
    entrypoints = (
        root / "auto_reframe_core" / "__main__.py",
        root / "auto_reframe_gui.py",
    )
    if not any(path.is_file() for path in entrypoints):
        return False, "找不到 Auto Reframe 統一入口。"
    if not os.access(str(root), os.W_OK):
        return False, "程式目錄不可寫入，無法自動安裝更新。"
    return True, ""


def prepare_update(
    info: UpdateInfo,
    install_root: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> StagedUpdate:
    """Download and stage an update in a private temporary directory."""
    allowed, reason = can_self_update(install_root)
    if not allowed:
        raise UpdateError(reason)
    workspace = Path(tempfile.mkdtemp(prefix="auto-reframe-update-"))
    try:
        archive = download_update(info, workspace, progress=progress)
        return stage_update(archive, info, workspace)
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def launch_installer(
    staged: StagedUpdate,
    install_root: Path,
    restart_command: Sequence[str],
    parent_pid: Optional[int] = None,
) -> subprocess.Popen:
    """Launch a detached helper that applies the staged files after GUI exit."""
    root = Path(install_root).resolve()
    allowed, reason = can_self_update(root)
    if not allowed:
        raise UpdateError(reason)
    if not restart_command or any("\x00" in str(part) for part in restart_command):
        raise UpdateError("重新啟動命令無效。")

    source_helper = Path(__file__).with_name("update_installer.py")
    if not source_helper.is_file():
        raise UpdateError("找不到更新安裝程式。")
    helper = staged.work_dir / "update_installer.py"
    shutil.copy2(source_helper, helper)
    plan = {
        "format_version": 1,
        "parent_pid": int(parent_pid or os.getpid()),
        "install_root": str(root),
        "staged_root": str(staged.staged_root.resolve()),
        "manifest_path": str(staged.manifest_path.resolve()),
        "version": staged.info.latest_version,
        "restart_command": [str(part) for part in restart_command],
    }
    plan_path = staged.work_dir / "install-plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    kwargs = {
        "cwd": str(staged.work_dir),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(
            [sys.executable, str(helper), "--plan", str(plan_path)],
            **kwargs,
        )
    except OSError as exc:
        raise UpdateError(f"無法啟動更新安裝程式: {exc}") from exc
