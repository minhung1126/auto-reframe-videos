# -*- coding: utf-8 -*-
"""Versioned JSON storage for optional GUI settings."""

import json
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Optional

from auto_reframe_core.output_locks import FileLock, OutputBusyError
from auto_reframe_core.job_errors import replace_with_retry


CONFIG_VERSION = 1
_UNSPECIFIED = object()


class ConfigStoreError(RuntimeError):
    """Raised when a GUI config file cannot be read or written."""


class ConfigConflictError(ConfigStoreError):
    """Another application instance changed this document after it was read."""


def config_revision(path):
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _check_revision(path, expected):
    if expected is not _UNSPECIFIED and config_revision(path) != expected:
        raise ConfigConflictError("設定已被另一個程式實例或外部編輯器修改，請重新載入後再儲存。")


def load_config(path: Path) -> Optional[dict]:
    """Load a supported config document; missing files use code defaults."""
    config_path = Path(path)
    if not config_path.is_file():
        return None
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigStoreError(f"無法讀取設定檔 {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigStoreError(f"設定檔格式錯誤: {config_path}")
    if document.get("version") != CONFIG_VERSION:
        raise ConfigStoreError(
            f"不支援的設定檔版本: {document.get('version')!r}"
        )
    settings = document.get("settings")
    if not isinstance(settings, dict):
        raise ConfigStoreError(f"設定檔缺少 settings 物件: {config_path}")
    return settings


def save_config(path: Path, settings: dict, *, expected_revision=_UNSPECIFIED) -> str:
    """Atomically save versioned settings in the selected user-data location."""
    config_path = Path(path)
    payload = json.dumps(
        {"version": CONFIG_VERSION, "settings": settings},
        ensure_ascii=False,
        indent=2,
    )
    temporary_path = None
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(config_path.with_name(config_path.name + ".lock"), timeout=2):
            _check_revision(config_path, expected_revision)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".config-",
                                             suffix=".tmp", dir=config_path.parent, delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            replace_with_retry(temporary_path, config_path)
            return config_revision(config_path)
    except (OSError, OutputBusyError) as exc:
        raise ConfigStoreError(f"無法儲存設定檔 {config_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # Keep the original write/replace diagnostic if cleanup is also blocked.
                pass


def clear_config(path: Path, *, expected_revision=_UNSPECIFIED) -> None:
    """Remove only the optional saved GUI config."""
    try:
        config_path = Path(path)
        with FileLock(config_path.with_name(config_path.name + ".lock"), timeout=2):
            _check_revision(config_path, expected_revision)
            config_path.unlink(missing_ok=True)
    except (OSError, OutputBusyError) as exc:
        raise ConfigStoreError(f"無法移除設定檔 {path}: {exc}") from exc


class ConfigSession:
    """Optimistic concurrency between independent GUI processes and editors."""
    def __init__(self, path):
        self.path = Path(path)
        self.revision = None

    def load(self):
        try:
            with FileLock(self.path.with_name(self.path.name + ".lock"), timeout=2):
                settings = load_config(self.path)
                self.revision = config_revision(self.path)
                return settings
        except (OSError, OutputBusyError) as exc:
            raise ConfigStoreError(f"無法讀取設定檔 {self.path}: {exc}") from exc

    def save(self, settings):
        self.revision = save_config(self.path, settings, expected_revision=self.revision)

    def clear(self):
        clear_config(self.path, expected_revision=self.revision)
        self.revision = None

    def recover(self):
        """Preserve the exact broken file before the user elects to use defaults."""
        from uuid import uuid4
        with FileLock(self.path.with_name(self.path.name + ".lock"), timeout=2):
            backup = self.path.with_name(f"config.recovered-{uuid4().hex}.json")
            self.path.replace(backup)
            self.revision = None
            return backup
