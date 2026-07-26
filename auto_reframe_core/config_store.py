# -*- coding: utf-8 -*-
"""Versioned JSON storage for optional GUI settings."""

import json
from pathlib import Path
from typing import Optional


CONFIG_VERSION = 1


class ConfigStoreError(RuntimeError):
    """Raised when a GUI config file cannot be read or written."""


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


def save_config(path: Path, settings: dict) -> None:
    """Atomically save settings beside the application."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_name(config_path.name + ".tmp")
    payload = json.dumps(
        {"version": CONFIG_VERSION, "settings": settings},
        ensure_ascii=False,
        indent=2,
    )
    try:
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(config_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigStoreError(f"無法儲存設定檔 {config_path}: {exc}") from exc


def clear_config(path: Path) -> None:
    """Remove only the optional saved GUI config."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        raise ConfigStoreError(f"無法移除設定檔 {path}: {exc}") from exc
