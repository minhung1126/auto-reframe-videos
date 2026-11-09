"""Temporary, five-release migration of installation-local GUI settings."""

from copy import deepcopy
from pathlib import Path
import shutil

from auto_reframe_core.config_store import (
    ConfigStoreError,
    clear_config,
    config_revision,
    load_config,
    save_config,
)
from auto_reframe_core.output_locks import FileLock
from auto_reframe_core.version import VERSION

# The migration ships in five patch releases, including its first release.
# Remove this module and its call site at 3.0.19; the guard already disables it.
MIGRATION_RELEASES = tuple((3, 0, patch) for patch in range(14, 19))


def migrate_config(destination, candidates, defaults, validator, *, version=None):
    if tuple(VERSION if version is None else version) not in MIGRATION_RELEASES:
        return None
    destination = Path(destination)
    if destination.exists():
        return None
    for candidate in candidates:
        source = Path(candidate)
        if source.resolve() == destination.resolve() or not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ConfigStoreError(f"舊設定檔不是一般檔案，已保留原檔：{source}")
        with FileLock(source.with_name(source.name + ".lock"), timeout=2):
            if destination.exists() or not source.exists():
                return None
            original_revision = config_revision(source)
            saved = load_config(source)
            settings = deepcopy(defaults)
            settings.update(saved)
            for field, filename in (
                ("top_text", "top_text.txt"),
                ("bottom_text", "bottom_text.txt"),
            ):
                text_path = source.parent / filename
                if field not in saved and text_path.is_file():
                    settings[field] = (
                        text_path.read_text(encoding="utf-8-sig")
                        .replace("\r", "")
                        .rstrip("\n")
                    )
            font = Path(settings["font_path"])
            if not font.is_absolute() and font.as_posix() != "fonts/NotoSerifTC.ttf":
                settings["font_path"] = str((source.parent / font).resolve())
            settings = validator(settings)
            # Preserve selected legacy PNG references without moving the user's artwork.
            for watermark in settings["watermarks"].values():
                filename = watermark["file"]
                if not filename:
                    continue
                old_png = Path(filename)
                if not old_png.is_absolute():
                    old_png = source.parent / old_png
                    if not old_png.is_file():
                        old_png = source.parent / "watermark" / filename
                if old_png.is_file() and old_png.suffix.lower() == ".png":
                    new_png = destination.parent / "watermark" / old_png.name
                    new_png.parent.mkdir(parents=True, exist_ok=True)
                    if new_png.exists():
                        if new_png.read_bytes() != old_png.read_bytes():
                            raise ConfigStoreError(
                                f"浮水印同名但內容不同，已保留原設定檔：{new_png}"
                            )
                    else:
                        # Exclusive creation avoids overwriting another instance's PNG.
                        created = False
                        try:
                            with (
                                old_png.open("rb") as artwork,
                                new_png.open("xb") as output,
                            ):
                                created = True
                                shutil.copyfileobj(artwork, output)
                        except BaseException:
                            if created:
                                new_png.unlink(missing_ok=True)
                            raise
                    watermark["file"] = new_png.name
            if config_revision(source) != original_revision:
                raise ConfigStoreError("舊設定檔在遷移期間已變更，已保留原檔。")
            revision = save_config(destination, settings, expected_revision=None)
            try:
                # Verify persisted bytes before deleting the only older copy.
                if validator(load_config(destination)) != settings:
                    raise ConfigStoreError("遷移後的設定檔驗證失敗，已保留原檔。")
                if config_revision(source) != original_revision:
                    raise ConfigStoreError("舊設定檔在遷移期間已變更，已保留原檔。")
                source.unlink()
            except BaseException:
                clear_config(destination, expected_revision=revision)
                raise
            return source
    return None
