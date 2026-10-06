"""Read-only application resources and persistent, movable user workspaces."""
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from auto_reframe_core.config_store import ConfigStoreError, load_config, save_config

APP_NAME = "Auto Reframe Videos"


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def resource_root():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])).resolve()


def user_data_root(system=None, home=None, environ=None):
    system = system or sys.platform
    home = Path(home) if home is not None else Path.home()
    environ = os.environ if environ is None else environ
    if system == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if system == "win32":
        return Path(environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / APP_NAME
    return Path(environ.get("XDG_CONFIG_HOME", home / ".config")) / APP_NAME


def tool_path(name, configured=None):
    """Frozen apps always use shipped tools; source users may override PATH."""
    if is_frozen():
        suffix = ".exe" if sys.platform == "win32" else ""
        path = resource_root() / "bin" / (name + suffix)
        if not path.is_file():
            raise ConfigStoreError(f"隨附執行檔遺失，請重新安裝：{path}")
        return str(path)
    return str(configured or name)


def logs_root():
    root = user_data_root() / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def input(self):
        return self.root / "input"

    @property
    def output(self):
        return self.root / "output"

    @property
    def watermark(self):
        return self.root / "watermark"

    def ensure(self):
        for path in (self.input, self.output, self.watermark):
            path.mkdir(parents=True, exist_ok=True)
        return self


def validate_workspace(path, resources=None, frozen=None):
    root = Path(path).expanduser().resolve()
    resources = Path(resources or resource_root()).resolve()
    frozen = is_frozen() if frozen is None else frozen
    # Reject both the bundle's resource tree and the enclosing .app/onedir tree.
    install = Path(sys.executable).resolve().parent if frozen else resources
    for parent in resources.parents:
        if parent.suffix.lower() == ".app":
            install = parent
            break
    if frozen and (root == install or install in root.parents or root == resources or resources in root.parents):
        raise ConfigStoreError("影片工作區必須位於程式安裝目錄之外。")
    if root.exists() and not root.is_dir():
        raise ConfigStoreError("影片工作區必須是資料夾。")
    return Workspace(root)


def load_workspace(data_root=None):
    data = Path(data_root or user_data_root())
    settings = load_config(data / "workspace.json")
    if settings is None:
        return None if is_frozen() else Workspace(resource_root())
    value = settings.get("root")
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ConfigStoreError("工作區設定不是有效的絕對路徑。")
    return validate_workspace(value)


def save_workspace(workspace, data_root=None):
    workspace = validate_workspace(workspace.root).ensure()
    save_config(Path(data_root or user_data_root()) / "workspace.json", {"root": str(workspace.root)})


def import_legacy_settings(project, destination, validator=None):
    """Import a copy; preserve original config, text files and all media."""
    project = Path(project).resolve()
    settings = load_config(project / "config.json")
    if settings is None:
        raise ConfigStoreError("所選資料夾沒有舊版 config.json。")
    settings = dict(settings)
    for key, filename in (("top_text", "top_text.txt"), ("bottom_text", "bottom_text.txt")):
        path = project / filename
        if key not in settings and path.is_file():
            settings[key] = path.read_text(encoding="utf-8-sig").replace("\r", "").rstrip("\n")
    font = Path(str(settings.get("font_path", "fonts/NotoSerifTC.ttf")))
    if not font.is_absolute() and font.as_posix() != "fonts/NotoSerifTC.ttf":
        settings["font_path"] = str(project / font)
    # Do not import stale executable paths into a desktop installation.
    if is_frozen():
        settings["ffmpeg"], settings["ffprobe"] = "ffmpeg", "ffprobe"
    if validator:
        validator(settings)
    if Path(destination).resolve() == (project / "config.json").resolve():
        raise ConfigStoreError("匯入目的地不可覆寫原設定檔。")
    save_config(destination, settings)
    return settings


def migrate_legacy_project(project, data_root=None, validator=None):
    """Commit settings and workspace together, restoring previous files on failure."""
    data = Path(data_root or user_data_root())
    workspace = validate_workspace(project).ensure()
    paths = (data / 'config.json', data / 'workspace.json')
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        settings = import_legacy_settings(project, paths[0], validator)
        save_workspace(workspace, data)
    except BaseException:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                temporary = path.with_name(path.name + '.rollback.tmp')
                temporary.write_bytes(content)
                temporary.replace(path)
        raise
    return settings
