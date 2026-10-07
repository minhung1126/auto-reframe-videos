"""Read-only application resources and per-user application data."""
import os
from pathlib import Path
import sys

from auto_reframe_core.config_store import ConfigStoreError

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


def watermark_root():
    return user_data_root() / "watermark"


def logs_root(system=None, home=None, environ=None, *, create=True):
    system = system or sys.platform
    home = Path(home) if home is not None else Path.home()
    root = (home / "Library" / "Logs" / APP_NAME if system == "darwin"
            else user_data_root(system, home, environ) / "logs")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
