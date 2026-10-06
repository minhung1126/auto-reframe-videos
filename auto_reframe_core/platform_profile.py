# -*- coding: utf-8 -*-
"""Platform-specific runtime decisions for the video tools."""

from dataclasses import dataclass
import os
import platform
import subprocess
import sys
from typing import Dict, Optional


@dataclass(frozen=True)
class PlatformProfile:
    system: str
    os_name: str
    cpu_count: int

    @property
    def is_macos(self) -> bool:
        return self.system == "darwin"

    @property
    def is_windows(self) -> bool:
        return self.os_name == "nt"

    @property
    def worker_limit(self) -> int:
        if self.is_macos:
            return 4
        return min(8, self.cpu_count)


def current_platform() -> PlatformProfile:
    return PlatformProfile(
        system=sys.platform,
        os_name=os.name,
        cpu_count=os.cpu_count() or 2,
    )


def resolve_workers(max_workers: int, profile: Optional[PlatformProfile] = None) -> int:
    """Resolve configured worker count with platform-specific safety caps."""
    p = profile or current_platform()
    workers = max_workers
    if workers <= 0:
        workers = p.cpu_count // 2
    return max(1, min(workers, p.worker_limit))


def should_pause_with_windows_prompt(profile: Optional[PlatformProfile] = None) -> bool:
    return (profile or current_platform()).is_windows


def pause_for_windows_shell(profile: Optional[PlatformProfile] = None) -> None:
    if should_pause_with_windows_prompt(profile):
        os.system("pause")


def open_directory(path, profile: Optional[PlatformProfile] = None) -> None:
    """Open a directory in the platform's native file manager."""

    directory = os.fspath(path)
    p = profile or current_platform()
    if p.is_windows:
        # ``startfile`` delegates to Explorer without creating a console window.
        os.startfile(directory)  # type: ignore[attr-defined]
        return

    command = ["open", directory] if p.is_macos else ["xdg-open", directory]
    subprocess.Popen(command, **hidden_subprocess_kwargs(p))


def hidden_subprocess_kwargs(
    profile: Optional[PlatformProfile] = None,
) -> Dict[str, int]:
    """Prevent Windows console windows from being created for helper processes."""
    if not (profile or current_platform()).is_windows:
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def desktop_target(system=None, machine=None):
    """Select by running process architecture, including an Intel app under Rosetta."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    cpu = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if system == "darwin" and cpu in ("arm64", "x64"):
        return f"macos-{cpu}"
    if system == "win32" and cpu == "x64":
        return "windows-x64"
    raise ValueError(f"不支援的桌面平台／架構：{system}/{machine}")


def desktop_asset_name(version, target=None):
    target = target or desktop_target()
    if target not in ("macos-arm64", "macos-x64", "windows-x64"):
        raise ValueError(f"不支援的桌面成品：{target}")
    extension = "-Setup.exe" if target == "windows-x64" else ".dmg"
    return f"auto-reframe-videos-v{version}-{target}{extension}"


_installer_mutex = None


def register_installer_mutex():
    """Expose a process-lifetime mutex so Inno Setup refuses a live application."""
    global _installer_mutex
    if current_platform().is_windows:
        import ctypes
        from ctypes import wintypes
        create = ctypes.windll.kernel32.CreateMutexW
        create.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        create.restype = wintypes.HANDLE
        _installer_mutex = create(None, False, "AutoReframeVideosDesktop")
        if not _installer_mutex:
            raise OSError("Unable to create installer mutex")
