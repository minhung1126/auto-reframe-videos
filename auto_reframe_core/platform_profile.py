# -*- coding: utf-8 -*-
"""Platform-specific runtime decisions for the video tools."""

from dataclasses import dataclass
import os
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


def hidden_subprocess_kwargs(
    profile: Optional[PlatformProfile] = None,
) -> Dict[str, int]:
    """Prevent Windows console windows from being created for helper processes."""
    if not (profile or current_platform()).is_windows:
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
