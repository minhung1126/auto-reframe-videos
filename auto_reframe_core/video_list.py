"""Shared original-file selection, independent of any workspace."""
import os
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.ts', '.m4v'})


def collect_videos(items, existing=(), recursive=False):
    videos = list(existing)
    seen = {os.path.normcase(str(Path(path).resolve())) for path in videos}
    identities = set()
    for path in videos:
        try:
            stat = Path(path).stat()
            identities.add((stat.st_dev, stat.st_ino))
        except OSError:
            pass
    errors = []
    for item in items:
        path = Path(item).expanduser()
        try:
            if path.is_dir():
                candidates = sorted(path.rglob('*') if recursive else path.iterdir())
                candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
                if not candidates:
                    errors.append(f'{path}：沒有支援的影片')
            else:
                candidates = [path]
            for candidate in candidates:
                if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                    errors.append(f'{candidate}：不支援的格式')
                    continue
                try:
                    with candidate.open('rb') as stream:
                        if not stream.read(1):
                            raise ValueError('檔案是空的')
                    resolved = candidate.resolve(strict=True)
                    key = os.path.normcase(str(resolved))
                    stat = resolved.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if key not in seen and identity not in identities:
                        videos.append(resolved)
                        seen.add(key)
                        identities.add(identity)
                except (OSError, ValueError) as exc:
                    errors.append(f'{candidate}：無法讀取（{exc}）')
        except OSError as exc:
            errors.append(f'{path}：無法讀取（{exc}）')
    return tuple(videos), errors
