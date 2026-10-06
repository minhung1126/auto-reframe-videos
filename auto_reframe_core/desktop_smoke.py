"""Self-test the shipped Tk, fonts and FFmpeg using generated, non-personal media."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

from auto_reframe_core.runtime_paths import resource_root, tool_path
from auto_reframe_core.platform_profile import hidden_subprocess_kwargs


def run_desktop_smoke(report, *, verify_tk=True):
    import tkinter as tk
    from auto_reframe_core.compress import CompressConfig, VideoCompressor
    from auto_reframe_core.reframe import ReframeConfig, VideoReframer
    from auto_reframe_core.video_utils import get_video_info

    tk_version = None
    if verify_tk:
        root = tk.Tk()
        root.withdraw()
        root.update()
        tk_version = root.tk.call('info', 'patchlevel')
        root.destroy()
    ffmpeg, ffprobe = tool_path('ffmpeg'), tool_path('ffprobe')
    with tempfile.TemporaryDirectory(prefix='arv-中文 空白-') as temp:
        workspace = Path(temp)
        source = workspace / '來源 影片.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=320x180:rate=12:duration=1',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
                       check=True, timeout=60, **hidden_subprocess_kwargs())
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        watermark = workspace / '浮水印 圖.png'
        watermark.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 16, 16, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\xff\xff\xff' * 16) * 16)) + chunk(b'IEND', b''))
        common = dict(input_dir=str(workspace), output_dir=str(workspace / 'output'),
                      ffmpeg_path=ffmpeg, ffprobe_path=ffprobe, max_workers=1,
                      watermark_enabled=True, watermark_file=str(watermark),
                      watermark_width_ratio=0.15, skip_existing=False)
        reframe = VideoReframer(ReframeConfig(**common,
            font_path=str(resource_root() / 'fonts' / 'NotoSerifTC.ttf'),
            top_text_override='中文 100%\n上方文字', bottom_text_override='下方文字',
            targets=[{'ratio': (4, 5), 'resolution': 'source', 'vcodec': 'h264'},
                     {'ratio': (4, 5), 'resolution': 'source', 'vcodec': 'h265'}]))
        compress = VideoCompressor(CompressConfig(**common,
            targets=[{'resolution': 'source', 'vcodec': 'h264'},
                     {'resolution': 'source', 'vcodec': 'h265'}]))
        for processor in (reframe, compress):
            processor.h264_encoder, processor.h264_hwaccel = 'libx264', None
            processor.h265_encoder, processor.h265_hwaccel = 'libx265', None
            successes, failures = processor.run()
            if successes != 1 or failures:
                raise RuntimeError('Bundled software transcode failed')
        outputs = sorted((workspace / 'output').rglob('*.mp4'))
        if len(outputs) != 4 or list((workspace / 'output').rglob('*.tmp')):
            raise RuntimeError('Incomplete multi-target outputs')
        info = [get_video_info(ffprobe, path) for path in outputs]
        if any(not entry or entry['duration'] < 0.8 or entry['width'] % 2 or entry['height'] % 2 for entry in info):
            raise RuntimeError('Invalid transcoded media')
        payload = {'status': 'passed' if verify_tk else 'media-only', 'tk': tk_version, 'outputs': [p.name for p in outputs],
                   'build': json.loads((resource_root() / 'build-info.json').read_text(encoding='utf-8')) if (resource_root() / 'build-info.json').is_file() else {'source': True}}
    Path(report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0
