"""Native desktop build orchestration with explicit resources and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_reframe_core.platform_profile import desktop_target, desktop_asset_name
from auto_reframe_core.version import __version__
from scripts.desktop_icons import write_icons


def run(*command, **kwargs):
    return subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def validate_vendor(vendor, target):
    """Require audited static FFmpeg tools, codecs, filters and redistribution records."""
    vendor = Path(vendor).resolve()
    provenance = json.loads((vendor / 'provenance.json').read_text(encoding='utf-8'))
    if provenance.get('target') != target or not provenance.get('version'):
        raise ValueError('Vendor provenance target/version mismatch')
    def public_url(value):
        parsed = urlparse(str(value))
        return parsed.scheme == 'https' and bool(parsed.hostname) and not any((parsed.username, parsed.password, parsed.query, parsed.fragment))
    for field in ('source_url', 'build_recipe_url'):
        if not public_url(provenance.get(field, '')):
            raise ValueError(f'Missing vendor {field}')
    # x264/x265 turn FFmpeg into a GPL distribution. A corresponding source offer
    # must cover every component, not merely point at an upstream homepage.
    sources = provenance.get('corresponding_sources', {})
    if not isinstance(sources, dict) or any(not public_url(sources.get(name, '')) for name in ('ffmpeg', 'x264', 'x265')):
        raise ValueError('Missing exact corresponding source downloads')
    for name in ('FFmpeg.txt', 'x264.txt', 'x265.txt', 'Python.txt', 'Tcl.txt', 'Tk.txt'):
        if not (vendor / 'licenses' / name).is_file() or (vendor / 'licenses' / name).stat().st_size < 100:
            raise ValueError(f'Missing license: {name}')
    hashes = provenance.get('binaries', {})
    actual = {p.name for p in (vendor / 'bin').iterdir()}
    suffix = '.exe' if target == 'windows-x64' else ''
    if actual != {'ffmpeg' + suffix, 'ffprobe' + suffix} or set(hashes) != actual:
        raise ValueError('Vendor bin/ must contain only the two self-contained tools')
    for name in sorted(actual):
        path = vendor / 'bin' / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Vendor binary checksum mismatch')
        if target.startswith('macos'):
            deps = run('otool', '-L', str(path), capture_output=True, text=True).stdout.splitlines()[1:]
            if any(not line.strip().split(' ')[0].startswith(('/usr/lib/', '/System/Library/')) for line in deps):
                raise ValueError('FFmpeg links to an external non-system library')
            arch = run('lipo', '-archs', str(path), capture_output=True, text=True).stdout.strip()
            if arch != ('arm64' if target.endswith('arm64') else 'x86_64'):
                raise ValueError('FFmpeg CPU architecture mismatch')
        else:
            import pefile
            pe = pefile.PE(str(path))
            if pe.FILE_HEADER.Machine != 0x8664:
                raise ValueError('FFmpeg must be Windows x64')
            system_dlls = {'kernel32.dll', 'msvcrt.dll', 'ucrtbase.dll', 'user32.dll', 'gdi32.dll',
                'advapi32.dll', 'shell32.dll', 'ole32.dll', 'oleaut32.dll', 'ws2_32.dll', 'bcrypt.dll',
                'secur32.dll', 'crypt32.dll', 'winmm.dll', 'version.dll', 'psapi.dll', 'shlwapi.dll',
                'imm32.dll', 'setupapi.dll', 'd3d11.dll', 'dxgi.dll', 'dxva2.dll', 'dwmapi.dll',
                'mf.dll', 'mfplat.dll', 'mfuuid.dll', 'strmiids.dll', 'cfgmgr32.dll', 'avrt.dll', 'ntdll.dll', 'normaliz.dll', 'comdlg32.dll', 'comctl32.dll'}
            for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []) + getattr(pe, 'DIRECTORY_ENTRY_DELAY_IMPORT', []):
                dll = entry.dll.decode('ascii').lower()
                if dll not in system_dlls and not dll.startswith(('api-ms-win-', 'ext-ms-win-')):
                    raise ValueError(f'Non-system FFmpeg dependency: {dll}')
    ffmpeg = str(vendor / 'bin' / ('ffmpeg' + suffix))
    configuration = run(ffmpeg, '-buildconf', capture_output=True, text=True)
    conf = configuration.stdout + configuration.stderr
    if '--enable-nonfree' in conf or '--enable-gpl' not in conf:
        raise ValueError('FFmpeg must be redistributable GPL without nonfree components')
    encoders = run(ffmpeg, '-encoders', capture_output=True, text=True).stdout
    required = ['libx264', 'libx265', 'aac'] + (['h264_videotoolbox', 'hevc_videotoolbox'] if target.startswith('macos') else ['h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'h264_qsv', 'hevc_qsv'])
    if any(not re.search(r'\b' + name + r'\b', encoders) for name in required):
        raise ValueError('FFmpeg lacks a required encoder')
    decoders = run(ffmpeg, '-decoders', capture_output=True, text=True).stdout
    if not re.search(r'\bpng\b', decoders):
        raise ValueError('FFmpeg lacks the PNG watermark decoder')
    filters = run(ffmpeg, '-filters', capture_output=True, text=True).stdout
    if any(not re.search(r'\b' + name + r'\b', filters) for name in ('drawtext', 'scale', 'crop', 'pad', 'overlay', 'split', 'colorchannelmixer')):
        raise ValueError('FFmpeg lacks a required filter')
    for name in actual:
        run(str(vendor / 'bin' / name), '-version', capture_output=True)
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor-dir', type=Path, required=True)
    parser.add_argument('--target', choices=('macos-arm64', 'macos-x64', 'windows-x64'), required=True)
    parser.add_argument('--iscc', default='iscc')
    parser.add_argument('--unsigned', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if desktop_target() != args.target:
        parser.error('Build on the native target OS and CPU')
    vendor = args.vendor_dir.resolve()
    provenance = validate_vendor(vendor, args.target)
    staging = ROOT / 'build' / 'desktop'
    write_icons(staging)
    from importlib.metadata import distribution
    runtime_licenses = staging / 'runtime-licenses'
    runtime_licenses.mkdir(exist_ok=True)
    for package in ('certifi', 'pyinstaller'):
        dist = distribution(package)
        licenses = [file for file in (dist.files or []) if file.name.upper().startswith(('LICENSE', 'COPYING'))]
        if not licenses:
            raise RuntimeError('Missing build-runtime license: ' + package)
        for index, file in enumerate(licenses):
            shutil.copyfile(dist.locate_file(file), runtime_licenses / f'{package}-{index}.txt')
    revision = run('git', 'rev-parse', 'HEAD', capture_output=True, text=True).stdout.strip()
    info = {'version': __version__, 'target': args.target, 'commit': revision,
            'python': sys.version, 'vendor': provenance,
            'dirty': bool(run('git', 'status', '--porcelain', capture_output=True, text=True).stdout.strip()),
            'minimum_os': 'macOS 13.0 (candidate; hardware qualification required)' if args.target.startswith('macos') else 'Windows 10 22H2; Windows 11 preferred'}
    (staging / 'build-info.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    env = dict(os.environ, ARV_VENDOR_DIR=str(vendor), ARV_TARGET=args.target)
    env.pop('APPLE_SIGN_IDENTITY', None)
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--distpath', str(ROOT / 'build' / 'frozen'), str(ROOT / 'packaging' / 'desktop.spec'), env=env)
    app = ROOT / 'build' / 'frozen' / 'Auto Reframe Videos'
    if args.target.startswith('macos'):
        app = app.with_suffix('.app')
    from scripts.verify_desktop import verify_bundle, smoke_bundle
    verify_bundle(app, args.target)
    smoke_bundle(app, args.target, staging / 'smoke.json')
    output = ROOT / 'dist' / 'desktop'
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / desktop_asset_name(__version__, args.target)
    if args.target.startswith('macos'):
        dmg_source = staging / 'dmg'
        if dmg_source.exists():
            shutil.rmtree(dmg_source)
        dmg_source.mkdir()
        shutil.copytree(app, dmg_source / app.name, symlinks=True)
        (dmg_source / 'Applications').symlink_to('/Applications')
        artifact.unlink(missing_ok=True)
        run('hdiutil', 'create', '-volname', 'Auto Reframe Videos', '-srcfolder', str(dmg_source), '-format', 'UDZO', str(artifact))
    else:
        run(args.iscc, f'/DAppVersion={__version__}', f'/DBuildRoot={app.parent}', f'/DOutputRoot={output}', str(ROOT / 'packaging' / 'windows.iss'))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / (artifact.name + '.sha256')).write_text(f'{digest}  {artifact.name}\n', encoding='ascii')
    info['bundle_files'] = [
        {'path': str(path.relative_to(app)), 'size': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(app.rglob('*')) if path.is_file() and not path.is_symlink()
    ]
    info['artifact'] = artifact.name
    info['sha256'] = digest
    info['signed'] = False
    info['smoke'] = json.loads((staging / 'smoke.json').read_text(encoding='utf-8'))
    (output / (artifact.name + '.build.json')).write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(artifact)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
