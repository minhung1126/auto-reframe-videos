"""Audit a frozen bundle and run its own test with external tools removed from PATH."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_reframe_core.version import __version__


def is_macho(path):
    with path.open('rb') as source:
        return source.read(4) in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                                  b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca',
                                  b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce')


def bundle_paths(app, target):
    app = Path(app)
    if target.startswith('macos'):
        return app / 'Contents' / 'Frameworks', app / 'Contents' / 'MacOS' / 'Auto Reframe Videos'
    suffix = '.exe' if target == 'windows-x64' else ''
    return app / '_internal', app / ('Auto Reframe Videos' + suffix)


def macho_minimum_versions(commands):
    versions = []
    field = None
    for line in commands.splitlines():
        if line.strip().startswith('cmd '):
            field = {'cmd LC_BUILD_VERSION': 'minos',
                     'cmd LC_VERSION_MIN_MACOSX': 'version'}.get(line.strip())
        if field:
            match = re.match(r'\s*' + field + r'\s+(\d+\.\d+(?:\.\d+)?)', line)
            if match:
                versions.append(match.group(1))
    return versions


def verify_bundle(app, target):
    resources, executable = bundle_paths(app, target)
    if not executable.is_file():
        raise RuntimeError('Missing frozen entry point')
    suffix = '.exe' if target == 'windows-x64' else ''
    required = ('config.json.example', 'fonts/NotoSerifTC.ttf', 'fonts/LICENSE',
                'LICENSE', 'THIRD_PARTY_NOTICES.md', 'build-info.json', 'provenance.json',
                'bin/ffmpeg' + suffix, 'bin/ffprobe' + suffix, 'certs/cacert.pem')
    for name in required:
        if not (resources / name).is_file():
            raise RuntimeError('Missing frozen resource: ' + name)
    info = json.loads((resources / 'build-info.json').read_text(encoding='utf-8'))
    if info['version'] != __version__ or info['target'] != target or not re.fullmatch('[0-9a-f]{40,64}', info['commit']):
        raise RuntimeError('Build provenance mismatch')
    for path in Path(app).rglob('*'):
        if path.name.lower() in {'.git', '.env', 'config.json', 'credentials.json', 'secrets.json', 'input', 'output', 'watermark', 'top_text.txt', 'bottom_text.txt'} or path.suffix.lower() in ('.p12', '.pfx', '.key', '.mp4', '.mov', '.ipynb') or (path.suffix.lower() == '.pem' and path.resolve() != (resources / 'certs/cacert.pem').resolve()):
            raise RuntimeError('Unexpected private/runtime file: ' + str(path))
    if target.startswith('macos'):
        for path in Path(app).rglob('*'):
            if path.is_file() and not path.is_symlink() and is_macho(path):
                archs = subprocess.run(['lipo', '-archs', str(path)], check=True, capture_output=True, text=True).stdout.split()
                if ('arm64' if target.endswith('arm64') else 'x86_64') not in archs:
                    raise RuntimeError('Bundle architecture mismatch: ' + str(path))
                commands = subprocess.run(['otool', '-l', str(path)], check=True, capture_output=True, text=True).stdout
                versions = macho_minimum_versions(commands)
                if not versions or any(tuple(map(int, v.split('.'))) > (13, 0, 0) for v in versions):
                    raise RuntimeError('Cannot establish macOS 13 compatibility: ' + str(path))
    return executable


def smoke_bundle(app, target, report):
    _, executable = bundle_paths(app, target)
    report = Path(report).resolve()
    report.unlink(missing_ok=True)
    env = dict(os.environ, PATH='')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'TCL_LIBRARY', 'TK_LIBRARY'):
        env.pop(key, None)
    # macOS Finder supplies a different cwd; use a directory outside the bundle.
    subprocess.run([str(executable.resolve()), '--desktop-smoke', str(report)],
                   check=True, timeout=240, cwd=report.parent, env=env)
    if not report.is_file() or json.loads(report.read_text(encoding='utf-8')).get('status') != 'passed':
        raise RuntimeError('Frozen smoke test did not pass')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verify_bundle(args.bundle, args.target)
    smoke_bundle(args.bundle, args.target, args.report)


if __name__ == '__main__':
    main()
