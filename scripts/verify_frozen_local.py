"""Exercise the production spec on Linux without pretending to build Mac/Windows installers.

Only local test fixtures are used. The bundle is not a release or a redistributable
Linux product; FFmpeg's linked system libraries are collected by PyInstaller.
"""
import argparse
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_reframe_core.version import __version__
from scripts.desktop_icons import write_icons
from scripts.verify_desktop import verify_bundle, smoke_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'build' / 'frozen-verification')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Use the native desktop build on macOS/Windows')
    directory = args.output_root.resolve()
    metadata = directory / 'metadata'
    vendor = directory / 'vendor'
    (vendor / 'bin').mkdir(parents=True, exist_ok=True)
    (vendor / 'licenses').mkdir(exist_ok=True)
    for name in ('ffmpeg', 'ffprobe'):
        executable = shutil.which(name)
        if not executable:
            parser.error('Local verification needs ' + name)
        shutil.copy2(executable, vendor / 'bin' / name)
    (vendor / 'licenses' / 'TEST-FIXTURE.txt').write_text(
        'Local verification fixture only; not an approved distribution.\n', encoding='utf-8')
    provenance = {'target': 'local-linux', 'purpose': 'Local frozen-bundle verification only'}
    (vendor / 'provenance.json').write_text(json.dumps(provenance) + '\n', encoding='utf-8')
    write_icons(metadata)
    (metadata / 'runtime-licenses').mkdir(exist_ok=True)
    for package in ('certifi', 'pyinstaller'):
        dist = distribution(package)
        for index, file in enumerate(dist.files or []):
            if file.name.upper().startswith(('LICENSE', 'COPYING')):
                shutil.copyfile(dist.locate_file(file), metadata / 'runtime-licenses' / f'{package}-{index}.txt')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    info = {'version': __version__, 'target': 'local-linux', 'commit': commit,
            'purpose': 'Test only; does not qualify native installers'}
    (metadata / 'build-info.json').write_text(json.dumps(info) + '\n', encoding='utf-8')
    env = dict(os.environ, ARV_TARGET='local-linux', ARV_VENDOR_DIR=str(vendor), ARV_METADATA_DIR=str(metadata))
    env.pop('APPLE_SIGN_IDENTITY', None)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--workpath', str(directory / 'work'), '--distpath', str(directory / 'dist'),
                    str(ROOT / 'packaging' / 'desktop.spec')], check=True, cwd=ROOT, env=env)
    app = directory / 'dist' / 'Auto Reframe Videos'
    verify_bundle(app, 'local-linux')
    smoke_bundle(app, 'local-linux', directory / 'smoke.json')
    print('Frozen Tk/FFmpeg/resource verification passed:', directory / 'smoke.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
