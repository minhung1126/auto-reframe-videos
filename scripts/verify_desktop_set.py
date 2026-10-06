"""Release gate: exactly three unsigned personal-use installers from the same version/commit."""
import argparse
import hashlib
import json
from pathlib import Path
import os

from auto_reframe_core.platform_profile import desktop_asset_name
from auto_reframe_core.version import __version__

TARGETS = ('macos-arm64', 'macos-x64', 'windows-x64')


def verify_set(directory, expected_commit=None):
    directory = Path(directory)
    commits = set()
    expected = set()
    for target in TARGETS:
        name = desktop_asset_name(__version__, target)
        expected.update((name, name + '.sha256', name + '.build.json'))
        artifact = directory / name
        info = json.loads((directory / (name + '.build.json')).read_text(encoding='utf-8'))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if (info.get('version') != __version__ or info.get('target') != target
                or info.get('artifact') != name or info.get('signed') is not False
                or info.get('dirty') is not False or info.get('sha256') != digest
                or info.get('smoke', {}).get('status') != 'passed' or not info.get('bundle_files')):
            raise ValueError('Desktop release validation failed: ' + name)
        if (directory / (name + '.sha256')).read_text(encoding='ascii') != f'{digest}  {name}\n':
            raise ValueError('Desktop checksum mismatch')
        commits.add(info.get('commit'))
    if {p.name for p in directory.iterdir()} != expected or len(commits) != 1 or None in commits:
        raise ValueError('Desktop set is incomplete or contains mismatched builds')
    if expected_commit and commits != {expected_commit}:
        raise ValueError('Desktop installers do not match release commit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    verify_set(args.directory, os.environ.get('GITHUB_SHA'))
    print('Verified three unsigned desktop installers')


if __name__ == '__main__':
    main()
