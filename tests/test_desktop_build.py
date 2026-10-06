import hashlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile

from auto_reframe_core.platform_profile import desktop_asset_name
from auto_reframe_core.version import __version__
from scripts.prepare_desktop_vendor import unpack_vendor
from scripts.build_ffmpeg_vendor import prepare_source
from scripts.verify_desktop_set import verify_set, TARGETS


class DesktopBuildTests(unittest.TestCase):
    def test_source_download_checksum_and_archive_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            destination = cache / 'extracted'
            with patch('scripts.build_ffmpeg_vendor.urlopen', return_value=io.BytesIO(b'wrong content')):
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    prepare_source('fixture', {'url': 'https://example.com/source', 'sha256': '0' * 64}, cache, destination)
            self.assertFalse(destination.exists())
            archive = cache / 'fixture.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                member = tarfile.TarInfo('../escape')
                member.size = 4
                bundle.addfile(member, io.BytesIO(b'data'))
            with self.assertRaises(tarfile.FilterError):
                prepare_source('fixture', {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, cache, destination)
            self.assertFalse((cache / 'escape').exists())

    def test_vendor_rejects_traversal_collisions_and_unexpected_data(self):
        for names in [('bin/../../secret',), ('bin/FFmpeg', 'bin/ffmpeg'), ('config.json',), ('bin/evil\\file',), ('/bin/ffmpeg',), ('licenses/CON.txt',), ('bin/./ffmpeg',)]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                archive, destination = Path(directory) / 'vendor.zip', Path(directory) / 'vendor'
                with zipfile.ZipFile(archive, 'w') as bundle:
                    for name in names:
                        # ZipInfo otherwise rewrites backslashes on Windows before
                        # the malicious path ever reaches our validation code.
                        entry = zipfile.ZipInfo()
                        entry.filename = name
                        bundle.writestr(entry, b'data')
                with self.assertRaises(ValueError):
                    unpack_vendor(archive, destination)
                self.assertFalse(destination.exists())

    def test_three_installer_gate_accepts_unsigned_and_rejects_mixed_or_corrupt_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(target, signed=False, commit='a' * 40):
                name = desktop_asset_name(__version__, target)
                artifact = root / name
                artifact.write_bytes(b'fixture-installer')
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                (root / (name + '.sha256')).write_text(f'{digest}  {name}\n', encoding='ascii')
                (root / (name + '.build.json')).write_text(json.dumps({
                    'version': __version__, 'target': target, 'artifact': name,
                    'signed': signed, 'dirty': False, 'sha256': digest, 'commit': commit,
                    'smoke': {'status': 'passed'}, 'bundle_files': [{'path': 'main'}],
                }), encoding='utf-8')
            for target in TARGETS:
                write(target)
            verify_set(root, 'a' * 40)
            write(TARGETS[-1], signed=True)
            with self.assertRaises(ValueError):
                verify_set(root)
            write(TARGETS[-1], commit='b' * 40)
            with self.assertRaises(ValueError):
                verify_set(root)
            write(TARGETS[-1])
            (root / desktop_asset_name(__version__, TARGETS[-1])).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                verify_set(root)
