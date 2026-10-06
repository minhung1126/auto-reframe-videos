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
from scripts.verify_desktop import macho_minimum_versions
from scripts.build_desktop import bundle_windows_runtime


class DesktopBuildTests(unittest.TestCase):
    def test_windows_runtime_is_bundled_transitively_and_rejects_foreign_or_altered_dll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor, toolchain = root / 'vendor', root / 'toolchain'
            (vendor / 'bin').mkdir(parents=True)
            (toolchain / 'mingw64/bin').mkdir(parents=True)
            hashes = {}
            for name in ('ffmpeg.exe', 'ffprobe.exe'):
                (vendor / 'bin' / name).write_bytes(name.encode())
                hashes[name] = hashlib.sha256(name.encode()).hexdigest()
            (vendor / 'provenance.json').write_text(json.dumps({'binaries': hashes}))
            for name in ('libgcc_s_seh-1.dll', 'libwinpthread-1.dll'):
                (toolchain / 'mingw64/bin' / name).write_bytes(name.encode())
            imports = {'ffmpeg.exe': ['libgcc_s_seh-1.dll'], 'ffprobe.exe': ['avicap32.dll', 'avifil32.dll'],
                       'libgcc_s_seh-1.dll': ['libwinpthread-1.dll'],
                       'libwinpthread-1.dll': ['kernel32.dll']}
            with patch('scripts.build_desktop.windows_imports', side_effect=lambda p: imports[p.name]), \
                 patch('scripts.build_desktop.run') as run:
                run.return_value.stdout = '15.2.0\n'
                bundle_windows_runtime(vendor, toolchain)
                provenance = json.loads((vendor / 'provenance.json').read_text())
                self.assertEqual(set(provenance['binaries']), set(hashes) | {'libgcc_s_seh-1.dll', 'libwinpthread-1.dll'})
                for name, digest in provenance['binaries'].items():
                    self.assertEqual(hashlib.sha256((vendor / 'bin' / name).read_bytes()).hexdigest(), digest)
                (vendor / 'bin/libgcc_s_seh-1.dll').write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    bundle_windows_runtime(vendor, toolchain)
            with patch('scripts.build_desktop.windows_imports', return_value=['private.dll']):
                with self.assertRaisesRegex(ValueError, 'Unapproved'):
                    bundle_windows_runtime(vendor, toolchain)

    def test_macos_floor_uses_minos_and_excludes_linker_tool_version(self):
        modern = 'cmd LC_BUILD_VERSION\ncmdsize 32\nplatform 1\nminos 11.0\nsdk 15.5\nntools 1\ntool LD\nversion 1167.5\n'
        legacy = 'cmd LC_VERSION_MIN_MACOSX\nversion 10.13\nsdk 12.1\n'
        self.assertEqual(macho_minimum_versions(modern + legacy), ['11.0', '10.13'])

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
