from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auto_reframe_core.platform_profile import desktop_asset_name
from auto_reframe_core.updater import LATEST_RELEASE_API, UpdateError, can_self_update, check_for_update, prepare_update


class Response(BytesIO):
    headers = {}
    def geturl(self):
        return LATEST_RELEASE_API


def release_document():
    base = 'https://github.com/minhung1126/auto-reframe-videos/releases'
    names = ['auto-reframe-videos-v3.1.0.zip'] + [desktop_asset_name('3.1.0', target) for target in ('macos-arm64', 'macos-x64', 'windows-x64')]
    return {'tag_name': 'v3.1.0', 'html_url': base + '/tag/v3.1.0', 'assets': [
        {'name': name, 'digest': 'sha256:' + 'a' * 64, 'size': 100,
         'browser_download_url': base + '/download/v3.1.0/' + name} for name in names]}


class DesktopUpdateTests(unittest.TestCase):
    def test_all_three_targets_select_only_native_installer(self):
        release = release_document()
        for target in ('macos-arm64', 'macos-x64', 'windows-x64'):
            with self.subTest(target=target), patch('auto_reframe_core.platform_profile.desktop_target', return_value=target):
                info = check_for_update('3.0.10', opener=lambda *a, **k: Response(json.dumps(release).encode()), desktop=True)
                self.assertEqual(info.asset_name, desktop_asset_name('3.1.0', target))
                self.assertTrue(info.available)
        info = check_for_update('3.0.10', opener=lambda *a, **k: Response(json.dumps(release).encode()), desktop=False)
        self.assertTrue(info.asset_name.endswith('.zip'))

    def test_desktop_missing_asset_never_falls_back_to_source_zip(self):
        release = release_document()
        release['assets'] = release['assets'][:1]
        with patch('auto_reframe_core.platform_profile.desktop_target', return_value='windows-x64'), self.assertRaises(UpdateError):
            check_for_update('3.0.10', opener=lambda *a, **k: Response(json.dumps(release).encode()), desktop=True)

    def test_duplicate_and_foreign_asset_urls_are_rejected(self):
        for duplicate in (True, False):
            release = release_document()
            if duplicate:
                release['assets'].append(release['assets'][-1])
            else:
                release['assets'][-1]['browser_download_url'] = 'https://example.com/Setup.exe'
            with patch('auto_reframe_core.platform_profile.desktop_target', return_value='windows-x64'), self.assertRaises(UpdateError):
                check_for_update('3.0.10', opener=lambda *a, **k: Response(json.dumps(release).encode()), desktop=True)

    def test_frozen_process_cannot_prepare_source_update(self):
        with tempfile.TemporaryDirectory() as directory, patch('auto_reframe_core.updater.is_frozen', return_value=True), patch('auto_reframe_core.updater.download_update') as download:
            self.assertFalse(can_self_update(Path(directory))[0])
            with self.assertRaises(UpdateError):
                prepare_update(None, Path(directory))
            download.assert_not_called()
