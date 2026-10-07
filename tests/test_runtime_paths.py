from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auto_reframe_core.config_store import ConfigStoreError
from auto_reframe_core.runtime_paths import (
    user_data_root, resource_root, tool_path, logs_root, watermark_root,
)
from auto_reframe_core.platform_profile import desktop_target, desktop_asset_name

class RuntimePathsTests(unittest.TestCase):
    def test_native_user_data_locations(self):
        home = Path('/users/test')
        self.assertEqual(user_data_root('darwin', home, {}), home / 'Library/Application Support/Auto Reframe Videos')
        self.assertEqual(user_data_root('win32', home, {'LOCALAPPDATA': '/local'}), Path('/local/Auto Reframe Videos'))

    def test_frozen_resources_and_tools_ignore_cwd_and_stale_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            suffix = '.exe' if __import__('sys').platform == 'win32' else ''
            (root / 'bin' / ('ffmpeg' + suffix)).touch()
            with patch('sys.frozen', True, create=True), patch('sys._MEIPASS', str(root), create=True):
                self.assertEqual(resource_root(), root.resolve())
                self.assertEqual(tool_path('ffmpeg', '/old/ffmpeg'), str(root.resolve() / 'bin' / ('ffmpeg' + suffix)))
                with self.assertRaises(ConfigStoreError):
                    tool_path('ffprobe')




    def test_watermark_and_platform_logs_are_per_user(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertEqual(logs_root('darwin', home, {}, create=False), home / 'Library/Logs/Auto Reframe Videos')
            self.assertEqual(logs_root('win32', home, {'LOCALAPPDATA': str(home / 'local')}, create=False), home / 'local/Auto Reframe Videos/logs')
            with patch('auto_reframe_core.runtime_paths.user_data_root', return_value=home):
                self.assertEqual(watermark_root(), home / 'watermark')
            self.assertFalse((home / 'input').exists())
            self.assertFalse((home / 'output').exists())

    def test_target_cpu_mapping_and_rejection(self):
        self.assertEqual(desktop_target('darwin', 'arm64'), 'macos-arm64')
        self.assertEqual(desktop_target('darwin', 'x86_64'), 'macos-x64')
        self.assertEqual(desktop_target('win32', 'AMD64'), 'windows-x64')
        for system, machine in [('win32', 'ARM64'), ('linux', 'x86_64')]:
            with self.assertRaises(ValueError):
                desktop_target(system, machine)
        self.assertEqual(desktop_asset_name('3.0.10', 'windows-x64'), 'auto-reframe-videos-v3.0.10-windows-x64-Setup.exe')
