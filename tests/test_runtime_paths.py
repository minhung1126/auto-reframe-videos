import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auto_reframe_core.config_store import ConfigStoreError, load_config, save_config
from auto_reframe_core.runtime_paths import (
    Workspace, user_data_root, resource_root, validate_workspace, load_workspace,
    save_workspace, import_legacy_settings, tool_path, migrate_legacy_project,
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
                with self.assertRaises(ConfigStoreError):
                    validate_workspace(root / 'videos')

    def test_workspace_survives_bundle_moves_and_setting_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            data, videos = Path(directory) / 'data', Path(directory) / '中文 空白'
            with patch('auto_reframe_core.runtime_paths.is_frozen', return_value=False):
                save_workspace(Workspace(videos), data)
                save_config(data / 'config.json', {'mode': 'compress'})
                (data / 'config.json').unlink()
                with patch('auto_reframe_core.runtime_paths.resource_root', return_value=Path('/new/app')):
                    workspace = load_workspace(data)
            self.assertEqual(workspace.root, videos.resolve())
            self.assertTrue(all(p.is_dir() for p in (workspace.input, workspace.output, workspace.watermark)))

    def test_first_launch_cancellation_does_not_create_data(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            with patch('auto_reframe_core.runtime_paths.is_frozen', return_value=True):
                self.assertIsNone(load_workspace(data))
            self.assertFalse(data.exists())

    def test_legacy_import_preserves_original_and_normalizes_text(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'old'
            project.mkdir()
            original = project / 'config.json'
            save_config(original, {'mode': 'compress', 'ffmpeg': '/old/ffmpeg'})
            before = original.read_bytes()
            (project / 'top_text.txt').write_text('\ufeff第一行\n\n第二行\n', encoding='utf-8')
            destination = Path(directory) / 'new/config.json'
            with patch('auto_reframe_core.runtime_paths.is_frozen', return_value=True):
                imported = import_legacy_settings(project, destination)
            self.assertEqual(imported['top_text'], '第一行\n\n第二行')
            self.assertEqual(load_config(destination)['ffmpeg'], 'ffmpeg')
            self.assertEqual(original.read_bytes(), before)
            with self.assertRaises(ConfigStoreError):
                import_legacy_settings(project, original)

    def test_invalid_import_does_not_replace_existing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_config(root / 'config.json', {'mode': 'invalid'})
            destination = root / 'new.json'
            save_config(destination, {'mode': 'compress'})
            def reject(_settings):
                raise ConfigStoreError('invalid settings')
            with self.assertRaises(ConfigStoreError):
                import_legacy_settings(root, destination, reject)
            self.assertEqual(load_config(destination), {'mode': 'compress'})

    def test_migration_rolls_back_both_documents_on_workspace_save_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, data = root / 'old', root / 'data'
            project.mkdir()
            save_config(project / 'config.json', {'mode': 'compress'})
            save_config(data / 'config.json', {'mode': 'reframe'})
            save_config(data / 'workspace.json', {'root': str(root / 'original')})
            originals = {p.name: p.read_bytes() for p in data.iterdir()}
            with patch('auto_reframe_core.runtime_paths.save_workspace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    migrate_legacy_project(project, data)
            self.assertEqual({p.name: p.read_bytes() for p in data.iterdir()}, originals)
            self.assertEqual(load_config(project / 'config.json')['mode'], 'compress')

    def test_target_cpu_mapping_and_rejection(self):
        self.assertEqual(desktop_target('darwin', 'arm64'), 'macos-arm64')
        self.assertEqual(desktop_target('darwin', 'x86_64'), 'macos-x64')
        self.assertEqual(desktop_target('win32', 'AMD64'), 'windows-x64')
        for system, machine in [('win32', 'ARM64'), ('linux', 'x86_64')]:
            with self.assertRaises(ValueError):
                desktop_target(system, machine)
        self.assertEqual(desktop_asset_name('3.0.10', 'windows-x64'), 'auto-reframe-videos-v3.0.10-windows-x64-Setup.exe')
