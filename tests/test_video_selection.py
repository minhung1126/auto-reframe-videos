"""Original-file selection and multi-destination safety contracts."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from auto_reframe_core.compress import CompressConfig
from auto_reframe_core.reframe import ReframeConfig
from auto_reframe_core.output_plans import preflight_outputs, delete_planned_conflicts
from auto_reframe_core.video_list import collect_videos
from auto_reframe_core.batch_runner import run_video_batch
from auto_reframe_core.gui import AutoReframeGUI, build_job_confirmation_message
from auto_reframe_core.config_store import load_config

INFO = {'width': 320, 'height': 180, 'fps': 30}


class SelectionTests(unittest.TestCase):
    def test_folder_depth_duplicates_and_rejected_items(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / '中文 空白.MP4'
            first.write_bytes(b'video')
            nested = root / 'nested'
            nested.mkdir()
            second = nested / 'second.mov'
            second.write_bytes(b'video')
            invalid = root / 'note.txt'
            invalid.write_text('note')
            videos, errors = collect_videos((root, first, invalid, root / 'missing.mp4'))
            self.assertEqual(videos, (first.resolve(),))
            self.assertEqual(len(errors), 2)
            videos, errors = collect_videos((root,), videos, recursive=True)
            self.assertEqual(set(videos), {first.resolve(), second.resolve()})
            self.assertFalse(errors)

    def test_empty_files_and_file_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original.mp4'
            original.write_bytes(b'video')
            alias = root / 'alias.mp4'
            try:
                alias.hardlink_to(original)
            except OSError:
                self.skipTest('Hard links unavailable')
            empty = root / 'empty.mp4'
            empty.touch()
            videos, errors = collect_videos((original, alias, empty))
            self.assertEqual(videos, (original.resolve(),))
            self.assertEqual(len(errors), 1)
            self.assertIn('檔案是空的', errors[0])

    def test_snapshot_survives_selection_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, later = root / 'first.mp4', root / 'later.mp4'
            first.write_bytes(b'a')
            later.write_bytes(b'b')
            selection, _ = collect_videos((first,))
            config = CompressConfig(video_files=selection, output_mode='source', max_workers=1)
            selection, _ = collect_videos((later,), selection)
            with patch('auto_reframe_core.video_utils.get_video_info', return_value=INFO), \
                 patch('auto_reframe_core.batch_runner.run_parallel', return_value=(1, [])) as parallel:
                self.assertEqual(run_video_batch(config, Mock(), '測試'), (1, []))
            tasks = parallel.call_args.args[0]
            self.assertEqual(tasks, [(1, 1, first.resolve())])
            self.assertEqual(len(selection), 2)
            self.assertFalse((root / 'input').exists())


class MultiOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = []
        for name in ('A', 'B'):
            folder = self.root / name
            folder.mkdir()
            video = folder / 'same.mp4'
            video.write_bytes(b'original')
            self.sources.append(video)
        self.probe = patch('auto_reframe_core.video_utils.get_video_info', return_value=INFO)
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def config(self, mode='compress', **kwargs):
        cls = CompressConfig if mode == 'compress' else ReframeConfig
        return cls(video_files=tuple(self.sources), output_mode='source', **kwargs)

    def test_same_names_in_distinct_source_destinations_are_safe(self):
        for mode in ('compress', 'reframe'):
            destinations, finals = preflight_outputs(self.config(mode), mode)
            self.assertEqual(set(destinations), {p.parent / 'auto-reframe' for p in self.sources})
            self.assertGreaterEqual(len(finals), 2)
            self.assertEqual(len(finals), len(set(finals)))
            self.assertFalse(any(p.exists() for p in finals))
            self.assertFalse(list(self.root.rglob('.arv-write-*')))

    def test_combined_folder_collision_is_refused_before_creating_output(self):
        output = self.root / 'exports'
        for mode in ('compress', 'reframe'):
            config = replace(self.config(mode), output_mode='specified', output_dir=str(output))
            with self.assertRaisesRegex(ValueError, '碰撞'):
                preflight_outputs(config, mode)
            self.assertFalse(output.exists())
        self.assertTrue(all(p.read_bytes() == b'original' for p in self.sources))

    def test_native_case_collisions_are_refused(self):
        renamed = self.sources[1].with_name('SAME.mp4')
        self.sources[1].rename(renamed)
        self.sources[1] = renamed
        for platform in ('win32', 'darwin'):
            config = replace(self.config(), output_mode='specified', output_dir=str(self.root / 'combined'))
            with patch('auto_reframe_core.output_plans.sys.platform', platform):
                with self.assertRaisesRegex(ValueError, '碰撞'):
                    preflight_outputs(config, 'compress')

    def test_batch_cleans_only_its_planned_stale_temps(self):
        config = self.config(max_workers=1)
        _, finals = preflight_outputs(config, 'compress')
        stale = finals[0].with_name(finals[0].name + '.tmp')
        stale.write_bytes(b'incomplete')
        unrelated = finals[0].parent / 'unrelated.tmp'
        unrelated.write_bytes(b'keep')
        with patch('auto_reframe_core.batch_runner.run_parallel', return_value=(2, [])):
            run_video_batch(config, Mock(), '測試')
        self.assertFalse(stale.exists())
        self.assertEqual(unrelated.read_bytes(), b'keep')

    def test_original_selected_file_cannot_be_used_as_an_output(self):
        config = replace(self.config(), video_files=(self.sources[0],))
        _, finals = preflight_outputs(config, 'compress')
        original = finals[0]
        original.write_bytes(b'original target-looking filename')
        config.video_files += (original,)
        with self.assertRaisesRegex(ValueError, '原始影片'):
            preflight_outputs(config, 'compress')
        self.assertEqual(original.read_bytes(), b'original target-looking filename')

    def test_unreadable_video_and_unwritable_output_are_errors(self):
        with patch('auto_reframe_core.video_utils.get_video_info', return_value=None):
            with self.assertRaisesRegex(ValueError, '無法讀取影片資訊'):
                preflight_outputs(self.config(), 'compress')
        with patch('auto_reframe_core.output_plans.tempfile.NamedTemporaryFile', side_effect=PermissionError('read-only')):
            with self.assertRaises(PermissionError):
                preflight_outputs(self.config(), 'compress')

    def test_existing_actions_touch_only_planned_files(self):
        _, finals = preflight_outputs(self.config(), 'compress')
        for final in finals:
            final.write_bytes(b'old')
        unrelated = finals[0].parent / 'unrelated.mp4'
        unrelated.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, '未規劃'):
            delete_planned_conflicts((unrelated,), finals)
        delete_planned_conflicts(finals, finals)
        self.assertEqual(unrelated.read_bytes(), b'keep')
        self.assertTrue(all(not p.exists() for p in finals))

    def test_confirmation_lists_count_mode_and_every_destination(self):
        message = build_job_confirmation_message('compress', self.config())
        self.assertIn('影片數量：2', message)
        self.assertIn('各影片所在資料夾的 auto-reframe/', message)
        for source in self.sources:
            self.assertIn(str(source.parent / 'auto-reframe'), message)


class ConfigOpeningTests(unittest.TestCase):
    def test_open_watermark_folder_creates_actual_location(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'data/watermark'
            app = AutoReframeGUI.__new__(AutoReframeGUI)
            app.root = object()
            with patch('auto_reframe_core.gui.open_directory') as opener:
                app._open_directory(folder)
                self.assertTrue(folder.is_dir())
                opener.assert_called_once_with(folder)

    def test_open_config_creates_complete_defaults_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data/config.json'
            defaults = load_config(Path(__file__).resolve().parents[1] / 'config.json.example')
            app = AutoReframeGUI.__new__(AutoReframeGUI)
            app.default_settings = defaults
            app.root = object()
            with patch('auto_reframe_core.gui.CONFIG_PATH', path), patch('auto_reframe_core.gui.open_directory') as opener:
                app.open_config()
                self.assertEqual(load_config(path), defaults)
                opener.assert_called_once_with(path)
                path.write_text('externally edited')
                app.open_config()
                self.assertEqual(path.read_text(), 'externally edited')
