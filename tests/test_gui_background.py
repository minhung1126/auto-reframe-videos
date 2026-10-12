"""Background preparation must keep Tk calls and user decisions on the UI thread."""

from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from auto_reframe_core.gui import AutoReframeGUI, OUTPUT_CONFLICT_CANCEL, OUTPUT_CONFLICT_DELETE


class MainThreadVariable:
    def __init__(self, value):
        self.value = value
        self.thread = threading.get_ident()

    def get(self):
        if threading.get_ident() != self.thread:
            raise AssertionError('Tk variable read from background thread')
        return self.value

    def set(self, value):
        if threading.get_ident() != self.thread:
            raise AssertionError('Tk variable written from background thread')
        self.value = value


class GUIBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.app = AutoReframeGUI.__new__(AutoReframeGUI)
        self.app.root = SimpleNamespace(after=Mock())
        self.app.event_queue = queue.Queue()
        self.app.running = self.app.scan_busy = self.app.preflight_busy = False
        self.app.update_busy = False
        self.app.video_files = (Path('existing.mp4'),)
        self.app.recursive_var = MainThreadVariable(True)
        self.app.status_var = MainThreadVariable('ready')
        self.app.skip_existing_var = MainThreadVariable(True)
        self.app.start_button = Mock()
        self.app.save_button = Mock()
        self.app.restore_button = Mock()
        self.app._refresh_files = Mock()
        self.app._append_log = Mock()
        self.app._reset_video_progress = Mock()
        self.app.notebook = Mock()
        self.app.activity_tab = object()
        self.app._run_worker = Mock()
        self.app.check_update_button = Mock()
        self.app.install_update_button = Mock()
        self.app.update_status_var = MainThreadVariable('ready')
        self.app._set_release_notes = Mock()
        self.dialogs = patch('auto_reframe_core.gui.messagebox').start()
        self.addCleanup(patch.stopall)

    def join_worker(self, worker):
        worker.join(5)
        self.assertFalse(worker.is_alive(), 'background worker did not finish')

    def test_scan_returns_before_io_and_only_applies_result_on_main_thread(self):
        entered, release = threading.Event(), threading.Event()
        thread_ids = []

        def scan(items, existing, recursive):
            thread_ids.append(threading.get_ident())
            self.assertEqual(items, ('new.mp4',))
            self.assertEqual(existing, (Path('existing.mp4'),))
            self.assertTrue(recursive)
            entered.set()
            if not release.wait(5):
                raise AssertionError('scan was never released')
            return existing + (Path('new.mp4'),), []

        with patch('auto_reframe_core.video_list.collect_videos', side_effect=scan):
            try:
                self.app._add_items(('new.mp4',))
                self.assertTrue(entered.wait(2))
                self.assertTrue(self.app.scan_busy)
                self.assertEqual(self.app.video_files, (Path('existing.mp4'),))
                self.app._clear_videos()
                self.assertEqual(self.app.video_files, (Path('existing.mp4'),))
                self.app._refresh_files.assert_not_called()
            finally:
                release.set()
                self.join_worker(self.app.scan_worker)
            self.app._drain_events()
        self.assertFalse(self.app.scan_busy)
        self.assertEqual(self.app.video_files, (Path('existing.mp4'), Path('new.mp4')))
        self.assertNotEqual(thread_ids, [threading.get_ident()])
        self.app._refresh_files.assert_called_once()
        self.app.start_button.configure.assert_called_with(state='normal')

    def test_scan_failure_retains_selection_and_restores_controls(self):
        with patch('auto_reframe_core.video_list.collect_videos', side_effect=OSError('unreadable')):
            self.app._add_items(('new.mp4',))
            self.join_worker(self.app.scan_worker)
        self.dialogs.showerror.assert_not_called()
        self.app._drain_events()
        self.assertFalse(self.app.scan_busy)
        self.assertEqual(self.app.video_files, (Path('existing.mp4'),))
        self.dialogs.showerror.assert_called_once()
        self.app.start_button.configure.assert_called_with(state='normal')

    def test_preflight_returns_before_probe_and_confirmation_can_cancel(self):
        entered, release = threading.Event(), threading.Event()
        config = SimpleNamespace(skip_existing=True, prepared_batch=None)
        self.app._build_config = Mock(return_value=('compress', config))
        final = Mock()
        final.exists.return_value = True
        prepared = SimpleNamespace(finals=(final,))
        thread_ids = []

        def prepare(*_args):
            thread_ids.append(threading.get_ident())
            entered.set()
            if not release.wait(5):
                raise AssertionError('preflight was never released')
            return prepared

        with patch('auto_reframe_core.gui.prepare_video_batch', side_effect=prepare), patch(
            'auto_reframe_core.gui.OutputConflictDialog.ask', return_value=OUTPUT_CONFLICT_CANCEL
        ) as ask:
            try:
                self.app.start_job()
                self.assertTrue(entered.wait(2))
                self.assertTrue(self.app.running)
                self.assertTrue(self.app.preflight_busy)
                ask.assert_not_called()
                self.app.start_job()
                self.app._build_config.assert_called_once()
            finally:
                release.set()
                self.join_worker(self.app.worker)
            ask.assert_not_called()
            self.app._drain_events()
            ask.assert_called_once()
        self.assertIs(config.prepared_batch, prepared)
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.preflight_busy)
        self.app._run_worker.assert_not_called()
        self.assertNotEqual(thread_ids, [threading.get_ident()])

    def test_changed_source_after_confirmation_never_deletes_or_starts_encoding(self):
        self.app.running = self.app.preflight_busy = True
        final = Path('existing-output.mp4')
        config = SimpleNamespace(skip_existing=True, prepared_batch=SimpleNamespace(finals=(final,)))
        with patch('auto_reframe_core.gui.OutputConflictDialog.ask', return_value=OUTPUT_CONFLICT_DELETE), patch(
            'auto_reframe_core.gui.build_job_confirmation_message', return_value='confirm'
        ), patch('auto_reframe_core.gui.validate_prepared_batch', side_effect=ValueError('source changed')), patch(
            'auto_reframe_core.gui.delete_planned_conflicts'
        ) as delete:
            self.dialogs.askyesno.return_value = True
            self.app._finish_preflight(('compress', config, (final,)))
            self.join_worker(self.app.worker)
            delete.assert_not_called()
            self.app._run_worker.assert_not_called()
            self.app._drain_events()
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.preflight_busy)
        self.dialogs.showerror.assert_called_once()
        self.app._run_worker.assert_not_called()

    def test_confirmation_failure_restores_controls_and_keeps_queue_dispatch_alive(self):
        self.app.running = self.app.preflight_busy = True
        config = SimpleNamespace(skip_existing=True)
        self.app.event_queue.put(('preflight_done', ('compress', config, ())))
        with patch('auto_reframe_core.gui.build_job_confirmation_message', side_effect=OSError('unavailable')):
            self.app._drain_events()
        self.assertFalse(self.app.running)
        self.assertFalse(self.app.preflight_busy)
        self.app.start_button.configure.assert_called_with(state='normal')
        self.app.root.after.assert_called_once()
        self.dialogs.showerror.assert_called_once()

    def test_prepared_confirmation_uses_cached_destinations_without_filesystem_io(self):
        from auto_reframe_core.compress import CompressConfig
        from auto_reframe_core.gui import build_job_confirmation_message
        config = CompressConfig(video_files=(Path('source.mp4'),))
        config.prepared_batch = SimpleNamespace(destinations=(Path('cached-output'),))
        with patch('auto_reframe_core.gui.output_root_for', side_effect=AssertionError('UI filesystem access')):
            message = build_job_confirmation_message('compress', config)
        self.assertIn('cached-output', message)

    def test_validated_delete_runs_off_main_thread_and_encoding_waits_for_dispatch(self):
        self.app.running = self.app.preflight_busy = True
        final = Path('existing-output.mp4')
        config = SimpleNamespace(skip_existing=False, prepared_batch=SimpleNamespace(finals=(final,)))
        actions = []
        with patch('auto_reframe_core.gui.validate_prepared_batch', side_effect=lambda *_: actions.append(
            ('validate', threading.get_ident())
        )), patch('auto_reframe_core.gui.delete_planned_conflicts', side_effect=lambda *_: actions.append(
            ('delete', threading.get_ident())
        )):
            worker = threading.Thread(target=self.app._validate_job,
                                      args=('compress', config, OUTPUT_CONFLICT_DELETE, (final,)))
            worker.start()
            self.join_worker(worker)
        self.assertEqual([action[0] for action in actions], ['validate', 'delete'])
        self.assertTrue(all(action[1] != threading.get_ident() for action in actions))
        self.app._run_worker.assert_not_called()
        self.app._drain_events()
        self.join_worker(self.app.worker)
        self.app._run_worker.assert_called_once_with('compress', config)
        self.assertTrue(self.app.running)
        self.assertFalse(self.app.preflight_busy)

    def test_update_check_and_install_wait_for_scan_and_preflight(self):
        for phase in ('scan_busy', 'running'):
            with self.subTest(phase=phase):
                setattr(self.app, phase, True)
                with patch('auto_reframe_core.gui.threading.Thread') as thread:
                    self.app.check_for_updates()
                    thread.assert_not_called()
                setattr(self.app, phase, False)
        self.app.scan_busy = True
        with patch('auto_reframe_core.gui.is_frozen') as frozen:
            self.app.install_update()
            frozen.assert_not_called()

    def test_update_check_completion_and_error_restore_idle_controls(self):
        info = SimpleNamespace(available=False, immutable=True, current_version='1.0.0',
                               latest_version='1.0.0', notes='')
        self.app.update_busy = True
        self.app._finish_update_check(info)
        self.assertFalse(self.app.update_busy)
        self.app.start_button.configure.assert_called_with(state='normal')
        self.app.update_busy = True
        self.app.scan_busy = True
        self.app._finish_update_check_error('offline')
        self.assertFalse(self.app.update_busy)
        self.app.start_button.configure.assert_called_with(state='disabled')
        self.app.scan_busy = False
        self.app._finish_update_check_error('offline')
        self.app.start_button.configure.assert_called_with(state='normal')


if __name__ == '__main__':
    unittest.main()
