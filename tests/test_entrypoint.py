# -*- coding: utf-8 -*-

import io
from pathlib import Path
import unittest
from unittest.mock import patch

from auto_reframe_core import cli


ROOT = Path(__file__).resolve().parents[1]


class UnifiedEntrypointTests(unittest.TestCase):
    def test_stdio_is_reconfigured_for_localized_windows_output(self):
        class ReconfigurableStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.settings = None

            def reconfigure(self, **settings):
                self.settings = settings

        stdout = ReconfigurableStream()
        stderr = ReconfigurableStream()
        with patch.object(cli.sys, "stdout", stdout), patch.object(
            cli.sys, "stderr", stderr
        ):
            cli.configure_utf8_stdio()

        expected = {"encoding": "utf-8", "errors": "replace"}
        self.assertEqual(stdout.settings, expected)
        self.assertEqual(stderr.settings, expected)

    def test_default_mode_is_gui(self):
        args = cli.build_parser().parse_args([])

        self.assertEqual(args.mode, "gui")

    def test_each_mode_routes_through_the_unified_entry(self):
        targets = {
            "gui": "auto_reframe_core.gui.main",
            "reframe": "auto_reframe_core.reframe.main",
            "compress": "auto_reframe_core.compress.main",
        }
        for mode, target in targets.items():
            with self.subTest(mode=mode), patch(target, return_value=None) as run:
                self.assertEqual(cli.main([mode]), 0)
                run.assert_called_once_with()

    def test_platform_launchers_use_only_the_unified_gui_entry(self):
        for launcher in ("run.bat", "run.command"):
            with self.subTest(launcher=launcher):
                content = (ROOT / launcher).read_text(encoding="utf-8")
                self.assertIn("-m auto_reframe_core gui", content)
                self.assertNotIn("auto_reframe_gui.py", content)


if __name__ == "__main__":
    unittest.main()
