# -*- coding: utf-8 -*-

import importlib
import unittest
from unittest.mock import patch

from auto_reframe_core import cli


class UnifiedEntrypointTests(unittest.TestCase):
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

    def test_legacy_modules_alias_package_implementations(self):
        aliases = {
            "auto_reframe": "auto_reframe_core.reframe",
            "auto_compress": "auto_reframe_core.compress",
            "auto_reframe_gui": "auto_reframe_core.gui",
            "video_utils": "auto_reframe_core.video_utils",
        }
        for legacy_name, implementation_name in aliases.items():
            with self.subTest(module=legacy_name):
                self.assertIs(
                    importlib.import_module(legacy_name),
                    importlib.import_module(implementation_name),
                )


if __name__ == "__main__":
    unittest.main()
