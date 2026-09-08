# -*- coding: utf-8 -*-
"""Unit and integration tests for per-mode watermark settings in GUI and config."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auto_reframe_core.config_store import ConfigStoreError, load_config
from auto_reframe_core.gui import (
    AutoReframeGUI,
    DEFAULT_WATERMARK_WIDTH_RATIOS,
    normalize_watermark_settings,
)
from auto_reframe_core.gui_options import WATERMARK_POSITION_LABELS


class WatermarkSettingsNormalizationTests(unittest.TestCase):
    def test_default_empty_settings_yields_per_mode_defaults(self):
        settings = {}
        normalized = normalize_watermark_settings(settings)

        self.assertIn("reframe", normalized)
        self.assertIn("compress", normalized)

        self.assertFalse(normalized["reframe"]["enabled"])
        self.assertEqual(normalized["reframe"]["file"], "")
        self.assertEqual(normalized["reframe"]["position"], "bottom-center")
        self.assertEqual(normalized["reframe"]["width_ratio"], 0.15)
        self.assertEqual(normalized["reframe"]["margin"], 3)
        self.assertNotIn("opacity", normalized["reframe"])

        self.assertFalse(normalized["compress"]["enabled"])
        self.assertEqual(normalized["compress"]["file"], "")
        self.assertEqual(normalized["compress"]["position"], "bottom-center")
        self.assertEqual(normalized["compress"]["width_ratio"], 0.10)
        self.assertEqual(normalized["compress"]["margin"], 3)
        self.assertNotIn("opacity", normalized["compress"])

    def test_legacy_flat_config_migrates_to_both_modes(self):
        legacy_settings = {
            "watermark_enabled": True,
            "watermark_file": "my_logo.png",
            "watermark_position": "top-right",
            "watermark_width_ratio": 0.08,
            "watermark_opacity": 0.5,  # Should be ignored/dropped
            "watermark_margin": 5,
        }
        normalized = normalize_watermark_settings(legacy_settings)

        for mode in ("reframe", "compress"):
            self.assertTrue(normalized[mode]["enabled"])
            self.assertEqual(normalized[mode]["file"], "my_logo.png")
            self.assertEqual(normalized[mode]["position"], "top-right")
            self.assertEqual(normalized[mode]["width_ratio"], 0.08)
            self.assertEqual(normalized[mode]["margin"], 5)
            self.assertNotIn("opacity", normalized[mode])

    def test_legacy_flat_config_without_ratio_uses_mode_defaults(self):
        legacy_settings = {
            "watermark_enabled": True,
            "watermark_file": "my_logo.png",
        }
        normalized = normalize_watermark_settings(legacy_settings)

        self.assertEqual(normalized["reframe"]["width_ratio"], 0.15)
        self.assertEqual(normalized["compress"]["width_ratio"], 0.10)

    def test_valid_per_mode_structure_is_preserved(self):
        settings = {
            "watermarks": {
                "reframe": {
                    "enabled": True,
                    "file": "reframe.png",
                    "position": "bottom-left",
                    "width_ratio": 0.18,
                    "margin": 6,
                },
                "compress": {
                    "enabled": False,
                    "file": "compress.png",
                    "position": "bottom-right",
                    "width_ratio": 0.12,
                    "margin": 2,
                },
            }
        }
        normalized = normalize_watermark_settings(settings)
        self.assertEqual(normalized["reframe"]["file"], "reframe.png")
        self.assertEqual(normalized["reframe"]["position"], "bottom-left")
        self.assertEqual(normalized["reframe"]["width_ratio"], 0.18)
        self.assertEqual(normalized["reframe"]["margin"], 6)

        self.assertEqual(normalized["compress"]["file"], "compress.png")
        self.assertEqual(normalized["compress"]["position"], "bottom-right")
        self.assertEqual(normalized["compress"]["width_ratio"], 0.12)
        self.assertEqual(normalized["compress"]["margin"], 2)

    def test_invalid_position_raises_config_store_error(self):
        settings = {
            "watermarks": {
                "reframe": {"position": "floating-nowhere"},
            }
        }
        with self.assertRaises(ConfigStoreError) as ctx:
            normalize_watermark_settings(settings)
        self.assertIn("position 無效", str(ctx.exception))

    def test_invalid_width_ratio_raises_config_store_error(self):
        with self.subTest(case="negative"):
            with self.assertRaises(ConfigStoreError):
                normalize_watermark_settings({"watermarks": {"reframe": {"width_ratio": -0.1}}})
        with self.subTest(case="too_large"):
            with self.assertRaises(ConfigStoreError):
                normalize_watermark_settings({"watermarks": {"reframe": {"width_ratio": 1.5}}})
        with self.subTest(case="string_not_number"):
            with self.assertRaises(ConfigStoreError):
                normalize_watermark_settings({"watermarks": {"reframe": {"width_ratio": "abc"}}})

    def test_invalid_margin_raises_config_store_error(self):
        with self.subTest(case="negative"):
            with self.assertRaises(ConfigStoreError):
                normalize_watermark_settings({"watermarks": {"compress": {"margin": -1}}})
        with self.subTest(case="too_large"):
            with self.assertRaises(ConfigStoreError):
                normalize_watermark_settings({"watermarks": {"compress": {"margin": 105}}})

    def test_load_effective_settings_migrates_legacy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            example_file = tmp_path / "config.json.example"
            config_file = tmp_path / "config.json"

            example_settings = {
                "version": 1,
                "settings": {
                    "mode": "reframe",
                    "watermarks": {
                        "reframe": {
                            "enabled": False,
                            "file": "",
                            "position": "bottom-center",
                            "width_ratio": 0.15,
                            "margin": 3,
                        },
                        "compress": {
                            "enabled": False,
                            "file": "",
                            "position": "bottom-center",
                            "width_ratio": 0.10,
                            "margin": 3,
                        },
                    },
                },
            }
            example_file.write_text(json.dumps(example_settings), encoding="utf-8")

            legacy_saved = {
                "version": 1,
                "settings": {
                    "mode": "compress",
                    "watermark_enabled": True,
                    "watermark_file": "my_logo.png",
                    "watermark_position": "top-right",
                    "watermark_width_ratio": 0.08,
                    "watermark_opacity": 0.75,
                    "watermark_margin": 6,
                },
            }
            config_file.write_text(json.dumps(legacy_saved), encoding="utf-8")

            with (
                patch("auto_reframe_core.gui.CONFIG_EXAMPLE_PATH", example_file),
                patch("auto_reframe_core.gui.CONFIG_PATH", config_file),
            ):
                from auto_reframe_core.gui import load_effective_settings
                defaults, effective = load_effective_settings()
                normalized = normalize_watermark_settings(effective)

                for mode in ("reframe", "compress"):
                    self.assertTrue(normalized[mode]["enabled"])
                    self.assertEqual(normalized[mode]["file"], "my_logo.png")
                    self.assertEqual(normalized[mode]["position"], "top-right")
                    self.assertEqual(normalized[mode]["width_ratio"], 0.08)
                    self.assertEqual(normalized[mode]["margin"], 6)
                    self.assertNotIn("opacity", normalized[mode])


def _can_create_tk_root() -> bool:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


HAS_DISPLAY = _can_create_tk_root()


@unittest.skipUnless(HAS_DISPLAY, "Tkinter GUI display environment is required")
class WatermarkGUISavingBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name)
        (self.root_path / "watermark").mkdir()
        (self.root_path / "watermark" / "logo1.png").write_bytes(b"png1")
        (self.root_path / "watermark" / "logo2.png").write_bytes(b"png2")
        self.patcher = patch(
            "auto_reframe_core.gui.WATERMARK_DIR", self.root_path / "watermark"
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def test_gui_initializes_separate_watermark_variables_with_mode_defaults(self):
        with patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)
                # Verify reframe default ratio is 0.15
                self.assertEqual(app.watermark_width_ratio_vars["reframe"].get(), "0.15")
                # Verify compress default ratio is 0.10
                self.assertEqual(app.watermark_width_ratio_vars["compress"].get(), "0.1")
                # Verify margins
                self.assertEqual(app.watermark_margin_vars["reframe"].get(), "3")
                self.assertEqual(app.watermark_margin_vars["compress"].get(), "3")
                # Verify positions default to bottom-center
                self.assertEqual(
                    app.watermark_position_vars["reframe"].get(),
                    WATERMARK_POSITION_LABELS["bottom-center"],
                )
                self.assertEqual(
                    app.watermark_position_vars["compress"].get(),
                    WATERMARK_POSITION_LABELS["bottom-center"],
                )
            finally:
                root.destroy()

    def test_collect_settings_validates_and_rejects_out_of_range_ratio(self):
        with patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)

                # Set invalid width ratio for reframe
                app.watermark_width_ratio_vars["reframe"].set("1.2")
                with self.assertRaises(ValueError) as ctx:
                    app._collect_settings()
                self.assertIn("裁切重製", str(ctx.exception))
                self.assertIn("0.01 與 1.0", str(ctx.exception))

                # Fix reframe, set invalid width ratio for compress
                app.watermark_width_ratio_vars["reframe"].set("0.15")
                app.watermark_width_ratio_vars["compress"].set("abc")
                with self.assertRaises(ValueError) as ctx:
                    app._collect_settings()
                self.assertIn("影片壓縮", str(ctx.exception))
                self.assertIn("必須是數字", str(ctx.exception))
            finally:
                root.destroy()

    def test_collect_settings_validates_and_rejects_out_of_range_margin(self):
        with patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)

                # Set invalid margin for compress
                app.watermark_margin_vars["compress"].set("-5")
                with self.assertRaises(ValueError) as ctx:
                    app._collect_settings()
                self.assertIn("影片壓縮", str(ctx.exception))
                self.assertIn("0 與 100", str(ctx.exception))
            finally:
                root.destroy()

    def test_save_settings_aborts_and_does_not_corrupt_config_on_error(self):
        config_file = self.root_path / "config.json"
        with (
            patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path),
            patch("auto_reframe_core.gui.CONFIG_PATH", config_file),
            patch("auto_reframe_core.gui.messagebox.showerror") as mock_err,
        ):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)
                app.watermark_width_ratio_vars["reframe"].set("999")

                app.save_settings()

                self.assertTrue(mock_err.called)
                self.assertFalse(config_file.exists())
            finally:
                root.destroy()

    def test_save_settings_writes_per_mode_watermarks_without_opacity(self):
        config_file = self.root_path / "config.json"
        with (
            patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path),
            patch("auto_reframe_core.gui.CONFIG_PATH", config_file),
            patch("auto_reframe_core.gui.messagebox.showinfo"),
        ):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)
                app.watermark_enabled_vars["reframe"].set(True)
                app.watermark_file_vars["reframe"].set("logo1.png")
                app.watermark_width_ratio_vars["reframe"].set("0.18")
                app.watermark_margin_vars["reframe"].set("4")

                app.watermark_enabled_vars["compress"].set(False)
                app.watermark_file_vars["compress"].set("logo2.png")
                app.watermark_width_ratio_vars["compress"].set("0.09")
                app.watermark_margin_vars["compress"].set("2")

                app.save_settings()

                self.assertTrue(config_file.exists())
                data = json.loads(config_file.read_text(encoding="utf-8"))
                settings = data["settings"]
                self.assertIn("watermarks", settings)
                self.assertNotIn("watermark_opacity", settings)
                self.assertNotIn("watermark_enabled", settings)

                wm_rf = settings["watermarks"]["reframe"]
                self.assertTrue(wm_rf["enabled"])
                self.assertEqual(wm_rf["file"], "logo1.png")
                self.assertEqual(wm_rf["width_ratio"], 0.18)
                self.assertEqual(wm_rf["margin"], 4)
                self.assertNotIn("opacity", wm_rf)

                wm_cp = settings["watermarks"]["compress"]
                self.assertFalse(wm_cp["enabled"])
                self.assertEqual(wm_cp["file"], "logo2.png")
                self.assertEqual(wm_cp["width_ratio"], 0.09)
                self.assertEqual(wm_cp["margin"], 2)
                self.assertNotIn("opacity", wm_cp)
            finally:
                root.destroy()

    def test_restore_default_settings_resets_both_tabs(self):
        config_file = self.root_path / "config.json"
        with (
            patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path),
            patch("auto_reframe_core.gui.CONFIG_PATH", config_file),
            patch("auto_reframe_core.gui.messagebox.showinfo"),
        ):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)

                # Mutate both tabs
                app.watermark_enabled_vars["reframe"].set(True)
                app.watermark_width_ratio_vars["reframe"].set("0.25")
                app.watermark_width_ratio_vars["compress"].set("0.05")

                app.save_settings()
                self.assertTrue(config_file.exists())

                app.restore_default_settings()
                self.assertFalse(config_file.exists())

                # Verified restored
                self.assertFalse(app.watermark_enabled_vars["reframe"].get())
                self.assertEqual(app.watermark_width_ratio_vars["reframe"].get(), "0.15")
                self.assertEqual(app.watermark_width_ratio_vars["compress"].get(), "0.1")
            finally:
                root.destroy()

    def test_upgrade_from_legacy_config_json_preserves_user_settings(self):
        config_file = self.root_path / "config.json"
        legacy_payload = {
            "version": 1,
            "settings": {
                "mode": "compress",
                "targets": {
                    "reframe": [{"ratio": [4, 5], "resolution": "1080p", "vcodec": "h264"}],
                    "compress": [{"resolution": "source", "vcodec": "h265"}],
                },
                "final_ratio": [9, 16],
                "watermark_enabled": True,
                "watermark_file": "logo1.png",
                "watermark_position": "top-left",
                "watermark_width_ratio": 0.08,
                "watermark_opacity": 0.75,
                "watermark_margin": 6,
                "font_path": "fonts/NotoSerifTC.ttf",
                "font_color": "white",
                "top_font_size": 48,
                "bottom_font_size": 24,
                "text_margin": 20,
                "top_spacing": 1.08,
                "bottom_spacing": 1.2,
                "ffmpeg": "ffmpeg",
                "ffprobe": "ffprobe",
                "workers": 0,
                "skip_existing": True,
                "debug": False,
            },
        }
        config_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

        with (
            patch("auto_reframe_core.gui.SCRIPT_DIR", self.root_path),
            patch("auto_reframe_core.gui.CONFIG_PATH", config_file),
            patch("auto_reframe_core.gui.messagebox.showinfo"),
        ):
            import tkinter as tk
            root = tk.Tk()
            try:
                root.withdraw()
                app = AutoReframeGUI(root)

                # Verified legacy settings migrated to widgets in both tabs
                for mode in ("reframe", "compress"):
                    self.assertTrue(app.watermark_enabled_vars[mode].get())
                    self.assertEqual(app.watermark_file_vars[mode].get(), "logo1.png")
                    self.assertEqual(
                        app.watermark_position_vars[mode].get(),
                        WATERMARK_POSITION_LABELS["top-left"],
                    )
                    self.assertEqual(app.watermark_width_ratio_vars[mode].get(), "0.08")
                    self.assertEqual(app.watermark_margin_vars[mode].get(), "6")

                # Saving should convert to new format without opacity
                app.save_settings()

                saved_data = json.loads(config_file.read_text(encoding="utf-8"))
                saved_settings = saved_data["settings"]
                self.assertIn("watermarks", saved_settings)
                self.assertNotIn("watermark_opacity", saved_settings)
                self.assertNotIn("watermark_enabled", saved_settings)
                self.assertEqual(saved_settings["watermarks"]["reframe"]["position"], "top-left")
                self.assertEqual(saved_settings["watermarks"]["reframe"]["width_ratio"], 0.08)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
