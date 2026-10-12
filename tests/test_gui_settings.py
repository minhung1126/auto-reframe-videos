"""Settings validation works before Tk widgets or media processing exist."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from auto_reframe_core.config_store import ConfigStoreError, load_config, save_config
from auto_reframe_core.gui_settings import (
    load_effective_settings,
    normalize_gui_settings,
    normalize_target_sets,
    normalize_watermark_settings,
)


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "config.json.example"


class GUISettingsTests(unittest.TestCase):
    def setUp(self):
        self.defaults = load_config(EXAMPLE_PATH)

    def test_null_noninteger_and_nonfinite_ratios_have_field_errors(self):
        for field in ("final_ratio", "target_ratio"):
            for value in (None, True, 1.5, float("inf"), "bad"):
                with self.subTest(field=field, value=value):
                    settings = deepcopy(self.defaults)
                    if field == "final_ratio":
                        settings["final_ratio"] = [value, 16]
                        label = "final_ratio"
                    else:
                        settings["targets"]["reframe"][0]["ratio"] = [4, value]
                        label = "ratio"
                    with self.assertRaisesRegex(ConfigStoreError, label):
                        normalize_target_sets(settings)

    def test_invalid_scalar_fields_report_the_key_before_widget_creation(self):
        cases = (
            ("workers", None), ("workers", -1), ("top_font_size", 0),
            ("bottom_font_size", "bad"), ("text_margin", -1),
            ("top_spacing", "NaN"), ("bottom_spacing", float("inf")),
            ("font_path", None), ("font_path", ""), ("font_color", []),
            ("skip_existing", "false"), ("include_subfolders", 1),
            ("mode", []), ("output_mode", "elsewhere"),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                settings = deepcopy(self.defaults)
                settings[key] = value
                with self.assertRaisesRegex(ConfigStoreError, key):
                    normalize_gui_settings(settings)

    def test_watermark_null_and_boolean_values_are_not_silently_coerced(self):
        for key, value in (
            ("enabled", "false"), ("file", None), ("position", None),
            ("width_ratio", None), ("width_ratio", True),
            ("opacity", None), ("opacity", True), ("margin", 1.5),
        ):
            with self.subTest(key=key, value=value):
                settings = {"watermarks": {"compress": {key: value}}}
                with self.assertRaisesRegex(ConfigStoreError, f"watermarks.compress.{key}"):
                    normalize_watermark_settings(settings)

    def test_defaults_remain_complete_and_numeric_widget_input_is_normalized(self):
        raw = deepcopy(self.defaults)
        raw.update(workers="2", top_font_size="48", top_spacing="1.08")
        raw["watermarks"]["compress"].update(opacity="60%", margin="3")
        normalized = normalize_gui_settings(raw)
        self.assertEqual(normalized["workers"], 2)
        self.assertEqual(normalized["top_font_size"], 48)
        self.assertEqual(normalized["top_spacing"], 1.08)
        self.assertEqual(normalized["watermarks"]["compress"]["opacity"], 0.6)
        self.assertEqual(normalized["watermarks"]["reframe"]["width_ratio"], 0.15)
        self.assertEqual(normalized["watermarks"]["compress"]["width_ratio"], 0.10)
        self.assertEqual(set(normalized), set(self.defaults))
        self.assertEqual(raw["workers"], "2")

    def test_example_is_the_default_source_and_overlay_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            save_config(config, {"workers": "3", "top_text": "標題\n\n第二段\n"})
            defaults, settings = load_effective_settings(EXAMPLE_PATH, config)
            self.assertEqual(settings["workers"], 3)
            self.assertEqual(defaults["workers"], self.defaults["workers"])
            self.assertEqual(settings["top_text"], "標題\n\n第二段\n")
            save_config(config, {"final_ratio": [None, 16]})
            with self.assertRaisesRegex(ConfigStoreError, "final_ratio"):
                load_effective_settings(EXAMPLE_PATH, config)
            with self.assertRaisesRegex(ConfigStoreError, "找不到預設設定"):
                load_effective_settings(Path(temporary) / "missing.json", config)

    def test_settings_module_import_does_not_import_tkinter(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import auto_reframe_core.gui_settings; "
             "assert 'tkinter' not in sys.modules"],
            capture_output=True, text=True,
            cwd=EXAMPLE_PATH.parent,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compress_job_ignores_incomplete_reframe_edits_but_document_rejects_them(self):
        raw = deepcopy(self.defaults)
        raw["mode"] = "compress"
        raw["targets"]["reframe"] = []
        raw["watermarks"]["reframe"]["width_ratio"] = "editing"
        raw.update(final_ratio=[None, 16], top_font_size="editing", font_path=None)
        normalized = normalize_gui_settings(raw, validation_mode="compress")
        self.assertEqual(normalized["targets"]["compress"], self.defaults["targets"]["compress"])
        self.assertEqual(normalized["watermarks"]["compress"]["width_ratio"], 0.10)
        self.assertEqual(normalized["top_font_size"], "editing")
        with self.assertRaises(ConfigStoreError):
            normalize_gui_settings(raw)

    def test_reframe_job_ignores_incomplete_compress_edits_but_document_rejects_them(self):
        raw = deepcopy(self.defaults)
        raw["targets"]["compress"] = []
        raw["watermarks"]["compress"]["margin"] = "editing"
        normalized = normalize_gui_settings(raw, validation_mode="reframe")
        self.assertEqual(normalized["targets"]["reframe"][0]["ratio"], (4, 5))
        with self.assertRaisesRegex(ConfigStoreError, "targets.compress"):
            normalize_gui_settings(raw)

    def test_active_mode_and_shared_errors_match_full_document_validation(self):
        for mode in ("reframe", "compress"):
            for invalid_field in ("workers", "watermark"):
                with self.subTest(mode=mode, invalid_field=invalid_field):
                    raw = deepcopy(self.defaults)
                    raw["mode"] = mode
                    if invalid_field == "workers":
                        raw["workers"] = -1
                    else:
                        raw["watermarks"][mode]["opacity"] = "editing"
                    with self.assertRaises(ConfigStoreError) as full:
                        normalize_gui_settings(raw)
                    with self.assertRaises(ConfigStoreError) as active:
                        normalize_gui_settings(raw, validation_mode=mode)
                    self.assertEqual(str(full.exception), str(active.exception))


if __name__ == "__main__":
    unittest.main()
