# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import unittest

from scripts.build_release import application_files, manifest_for, project_version


class ReleaseBuilderTests(unittest.TestCase):
    def test_release_allowlist_contains_runtime_and_excludes_personal_data(self):
        paths = application_files()
        relative = {
            path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
            for path in paths
        }
        self.assertIn("auto_reframe_core/__main__.py", relative)
        self.assertIn("auto_reframe_core/cli.py", relative)
        self.assertIn("auto_reframe_core/gui.py", relative)
        self.assertIn("auto_reframe_core/updater.py", relative)
        self.assertIn("run.bat", relative)
        self.assertIn("run.command", relative)
        self.assertIn("fonts/NotoSerifTC.ttf", relative)
        self.assertIn("fonts/LICENSE", relative)
        self.assertIn("SECURITY.md", relative)
        self.assertFalse(any(name.startswith("watermark/") for name in relative))
        self.assertFalse(any(name.startswith("tests/") for name in relative))
        self.assertFalse(any(name.startswith(".github/") for name in relative))
        self.assertNotIn("AGENTS.md", relative)
        self.assertNotIn("config.json", relative)
        self.assertNotIn("auto_reframe.py", relative)
        self.assertNotIn("auto_compress.py", relative)
        self.assertNotIn("auto_reframe_gui.py", relative)
        self.assertNotIn("video_utils.py", relative)

    def test_manifest_matches_project_version_and_has_unique_paths(self):
        version = project_version()
        manifest, payloads = manifest_for(version, application_files())
        paths = [item["path"] for item in manifest["files"]]
        modes = {item["path"]: item["mode"] for item in manifest["files"]}
        self.assertEqual(manifest["version"], version)
        self.assertEqual(len(paths), len(set(path.casefold() for path in paths)))
        self.assertEqual(set(paths), set(payloads))
        self.assertEqual(modes["run.command"], 0o755)
        self.assertEqual(modes["run.bat"], 0o644)

    def test_bundled_font_hash_and_license_notice_are_pinned(self):
        root = Path(__file__).resolve().parents[1]
        font = root / "fonts" / "NotoSerifTC.ttf"
        digest = hashlib.sha256(font.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "0077e18f57c6908f4a000969880940bdb0dad057c0e8d98b49dc364c3d1b09c6",
        )
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn(digest, notices)
        self.assertIn("2017-2024 Adobe", notices)
        self.assertIn("SIL Open Font License 1.1", notices)


if __name__ == "__main__":
    unittest.main()
