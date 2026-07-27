# -*- coding: utf-8 -*-

from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from auto_reframe_core import update_installer
from auto_reframe_core.updater import (
    LATEST_RELEASE_API,
    MANIFEST_NAME,
    UpdateError,
    UpdateInfo,
    can_self_update,
    check_for_update,
    download_update,
    parse_version,
    stage_update,
)


class FakeResponse(BytesIO):
    def __init__(self, payload, url, headers=None):
        super().__init__(payload)
        self._url = url
        self.headers = headers or {}

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def make_info(payload=b"zip", current="2.3.0", latest="2.4.0"):
    name = f"auto-reframe-videos-v{latest}.zip"
    return UpdateInfo(
        current_version=current,
        latest_version=latest,
        tag_name=f"v{latest}",
        asset_name=name,
        download_url=(
            "https://github.com/minhung1126/auto-reframe-videos/releases/download/"
            f"v{latest}/{name}"
        ),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        release_url=(
            "https://github.com/minhung1126/auto-reframe-videos/releases/tag/"
            f"v{latest}"
        ),
        notes="notes",
        published_at="2026-07-27T00:00:00Z",
        immutable=True,
    )


def write_update_archive(path, version, files, extra_entries=()):
    manifest_files = []
    for name, payload, mode in files:
        manifest_files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "mode": mode,
            }
        )
    manifest = {
        "format_version": 1,
        "version": version,
        "files": manifest_files,
    }
    prefix = f"auto-reframe-videos-v{version}/"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, payload, mode in files:
            entry = zipfile.ZipInfo(prefix + name)
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | mode) << 16
            bundle.writestr(entry, payload)
        bundle.writestr(
            prefix + MANIFEST_NAME,
            json.dumps(manifest).encode("utf-8"),
        )
        for entry, payload in extra_entries:
            bundle.writestr(entry, payload)
    return manifest


def write_installed_manifest(root, version, files):
    manifest_files = []
    for name, payload, mode in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        manifest_files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "mode": mode,
            }
        )
    manifest = {
        "format_version": 1,
        "version": version,
        "files": manifest_files,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class VersionAndDiscoveryTests(unittest.TestCase):
    def test_stable_version_comparison(self):
        self.assertEqual(parse_version("v2.4.0"), (2, 4, 0))
        self.assertGreater(parse_version("2.10.0"), parse_version("2.9.9"))
        with self.assertRaises(UpdateError):
            parse_version("2.4")
        with self.assertRaises(UpdateError):
            parse_version("2.4.0-beta")

    def test_check_for_update_requires_canonical_asset_and_digest(self):
        asset_payload = b"archive"
        info = make_info(asset_payload)
        release = {
            "tag_name": info.tag_name,
            "html_url": info.release_url,
            "draft": False,
            "prerelease": False,
            "immutable": True,
            "body": "release notes",
            "published_at": info.published_at,
            "assets": [
                {
                    "name": info.asset_name,
                    "browser_download_url": info.download_url,
                    "digest": f"sha256:{info.sha256}",
                    "size": info.size,
                }
            ],
        }

        def opener(request, timeout):
            self.assertEqual(request.full_url, LATEST_RELEASE_API)
            self.assertEqual(timeout, 15)
            return FakeResponse(
                json.dumps(release).encode("utf-8"),
                LATEST_RELEASE_API,
            )

        result = check_for_update(current_version="2.3.0", opener=opener)
        self.assertTrue(result.available)
        self.assertEqual(result.latest_version, "2.4.0")
        self.assertEqual(result.sha256, info.sha256)
        self.assertTrue(result.immutable)

        release["assets"][0]["browser_download_url"] = "https://example.com/update.zip"
        with self.assertRaisesRegex(UpdateError, "官方下載網址"):
            check_for_update(current_version="2.3.0", opener=opener)

    def test_download_rejects_digest_mismatch_and_cleans_partial_file(self):
        payload = b"downloaded archive"
        info = make_info(payload)

        def opener(_request, timeout):
            self.assertEqual(timeout, 30)
            return FakeResponse(
                payload,
                "https://release-assets.githubusercontent.com/example",
                {"Content-Length": str(len(payload))},
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = download_update(info, destination, opener=opener)
            self.assertEqual(result.read_bytes(), payload)

            bad = UpdateInfo(
                **{**info.__dict__, "sha256": "0" * 64}
            )
            with self.assertRaisesRegex(UpdateError, "SHA-256"):
                download_update(bad, destination, opener=opener)
            self.assertFalse((destination / (info.asset_name + ".part")).exists())


class ArchiveStagingTests(unittest.TestCase):
    def test_stage_valid_archive_and_reject_user_data_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "update.zip"
            files = [
                ("auto_reframe_core/__main__.py", b"print('new')\n", 0o644),
                ("auto_reframe_core/version.py", b"__version__='2.4.0'\n", 0o644),
            ]
            write_update_archive(archive, "2.4.0", files)
            payload = archive.read_bytes()
            info = make_info(payload)
            work = root / "work"
            work.mkdir()
            staged = stage_update(archive, info, work)
            self.assertEqual(
                (staged.staged_root / "auto_reframe_core" / "__main__.py").read_bytes(),
                files[0][1],
            )
            self.assertTrue(staged.manifest_path.is_file())

            bad_archive = root / "bad.zip"
            write_update_archive(
                bad_archive,
                "2.4.0",
                [("config.json", b"{}", 0o644)],
            )
            bad_payload = bad_archive.read_bytes()
            bad_info = make_info(bad_payload)
            bad_work = root / "bad-work"
            bad_work.mkdir()
            with self.assertRaisesRegex(UpdateError, "使用者資料"):
                stage_update(bad_archive, bad_info, bad_work)

    def test_stage_rejects_zip_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "traversal.zip"
            write_update_archive(
                archive,
                "2.4.0",
                [("auto_reframe_core/__main__.py", b"safe", 0o644)],
                [
                    (
                        "auto-reframe-videos-v2.4.0/../outside.py",
                        b"unsafe",
                    )
                ],
            )
            payload = archive.read_bytes()
            info = make_info(payload)
            work = root / "work"
            work.mkdir()
            with self.assertRaisesRegex(UpdateError, "不安全路徑"):
                stage_update(archive, info, work)
            self.assertFalse((root / "outside.py").exists())

    def test_stage_rejects_reserved_windows_name_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserved_archive = root / "reserved.zip"
            write_update_archive(
                reserved_archive,
                "2.4.0",
                [("CON.txt", b"unsafe", 0o644)],
            )
            payload = reserved_archive.read_bytes()
            work = root / "reserved-work"
            work.mkdir()
            with self.assertRaisesRegex(UpdateError, "不安全路徑"):
                stage_update(reserved_archive, make_info(payload), work)

            symlink_archive = root / "symlink.zip"
            write_update_archive(
                symlink_archive,
                "2.4.0",
                [("auto_reframe_core/__main__.py", b"safe", 0o644)],
            )
            with zipfile.ZipFile(symlink_archive, "a") as bundle:
                entry = zipfile.ZipInfo(
                    "auto-reframe-videos-v2.4.0/link-to-outside"
                )
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, "../outside")
            payload = symlink_archive.read_bytes()
            symlink_work = root / "symlink-work"
            symlink_work.mkdir()
            with self.assertRaisesRegex(UpdateError, "符號連結"):
                stage_update(symlink_archive, make_info(payload), symlink_work)


class InstallerTests(unittest.TestCase):
    def test_self_update_accepts_unified_and_legacy_install_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified = root / "auto_reframe_core" / "__main__.py"
            unified.parent.mkdir()
            unified.write_text("", encoding="utf-8")
            self.assertEqual(can_self_update(root), (True, ""))

            unified.unlink()
            legacy = root / "auto_reframe_gui.py"
            legacy.write_text("", encoding="utf-8")
            self.assertEqual(can_self_update(root), (True, ""))

            legacy.unlink()
            allowed, reason = can_self_update(root)
            self.assertFalse(allowed)
            self.assertIn("統一入口", reason)

    def test_transaction_updates_managed_files_and_preserves_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "app"
            staged = Path(directory) / "staged"
            root.mkdir()
            staged.mkdir()
            old_files = [
                ("auto_reframe_gui.py", b"old gui", 0o644),
                ("obsolete.py", b"old obsolete", 0o644),
            ]
            write_installed_manifest(root, "2.3.0", old_files)
            (root / "config.json").write_text('{"personal": true}', encoding="utf-8")
            (root / "input").mkdir()
            (root / "input" / "video.mp4").write_bytes(b"personal video")
            (root / "watermark").mkdir()
            (root / "watermark" / "mine.png").write_bytes(b"personal watermark")

            new_files = [
                ("auto_reframe_core/__main__.py", b"new entry", 0o644),
                ("new_module.py", b"new module", 0o644),
            ]
            write_installed_manifest(staged, "2.4.0", new_files)
            manifest_path = staged / MANIFEST_NAME

            update_installer.apply_update(
                root,
                staged,
                manifest_path,
                "2.4.0",
            )
            self.assertEqual(
                (root / "auto_reframe_core" / "__main__.py").read_bytes(),
                b"new entry",
            )
            self.assertFalse((root / "auto_reframe_gui.py").exists())
            self.assertEqual((root / "new_module.py").read_bytes(), b"new module")
            self.assertFalse((root / "obsolete.py").exists())
            self.assertEqual(
                (root / "config.json").read_text(encoding="utf-8"),
                '{"personal": true}',
            )
            self.assertEqual((root / "input" / "video.mp4").read_bytes(), b"personal video")
            self.assertEqual(
                (root / "watermark" / "mine.png").read_bytes(),
                b"personal watermark",
            )
            self.assertFalse((root / MANIFEST_NAME).exists())
            self.assertFalse((root / ".update-backups").exists())
            self.assertEqual(list(root.glob(".update-rollback-*")), [])

    def test_transaction_refuses_local_managed_changes_and_git_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "app"
            staged = Path(directory) / "staged"
            root.mkdir()
            staged.mkdir()
            write_installed_manifest(
                root,
                "2.3.0",
                [("auto_reframe_gui.py", b"old gui", 0o644)],
            )
            (root / "auto_reframe_gui.py").write_bytes(b"locally edited")
            write_installed_manifest(
                staged,
                "2.4.0",
                [("auto_reframe_core/__main__.py", b"new entry", 0o644)],
            )
            with self.assertRaisesRegex(
                update_installer.InstallError,
                "Locally modified",
            ):
                update_installer.apply_update(
                    root,
                    staged,
                    staged / MANIFEST_NAME,
                    "2.4.0",
                )

            (root / ".git").mkdir()
            allowed, reason = can_self_update(root)
            self.assertFalse(allowed)
            self.assertIn("Git", reason)

    def test_transaction_rolls_back_if_replacement_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "app"
            staged = Path(directory) / "staged"
            root.mkdir()
            staged.mkdir()
            write_installed_manifest(
                root,
                "2.3.0",
                [
                    ("auto_reframe_gui.py", b"old gui", 0o644),
                    ("module.py", b"old module", 0o644),
                ],
            )
            write_installed_manifest(
                staged,
                "2.4.0",
                [
                    ("auto_reframe_core/__main__.py", b"new entry", 0o644),
                    ("module.py", b"new module", 0o644),
                ],
            )
            original_copy = update_installer._copy_into_place
            calls = {"count": 0}

            def fail_once(source, target, mode):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("simulated replacement failure")
                return original_copy(source, target, mode)

            with patch(
                "auto_reframe_core.update_installer._copy_into_place",
                side_effect=fail_once,
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    update_installer.apply_update(
                        root,
                        staged,
                        staged / MANIFEST_NAME,
                        "2.4.0",
                    )

            self.assertEqual((root / "auto_reframe_gui.py").read_bytes(), b"old gui")
            self.assertFalse((root / "auto_reframe_core" / "__main__.py").exists())
            self.assertEqual((root / "module.py").read_bytes(), b"old module")
            self.assertEqual(
                json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))["version"],
                "2.3.0",
            )

    def test_installer_cleanup_only_removes_owned_temp_directory(self):
        owned = Path(tempfile.mkdtemp(prefix="auto-reframe-update-"))
        plan = owned / "install-plan.json"
        plan.write_text("{}", encoding="utf-8")
        update_installer._cleanup_work_dir(plan)
        self.assertFalse(owned.exists())

        unrelated = Path(tempfile.mkdtemp(prefix="unrelated-update-"))
        try:
            plan = unrelated / "install-plan.json"
            plan.write_text("{}", encoding="utf-8")
            update_installer._cleanup_work_dir(plan)
            self.assertTrue(unrelated.exists())
        finally:
            shutil.rmtree(unrelated, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
