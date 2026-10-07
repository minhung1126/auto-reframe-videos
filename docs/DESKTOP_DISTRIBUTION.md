# Desktop distribution — first edition

This repository contains the native build, installation and manual-update implementation.
Desktop installers are permanently unsigned for personal use. No developer certificates, paid
signing services or Apple notarization are required. Linux tests do not qualify native installer UX.

## Runtime and data contract

| Item | macOS | Windows |
| --- | --- | --- |
| Current-user installation | `~/Applications/Auto Reframe Videos.app` | `%LOCALAPPDATA%\Programs\Auto Reframe Videos\` |
| System installation | `/Applications/Auto Reframe Videos.app` | `%ProgramFiles%\Auto Reframe Videos\` |
| Settings | `~/Library/Application Support/Auto Reframe Videos/config.json` | `%LOCALAPPDATA%\Auto Reframe Videos\config.json` |
| Watermark | `~/Library/Application Support/Auto Reframe Videos/watermark/` | `%LOCALAPPDATA%\Auto Reframe Videos\watermark\` |
| Logs | `~/Library/Logs/Auto Reframe Videos/` | `%LOCALAPPDATA%\Auto Reframe Videos\logs\` |

Windows Setup offers current-user and all-users installation. Only system installation elevates;
normal application execution and the post-install launch use the original user's privileges.
macOS keeps DMG drag installation without a custom wizard. System installation shares only
application files; each user owns independent settings, watermark PNGs and logs. Upgrades and
uninstall retain these data. Unsigned build policy is unchanged.

Resources resolve from the source root or PyInstaller `_MEIPASS`. Frozen tools use shipped
binaries. Linux development data uses `$XDG_CONFIG_HOME/Auto Reframe Videos`. No first-launch
workspace dialog, `workspace.json`, legacy import/migration, or automatic input/output creation
exists. GUI text is saved in config, without text-file fallback. The About page opens the actual
config with the default application, first atomically creating complete defaults if absent;
external changes load on next launch, and users are instructed to close before editing.

Both modes share original-file selection with multi-file/folder native drag/drop, deduplication,
readability feedback and an immutable running-task snapshot. Default output is `auto-reframe/`
beside each source; specified-folder mode combines outputs. Preflight rejects cross-source path
collisions or original-file overwrite and probes actual destination writes. Confirmation shows
count, mode and every destination. Existing output actions apply only to planned files.
CLI supports `--input-dir` and `--output-dir` without a workspace.

Native drag/drop uses pinned `tkinterdnd2==0.4.3` / TkDND 2.9.3 (macOS arm64), 2.9.4 (other targets) with Tcl/Tk 8.x. The spec includes
only the target platform native directory. Build/audit gates require its binaries and both wrapper
and native licenses. Frozen smoke loads TkDND and registers a drop target. See THIRD_PARTY_NOTICES.

The source updater keeps its ZIP, manifest and transactional installer. Frozen processes cannot
invoke that installer: they select exactly their OS/CPU asset, validate release URL/digest metadata,
save current settings and open the corresponding download. The user closes the app and installs
manually. The browser performs the download; users can compare the published SHA-256 before installing.
No automatic desktop installation is included. Windows Setup uses Restart Manager and an application
mutex to block replacing a live app. The uninstaller only removes installation files. Deleting a
Mac app or uninstalling Windows leaves settings, watermark PNGs and logs intact.

## Reviewed FFmpeg inputs

The default native CI build compiles FFmpeg 8.0.1, x264, x265, FreeType, HarfBuzz and zlib
from the exact commits and SHA-256 archives in `packaging/ffmpeg-sources.json`.
Windows also compiles oneVPL and includes pinned NVIDIA codec and AMD AMF headers.
`scripts/build_ffmpeg_vendor.py` builds static dependencies, copies their license texts,
and records exact corresponding sources and the build recipe. Runtime license texts
are hash-pinned in `packaging/runtime-licenses.json`. No manually hosted bundle is required.

To build this input locally on a native runner with CMake, Ninja, NASM, make and
pkg-config (MinGW64 GCC/MSYS2 on Windows), run:

```bash
python -m scripts.build_ffmpeg_vendor --target macos-arm64 --destination build/vendor
```

An optional reviewed **self-contained** vendor ZIP can replace this input. Its root contains:

```text
bin/ffmpeg                 # Windows: ffmpeg.exe
bin/ffprobe                # Windows: ffprobe.exe
bin/libgcc_s_seh-1.dll      # Windows only, if actually imported; likewise libstdc++/winpthread
licenses/FFmpeg.txt
licenses/x264.txt
licenses/x265.txt
licenses/Python.txt
licenses/Tcl.txt
licenses/Tk.txt
licenses/...               # All other linked component licenses and notices
provenance.json
```

macOS tools may link only `/usr/lib` and `/System/Library` dependencies. Windows tools must
be x64 PE files importing only audited system DLLs or hash-recorded MinGW runtime DLLs
shipped beside the tools. Their transitive imports are audited, and runtime license texts are
included. The build rejects unbundled or unapproved dynamic libraries,
incorrect architectures, altered binary hashes, `--enable-nonfree`, missing GPL configuration,
missing libx264/libx265/AAC, missing platform hardware encoders, and missing filters used by the app.
An encoder's presence is not proof that a particular machine has usable hardware.

Example `provenance.json` structure (replace placeholders with reviewed, exact values):

```json
{
  "target": "macos-arm64",
  "version": "EXACT_FFMPEG_VERSION",
  "source_url": "https://YOUR_SOURCE_ARCHIVE_HOST/EXACT_FFMPEG_SOURCE",
  "build_recipe_url": "https://YOUR_REPOSITORY/EXACT_COMMIT/build-recipe",
  "corresponding_sources": {
    "ffmpeg": "https://YOUR_SOURCE_ARCHIVE_HOST/EXACT_FFMPEG_SOURCE",
    "x264": "https://YOUR_SOURCE_ARCHIVE_HOST/EXACT_X264_SOURCE",
    "x265": "https://YOUR_SOURCE_ARCHIVE_HOST/EXACT_X265_SOURCE"
  },
  "binaries": {
    "ffmpeg": "64_HEX_SHA256",
    "ffprobe": "64_HEX_SHA256"
  },
  "components": ["Record every linked component, version, license and source here"]
}
```

The release operator must audit **all** linked libraries, their versions, license texts, notices,
source availability and reproducible build instructions. libx264/libx265 enable GPL requirements;
retain corresponding source for the distributed versions and satisfy source distribution obligations.
Do not substitute generic upstream homepages for exact corresponding sources. Preserve the
application's All Rights Reserved license and the bundled font's SIL OFL. Python/Tcl/Tk licenses
must match the selected Python runtime. PyInstaller bootloader and certifi license files are copied
from the pinned installed distributions. CA data is bundled so a clean Mac can check HTTPS updates.
Provenance records vendor hashes; the installer sidecar additionally records the final bundle's per-file hashes.

Never put credential-bearing URLs or private build paths into provenance. Vendor bundles go in
ignored `build/vendor/`, not version control. ZIP fetching verifies its pinned SHA-256 before safe
extraction. The default source build verifies every archive before extraction and compilation.

## Native build

Use Python 3.12 containing Tk, on the actual target architecture, then:

```bash
python -m pip install -r packaging/requirements-build.txt
python -m compileall -q auto_reframe_core tests scripts
python -m unittest discover -s tests
python scripts/build_desktop.py --target macos-arm64 --vendor-dir build/vendor
```

Use `macos-x64` on Intel and `windows-x64` on Windows. On Windows, pass `--iscc` with the
full path to Inno Setup 6's `ISCC.exe` if it is not on PATH. Every build is unsigned by default,
and no credentials are read. The older `--unsigned` flag is accepted as a no-op. Assets are:

```text
auto-reframe-videos-vX.Y.Z-macos-arm64.dmg
auto-reframe-videos-vX.Y.Z-macos-x64.dmg
auto-reframe-videos-vX.Y.Z-windows-x64-Setup.exe
```

Each has `.sha256` and `.build.json` sidecars, including version, source commit, dependency
provenance, final bundle hashes and smoke result. Version always comes from
`auto_reframe_core/version.py`. Only explicitly selected code, defaults, tools, font and licenses
are included; the repository, settings, videos, tests and personal files are not collected wholesale.
An additional bundle audit rejects runtime/private files.

The macOS build declares **13.0 as a candidate floor**. Every Mach-O must include the requested
CPU architecture and declare a minimum OS no newer than 13.0. Python/Tk and FFmpeg compatibility
must still be proven on macOS 13 physical machines before this is a supported-version promise.
Windows Setup requires 10.0.19045 (10 22H2); Windows 11 is the primary validation platform.
Windows ARM64, stores and automatic desktop updating are outside this edition.

## CI and personal installation

`.github/workflows/desktop.yml` runs three native jobs (`macos-15` arm64,
`macos-15-intel` x64 and `windows-latest` x64). Both manual and tag-triggered builds are always
unsigned. The release workflow checks that all three assets match the version, commit and checksums;
it does not require signing secrets, certificates, notarization or timestamp services.

Optionally set repository variable `DESKTOP_VENDOR_MANIFEST` to a JSON object mapping `macos-arm64`,
`macos-x64`, `windows-x64` to reviewed `{"url": "https://...vendor.zip", "sha256": "..."}` entries.
Without that variable, each runner builds the pinned sources automatically. Do not publish from a dirty worktree.
Ordinary CI also builds all three installers before a release tag is created.

On macOS, drag the app into Applications. If macOS blocks first launch, use System Settings →
Privacy & Security → Open Anyway for this app and confirm. On Windows, run Setup.exe; if SmartScreen
shows its warning for the installer you built, use More info → Run anyway. Organization policies may
hide these actions. Global security settings do not need to be disabled.

PyInstaller may automatically add the ad-hoc structural signature required to execute Mach-O code
on Apple Silicon. This uses no developer identity, certificate or notarization and is not a publisher
signature. Application and installer build metadata records `signed: false` for developer signing.

## Automated and physical acceptance

The frozen executable's `--desktop-smoke REPORT.json` invokes the bundled Tk and creates synthetic
one-second media with Chinese/space paths. It tests software Reframe/Compress, H.264/H.265,
multiline Chinese captions with literal `%`, PNG opacity and two targets per mode. It requires
four complete valid MP4 outputs and no remaining `.tmp` files. The build invokes it with empty PATH
and without external Python/Tcl/Tk environment overrides; all test media stays in temporary storage.
This establishes bundled resource use, not full clean-machine or hardware qualification.

Record evidence per target, installer version, machine, OS and GPU in this checklist:

| Acceptance | macOS arm64 | macOS Intel | Windows x64 |
| --- | --- | --- | --- |
| No installed Python/Tk/FFmpeg; install and transcode | Pending | Pending | Pending |
| Finder launch / no console window; Chinese and space paths | Pending | Pending | Pending |
| Top/video/bottom layers, fonts, literal text, watermark, multiple targets | Pending | Pending | Pending |
| VideoToolbox / available NVENC, AMF, QSV; actual software fallback | Pending | Pending | Pending |
| Cancel all workers; remove incomplete tmp; retain completed finals | Pending | Pending | Pending |
| First launch, config open/save/reset, watermark folder opening | Pending | Pending | Pending |
| Move app, overwrite upgrade, upgrade while running | Pending | Pending | Pending |
| Delete/uninstall retains per-user settings, watermark and logs | Pending | Pending | Pending |
| Personal installation via per-app OS confirmation | Pending | Pending | Pending |
| macOS 13 / Windows 10 22H2 and Windows 11 | Pending | Pending | Pending |
| License/source-offer audit and artifact privacy scan | Pending | Pending | Pending |

CI cannot simulate physical GPUs/installer UX. Attach the completed
matrix and logs to the release review. Second edition: evaluate Sparkle for macOS and an independent
Windows update helper/installer; keep the source updater separate.

## Local frozen verification without native test machines

On Linux, `scripts.verify_frozen_local` exercises the **same production PyInstaller spec** using
local FFmpeg tools as test fixtures. It builds a real frozen executable, audits its resources and
runs the bundled Tk/transcoding self-test with empty PATH and no external Python/Tcl/Tk overrides:

```bash
python -m pip install -r packaging/requirements-build.txt
xvfb-run -a python -m scripts.verify_frozen_local
```

Install Xvfb, xauth, FFmpeg and Python's Tk support first. The output report is
`build/frozen-verification/smoke.json`; the test bundle is under the adjacent `dist/` directory.
This directory is ignored and is **not** a licensed Linux release or a macOS/Windows installer.
Native installer, hardware and minimum-OS qualification remain separate.
The ordinary CI workflow runs this test automatically and stores its report.

The current drag/drop runtime requires Tcl/Tk 8.x. Previous Tcl/Tk 9-only smoke evidence does
not qualify this build. Native runners use Python 3.12 / Tk 8.6; the build gate rejects incompatible
Tk versions and frozen smoke must load TkDND successfully.

## Required native UX qualification (pending)

Linux checks cannot qualify Setup.exe or DMG. For each Windows installation scope, verify default
paths, installer-only elevation, ordinary daily execution, another user's independent data, and
upgrade/uninstall data retention. On macOS test both drag destinations and data retention.
On both OSes verify first launch without workspace prompts; drag multiple files/folders including
spaces and Chinese names; nested-folder selection; duplicate/unreadable feedback; both output
modes with multiple sources; same-name collision refusal; confirmation destinations; cancellation;
config opening/creation and next-launch reload; both watermark open-folder/refresh controls.

Windows checkbox repair remains a candidate until actual screenshots are reviewed at **100%,
125%, 150%, 200%** scaling on Windows 11 and Windows 10 22H2, covering desktop-shortcut tasks
and the completion-page launch checkbox. Check the Chinese font, glyph visibility, control
bounds, row height and spacing, and click/keyboard operation. Archive screenshots with scaling,
OS, installer version and scope. No native qualification or screenshot evidence was available
in the Linux implementation environment; do not mark this gate passed from `.iss` alone.
