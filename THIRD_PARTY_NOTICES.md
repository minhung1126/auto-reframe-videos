# Third-party notices

## Noto Serif TC

`fonts/NotoSerifTC.ttf` is part of the Noto CJK font family. It is
distributed under the SIL Open Font License 1.1. The complete license is
included at `fonts/LICENSE`.

Font metadata:

- Copyright: `(c) 2017-2024 Adobe (http://www.adobe.com/).`
- Version: `2.003-H1`
- Bundled SHA-256:
  `0077e18f57c6908f4a000969880940bdb0dad057c0e8d98b49dc364c3d1b09c6`

Verified source: Google Fonts repository commit
[`6d17dab13b85129360f9748f057c7f67c5f484d4`](https://github.com/google/fonts/blob/6d17dab13b85129360f9748f057c7f67c5f484d4/ofl/notoseriftc/NotoSerifTC%5Bwght%5D.ttf).


## Desktop runtime distributions

Desktop installers additionally contain Python (PSF license), Tcl/Tk (their BSD-style licenses),
PyInstaller bootloader (GPL with its distribution exception), and certifi (MPL 2.0 CA data).
Their license texts are shipped under `licenses/`, along with FFmpeg and all linked components.
The application's All Rights Reserved notice does not replace any third-party license.

FFmpeg with libx264 and libx265 is a GPL distribution. `--enable-nonfree` builds are rejected.
Each reviewed vendor bundle must contain `provenance.json` with exact versions, build recipe,
SHA-256 hashes, corresponding source downloads and license records for all components.
The default build uses FFmpeg 8.0.1, x264, x265, FreeType 2.13.3, HarfBuzz 12.1.0
and zlib 1.3.1 (zlib license, used for PNG watermarks).
Windows additionally uses oneVPL 2.13.0, NVIDIA codec headers and AMD AMF headers.
Exact upstream commits, source archive URLs and SHA-256 hashes are recorded in
`packaging/ffmpeg-sources.json` and each installer's provenance. FreeType uses its FTL
license; HarfBuzz uses its permissive license; NVIDIA headers retain their copyright
and BSD notices; AMD AMF and oneVPL retain their upstream license texts.
The source build recipe is `scripts/build_ffmpeg_vendor.py`.
Windows statically linked GCC support libraries use GPLv3 with the GCC Runtime
Library Exception; both the GPLv3 text and exception are included in the bundle.
When MinGW runtime DLLs are required, they are shipped beside FFmpeg/FFprobe and
recorded by SHA-256 in provenance. Their native package notices, including MinGW CRT
and winpthreads notices, are copied into `licenses/runtime`.
See `docs/DESKTOP_DISTRIBUTION.md` for the mandatory redistribution checklist.

## GUI drag and drop

- `tkinterdnd2` 0.4.3 (MIT), https://github.com/Eliav2/tkinterdnd2.
  PyPI wheel SHA-256: `8804f5d2e2a99713ec93e85384397fec6bf66fdf2065e3750938d55018971c4a`.
- The wheel ships native TkDND libraries (macOS arm64: 2.9.3; Windows x64/macOS x64/Linux x64: 2.9.4) for Windows/macOS/Linux.
  Upstream: https://github.com/petasis/tkdnd/tree/tkdnd-release-test-v2.9.4.
  License: `packaging/licenses/tkdnd.txt`, SHA-256
  `86501f2f0b7dc0ade34deef0cfba930286ec28a12c63949c7417ef53f01aae88`.
- Native desktop bundles include only the target OS/CPU TkDND directory and both
  license notices under `licenses/runtime/`. The TkDND license grants redistribution
  with its notice retained verbatim; it does not change this project's license.
