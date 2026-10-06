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
The default build uses FFmpeg 8.0.1, x264, x265, FreeType 2.13.3 and HarfBuzz 12.1.0.
Windows additionally uses oneVPL 2.13.0, NVIDIA codec headers and AMD AMF headers.
Exact upstream commits, source archive URLs and SHA-256 hashes are recorded in
`packaging/ffmpeg-sources.json` and each installer's provenance. FreeType uses its FTL
license; HarfBuzz uses its permissive license; NVIDIA headers retain their copyright
and BSD notices; AMD AMF and oneVPL retain their upstream license texts.
The source build recipe is `scripts/build_ffmpeg_vendor.py`.
Windows statically linked GCC support libraries use GPLv3 with the GCC Runtime
Library Exception; both the GPLv3 text and exception are included in the bundle.
See `docs/DESKTOP_DISTRIBUTION.md` for the mandatory redistribution checklist.
