# Public repository and release safety checklist

Audit date: 2026-07-27

## Audit result

The GitHub repository is public. Before the first Release, its history was
rewritten to remove:

- a personal Gmail author address, replaced by the owner's GitHub noreply
  address;
- a deleted Colab notebook and its personal account, authorship, cell/output,
  and Drive-path metadata;
- a personal PNG watermark and matching brand references.

The rewritten current tree and every reachable commit were scanned for these
identifiers and for common API keys, access tokens, passwords, private keys,
and cloud credentials. No matches remained. This rewrite cannot recall old
clones, forks, caches, or archives that third parties may already possess.

Runtime settings, input/output videos, text files, logs, notebook files, local
secret-file patterns, and signing material are ignored. Release ZIP files are
built from an explicit allowlist and do not include runtime data, tests,
workflows, repository instructions, or user watermarks.

The project source is publicly viewable but **All Rights Reserved**. The root
`LICENSE` grants no permission to use, modify, create derivative works from,
or redistribute the software. The Release workflow refuses to publish without
this notice.

The bundled `fonts/NotoSerifTC.ttf` is the official Google Fonts artifact
pinned to repository commit
`6d17dab13b85129360f9748f057c7f67c5f484d4`, with SHA-256
`0077e18f57c6908f4a000969880940bdb0dad057c0e8d98b49dc364c3d1b09c6`.
It is separately licensed under the SIL Open Font License 1.1; its license and
Adobe copyright notice are included in `fonts/LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## One-time repository settings

Before publishing the first Release:

- Enable **Settings → Releases → Enable release immutability**.
- Keep the default `GITHUB_TOKEN` permission read-only.
- Require full-length commit SHA pinning for Actions if the repository setting
  is available.
- Enable private vulnerability reporting.
- Enable secret scanning and push protection if GitHub offers them for the
  repository.
- Protect `main`: require the cross-platform CI checks and prevent force
  pushes.

## Release procedure

1. Update `VERSION` in `auto_reframe_core/version.py`.
2. Commit and push the release-ready code to `main`.
3. Confirm CI passes on Windows, macOS, and Linux.
4. Open **Actions → Release → Run workflow** on `main`.
5. Enter the same `MAJOR.MINOR.PATCH` version without a `v` prefix.
6. Confirm that the resulting Release is marked **Immutable** and contains:
   - `auto-reframe-videos-vX.Y.Z.zip`
   - `SHA256SUMS-vX.Y.Z.txt`
7. From an extracted older Release ZIP, use **關於／更新** to test the update
   and restart path. Automatic install intentionally stays disabled inside Git
   checkouts.
