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

### Server-cache purge completed

The rewritten GitHub branches and tags were fetched back into a fresh mirror
and passed the same audit. Immediately after the force-push, direct requests
for pre-rewrite commit and file object IDs still returned HTTP 200, so a
GitHub Support cached-view purge was requested.

On 2026-07-27, after GitHub Support reported that the purge was complete,
unauthenticated no-cache requests were made for the old commit page, deleted
Colab notebook, and personal watermark. All three direct URLs returned HTTP
404. The server-cache cleanup is therefore verified complete for the known
objects.

The first changed commits reported by the history-rewrite tool were:

- `fb3b20c4e77de6799881259a16f63b3ea3950298`
- `5911f7d7a6a0f12ce5eb44e951feb1db88302a6a`

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

Current status:

- Completed: GitHub Support purged the pre-rewrite cached objects and the
  three known direct URLs were independently verified to return HTTP 404.
- Completed: Release v2.5.0 was published from commit
  `28da1fd5ceaa39d873180ada57dd6e6719e61add` after Windows, macOS, and Linux
  verification passed. Its ZIP and checksum assets were verified against the
  published SHA-256 digest.
- Pending: Release v2.5.0 currently reports `immutable: false` through the
  GitHub Releases API. Enable release immutability only after confirming that
  no asset or release-note correction is required.

Repository hardening still to confirm:

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
