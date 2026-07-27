# Security Policy

## Supported versions

Security fixes are provided in the newest stable release. Update through
the GUI's **關於／更新** tab or download the newest release from GitHub.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue.

Use GitHub's **Security → Advisories → Report a vulnerability** form for
this repository. If private vulnerability reporting is not available,
open an issue that asks the maintainer to provide a private contact method,
but do not include vulnerability details in that issue.

Include the affected version, operating system, reproduction steps, impact,
and any suggested mitigation. Reports will be acknowledged as soon as
practical.

## Update trust model

The in-app updater:

- checks only the stable `latest` release of
  `minhung1126/auto-reframe-videos`;
- accepts only the canonical release asset name and URL;
- verifies the SHA-256 digest returned by GitHub's Releases API;
- rejects unsafe ZIP paths, symlinks, duplicate paths, oversized archives,
  abnormal compression ratios, and manifest mismatches;
- preserves user settings, videos, text, and watermarks;
- creates a local backup and refuses to overwrite locally modified managed
  files or a Git working tree.

Repository administrators should enable immutable releases so published
tags and assets cannot later be changed.
