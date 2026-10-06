"""Fetch a reviewed, hash-pinned vendor ZIP; never use unpinned latest binaries."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import unicodedata
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

MAX_BYTES = 1024 * 1024 * 1024


def unpack_vendor(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Vendor destination must not exist')
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 1000 or sum(e.file_size for e in entries) > MAX_BYTES:
            raise ValueError('Vendor ZIP exceeds limits')
        seen = set()
        for entry in entries:
            # Windows ZipInfo normalizes separators while reading. Inspect the
            # original archive name before that transformation or NUL truncation.
            if entry.orig_filename != entry.filename or '\\' in entry.orig_filename:
                raise ValueError('Non-canonical vendor ZIP path')
            path = PurePosixPath(entry.filename)
            if any(part in ('', '.', '..') for part in entry.filename.rstrip('/').split('/')):
                raise ValueError('Non-canonical vendor ZIP path')
            if (not path.parts or path.is_absolute() or '..' in path.parts or '\\' in entry.filename
                    or ':' in entry.filename or any(part.rstrip(' .') != part for part in path.parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe vendor ZIP path')
            reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in range(1, 10)}
            if (len(entry.filename) > 1024 or any(ord(c) < 32 for c in entry.filename)
                    or any(c in '<>"|?*' for c in entry.filename)
                    or any(len(part) > 255 or part.split('.', 1)[0].upper() in reserved for part in path.parts)):
                raise ValueError('Unsafe vendor ZIP name')
            key = unicodedata.normalize('NFC', entry.filename.rstrip('/')).casefold()
            if key in seen:
                raise ValueError('Duplicate vendor ZIP path')
            seen.add(key)
            if path.parts[0] not in ('bin', 'licenses', 'provenance.json'):
                raise ValueError('Vendor ZIP must contain only bin/, licenses/, provenance.json')
        destination.mkdir(parents=True)
        try:
            for entry in entries:
                target = destination / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as source, target.open('wb') as sink:
                        shutil.copyfileobj(source, sink)
                    target.chmod(0o755 if PurePosixPath(entry.filename).parts[0] == 'bin' else 0o644)
        except BaseException:
            shutil.rmtree(destination)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-fA-F0-9]{64}', args.sha256) or urlparse(args.url).scheme != 'https':
        parser.error('A reviewed HTTPS URL and SHA-256 are required')
    with tempfile.TemporaryDirectory(prefix='arv-vendor-') as directory:
        archive = Path(directory) / 'vendor.zip'
        digest = hashlib.sha256()
        with urlopen(args.url, timeout=60) as response, archive.open('wb') as sink:
            if urlparse(response.geturl()).scheme != 'https':
                raise ValueError('Insecure vendor redirect')
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError('Vendor archive exceeds limit')
                digest.update(chunk)
                sink.write(chunk)
        if digest.hexdigest() != args.sha256.lower():
            raise ValueError('Vendor SHA-256 mismatch')
        unpack_vendor(archive, args.destination)
    print(args.destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
