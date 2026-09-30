"""Small filesystem helpers shared across the pipeline stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

READ_BLOCK = 1024 * 1024


def sha256_of_file(path: Path | str) -> str:
    """Fingerprint a file's contents.

    Used three ways, all of them about trust rather than speed:
      * Phase 1 stamps the input it validated into its report (provenance).
      * Phase 2 checks the clean file has not changed since it was embedded
        (staleness -- see publish.py).
      * The published manifest records each bundle file, so the Go loader can
        detect a truncated or corrupted download before trusting the bytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON so a reader never observes a partially written file.

    Write to a temporary file in the same directory, flush it all the way to
    disk, then rename. Rename within one filesystem is atomic: any reader sees
    either the whole old file or the whole new one, never a mixture.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
