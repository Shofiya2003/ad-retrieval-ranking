"""Stage 3 of the ingestion pipeline: package vectors + metadata for Go.

    uv run python -m ingestion.publish

Publishing is cheap (it moves bytes around) while embedding is expensive (it
runs a model), which is why they are separate stages: the output format can
change, or a bundle can be rebuilt, without paying for the model again.

The bundle is three files in a versioned directory:

    vectors.f32    raw little-endian float32, count x dim, row-major
    ads.jsonl      one ad per line; line i corresponds to vector row i
    manifest.json  the contract: counts, dims, checksums, model identity

Raw float32 rather than JSON because Go can read it with encoding/binary in one
pass -- no parsing 17 million float strings at startup. Rather than .npy
because that would need a third-party Go parser to learn the shape; the
manifest carries the same information in plain JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ingestion.embed import load_clean_records
from ingestion.fileutil import sha256_of_file, write_json_atomic

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_INPUT_ERROR = 2

FORMAT_VERSION = 1
LATEST_POINTER = "LATEST"

# Vectors are unit length by construction. Floating point makes that "1.0 give
# or take", so the check needs a tolerance -- but a loose one would let a
# genuinely broken row through, and an all-zero row (norm 0.0) is the failure
# this is really hunting for.
NORM_TOLERANCE = 1e-3


class PublishError(RuntimeError):
    """A check failed; nothing is written and nothing is swapped into place."""


def verify_alignment(records: list[dict], vectors: np.ndarray, embed_manifest: dict,
                     clean_path: Path) -> None:
    """Prove that vector row i still belongs to ad i.

    The failure this prevents is quiet and nasty: re-run validate.py with a
    different threshold, skip re-embedding, publish, and every vector is now
    paired with a different ad. Nothing crashes. Search returns confident
    nonsense. The hash is what makes that impossible.
    """
    actual_sha = sha256_of_file(clean_path)
    if actual_sha != embed_manifest["source_sha256"]:
        raise PublishError(
            f"{clean_path} has changed since it was embedded "
            f"(now {actual_sha[:12]}, embedded {embed_manifest['source_sha256'][:12]}). "
            "Re-run ingestion.embed before publishing."
        )

    if len(records) != embed_manifest["count"]:
        raise PublishError(
            f"clean file has {len(records)} ads but the embed manifest claims "
            f"{embed_manifest['count']}"
        )

    if vectors.shape[0] != len(records):
        raise PublishError(
            f"{vectors.shape[0]} vectors for {len(records)} ads -- rows and ads must correspond"
        )

    if vectors.shape[1] != embed_manifest["dim"]:
        raise PublishError(
            f"vectors are {vectors.shape[1]}-dimensional, manifest says {embed_manifest['dim']}"
        )

    if records and records[0]["ad_id"] != embed_manifest["first_ad_id"]:
        raise PublishError("first ad_id does not match the embed manifest -- ordering changed")

    if records and records[-1]["ad_id"] != embed_manifest["last_ad_id"]:
        raise PublishError("last ad_id does not match the embed manifest -- ordering changed")


def verify_vectors(vectors: np.ndarray) -> None:
    """Refuse to publish numbers that cannot mean anything.

    A NaN poisons every comparison it touches (NaN compares false against
    everything, so the row becomes invisible to search). An all-zero row has
    dot product 0 with every query -- it can never be retrieved, and nothing
    would ever tell you.
    """
    if not np.isfinite(vectors).all():
        bad = int((~np.isfinite(vectors)).any(axis=1).sum())
        raise PublishError(f"{bad} vector(s) contain NaN or infinity")

    norms = np.linalg.norm(vectors, axis=1)
    off = np.abs(norms - 1.0) > NORM_TOLERANCE
    if off.any():
        worst = float(norms[off].min())
        raise PublishError(
            f"{int(off.sum())} vector(s) are not unit length (worst norm {worst:.6f}); "
            "Phase 4 relies on normalisation so that a dot product IS cosine similarity"
        )


def write_bundle(target: Path, records: list[dict], vectors: np.ndarray,
                 embed_manifest: dict, clean_path: Path) -> dict:
    """Write the three files into `target` and return the manifest."""
    target.mkdir(parents=True, exist_ok=True)

    # astype("<f4") pins little-endian float32 explicitly rather than relying on
    # this machine's native order, so the Go loader's assumption is guaranteed
    # by the writer instead of by luck.
    payload = np.ascontiguousarray(vectors, dtype="<f4").tobytes()
    vectors_path = target / "vectors.f32"
    with vectors_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    ads_path = target / "ads.jsonl"
    with ads_path.open("w", encoding="utf-8") as handle:
        for row, record in enumerate(records):
            # The explicit row index makes the alignment contract checkable
            # from the file itself rather than assumed from line order.
            handle.write(json.dumps({"row": row, **record}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Phase 4's query sidecar reads model identity FROM HERE rather than
        # hard-coding it, so query vectors and ad vectors cannot come from
        # different models. A mismatch would produce plausible-looking
        # similarity scores that mean nothing.
        "model": {
            "name": embed_manifest["model"],
            "revision": embed_manifest["revision"],
            "dim": embed_manifest["dim"],
            "max_tokens": embed_manifest["max_tokens"],
            "normalized": True,
            "similarity": "dot",
        },
        "vectors": {
            "file": vectors_path.name,
            "dtype": "float32",
            "byte_order": "little",
            "layout": "row_major",
            "count": int(vectors.shape[0]),
            "dim": int(vectors.shape[1]),
            "bytes": len(payload),
            "sha256": sha256_of_file(vectors_path),
        },
        "ads": {
            "file": ads_path.name,
            "count": len(records),
            "sha256": sha256_of_file(ads_path),
        },
        "source": {
            "clean_file": str(clean_path),
            "clean_sha256": embed_manifest["source_sha256"],
            "embedded_at": embed_manifest["created_at"],
            "device": embed_manifest["device"],
            "tokens": embed_manifest["tokens"],
        },
    }
    write_json_atomic(target / "manifest.json", manifest)
    return manifest


def run(clean_path: Path, embedded_dir: Path, out_root: Path) -> dict:
    """Verify everything, then swap a complete bundle into place atomically."""
    embed_manifest = json.loads((embedded_dir / "embed_manifest.json").read_text())
    vectors = np.load(embedded_dir / "embeddings.npy")
    records = load_clean_records(clean_path)

    verify_alignment(records, vectors, embed_manifest, clean_path)
    verify_vectors(vectors)

    # Version names sort chronologically and carry a content fingerprint, so
    # two runs over identical data are distinguishable by time but obviously
    # related by hash.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(np.ascontiguousarray(vectors, dtype="<f4").tobytes()).hexdigest()
    version = f"{stamp}-{digest[:8]}"

    out_root.mkdir(parents=True, exist_ok=True)
    staging = out_root / f".tmp-{version}"
    if staging.exists():
        shutil.rmtree(staging)

    try:
        manifest = write_bundle(staging, records, vectors, embed_manifest, clean_path)
        final = out_root / version
        if final.exists():
            shutil.rmtree(final)
        # Rename of a fully written directory: a reader following LATEST sees
        # either the previous complete bundle or this one, never a half-written
        # mixture of the two.
        os.replace(staging, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    # LATEST is updated LAST, and atomically. Until this line runs, the new
    # bundle exists on disk but nothing points at it -- so a crash anywhere
    # above leaves the previous version serving, untouched.
    pointer = out_root / LATEST_POINTER
    tmp_pointer = pointer.with_name(pointer.name + ".tmp")
    tmp_pointer.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp_pointer, pointer)

    manifest["version"] = version
    manifest["path"] = str(final)
    return manifest


def read_latest(out_root: Path) -> Path:
    """Resolve the LATEST pointer to a bundle directory."""
    version = (out_root / LATEST_POINTER).read_text(encoding="utf-8").strip()
    return out_root / version


def format_summary(manifest: dict) -> str:
    vectors, model = manifest["vectors"], manifest["model"]
    return "\n".join(
        [
            "",
            "=" * 62,
            "  PUBLISHED BUNDLE",
            "=" * 62,
            f"  version    {manifest['version']}",
            f"  path       {manifest['path']}",
            f"  model      {model['name']} @ {model['revision'][:12]}",
            f"  vectors    {vectors['count']:,} x {vectors['dim']} "
            f"{vectors['dtype']} ({vectors['bytes']:,} bytes)",
            f"  similarity {model['similarity']} (vectors are unit length)",
            f"  checksum   {vectors['sha256'][:16]}",
            "=" * 62,
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingestion.publish",
        description="Package embeddings + metadata into a versioned bundle for the Go service.",
    )
    parser.add_argument("--clean", type=Path, default=Path("data/clean/ads_clean.jsonl"))
    parser.add_argument("--embedded", type=Path, default=Path("data/embedded"))
    parser.add_argument("--out-root", type=Path, default=Path("data/published"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path in (args.clean, args.embedded / "embeddings.npy", args.embedded / "embed_manifest.json"):
        if not path.exists():
            print(f"error: required input not found: {path}", file=sys.stderr)
            return EXIT_INPUT_ERROR

    try:
        manifest = run(args.clean, args.embedded, args.out_root)
    except PublishError as exc:
        print(f"PUBLISH REFUSED: {exc}", file=sys.stderr)
        return EXIT_GATE_FAILED

    if not args.quiet:
        print(format_summary(manifest))
        print(f"  LATEST -> {args.out_root / LATEST_POINTER}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
