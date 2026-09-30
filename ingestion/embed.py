"""Stage 2 of the ingestion pipeline: clean ads in, vectors out.

    uv run python -m ingestion.embed --input data/clean/ads_clean.jsonl \
        --out-dir data/embedded

An embedding is a list of numbers positioning a piece of text in space, such
that text meaning similar things lands nearby. Phase 4 searches that space
instead of matching keywords, which is why "trail running shoes" can retrieve
an ad that never uses the word "trail".

This runs OFFLINE, over all 44k ads at once. Nobody is waiting on it. The
serving path in Phase 4 embeds only the single short query per request -- if it
had to embed candidate ads at request time, latency would scale with the
candidate set and the SLA would be unreachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ingestion.embedder import MODEL_NAME, MODEL_REVISION, Embedder, load_embedder
from ingestion.fileutil import sha256_of_file, write_json_atomic

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_INPUT_ERROR = 2

# How many texts to hand the tokenizer per call when counting tokens.
#
# The gate only needs each text's LENGTH, but the tokenizer returns the full
# list of token ids for every text. Asking it for all 44,200 at once would
# materialise ~1.8 million Python ints (tens of MB of objects) just to call
# len() on them and throw them away. Chunking caps that at 2048 texts' worth,
# and the total work is identical either way.
#
# At our size this is a guard rail rather than a rescue; it earns its keep if
# this pipeline is ever pointed at millions of ads.
TOKEN_CHUNK = 2048


def compose_text(record: dict) -> str:
    """Build the string that actually gets embedded: headline and description.

    Category and bid are deliberately excluded, for different reasons.

    CATEGORY is excluded because it is used later as a separate, explicit
    ranking signal. Embedding it too would count the same evidence twice --
    once blurred into the vector where its influence cannot be measured or
    tuned, and again as a ranking feature where it can. Keeping it out means
    retrieval answers only "what is this ad about?" and ranking independently
    answers "which relevant ad should win?". Those stay separable, which is
    what makes the ranking weights in Phase 6 meaningful.

    Note this is a RANKING SIGNAL, not a pre-filter: incoming queries are free
    text and carry no category, so there is nothing to filter on before the
    search. Retrieval runs over every ad and category match is scored
    afterwards. (Real ad servers do filter first, on eligibility -- budget
    exhausted, targeting, frequency caps -- which is a different thing from
    relevance.)

    BID is excluded because it is not meaning. Two identical ads at $2 and $20
    should retrieve identically and be separated by the auction, not by the
    vector space.
    """
    headline = record["headline"]
    description = record.get("description", "")
    if not description:
        return headline
    separator = " " if headline[-1:] in ".!?" else ". "
    return f"{headline}{separator}{description}"


def load_clean_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def count_tokens_chunked(embedder: Embedder, texts: list[str]) -> list[int]:
    counts: list[int] = []
    for start in range(0, len(texts), TOKEN_CHUNK):
        counts.extend(embedder.count_tokens(texts[start : start + TOKEN_CHUNK]))
    return counts


class TruncationError(RuntimeError):
    """Raised when an ad would be silently cut short by the model."""

    def __init__(self, offenders: list[tuple[str, int]], max_tokens: int) -> None:
        self.offenders = offenders
        self.max_tokens = max_tokens
        shown = ", ".join(f"{ad_id} ({n} tokens)" for ad_id, n in offenders[:10])
        more = "" if len(offenders) <= 10 else f" and {len(offenders) - 10} more"
        super().__init__(
            f"{len(offenders)} ad(s) exceed the model's {max_tokens}-token limit: {shown}{more}"
        )


def run(
    input_path: Path,
    out_dir: Path,
    embedder: Embedder,
    *,
    batch_size: int = 256,
    allow_truncation: bool = False,
    limit: int | None = None,
) -> dict:
    """Embed every clean ad and write embeddings.npy + embed_manifest.json."""
    records = load_clean_records(input_path, limit=limit)
    texts = [compose_text(record) for record in records]

    # --- token gate --------------------------------------------------------
    #
    # Phase 1's length limits are in CHARACTERS. The model truncates by TOKENS,
    # at max_tokens. That mismatch is the one seam between the two phases, so
    # it is checked here BEFORE any embedding work happens -- failing after
    # running the model over 44k ads would waste the expensive part.
    token_counts = count_tokens_chunked(embedder, texts)
    offenders = [
        (records[i]["ad_id"], n) for i, n in enumerate(token_counts) if n > embedder.max_tokens
    ]
    if offenders and not allow_truncation:
        raise TruncationError(offenders, embedder.max_tokens)

    # --- embed -------------------------------------------------------------
    started = time.perf_counter()
    vectors = embedder.encode(texts, batch_size=batch_size)
    duration = time.perf_counter() - started

    if vectors.shape != (len(records), embedder.dim):
        raise RuntimeError(
            f"embedder returned {vectors.shape}, expected {(len(records), embedder.dim)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vectors)

    percentiles = np.percentile(token_counts, [50, 99]) if token_counts else [0, 0]
    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": embedder.name,
        "revision": embedder.revision,
        "dim": embedder.dim,
        "max_tokens": embedder.max_tokens,
        "normalized": True,
        "count": len(records),
        # Alignment tripwires: publish.py re-checks these against the clean
        # file so vectors can never be paired with the wrong ads.
        "source_path": str(input_path),
        "source_sha256": sha256_of_file(input_path),
        "first_ad_id": records[0]["ad_id"] if records else None,
        "last_ad_id": records[-1]["ad_id"] if records else None,
        "device": getattr(embedder, "device", "unknown"),
        "batch_size": batch_size,
        "tokens": {
            "p50": int(percentiles[0]),
            "p99": int(percentiles[1]),
            "max": int(max(token_counts)) if token_counts else 0,
            "truncated": len(offenders),
        },
        "duration_seconds": round(duration, 3),
        "ads_per_second": round(len(records) / duration, 1) if duration > 0 else None,
    }
    write_json_atomic(out_dir / "embed_manifest.json", manifest)
    return manifest


def format_summary(manifest: dict) -> str:
    tokens = manifest["tokens"]
    return "\n".join(
        [
            "",
            "=" * 62,
            "  EMBEDDING REPORT",
            "=" * 62,
            f"  model      {manifest['model']}",
            f"  revision   {manifest['revision'][:12]}",
            f"  ads        {manifest['count']:>9,}  x {manifest['dim']} dims",
            f"  device     {manifest['device']}  (batch size {manifest['batch_size']})",
            f"  tokens     p50={tokens['p50']}  p99={tokens['p99']}  max={tokens['max']}"
            f"  limit={manifest['max_tokens']}  truncated={tokens['truncated']}",
            f"  elapsed    {manifest['duration_seconds']:>9.2f}s"
            f"  ({manifest['ads_per_second']:,.0f} ads/sec)",
            "=" * 62,
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingestion.embed",
        description="Generate embeddings for cleaned ads.",
    )
    parser.add_argument("--input", type=Path, default=Path("data/clean/ads_clean.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/embedded"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Embed anyway when an ad exceeds the model's token limit (default: fail).",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Override the pinned model.")
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Override the pinned model revision (recorded in the manifest either way).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N ads.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    embedder = load_embedder(device=args.device, model_name=args.model, revision=args.revision)

    try:
        manifest = run(
            args.input,
            args.out_dir,
            embedder,
            batch_size=args.batch_size,
            allow_truncation=args.allow_truncation,
            limit=args.limit,
        )
    except TruncationError as exc:
        print(f"GATE FAILED: {exc}", file=sys.stderr)
        print(
            "These ads would be silently cut short by the model, which is exactly what "
            "Phase 1 promised not to allow. Tighten the Phase 1 length limits, or re-run "
            "with --allow-truncation if you accept the loss.",
            file=sys.stderr,
        )
        return EXIT_GATE_FAILED

    if not args.quiet:
        print(format_summary(manifest))
        print(f"  vectors  -> {args.out_dir / 'embeddings.npy'}")
        print(f"  manifest -> {args.out_dir / 'embed_manifest.json'}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
