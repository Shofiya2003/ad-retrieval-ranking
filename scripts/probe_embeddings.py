#!/usr/bin/env python3
"""Brute-force search over a published bundle. Two jobs, both important.

1. SANITY. Before any Go exists, check the vectors actually capture meaning:
   run real queries and look at what comes back. If "lower my tax bill" returns
   dog beds, everything downstream is built on sand and better to know now.

2. GROUND TRUTH FOR PHASE 4. This search is EXACT -- it compares the query
   against all 44,200 ads. HNSW in Phase 4 is APPROXIMATE, so its recall
   ("what fraction of the true top-K did it actually find?") can only be
   measured against an exact answer. --dump-ground-truth writes these results
   for Phase 4 to score itself against, which is what turns the
   accuracy-vs-speed tradeoff into a measurement instead of a claim.

    python scripts/probe_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.embedder import load_embedder  # noqa: E402
from ingestion.publish import read_latest  # noqa: E402

# Each query is paired with the category a human would expect it to hit. The
# category is a rough stand-in for a relevance label -- enough to spot a broken
# pipeline, not enough to rank models against each other.
DEFAULT_QUERIES: list[tuple[str, str]] = [
    ("trail running shoes with good grip", "footwear"),
    ("how do I lower my tax bill", "finance"),
    ("orthopedic bed for an old dog", "pet_supplies"),
    ("noise cancelling headphones for flights", "electronics"),
    ("learn python for data analysis", "education"),
    ("warm waterproof jacket for winter", "apparel"),
    ("home espresso machine", "home_kitchen"),
    ("weights I can use in a small apartment", "fitness"),
]


def load_bundle(bundle: Path) -> tuple[dict, np.ndarray, list[dict]]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    spec = manifest["vectors"]

    # Read the raw bytes back the same way Go will: a flat little-endian
    # float32 array, reshaped using the count and dim from the manifest.
    vectors = np.fromfile(bundle / spec["file"], dtype="<f4").reshape(spec["count"], spec["dim"])

    ads = [json.loads(line) for line in (bundle / manifest["ads"]["file"]).read_text().splitlines()]
    return manifest, vectors, ads


def search(query_vectors: np.ndarray, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by dot product.

    The vectors are unit length, so a dot product IS cosine similarity -- no
    normalising and no square roots here. One matrix multiply compares every
    query against every ad at once.
    """
    scores = query_vectors @ vectors.T
    top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1), 1)
    return ordered, np.take_along_axis(scores, ordered, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=Path("data/published"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", action="append", help="Run a custom query (repeatable).")
    parser.add_argument("--dump-ground-truth", type=Path, default=None,
                        help="Write exact top-k results for Phase 4 to score HNSW recall against.")
    args = parser.parse_args()

    try:
        bundle = read_latest(args.published)
    except FileNotFoundError:
        print(f"error: no published bundle under {args.published} -- run ingestion.publish first",
              file=sys.stderr)
        return 2

    manifest, vectors, ads = load_bundle(bundle)

    # The model is read FROM THE MANIFEST, never hard-coded. That is the whole
    # contract: query vectors and ad vectors are guaranteed to come from the
    # same model, because there is only one place the name is written down.
    model_spec = manifest["model"]
    model = load_embedder(
        device="auto", model_name=model_spec["name"], revision=model_spec["revision"]
    )

    pairs = [(q, None) for q in args.query] if args.query else DEFAULT_QUERIES
    queries = [q for q, _ in pairs]
    query_vectors = model.encode(queries, batch_size=32)

    indices, scores = search(query_vectors, vectors, args.k)

    print(f"\nbundle   {bundle}")
    print(f"model    {model_spec['name']} @ {model_spec['revision'][:12]}")
    print(f"ads      {len(ads):,} x {model_spec['dim']} dims (exact brute-force search)\n")

    hits = 0
    scored = 0
    for row, (query, expected) in enumerate(pairs):
        print(f"  \"{query}\"")
        for rank in range(args.k):
            ad = ads[int(indices[row][rank])]
            mark = ""
            if expected is not None:
                mark = " <-- expected" if ad["category"] == expected else ""
            print(f"    {scores[row][rank]:.3f}  [{ad['category']:<13}] {ad['headline'][:62]}{mark}")
        if expected is not None:
            top_hits = sum(ads[int(i)]["category"] == expected for i in indices[row])
            hits += top_hits
            scored += args.k
            print(f"    -> {top_hits}/{args.k} in the expected category ({expected})")
        print()

    if scored:
        print(f"  category precision@{args.k}: {hits}/{scored} = {hits / scored:.1%}")
        print("  (a rough signal only -- category is a stand-in for a real relevance label)\n")

    if args.dump_ground_truth:
        payload = {
            "bundle": str(bundle),
            "model": model_spec,
            "k": args.k,
            "queries": [
                {
                    "query": query,
                    "expected_category": expected,
                    "top_k": [
                        {"ad_id": ads[int(i)]["ad_id"], "row": int(i), "score": float(s)}
                        for i, s in zip(indices[row], scores[row])
                    ],
                }
                for row, (query, expected) in enumerate(pairs)
            ],
        }
        args.dump_ground_truth.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ground_truth.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  exact ground truth -> {args.dump_ground_truth}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
