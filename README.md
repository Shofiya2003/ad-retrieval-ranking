# Ad Retrieval, Ranking & Ingestion Pipeline

A small but realistic ad-serving system: a Python ingestion pipeline that validates, embeds and
publishes ad data, feeding a Go serving layer that does vector retrieval and ranking under a
latency SLA, with Redis as both feature store and cache.

**Status: Phases 1–2 complete** (validate → clean → embed → publish). Phases 3–10 to follow.

## Architecture (target)

```
                 INGESTION (Python, offline)            SERVING (Go, online)
  raw ads ──▶ validate ──▶ embed ──▶ publish ──▶ ┌── loader ──▶ HNSW index ──▶ rank ──▶ /rank
             (Phase 1)   (Phase 2)  (Phase 2)   │      ▲                         ▲
                                                 │      │                         │
                                                 │  Python /embed sidecar    Redis feature
                                                 │  (query-time only)        store + cache
                                                 └───────────────────────────────┘
```

## Phase 1 — Validate + Clean

Turns raw, messy ad records into a clean dataset plus a report of exactly what was thrown away
and why. The point is **fail fast**: bad data that slips through here doesn't announce itself
later — it silently becomes a garbage embedding in Phase 2, a meaningless nearest neighbour in
Phase 4, and a served ad with a negative bid in Phase 6.

### The central design decision: clean first, then validate

> A headline of `"   "` isn't empty until you trim it.
> A headline of `"<b>Sale!</b>"` isn't 12 characters of ad copy, it's 5.

Length and emptiness checks run against the **cleaned** text, because cleaned text is what
Phase 2 actually embeds. Judging the raw string would both reject good ads and accept junk ones.

That splits every data problem in two:

| | Example | Outcome |
|---|---|---|
| **Fixable** | stray HTML, doubled whitespace, zero-width spaces, `"$4.25"` as a bid | cleaned, kept, **warning** logged |
| **Fatal** | missing headline, bid of `-3.00`, unknown category, duplicate | **rejected** with a stable reason code |

Reason codes are stable strings (`E008_BID_OUT_OF_RANGE`), not free text. That's what makes the
report countable, chartable and alertable — the difference between *"some ads failed"* and
*"4.1% of this batch failed on bid range, up from 0.2% yesterday."*

### Rules

**Cleaning (kept):** `W001` whitespace · `W002` HTML · `W003` unicode NFKC · `W004` control chars ·
`W005` bid coerced from string · `W006` description missing · `W007` timestamp normalised

**Fatal (rejected):**

| Code | Check | Why it matters downstream |
|---|---|---|
| `E001_MALFORMED_JSON` | line won't parse | — |
| `E002_MISSING_FIELD` | required key absent or null | — |
| `E003_WRONG_TYPE` | e.g. headline is a list | — |
| `E004_EMPTY_TEXT` | headline blank after cleaning | embeds to noise |
| `E005_HEADLINE_TOO_SHORT` | < 10 chars after cleaning | too little signal for a useful embedding |
| `E006_TEXT_TOO_LONG` | headline > 120, description > 1000 | would be silently truncated at embed time |
| `E007_BID_NOT_NUMERIC` | bid won't coerce | — |
| `E008_BID_OUT_OF_RANGE` | bid ≤ 0 or > $50 CPM | catches cents-vs-dollars unit bugs and fat-fingers |
| `E009_UNKNOWN_CATEGORY` | not in the 12-value taxonomy | can't match in Phase 6 ranking |
| `E010_DUPLICATE_AD_ID` | id already seen | second copy would overwrite the first in the feature store |
| `E011_DUPLICATE_CONTENT` | identical cleaned text | returns what looks like the same ad twice in one top-K |
| `E012_BAD_TIMESTAMP` | `created_at` unparseable | — |

Duplicate policy is **first-wins**, and only *accepted* records claim an id — so a rejected
duplicate never poisons a later legitimate record.

### Known limitation

`E011` is **exact**-match only. `"Buy now!"` and `"Buy now!!"` both survive. Catching semantic
near-duplicates needs the Phase 2 embeddings; the hash is deliberately the cheap version.

### Scale note

Everything streams — one line in, one line out, constant memory. The one exception is duplicate
detection, which is inherently stateful: two sets over 50k records is nothing, but at 100M you'd
reach for a Bloom filter (accepting a small false-positive rate) or an external sort on the key.

## Phase 2 — Embed + Publish

Turns each clean ad into a 384-number vector (an embedding) and packages those vectors into a
versioned bundle the Go service can load with only its standard library.

An embedding places text in space so that text meaning similar things lands nearby. That's what
lets retrieval work without shared keywords: *"how do I lower my tax bill"* returns tax-filing
ads that barely share a word with the query.

### Model

`sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4`. 384 dimensions,
256-token limit, Apache-2.0. Chosen for speed (the Phase 4 sidecar embeds a query on every
request), small vectors (half the distance cost of a 768-dim model), and because queries and ads
are embedded identically — `bge`/`e5` need different prefixes for each, and getting that wrong in
the sidecar degrades quality with no error.

### Why ads are embedded offline but queries at request time

Embedding 44k ads takes ~25 seconds and nobody is waiting on it. The serving path embeds only the
single short query per request. Embedding candidate ads at request time would make latency scale
with the candidate set.

### What goes into the vector

`"{headline}. {description}"` — **category and bid are deliberately excluded.** Phase 6 scores
category match and bid as separate, explicit ranking signals; folding category into the vector
would count it twice and blur retrieval ("what is this ad about?") against ranking ("which
relevant ad should win?"). A bid isn't meaning at all.

### Vectors are unit length, so similarity is a dot product

Every vector is L2-normalised to length 1.0. For unit vectors, cosine similarity **equals** the
dot product — so Phase 4 needs no square roots in the hot path. `publish.py` refuses to publish
if any vector's length drifts from 1.0.

### Four ways this stage refuses to fail silently

| Guard | The quiet disaster it prevents |
|---|---|
| **Token gate** | Phase 1 limits *characters*; the model truncates by *tokens* at 256. `embed.py` counts tokens with the model's own tokenizer **before** embedding, and fails listing the offending `ad_id`s. Without it, over-long ads are cut short and nothing reports it. |
| **Staleness guard** | `publish.py` re-hashes `ads_clean.jsonl` and refuses if it changed since embedding. Otherwise re-running `validate.py` and publishing stale vectors pairs every vector with a *different* ad — nothing crashes, search just returns confident nonsense. |
| **Integrity checks** | NaN rows (a NaN compares false against everything, so the ad becomes invisible to search) and all-zero rows (dot product 0 against every query, can never be retrieved). |
| **Model identity in the manifest** | The Phase 4 sidecar reads the model name and revision *from the manifest* instead of hard-coding them, so query and ad vectors cannot come from different models. A mismatch produces plausible-looking similarity scores that mean nothing. |

### The published bundle

```
data/published/
  LATEST                          -> text file naming the current version
  20260922T201033Z-4c4c2646/
    vectors.f32     raw little-endian float32, 44,200 x 384, row-major (67,891,200 bytes)
    ads.jsonl       one ad per line with an explicit `row` field; line i ↔ vector row i
    manifest.json   counts, dims, checksums, model identity, source provenance
```

Raw float32 rather than JSON because Go reads it with `encoding/binary` in one pass instead of
parsing 17 million float strings at startup. Rather than `.npy` because that needs a third-party
Go parser; the manifest carries the same information as plain JSON. Rather than Redis because
that's Phase 5 — flat files keep Go startup deterministic and debuggable.

**Publishing is atomic.** The bundle is written to a staging directory, `fsync`ed, then renamed
into place; `LATEST` is updated last, also atomically. A reader following `LATEST` sees either
the complete old bundle or the complete new one, never a mixture — and a crash part-way through
leaves the previous version serving untouched. This is also the hook Phase 9 needs for hot reload.

### Why `embed` and `publish` are separate stages

Embedding is expensive (runs a model); publishing is cheap (moves bytes). Splitting them means
the output format can change, or a bundle be rebuilt, without paying for the model again.

## Results — Phase 2 (44,200 ads, Apple M4)

```
embed     24.66s   1,792 ads/sec on the M4 GPU (mps), batch size 256
tokens    p50=37  p99=46  max=55   limit=256   truncated=0
publish   67,891,200 bytes = 44,200 x 384 x 4, checksum verified
probe     category precision@5 = 40/40 = 100%
```

Memory math worth knowing: 68 MB for 44k ads → roughly **15 GB at 10M ads**, which is why
production systems quantize (int8 / product quantization) at that scale.

**One honest finding.** For *"warm waterproof jacket for winter"* the top hit is a denim jacket,
ahead of the rain shells and parkas in the corpus. The model matched "jacket" and "winter" more
strongly than "waterproof". Category precision was still 5/5 because a denim jacket is apparel —
which is exactly the limitation of using category as a stand-in for a real relevance label.

### The probe script is Phase 4's ground truth

`scripts/probe_embeddings.py` does **exact** brute-force search: one dot product against all
44,200 rows. HNSW in Phase 4 is **approximate**, so its recall — what fraction of the true top-K
it actually found — can only be measured against an exact answer.
`--dump-ground-truth` writes those exact results for Phase 4 to score itself against, which turns
the accuracy-vs-speed tradeoff into a measurement rather than a claim.


## Running it

```bash
uv sync --extra dev

# everything, end to end: generate -> validate -> embed -> publish -> probe
make pipeline

# or one stage at a time
uv run python scripts/generate_raw_ads.py --count 50000 --seed 42
uv run python -m ingestion.validate --fail-under 0.85   # gate: non-zero exit blocks Phase 2
uv run python scripts/check_oracle.py                   # counts match the seeded defects?
uv run python -m ingestion.embed                        # gate: fails on token-limit overflow
uv run python -m ingestion.publish                      # gate: fails on stale or broken vectors
uv run python scripts/probe_embeddings.py               # brute-force sanity search

make test        # fast suite, no model download
make test-slow   # the 6 tests that load the real model
```

Each stage returns a non-zero exit code when its gate fails, so `make` stops. That's the point of
gates rather than warnings.

Phase 1 outputs land in `data/clean/`: `ads_clean.jsonl`, `ads_rejected.jsonl`,
`validation_report.json`. Phase 2 outputs land in `data/embedded/` and `data/published/`.

### Why JSONL everywhere

Streamable line by line, appendable (which Phase 9's incremental ingestion needs), and one
corrupt line doesn't destroy the whole file — a JSON array would fail entirely on the same input.

### The generator is a test oracle

`scripts/generate_raw_ads.py` injects **exactly one** defect per record and writes
`data/raw/expected_defects.json` recording what it injected. The validator's report should match
it code for code; `scripts/check_oracle.py` proves it does. Without seeded defects the validator
has nothing to catch and the report is a wall of zeros — you can't trust a data-quality check
you've never seen fire.

## Results — Phase 1 (50,000 ads, seed 42)

```
read         50,000
accepted     44,200   (88.40%)
rejected      5,800
elapsed        0.82s        ≈ 61k records/sec, single-threaded
```

All 12 reject codes match the generator's oracle exactly.

## Layout

```
ingestion/
  schema.py     AdRecord, the category taxonomy, every tunable limit
  clean.py      pure normalisation helpers (no I/O, no rejections)
  rules.py      one small function per rule -> None | Violation
  report.py     count aggregation, JSON artifact + terminal summary
  validate.py   Phase 1 CLI: streams raw -> clean/reject -> report
  fileutil.py   checksums + atomic JSON writes, shared by every stage
  embedder.py   Embedder protocol + the pinned MiniLM implementation
  embed.py      Phase 2 CLI: token gate -> batched embeddings.npy
  publish.py    Phase 2 CLI: verify -> versioned bundle -> atomic LATEST
scripts/
  generate_raw_ads.py   synthetic corpus with seeded defects + oracle
  check_oracle.py       diffs the report against the oracle
  probe_embeddings.py   brute-force search; also Phase 4's ground truth
tests/                  134 tests (128 fast + 6 that load the real model)
```

`embed.py` depends on the small `Embedder` protocol rather than on
`sentence-transformers` directly, so the fast suite plugs in a deterministic fake and never
downloads 90 MB or imports torch. Everything that can genuinely go wrong in Phase 2 — the token
gate, row alignment, checksums, atomic publishing — is pipeline logic that needs no real neural
network to test.

`rules.py` holds one function per rule rather than one fused `validate()` so that any single
rule can be read, tested and explained in isolation. In production this layer is usually a
Pydantic model; it's hand-rolled here so the logic is visible rather than declared.

## Stack

Python 3.11 (pinned via `uv` — the ML stack lags newer releases). Phase 1 is stdlib-only;
Phase 2 adds `sentence-transformers` 6.0.1, `torch` 2.14 and `numpy` 2.4. `pytest` for tests.
