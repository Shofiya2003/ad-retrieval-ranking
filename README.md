# Ad Retrieval, Ranking & Ingestion Pipeline

A small but realistic ad-serving system: a Python ingestion pipeline that validates, embeds and
publishes ad data, feeding a Go serving layer that does vector retrieval and ranking under a
latency SLA, with Redis as both feature store and cache.

**Status: Phase 1 complete** (validate + clean). Phases 2–10 to follow.

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

## Running it

```bash
uv sync --extra dev

# 1. generate a synthetic corpus with deliberately seeded defects
uv run python scripts/generate_raw_ads.py --count 50000 --seed 42

# 2. validate it (--fail-under is the pipeline gate: non-zero exit blocks Phase 2)
uv run python -m ingestion.validate --fail-under 0.85

# 3. confirm the validator's counts match the generator's seeded defects
uv run python scripts/check_oracle.py

# 4. unit + end-to-end tests
uv run pytest
```

Outputs land in `data/clean/`: `ads_clean.jsonl`, `ads_rejected.jsonl`, `validation_report.json`.

### Why JSONL everywhere

Streamable line by line, appendable (which Phase 9's incremental ingestion needs), and one
corrupt line doesn't destroy the whole file — a JSON array would fail entirely on the same input.

### The generator is a test oracle

`scripts/generate_raw_ads.py` injects **exactly one** defect per record and writes
`data/raw/expected_defects.json` recording what it injected. The validator's report should match
it code for code; `scripts/check_oracle.py` proves it does. Without seeded defects the validator
has nothing to catch and the report is a wall of zeros — you can't trust a data-quality check
you've never seen fire.

## Results (50,000 ads, seed 42)

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
  validate.py   CLI: streams raw -> clean/reject -> report
scripts/
  generate_raw_ads.py   synthetic corpus with seeded defects + oracle
  check_oracle.py       diffs the report against the oracle
tests/                  98 tests: cleaning, each rule, end-to-end + the gate
```

`rules.py` holds one function per rule rather than one fused `validate()` so that any single
rule can be read, tested and explained in isolation. In production this layer is usually a
Pydantic model; it's hand-rolled here so the logic is visible rather than declared.

## Stack

Python 3.11 (pinned via `uv` — the Phase 2 ML stack lags newer releases), stdlib only for
Phase 1, `pytest` for tests.
