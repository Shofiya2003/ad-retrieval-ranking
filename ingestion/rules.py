"""One small function per validation rule.

Each rule takes an already-cleaned value and returns either None (it passed) or
a Violation carrying a STABLE code. Stable codes are the whole point: they are
what make the report countable, chartable and alertable. The difference between

    "some ads failed"

and

    "4.1% of this batch failed on bid range, up from 0.2% yesterday"

is entirely that the reason has a name that does not change between runs.

The rules are kept one-per-function rather than fused into a single validate()
so that any single rule can be read, tested and explained in isolation. In
production this layer is usually a Pydantic model; we hand-roll it here so the
logic is visible rather than declared.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingestion import schema
from ingestion.clean import content_hash

# Fatal codes -- the record is REJECTED.
E_MALFORMED_JSON = "E001_MALFORMED_JSON"
E_MISSING_FIELD = "E002_MISSING_FIELD"
E_WRONG_TYPE = "E003_WRONG_TYPE"
E_EMPTY_TEXT = "E004_EMPTY_TEXT"
E_HEADLINE_TOO_SHORT = "E005_HEADLINE_TOO_SHORT"
E_TEXT_TOO_LONG = "E006_TEXT_TOO_LONG"
E_BID_NOT_NUMERIC = "E007_BID_NOT_NUMERIC"
E_BID_OUT_OF_RANGE = "E008_BID_OUT_OF_RANGE"
E_UNKNOWN_CATEGORY = "E009_UNKNOWN_CATEGORY"
E_DUPLICATE_AD_ID = "E010_DUPLICATE_AD_ID"
E_DUPLICATE_CONTENT = "E011_DUPLICATE_CONTENT"
E_BAD_TIMESTAMP = "E012_BAD_TIMESTAMP"

ALL_FATAL_CODES: tuple[str, ...] = (
    E_MALFORMED_JSON,
    E_MISSING_FIELD,
    E_WRONG_TYPE,
    E_EMPTY_TEXT,
    E_HEADLINE_TOO_SHORT,
    E_TEXT_TOO_LONG,
    E_BID_NOT_NUMERIC,
    E_BID_OUT_OF_RANGE,
    E_UNKNOWN_CATEGORY,
    E_DUPLICATE_AD_ID,
    E_DUPLICATE_CONTENT,
    E_BAD_TIMESTAMP,
)


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    detail: str


# --- Structural rules (run on the RAW dict, before cleaning) --------------
#
# These have to come first: you cannot strip HTML from a list, and you cannot
# measure the length of a field that is not there.


def check_required_fields(raw: dict) -> Violation | None:
    for field in schema.REQUIRED_FIELDS:
        if field not in raw or raw[field] is None:
            return Violation(E_MISSING_FIELD, f"{field} is missing or null")
    return None


def check_field_types(raw: dict) -> Violation | None:
    """Only structural type errors are fatal here.

    `bid_cpm_usd` is deliberately not checked -- a bid that arrives as the
    string "4.25" is an upstream formatting quirk we can fix, so it goes
    through coerce_bid() instead of being rejected outright.
    """
    for field in schema.STRING_FIELDS:
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            return Violation(
                E_WRONG_TYPE,
                f"{field} must be a string, got {type(value).__name__}",
            )
    return None


# --- Content rules (run on CLEANED values) --------------------------------


def check_headline(headline: str) -> Violation | None:
    """Emptiness and length, judged on the cleaned text.

    Too short is fatal because Phase 2 cannot build a useful embedding from a
    handful of characters -- "Sale!" lands somewhere meaningless in vector
    space and would surface as a neighbour for unrelated queries.

    Too long is fatal because the embedding model would silently TRUNCATE it.
    We would rather reject loudly than embed half an ad and never find out.
    """
    if not headline:
        return Violation(E_EMPTY_TEXT, "headline is empty after cleaning")
    if len(headline) < schema.HEADLINE_MIN_LEN:
        return Violation(
            E_HEADLINE_TOO_SHORT,
            f"headline is {len(headline)} chars, minimum is {schema.HEADLINE_MIN_LEN}",
        )
    if len(headline) > schema.HEADLINE_MAX_LEN:
        return Violation(
            E_TEXT_TOO_LONG,
            f"headline is {len(headline)} chars, maximum is {schema.HEADLINE_MAX_LEN}",
        )
    return None


def check_description(description: str) -> Violation | None:
    if len(description) > schema.DESCRIPTION_MAX_LEN:
        return Violation(
            E_TEXT_TOO_LONG,
            f"description is {len(description)} chars, maximum is {schema.DESCRIPTION_MAX_LEN}",
        )
    return None


def check_bid(bid: float | None, raw_value: object) -> Violation | None:
    """`bid` is None when coerce_bid() could not read it as a number at all.

    The range check is the one that earns its keep: the ceiling catches a
    source system reporting cents (425) where we expect dollars (4.25), which
    would otherwise win every auction in Phase 6 and look like a ranking bug
    rather than a data bug.
    """
    if bid is None:
        return Violation(E_BID_NOT_NUMERIC, f"bid_cpm_usd={raw_value!r} is not a number")
    if bid != bid or bid in (float("inf"), float("-inf")):  # NaN / infinity
        return Violation(E_BID_OUT_OF_RANGE, f"bid_cpm_usd={bid} is not finite")
    if bid <= schema.BID_MIN_EXCLUSIVE:
        return Violation(E_BID_OUT_OF_RANGE, f"bid_cpm_usd={bid} must be > 0")
    if bid > schema.BID_MAX:
        return Violation(
            E_BID_OUT_OF_RANGE,
            f"bid_cpm_usd={bid} exceeds ceiling of {schema.BID_MAX}",
        )
    return None


def check_category(category: str) -> Violation | None:
    """Unknown category is fatal, unlike stray HTML which we just clean away.

    The difference is repairability. We know what a headline with tags in it
    was meant to say; we do not know what an ad tagged "misc" was meant to
    target, and guessing would put a wrong ad in front of a real user.
    """
    if category not in schema.CATEGORIES:
        return Violation(E_UNKNOWN_CATEGORY, f"category={category!r} is not in the taxonomy")
    return None


def check_timestamp(normalized: str | None, raw_value: object) -> Violation | None:
    if normalized is None:
        return Violation(E_BAD_TIMESTAMP, f"created_at={raw_value!r} is not valid ISO-8601")
    return None


# --- Duplicate rules (stateful) -------------------------------------------


class Deduplicator:
    """First-wins duplicate detection across a single pass of the input.

    Two sets, two different failures:

      * ad_id     -- a second record with the same id would overwrite the first
                     in the Phase 5 feature store, so whichever arrived last
                     would silently win.
      * content   -- identical ad copy wastes index space and, worse, returns
                     what looks like the same ad twice in one top-K result.

    This is the one part of the pipeline that is NOT constant-memory: dedup is
    inherently stateful. Two sets over 50k records is nothing. At 100M you
    would reach for a Bloom filter (accepting a small false-positive rate) or
    an external sort on the key -- worth knowing as the honest scale answer.
    """

    __slots__ = ("_seen_ids", "_seen_content")

    def __init__(self) -> None:
        self._seen_ids: set[str] = set()
        self._seen_content: set[str] = set()

    def check_and_record(self, ad_id: str, headline: str, description: str) -> Violation | None:
        if ad_id in self._seen_ids:
            return Violation(E_DUPLICATE_AD_ID, f"ad_id={ad_id!r} already seen in this batch")

        fingerprint = content_hash(headline, description)
        if fingerprint in self._seen_content:
            return Violation(
                E_DUPLICATE_CONTENT,
                f"identical cleaned text to an earlier ad (sha256={fingerprint[:12]})",
            )

        # Only record once the record has cleared both checks, so a rejected
        # record never blocks a later legitimate one.
        self._seen_ids.add(ad_id)
        self._seen_content.add(fingerprint)
        return None
