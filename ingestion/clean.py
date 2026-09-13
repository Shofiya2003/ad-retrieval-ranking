"""Pure normalisation helpers. No I/O, no rejections -- these only ever *fix*.

Why this module exists separately, and why it runs BEFORE validation:

    A headline of "   " is not empty until you trim it.
    A headline of "<b>Sale!</b>" is not 12 characters of ad copy, it is 5.

Length and emptiness checks therefore have to run against cleaned text, because
cleaned text is what Phase 2 actually embeds. Judging the raw string would both
reject good ads and accept junk ones.

Every function here is total: it takes a value and returns a cleaned value plus
the list of warning codes explaining what it touched. Nothing raises.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone

# Warning codes -- the record is KEPT, we just record that we changed it.
W_WHITESPACE = "W001_WHITESPACE_NORMALIZED"
W_HTML = "W002_HTML_STRIPPED"
W_UNICODE = "W003_UNICODE_NORMALIZED"
W_CONTROL = "W004_CONTROL_CHARS_REMOVED"
W_BID_COERCED = "W005_BID_COERCED_FROM_STRING"
W_DESC_MISSING = "W006_DESCRIPTION_MISSING"
W_TIMESTAMP_NORMALIZED = "W007_TIMESTAMP_NORMALIZED"

_TAG_RE = re.compile(r"<[^>]{1,200}>")
_WHITESPACE_RE = re.compile(r"\s+")
# Control characters, but NOT \t \n \r -- those are whitespace and are handled
# by the collapse step, which turns them into a single space rather than
# deleting them and gluing two words together.
_CONTROL_STRIP_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿]")


def normalize_unicode(text: str) -> str:
    """NFKC fold: curly quotes, full-width letters and ligatures collapse to
    their plain ASCII-ish equivalents so that two ads written differently but
    reading identically hash the same in the duplicate check."""
    return unicodedata.normalize("NFKC", text)


def strip_html(text: str) -> str:
    """Remove tags, then decode entities.

    Tags become a SPACE, not an empty string: "<b>Big</b><i>Sale</i>" must not
    become "BigSale". The collapse step tidies up the extra spaces afterwards.

    Tags are removed before entities are decoded, so a literal "&lt;b&gt;" in
    ad copy survives as visible text instead of being decoded into a tag and
    then deleted.
    """
    return html.unescape(_TAG_RE.sub(" ", text))


def remove_control_chars(text: str) -> str:
    """Drop non-printing characters (NULs, zero-width spaces, BOMs).

    These are invisible in a terminal but are real bytes to a tokenizer, and a
    zero-width space in the middle of a word splits it into two tokens.
    """
    return _CONTROL_STRIP_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Any run of whitespace becomes a single space; trim the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> tuple[str, list[str]]:
    """Run the full normalisation chain, reporting what it had to change."""
    warnings: list[str] = []

    step = normalize_unicode(text)
    if step != text:
        warnings.append(W_UNICODE)

    before = step
    step = strip_html(step)
    if step != before:
        warnings.append(W_HTML)

    before = step
    step = remove_control_chars(step)
    if step != before:
        warnings.append(W_CONTROL)

    before = step
    step = collapse_whitespace(step)
    if step != before:
        warnings.append(W_WHITESPACE)

    return step, warnings


def coerce_bid(value: object) -> tuple[float | None, list[str]]:
    """Return (bid, warnings), or (None, ...) if it is not a number at all.

    Returning None means "not numeric" -- deciding whether that is fatal is
    rules.py's job, not ours. A bid that arrives as the string "4.25" (or
    "$4.25") is a fixable upstream quirk, so we coerce it and warn.
    """
    # bool is a subclass of int in Python, and True would silently become 1.0.
    if isinstance(value, bool):
        return None, []
    if isinstance(value, (int, float)):
        return float(value), []
    if isinstance(value, str):
        candidate = value.strip().replace("$", "").replace(",", "")
        try:
            return float(candidate), [W_BID_COERCED]
        except ValueError:
            return None, []
    return None, []


def normalize_timestamp(value: str) -> tuple[str | None, list[str]]:
    """Parse an ISO-8601 timestamp and re-emit it in one canonical UTC form.

    Returns (None, []) if it will not parse. A naive timestamp (no offset) is
    assumed to be UTC -- an assumption worth stating, because guessing a local
    timezone here would silently shift every ad's age in Phase 6.
    """
    if not isinstance(value, str):
        return None, []
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None, []

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    canonical = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return canonical, [] if canonical == value else [W_TIMESTAMP_NORMALIZED]


def content_hash(headline: str, description: str) -> str:
    """Stable fingerprint of an ad's cleaned text, used for duplicate detection.

    Case-folded so that "BUY NOW, SAVE BIG" and "Buy Now, Save Big" collide.
    Note this is EXACT match only -- "Buy now!" and "Buy now!!" are different
    fingerprints. Catching semantic near-duplicates needs the Phase 2
    embeddings; this is deliberately the cheap version.
    """
    payload = f"{headline.casefold()}\n{description.casefold()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
