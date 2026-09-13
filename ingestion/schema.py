"""Shape of an ad record, plus every limit the validator enforces.

Everything tunable lives here so the rules in `rules.py` stay pure logic and you
only have one place to look when someone asks "why is the bid ceiling $50?".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# --- Taxonomy -------------------------------------------------------------
#
# A CLOSED set, not free text. Phase 6 ranking scores a category match between
# the query and the ad, so a category we don't recognise can never match
# anything -- it is a data error, not a new category. Keeping it closed here is
# what makes that ranking signal trustworthy later.
CATEGORIES: frozenset[str] = frozenset(
    {
        "apparel",
        "footwear",
        "electronics",
        "home_kitchen",
        "beauty",
        "fitness",
        "travel",
        "finance",
        "gaming",
        "pet_supplies",
        "food_delivery",
        "education",
    }
)


# --- Field contract -------------------------------------------------------

# Missing or null -> the record cannot be served, so it is rejected (E002).
REQUIRED_FIELDS: tuple[str, ...] = (
    "ad_id",
    "headline",
    "category",
    "advertiser_id",
    "bid_cpm_usd",
    "created_at",
)

# Absent is fine; we default it to "" and record a warning. An ad with only a
# headline is still a servable ad, just a weaker embedding.
OPTIONAL_FIELDS: tuple[str, ...] = ("description",)

# Fields that must arrive as strings. `bid_cpm_usd` is deliberately absent --
# it gets its own coercion path in clean.py because "4.25" is a fixable
# upstream quirk, not a fatal type error.
STRING_FIELDS: tuple[str, ...] = (
    "ad_id",
    "headline",
    "category",
    "advertiser_id",
    "created_at",
    "description",
)


# --- Limits ---------------------------------------------------------------

# Below this there is not enough text to produce a meaningful embedding in
# Phase 2 -- "Sale!" embeds to essentially noise.
HEADLINE_MIN_LEN = 10

# Above these the embedding model would silently truncate. We would rather
# reject loudly than embed half an ad and never know.
HEADLINE_MAX_LEN = 120
DESCRIPTION_MAX_LEN = 1000

# Bid is CPM in US dollars. The floor is exclusive: a zero or negative bid is
# never servable. The ceiling catches the classic unit bug -- a system that
# reports cents sending 425 where we expect 4.25 -- and fat-fingered manual
# entry. Both would otherwise dominate every auction in Phase 6.
BID_MIN_EXCLUSIVE = 0.0
BID_MAX = 50.0


@dataclass(slots=True)
class AdRecord:
    """A validated, cleaned ad. This is the contract Phase 2 consumes."""

    ad_id: str
    headline: str
    description: str
    category: str
    advertiser_id: str
    bid_cpm_usd: float
    created_at: str  # normalised ISO-8601 UTC

    def to_dict(self) -> dict:
        return asdict(self)
