"""One test per rule, each with a passing case and a failing case.

Keeping the rules as separate small functions is what makes this file possible:
every check can be driven directly, without constructing a whole valid record
just to probe one condition.
"""

import pytest

from ingestion import rules, schema


# --- structural -----------------------------------------------------------


def valid_raw() -> dict:
    return {
        "ad_id": "ad_0000001",
        "headline": "Premium running shoes from Strider",
        "description": "Tested over 500 miles.",
        "category": "footwear",
        "advertiser_id": "adv_0007",
        "bid_cpm_usd": 4.25,
        "created_at": "2026-03-14T09:21:00Z",
    }


def test_required_fields_present():
    assert rules.check_required_fields(valid_raw()) is None


@pytest.mark.parametrize("field", schema.REQUIRED_FIELDS)
def test_absent_required_field_is_fatal(field):
    raw = valid_raw()
    del raw[field]
    violation = rules.check_required_fields(raw)
    assert violation.code == rules.E_MISSING_FIELD
    assert field in violation.detail


@pytest.mark.parametrize("field", schema.REQUIRED_FIELDS)
def test_null_required_field_is_just_as_fatal_as_an_absent_one(field):
    raw = valid_raw()
    raw[field] = None
    assert rules.check_required_fields(raw).code == rules.E_MISSING_FIELD


def test_description_is_optional():
    raw = valid_raw()
    del raw["description"]
    assert rules.check_required_fields(raw) is None


def test_wrong_type_is_fatal():
    raw = valid_raw()
    raw["headline"] = ["Premium", "running", "shoes"]
    assert rules.check_field_types(raw).code == rules.E_WRONG_TYPE


def test_numeric_bid_is_not_a_type_error():
    # bid_cpm_usd is deliberately excluded from the string-type check so that
    # "4.25" can be coerced rather than rejected.
    raw = valid_raw()
    raw["bid_cpm_usd"] = "4.25"
    assert rules.check_field_types(raw) is None


# --- headline -------------------------------------------------------------


def test_good_headline_passes():
    assert rules.check_headline("Premium running shoes") is None


def test_empty_headline_is_fatal():
    assert rules.check_headline("").code == rules.E_EMPTY_TEXT


def test_short_headline_is_fatal():
    # "Sale!" carries no signal for an embedding -- it would land somewhere
    # meaningless in vector space and surface for unrelated queries.
    assert rules.check_headline("Sale!").code == rules.E_HEADLINE_TOO_SHORT


def test_headline_at_the_minimum_length_is_accepted():
    assert rules.check_headline("a" * schema.HEADLINE_MIN_LEN) is None


def test_long_headline_is_fatal_rather_than_silently_truncated():
    assert rules.check_headline("a" * (schema.HEADLINE_MAX_LEN + 1)).code == rules.E_TEXT_TOO_LONG


def test_headline_at_the_maximum_length_is_accepted():
    assert rules.check_headline("a" * schema.HEADLINE_MAX_LEN) is None


# --- description ----------------------------------------------------------


def test_empty_description_is_fine():
    assert rules.check_description("") is None


def test_long_description_is_fatal():
    assert rules.check_description("a" * (schema.DESCRIPTION_MAX_LEN + 1)).code == rules.E_TEXT_TOO_LONG


# --- bid ------------------------------------------------------------------


def test_normal_bid_passes():
    assert rules.check_bid(4.25, 4.25) is None


def test_non_numeric_bid_is_fatal():
    assert rules.check_bid(None, "N/A").code == rules.E_BID_NOT_NUMERIC


@pytest.mark.parametrize("bid", [0.0, -0.01, -9.5])
def test_non_positive_bid_is_fatal(bid):
    assert rules.check_bid(bid, bid).code == rules.E_BID_OUT_OF_RANGE


def test_cents_sent_as_dollars_is_caught_by_the_ceiling():
    # 425 cents arriving where 4.25 dollars was expected. Without this rule the
    # ad wins every auction in Phase 6 and it looks like a ranking bug.
    assert rules.check_bid(425.0, 425.0).code == rules.E_BID_OUT_OF_RANGE


@pytest.mark.parametrize("bid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_bid_is_fatal(bid):
    assert rules.check_bid(bid, bid).code == rules.E_BID_OUT_OF_RANGE


def test_bid_exactly_at_the_ceiling_is_accepted():
    assert rules.check_bid(schema.BID_MAX, schema.BID_MAX) is None


# --- category and timestamp ----------------------------------------------


@pytest.mark.parametrize("category", sorted(schema.CATEGORIES))
def test_every_taxonomy_category_passes(category):
    assert rules.check_category(category) is None


@pytest.mark.parametrize("category", ["misc", "other", "", "Footwear", "sports"])
def test_unknown_category_is_fatal(category):
    # Note "Footwear" fails here: validate.py lowercases before calling this,
    # so by this point casing has already been normalised.
    assert rules.check_category(category).code == rules.E_UNKNOWN_CATEGORY


def test_good_timestamp_passes():
    assert rules.check_timestamp("2026-03-14T09:21:00Z", "2026-03-14T09:21:00Z") is None


def test_unparseable_timestamp_is_fatal():
    assert rules.check_timestamp(None, "yesterday").code == rules.E_BAD_TIMESTAMP


# --- duplicates -----------------------------------------------------------


def test_first_occurrence_of_an_ad_is_accepted():
    deduper = rules.Deduplicator()
    assert deduper.check_and_record("ad_1", "Premium running shoes", "desc") is None


def test_repeated_ad_id_is_fatal():
    deduper = rules.Deduplicator()
    deduper.check_and_record("ad_1", "Premium running shoes", "desc")
    violation = deduper.check_and_record("ad_1", "A completely different headline", "other")
    assert violation.code == rules.E_DUPLICATE_AD_ID


def test_repeated_content_under_a_new_id_is_fatal():
    deduper = rules.Deduplicator()
    deduper.check_and_record("ad_1", "Premium running shoes", "desc")
    violation = deduper.check_and_record("ad_2", "Premium running shoes", "desc")
    assert violation.code == rules.E_DUPLICATE_CONTENT


def test_a_rejected_record_does_not_reserve_its_id():
    """First-wins means the FIRST ACCEPTED record wins, not the first seen.

    If a rejected duplicate still recorded its id, a later legitimate record
    carrying that id would be wrongly thrown away too -- one bad row poisoning
    a good one.
    """
    deduper = rules.Deduplicator()
    deduper.check_and_record("ad_1", "Premium running shoes", "desc")
    deduper.check_and_record("ad_1", "Duplicate id, gets rejected", "desc2")
    # "ad_2" was never accepted before, and its text is new, so it must pass.
    assert deduper.check_and_record("ad_2", "Duplicate id, gets rejected", "desc2") is None
