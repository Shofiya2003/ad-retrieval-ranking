"""Tests for the pure normalisation helpers.

These are the easiest things in the project to test precisely, because they
have no I/O and no state: text in, text out.
"""

import pytest

from ingestion import clean


def test_collapse_whitespace_trims_and_squashes():
    assert clean.collapse_whitespace("  Big   Sale\n\ttoday  ") == "Big Sale today"


def test_strip_html_inserts_a_space_so_words_do_not_merge():
    # The whole reason tags become " " rather than "": otherwise this is "BigSale".
    assert clean.clean_text("<b>Big</b><i>Sale</i>")[0] == "Big Sale"


def test_strip_html_decodes_entities():
    assert clean.clean_text("Coffee &amp; Tea")[0] == "Coffee & Tea"


def test_tags_are_removed_before_entities_are_decoded():
    # A literal "&lt;b&gt;" in ad copy is text the advertiser wrote, not markup.
    # Decoding first would turn it into a tag and then delete it.
    assert clean.clean_text("Use &lt;b&gt; for bold")[0] == "Use <b> for bold"


def test_zero_width_space_is_removed():
    cleaned, warnings = clean.clean_text("run​ning shoes")
    assert cleaned == "running shoes"
    assert clean.W_CONTROL in warnings


def test_unicode_is_nfkc_folded():
    cleaned, warnings = clean.clean_text("Big Sale！")  # fullwidth "!"
    assert cleaned == "Big Sale!"
    assert clean.W_UNICODE in warnings


def test_clean_text_reports_every_transformation_it_applied():
    cleaned, warnings = clean.clean_text("  <b>Big​   Sale</b>  ")
    assert cleaned == "Big Sale"
    assert set(warnings) == {clean.W_HTML, clean.W_CONTROL, clean.W_WHITESPACE}


def test_clean_text_on_already_clean_input_warns_about_nothing():
    assert clean.clean_text("Premium running shoes") == ("Premium running shoes", [])


def test_whitespace_only_headline_becomes_empty():
    # This is exactly why cleaning has to run before the emptiness rule.
    assert clean.clean_text("   \t\n ")[0] == ""


@pytest.mark.parametrize(
    "value,expected",
    [(4.25, 4.25), (4, 4.0), (-3.0, -3.0), (0, 0.0)],
)
def test_coerce_bid_passes_numbers_through_without_warning(value, expected):
    assert clean.coerce_bid(value) == (expected, [])


@pytest.mark.parametrize("value,expected", [("4.25", 4.25), ("$4.25", 4.25), ("1,250.00", 1250.0)])
def test_coerce_bid_repairs_numeric_strings(value, expected):
    bid, warnings = clean.coerce_bid(value)
    assert bid == expected
    assert warnings == [clean.W_BID_COERCED]


def test_coerce_bid_rejects_booleans():
    # bool subclasses int in Python, so True would otherwise become a 1.0 bid.
    assert clean.coerce_bid(True) == (None, [])


@pytest.mark.parametrize("value", ["N/A", "", ["4.25"], {"bid": 4.25}, None])
def test_coerce_bid_returns_none_for_non_numbers(value):
    assert clean.coerce_bid(value)[0] is None


def test_timestamp_is_normalised_to_utc():
    assert clean.normalize_timestamp("2026-03-14T11:21:00+02:00")[0] == "2026-03-14T09:21:00Z"


def test_naive_timestamp_is_assumed_utc():
    assert clean.normalize_timestamp("2026-03-14T09:21:00")[0] == "2026-03-14T09:21:00Z"


@pytest.mark.parametrize("value", ["14/03/2026", "yesterday", "2026-13-45T00:00:00Z", ""])
def test_unparseable_timestamp_returns_none(value):
    assert clean.normalize_timestamp(value)[0] is None


def test_content_hash_ignores_case():
    assert clean.content_hash("BUY NOW", "Save big") == clean.content_hash("buy now", "SAVE BIG")


def test_content_hash_is_exact_match_only():
    # The documented limitation: near-duplicates survive. Catching these needs
    # the Phase 2 embeddings, not a hash.
    assert clean.content_hash("Buy now!", "x") != clean.content_hash("Buy now!!", "x")
