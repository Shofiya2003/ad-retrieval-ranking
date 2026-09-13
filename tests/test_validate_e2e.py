"""End-to-end: a small hand-built fixture file through the real CLI.

The unit tests prove each rule in isolation. This file proves the ORDER is
right -- that cleaning runs before validation, that duplicates are judged last,
and that the three output artifacts and the exit-code gate actually work.
"""

import json

import pytest

from ingestion import rules, schema, validate


@pytest.fixture(autouse=True)
def restore_thresholds():
    """validate.main() can override module-level thresholds from CLI flags.
    Snapshot and restore them so one test cannot leak into the next."""
    saved = (schema.BID_MAX, schema.HEADLINE_MIN_LEN)
    yield
    schema.BID_MAX, schema.HEADLINE_MIN_LEN = saved


def base(**overrides) -> dict:
    record = {
        "ad_id": "ad_base",
        "headline": "Premium running shoes from Strider",
        "description": "Tested over 500 miles.",
        "category": "footwear",
        "advertiser_id": "adv_0007",
        "bid_cpm_usd": 4.25,
        "created_at": "2026-03-14T09:21:00Z",
    }
    record.update(overrides)
    return record


@pytest.fixture
def fixture_file(tmp_path):
    """One record per rule, in file order. Line 1 is the record that the two
    duplicate cases later collide with."""
    lines = [
        json.dumps(base(ad_id="ad_001")),                                    # accepted
        json.dumps(base(ad_id="ad_002", headline="<b>Everyday  trail runners</b>",
                        bid_cpm_usd="$3.10")),                               # accepted, cleaned
        "{not valid json",                                                   # E001
        json.dumps({k: v for k, v in base(ad_id="ad_004").items() if k != "category"}),  # E002
        json.dumps(base(ad_id="ad_005", headline=["a", "list"])),            # E003
        json.dumps(base(ad_id="ad_006", headline="   \t ")),                 # E004
        json.dumps(base(ad_id="ad_007", headline="<b>Sale!</b>")),           # E005
        json.dumps(base(ad_id="ad_008", headline="x" * 200)),                # E006
        json.dumps(base(ad_id="ad_009", bid_cpm_usd="N/A")),                 # E007
        json.dumps(base(ad_id="ad_010", bid_cpm_usd=425.0)),                 # E008
        json.dumps(base(ad_id="ad_011", category="misc")),                   # E009
        json.dumps(base(ad_id="ad_001", headline="A totally different headline here")),  # E010
        json.dumps(base(ad_id="ad_013")),                                    # E011 (same text as ad_001)
        json.dumps(base(ad_id="ad_014", created_at="yesterday")),            # E012
        "",                                                                  # blank: skipped entirely
        json.dumps(base(ad_id="ad_016", category="  Footwear ",
                        headline="All-season hiking boots from Oakstep",
                        description="Grip that holds on wet rock.")),      # accepted, category normalised
    ]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_every_rule_fires_exactly_once(fixture_file, tmp_path):
    report = validate.run(fixture_file, tmp_path / "out")

    assert report.total_read == 15  # 16 lines, one blank and not counted
    assert report.accepted == 3
    assert report.rejected == 12
    assert dict(report.reject_counts) == {code: 1 for code in rules.ALL_FATAL_CODES}


def test_cleaning_runs_before_validation(fixture_file, tmp_path):
    """"<b>Sale!</b>" is 12 raw characters but 5 real ones.

    Judged raw it would sail past the 10-character minimum and land a useless
    embedding in the index. This is the single most important ordering claim in
    the phase, so it gets its own assertion.
    """
    report = validate.run(fixture_file, tmp_path / "out")
    assert report.reject_counts[rules.E_HEADLINE_TOO_SHORT] == 1


def test_accepted_records_are_written_clean(fixture_file, tmp_path):
    out_dir = tmp_path / "out"
    validate.run(fixture_file, out_dir)

    records = [json.loads(line) for line in (out_dir / "ads_clean.jsonl").read_text().splitlines()]
    by_id = {r["ad_id"]: r for r in records}

    # HTML stripped, whitespace collapsed, and the string bid coerced to a float.
    assert by_id["ad_002"]["headline"] == "Everyday trail runners"
    assert by_id["ad_002"]["bid_cpm_usd"] == 3.10
    assert isinstance(by_id["ad_002"]["bid_cpm_usd"], float)

    # Category cleaned and lowercased, so "  Footwear " is not a new category.
    assert by_id["ad_016"]["category"] == "footwear"


def test_rejected_records_keep_their_reason_and_line_number(fixture_file, tmp_path):
    out_dir = tmp_path / "out"
    validate.run(fixture_file, out_dir)

    rejects = [json.loads(line) for line in (out_dir / "ads_rejected.jsonl").read_text().splitlines()]
    assert len(rejects) == 12
    assert all({"line", "code", "detail", "record"} <= set(r) for r in rejects)

    malformed = next(r for r in rejects if r["code"] == rules.E_MALFORMED_JSON)
    assert malformed["line"] == 3  # a corrupt line is isolated, the file survives


def test_report_json_is_written_with_provenance(fixture_file, tmp_path):
    out_dir = tmp_path / "out"
    validate.run(fixture_file, out_dir, config={"bid_max": schema.BID_MAX})

    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["totals"]["read"] == 15
    assert len(report["input_sha256"]) == 64
    assert report["config"]["bid_max"] == schema.BID_MAX
    # Examples are kept so you can eyeball whether a rule is doing what you think.
    assert report["examples"][rules.E_BID_OUT_OF_RANGE][0]["detail"]


def test_warnings_are_counted_without_rejecting(fixture_file, tmp_path):
    from ingestion import clean

    report = validate.run(fixture_file, tmp_path / "out")
    assert report.warning_counts[clean.W_HTML] == 1
    assert report.warning_counts[clean.W_BID_COERCED] == 1
    assert report.accepted == 3


def test_empty_input_is_not_an_error(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    report = validate.run(empty, tmp_path / "out")
    assert report.total_read == 0
    assert report.pass_rate == 1.0  # nothing wrong with it, it is just empty


# --- the pipeline gate ----------------------------------------------------


def test_gate_passes_when_pass_rate_is_high_enough(fixture_file, tmp_path):
    code = validate.main(
        ["--input", str(fixture_file), "--out-dir", str(tmp_path / "out"),
         "--fail-under", "0.10", "--quiet"]
    )
    assert code == validate.EXIT_OK


def test_gate_blocks_the_batch_when_too_much_was_rejected(fixture_file, tmp_path):
    # 3/15 accepted is 20%, well under this threshold.
    code = validate.main(
        ["--input", str(fixture_file), "--out-dir", str(tmp_path / "out"),
         "--fail-under", "0.90", "--quiet"]
    )
    assert code == validate.EXIT_GATE_FAILED


def test_missing_input_file_is_reported_distinctly(tmp_path):
    code = validate.main(["--input", str(tmp_path / "nope.jsonl"), "--quiet"])
    assert code == validate.EXIT_INPUT_ERROR


def test_threshold_override_changes_the_outcome(tmp_path):
    """Proves --max-bid actually reaches the rules, and is recorded in the report."""
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(base(ad_id="ad_x", bid_cpm_usd=80.0)) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    validate.main(["--input", str(path), "--out-dir", str(out_dir), "--quiet"])
    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["totals"]["accepted"] == 0  # 80.0 is over the default $50 ceiling

    validate.main(["--input", str(path), "--out-dir", str(out_dir), "--max-bid", "100", "--quiet"])
    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["totals"]["accepted"] == 1
    assert report["config"]["bid_max"] == 100.0
