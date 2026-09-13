"""Stage 1 of the ingestion pipeline: raw ads in, clean ads + a report out.

    uv run python -m ingestion.validate --input data/raw/ads_raw.jsonl \
        --out-dir data/clean --fail-under 0.90

The design in one line: CLEAN FIRST, THEN VALIDATE -- because cleaned text is
what Phase 2 embeds, so cleaned text is what the length and emptiness rules
have to judge.

Everything streams. One line in, one line out, constant memory regardless of
input size. The only state that grows is the duplicate-detection sets in
rules.Deduplicator, which is unavoidable: you cannot know something is a
duplicate without remembering what you have seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from ingestion import clean, rules, schema
from ingestion.report import ValidationReport
from ingestion.schema import AdRecord

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_INPUT_ERROR = 2


def sha256_of_file(path: Path) -> str:
    """Fingerprint the input so a report can always be traced to exactly the
    bytes that produced it. Costs one extra read of the file, which is cheap
    next to the value of a reproducible artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(raw: dict, deduper: rules.Deduplicator):
    """Validate one parsed record.

    Returns (AdRecord, warnings) on success, or (None, Violation) on rejection.
    The ordering below is the heart of the phase, so it is worth reading in
    order:

      1. structural checks on the RAW dict   -- you cannot clean a field that
                                                is missing or is a list
      2. cleaning                            -- normalise into what will
                                                actually be embedded
      3. content checks on the CLEANED values
      4. duplicate checks last               -- only a record that is otherwise
                                                valid deserves to claim an id
    """
    warnings: list[str] = []

    # --- 1. structural ---------------------------------------------------
    violation = rules.check_required_fields(raw)
    if violation:
        return None, violation

    violation = rules.check_field_types(raw)
    if violation:
        return None, violation

    # --- 2. cleaning -----------------------------------------------------
    headline, w = clean.clean_text(raw["headline"])
    warnings += w

    raw_description = raw.get("description")
    if raw_description is None:
        description = ""
        warnings.append(clean.W_DESC_MISSING)
    else:
        description, w = clean.clean_text(raw_description)
        warnings += w

    # Category is lowercased as well as cleaned: "Footwear " and "footwear"
    # are the same category, and case is never meaningful in a taxonomy key.
    category, w = clean.clean_text(raw["category"])
    category = category.lower()
    warnings += w

    ad_id = raw["ad_id"].strip()
    advertiser_id = raw["advertiser_id"].strip()

    bid, w = clean.coerce_bid(raw["bid_cpm_usd"])
    warnings += w

    created_at, w = clean.normalize_timestamp(raw["created_at"])
    warnings += w

    # --- 3. content ------------------------------------------------------
    for violation in (
        rules.check_headline(headline),
        rules.check_description(description),
        rules.check_bid(bid, raw["bid_cpm_usd"]),
        rules.check_category(category),
        rules.check_timestamp(created_at, raw["created_at"]),
    ):
        if violation:
            return None, violation

    # --- 4. duplicates ---------------------------------------------------
    violation = deduper.check_and_record(ad_id, headline, description)
    if violation:
        return None, violation

    record = AdRecord(
        ad_id=ad_id,
        headline=headline,
        description=description,
        category=category,
        advertiser_id=advertiser_id,
        bid_cpm_usd=bid,
        created_at=created_at,
    )
    return record, warnings


def run(
    input_path: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    config: dict | None = None,
) -> ValidationReport:
    """Stream the input file through the validator, writing all three outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "ads_clean.jsonl"
    reject_path = out_dir / "ads_rejected.jsonl"
    report_path = out_dir / "validation_report.json"

    report = ValidationReport(
        input_path=str(input_path),
        input_sha256=sha256_of_file(input_path),
        config=config or {},
    )
    deduper = rules.Deduplicator()
    started = time.perf_counter()

    with (
        input_path.open("r", encoding="utf-8") as source,
        clean_path.open("w", encoding="utf-8") as clean_out,
        reject_path.open("w", encoding="utf-8") as reject_out,
    ):

        def reject(line_no: int, code: str, detail: str, payload, excerpt: str) -> None:
            report.record_reject(code, detail, line_no, excerpt)
            reject_out.write(
                json.dumps(
                    {"line": line_no, "code": code, "detail": detail, "record": payload},
                    ensure_ascii=False,
                )
                + "\n"
            )

        for line_no, line in enumerate(source, start=1):
            if limit is not None and line_no > limit:
                break

            line = line.strip()
            if not line:
                continue  # blank lines are noise, not records -- not counted

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                # One corrupt line does not kill the file. This is a concrete
                # reason the format is JSONL rather than one big JSON array.
                reject(line_no, rules.E_MALFORMED_JSON, str(exc), None, line)
                continue

            if not isinstance(raw, dict):
                reject(
                    line_no,
                    rules.E_WRONG_TYPE,
                    f"record must be a JSON object, got {type(raw).__name__}",
                    raw,
                    line,
                )
                continue

            record, result = validate_record(raw, deduper)
            if record is None:
                reject(line_no, result.code, result.detail, raw, line)
                continue

            report.record_accept(result)
            clean_out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    report.duration_seconds = time.perf_counter() - started
    report.write_json(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingestion.validate",
        description="Validate and clean raw ad records for the serving pipeline.",
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/ads_raw.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "Exit non-zero if the pass rate falls below RATE (e.g. 0.90). This is "
            "the pipeline gate: it is what stops a bad batch reaching Phase 2."
        ),
    )
    parser.add_argument(
        "--max-bid",
        type=float,
        default=None,
        help=f"Override the bid ceiling in USD CPM (default {schema.BID_MAX}).",
    )
    parser.add_argument(
        "--min-headline-len",
        type=int,
        default=None,
        help=f"Override the minimum headline length (default {schema.HEADLINE_MIN_LEN}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only read the first N lines.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the terminal summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # The single place thresholds are mutated. Rules read them from schema at
    # call time, so overriding here is enough -- and recording them in the
    # report below means a run is never ambiguous about which limits applied.
    if args.max_bid is not None:
        schema.BID_MAX = args.max_bid
    if args.min_headline_len is not None:
        schema.HEADLINE_MIN_LEN = args.min_headline_len

    config = {
        "bid_max": schema.BID_MAX,
        "bid_min_exclusive": schema.BID_MIN_EXCLUSIVE,
        "headline_min_len": schema.HEADLINE_MIN_LEN,
        "headline_max_len": schema.HEADLINE_MAX_LEN,
        "description_max_len": schema.DESCRIPTION_MAX_LEN,
        "categories": sorted(schema.CATEGORIES),
        "fail_under": args.fail_under,
    }

    report = run(args.input, args.out_dir, limit=args.limit, config=config)

    if not args.quiet:
        print(report.format_summary())
        print(f"  clean   -> {args.out_dir / 'ads_clean.jsonl'}")
        print(f"  rejects -> {args.out_dir / 'ads_rejected.jsonl'}")
        print(f"  report  -> {args.out_dir / 'validation_report.json'}\n")

    if args.fail_under is not None and report.pass_rate < args.fail_under:
        print(
            f"GATE FAILED: pass rate {report.pass_rate:.2%} is below the "
            f"required {args.fail_under:.2%}. Downstream stages will not run.",
            file=sys.stderr,
        )
        return EXIT_GATE_FAILED

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
