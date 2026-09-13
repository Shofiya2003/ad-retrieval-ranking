#!/usr/bin/env python3
"""Compare the validator's report against the generator's seeded-defect oracle.

Because the generator injects exactly one defect per record and records what it
injected, the validator's per-code counts should match it exactly. Any
disagreement means one of the two is wrong -- and this script is how you find
out which, instead of eyeballing two JSON files side by side.

    python scripts/check_oracle.py

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=Path("data/raw/expected_defects.json"))
    parser.add_argument("--report", type=Path, default=Path("data/clean/validation_report.json"))
    args = parser.parse_args()

    for path in (args.oracle, args.report):
        if not path.is_file():
            print(f"error: {path} not found -- run the generator and validator first", file=sys.stderr)
            return 2

    oracle = json.loads(args.oracle.read_text())
    report = json.loads(args.report.read_text())

    expected = oracle["expected_rejections_by_code"]
    actual = report["rejections_by_code"]

    print(f"{'CODE':<28} {'EXPECTED':>9} {'ACTUAL':>9}   ")
    print("-" * 56)

    mismatches = 0
    for code in sorted(set(expected) | set(actual)):
        want, got = expected.get(code, 0), actual.get(code, 0)
        ok = want == got
        mismatches += not ok
        print(f"{code:<28} {want:>9,} {got:>9,}   {'ok' if ok else 'MISMATCH'}")

    print("-" * 56)
    for label, want, got in (
        ("accepted", oracle["expected_accepted_total"], report["totals"]["accepted"]),
        ("rejected", oracle["expected_rejected_total"], report["totals"]["rejected"]),
        ("read", oracle["count"], report["totals"]["read"]),
    ):
        ok = want == got
        mismatches += not ok
        print(f"{label:<28} {want:>9,} {got:>9,}   {'ok' if ok else 'MISMATCH'}")

    if mismatches:
        print(f"\n{mismatches} mismatch(es). Either a rule or the generator is wrong.", file=sys.stderr)
        return 1

    print("\nAll counts match the oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
