"""Accumulates what happened during a validation run and renders it two ways:
a machine-readable JSON artifact and a human-readable terminal summary.

The report is not decoration. It is the only thing standing between "we ingested
50,000 ads" and "we ingested 44,000 ads and threw 6,000 away for reasons nobody
looked at". Every rejection is counted by code, and a handful of real examples
are kept per code so you can eyeball whether the rule is doing what you think.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

MAX_EXAMPLES_PER_CODE = 5


class ValidationReport:
    def __init__(self, *, input_path: str, input_sha256: str, config: dict) -> None:
        self.input_path = input_path
        self.input_sha256 = input_sha256
        self.config = config

        self.total_read = 0
        self.accepted = 0
        self.reject_counts: Counter[str] = Counter()
        self.warning_counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict]] = defaultdict(list)
        self.duration_seconds = 0.0

    # --- recording -------------------------------------------------------

    def record_accept(self, warnings: list[str]) -> None:
        self.total_read += 1
        self.accepted += 1
        for code in warnings:
            self.warning_counts[code] += 1

    def record_reject(self, code: str, detail: str, line_number: int, raw_excerpt: str) -> None:
        self.total_read += 1
        self.reject_counts[code] += 1
        if len(self.examples[code]) < MAX_EXAMPLES_PER_CODE:
            self.examples[code].append(
                {"line": line_number, "detail": detail, "excerpt": raw_excerpt[:200]}
            )

    # --- derived ---------------------------------------------------------

    @property
    def rejected(self) -> int:
        return sum(self.reject_counts.values())

    @property
    def pass_rate(self) -> float:
        """Fraction of input records that survived. 1.0 for an empty input --
        an empty batch has nothing wrong with it, it is just empty."""
        if self.total_read == 0:
            return 1.0
        return self.accepted / self.total_read

    # --- rendering -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            # Provenance: which exact file, and with which thresholds. Without
            # this a report is unreproducible six months later.
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "config": self.config,
            "duration_seconds": round(self.duration_seconds, 3),
            "totals": {
                "read": self.total_read,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "pass_rate": round(self.pass_rate, 6),
            },
            "rejections_by_code": dict(self.reject_counts.most_common()),
            "warnings_by_code": dict(self.warning_counts.most_common()),
            "examples": {code: self.examples[code] for code, _ in self.reject_counts.most_common()},
        }

    def write_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    def format_summary(self) -> str:
        lines = [
            "",
            "=" * 62,
            f"  VALIDATION REPORT  ({self.input_path})",
            "=" * 62,
            f"  read      {self.total_read:>9,}",
            f"  accepted  {self.accepted:>9,}   ({self.pass_rate:.2%})",
            f"  rejected  {self.rejected:>9,}",
            f"  elapsed   {self.duration_seconds:>9.2f}s",
        ]

        if self.reject_counts:
            lines += ["", "  Rejections by reason:"]
            for code, count in self.reject_counts.most_common():
                share = count / self.total_read if self.total_read else 0.0
                lines.append(f"    {code:<28} {count:>7,}  ({share:6.2%})")

        if self.warning_counts:
            lines += ["", "  Records cleaned (kept, not rejected):"]
            for code, count in self.warning_counts.most_common():
                lines.append(f"    {code:<28} {count:>7,}")

        lines.append("=" * 62)
        return "\n".join(lines)
