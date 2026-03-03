"""Basic parity metrics for command/response trace comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceMetrics:
    total_commands: int
    ok_matches: int
    ok_mismatches: int

    @property
    def ok_match_ratio(self) -> float:
        if self.total_commands == 0:
            return 1.0
        return self.ok_matches / float(self.total_commands)


def compute_ok_parity(left_ok: list[bool], right_ok: list[bool]) -> TraceMetrics:
    if len(left_ok) != len(right_ok):
        raise ValueError("left_ok and right_ok length mismatch")
    matches = 0
    mismatches = 0
    for left_value, right_value in zip(left_ok, right_ok, strict=True):
        if left_value == right_value:
            matches += 1
        else:
            mismatches += 1
    return TraceMetrics(
        total_commands=len(left_ok),
        ok_matches=matches,
        ok_mismatches=mismatches,
    )

