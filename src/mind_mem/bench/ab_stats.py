"""The delta, with its uncertainty attached.

The design is paired: both arms attempt the same task, so the right test
is McNemar's, computed **exactly** (an exact binomial on the discordant
pairs) rather than through the chi-square approximation, which is not
trustworthy at the counts this benchmark produces.

Only the pairs where the arms disagreed carry information.  Ten tasks where
both arms failed say nothing; two tasks where they differed say a little.
That is why every summary reports ``n_discordant`` next to the headline,
and why :func:`smallest_significant_discordant` is reported too: below that
many discordant pairs **no** split can reach the significance level, so a
one-task difference is arithmetically incapable of being evidence and the
report says so instead of headlining it.

All arithmetic is exact integer / rational (``math.comb`` and
``fractions.Fraction``); the reported p-value is rounded once, at the end,
for display.  Nothing here reads a clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any, Iterable, Sequence

#: Conventional threshold. Stated, not hidden, and reported with the result.
DEFAULT_ALPHA = Fraction(1, 20)

#: Guard against a pathological input dominating a run.
MAX_PAIRS = 100_000


@dataclass(frozen=True)
class PairedSummary:
    """Successes per arm and the exact paired test over their disagreements."""

    n_tasks: int
    memory_successes: int
    control_successes: int
    both: int
    memory_only: int
    control_only: int
    neither: int
    p_value: Fraction
    alpha: Fraction
    min_discordant_for_significance: int
    verdict: str
    note: str

    @property
    def n_discordant(self) -> int:
        return self.memory_only + self.control_only

    @property
    def delta(self) -> int:
        return self.memory_successes - self.control_successes

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "memory_successes": self.memory_successes,
            "control_successes": self.control_successes,
            "delta_successes": self.delta,
            "both_passed": self.both,
            "memory_only": self.memory_only,
            "control_only": self.control_only,
            "neither_passed": self.neither,
            "n_discordant": self.n_discordant,
            "test": "mcnemar_exact_two_sided",
            "p_value": round(float(self.p_value), 6),
            "alpha": round(float(self.alpha), 6),
            "min_discordant_for_significance": self.min_discordant_for_significance,
            "verdict": self.verdict,
            "note": self.note,
        }


def mcnemar_exact(memory_only: int, control_only: int) -> Fraction:
    """Two-sided exact McNemar p-value over the discordant pairs."""
    if min(memory_only, control_only) < 0:
        raise ValueError("discordant counts cannot be negative")
    total = memory_only + control_only
    if total == 0 or memory_only == control_only:
        return Fraction(1)
    if total > MAX_PAIRS:
        raise ValueError(f"{total} discordant pairs exceeds the {MAX_PAIRS} guard")
    smaller = min(memory_only, control_only)
    tail = sum(comb(total, i) for i in range(smaller + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**total))


def smallest_significant_discordant(alpha: Fraction = DEFAULT_ALPHA, limit: int = 200) -> int:
    """Fewest discordant pairs that could ever reach ``alpha``.

    A perfect split (every disagreement favouring one arm) is the most
    extreme outcome available at a given number of discordant pairs; if
    even that cannot clear ``alpha``, the sample is arithmetically
    incapable of significance and no result at that size is evidence.
    """
    for total in range(1, limit + 1):
        if mcnemar_exact(total, 0) <= alpha:
            return total
    return limit + 1  # pragma: no cover - unreachable for any sane alpha


def _verdict(summary: tuple[int, int], p_value: Fraction, alpha: Fraction, floor: int) -> tuple[str, str]:
    """Name the outcome honestly, including when there is nothing to name."""
    memory_only, control_only = summary
    total = memory_only + control_only
    if total == 0:
        return "no_evidence", "The arms agreed on every task; a paired test has nothing to work with."
    if total < floor:
        return (
            "underpowered",
            f"{total} discordant pair(s): below {floor}, no split can reach p<={float(alpha):g}, "
            "so this difference cannot be evidence at any effect size. Report it as noise.",
        )
    if p_value > alpha:
        return (
            "not_significant",
            f"{total} discordant pair(s) split {memory_only}/{control_only}; p={float(p_value):.4f} does not clear {float(alpha):g}.",
        )
    direction = "memory_better" if memory_only > control_only else "control_better"
    return direction, f"{total} discordant pair(s) split {memory_only}/{control_only}; p={float(p_value):.4f}."


def summarise(pairs: Sequence[tuple[bool, bool]] | Iterable[tuple[bool, bool]], alpha: Fraction = DEFAULT_ALPHA) -> PairedSummary:
    """Summarise ``(memory_success, control_success)`` pairs, one per task."""
    rows = list(pairs)
    both = sum(1 for m, c in rows if m and c)
    memory_only = sum(1 for m, c in rows if m and not c)
    control_only = sum(1 for m, c in rows if c and not m)
    neither = sum(1 for m, c in rows if not m and not c)
    p_value = mcnemar_exact(memory_only, control_only)
    floor = smallest_significant_discordant(alpha)
    verdict, note = _verdict((memory_only, control_only), p_value, alpha, floor)
    return PairedSummary(
        n_tasks=len(rows),
        memory_successes=both + memory_only,
        control_successes=both + control_only,
        both=both,
        memory_only=memory_only,
        control_only=control_only,
        neither=neither,
        p_value=p_value,
        alpha=alpha,
        min_discordant_for_significance=floor,
        verdict=verdict,
        note=note,
    )
