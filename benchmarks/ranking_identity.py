# Copyright 2026 STARGA, Inc.
"""The reusable form of "latency only, no ranking movement".

A change that claims to move no ranking is making a *measurement* claim, and
until this module existed it was argued instead: the 5.0.2 hot-path work ran
two source trees over one workspace by hand and eyeballed that the served
``(id, score)`` lists matched. That worked once. It is not a gate, it left no
artifact, and nobody could re-run it.

What a fingerprint commits to
-----------------------------
The **served answer**: every returned id, in rank order, each paired with the
exact score that put it there. Not a set — collapsing the order would let a
reordering pass as identical, which is precisely the movement being watched
for. Scores go in as ``float.hex()``, so a one-ULP drift is a difference; a
"byte-identical" claim that tolerated rounding would not be one.

Three ways this kind of check silently proves nothing, all closed here:

* **Both sides empty.** ``() == ()`` passes and means nothing. Comparing
  fingerprints goes through :func:`assert_ranking_unchanged`, which refuses a
  comparison thinner than ``min_results`` rather than returning a pass.
* **The field got renamed.** A ``.get("_id", "")`` would turn every row into
  the same empty string and every comparison into a pass. Extraction raises on
  a missing id or score instead.
* **The probe never ran.** A battery that produced no queries would report no
  differences; :func:`fingerprint_battery` refuses an empty query list.

Everything here is pure: no clock, no I/O, no randomness. The digest reuses
``mind_mem.recall_digests.served_set_digest`` — one canonical encoding of "a
served answer" with one owner, rather than a second spelling of it living in
the benchmark tree.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_mem.recall_digests import served_set_digest  # noqa: E402

#: Default field names on a recall hit dict.
ID_FIELD = "_id"
SCORE_FIELD = "score"

#: A fingerprint is a tuple of ``(id, score_token)`` pairs in rank order.
Fingerprint = tuple[tuple[str, str], ...]


class VacuousComparison(Exception):
    """The comparison could not have failed, so its pass is not evidence.

    Raised rather than returning "identical" when there is too little to
    compare: an empty result list, or a battery with no queries. A gate that
    reports success over work it never inspected is the failure mode this
    project has already been bitten by twice.
    """


class RankingMoved(AssertionError):
    """Two fingerprints differ — the served ranking is not byte-identical.

    Carries the first divergence rather than only the fact of one, because
    "the ranking moved" is not actionable and "rank 3 went from block X at
    12.5 to block Y at 12.5000001" is.
    """


def _score_token(value: Any) -> str:
    """Canonical, lossless text for one score.

    ``float.hex()`` is exact and round-trips, so two scores share a token if
    and only if they share their bits — including the ``-0.0`` / ``0.0``
    distinction that ``==`` erases. ``bool`` is rejected before ``int``
    catches it: ``True`` is an ``int`` in Python, and a score that silently
    became a flag is a defect, not a value worth hashing.
    """
    if isinstance(value, bool):
        raise TypeError(f"score must be a number, got bool {value!r}")
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value.hex()}"
    raise TypeError(f"score must be int or float, got {type(value).__name__} {value!r}")


def ranking_fingerprint(
    results: Iterable[Mapping[str, Any]],
    *,
    id_field: str = ID_FIELD,
    score_field: str = SCORE_FIELD,
) -> Fingerprint:
    """The served ``(id, score)`` list, in rank order, canonically encoded.

    Args:
        results: Recall hits, in the order they were served.
        id_field: Key holding the block id.
        score_field: Key holding the ranking score.

    Raises:
        KeyError: A hit is missing its id or its score. Absent fields are not
            defaulted — a rename that emptied every row would otherwise make
            every comparison pass.
        TypeError: A score is not a number.
    """
    out: list[tuple[str, str]] = []
    for position, hit in enumerate(results):
        if id_field not in hit:
            raise KeyError(f"hit at rank {position + 1} has no {id_field!r}: {sorted(hit)}")
        if score_field not in hit:
            raise KeyError(f"hit at rank {position + 1} has no {score_field!r}: {sorted(hit)}")
        out.append((str(hit[id_field]), _score_token(hit[score_field])))
    return tuple(out)


def fingerprint_digest(fingerprint: Fingerprint) -> str:
    """One hex digest naming a served ranking.

    The pairs are flattened to ``id, score, id, score, …`` and handed to the
    canonical served-set encoder, whose length-prefixed framing makes the
    flattening unambiguous: ``("AB", "C")`` and ``("A", "BC")`` hash apart, so
    no id or score token can be split or merged into a colliding answer.
    """
    flat: list[str] = []
    for block_id, score in fingerprint:
        flat.append(block_id)
        flat.append(score)
    return served_set_digest(flat)


@dataclass(frozen=True)
class RankingDiff:
    """What changed between two served rankings, or that nothing did."""

    moved: bool
    n_before: int
    n_after: int
    digest_before: str
    digest_after: str
    first_divergence: int | None
    before_at_divergence: tuple[str, str] | None
    after_at_divergence: tuple[str, str] | None

    def describe(self) -> str:
        if not self.moved:
            return f"identical: {self.n_before} hit(s), digest {self.digest_before[:16]}"
        if self.first_divergence is None:  # pragma: no cover - defensive
            return "moved: length differs and no positional divergence was found"
        return (
            f"moved at rank {self.first_divergence + 1}: "
            f"{self.before_at_divergence!r} -> {self.after_at_divergence!r} "
            f"({self.n_before} -> {self.n_after} hit(s); "
            f"digest {self.digest_before[:16]} -> {self.digest_after[:16]})"
        )


def compare_rankings(before: Fingerprint, after: Fingerprint) -> RankingDiff:
    """Diff two fingerprints. Pure; reports, never raises on a difference."""
    first: int | None = None
    for position in range(min(len(before), len(after))):
        if before[position] != after[position]:
            first = position
            break
    if first is None and len(before) != len(after):
        first = min(len(before), len(after))
    return RankingDiff(
        moved=before != after,
        n_before=len(before),
        n_after=len(after),
        digest_before=fingerprint_digest(before),
        digest_after=fingerprint_digest(after),
        first_divergence=first,
        before_at_divergence=before[first] if first is not None and first < len(before) else None,
        after_at_divergence=after[first] if first is not None and first < len(after) else None,
    )


def assert_ranking_unchanged(
    before: Fingerprint,
    after: Fingerprint,
    *,
    label: str = "ranking",
    min_results: int = 1,
) -> RankingDiff:
    """Assert an OFF/unchanged path served a byte-identical ranking.

    Args:
        before: Fingerprint of the reference run.
        after: Fingerprint of the run under test.
        label: What is being compared, quoted in both failure messages.
        min_results: Fewest hits each side must carry for the comparison to
            mean anything. ``0`` disables the guard and must be justified at
            the call site — an empty-vs-empty pass is not evidence.

    Raises:
        VacuousComparison: Either side is thinner than ``min_results``.
        RankingMoved: The fingerprints differ.

    Returns:
        The (identical) diff, so a caller can record the digest it pinned.
    """
    if min_results > 0 and (len(before) < min_results or len(after) < min_results):
        raise VacuousComparison(
            f"{label}: refusing to call {len(before)} vs {len(after)} hit(s) identical; "
            f"at least {min_results} on each side are needed for the comparison to be able to fail"
        )
    diff = compare_rankings(before, after)
    if diff.moved:
        raise RankingMoved(f"{label}: {diff.describe()}")
    return diff


def fingerprint_battery(
    recall_fn: Callable[[str], Iterable[Mapping[str, Any]]],
    queries: Sequence[str],
    *,
    id_field: str = ID_FIELD,
    score_field: str = SCORE_FIELD,
) -> dict[str, Fingerprint]:
    """Fingerprint one ranking per query.

    A single query is a thin gate: a change can leave one ordering alone and
    move every other. Passing a battery makes the claim proportional to the
    change being defended.

    Raises:
        VacuousComparison: ``queries`` is empty, so the battery would report
            no differences having looked for none.
    """
    if not queries:
        raise VacuousComparison("a battery with no queries cannot detect a ranking change")
    return {query: ranking_fingerprint(recall_fn(query), id_field=id_field, score_field=score_field) for query in queries}


def assert_battery_unchanged(
    before: Mapping[str, Fingerprint],
    after: Mapping[str, Fingerprint],
    *,
    label: str = "battery",
    min_results: int = 1,
) -> dict[str, RankingDiff]:
    """Assert every query in a battery served a byte-identical ranking.

    Raises:
        VacuousComparison: The batteries cover different queries, or either
            is empty. Comparing only the intersection would let a query that
            stopped being asked read as a pass.
        RankingMoved: Any query's ranking moved; the message names every one
            that did, not just the first.
    """
    if not before or not after:
        raise VacuousComparison(f"{label}: an empty battery cannot detect a ranking change")
    if set(before) != set(after):
        only_before = sorted(set(before) - set(after))
        only_after = sorted(set(after) - set(before))
        raise VacuousComparison(f"{label}: the batteries ask different questions (before-only {only_before}, after-only {only_after})")
    diffs: dict[str, RankingDiff] = {}
    moved: list[str] = []
    for query in before:
        if min_results > 0 and (len(before[query]) < min_results or len(after[query]) < min_results):
            raise VacuousComparison(
                f"{label}: query {query!r} served {len(before[query])} vs {len(after[query])} hit(s); "
                f"at least {min_results} on each side are needed for the comparison to be able to fail"
            )
        diffs[query] = compare_rankings(before[query], after[query])
        if diffs[query].moved:
            moved.append(query)
    if moved:
        detail = "; ".join(f"{query!r} {diffs[query].describe()}" for query in moved)
        raise RankingMoved(f"{label}: {len(moved)} of {len(before)} quer(ies) moved -- {detail}")
    return diffs


__all__ = [
    "Fingerprint",
    "ID_FIELD",
    "SCORE_FIELD",
    "RankingDiff",
    "RankingMoved",
    "VacuousComparison",
    "assert_battery_unchanged",
    "assert_ranking_unchanged",
    "compare_rankings",
    "fingerprint_battery",
    "fingerprint_digest",
    "ranking_fingerprint",
]
