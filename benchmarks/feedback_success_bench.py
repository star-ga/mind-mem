#!/usr/bin/env python3
"""Feedback-quality -> downstream-success bench (Group I item 3).

Standing, deterministic proof that the v4.7.0 per-hit feedback-quality
credit (``retrieval_graph.feedback_quality_credit``) and the v4.8.0
recall-sufficiency score (``retrieval_graph.recall_sufficiency``) predict
whether a recall actually delivered enough for the downstream agent to
succeed.

Eval set: 48 synthetic, hand-authored episodes -- 8 intent classes
(``ENTITY`` .. ``TRACE``, i.e. ``INTENT_DEMAND`` 1.0-5.0) x 6 families
(sufficient / noisy-sufficient / starved / redundant / invalid /
not-retained). Every episode is a pure literal: fixed hits with a
pre-set ``validity`` annotation (so ``feedback_quality_credit`` never
touches the workspace -- it is a zero-I/O pure call over the hit list),
plus a ground-truth ``label`` computed once at import time by
:func:`_ground_truth_label`, which independently reimplements
"durable, valid, non-redundant" straight off the raw hit metadata --
it never calls the shared credit/validity helpers, so this bench
cannot trivially agree with itself.

Predictor (fixed, untrained, one threshold, no weights to fit):
``recall_sufficiency(hits, intent_type)["score"] >= SUFFICIENCY_STARVED_THRESHOLD``.

Fully deterministic: no clock, no randomness, no model, no network.
Episodes are enumerated (never sampled) and dates/ids are fixed
strings/counters.

Run:
    python benchmarks/feedback_success_bench.py
"""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Callable
from typing import Any, NamedTuple

from mind_mem._recall_constants import SUFFICIENCY_STARVED_THRESHOLD
from mind_mem.retrieval_graph import feedback_quality_credit, recall_sufficiency

# ---------------------------------------------------------------------------
# Episode literal
# ---------------------------------------------------------------------------


class Episode(NamedTuple):
    """One frozen synthetic recall episode.

    ``label`` is ground truth (computed independently of the predictor
    under test, see :func:`_ground_truth_label`) and is never seen by
    the predictor.
    """

    episode_id: str
    intent_type: str
    family: str
    hits: tuple[dict[str, Any], ...]
    needed_fact_ids: tuple[str, ...]
    label: bool


# Intent classes routed by the IntentRouter, in ``INTENT_DEMAND`` order,
# excluding LIST (demand 6.0) to keep the grid at exactly ENTITY..TRACE
# (demand 1.0-5.0) per the bench design.
_INTENTS: tuple[tuple[str, int], ...] = (
    ("ENTITY", 1),
    ("WHEN", 1),
    ("WHAT", 2),
    ("VERIFY", 2),
    ("WHY", 3),
    ("HOW", 3),
    ("COMPARE", 4),
    ("TRACE", 5),
)

# A near-duplicate text similarity above this fraction of shared tokens
# counts as "the same fact" for ground-truth purposes. Deliberately a
# different algorithm (token-set Jaccard) than the credit's cosine
# term-frequency similarity in ``dedup._cosine_similarity`` -- the label
# rule must not share code with the thing it grades.
_LABEL_DUP_JACCARD_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Fixture builders -- pure, deterministic hit construction
# ---------------------------------------------------------------------------


def _topic_text(topic_id: int) -> str:
    """A globally unique single-token excerpt for ``topic_id``.

    Unique tokens between genuinely distinct facts keep their cosine
    similarity at 0.0 (``non_redundant`` == 1.0); reusing the exact same
    string for two hits is how the F2 (redundant) family manufactures a
    near-duplicate on purpose.
    """
    return f"topicmarker{topic_id:05d}"


def _make_hit(
    block_id: int,
    excerpt: str,
    *,
    fact_id: str | None,
    score: float,
    validity_score: float,
    lifecycle: str = "durable",
    status: str = "active",
) -> dict[str, Any]:
    """Build one pre-annotated synthetic recall hit (Stage 2.65 shape).

    ``validity`` is pre-set so ``feedback_quality_credit`` never reads
    the contradiction log or staleness scores for this hit -- zero I/O.
    ``fact_id`` is bench-authoring metadata only (an underscore-prefixed
    key the credit/sufficiency code never inspects); it is read solely
    by :func:`_ground_truth_label` below.
    """
    return {
        "_id": f"SYN-{block_id:05d}",
        "score": round(score, 4),
        "Statement": f"Synthetic fact statement #{block_id}.",
        "excerpt": excerpt,
        "Status": status,
        "Lifecycle": lifecycle,
        "validity": {"score": round(validity_score, 4)},
        "_fact_id": fact_id,
    }


def _fresh_hit(topic_ids: itertools.count, *, fact_id: str | None, score: float, validity_score: float, **kw: Any) -> dict[str, Any]:
    """``_make_hit`` with a freshly minted, globally-unique excerpt."""
    topic_id = next(topic_ids)
    return _make_hit(topic_id, _topic_text(topic_id), fact_id=fact_id, score=score, validity_score=validity_score, **kw)


def _family_s1_sufficient(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """S1 -- ``demand`` distinct, durable, valid, on-task hits."""
    facts = [f"F{i}" for i in range(n)]
    hits = [_fresh_hit(topic_ids, fact_id=facts[i], score=1.0 - 0.01 * i, validity_score=1.0) for i in range(n)]
    return hits, facts


def _family_s2_noisy_sufficient(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """S2 -- S1 plus low-score, off-topic junk hits appended.

    ``recall_sufficiency`` sums useful-context mass, so low-informative
    noise can only add to the total, never subtract -- this proves it
    cannot flip an already-sufficient recall into a starved one.
    """
    hits, facts = _family_s1_sufficient(topic_ids, n)
    junk = [_fresh_hit(topic_ids, fact_id=None, score=0.02, validity_score=1.0) for _ in range(3)]
    return [*hits, *junk], facts


def _family_f1_starved(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """F1 -- fewer distinct on-task hits than demand.

    Only ``n // 3`` of the ``n`` needed facts are covered at all (zero
    for ``demand`` 1 or 2). ``informative`` is score *relative to the
    top score in the delivered set*, so padding a short list with an
    extra low-score off-topic hit does not depress it -- a single junk
    hit left on its own would score ``informative == 1.0`` (it is its
    own top score) and register as misleadingly "sufficient". The
    honest construction of "delivered less than demand" is therefore to
    simply deliver fewer hits, including zero for the low-demand
    classes -- ``recall_sufficiency`` correctly returns ``None`` (no
    ``feedback_credit`` at all) on an empty hit list.
    """
    facts = [f"F{i}" for i in range(n)]
    covered = n // 3
    hits = [_fresh_hit(topic_ids, fact_id=facts[i], score=1.0 - 0.01 * i, validity_score=1.0) for i in range(covered)]
    return hits, facts


def _family_f2_redundant(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """F2 -- hit count matches demand, but every hit is a near-paraphrase
    of the SAME single fact.

    Only ``facts[0]`` is ever genuinely (non-duplicate) covered; hits
    for ``facts[1:]`` reuse ``facts[0]``'s exact excerpt text, so both
    the credit's ``non_redundant`` component and the independent
    ground-truth near-duplicate check collapse them back onto fact 0.
    At ``demand == 1`` (ENTITY, WHEN) there is only one fact and no
    duplicate is even generated (``range(1, 1)`` is empty) -- padding
    cannot manufacture a coverage gap when only one thing was ever
    needed, so that case is a legitimate sufficient/success episode,
    not a distinct "redundant" story. That is mathematically forced:
    one genuinely valid, non-redundant hit always clears a demand of
    1.0 by itself, however many duplicates follow it.
    """
    facts = [f"F{i}" for i in range(n)]
    base_id = next(topic_ids)
    base_text = _topic_text(base_id)
    hits = [_make_hit(base_id, base_text, fact_id=facts[0], score=1.0, validity_score=0.9)]
    for i in range(1, n):
        dup_id = next(topic_ids)  # consume a fresh id for bookkeeping; reuse base_text on purpose
        hits.append(_make_hit(dup_id, base_text, fact_id=facts[i], score=1.0 - 0.01 * i, validity_score=0.9))
    return hits, facts


def _family_f3_invalid(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """F3 -- a needed hit carries ``validity.score == 0.0`` (contradicted).

    ``n // 3`` facts get a genuinely valid hit; the rest get a hit that
    nominally targets the fact but is contradicted (``validity_score=0``).
    """
    facts = [f"F{i}" for i in range(n)]
    valid_count = n // 3
    hits = []
    for i in range(n):
        validity_score = 1.0 if i < valid_count else 0.0
        hits.append(_fresh_hit(topic_ids, fact_id=facts[i], score=1.0 - 0.01 * i, validity_score=validity_score))
    return hits, facts


def _family_f4_not_retained(topic_ids: itertools.count, n: int) -> tuple[list[dict[str, Any]], list[str]]:
    """F4 -- a needed hit is ``Lifecycle: ephemeral`` / ``Status: superseded``.

    ``n // 3`` facts get a genuinely retained hit; the rest get a hit
    that nominally targets the fact but was pruned from the durable
    record (both fields set so either the lifecycle- or status- driven
    read of "not retained" trips on it).
    """
    facts = [f"F{i}" for i in range(n)]
    valid_count = n // 3
    hits = []
    for i in range(n):
        if i < valid_count:
            hits.append(_fresh_hit(topic_ids, fact_id=facts[i], score=1.0 - 0.01 * i, validity_score=1.0))
        else:
            hits.append(
                _fresh_hit(
                    topic_ids,
                    fact_id=facts[i],
                    score=1.0 - 0.01 * i,
                    validity_score=1.0,
                    lifecycle="ephemeral",
                    status="superseded",
                )
            )
    return hits, facts


_FamilyBuilder = Callable[[itertools.count, int], tuple[list[dict[str, Any]], list[str]]]

_FAMILIES: tuple[tuple[str, _FamilyBuilder], ...] = (
    ("S1_sufficient", _family_s1_sufficient),
    ("S2_noisy_sufficient", _family_s2_noisy_sufficient),
    ("F1_starved", _family_f1_starved),
    ("F2_redundant", _family_f2_redundant),
    ("F3_invalid", _family_f3_invalid),
    ("F4_not_retained", _family_f4_not_retained),
)


# ---------------------------------------------------------------------------
# Ground truth -- independent of feedback_quality_credit / recall_sufficiency
# ---------------------------------------------------------------------------


def _label_tokens(text: str) -> frozenset[str]:
    """Lowercase alnum token set. Independent of ``dedup._text_tokens``:
    no stop-word list, no length filter beyond non-empty -- a
    deliberately different (simpler) tokenizer than the credit path."""
    return frozenset(text.lower().split())


def _is_near_duplicate(tokens: frozenset[str], counted: list[frozenset[str]]) -> bool:
    """Token-set Jaccard near-duplicate check against already-counted
    hits. Independent of ``dedup._cosine_similarity`` (different metric,
    different code path) -- the label must not share math with the
    metric it is grading."""
    if not tokens:
        return False
    for prior in counted:
        union = tokens | prior
        if not union:
            continue
        jaccard = len(tokens & prior) / len(union)
        if jaccard >= _LABEL_DUP_JACCARD_THRESHOLD:
            return True
    return False


def _ground_truth_label(hits: list[dict[str, Any]], needed_fact_ids: list[str]) -> bool:
    """Success iff every needed fact is covered by a delivered hit that
    is durable+active, has ``validity.score >= 0.5``, and is not a
    token-set near-duplicate of a fact already counted.

    Reads only raw hit fields (``Status``, ``Lifecycle``, ``validity``,
    ``excerpt``, the authoring-only ``_fact_id``) -- never calls
    ``feedback_quality_credit``, ``recall_sufficiency``, or any shared
    validity/dedup helper.
    """
    covered: set[str] = set()
    counted_tokens: list[frozenset[str]] = []
    for hit in hits:
        fact_id = hit.get("_fact_id")
        if fact_id is None:
            continue
        tokens = _label_tokens(str(hit.get("excerpt", "")))
        if _is_near_duplicate(tokens, counted_tokens):
            continue
        lifecycle = str(hit.get("Lifecycle", "")).strip().lower()
        status = str(hit.get("Status", "")).strip().lower()
        validity_score = float((hit.get("validity") or {}).get("score", 0.0))
        if lifecycle == "durable" and status == "active" and validity_score >= 0.5:
            covered.add(fact_id)
            counted_tokens.append(tokens)
    return set(needed_fact_ids).issubset(covered)


# ---------------------------------------------------------------------------
# The 48-episode grid, built once at import time
# ---------------------------------------------------------------------------


def _build_episodes() -> tuple[Episode, ...]:
    topic_ids = itertools.count()
    episodes: list[Episode] = []
    for intent, demand in _INTENTS:
        for family_name, builder in _FAMILIES:
            hits, facts = builder(topic_ids, demand)
            label = _ground_truth_label(hits, facts)
            episodes.append(
                Episode(
                    episode_id=f"{intent}-{family_name}",
                    intent_type=intent,
                    family=family_name,
                    hits=tuple(hits),
                    needed_fact_ids=tuple(facts),
                    label=label,
                )
            )
    return tuple(episodes)


EPISODES: tuple[Episode, ...] = _build_episodes()

_FEEDBACK_CREDIT_CFG: dict[str, Any] = {"feedback_credit": {"enabled": True}}
_DUMMY_WORKSPACE = "unused://feedback-success-bench"


# ---------------------------------------------------------------------------
# Predictor + metrics
# ---------------------------------------------------------------------------


def _predict(episode: Episode) -> float:
    """Annotate a deep copy of the episode's hits and return the
    predicted ``recall_sufficiency`` score in ``[0, 1]``.

    Deep-copies before annotating so ``EPISODES`` (module-level,
    imported once) is never mutated -- ``run_bench()`` stays safe to
    call repeatedly and byte-identical.
    """
    hits = copy.deepcopy(list(episode.hits))
    feedback_quality_credit(hits, _DUMMY_WORKSPACE, _FEEDBACK_CREDIT_CFG)
    sufficiency = recall_sufficiency(hits, episode.intent_type)
    return float(sufficiency["score"]) if isinstance(sufficiency, dict) else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    total = len(y_true) or 1
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_bench() -> dict[str, Any]:
    """Run the full 48-episode grid and return the summary metrics dict.

    Pure and deterministic: same inputs every call, no I/O beyond the
    dummy in-memory workspace string, no clock, no randomness.
    """
    y_true: list[bool] = []
    y_pred: list[bool] = []
    scores_by_label: dict[bool, list[float]] = {True: [], False: []}
    labels_by_predicted: dict[bool, list[bool]] = {True: [], False: []}
    per_family: dict[str, dict[str, Any]] = {}

    for episode in EPISODES:
        score = _predict(episode)
        predicted = score >= SUFFICIENCY_STARVED_THRESHOLD

        y_true.append(episode.label)
        y_pred.append(predicted)
        scores_by_label[episode.label].append(score)
        labels_by_predicted[predicted].append(episode.label)

        fam = per_family.setdefault(episode.family, {"n": 0, "_correct": 0, "_scores": []})
        fam["n"] += 1
        fam["_correct"] += int(predicted == episode.label)
        fam["_scores"].append(score)

    metrics = _classification_metrics(y_true, y_pred)
    mean_sufficiency_success = round(_mean(scores_by_label[True]), 4)
    mean_sufficiency_failure = round(_mean(scores_by_label[False]), 4)

    per_family_summary = {
        name: {
            "n": stats["n"],
            "accuracy": round(stats["_correct"] / stats["n"], 4),
            "mean_sufficiency": round(_mean(stats["_scores"]), 4),
        }
        for name, stats in per_family.items()
    }

    return {
        **metrics,
        "mean_sufficiency_success": mean_sufficiency_success,
        "mean_sufficiency_failure": mean_sufficiency_failure,
        "separation": round(mean_sufficiency_success - mean_sufficiency_failure, 4),
        "success_rate_predicted_starved": round(_mean([1.0 if lbl else 0.0 for lbl in labels_by_predicted[False]]), 4),
        "success_rate_predicted_sufficient": round(_mean([1.0 if lbl else 0.0 for lbl in labels_by_predicted[True]]), 4),
        "n_episodes": len(EPISODES),
        "threshold": SUFFICIENCY_STARVED_THRESHOLD,
        "per_family": per_family_summary,
    }


if __name__ == "__main__":
    print(json.dumps(run_bench(), sort_keys=True, indent=2))
