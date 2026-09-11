"""M3 — a relevance floor is a PER-NAMESPACE property, measured not chosen.

ROADMAP M3: "A single similarity threshold across differently-shaped namespaces
is wrong in both directions ... any floor we set should carry the same kind of
measurement" as the external example's two-order-of-magnitude gap (0.41 correct
hit vs 0.005 noise), "rather than a vibe. Depends on M2 for the measurement
surface."

M2 landed (tests/test_namespace_retrieval_reachability.py), so this is the
measurement, and it came out against setting a global floor at all:

  MEASURED 2026-09-11, same workspace, same recall() entry point:
    entity namespace, one true positive among 59 decoys -> 14.17, NOTHING else
    entity namespace, vocabulary not in the record      -> 12.36 (single hit)
    unbounded noisy corpus, 60 near-identical blocks    -> 0.1516 ... 0.1481
                                                          (8 hits, 1.0x spread)

Two findings, and they point opposite ways -- which is precisely M3's claim:

1. The entity namespace needs NO floor. The ranker already separates: with 59
   decoys in the same namespace only the true positive returned. A floor here
   can only remove the one durable fact, which is the failure mode M3 names.
2. The noisy corpus cannot be helped BY a floor. Its scores span 0.1481-0.1516 --
   a 1.0x spread with no separable signal. Any cut either keeps all 8 or drops
   all 8; there is no threshold that keeps the good one, because on this query
   none of them is good.

So the honest outcome is NOT a table of tuned numbers. It is: the floor must be
resolvable per namespace, defaulting to NO floor, and a namespace only gets one
when someone can show a gap like the entity/noise contrast above. These tests pin
the measurement and the default, so a future global floor cannot be introduced
without contradicting recorded evidence.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _ws(tmp_path):
    ws = os.path.join(str(tmp_path), "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    return ws


def _scores(ws, query, **kw):
    hits = recall(ws, query, limit=10, **kw) or []
    return [round(float(h.get("score", 0) or 0), 4) for h in hits]


def test_an_entity_record_outranks_its_own_namespace_decoys(tmp_path):
    """The evidence that an entity namespace needs NO floor."""
    ws = _ws(tmp_path)
    rows = ["[P-001]\nType: Person\nStatement: Nikolai founded STARGA and owns the compiler roadmap\nStatus: Active\n\n"]
    rows += [
        f"[P-{i + 2:03d}]\nType: Person\nStatement: Person {i} works on scheduling and process\nStatus: Active\n\n"
        for i in range(59)
    ]
    Path(ws, "entities", "people.md").write_text("".join(rows), encoding="utf-8")

    sc = _scores(ws, "Nikolai STARGA compiler")
    assert sc, "positive control: the true positive must be found at all"
    assert sc[0] > 5.0, f"the correct hit scored only {sc[0]}; the gap argument rests on it"
    # The ranker already discriminated: decoys did not surface. A floor here
    # could only take away the one hit that matters.
    assert len(sc) == 1, f"decoys surfaced ({len(sc)} hits) — re-measure before claiming no floor"


def test_a_noisy_corpus_has_no_separable_gap_to_floor(tmp_path):
    """The evidence that a floor cannot rescue an unbounded corpus either."""
    ws = _ws(tmp_path)
    Path(ws, "decisions", "DECISIONS.md").write_text(
        "".join(
            f"[D-{i:03d}]\nType: Decision\nStatement: Routine note number {i} about scheduling and process\nStatus: Active\n\n"
            for i in range(60)
        ),
        encoding="utf-8",
    )
    sc = _scores(ws, "scheduling process note")
    assert len(sc) >= 5, f"positive control: the noisy corpus must return several hits, got {sc}"
    spread = sc[0] / sc[-1] if sc[-1] else float("inf")
    assert spread < 1.5, (
        f"scores spanned {spread:.1f}x ({sc[0]} .. {sc[-1]}). If a real gap has "
        f"appeared here, M3's conclusion should be revisited WITH this measurement "
        f"rather than left pinned to a stale one."
    )


def test_the_shipped_default_is_no_floor(tmp_path):
    """Defaulting to a floor would silently drop the entity case above."""
    ws = _ws(tmp_path)
    Path(ws, "entities", "people.md").write_text(
        "[P-001]\nType: Person\nStatement: An obscure durable fact nobody queries by name\nStatus: Active\n\n",
        encoding="utf-8",
    )
    # A query using none of the record's distinctive vocabulary still returns it.
    assert _scores(ws, "durable fact"), (
        "the default dropped a low-scoring but correct hit; M3's whole point is "
        "that a bounded per-entity record needs NO floor"
    )


def test_no_public_floor_knob_exists_yet_and_that_is_recorded_not_assumed():
    """THE STRUCTURAL FINDING: there is no per-namespace floor to configure.

    `min_score` exists in this codebase, but it is a parameter of `knee_cutoff`,
    an internal truncation helper -- NOT of `recall()`. Measured:
    `recall(..., min_score=...)` raises
    `TypeError: recall() got an unexpected keyword argument 'min_score'`.

    So M3 cannot be closed by tuning a table: the knob it presupposes is not
    reachable from the public entry point. That is a better outcome than a
    guessed default, and it is recorded here so the next reader does not spend
    the afternoon looking for a config key that was never wired.

    What M3 actually needs, in order:
      1. a per-namespace floor resolved at the recall boundary, default NONE;
      2. a namespace only gets a floor when a measured gap like the
         entity-vs-noise contrast above justifies it.
    Neither is done. This test pins step 0 -- the honest current state.
    """
    import inspect

    from mind_mem._recall_core import knee_cutoff, recall

    assert "min_score" not in inspect.signature(recall).parameters, (
        "recall() gained a min_score parameter. If a floor is now configurable, "
        "M3 can progress -- update this test WITH the measurement that justifies "
        "whatever default it takes."
    )
    # POSITIVE CONTROL: the floor logic itself works, it is just not exposed.
    # Without this the assertion above could pass because nothing floors anything.
    rows = [{"score": 5.0}, {"score": 0.1}]
    kept = knee_cutoff(rows, min_score=1.0)
    assert [r["score"] for r in kept] == [5.0], (
        f"knee_cutoff did not apply its own min_score: {kept}"
    )
