"""Regression gate for the validity gate's FIFTH component (provenance class).

Locks in the respec acceptance gate:

  * one composite path — the standalone ``trust_scores`` surface emits the
    exact float the gate folds into ``V``; there is no second scorer.
  * ordering — ``operator > agent-verified > agent-inferred > external-ingest``
    strictly, both as a component and as a composite.
  * demotion — externally-ingested content is demoted below otherwise
    identical operator content in a real end-to-end recall.
  * opt-in — with the sub-flag off the composite is the original
    four-criteria mean and recall ordering is byte-identical (scores AND
    order compared against the pre-gate pipeline, not merely "close").
  * determinism — repeated runs are byte-identical; no clock, no randomness.
"""

from __future__ import annotations

import json
import os

from mind_mem._recall_constants import VALIDITY_DEMOTION, VALIDITY_GATE_THRESHOLD
from mind_mem._recall_core import recall
from mind_mem.init_workspace import init
from mind_mem.provenance_class import (
    AGENT_INFERRED,
    AGENT_VERIFIED,
    EXTERNAL_INGEST,
    OPERATOR,
    PROVENANCE_ORDER,
    UNKNOWN,
)
from mind_mem.trust_scores import TRUST_FIELD, apply_trust_scores
from mind_mem.validity_gate import provenance_component, validity_components

_QUERY = "provenance ingest ledger widget rollout"

# Three blocks that differ ONLY in provenance. All three are `wip`, which
# debits the status component to 0.5 — enough that the fifth component
# decides which side of the 0.8 threshold each block lands on.
_DECISIONS_BODY = """
[D-20260401-001]
Date: 2026-04-01
Status: wip
Scope: global
ActorId: operator-1
ActorRole: operator
Statement: Provenance ingest ledger widget rollout entry authored under an operator role
Rationale: Provenance class regression fixture
Tags: provenance-class-fixture

[D-20260401-002]
Date: 2026-04-01
Status: wip
Scope: global
ActorId: forum-sync
ActorRole: importer
ToolId: imported:forum
Source: imported:forum
Statement: Provenance ingest ledger widget rollout provenance ingest ledger widget rollout pulled from an external feed
Rationale: Provenance class regression fixture
Tags: provenance-class-fixture

[D-20260401-003]
Date: 2026-04-01
Status: wip
Scope: global
Statement: Provenance ingest ledger widget rollout entry carrying no provenance fields at all
Rationale: Provenance class regression fixture
Tags: provenance-class-fixture
"""

_ID_OPERATOR = "D-20260401-001"
_ID_EXTERNAL = "D-20260401-002"
_ID_LEGACY = "D-20260401-003"

# Expected four-criteria composite for a `wip` block with no contradiction
# and no staleness: 0.25 * (1.0 + 0.5 + 1.0 + 1.0).
_FOUR_COMPONENT_WIP = 0.875
_FOUR_KEYS = {"corroboration", "status", "contradiction", "staleness", "score"}
_FIVE_KEYS = _FOUR_KEYS | {"provenance", "provenance_class"}


def _seed_workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(_DECISIONS_BODY)
    return ws


def _configure_gate(ws: str, *, provenance: bool) -> None:
    """Enable the gate (and optionally its fifth component), forcing a
    distinct config mtime so ``_recall_core``'s mtime-cached reload picks the
    change up within one test process."""
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["recall"]["validity_gate"] = {"enabled": True, "provenance_class": {"enabled": provenance}}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    new_mtime = os.path.getmtime(cfg_path) + 5.0
    os.utime(cfg_path, (new_mtime, new_mtime))


def _by_id(results: list[dict]) -> dict[str, dict]:
    return {r["_id"]: r for r in results}


def _id_score_tuples(results: list[dict]) -> list[tuple[str, float]]:
    return [(r["_id"], r["score"]) for r in results]


def _hit(**fields: object) -> dict:
    base = {"_id": "B-1", "Status": "active"}
    base.update(fields)
    return base


# --- pure composite math ----------------------------------------------------


class TestCompositeMath:
    def test_provenance_off_is_exactly_the_four_component_composite(self) -> None:
        components = validity_components(_hit(ActorRole="importer"), set(), {})
        assert set(components) == _FOUR_KEYS
        assert components["score"] == 1.0  # active + clean + fresh + neutral

    def test_provenance_on_folds_the_fifth_component(self) -> None:
        components = validity_components(_hit(ActorRole="importer"), set(), {}, provenance_enabled=True)
        assert set(components) == _FIVE_KEYS
        assert components["provenance"] == 0.25
        assert components["provenance_class"] == EXTERNAL_INGEST
        assert components["score"] == round(0.2 * (1.0 + 1.0 + 1.0 + 1.0 + 0.25), 4)

    def test_composite_is_strictly_ordered_by_provenance_class(self) -> None:
        hits = {
            OPERATOR: _hit(ActorRole="operator"),
            AGENT_VERIFIED: _hit(ActorRole="planner", Verified="true"),
            AGENT_INFERRED: _hit(ActorRole="planner"),
            EXTERNAL_INGEST: _hit(ActorRole="importer"),
        }
        scores = [validity_components(hits[name], set(), {}, provenance_enabled=True)["score"] for name in PROVENANCE_ORDER]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(PROVENANCE_ORDER)
        classes = [validity_components(hits[n], set(), {}, provenance_enabled=True)["provenance_class"] for n in PROVENANCE_ORDER]
        assert classes == list(PROVENANCE_ORDER)

    def test_absent_provenance_is_neutral_not_demoted(self) -> None:
        """A corpus predating provenance fields must never be punished."""
        components = validity_components(_hit(), set(), {}, provenance_enabled=True)
        assert components["provenance_class"] == UNKNOWN
        assert components["provenance"] == 1.0
        assert components["score"] == 1.0

    def test_repeated_runs_are_byte_identical(self) -> None:
        hit = _hit(ActorRole="importer", ToolId="imported:forum")
        runs = {json.dumps(validity_components(hit, set(), {}, provenance_enabled=True), sort_keys=True) for _ in range(50)}
        assert len(runs) == 1

    def test_confirmed_ids_promote_only_agent_blocks(self) -> None:
        confirmed = frozenset({"B-1"})
        agent = validity_components(_hit(ActorId="a"), set(), {}, provenance_enabled=True, confirmed_ids=confirmed)
        external = validity_components(_hit(ActorRole="importer"), set(), {}, provenance_enabled=True, confirmed_ids=confirmed)
        legacy = validity_components(_hit(), set(), {}, provenance_enabled=True, confirmed_ids=confirmed)
        assert agent["provenance_class"] == AGENT_VERIFIED
        assert external["provenance_class"] == EXTERNAL_INGEST
        assert legacy["provenance_class"] == UNKNOWN

    def test_standalone_surface_emits_the_gate_component(self) -> None:
        """One composite path: ``actor_trust`` IS ``validity["provenance"]``."""
        hits = [_hit(_id="B-1", ActorRole="operator"), _hit(_id="B-2", ActorRole="importer")]
        annotated = apply_trust_scores(
            [dict(h) for h in hits],
            config={"retrieval": {"trust_scores": {"enabled": True}}},
        )
        gate = [validity_components(h, set(), {}, provenance_enabled=True)["provenance"] for h in hits]
        assert [r[TRUST_FIELD] for r in annotated] == gate == [provenance_component(h) for h in hits]


# --- end-to-end recall ------------------------------------------------------


def test_provenance_component_is_opt_in_and_flag_off_recall_is_byte_identical(tmp_path) -> None:
    ws = _seed_workspace(tmp_path)

    # 1. Gate entirely off — the pre-gate pipeline, twice, no annotation.
    off_1 = recall(ws, _QUERY, limit=10)
    off_2 = recall(ws, _QUERY, limit=10)
    assert len(off_1) >= 3, f"fixture blocks not all recalled: {[r['_id'] for r in off_1]}"
    assert _id_score_tuples(off_1) == _id_score_tuples(off_2)
    for r in off_1:
        assert "validity" not in r and "_validity_demoted" not in r

    # 2. Gate ON, fifth component OFF — the original four-criteria gate:
    #    same order, same scores, and no provenance keys anywhere.
    _configure_gate(ws, provenance=False)
    gate_only = recall(ws, _QUERY, limit=10)
    assert _id_score_tuples(gate_only) == _id_score_tuples(off_1)
    for r in gate_only:
        assert set(r["validity"]) == _FOUR_KEYS
        assert "_validity_demoted" not in r
    for block_id in (_ID_OPERATOR, _ID_EXTERNAL, _ID_LEGACY):
        assert _by_id(gate_only)[block_id]["validity"]["score"] == _FOUR_COMPONENT_WIP


def test_external_ingest_is_demoted_below_operator_content(tmp_path) -> None:
    ws = _seed_workspace(tmp_path)

    baseline = _by_id(recall(ws, _QUERY, limit=10))
    baseline_order = [r["_id"] for r in recall(ws, _QUERY, limit=10)]
    assert baseline_order.index(_ID_EXTERNAL) < baseline_order.index(_ID_OPERATOR), (
        f"fixture must start with the external block on top: {baseline_order}"
    )

    _configure_gate(ws, provenance=True)
    on_1 = recall(ws, _QUERY, limit=10)
    on_by_id = _by_id(on_1)
    order = [r["_id"] for r in on_1]

    external = on_by_id[_ID_EXTERNAL]
    assert external["validity"]["provenance_class"] == EXTERNAL_INGEST
    assert external["validity"]["provenance"] == 0.25
    assert external["validity"]["score"] == round(0.2 * (1.0 + 0.5 + 1.0 + 1.0 + 0.25), 4)
    assert external["validity"]["score"] < VALIDITY_GATE_THRESHOLD
    assert external["_validity_demoted"] is True
    assert external["score"] == round(baseline[_ID_EXTERNAL]["score"] * VALIDITY_DEMOTION, 4)

    operator = on_by_id[_ID_OPERATOR]
    assert operator["validity"]["provenance_class"] == OPERATOR
    assert operator["validity"]["score"] == round(0.2 * (1.0 + 0.5 + 1.0 + 1.0 + 1.0), 4)
    assert "_validity_demoted" not in operator
    assert operator["score"] == baseline[_ID_OPERATOR]["score"]

    # A block with no provenance is neutral — never demoted for being legacy.
    legacy = on_by_id[_ID_LEGACY]
    assert legacy["validity"]["provenance_class"] == UNKNOWN
    assert "_validity_demoted" not in legacy
    assert legacy["score"] == baseline[_ID_LEGACY]["score"]

    # The ordering actually flipped: external ingest now sits below both.
    assert operator["score"] > external["score"]
    assert legacy["score"] > external["score"]
    assert order.index(_ID_OPERATOR) < order.index(_ID_EXTERNAL)

    # Determinism: two more flag-on runs, byte-identical annotations.
    def _ordered(results: list[dict]) -> list[tuple[str, float, str]]:
        return [(r["_id"], r["score"], json.dumps(r["validity"], sort_keys=True)) for r in results]

    assert _ordered(on_1) == _ordered(recall(ws, _QUERY, limit=10))
    assert _ordered(on_1) == _ordered(recall(ws, _QUERY, limit=10))
