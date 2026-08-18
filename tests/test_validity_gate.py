"""Regression gate for the Phase-2 recall validity gate (Stage 2.65).

Locks in:
  * opt-in — flag absent is a byte-for-byte no-op (no ``validity`` /
    ``_validity_demoted`` annotation, deterministic default path).
  * discrimination — a clean/active block scores ``validity["score"] == 1.0``
    and is never demoted; a deprecated + contradicted block is demoted below
    both of its clean peers.
  * determinism — repeated flag-on runs produce byte-identical ordered
    ``(id, score, validity)`` tuples (no clock/rand leak into the preimage).
"""

from __future__ import annotations

import json
import os

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init

_QUERY = "corroboration staleness fixture widget rollout"

_DECISIONS_BODY = """
[D-20260301-001]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture widget rollout entry Alpha
Rationale: Validity gate regression fixture
Tags: validity-gate-fixture

[D-20260301-002]
Date: 2026-03-01
Status: deprecated
Scope: global
Statement: Corroboration staleness fixture widget rollout entry Beta
Rationale: Validity gate regression fixture
Tags: validity-gate-fixture

[D-20260301-003]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture widget rollout entry Gamma
Rationale: Validity gate regression fixture
Tags: validity-gate-fixture
"""

# Seeded UNRESOLVED contradiction naming B (D-20260301-002) as a party.
_CONTRADICTIONS_BODY = """
[C-20260301-001]
Date: 2026-03-01
Severity: high
Type: decision_vs_decision
Statement: Block D-20260301-002 conflicts with an established rollout decision
Status: open
Resolution: none
"""

_ID_A = "D-20260301-001"
_ID_B = "D-20260301-002"
_ID_C = "D-20260301-003"


def _seed_workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)

    decisions_path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(decisions_path, "a", encoding="utf-8") as fh:
        fh.write(_DECISIONS_BODY)

    intel_dir = os.path.join(ws, "intelligence")
    os.makedirs(intel_dir, exist_ok=True)
    with open(os.path.join(intel_dir, "CONTRADICTIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(_CONTRADICTIONS_BODY)

    return ws


def _enable_validity_gate(ws: str) -> None:
    """Flip ``recall.validity_gate.enabled`` on and force a distinct config
    mtime so ``_recall_core``'s mtime-cached config reload picks it up
    within a single test process (real-clock mtime resolution is too
    coarse to rely on across two writes a few milliseconds apart)."""
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["recall"]["validity_gate"] = {"enabled": True}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    new_mtime = os.path.getmtime(cfg_path) + 5.0
    os.utime(cfg_path, (new_mtime, new_mtime))


def _by_id(results: list[dict]) -> dict[str, dict]:
    return {r["_id"]: r for r in results}


def _id_score_tuples(results: list[dict]) -> list[tuple[str, float]]:
    return [(r["_id"], r["score"]) for r in results]


def test_validity_gate_discriminates_and_is_deterministic(tmp_path) -> None:
    ws = _seed_workspace(tmp_path)

    # --- 1. Opt-in proof: flag absent is a complete, deterministic no-op ---
    off_1 = recall(ws, _QUERY, limit=10)
    off_2 = recall(ws, _QUERY, limit=10)

    assert len(off_1) >= 3, f"fixture blocks not all recalled: {[r['_id'] for r in off_1]}"
    for r in off_1:
        assert "validity" not in r
        assert "_validity_demoted" not in r
    assert _id_score_tuples(off_1) == _id_score_tuples(off_2)

    baseline = _by_id(off_1)
    baseline_b_score = baseline[_ID_B]["score"]

    # --- 2. Discrimination proof: enable the flag, rerun ---
    _enable_validity_gate(ws)
    on_1 = recall(ws, _QUERY, limit=10)
    on_by_id = _by_id(on_1)

    assert _ID_A in on_by_id and _ID_B in on_by_id and _ID_C in on_by_id

    hit_a = on_by_id[_ID_A]
    assert hit_a["validity"]["score"] == 1.0
    assert "_validity_demoted" not in hit_a

    hit_b = on_by_id[_ID_B]
    assert hit_b["validity"]["status"] == 0.0
    assert hit_b["validity"]["contradiction"] == 0.0
    assert hit_b["validity"]["score"] == 0.5
    assert hit_b["validity"]["score"] < 0.8
    assert hit_b["_validity_demoted"] is True
    assert hit_b["score"] == round(baseline_b_score * 0.5, 4)

    hit_c = on_by_id[_ID_C]
    assert hit_c["score"] > hit_b["score"]
    assert hit_a["score"] > hit_b["score"]

    # --- 3. Determinism proof: two more flag-on runs, byte-identical ---
    on_2 = recall(ws, _QUERY, limit=10)
    on_3 = recall(ws, _QUERY, limit=10)

    def _ordered_tuples(results: list[dict]) -> list[tuple[str, float, dict]]:
        return [(r["_id"], r["score"], r["validity"]) for r in results]

    assert _ordered_tuples(on_1) == _ordered_tuples(on_2)
    assert _ordered_tuples(on_1) == _ordered_tuples(on_3)
