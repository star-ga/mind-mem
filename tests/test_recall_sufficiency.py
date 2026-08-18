"""Regression gate for Group I item 2 recall-sufficiency score.

Locks in:
  * flag-off no-op — no ``feedback_credit`` on any hit means the pure
    function returns ``None`` and diagnostics stay silent
    (``queries_scored == 0``).
  * discrimination — the SAME corpus, scored for two different
    ``IntentRouter`` classes, produces "enough" (ENTITY, demand 1.0) vs
    "starved" (LIST, demand 6.0) sufficiency scores.
  * both surfacing paths — ``retrieval_diagnostics``'s additive
    ``recall_sufficiency`` block, and the MCP ``pack_recall_budget`` tool's
    additive ``sufficiency`` key.
  * determinism — repeated flag-on runs produce byte-identical sufficiency
    dicts (no clock/rand leak into the preimage).
"""

from __future__ import annotations

import json
import os

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init
from mind_mem.mcp.tools.recall import pack_recall_budget
from mind_mem.retrieval_graph import recall_sufficiency, retrieval_diagnostics

_DECISIONS_BODY = """
[D-20260301-001]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Alpha
Rationale: Recall sufficiency regression fixture
Tags: recall-sufficiency-fixture

[D-20260301-002]
Date: 2026-03-01
Status: deprecated
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Beta
Rationale: Recall sufficiency regression fixture
Tags: recall-sufficiency-fixture

[D-20260301-004]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Alpha
Rationale: Recall sufficiency regression fixture (near-duplicate of 001)
Tags: recall-sufficiency-fixture
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


def _enable_flags(ws: str) -> None:
    """Flip ``recall.validity_gate`` and ``recall.feedback_credit`` on and
    force a distinct config mtime so ``_recall_core``'s mtime-cached config
    reload picks it up within a single test process (real-clock mtime
    resolution is too coarse to rely on across two writes a few
    milliseconds apart)."""
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["recall"]["validity_gate"] = {"enabled": True}
    cfg["recall"]["feedback_credit"] = {"enabled": True}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    new_mtime = os.path.getmtime(cfg_path) + 5.0
    os.utime(cfg_path, (new_mtime, new_mtime))


def test_sufficiency_discriminates_and_is_deterministic(tmp_path, monkeypatch) -> None:
    ws = _seed_workspace(tmp_path)
    ENOUGH = "tell me about the credit widget rollout"  # -> ENTITY, demand 1.0
    STARVED = "list credit widget rollout entries"  # -> LIST, demand 6.0

    # 1. Flag off: no credits -> pure fn returns None; diagnostics silent.
    off = recall(ws, ENOUGH, limit=10)
    assert recall_sufficiency(off, "ENTITY") is None
    assert retrieval_diagnostics(ws)["recall_sufficiency"]["queries_scored"] == 0

    # 2. Discrimination: same corpus, two query classes.
    _enable_flags(ws)  # validity_gate + feedback_credit
    hits_e, hits_s = recall(ws, ENOUGH, limit=10), recall(ws, STARVED, limit=10)
    s_e, s_s = recall_sufficiency(hits_e, "ENTITY"), recall_sufficiency(hits_s, "LIST")
    assert s_e["score"] == 1.0  # E>=1 clean hit / demand 1 -> enough
    assert s_s["score"] <= 0.5 and s_s["demand"] == 6.0  # same evidence mass / demand 6 -> starved
    assert 0.0 <= s_s["score"] < s_e["score"] <= 1.0

    # 3. Surfaces: persisted stage_counts -> diagnostics block, per class.
    diag = retrieval_diagnostics(ws)["recall_sufficiency"]
    assert diag["queries_scored"] == 2 and diag["starved_rate"] == 0.5
    assert diag["by_intent"]["ENTITY"]["avg"] == 1.0
    assert diag["by_intent"]["LIST"]["avg"] <= 0.5

    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)  # MCP tool surface
    packed = json.loads(pack_recall_budget(ENOUGH, max_tokens=2000, limit=10))
    assert packed["sufficiency"]["score"] == 1.0
    assert packed["sufficiency"]["pre_pack_score"] == 1.0

    # 4. Determinism: repeated flag-on runs byte-identical end to end.
    rerun = [recall_sufficiency(recall(ws, STARVED, limit=10), "LIST") for _ in range(2)]
    assert json.dumps(s_s, sort_keys=True) == json.dumps(rerun[0], sort_keys=True) == json.dumps(rerun[1], sort_keys=True)
