"""Regression gate for Group I per-hit feedback-quality credit (Stage 3.1).

Locks in:
  * opt-in — flag absent is a byte-for-byte no-op (no ``feedback_credit``
    annotation, deterministic default path).
  * discrimination — each of the four components separates a crafted good
    hit from a crafted bad hit: top-rank vs validity-demoted for
    ``informative``, clean vs contradicted+deprecated for ``valid``, unique
    vs identical-text for ``non_redundant``, active vs deprecated for
    ``retained``.
  * the shared-helper contract — ``feedback_credit["valid"]`` equals
    ``validity["score"]`` for every hit (one source of truth, not parallel
    code).
  * determinism — repeated flag-on runs produce byte-identical
    ``(id, score, feedback_credit)`` tuples (no clock/rand leak into the
    preimage).
"""

from __future__ import annotations

import json
import os

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init

_QUERY = "corroboration staleness fixture credit widget rollout"

_DECISIONS_BODY = """
[D-20260301-001]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Alpha
Rationale: Feedback credit regression fixture
Tags: feedback-credit-fixture

[D-20260301-002]
Date: 2026-03-01
Status: deprecated
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Beta
Rationale: Feedback credit regression fixture
Tags: feedback-credit-fixture

[D-20260301-004]
Date: 2026-03-01
Status: active
Scope: global
Statement: Corroboration staleness fixture credit widget rollout entry Alpha
Rationale: Feedback credit regression fixture (near-duplicate of 001)
Tags: feedback-credit-fixture
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
_ID_D = "D-20260301-004"


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


def test_feedback_credit_discriminates_and_is_deterministic(tmp_path) -> None:
    ws = _seed_workspace(tmp_path)

    # 1. Opt-in: flag absent is a complete no-op.
    off_1 = recall(ws, _QUERY, limit=10)
    off_2 = recall(ws, _QUERY, limit=10)
    assert all("feedback_credit" not in r for r in off_1)
    assert [(r["_id"], r["score"]) for r in off_1] == [(r["_id"], r["score"]) for r in off_2]

    # 2. Discrimination: both flags on.
    _enable_flags(ws)  # validity_gate + feedback_credit, mtime-bumped
    on_1 = recall(ws, _QUERY, limit=10)
    by_id = {r["_id"]: r for r in on_1}

    for r in on_1:  # shape + range invariant
        fc = r["feedback_credit"]
        assert set(fc) == {"informative", "valid", "non_redundant", "retained"}
        assert all(0.0 <= v <= 1.0 for v in fc.values())

    # informative: top-of-list = 1.0; the validity-demoted B scores lower.
    assert on_1[0]["feedback_credit"]["informative"] == 1.0
    assert by_id[_ID_B]["feedback_credit"]["informative"] < 1.0

    # valid: ONE source of truth — equals the Stage-2.65 composite per hit.
    assert all(r["feedback_credit"]["valid"] == r["validity"]["score"] for r in on_1)
    assert by_id[_ID_B]["feedback_credit"]["valid"] == 0.5  # deprecated + contradicted
    assert by_id[_ID_A]["feedback_credit"]["valid"] == 1.0

    # non_redundant: the identical-text pair splits — survivor vs absorbed.
    nr = sorted(
        [
            by_id[_ID_A]["feedback_credit"]["non_redundant"],
            by_id[_ID_D]["feedback_credit"]["non_redundant"],
        ]
    )
    assert nr[0] <= 0.05 and nr[1] >= 0.9

    # retained: deprecated -> 0.0, active/durable -> 1.0.
    assert by_id[_ID_B]["feedback_credit"]["retained"] == 0.0
    assert by_id[_ID_A]["feedback_credit"]["retained"] == 1.0

    # 3. Determinism: repeated flag-on runs byte-identical.
    on_2, on_3 = recall(ws, _QUERY, limit=10), recall(ws, _QUERY, limit=10)

    def _dump(rs: list[dict]) -> str:
        return json.dumps([(r["_id"], r["score"], r["feedback_credit"]) for r in rs], sort_keys=True)

    assert _dump(on_1) == _dump(on_2) == _dump(on_3)
