"""An attestation must distinguish two runs that served different answers.

``RECALL_ATTEST_v1`` rail 3 promises determinism: "no wall-clock in the
preimage". That was true of the *record* and false of the *run* it attested.
The preimage described the result set by a single ``result_count``, so two runs
of the same corpus and query that ranked differently — because the hidden
wall-clock input moved — produced a byte-identical ``attestation_hash``. An
attestation that cannot tell two different served orders apart asserts a
reproducibility it cannot deliver, which is worse than no attestation.

Binding the resolved ``scoring_instant`` closes it: the instant is a run
*input*, in the same class as ``config_hash`` and ``index_anchor``, so it is
recorded and replayable without turning the record into a stored verdict
(rail 2 — the attestation is still never persisted).

Acceptance criteria covered: T4 attestation completeness, T5 replay.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import pytest

from mind_mem.recall_attestation import RecallAttestation, build_recall_attestation

GENESIS_ANCHOR = "0" * 64

NEAR_INSTANT = date(2026, 8, 27)
FAR_INSTANT = date(2027, 8, 27)
QUERY = "retrieval ranking determinism"

# One block wins on BM25, the other on recency. At NEAR_INSTANT the recency
# multiplier lifts the newer block above the stronger match; at FAR_INSTANT both
# have decayed to the 0.1 floor and raw BM25 decides. Same corpus, same config,
# two different served orders.
_BLOCKS = (
    ("D-20230101-001", "retrieval ranking retrieval ranking determinism", "2023-01-01"),
    ("D-20260801-002", "retrieval ranking determinism", "2026-08-01"),
)


def _write_workspace(root: str) -> None:
    os.makedirs(os.path.join(root, "decisions"), exist_ok=True)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        for bid, statement, when in _BLOCKS:
            fh.write(f"[{bid}]\nStatement: {statement}\nStatus: active\nDate: {when}\n\n")


@pytest.fixture
def envelope_for(tmp_path, monkeypatch):
    """Return ``run(scoring_instant) -> envelope`` against one pinned workspace."""
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = str(tmp_path)
    _write_workspace(ws)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": False}})

    def _run(scoring_instant: date | str | None) -> dict:
        raw = mcp_recall._recall_impl(QUERY, limit=5, backend="bm25", scoring_instant=scoring_instant)
        return json.loads(raw)

    # Warm-up: the first call materialises the FTS index, so discard it and
    # compare only runs that took an identical code path.
    _run(NEAR_INSTANT)
    return _run


def _order(envelope: dict) -> list[str]:
    return [str(r.get("_id")) for r in envelope.get("results", [])]


# ---------------------------------------------------------------------------
# T4 — the headline
# ---------------------------------------------------------------------------


def test_different_served_orders_produce_different_attestation_hashes(envelope_for) -> None:
    """T4: the defect. Two rankings, one hash — no longer."""
    near = envelope_for(NEAR_INSTANT)
    far = envelope_for(FAR_INSTANT)

    assert _order(near) != _order(far), "fixture did not reorder — the test would be vacuous"
    assert near["attestation"]["attestation_hash"] != far["attestation"]["attestation_hash"], (
        "two runs that served different orders collided on one attestation hash"
    )


def test_different_orders_at_one_instant_produce_different_hashes(tmp_path, monkeypatch) -> None:
    """T4 as *worded*, not just the instance the seam was built for.

    The criterion is "two runs that produce different served orders must
    produce different attestation hashes" — for any cause, not only for a
    moved instant. Before ``results_digest`` was bound, the preimage described
    the answer by its **cardinality**, so at one fixed instant these three runs
    — same query, same config, same count, one reordered corpus and one wholly
    disjoint one — all collided on a single hash.
    """
    import mind_mem.mcp.tools.recall as mcp_recall

    # Swap which id carries which text/date, so the same two ids come back in
    # the opposite order. Then a corpus sharing no id with either.
    reordered = ((_BLOCKS[1][0], _BLOCKS[0][1], _BLOCKS[0][2]), (_BLOCKS[0][0], _BLOCKS[1][1], _BLOCKS[1][2]))
    disjoint = (("D-20230101-777", _BLOCKS[0][1], _BLOCKS[0][2]), ("D-20260801-888", _BLOCKS[1][1], _BLOCKS[1][2]))

    def _run(blocks: tuple[tuple[str, str, str], ...]) -> dict:
        ws = str(tmp_path / f"ws{abs(hash(blocks))}")
        os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
            for bid, statement, when in blocks:
                fh.write(f"[{bid}]\nStatement: {statement}\nStatus: active\nDate: {when}\n\n")
        monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
        monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": False}})
        mcp_recall._recall_impl(QUERY, limit=5, backend="bm25", scoring_instant=NEAR_INSTANT)  # warm-up
        return json.loads(mcp_recall._recall_impl(QUERY, limit=5, backend="bm25", scoring_instant=NEAR_INSTANT))

    baseline, swapped, elsewhere = _run(_BLOCKS), _run(reordered), _run(disjoint)

    assert _order(baseline) != _order(swapped), "fixture did not reorder — the test would be vacuous"
    assert set(_order(baseline)).isdisjoint(_order(elsewhere)), "fixture is not disjoint"
    counts = {e["attestation"]["result_count"] for e in (baseline, swapped, elsewhere)}
    instants = {e["attestation"]["scoring_instant"] for e in (baseline, swapped, elsewhere)}
    assert len(counts) == 1 and len(instants) == 1, "the runs differ in count or instant, so this is the easy case"

    hashes = [e["attestation"]["attestation_hash"] for e in (baseline, swapped, elsewhere)]
    assert len(set(hashes)) == 3, f"two runs that served different answers collided: {hashes}"


def test_served_id_order_is_bound_not_just_the_set() -> None:
    """``(A, B)`` and ``(B, A)`` are two different answers, so two hashes.

    The leg tuples are deduped and sorted before hashing; the served ids must
    not be, because their order *is* the ranking under attestation.
    """
    common = dict(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=2,
        scoring_instant=NEAR_INSTANT,
    )
    forward = build_recall_attestation(**common, served_ids=("D-A", "D-B"))
    reverse = build_recall_attestation(**common, served_ids=("D-B", "D-A"))
    assert forward.attestation_hash != reverse.attestation_hash
    assert forward.results_digest != reverse.results_digest


def test_results_digest_is_hash_bound_and_tamper_evident() -> None:
    """Rewriting the recorded answer without rehashing is detectable."""
    import dataclasses

    att = build_recall_attestation(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=2,
        served_ids=("D-A", "D-B"),
        scoring_instant=NEAR_INSTANT,
    )
    assert att.is_internally_consistent()
    assert not dataclasses.replace(att, results_digest="0" * 64).is_internally_consistent()


def test_identical_inputs_produce_an_identical_hash(envelope_for) -> None:
    """T4 converse: the record stays a pure function of the run's inputs."""
    first = envelope_for(NEAR_INSTANT)
    second = envelope_for(NEAR_INSTANT)

    assert _order(first) == _order(second)
    assert first["attestation"]["attestation_hash"] == second["attestation"]["attestation_hash"]


def test_attestation_records_the_instant_the_run_actually_scored_with(envelope_for) -> None:
    envelope = envelope_for(FAR_INSTANT)
    assert envelope["attestation"]["scoring_instant"] == "2027-08-27"


def test_scoring_instant_is_hash_bound_and_tamper_evident() -> None:
    """Rewriting the recorded instant without rehashing is detectable."""
    import dataclasses

    att = build_recall_attestation(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=2,
        scoring_instant=NEAR_INSTANT,
    )
    assert att.is_internally_consistent()
    tampered = dataclasses.replace(att, scoring_instant="2027-08-27")
    assert not tampered.is_internally_consistent()


def test_two_instants_alone_change_the_hash() -> None:
    common = dict(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=2,
    )
    near = build_recall_attestation(**common, scoring_instant=NEAR_INSTANT)
    far = build_recall_attestation(**common, scoring_instant=FAR_INSTANT)
    assert near.attestation_hash != far.attestation_hash


def test_serialised_instant_is_a_bare_iso_date() -> None:
    """No time component, no offset suffix — 10 stable ASCII bytes."""
    att = build_recall_attestation(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=1,
        scoring_instant=datetime(2026, 8, 27, 23, 59, 59, tzinfo=timezone.utc),
    )
    assert att.scoring_instant == "2026-08-27"
    assert att.to_dict()["scoring_instant"] == "2026-08-27"


@pytest.mark.parametrize("dropped", ["scoring_instant", "results_digest"])
def test_pre_seam_record_is_rejected_not_silently_inconsistent(dropped: str) -> None:
    """A v1-shaped dict without either new field must fail loudly at the boundary.

    Reviving it with a guessed default would produce a record that simply
    reports ``is_internally_consistent() is False`` with no explanation.
    """
    att = build_recall_attestation(
        legs_ran=("bm25",),
        legs_degraded=(),
        config_hash="CFG",
        degraded=None,
        index_anchor=GENESIS_ANCHOR,
        result_count=1,
        served_ids=("D-A",),
        scoring_instant=NEAR_INSTANT,
    )
    legacy = att.to_dict()
    legacy.pop(dropped)
    with pytest.raises(ValueError, match=dropped):
        RecallAttestation.from_dict(legacy)


def test_live_consolidated_tool_exposes_the_seam(tmp_path, monkeypatch) -> None:
    """The MCP-visible ``recall`` is the consolidated dispatcher, not the base tool.

    ``public.recall`` deliberately shadows ``recall.recall``, so exposing the
    parameter only on the base tool would leave the actual client-facing surface
    unable to pin or replay an instant.
    """
    import mind_mem.mcp.tools.public as public
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = str(tmp_path)
    _write_workspace(ws)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": False}})

    run = public.recall.__wrapped__
    run(QUERY, backend="bm25")  # warm-up: materialise the index
    near = json.loads(run(QUERY, backend="bm25", scoring_instant="2026-08-27"))
    far = json.loads(run(QUERY, backend="bm25", scoring_instant="2027-08-27"))

    assert _order(near) != _order(far)
    assert near["attestation"]["attestation_hash"] != far["attestation"]["attestation_hash"]
    assert far["attestation"]["scoring_instant"] == "2027-08-27"


def test_live_tool_rejects_a_malformed_instant(tmp_path, monkeypatch) -> None:
    """Validated at the boundary, with a message that names the expected shape."""
    import mind_mem.mcp.tools.public as public
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = str(tmp_path)
    _write_workspace(ws)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": False}})

    # "2026-W01-1" is the one that matters: a valid ISO-8601 *week* date, ten
    # characters wide, which ``date.fromisoformat`` happily resolves to
    # 2025-12-29 — eight months from the date the client named.
    for bad in ("2026-8-7", "not-a-date", "2026-08-27T00:00:00", "2026-W01-1", " 2026-08-27 "):
        envelope = json.loads(public.recall.__wrapped__(QUERY, backend="bm25", scoring_instant=bad))
        assert "scoring_instant" in envelope.get("error", ""), f"{bad!r} was not rejected"


def test_cache_does_not_serve_one_instant_under_another(tmp_path, monkeypatch) -> None:
    """Two instants are two answers, so they must not share a cache entry."""
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = str(tmp_path)
    _write_workspace(ws)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(mcp_recall, "_load_config", lambda _ws: {"cache": {"enabled": True}})

    def _run(instant: str) -> dict:
        return json.loads(mcp_recall._recall_impl(QUERY, limit=5, backend="bm25", scoring_instant=instant))

    _run("2026-08-27")  # warm-up + populate the cache
    near = _run("2026-08-27")
    far = _run("2027-08-27")
    assert _order(near) != _order(far), "the far instant was served from the near instant's cache entry"


# ---------------------------------------------------------------------------
# T5 — replay
# ---------------------------------------------------------------------------


def test_attested_instant_replays_the_identical_served_order(envelope_for) -> None:
    """T5: read the instant back off the attestation, re-run, get the same answer."""
    original = envelope_for(FAR_INSTANT)
    attested = original["attestation"]["scoring_instant"]

    replay = envelope_for(date.fromisoformat(attested))

    assert _order(replay) == _order(original)
    assert [r["score"] for r in replay["results"]] == [r["score"] for r in original["results"]]
    assert replay["attestation"]["attestation_hash"] == original["attestation"]["attestation_hash"]


def test_replay_beats_a_different_instant(envelope_for) -> None:
    """The replay assertion above is only meaningful because the wrong instant fails."""
    original = envelope_for(FAR_INSTANT)
    wrong = envelope_for(NEAR_INSTANT)
    assert _order(wrong) != _order(original)
    assert wrong["attestation"]["attestation_hash"] != original["attestation"]["attestation_hash"]
