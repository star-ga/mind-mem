# Copyright 2026 STARGA, Inc.
"""Computed abstention on the MCP recall door (and the doors that share it).

``_recall_impl`` is the choke point ``recall``, ``hybrid_search``, the REST
``/v1/recall`` route and the gRPC handler all lean on, so the field is wired
there exactly once, POST-cache, on the rail the attestation and explain blocks
already use: the recall cache key knows nothing about an intended use, so a
verdict baked into the cached payload would be replayed for the wrong one.

The ranker is stubbed out for most of these tests on purpose. They pin the
WIRING -- what is returned, what is recorded, what is left untouched when the
feature is off -- not retrieval quality; that is measured against the real
engine in ``tests/test_abstention_separation.py``.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from tests._abstention_fixture import fake_engine as _fake_engine
from tests._abstention_fixture import log_events, make_workspace

#: Top-level keys of a recall envelope as it shipped BEFORE this feature. The
#: flag-off response must be exactly this set: a baseline pinned to a property,
#: not to a moving ref.
PRE_FEATURE_KEYS = frozenset({"_schema_version", "backend", "query", "query_id", "count", "scoring_instant", "results", "attestation"})


@pytest.fixture()
def door(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The MCP recall module pointed at a fresh indexed fixture workspace."""
    import mind_mem.mcp.tools.recall as mcp_recall
    from mind_mem import recall_cache

    ws = make_workspace(str(tmp_path))
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    recall_cache.reset_singleton()
    mcp_recall.WS = ws  # type: ignore[attr-defined]
    try:
        yield mcp_recall
    finally:
        recall_cache.reset_singleton()


def _stub(door: Any, monkeypatch: pytest.MonkeyPatch, scores: list[float], **kw: Any) -> None:
    monkeypatch.setattr(door, "_recall_impl_uncached", _fake_engine(scores, **kw))


def _call(door: Any, query: str = "frost telemetry", **kw: Any) -> dict[str, Any]:
    raw = door._recall_impl(query, limit=10, backend="bm25", **kw)
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), raw
    return parsed


def _tables(ws: str) -> set[str]:
    import os

    db = os.path.join(ws, ".mind-mem-index", "recall.db")
    conn = sqlite3.connect(db)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def _prediction_rows(ws: str) -> list[tuple[Any, ...]]:
    import os

    conn = sqlite3.connect(os.path.join(ws, ".mind-mem-index", "recall.db"))
    try:
        return conn.execute(
            "SELECT run_id, intended_use, band, separation, candidates, basis, recorded_at "
            "FROM recall_abstention ORDER BY run_id, intended_use"
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Flag OFF: invisible
# ---------------------------------------------------------------------------


def test_default_recall_has_no_abstention_and_creates_no_table(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [30.0, 2.0, 1.0])
    env = _call(door)
    assert set(env) == PRE_FEATURE_KEYS, sorted(set(env) ^ PRE_FEATURE_KEYS)
    assert "abstention" not in env
    tables = _tables(door.WS)
    # Positive control: the database exists and holds real tables, so the
    # absence below is a statement about the feature and not about a missing DB.
    assert tables, "recall.db has no tables at all -- the absence check would be vacuous"
    assert "recall_abstention" not in tables


def test_explicitly_disabled_config_is_the_same_as_absent(door: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    ws = make_workspace(str(tmp_path / "off"), config={"abstention": {"enabled": False}})
    monkeypatch.setattr(door, "_workspace", lambda: ws)
    _stub(door, monkeypatch, [30.0, 2.0])
    env = _call(door)
    assert set(env) == PRE_FEATURE_KEYS
    assert "recall_abstention" not in _tables(ws)


def test_off_path_adds_no_log_line(door: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Wiring-discipline rule 2: with the feature off the wired build must not
    emit an event the unwired build did not."""
    _stub(door, monkeypatch, [30.0, 2.0])
    _call(door)  # warm the lazily created loggers
    with log_events() as off_events:
        _call(door, "second query shape")
    # Positive control 1: the recording works -- recall logs SOMETHING here.
    assert off_events, "recorded nothing; the absence check below would be vacuous"
    assert not [e for e in off_events if "abstention" in e.lower()], off_events

    # Positive control 2: the same recorder DOES see an abstention event when
    # the feature is on and has something to say (a rejected policy).
    bad = make_workspace(str(tmp_path / "bad"), config={"abstention": {"enabled": True, "bands": {"answer": {"act": 7}}}})
    monkeypatch.setattr(door, "_workspace", lambda: bad)
    with log_events() as on_events:
        _call(door, "third query shape")
    assert [e for e in on_events if "abstention" in e.lower()], on_events


def test_the_config_section_is_a_known_recall_key(tmp_path: Any) -> None:
    """An operator who opts in must not trade it for an ``unknown_recall_config_keys``
    warning on every recall (positive control: a genuinely unknown key DOES warn)."""
    from mind_mem._recall_core import _load_backend

    known = make_workspace(str(tmp_path / "known"), config={"abstention": {"enabled": True}})
    unknown = make_workspace(str(tmp_path / "unknown"), config={"abstention_typo": {"enabled": True}})
    _load_backend(known)  # warm
    with log_events() as events:
        _load_backend(known)
    assert not [e for e in events if "unknown_recall_config_keys" in e], events
    with log_events() as events:
        _load_backend(unknown)
    assert [e for e in events if "unknown_recall_config_keys" in e], "the control did not fire; the check above is vacuous"


# ---------------------------------------------------------------------------
# Flag ON: the first-class field
# ---------------------------------------------------------------------------


def test_intended_use_adds_the_first_class_field(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])  # separation 0.6
    summary = _call(door, intended_use="summary")["abstention"]
    answer = _call(door, intended_use="answer")["abstention"]
    write = _call(door, intended_use="governed_write")["abstention"]
    assert summary["band"] == "act" and summary["abstain"] is False
    assert answer["band"] == "act"
    assert write["band"] == "proceed_with_caution" and write["abstain"] is False
    assert summary["separation"] == answer["separation"] == write["separation"] == 0.6
    assert (summary["candidates"], summary["basis"]) == (3, "sqlite")
    assert [summary["intended_use"], answer["intended_use"], write["intended_use"]] == ["summary", "answer", "governed_write"]


def test_field_distinguishes_five_weak_matches_from_one_clear_match(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [40.0, 5.0, 4.0, 3.0, 2.0])
    clear = _call(door, "clear one", intended_use="answer")
    _stub(door, monkeypatch, [8.1, 8.0, 7.9, 7.8, 7.7])
    weak = _call(door, "weak five", intended_use="answer")
    assert clear["count"] == weak["count"] == 5
    assert clear["abstention"]["abstain"] is False and clear["abstention"]["band"] == "act"
    assert weak["abstention"]["abstain"] is True and weak["abstention"]["band"] == "do_not_act"


def test_no_results_is_do_not_act(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [])
    env = _call(door, intended_use="summary")
    assert env["results"] == []
    assert env["abstention"]["band"] == "do_not_act"
    assert env["abstention"]["abstain"] is True
    assert env["abstention"]["reason"] == "no_candidates"


def test_the_field_does_not_change_what_was_served(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """It describes the set; it never filters, reorders or rescores it."""
    _stub(door, monkeypatch, [12.0, 11.0, 3.0])
    plain = _call(door, "same query")
    judged = _call(door, "same query", intended_use="governed_write")
    assert plain["results"] == judged["results"]
    assert plain["attestation"]["results_digest"] == judged["attestation"]["results_digest"]
    assert plain["attestation"]["query_id"] == judged["attestation"]["query_id"]


def test_config_enabled_applies_the_configured_default_use(door: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    ws = make_workspace(str(tmp_path / "cfg"), config={"abstention": {"enabled": True, "default_use": "governed_write"}})
    monkeypatch.setattr(door, "_workspace", lambda: ws)
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    env = _call(door)  # no per-call use
    assert env["abstention"]["intended_use"] == "governed_write"
    assert env["abstention"]["band"] == "proceed_with_caution"
    # A per-call use overrides the configured default.
    assert _call(door, "other", intended_use="summary")["abstention"]["intended_use"] == "summary"


def test_configured_thresholds_are_applied_and_bad_ones_surface_as_config_warnings(
    door: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    strict = make_workspace(str(tmp_path / "strict"), config={"abstention": {"enabled": True, "bands": {"answer": {"act": 0.65}}}})
    monkeypatch.setattr(door, "_workspace", lambda: strict)
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])  # 0.6: act by default, caution under act>=0.65
    assert _call(door, intended_use="answer")["abstention"]["band"] == "proceed_with_caution"

    bad = make_workspace(
        str(tmp_path / "bad"), config={"abstention": {"enabled": True, "bands": {"governed_write": {"act": 0.01, "caution": 0.0}}}}
    )
    monkeypatch.setattr(door, "_workspace", lambda: bad)
    env = _call(door, intended_use="governed_write")
    assert env["abstention"]["thresholds"]["act"] == 0.70  # defaults, not the loosened pair
    assert any("abstention" in str(w) for w in env["config_warnings"])


def test_invalid_intended_use_is_refused_before_any_work(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def spy(*_a: Any, **_k: Any) -> str:
        calls.append("ran")
        return _fake_engine([5.0, 1.0])(*_a, **_k)

    monkeypatch.setattr(door, "_recall_impl_uncached", spy)
    refused = _call(door, intended_use="delete_everything")
    assert "error" in refused and "intended_use" in refused["error"]
    assert "governed_write" in refused["error"]
    assert calls == []
    # Positive control: a valid use does reach the engine.
    _call(door, intended_use="summary")
    assert calls == ["ran"]


def test_empty_string_intended_use_means_not_given(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 1.0])
    env = _call(door, intended_use="")
    assert set(env) == PRE_FEATURE_KEYS


# ---------------------------------------------------------------------------
# The cache boundary
# ---------------------------------------------------------------------------


def test_verdict_is_derived_post_cache_and_never_stored_in_it(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache key knows no intended use, so a verdict stored in the cached
    payload would be replayed to a caller who asked for a different one.

    The recorder is stubbed here ONLY because a first-time prediction row is a
    write to ``recall.db``, which is part of the corpus stamp and would -- by
    design -- retire the cache entry this test is trying to hit. Recording is
    covered on its own below.
    """
    import mind_mem.abstention_store as store

    monkeypatch.setattr(store, "record_prediction", lambda *a, **k: False)
    calls: list[str] = []
    inner = _fake_engine([10.0, 4.0, 1.0])

    def counting(*a: Any, **k: Any) -> str:
        calls.append("miss")
        return inner(*a, **k)

    monkeypatch.setattr(door, "_recall_impl_uncached", counting)
    first = _call(door, "cached shape", intended_use="summary")
    second = _call(door, "cached shape", intended_use="governed_write")
    third = _call(door, "cached shape")
    assert calls == ["miss"], f"the second and third calls should have been cache hits, got {calls}"
    assert first["abstention"]["band"] == "act"
    assert second["abstention"]["band"] == "proceed_with_caution"
    assert second["abstention"]["intended_use"] == "governed_write"
    assert "abstention" not in third


# ---------------------------------------------------------------------------
# Shapes it does not apply to
# ---------------------------------------------------------------------------


def test_bundle_format_is_left_alone(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 1.0])
    env = _call(door, intended_use="summary", format="bundle")
    assert "facts" in env and "relations" in env  # still the bundle shape
    assert "abstention" not in env


def test_rrf_fused_hits_report_the_rrf_basis(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [2 / 61, 1 / 62, 1 / 63], backend="hybrid", extra={"fusion": "rrf"})
    env = _call(door, intended_use="summary")
    assert env["abstention"]["basis"] == "rrf"
    assert env["abstention"]["band"] == "act"
    assert _call(door, "q2", intended_use="answer")["abstention"]["band"] == "proceed_with_caution"


def test_guardrail_hits_do_not_vouch_for_the_answer(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    inner = _fake_engine([8.1, 8.0, 7.9])

    def with_guardrail(*a: Any, **k: Any) -> str:
        env = json.loads(inner(*a, **k))
        env["results"].insert(0, {"_id": "G-0001", "type": "guardrail", "score": 999.0, "guardrail": True})
        env["count"] = len(env["results"])
        return json.dumps(env)

    monkeypatch.setattr(door, "_recall_impl_uncached", with_guardrail)
    env = _call(door, intended_use="answer")
    assert env["abstention"]["band"] == "do_not_act"
    assert env["abstention"]["candidates"] == 3


# ---------------------------------------------------------------------------
# Entry points beyond _recall_impl
# ---------------------------------------------------------------------------


def test_recall_and_hybrid_search_tools_accept_intended_use(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    via_recall = json.loads(door.recall("tool path one", backend="bm25", intended_use="governed_write"))
    via_hybrid = json.loads(door.hybrid_search("tool path two", limit=5, intended_use="summary"))
    assert via_recall["abstention"]["intended_use"] == "governed_write"
    assert via_hybrid["abstention"]["intended_use"] == "summary"
    plain = json.loads(door.recall("tool path three", backend="bm25"))
    assert "abstention" not in plain


def test_recall_tool_rejects_a_bad_use_with_an_error_envelope(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0])
    refused = json.loads(door.recall("x", intended_use="yolo"))
    assert "error" in refused and "intended_use" in refused["error"]


def _wire_tools() -> dict[str, Any]:
    """The tools exactly as a client sees them: the registered FastMCP callables."""
    import asyncio

    from mind_mem.mcp.server import mcp

    return {tool.name: tool.fn for tool in asyncio.run(mcp.list_tools())}


def test_the_registered_wire_recall_tool_takes_intended_use(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``recall`` has ONE wire owner (``public.recall``); wiring the Python
    implementation alone would leave the field unreachable to every MCP client.
    Rule 5 of the wiring discipline: check the registered surface, not the
    function you edited."""
    import inspect

    from mind_mem.mcp.infra.workspace import use_workspace

    wire = _wire_tools()["recall"]
    assert "intended_use" in inspect.signature(wire).parameters
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    with use_workspace(door.WS):
        asked = json.loads(wire(query="through the wire", mode="bm25", intended_use="governed_write"))
        plain = json.loads(wire(query="through the wire two", mode="bm25"))
        refused = json.loads(wire(query="x", mode="bm25", intended_use="yolo"))
        wrong_mode = json.loads(wire(query="x", mode="classify", intended_use="summary"))
    assert asked["abstention"]["intended_use"] == "governed_write"
    assert asked["abstention"]["band"] == "proceed_with_caution"
    assert "abstention" not in plain and set(plain) == PRE_FEATURE_KEYS
    assert "intended_use" in refused["error"]
    # Refused, not accepted-and-dropped, for a mode that does not rank.
    assert "intended_use" in wrong_mode["error"] and "classify" in wrong_mode["error"]


def test_a_user_scope_verdict_writes_only_its_own_prediction(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``recall`` is a USER-scope tool, and user tools may write nothing but read
    accounting. The prediction is a record that a read with a stated use was
    served -- the same class as the served ledger and the retrieval log -- and
    this measures by execution that it is the ONLY thing the verdict adds."""
    from mind_mem.mcp.infra.workspace import use_workspace
    from tests.test_acl_user_scope_read_only import READ_ACCOUNTING, _diff, _snapshot_files

    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    wire = _wire_tools()["recall"]

    def run(**kw: Any) -> set[str]:
        before = _snapshot_files(door.WS)
        with use_workspace(door.WS):
            payload = json.loads(wire(query=f"scope probe {kw}", mode="bm25", **kw))
        assert "error" not in payload, payload
        return _diff(before, _snapshot_files(door.WS))

    plain_changed = run()
    asked_changed = run(intended_use="governed_write")
    table = ".mind-mem-index/recall.db::recall_abstention"
    assert table not in plain_changed  # no verdict asked, nothing recorded
    assert table in asked_changed  # positive control: the snapshot CAN see the write
    assert not (asked_changed - READ_ACCOUNTING), sorted(asked_changed - READ_ACCOUNTING)


# ---------------------------------------------------------------------------
# The prediction record (the half the calibration loop scores)
# ---------------------------------------------------------------------------


def test_requested_recall_records_one_prediction_keyed_by_run_id(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    env = _call(door, "predict me", intended_use="governed_write")
    run_id = env["attestation"]["query_id"]
    assert len(run_id) == 64
    rows = _prediction_rows(door.WS)
    assert len(rows) == 1
    got_run, use, band, separation, candidates, basis, recorded_at = rows[0]
    assert got_run == run_id
    assert (use, band, separation, candidates, basis) == ("governed_write", "proceed_with_caution", 0.6, 3, "sqlite")
    # Provenance is the injected scoring instant -- no clock read at record time.
    assert recorded_at == env["scoring_instant"]


def test_prediction_is_idempotent_and_keyed_per_use(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    for _ in range(3):
        _call(door, "same run", intended_use="answer")
    assert len(_prediction_rows(door.WS)) == 1
    _call(door, "same run", intended_use="summary")
    rows = _prediction_rows(door.WS)
    assert sorted(r[1] for r in rows) == ["answer", "summary"]
    assert len({r[0] for r in rows}) == 1  # one run, two uses


def test_replayed_prediction_does_not_touch_the_database_file(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``recall.db`` is part of the recall cache's corpus stamp, so a recorder
    that rewrote an existing row on every replay would retire every cache entry
    in the workspace for nothing. A (run, use) that is already recorded is
    detected by a READ and left alone."""
    import os
    import time

    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    _call(door, "replay me", intended_use="answer")
    db = os.path.join(door.WS, ".mind-mem-index")
    watched = [os.path.join(db, name) for name in ("recall.db", "recall.db-wal") if os.path.exists(os.path.join(db, name))]
    assert watched, "recall.db missing -- the mtime comparison would be vacuous"
    before = {path: (os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in watched}
    time.sleep(0.05)
    _call(door, "replay me", intended_use="answer")
    after = {path: (os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in watched}
    assert after == before
    # Positive control: a genuinely new (run, use) DOES write.
    _call(door, "replay me", intended_use="summary")
    changed = {path: (os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in watched}
    assert changed != before


def test_prediction_write_failure_never_breaks_recall_and_is_visible(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(door, monkeypatch, [10.0, 4.0, 1.0])
    import mind_mem.abstention_store as store

    def boom(*_a: Any, **_k: Any) -> bool:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "record_prediction", boom)
    env = _call(door, "unlucky", intended_use="answer")
    assert env["abstention"]["band"] == "act"  # the answer is still served
    assert any("prediction" in str(w).lower() for w in env["warnings"])


def test_no_prediction_when_the_run_has_no_published_identity(door: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An envelope with no attestation run id (derivation failed) has nothing
    to join an outcome to -- record nothing rather than a row keyed on junk."""
    _stub(door, monkeypatch, [10.0, 4.0])
    good = _call(door, "has identity", intended_use="answer")
    assert len(good["attestation"]["query_id"]) == 64
    assert len(_prediction_rows(door.WS)) == 1  # positive control: this path records

    def failing(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("attestation unavailable")

    monkeypatch.setattr("mind_mem.recall_attestation.derive_recall_attestation_for_workspace", failing)
    env = _call(door, "no identity", intended_use="answer")
    assert env["abstention"]["band"] == "act"  # the verdict is still served
    assert "query_id" not in env["attestation"]
    assert len(_prediction_rows(door.WS)) == 1  # ...but nothing new was recorded
