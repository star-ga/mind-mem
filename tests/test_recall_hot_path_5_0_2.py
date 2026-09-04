# Copyright 2026 STARGA, Inc.
"""Contract tests for the 5.0.2 recall hot-path work.

Every assertion here is paired with a mutation that turns it red, because a
performance fix whose guard cannot fail is a performance fix nobody can
check. The four properties pinned:

1. The build path still hashes. The query path stopped hashing, and the ONLY
   thing that makes that safe is that ``build_index``'s change detection did
   not — so the split is tested from both sides, including the exact edit
   (same size, same ``mtime_ns``) that only the hash can see.
2. Batched weights equal per-candidate weights. The batch forms replaced
   per-candidate lookups on two ranking legs; if they ever disagreed the
   swap would be a ranking change wearing a latency change's clothes.
3. The 50k truncation FIRES an in-band ``degraded`` marker. Not "the marker
   exists" — that a truncated result cannot come back looking complete.
4. ``auto_build_index`` is inert when off and builds when on.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timezone

import pytest

from mind_mem import sqlite_index
from mind_mem.block_metadata import BlockMetadataManager
from mind_mem.calibration import CalibrationManager
from mind_mem.init_workspace import init

_INSTANT = date(2026, 9, 3)


def _write_blocks(ws: str, n: int, *, start: int = 1) -> None:
    dec = os.path.join(ws, "decisions")
    os.makedirs(dec, exist_ok=True)
    body = ["# DECISIONS\n\n---\n"]
    for i in range(start, start + n):
        body.append(
            f"\n[D-20260101-{i:06d}]\n"
            f"Date: 2026-06-20\n"
            f"Status: active\n"
            f"Scope: global\n"
            f"Statement: deterministic compiler evidence chain recall index block {i}\n"
            f"Tags: compiler, recall\n"
        )
    with open(os.path.join(dec, "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write("".join(body))


@pytest.fixture
def workspace(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)
    _write_blocks(ws, 12)
    return ws


# ---------------------------------------------------------------------------
# 1. The hash moved off the query path and stayed on the build path
# ---------------------------------------------------------------------------


def _rewrite_preserving_metadata(path: str, old: str, new: str) -> None:
    """In-place edit that keeps byte size AND ``st_mtime_ns`` identical.

    This is the one class of change size+mtime cannot see, so it is the only
    fixture that can tell the two ``verify_hash`` modes apart.
    """
    assert len(old) == len(new), "the fixture must not change the file size"
    stat = os.stat(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert old in text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = os.stat(path)
    # Positive control on the FIXTURE: if this ever stops holding, the test
    # below would pass for the wrong reason (the cheap check would catch it).
    assert after.st_size == stat.st_size
    assert after.st_mtime_ns == stat.st_mtime_ns


def test_query_path_skips_the_hash_and_the_build_path_does_not(workspace: str) -> None:
    sqlite_index.build_index(workspace, incremental=False)
    assert sqlite_index.is_stale(workspace) is False

    path = os.path.join(workspace, "decisions", "DECISIONS.md")
    _rewrite_preserving_metadata(path, "recall index block 1\n", "recall index BLOCK 1\n")

    # The cheap (default, query-side) verdict cannot see this edit...
    assert sqlite_index.is_stale(workspace) is False
    # ...and the expensive one, which is what the build path runs, can.
    assert sqlite_index.is_stale(workspace, verify_hash=True) is True

    conn = sqlite_index._connect(workspace, readonly=True)
    try:
        assert sqlite_index._get_changed_files(conn, workspace, verify_hash=False) == []
        assert sqlite_index._get_changed_files(conn, workspace, verify_hash=True) != []
    finally:
        conn.close()


def test_build_index_still_reindexes_a_metadata_stable_edit(workspace: str) -> None:
    """The correctness half. Skipping the hash on queries is only safe while
    an incremental BUILD still hashes — so prove the build picks the edit up."""
    sqlite_index.build_index(workspace, incremental=False)
    path = os.path.join(workspace, "decisions", "DECISIONS.md")
    _rewrite_preserving_metadata(path, "recall index block 2\n", "recall index PLUTO 2\n")

    summary = sqlite_index.build_index(workspace, incremental=True)
    assert summary["files_indexed"] >= 1, "an incremental build must still hash and re-index this file"

    hits = sqlite_index.query_index(workspace, "PLUTO", limit=5, scoring_instant=_INSTANT)
    assert any("PLUTO" in (h.get("excerpt") or "") for h in hits), "the rewritten content never reached the index"


# ---------------------------------------------------------------------------
# 2. Batched lookups equal per-candidate lookups
# ---------------------------------------------------------------------------


def test_batch_calibration_weights_equal_per_candidate(workspace: str) -> None:
    from mind_mem._recall_core import batch_calibration_weights

    cal = CalibrationManager(workspace)
    ids = [f"D-20260101-{i:06d}" for i in range(1, 13)]
    now = datetime(2026, 9, 3, tzinfo=timezone.utc)
    # MIN_FEEDBACK_THRESHOLD is 3, so one round of feedback leaves every
    # weight at 1.0 and the equality below would compare two dicts of nothing.
    for round_ in range(4):
        cal.record_feedback(
            query_id=f"q{round_}",
            block_ids_useful=ids[:4],
            block_ids_not_useful=ids[4:8],
            feedback_type="accepted",
        )

    per_candidate = {bid: cal.get_block_weight(bid, now=now) for bid in ids}
    batched = batch_calibration_weights(cal, ids, now=now)
    assert batched == per_candidate
    # Positive control: the feedback above must actually MOVE a weight, or the
    # comparison is two dicts of 1.0 agreeing about nothing.
    assert any(w != 1.0 for w in per_candidate.values()), "no weight moved; the equality proves nothing"


def test_batch_calibration_weights_chunks_past_the_sqlite_variable_limit(workspace: str) -> None:
    """A single ``IN (...)`` of 50k ids raises; the chunked helper must not."""
    from mind_mem._recall_core import _WEIGHT_BATCH_SIZE, batch_calibration_weights

    cal = CalibrationManager(workspace)
    ids = [f"D-20260101-{i:06d}" for i in range(1, 5 * _WEIGHT_BATCH_SIZE + 7)]
    out = batch_calibration_weights(cal, ids, now=datetime(2026, 9, 3, tzinfo=timezone.utc))
    assert len(out) == len(ids)
    assert set(out) == set(ids)


def test_batch_importance_boosts_equal_per_candidate(workspace: str) -> None:
    meta_db = os.path.join(workspace, ".mind-mem", "block_meta.db")
    os.makedirs(os.path.dirname(meta_db), mode=0o700, exist_ok=True)
    mgr = BlockMetadataManager(meta_db)
    ids = [f"D-20260101-{i:06d}" for i in range(1, 13)]
    mgr.record_access(ids[:6], query="compiler", now=datetime(2026, 9, 3, tzinfo=timezone.utc))
    for bid in ids[:6]:
        mgr.update_importance(bid)

    per_candidate = {bid: mgr.get_importance_boost(bid) for bid in ids}
    batched = mgr.get_importance_boosts(ids)
    assert {bid: batched.get(bid, 1.0) for bid in ids} == per_candidate
    assert any(v != 1.0 for v in per_candidate.values()), "no boost moved; the equality proves nothing"


def test_batch_importance_boosts_chunks(workspace: str) -> None:
    from mind_mem.block_metadata import _IMPORTANCE_BATCH_SIZE

    meta_db = os.path.join(workspace, ".mind-mem", "block_meta.db")
    os.makedirs(os.path.dirname(meta_db), mode=0o700, exist_ok=True)
    mgr = BlockMetadataManager(meta_db)
    ids = [f"D-20260101-{i:06d}" for i in range(1, 3 * _IMPORTANCE_BATCH_SIZE + 5)]
    mgr.record_access(ids[:3], query="q", now=datetime(2026, 9, 3, tzinfo=timezone.utc))
    for bid in ids[:3]:
        mgr.update_importance(bid)
    out = mgr.get_importance_boosts(ids)
    assert set(out).issubset(set(ids))
    assert set(out) == {bid for bid in ids[:3] if mgr.get_importance_boost(bid) != 1.0} or len(out) >= 1


# ---------------------------------------------------------------------------
# 3. The corpus cap raises an in-band marker
# ---------------------------------------------------------------------------


def test_corpus_truncation_fires_an_in_band_degraded_marker(workspace: str, monkeypatch) -> None:
    """The marker must FIRE, not merely exist: a truncated answer has to be
    distinguishable from a complete one at the surface that serves it."""
    from mind_mem import _recall_core
    from mind_mem.recall import recall

    monkeypatch.setattr(_recall_core, "MAX_BLOCKS_PER_QUERY", 3)
    hits = recall(workspace, "compiler evidence chain", limit=10, scoring_instant=_INSTANT)

    marker = getattr(hits, "degraded", None)
    assert marker is not None, "a truncated corpus came back looking complete"
    assert marker["leg"] == "bm25"
    assert marker["reason"] == "corpus_truncated"
    assert int(marker["blocks_scored"]) == 3
    assert int(marker["blocks_total"]) > 3


def test_untruncated_recall_carries_no_marker_and_stays_a_plain_list(workspace: str) -> None:
    """The OFF path pays nothing: the engine returns the same plain ``list``
    it always returned, and no marker is fabricated."""
    from mind_mem._recall_core import recall as engine_recall

    hits = engine_recall(workspace, "compiler evidence chain", limit=10, scoring_instant=_INSTANT)
    assert hits, "fixture produced no hits; the negative assertion below would be vacuous"
    assert getattr(hits, "degraded", None) is None
    assert type(hits) is list, "the untruncated path must not pay for the marker"


def test_truncation_marker_survives_the_hybrid_bm25_arm(workspace: str, monkeypatch) -> None:
    """The early ``if results: return _as_results(results, None)`` in
    ``_bm25_search`` used to drop any marker the raw leg raised — which is
    exactly the non-empty degraded case this marker describes."""
    from mind_mem import _recall_core
    from mind_mem.hybrid_recall import HybridBackend

    monkeypatch.setattr(_recall_core, "MAX_BLOCKS_PER_QUERY", 3)
    hb = HybridBackend(config={})
    results = hb.search("compiler evidence chain", workspace, limit=5, scoring_instant=_INSTANT)
    assert results, "no hits; the marker assertion below would be vacuous"
    marker = getattr(results, "degraded", None)
    assert marker is not None and "corpus_truncated" in str(marker.get("reason"))


# ---------------------------------------------------------------------------
# 4. auto_build_index: inert when off, builds when on
# ---------------------------------------------------------------------------


def test_auto_build_index_is_inert_when_off(workspace: str) -> None:
    """Presence is the FTS schema, not the file: ``CalibrationManager`` creates
    ``recall.db`` on the recall path with a calibration table and nothing else,
    which is exactly why a default install never noticed it had no index."""
    assert sqlite_index._index_present(workspace) is False, "fixture already indexed; the assertions below would be vacuous"
    assert sqlite_index.ensure_index(workspace, enabled=False) is None
    assert sqlite_index._index_present(workspace) is False, "the off path built an index"

    hb = HybridBackendFactory(config={})
    assert hb._auto_build_index is False
    hb.search("compiler evidence chain", workspace, limit=5, scoring_instant=_INSTANT)
    assert sqlite_index._index_present(workspace) is False, "a default-config recall built an index"
    # The file DID appear (calibration owns it) — proving the file probe this
    # function used to rely on would have reported a phantom index.
    assert os.path.isfile(sqlite_index._db_path(workspace))


def test_flag_off_never_reaches_the_probe(workspace: str, monkeypatch) -> None:
    """Inertness as a contract, not an argument.

    Measured separately against a build with the feature stripped out
    entirely: identical ``stat`` / ``open`` / ``sqlite3.connect`` counts over
    10 flag-off searches (1730 / 40 / 80 either way). This pins the reason —
    the flag is tested BEFORE anything that could stat, connect or parse.
    """
    calls: list[tuple] = []

    def spy(ws, *, enabled):
        calls.append((ws, enabled))
        return None

    monkeypatch.setattr(sqlite_index, "ensure_index", spy)
    hb = HybridBackendFactory(config={})
    hb.search("compiler evidence chain", workspace, limit=5, scoring_instant=_INSTANT)
    assert calls == [], "the off path reached the probe"

    # Positive control: the spy IS installed on the path under test, so the
    # empty list above is evidence and not an artefact of patching the wrong
    # symbol. With the flag on, the very same spy records a call.
    hb_on = HybridBackendFactory(config={"auto_build_index": True})
    hb_on.search("compiler evidence chain", workspace, limit=5, scoring_instant=_INSTANT)
    assert calls and calls[0][1] is True, "the spy was never on the path; the assertion above proved nothing"


def test_auto_build_index_builds_once_when_on(workspace: str) -> None:
    sqlite_index._AUTO_BUILD_ATTEMPTED.discard(os.path.abspath(workspace))
    assert sqlite_index._index_present(workspace) is False

    summary = sqlite_index.ensure_index(workspace, enabled=True)
    assert summary is not None and summary["blocks_indexed"] == 12
    assert sqlite_index._index_present(workspace) is True
    # Second call is a no-op: "on first recall" means once, not every query.
    assert sqlite_index.ensure_index(workspace, enabled=True) is None


def test_auto_build_index_from_the_hybrid_arm(workspace: str) -> None:
    sqlite_index._AUTO_BUILD_ATTEMPTED.discard(os.path.abspath(workspace))
    assert sqlite_index._index_present(workspace) is False
    hb = HybridBackendFactory(config={"auto_build_index": True})
    hb.search("compiler evidence chain", workspace, limit=5, scoring_instant=_INSTANT)
    assert sqlite_index._index_present(workspace) is True, "auto_build_index=true did not build the index on first recall"


def HybridBackendFactory(*, config: dict):  # noqa: N802 — a fixture helper, not a class
    from mind_mem.hybrid_recall import HybridBackend

    return HybridBackend(config=config)


# ---------------------------------------------------------------------------
# 5. record_access finally receives the instant recall already resolved
# ---------------------------------------------------------------------------


def test_recall_creates_the_metadata_store_on_a_pre_5_0_2_workspace(workspace: str) -> None:
    """The reader-side guard that made the whole feature dead.

    ``.mind-mem`` joined ``init_workspace.DIRS`` in 5.0.2, so a workspace
    created by THIS version has the directory and the old
    ``if os.path.isdir(meta_dir)`` guard would pass. Every workspace
    ``mind-mem-init`` produced BEFORE 5.0.2 does not — and on those the guard
    pre-empted BlockMetadataManager's own create-on-first-use and
    ``meta_mgr`` was permanently None. Removing the directory here is what
    reproduces the shipped condition; without it this file could not tell a
    fixed reader from a broken one.
    """
    import shutil

    from mind_mem.recall import recall

    meta_dir = os.path.join(workspace, ".mind-mem")
    shutil.rmtree(meta_dir, ignore_errors=True)
    assert not os.path.isdir(meta_dir), "the pre-5.0.2 condition was not reproduced"

    hits = recall(workspace, "compiler evidence chain", limit=5, scoring_instant=_INSTANT)
    assert hits, "no hits; nothing would be recorded and the assertion would be vacuous"
    assert os.path.isfile(os.path.join(meta_dir, "block_meta.db")), "recall on a pre-5.0.2 workspace still records no telemetry"


def test_recall_stamps_last_accessed_from_the_resolved_instant(workspace: str) -> None:
    from mind_mem.recall import recall

    hits = recall(workspace, "compiler evidence chain", limit=5, scoring_instant=_INSTANT)
    assert hits, "no hits; nothing would be recorded and the assertion would be vacuous"

    meta_db = os.path.join(workspace, ".mind-mem", "block_meta.db")
    assert os.path.isfile(meta_db), "recall did not create the block-metadata store"
    conn = sqlite3.connect(meta_db)
    try:
        rows = conn.execute("SELECT id, access_count, last_accessed FROM block_meta").fetchall()
    finally:
        conn.close()
    assert rows, "recall recorded no access at all"
    stamped = [r for r in rows if r[2]]
    assert stamped, "last_accessed was never written — the instant is still not threaded through"
    assert all(r[2].startswith("2026-09-03") for r in stamped), "last_accessed did not come from the recall's own instant"
