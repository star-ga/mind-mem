"""The anticipation cache never serves an answer a non-governed write changed.

The anticipation (Group J) bundle cache was keyed only on the governed-ledger
head plus the config/schema generation identity. An admin ``reindex`` after a
status change on disk, ``delete_memory_item`` and every other write that does
not touch the governed ledger left the head where it was, so a block
quarantined on disk was still answered from the bundle. The fix folds the same
stat-only corpus stamp the recall cache uses into the bundle generation key.

Every negative assertion is paired with a positive control proving the bundle
genuinely served the query when nothing changed.
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.recall_cache import reset_singleton

_Q = "Ilona call Tuesday"
_B1 = "D-20261006-001"
_B2 = "D-20261006-002"


@pytest.fixture(autouse=True)
def _fresh_singletons(monkeypatch):
    from mind_mem import prefetch

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    # Fresh tmp_path files sit inside the racy-timestamp window, which makes
    # the corpus state unnameable and bypasses every cache. A zero window lets
    # the bundle be recorded and consulted without ``os.utime`` (which would
    # itself bump every ctime and retire the generation for the wrong reason).
    # The racy test below restores the real window explicitly.
    monkeypatch.setattr("mind_mem.recall_cache._RACY_WINDOW_NS", 0)
    reset_singleton()
    prefetch.reset_cache()
    yield
    reset_singleton()
    prefetch.reset_cache()


def _blk(bid: str, stmt: str, status: str) -> str:
    return f"[{bid}]\nType: Decision\nStatement: {stmt}\nDate: 2026-10-06\nStatus: {status}\n\n---\n\n"


def _seed(tmp_path) -> str:
    from mind_mem.init_workspace import init
    from mind_mem.sqlite_index import build_index

    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write("\n" + _blk(_B1, "Ilona call is on Tuesday. Ilona call Tuesday at 10.", "active"))
        fh.write(_blk(_B2, "Ilona call Tuesday backup note.", "active"))
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    cfg["recall"] = recall_cfg
    cfg["cache"] = {
        "enabled": True,
        "ttl_seconds": 3600,
        "anticipation": {"enabled": True, "min_corpus_stems": 1, "novel_ratio_threshold": 1.0},
    }
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    build_index(ws)
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.recall import _recall_impl

    # One-time lazy table creation in recall.db; a live store is past it.
    with use_workspace(ws):
        _recall_impl("warm-up", limit=1, backend="bm25")
    reset_singleton()
    return ws


def _recall(ws: str) -> tuple[str, list[str]]:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.recall import _recall_impl

    with use_workspace(ws):
        payload = json.loads(_recall_impl(_Q, limit=5))
    return str(payload.get("backend")), [str(h.get("_id")) for h in payload.get("results", [])]


def _warm_bundle(ws: str) -> None:
    """Record a bundle, then prove the next identical-state recall is served from it."""
    first_backend, first_ids = _recall(ws)
    assert first_backend != "anticipation_cache"
    assert _B1 in first_ids
    backend, ids = _recall(ws)
    # Positive control: with nothing changed the bundle genuinely answers.
    assert backend == "anticipation_cache", (backend, ids)
    assert _B1 in ids


def _quarantine_on_disk(ws: str) -> None:
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index(f"[{_B1}]")
    at = text.index("Status: active", start)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text[:at] + "Status: quarantined" + text[at + len("Status: active") :])


def test_quarantine_then_admin_reindex_is_not_served_from_anticipation_bundle(tmp_path):
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import memory_ops

    ws = _seed(tmp_path)
    _warm_bundle(ws)

    _quarantine_on_disk(ws)
    with use_workspace(ws):
        memory_ops.reindex()

    backend, ids = _recall(ws)
    assert backend != "anticipation_cache", (backend, ids)
    assert _B1 not in ids, ids
    # Positive control: the surviving block is still recalled live.
    assert _B2 in ids, ids


def test_out_of_band_corpus_edit_retires_anticipation_bundle(tmp_path):
    ws = _seed(tmp_path)
    _warm_bundle(ws)

    _quarantine_on_disk(ws)

    backend, _ids = _recall(ws)
    assert backend != "anticipation_cache", backend


def test_unnameable_corpus_state_bypasses_a_warm_bundle(tmp_path, monkeypatch):
    ws = _seed(tmp_path)
    _warm_bundle(ws)

    # Nothing on disk differs, so the warm bundle WOULD answer (the control in
    # ``_warm_bundle`` just proved it). A stamp of ``None`` -- a file inside the
    # racy-timestamp window, or an unstattable input -- means the state cannot
    # be named, and the bundle cache must be bypassed both ways.
    monkeypatch.setattr("mind_mem.mcp.tools.recall._corpus_stamp", lambda _ws, _config=None: None)

    backend, ids = _recall(ws)
    assert backend != "anticipation_cache", backend
    assert _B1 in ids


def test_racy_window_makes_the_corpus_state_unnameable(tmp_path, monkeypatch):
    from mind_mem.recall_cache import corpus_stamp

    ws = _seed(tmp_path)
    assert corpus_stamp(ws) is not None
    monkeypatch.setattr("mind_mem.recall_cache._RACY_WINDOW_NS", 1_000_000_000)
    os.utime(os.path.join(ws, "decisions", "DECISIONS.md"), None)
    assert corpus_stamp(ws) is None


def test_delete_memory_item_retires_anticipation_bundle(tmp_path):
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import memory_ops

    ws = _seed(tmp_path)
    _warm_bundle(ws)

    with use_workspace(ws):
        memory_ops.delete_memory_item(_B1, "anticipation stamp test")

    backend, ids = _recall(ws)
    assert backend != "anticipation_cache", (backend, ids)
    assert _B1 not in ids, ids


def test_prefetch_bundle_is_retired_by_its_own_retrieval_log_write(tmp_path):
    """Pins a known trade, in the safe direction.

    The prefetch door files its bundle under the generation it took before its
    fan-out, and the fan-out's engine logs into ``recall.db`` (``retrieval_log``).
    A stat cannot tell that telemetry from a lineage/edge write in the same file,
    so the stamp moves and the next recall goes to the store. If write-on-read
    telemetry ever leaves ``recall.db`` this test flips, and the prefetch door
    serves locally again by construction.
    """
    from mind_mem import prefetch
    from mind_mem.mcp.infra.config import _load_config
    from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.recall import _anticipation_generation_key
    from mind_mem.mcp.tools.recall import prefetch as prefetch_tool

    ws = _seed(tmp_path)
    with use_workspace(ws):
        filled = json.loads(prefetch_tool(_Q, limit=5))
    assert filled.get("results"), filled
    assert prefetch.get_cache().stats()["bundles"] >= 1, "the prefetch door recorded nothing"

    backend, _ids = _recall(ws)
    assert backend != "anticipation_cache", backend

    # Positive control: the same hits filed at the CURRENT generation do serve,
    # so the miss above is the generation moving, not a gate rejection.
    prefetch.reset_cache()
    key = _anticipation_generation_key(ws, prefetch.chain_head(ws))
    identity = prefetch.anticipation_generation_identity(_load_config(ws), str(MCP_SCHEMA_VERSION))
    assert key is not None and identity is not None
    prefetch.get_cache().record(ws, "prefetch", filled["results"], head=key, generation_identity=identity)
    backend, ids = _recall(ws)
    assert backend == "anticipation_cache", (backend, ids)
