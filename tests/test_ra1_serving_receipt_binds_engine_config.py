"""The recorded receipt must bind the config the ENGINE consumed — and the binding must not
downgrade storage.

TWO DEFECTS, both found by an independent source-bound reproduction against a candidate that already
had the engine binding, and neither visible to the tests that existed.

FIRST — ENGINE A, RECEIPT B. Binding the ranking to a captured config fixed which policy RANKED. It
did not fix which policy was RECORDED. The ranked door captured the config mapping and then derived
the pipeline hash by re-reading the workspace, so the mapping and the hash could come from two
different moments. The reproduction served one real result with ``HybridBackend`` observed TWICE under
config A while the recorded row's ``pipeline_hash`` was hash_B: engine A, receipt B, and
``served_proof="recorded"`` over the mixture. A row that names a configuration the ranking never used
is the exact claim this whole mechanism exists to prevent, so helper-level consistency was not enough.

The fix makes the derivation read the captured mapping: ``pipeline_hash._load_workspace_config``
honours the bound context, and the door captures the hash and the index anchor INSIDE a pre-context
carrying the mapping. The hash is then the hash of that mapping by construction rather than by
ordering.

SECOND — A SILENT STORAGE DOWNGRADE I INTRODUCED. The context originally wrapped its config in
``MappingProxyType`` to stop the engine rebinding top-level sections. ``storage._backend_name`` — like
other consumers written against this repo's plain-dict convention — tests ``isinstance(config, dict)``
and DEGRADES TO THE MARKDOWN BACKEND when that fails. A ``mappingproxy`` is not a ``dict``. Observed
live as ``block_store_config_malformed: config must be an object, got mappingproxy; degrading to
markdown backend``. On a postgres or encrypted workspace that serves plausible answers from the wrong
corpus, which is far worse than the rebind the proxy was guarding against. The proxy is gone; the
negative control below keeps that decision honest by proving the check still catches a proxy.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from mind_mem import hybrid_recall, prefetch
from mind_mem.mcp.infra.config import _load_config
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.request_context import (
    RequestContext,
    bind_request_context,
    context_config_for,
)
from mind_mem.served_ledger import ServedRunV2, read_served_runs
from mind_mem.storage import _backend_name

_QUERY = "deterministic compiler"

_CONFIG_A: dict[str, Any] = {
    "cache": {"enabled": False},
    "extraction": {"backend": "unknown", "model": "capture-a"},
    "recall": {"query_expansion": {"enabled": False, "auto_enable": False}, "vector_enabled": False},
}


def _seed(root: Path) -> str:
    (root / "decisions").mkdir(parents=True)
    for name in ("tasks", "entities", "intelligence"):
        (root / name).mkdir()
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-CTX-001]\nStatement: deterministic compiler retrieval context\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "mind-mem.json").write_text(json.dumps(_CONFIG_A), encoding="utf-8", newline="\n")
    return str(root)


@pytest.fixture(autouse=True)
def _reset_anticipation_cache() -> Any:
    prefetch.reset_cache()
    yield
    prefetch.reset_cache()


def test_the_recorded_row_binds_the_config_the_engine_actually_consumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproduction, as a test: flip the file on the door's own first load.

    The flip happens INSIDE ``_load_config``, right after it answers A, so the door's captured mapping
    is A while the workspace on disk is B for everything derived afterwards. That is the ordering that
    produced engine A / receipt B.
    """
    workspace = _seed(tmp_path / "serving")
    config_path = Path(workspace) / "mind-mem.json"
    config_b = copy.deepcopy(_CONFIG_A)
    config_b["extraction"]["model"] = "capture-b"

    hash_a = current_pipeline_hash(workspace)
    config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
    hash_b = current_pipeline_hash(workspace)
    config_path.write_text(json.dumps(_CONFIG_A), encoding="utf-8", newline="\n")
    assert hash_a != hash_b, "the two configs must hash differently or nothing is discriminated"

    loads: list[int] = []
    real_loader = recall_tool._load_config

    def capture_then_flip(ws: str) -> dict[str, Any]:
        loaded = real_loader(ws)
        loads.append(1)
        if len(loads) == 1:
            config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        return loaded

    seen: list[dict[str, Any]] = []
    real_factory = hybrid_recall.HybridBackend.from_config

    def spy(config: dict[str, Any]) -> Any:
        seen.append(json.loads(json.dumps(config)))
        return real_factory(config)

    monkeypatch.setattr(recall_tool, "_load_config", capture_then_flip)
    monkeypatch.setattr(hybrid_recall.HybridBackend, "from_config", staticmethod(spy))

    with use_workspace(workspace):
        payload = json.loads(recall_tool._recall_impl(_QUERY, limit=5, backend="auto"))

    assert payload.get("results"), payload
    assert seen, "the real HybridBackend was never constructed; this proves nothing"
    assert all(cfg.get("extraction", {}).get("model") == "capture-a" for cfg in seen), (
        f"the engine did not consume A, so the receipt check below is meaningless: {seen}"
    )
    assert json.loads(config_path.read_text(encoding="utf-8")) == config_b, "the flip never reached disk"

    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), payload
    rows = read_served_runs(workspace)

    if attestation.get("served_proof") == "unproven":
        # A refusal is acceptable: it records no claim at all.
        assert rows == (), rows
        assert attestation.get("ledger_error"), attestation
        return

    assert attestation.get("served_proof") == "recorded", attestation
    assert len(rows) == 1, rows
    row = rows[0]
    assert isinstance(row, ServedRunV2), row
    assert row.pipeline_hash == hash_a, (
        f"the engine consumed A but the recorded receipt binds {row.pipeline_hash[:16]}…; hash_a={hash_a[:16]}… hash_b={hash_b[:16]}…"
    )
    assert attestation.get("config_hash") == hash_a, attestation


def test_a_bound_context_does_not_downgrade_a_non_markdown_backend(tmp_path: Path) -> None:
    """A postgres workspace must still resolve to postgres through every bound path."""
    workspace = str(tmp_path / "pg")
    Path(workspace).mkdir(parents=True)
    config = {"block_store": {"backend": "postgres", "dsn": "postgresql://user@host/db"}}
    Path(workspace, "mind-mem.json").write_text(json.dumps(config), encoding="utf-8", newline="\n")

    assert _backend_name(workspace) == "postgres", "baseline: the file alone must resolve to postgres"

    context = RequestContext(workspace=workspace, config=config)
    assert type(context.config) is dict, (
        "the context must hold a plain dict; a mappingproxy fails isinstance(config, dict) and silently degrades storage to markdown"
    )
    with bind_request_context(context):
        assert _backend_name(workspace, config=context.config) == "postgres"
        assert _backend_name(workspace, config=_load_config(workspace)) == "postgres"


def test_the_downgrade_predicate_still_fires_on_a_mappingproxy(tmp_path: Path) -> None:
    """The negative control for the test above.

    Without this, ``_backend_name`` could stop checking the type at all and the test above would pass
    for the wrong reason. This pins that a proxy IS still rejected — which is precisely why the
    context must not hand one out.
    """
    workspace = str(tmp_path / "pg2")
    Path(workspace).mkdir(parents=True)
    config = {"block_store": {"backend": "postgres", "dsn": "postgresql://user@host/db"}}
    Path(workspace, "mind-mem.json").write_text(json.dumps(config), encoding="utf-8", newline="\n")
    assert _backend_name(workspace, config=MappingProxyType(config)) == "markdown", (
        "the predicate no longer degrades on a non-dict mapping, so the plain-dict requirement is "
        "no longer load-bearing and this whole guard has stopped measuring anything"
    )


def test_a_hash_taken_inside_a_bound_context_describes_that_context(tmp_path: Path) -> None:
    """The mechanism, isolated: the hash follows the bound config, not the file."""
    workspace = _seed(tmp_path / "hash")
    config_path = Path(workspace) / "mind-mem.json"
    config_b = copy.deepcopy(_CONFIG_A)
    config_b["extraction"]["model"] = "capture-b"

    hash_a = current_pipeline_hash(workspace)
    config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
    hash_b_from_disk = current_pipeline_hash(workspace)
    assert hash_a != hash_b_from_disk

    # Disk now says B. A hash taken under a context carrying A must still be A.
    with bind_request_context(RequestContext(workspace=workspace, config=_CONFIG_A)):
        assert current_pipeline_hash(workspace) == hash_a, "the hash was re-read from disk instead of derived from the bound config"
    # And outside the context it follows the file again, so the binding is not a permanent freeze.
    assert current_pipeline_hash(workspace) == hash_b_from_disk


def test_the_direct_python_door_records_the_config_it_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public ``recall()`` door, which derived its hash outside the captured context.

    ``capture_policy_snapshot`` loaded the config and then called ``current_pipeline_hash`` and
    ``_resolve_index_anchor`` against the workspace, so the new context-aware loader could not reach
    them. An independent source-bound reproduction observed the engine consuming A while the recorded
    row and attestation carried hash B. Flipping the file immediately after the capture returns is the
    ordering that produced it.
    """
    workspace = _seed(tmp_path / "direct")
    config_path = Path(workspace) / "mind-mem.json"
    config_b = copy.deepcopy(_CONFIG_A)
    config_b["extraction"]["model"] = "capture-b"

    hash_a = current_pipeline_hash(workspace)
    config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
    hash_b = current_pipeline_hash(workspace)
    config_path.write_text(json.dumps(_CONFIG_A), encoding="utf-8", newline="\n")
    assert hash_a != hash_b

    from mind_mem import recall as recall_module

    real_snapshot = recall_module.capture_policy_snapshot

    def snapshot_then_flip(ws: str) -> Any:
        captured = real_snapshot(ws)
        config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        return captured

    monkeypatch.setattr(recall_module, "capture_policy_snapshot", snapshot_then_flip)

    with use_workspace(workspace):
        served = recall_module.recall(workspace, _QUERY)

    attestation = served.attestation or {}
    rows = read_served_runs(workspace)
    assert json.loads(config_path.read_text(encoding="utf-8")) == config_b, "the flip never landed"

    if attestation.get("served_proof") == "unproven":
        assert rows == (), rows
        return
    assert attestation.get("config_hash") == hash_a, attestation
    assert [r.pipeline_hash for r in rows] == [hash_a], "the direct door recorded a hash taken after its own capture"


def test_the_public_prefetch_door_binds_its_fan_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefetch door captured AFTER the fan-out, so engine and receipt could disagree.

    The reproduction flipped the workspace to B for the actual N+1 core recalls and restored A before
    the post-retrieval snapshot: six real core reads observed model B while the row carried hash A —
    engine B, receipt A, the mirror of the ranked door's defect. The fix captures and binds BEFORE the
    fan-out, so the workers cannot see a later file at all.
    """
    workspace = _seed(tmp_path / "prefetch")
    config_path = Path(workspace) / "mind-mem.json"
    config_b = copy.deepcopy(_CONFIG_A)
    config_b["extraction"]["model"] = "capture-b"
    hash_a = current_pipeline_hash(workspace)

    seen: list[Any] = []
    from mind_mem import _recall_core

    real_get_config = _recall_core._get_config

    def observe(ws: str) -> Any:
        cfg = real_get_config(ws)
        seen.append((cfg or {}).get("extraction", {}).get("model"))
        return cfg

    monkeypatch.setattr(_recall_core, "_get_config", observe)

    from mind_mem import recall as recall_module

    real_prefetch = recall_module.prefetch_context

    def flip_then_prefetch(*args: Any, **kwargs: Any) -> Any:
        config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        try:
            return real_prefetch(*args, **kwargs)
        finally:
            config_path.write_text(json.dumps(_CONFIG_A), encoding="utf-8", newline="\n")

    monkeypatch.setattr(recall_module, "prefetch_context", flip_then_prefetch)

    with use_workspace(workspace):
        payload = json.loads(recall_tool.prefetch(_QUERY, limit=5))

    assert payload.get("error") is None, payload
    observed = [m for m in seen if m is not None]
    assert observed, "no core config read was observed; this test would prove nothing"
    assert all(m == "capture-a" for m in observed), f"the fan-out read the config written during the request: {observed}"
    rows = read_served_runs(workspace)
    if rows:
        assert [r.pipeline_hash for r in rows] == [hash_a], rows


def test_a_consumer_cannot_mutate_a_nested_section_of_the_context(tmp_path: Path) -> None:
    """Shallow copying was not enough, and a real consumer proved it.

    ``HybridBackend.__init__`` deletes a nested recall key when validation rejects its value. A control
    passed ``{'recall': {'rrf_k': 'bad', ...}}`` through ``from_config`` and ``rrf_k`` then disappeared
    from the captured context itself. A snapshot a consumer can edit is not a snapshot.
    """
    workspace = str(tmp_path / "nested")
    Path(workspace).mkdir(parents=True)
    config = {"recall": {"rrf_k": "bad", "vector_weight": 0.5}}
    Path(workspace, "mind-mem.json").write_text(json.dumps(config), encoding="utf-8", newline="\n")
    context = RequestContext(workspace=workspace, config=config)
    before = sorted(context.config["recall"])
    with bind_request_context(context):
        handed = context_config_for(workspace)
        assert handed is not None
        del handed["recall"]["rrf_k"]  # exactly what HybridBackend does on a bad value
        handed["recall"]["vector_weight"] = 99
    after = sorted(context.config["recall"])
    assert before == after == ["rrf_k", "vector_weight"], (before, after)
    assert context.config["recall"]["vector_weight"] == 0.5, "a consumer's nested edit reached the captured context"
