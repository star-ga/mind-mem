"""The public Python recall wrapper must snapshot policy before ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import mind_mem.recall as recall_module
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.served_ledger import read_served_runs

_QUERY = "deterministic compiler"
_CONFIG_A = {
    "cache": {"enabled": False},
    "extraction": {"backend": "unknown-a", "model": "a"},
}
_CONFIG_B = {
    "cache": {"enabled": False},
    "extraction": {"backend": "unknown-b", "model": "b"},
}
_RANKED_CONFIG_A = {
    "cache": {"enabled": False},
    "extraction": {"backend": "unknown-a", "model": "a"},
    "recall": {
        "query_expansion": {"enabled": False, "auto_enable": False},
        "vector_enabled": False,
    },
}
_RANKED_CONFIG_B = {
    "cache": {"enabled": False},
    "extraction": {"backend": "unknown-b", "model": "b"},
    "recall": {
        "query_expansion": {"enabled": True, "auto_enable": False},
        "vector_enabled": False,
    },
}


def _seed_workspace(root: Path) -> str:
    (root / "decisions").mkdir(parents=True)
    for name in ("tasks", "entities", "intelligence"):
        (root / name).mkdir()
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-RA1-PY-001]\nStatement: deterministic compiler retrieval context\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "mind-mem.json").write_text(json.dumps(_CONFIG_A), encoding="utf-8", newline="\n")
    return str(root)


def test_python_recall_snapshots_before_actual_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A governed config change after ranking cannot change the recorded hash."""
    workspace = _seed_workspace(tmp_path / "direct-python")
    config_path = Path(workspace) / "mind-mem.json"
    retrieval_hash = current_pipeline_hash(workspace)
    transitioned = False
    real_engine = recall_module._engine_recall

    def retrieve_then_change(*args: Any, **kwargs: Any) -> Any:
        nonlocal transitioned
        result = real_engine(*args, **kwargs)
        transitioned = True
        config_path.write_text(json.dumps(_CONFIG_B), encoding="utf-8", newline="\n")
        return result

    monkeypatch.setattr(recall_module, "_engine_recall", retrieve_then_change)

    with use_workspace(workspace):
        served = recall_module.recall(workspace, _QUERY)

    assert transitioned, "the real retrieval engine was not invoked"
    assert served, "the public Python door must return the real retrieval result"
    attestation = served.attestation
    assert isinstance(attestation, dict), served

    rows = read_served_runs(workspace)
    if attestation.get("served_proof") == "unproven":
        assert attestation.get("served_seq") is None, attestation
        assert attestation.get("served_row_hash") is None, attestation
        assert attestation.get("ledger_error"), attestation
        assert rows == (), rows
        return

    assert attestation.get("served_proof") == "recorded", attestation
    assert attestation.get("config_hash") == retrieval_hash, (
        "the direct Python door captured policy after actual ranking and recorded the later hash"
    )
    assert len(rows) == 1, rows
    assert rows[0].pipeline_hash == retrieval_hash, rows[0]


def _seed_ranked_workspace(root: Path) -> str:
    workspace = _seed_workspace(root)
    Path(workspace, "mind-mem.json").write_text(json.dumps(_RANKED_CONFIG_A), encoding="utf-8", newline="\n")
    return workspace


def test_ranked_positive_passes_the_real_a_config_to_hybrid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stable control observes the real constructor with A, not a fake result."""
    workspace = _seed_ranked_workspace(tmp_path / "ranked-positive")
    seen: list[dict[str, Any]] = []

    from mind_mem import hybrid_recall

    real_factory = hybrid_recall.HybridBackend.from_config

    def spy(config: dict[str, Any]) -> Any:
        seen.append(json.loads(json.dumps(config)))
        return real_factory(config)

    monkeypatch.setattr(hybrid_recall.HybridBackend, "from_config", staticmethod(spy))

    with use_workspace(workspace):
        payload = json.loads(recall_tool._recall_impl(_QUERY, limit=5, backend="auto"))

    assert payload.get("results"), payload
    assert seen, "the real HybridBackend constructor was not reached"
    assert all(not bool(cfg.get("recall", {}).get("query_expansion", {}).get("enabled")) for cfg in seen), seen


def test_ranked_pre_retrieval_mutation_keeps_engine_and_row_coherent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured A context must govern the real engine or produce an honest refusal."""
    workspace = _seed_ranked_workspace(tmp_path / "ranked-boundary")
    config_path = Path(workspace) / "mind-mem.json"
    hash_a = current_pipeline_hash(workspace)
    seen: list[dict[str, Any]] = []
    real_uncached = recall_tool._recall_impl_uncached

    from mind_mem import hybrid_recall

    real_factory = hybrid_recall.HybridBackend.from_config

    def spy(config: dict[str, Any]) -> Any:
        seen.append(json.loads(json.dumps(config)))
        return real_factory(config)

    def flip_before_real_engine(*args: Any, **kwargs: Any) -> Any:
        config_path.write_text(json.dumps(_RANKED_CONFIG_B), encoding="utf-8", newline="\n")
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(hybrid_recall.HybridBackend, "from_config", staticmethod(spy))
    monkeypatch.setattr(recall_tool, "_recall_impl_uncached", flip_before_real_engine)

    with use_workspace(workspace):
        payload = json.loads(recall_tool._recall_impl(_QUERY, limit=5, backend="auto"))

    assert payload.get("results"), payload
    assert seen, "the real HybridBackend constructor was not reached"
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), payload
    rows = read_served_runs(workspace)
    if attestation.get("served_proof") == "unproven":
        assert attestation.get("served_seq") is None, attestation
        assert attestation.get("served_row_hash") is None, attestation
        assert attestation.get("ledger_error"), attestation
        assert rows == (), rows
        return

    assert attestation.get("served_proof") == "recorded", attestation
    assert all(not bool(cfg.get("recall", {}).get("query_expansion", {}).get("enabled")) for cfg in seen), (
        "the recorded answer ran under B while its metadata claims captured A"
    )
    assert attestation.get("config_hash") == hash_a, attestation
    assert len(rows) == 1, rows
    assert rows[0].pipeline_hash == hash_a, rows[0]
