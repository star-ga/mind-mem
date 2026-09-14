"""The engine ranks under the CAPTURED config, not the file as it stands when a leg runs.

WHAT THIS ADDS OVER THE DOOR-LEVEL TESTS. Those assert the recorded row's coordinates agree with
each other. That is necessary and it is not the claim an independent review said was missing:
"Passing the context only to ``attest_and_record`` or ``_apply_attestation`` cannot prove which
policy produced the ranking." The engine reloads policy for itself — eight ``_get_config`` sites in
``_recall_core`` plus ``_load_config`` in the ranked door plus ``sqlite_index.query_index`` — so a
row could name a captured hash while ``HybridBackend`` had been constructed from a later config.

So this test watches the REAL backend constructor and the row together. It asserts what
configuration the ranking was actually built from, with the file changed underneath it, and
rejects a recorded A row unless the real engine consumed A. A fail-closed unproven response is
also accepted when the transition makes a coherent row impossible.

WHY THE NEGATIVE CONTROL IS THE LOAD-BEARING HALF. In the ordinary case the captured config and the
file are the same bytes, so a test that merely observes "the backend saw A" passes just as happily
when the binding does nothing at all. The second test therefore simulates the unbound world by
neutralising the lookup the binding depends on, and asserts the mutation DOES reach the backend
there. If that control ever stops going red, this file has stopped measuring anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem import hybrid_recall, prefetch
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.served_ledger import ServedRunV2, read_served_runs

_QUERY = "deterministic compiler"

_CONFIG_A: dict[str, Any] = {
    "cache": {"enabled": False},
    "recall": {"query_expansion": {"enabled": False, "auto_enable": False}, "vector_enabled": False},
}
_CONFIG_B: dict[str, Any] = {
    "cache": {"enabled": False},
    "recall": {"query_expansion": {"enabled": True, "auto_enable": False}, "vector_enabled": False},
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


def _run_with_midrequest_flip(workspace: str, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flip the config to B after the door captured, before the engine ranks.

    Returns every config the real ``HybridBackend`` was constructed from and the
    public envelope, preserving the evidence needed to check its row.
    """
    config_path = Path(workspace) / "mind-mem.json"
    seen: list[dict[str, Any]] = []
    real_factory = hybrid_recall.HybridBackend.from_config
    real_uncached = recall_tool._recall_impl_uncached

    def spy(config: dict[str, Any]) -> Any:
        seen.append(json.loads(json.dumps(config)))
        return real_factory(config)

    def flip_then_rank(*args: Any, **kwargs: Any) -> Any:
        config_path.write_text(json.dumps(_CONFIG_B), encoding="utf-8", newline="\n")
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(hybrid_recall.HybridBackend, "from_config", staticmethod(spy))
    monkeypatch.setattr(recall_tool, "_recall_impl_uncached", flip_then_rank)

    with use_workspace(workspace):
        payload = json.loads(recall_tool._recall_impl(_QUERY, limit=5, backend="auto"))

    assert payload.get("results"), payload
    assert json.loads(config_path.read_text(encoding="utf-8")) == _CONFIG_B, (
        "the flip never landed on disk — nothing was being discriminated"
    )
    assert seen, "the real HybridBackend was never constructed; this proves nothing"
    return seen, payload


def _expansion_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("recall", {}).get("query_expansion", {}).get("enabled"))


def test_the_backend_is_built_from_the_captured_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine and any recorded row agree on A, though the file says B."""
    workspace = _seed(tmp_path / "bound")
    hash_a = current_pipeline_hash(workspace)
    seen, payload = _run_with_midrequest_flip(workspace, monkeypatch)
    rows = read_served_runs(workspace)
    attestation = payload.get("attestation")
    assert isinstance(attestation, dict), payload
    if attestation.get("served_proof") == "unproven":
        assert attestation.get("served_seq") is None, attestation
        assert attestation.get("served_row_hash") is None, attestation
        assert attestation.get("ledger_error"), attestation
        assert rows == (), rows
        return

    assert not any(_expansion_enabled(cfg) for cfg in seen), (
        "the ranking was built from the config written DURING the request, so no recorded hash can describe the policy that produced it"
    )
    assert attestation.get("served_proof") == "recorded", attestation
    assert attestation.get("config_hash") == hash_a, attestation
    assert len(rows) == 1, rows
    row = rows[0]
    assert isinstance(row, ServedRunV2), row
    assert row.pipeline_hash == hash_a, row


def test_without_the_binding_the_mutation_does_reach_the_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control that keeps the test above honest: unbind, and B gets through.

    ``context_config_for`` is the single lookup every bound read goes through, so returning ``None``
    from it is exactly the pre-fix world: the doors still capture, still thread, still record — and
    the engine still reads the file for itself.
    """
    workspace = _seed(tmp_path / "unbound")

    import mind_mem.mcp.infra.config as infra_config
    import mind_mem.request_context as request_context
    from mind_mem import _recall_core

    monkeypatch.setattr(request_context, "context_config_for", lambda _ws: None)
    monkeypatch.setattr(_recall_core, "_config_cache", {}, raising=False)
    monkeypatch.setattr(_recall_core, "_config_mtime", {}, raising=False)
    assert infra_config is not None  # the loader under test resolves the patched symbol lazily

    seen, payload = _run_with_midrequest_flip(workspace, monkeypatch)
    assert payload.get("results"), payload
    assert any(_expansion_enabled(cfg) for cfg in seen), (
        "with the binding neutralised the mid-request config change did NOT reach the backend, so "
        "the positive test above cannot distinguish a working binding from a no-op one"
    )
