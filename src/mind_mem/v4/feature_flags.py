"""v4.0 feature-flag registry.

Every v4 surface is gated. Flags live in ``mind-mem.json`` under the
``v4`` key; absence of the key (or the specific sub-flag) means the
surface is OFF. v3.x deployments that never touch the config see no
change in behaviour.

Example ``mind-mem.json`` snippet to enable v4 block kinds + long-context
recall:

    {
        "version": "4.0.0-alpha.1",
        "v4": {
            "block_kinds": { "enabled": true },
            "long_context_recall": { "enabled": true, "max_tokens": 32000 }
        }
    }

The flags listed in :data:`ALL_V4_FLAGS` are the authoritative set; new
surfaces must register here when they're added.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Authoritative list of v4 feature flags. Each maps to a sub-key under
#: ``mind-mem.json: v4: {...}``. Order roughly matches ROADMAP §A→§E.
ALL_V4_FLAGS: Final[tuple[str, ...]] = (
    # Group A — cognition / model layer
    "tier_memory",
    "cognitive_kernel",
    "surprise_retrieval",
    # Group B — knowledge graph
    "block_kinds",
    "long_context_recall",
    "fusion",
    "streaming_recall",
    "chat",
    "prompt_schema",
    # Group C — governance / UX
    "idle_ingest",
    "lint",
    "contradiction_states",
    "self_heal",
    "viewer",
    "contradiction_stream",
    # Group D — platform scale (selected v4-introduced items)
    "rust_hot_path",
    "embedding_fallback",
    "pq",  # product-quantization codec (audit 4/4 consensus 2026-05-10)
    "hnsw_kind_index",  # HNSW kind-filtered ANN (audit 3/4)
    "consolidation_worker",  # write-time clustering worker (audit 2/4)
    "federation",  # cross-agent version vectors + conflict log (round 2 audit 4/4)
    "embedding_pipeline",  # auto-derive embeddings (round 2 audit 4/4)
    "kind_summaries",  # GraphRAG-style per-kind summaries (round 2 audit 3/4)
    "self_editing",  # MemGPT-style propose_edit / approve_edit (round 2 audit 2/4)
    "observability",  # counters / gauges / histograms (round 3 audit 4/4)
    "eviction",  # cold-tier eviction policies (round 3 audit 2/4)
    "backpressure",  # ingestion overload signal (round 4 audit, DeepSeek 9.75→10)
    "block_metadata",  # ChromaDB-style tags + Weaviate-style validators (round 4 audit)
    # Group E — compliance-sensitive opt-in
    "redaction",
    "time_bounded_recall",
    "vocabulary",
    "provenance",
    "evidence",
    "tenant_kms",
    "tenant_chains",
    "compliance_export",
    "contraindicates_edges",
)


class FeatureDisabledError(RuntimeError):
    """Raised when a v4 surface is invoked while its flag is OFF.

    Caller should either flip the flag in ``mind-mem.json`` or stay on
    the v3.x equivalent.
    """


# ---------------------------------------------------------------------------
# Config loading (read-only; we never write here)
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Resolve the active ``mind-mem.json``.

    Search order:
        1. ``$MIND_MEM_CONFIG`` env var (explicit override)
        2. ``$MIND_MEM_WORKSPACE/mind-mem.json`` (workspace-local)
        3. ``./mind-mem.json`` (cwd)
        4. ``~/.mind-mem/mind-mem.json`` (user-level)

    Returns the first path that exists; if nothing is found, returns
    the cwd default — which won't exist, so ``is_enabled`` falls back to
    "OFF" cleanly.
    """
    explicit = os.environ.get("MIND_MEM_CONFIG")
    if explicit:
        return Path(explicit)

    ws = os.environ.get("MIND_MEM_WORKSPACE")
    if ws:
        candidate = Path(ws) / "mind-mem.json"
        if candidate.is_file():
            return candidate

    cwd = Path.cwd() / "mind-mem.json"
    if cwd.is_file():
        return cwd

    user = Path.home() / ".mind-mem" / "mind-mem.json"
    return user


def _load_v4_block() -> dict:
    """Return the ``v4`` block from active config, or ``{}`` if absent."""
    p = _config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    block = data.get("v4")
    return block if isinstance(block, dict) else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled(flag: str) -> bool:
    """Return True iff the v4 sub-flag is set to ``{"enabled": true}``.

    Unknown flag names always return False — fail-closed so a typo in a
    config file can't accidentally turn things on.
    """
    if flag not in ALL_V4_FLAGS:
        return False
    cfg = _load_v4_block()
    sub = cfg.get(flag)
    if not isinstance(sub, dict):
        return False
    return sub.get("enabled") is True


def require_enabled(flag: str) -> None:
    """Raise :class:`FeatureDisabledError` if the flag is OFF.

    Surfaces should call this at the public-API entry point so callers
    get a clear, structured error instead of silent fallback.
    """
    if not is_enabled(flag):
        raise FeatureDisabledError(
            f'mind-mem v4 surface \'{flag}\' is disabled. Enable via mind-mem.json: "v4": {{ "{flag}": {{ "enabled": true }} }}'
        )


def flag_config(flag: str) -> dict:
    """Return the full sub-config dict for a flag (e.g.
    ``{"enabled": true, "max_tokens": 32000}``), or ``{}`` if unset.

    Surfaces use this to read their own tunables alongside the enable
    bit. Always returns a dict; never raises for missing flags.
    """
    if flag not in ALL_V4_FLAGS:
        return {}
    return _load_v4_block().get(flag, {}) or {}


__all__ = [
    "ALL_V4_FLAGS",
    "FeatureDisabledError",
    "is_enabled",
    "require_enabled",
    "flag_config",
]
