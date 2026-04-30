"""Shared tool-internal helpers — workspace paths + lazy-init singletons.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(PR-3 staging). These helpers are consumed by multiple tool
modules (signal, graph, ontology, core, consolidation, agent)
and are tool-private: no test references them directly. Keeping
them in one file avoids duplicating the lazy-init pattern across
every consumer.

Path helpers
------------
* :func:`_signal_store_path` — JSONL append-only signal log.
* :func:`_kg_path` — SQLite knowledge-graph DB.
* :func:`_core_dir` — on-disk context-core registry directory
  (created eagerly on first access).

Lazy singletons
---------------
* :func:`_ontology_registry` — preloads the bundled
  ``software_engineering_ontology`` on first access so
  ``ontology_validate`` works without a separate load step.
* :func:`_change_stream` — module-wide :class:`ChangeStream`.
* :func:`_core_registry` — module-wide :class:`CoreRegistry`.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Re-exports of cross-tier helpers used by every MCP tool. Importing them
# here lets each tool module say `from ._helpers import get_logger, metrics,
# traced` (within-package edge) instead of `from mind_mem.observability
# import get_logger, metrics` (cross-package edge). This keeps the
# arch-mind modularity_q16 metric high — every tool that reaches across
# package boundaries reduces it.
# ---------------------------------------------------------------------------

from mind_mem.observability import get_logger, metrics  # noqa: E402, F401
from mind_mem.telemetry import traced  # noqa: E402, F401

__all__ = [
    "_signal_store_path",
    "_kg_path",
    "_core_dir",
    "_ontology_registry",
    "_change_stream",
    "_core_registry",
    "get_logger",
    "metrics",
    "traced",
]


def _signal_store_path(ws: str) -> str:
    return os.path.join(ws, "memory", "interaction_signals.jsonl")


def _kg_path(ws: str) -> str:
    return os.path.join(ws, "memory", "knowledge_graph.db")


def _core_dir(ws: str) -> str:
    path = os.path.join(ws, "memory", "cores")
    os.makedirs(path, exist_ok=True)
    return path


_ONTOLOGY_REGISTRY: Any = None
_CHANGE_STREAM: Any = None
_CORE_REGISTRY: Any = None


def _ontology_registry() -> Any:
    global _ONTOLOGY_REGISTRY
    if _ONTOLOGY_REGISTRY is None:
        from mind_mem.ontology import OntologyRegistry, software_engineering_ontology

        _ONTOLOGY_REGISTRY = OntologyRegistry()
        # Preload the in-box SE ontology so ontology_validate works on
        # a fresh workspace without a separate ontology_load step.
        _ONTOLOGY_REGISTRY.load(software_engineering_ontology(), make_active=True)
    return _ONTOLOGY_REGISTRY


def _change_stream() -> Any:
    global _CHANGE_STREAM
    if _CHANGE_STREAM is None:
        from mind_mem.change_stream import ChangeStream

        _CHANGE_STREAM = ChangeStream()
    return _CHANGE_STREAM


def _core_registry() -> Any:
    global _CORE_REGISTRY
    if _CORE_REGISTRY is None:
        from mind_mem.context_core import CoreRegistry

        _CORE_REGISTRY = CoreRegistry()
    return _CORE_REGISTRY
