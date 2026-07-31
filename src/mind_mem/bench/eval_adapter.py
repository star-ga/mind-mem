#!/usr/bin/env python3
"""Pluggable retrieval-eval adapter contract + pipeline self-assertion.

The point of this module is *not* to invent a new retrieval engine — it is
to wrap the retrieval paths mind-mem already ships (the BM25 scan, the
SQLite FTS index, the BM25+vector hybrid) behind one small contract so a
single scorer loop can drive any of them, and — critically — so every run
records **which pipeline it actually exercised**.

Why the self-assertion matters (read
``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md``): a committed harness
built config-less temp workspaces, so ``recall()`` silently fell back to
the zero-dependency in-memory BM25 *scan* while the surrounding report
claimed a full-stack number. The measured reality (R@5 ≈ 0.30) and the
published claim (85.3) never touched the same code path. This adapter
contract makes that class of false-green impossible: each adapter emits a
:class:`PipelineProbe` — a declared-vs-effective backend label plus a hash
of the exact config it ran under — that the scorer writes into **every**
NDJSON row and into the scorecard header. A run whose ``declared`` and
``effective`` backends disagree is flagged, not hidden.

Adapter contract (four members):

    name: str
        Stable identifier used in artifacts.

    init(sessions, config) -> state
        Ingest the per-question session documents into whatever store the
        adapter uses, returning an opaque ``state``. The returned state
        MUST expose ``.probe`` (a :class:`PipelineProbe`).

    query(q, state, k) -> list[dict]
        Return up to ``k`` hits ranked by descending score, each a dict
        with ``doc_id`` (str) and ``score`` (float).

    teardown(state) -> None      (optional)
        Release any resources (temp workspace, open DB, …).

Adapters live in :mod:`mind_mem.bench.eval_adapters`; the scorer +
LongMemEval driver live in :mod:`mind_mem.bench.longmemeval_suite`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SessionDoc:
    """One retrievable unit: a session's text under a stable ``doc_id``.

    LongMemEval retrieves at *session* granularity, so a document is one
    session (its turns concatenated). ``doc_id`` is the dataset's own
    session id — the scorer compares retrieved ``doc_id``s against the
    gold ``answer_session_ids``.
    """

    doc_id: str
    text: str


def config_sha256(config: dict[str, Any] | None) -> str:
    """Stable short hash of a config dict.

    Sorted-key JSON so logically-equal configs hash equal regardless of
    insertion order. Truncated to 16 hex chars — enough to distinguish
    presets in an artifact, short enough to read.
    """
    payload = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PipelineProbe:
    """A self-asserted record of what an adapter *actually* ran.

    ``declared_backend`` is what the caller asked for (e.g. ``"hybrid"``);
    ``effective_backend`` is what the code path resolved to at runtime
    (e.g. ``"scan"`` because no config / no vector deps). When they
    disagree the run measured something other than what was requested —
    :prop:`mismatch` is the honesty tripwire the scorer surfaces.
    """

    adapter: str
    declared_backend: str
    effective_backend: str
    vector_available: bool
    config_sha256: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mismatch(self) -> bool:
        """True when the effective path is not the declared one.

        A declared ``"auto"`` never counts as a mismatch — the caller
        explicitly deferred the choice to the resolver.
        """
        if self.declared_backend == "auto":
            return False
        return self.declared_backend != self.effective_backend

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "declared_backend": self.declared_backend,
            "effective_backend": self.effective_backend,
            "vector_available": self.vector_available,
            "config_sha256": self.config_sha256,
            "pipeline_mismatch": self.mismatch,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


@runtime_checkable
class AdapterState(Protocol):
    """Opaque per-question state; must carry the pipeline probe."""

    @property
    def probe(self) -> PipelineProbe: ...


@runtime_checkable
class EvalAdapter(Protocol):
    """The four-member retrieval-eval adapter contract."""

    name: str

    def init(self, sessions: list[SessionDoc], config: dict[str, Any] | None) -> AdapterState: ...

    def query(self, q: str, state: AdapterState, k: int) -> list[dict[str, Any]]: ...

    def teardown(self, state: AdapterState) -> None: ...
