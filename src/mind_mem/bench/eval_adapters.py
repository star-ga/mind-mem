#!/usr/bin/env python3
"""Concrete retrieval-eval adapters.

Two adapters, deliberately thin — they *wrap* retrieval mind-mem already
ships, they do not reimplement it:

* :class:`Bm25BaselineAdapter` — the honesty floor. A self-contained
  in-memory BM25 over the session documents, zero store, zero deps. Not a
  product; it exists so every product number is reported next to the
  cheapest reasonable baseline. If the real pipeline can't beat this, the
  scorecard shows it.

* :class:`MindMemAdapter` — ingests the sessions into a **real** mind-mem
  workspace (Markdown corpus + optional SQLite index) and answers queries
  through the **real** public recall entry point,
  :func:`mind_mem.recall.recall`. Its probe records the backend the
  workspace config actually resolved to, so a run that meant to measure
  ``hybrid`` but fell to ``scan`` is caught, not buried.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .eval_adapter import PipelineProbe, SessionDoc, config_sha256

# --------------------------------------------------------------------------
# BM25 baseline (honesty floor)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BM25_K1 = 1.2
_BM25_B = 0.75


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Bm25State:
    docs: list[SessionDoc]
    tfs: list[Counter]
    df: Counter
    avgdl: float
    probe: PipelineProbe


class Bm25BaselineAdapter:
    """Classic BM25 over session text — the cheapest honest baseline."""

    name = "bm25_baseline"

    def init(self, sessions: list[SessionDoc], config: dict[str, Any] | None) -> _Bm25State:
        tfs = [Counter(_tok(d.text)) for d in sessions]
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        avgdl = (sum(sum(tf.values()) for tf in tfs) / len(tfs)) if tfs else 0.0
        probe = PipelineProbe(
            adapter=self.name,
            declared_backend="bm25_inmemory",
            effective_backend="bm25_inmemory",
            vector_available=False,
            config_sha256=config_sha256(config),
            notes="self-contained BM25; no store, no external deps",
        )
        return _Bm25State(docs=list(sessions), tfs=tfs, df=df, avgdl=avgdl, probe=probe)

    def query(self, q: str, state: _Bm25State, k: int) -> list[dict[str, Any]]:
        n = len(state.docs)
        if n == 0:
            return []
        q_terms = set(_tok(q))
        scored: list[tuple[float, str]] = []
        for doc, tf in zip(state.docs, state.tfs):
            dl = sum(tf.values()) or 1
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - state.df[term] + 0.5) / (state.df[term] + 0.5))
                denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (state.avgdl or 1))
                s += idf * (f * (_BM25_K1 + 1)) / denom
            if s > 0:
                scored.append((s, doc.doc_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"doc_id": did, "score": round(sc, 6)} for sc, did in scored[:k]]

    def teardown(self, state: _Bm25State) -> None:  # noqa: D401 - no resources
        return None


# --------------------------------------------------------------------------
# mind-mem adapter (real store + real recall path)
# --------------------------------------------------------------------------

_SID_UNSAFE = re.compile(r"[^A-Za-z0-9]")


def _sanitise(text: str) -> str:
    """Flatten to one line so it never splits a Markdown block."""
    flat = " / ".join(p.strip() for p in text.splitlines() if p.strip())
    return flat.replace("[", "(").replace("]", ")").strip() or "(empty)"


#: Block-id prefix the harness seeds under. Must be a prefix
#: ``corpus_registry.CORPUS_TABLE`` routes -- ``write_block`` raises for one
#: it cannot route, and the old hand-written ``SESSION-`` ids were not
#: routable at all. ``D`` routes to ``decisions/DECISIONS.md``, which is the
#: file this harness was writing by hand anyway.
_SEED_PREFIX = "D"


def _seed_governed(workspace: str, sessions: list[SessionDoc]) -> dict[str, str]:
    """Seed the eval workspace through the governed write path.

    Returns ``block_id -> doc_id``.

    This used to open ``decisions/DECISIONS.md`` and write the whole
    haystack by hand, with ``Status: active`` spelled into the text -- a
    servable status minted with no admission, no evidence row and no chain
    row. It is a synthetic workspace rather than an operator's corpus, which
    is why it was pinned as PENDING rather than treated as a leak; but a
    benchmark that seeds memory by bypassing governance is not measuring the
    product's write path, and the numbers it produces are for a system that
    does not ship. ``bench/ab_seed`` already learned this -- its module
    docstring records that an earlier draft appended directly and this
    repository's own structural invariant refused it -- so this takes the
    same route: one ``admit_proposal`` scope, then ``write_block`` per
    block.

    ``admit_proposal`` is the right scope and not a convenience: it is the
    only tier that mints ``ACTIVE``, and the haystack has to be servable or
    recall retrieves nothing and the benchmark measures zero.
    """
    from ..governance_gate import get_gate
    from ..storage import get_block_store

    store = get_block_store(workspace)
    id_map: dict[str, str] = {}
    blocks: list[dict[str, str]] = []
    for idx, doc in enumerate(sessions):
        block_id = f"{_SEED_PREFIX}-{idx}"
        id_map[block_id] = doc.doc_id
        blocks.append({"_id": block_id, "Statement": _sanitise(doc.text), "Status": "active"})

    if blocks:
        with get_gate(workspace).admit_proposal(
            f"P-bench-eval-seed-{len(blocks)}",
            "\n".join(b["_id"] for b in blocks),
            actor="bench_eval_adapter",
            target_file=os.path.join("decisions", "DECISIONS.md"),
            metadata={"benchmark": "eval_adapters", "blocks": str(len(blocks))},
        ):
            for block in blocks:
                store.write_block(dict(block))
    return id_map


@dataclass
class _MindMemState:
    workspace: str
    id_map: dict[str, str]  # block_id -> original doc_id
    probe: PipelineProbe


class MindMemAdapter:
    """Ingest sessions into a real mind-mem store; query the real recall path.

    ``config`` (defaults to a benchmark-mode preset) is written verbatim to
    the workspace's ``mind-mem.json`` and hashed into the probe. The probe's
    ``effective_backend`` starts from mind-mem's own ``_load_backend`` on the
    built workspace, and is then reconciled against the disk state that
    actually decides the dispatch — so it reflects what recall will *really*
    run, not what the caller hoped for.

    That second step is the whole point of the probe. ``_load_backend``
    answers from the config value alone: for ``backend: "sqlite"`` it returns
    the string ``"sqlite"`` having touched no disk, so declared and effective
    would be the same value by construction and ``PipelineProbe.mismatch``
    could never fire on the default path. ``query_index`` meanwhile falls
    back to the Markdown BM25 scan whenever the index db file is absent — so
    a ``build_index`` that raised would otherwise publish
    ``pipeline_mismatch: false`` for a whole run that never touched the index.
    """

    name = "mind_mem"

    #: Benchmark-mode default: SQLite backend with the recall caps that
    #: cripple a many-session haystack (knee cutoff, per-type dedup cap)
    #: turned OFF. See FINDINGS §2. Vector is left disabled by default so
    #: the harness runs zero-dep; pass a config with ``vector_enabled`` to
    #: exercise the hybrid path where the embedder is installed.
    DEFAULT_CONFIG: dict[str, Any] = {
        "recall": {
            "backend": "sqlite",
            "knee_cutoff": False,
            "dedup": {"enabled": False},
        }
    }

    def init(self, sessions: list[SessionDoc], config: dict[str, Any] | None) -> _MindMemState:
        import json as _json

        cfg = config if config is not None else self.DEFAULT_CONFIG
        recall_cfg = cfg.get("recall", {}) if isinstance(cfg, dict) else {}
        declared = recall_cfg.get("backend", "scan") if isinstance(recall_cfg, dict) else "scan"

        ws = tempfile.mkdtemp(prefix="mm_eval_")
        os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)

        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)

        id_map = _seed_governed(ws, sessions)

        effective, vector_available, notes, extra = self._probe_backend(ws, declared)

        probe = PipelineProbe(
            adapter=self.name,
            declared_backend=str(declared),
            effective_backend=effective,
            vector_available=vector_available,
            config_sha256=config_sha256(cfg),
            notes=notes,
            extra={"n_sessions": len(sessions), **extra},
        )
        return _MindMemState(workspace=ws, id_map=id_map, probe=probe)

    def _probe_backend(self, ws: str, declared: str) -> tuple[str, bool, str, dict[str, Any]]:
        """Resolve what recall will actually dispatch to on this workspace.

        Returns ``(effective_backend, vector_available, notes, extra)``.
        """
        notes: list[str] = []
        extra: dict[str, Any] = {}
        # Build the SQLite index if that is the configured backend, so the
        # probe reflects the path a real query takes.
        if declared == "sqlite":
            try:
                from ..sqlite_index import build_index

                build_index(ws, incremental=False)
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"build_index_failed:{type(exc).__name__}")

        effective = "scan"
        try:
            from .._recall_core import RecallBackend, _load_backend

            resolved = _load_backend(ws)
            if isinstance(resolved, str):
                effective = resolved
            elif isinstance(resolved, RecallBackend):
                effective = "recall_backend"
            else:
                effective = "scan"
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"load_backend_failed:{type(exc).__name__}")

        if effective == "sqlite":
            # _load_backend said "sqlite" from the config string alone. Ask
            # the disk whether the index it names is actually there, because
            # that — not the config — is what query_index branches on.
            effective, index_extra, index_notes = self._reconcile_sqlite_index(ws)
            extra.update(index_extra)
            notes.extend(index_notes)

        vector_available = False
        try:
            import importlib.util

            vector_available = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:  # pragma: no cover - defensive
            vector_available = False

        return effective, vector_available, "; ".join(notes), extra

    @staticmethod
    def _reconcile_sqlite_index(ws: str) -> tuple[str, dict[str, Any], list[str]]:
        """Downgrade a claimed ``sqlite`` backend to what the disk supports.

        ``query_index`` falls back to the Markdown BM25 scan when the index
        db file does not exist, so an absent index means the run measured the
        scan no matter what the config says — report ``"scan"`` and let the
        mismatch tripwire fire. A present-but-empty index still dispatches to
        sqlite, so it stays ``"sqlite"``; the row count is published in
        ``extra`` and flagged in ``notes`` instead of being disguised as a
        different backend.
        """
        notes: list[str] = []
        try:
            from ..sqlite_index import index_status

            status = index_status(ws)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"index_status_failed:{type(exc).__name__}")
            return "sqlite", {"index_probe": "failed"}, notes

        exists = bool(status.get("exists"))
        blocks = int(status.get("blocks") or 0)
        extra: dict[str, Any] = {"index_exists": exists, "index_blocks": blocks}
        if not exists:
            notes.append("sqlite_index_missing:recall falls back to markdown scan")
            return "scan", extra, notes
        if blocks == 0:
            notes.append("sqlite_index_empty:0 blocks indexed")
        return "sqlite", extra, notes

    def query(self, q: str, state: _MindMemState, k: int) -> list[dict[str, Any]]:
        from ..recall import recall

        hits = recall(state.workspace, q, limit=k)
        out: list[dict[str, Any]] = []
        for h in hits:
            raw_id = h.get("_id") or h.get("block_id") or h.get("id") or ""
            doc_id = state.id_map.get(raw_id, raw_id)
            score = h.get("score")
            if score is None:
                score = h.get("rerank_score") or h.get("rrf_score") or h.get("bm25_score") or 0.0
            out.append({"doc_id": doc_id, "score": float(score)})
        return out[:k]

    def teardown(self, state: _MindMemState) -> None:
        shutil.rmtree(state.workspace, ignore_errors=True)


def get_adapter(name: str):
    """Return an adapter instance by name (``bm25_baseline`` / ``mind_mem``)."""
    registry = {
        Bm25BaselineAdapter.name: Bm25BaselineAdapter,
        MindMemAdapter.name: MindMemAdapter,
    }
    if name not in registry:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(registry)}")
    return registry[name]()
