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
  through the **real** product retrieval path. Its probe records the backend
  the workspace config actually resolved to, so a run that meant to measure
  ``hybrid`` but fell to ``scan`` is caught, not buried.

  It has three dispatches, chosen by the declared ``recall.backend``:
  ``scan``/``sqlite``/``vector`` go through :func:`mind_mem.recall.recall`
  (the facade ``_load_backend`` resolves), and ``hybrid`` goes through
  :class:`~mind_mem.hybrid_recall.HybridBackend` directly. The third one
  exists because ``_load_backend`` has no ``hybrid`` case at all: it knows
  ``scan``/``tfidf``, ``sqlite`` and ``vector``, logs unknown config *keys*
  but not unknown *values*, and so answered a ``hybrid`` workspace with the
  Markdown scan and said nothing. Every hybrid number this harness had ever
  produced was a scan number wearing a hybrid label. Adding ``hybrid`` to
  ``_load_backend`` is not the fix — ``HybridBackend`` is not a
  ``RecallBackend`` (its ``search`` is ``(query, workspace)``, the mirror of
  ``RecallBackend.search``; see the bug recorded at ``mm_cli.py:1394``), and
  its BM25 arm re-enters ``recall``/``query_index``, so registering it there
  would be structural recursion. The dispatch is therefore bench-side, and
  the product ranking paths are untouched.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
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


#: ``recall.backend`` values this adapter dispatches through
#: :class:`~mind_mem.hybrid_recall.HybridBackend` rather than through the
#: ``recall()`` facade. ``_load_backend`` has no case for any of them.
_HYBRID_BACKENDS = frozenset({"hybrid"})

#: Label prefix per :class:`~mind_mem._recall_core.RecallBackend` subclass.
#: ``_load_backend`` returns an *instance* for these, so the probe used to
#: record the single opaque string ``"recall_backend"`` for every one of them
#: -- a Postgres run and a vector run were indistinguishable in the artifact.
#: An unmapped class still reports its own name rather than collapsing.
_RECALL_BACKEND_LABELS = {
    "VectorBackend": "vector",
    "PostgresRecallBackend": "postgres",
}


class _HitList(list):
    """Hits, carrying the legs the run that produced them actually ran.

    A ``list`` subclass on purpose -- the adapter contract says ``query``
    returns a list of ``{doc_id, score}`` dicts and the scorer iterates it, so
    every existing caller is unaffected. The two extra attributes ride beside
    the hits for a caller that wants the per-question leg record without
    reaching into the probe. Same shape, and same reason, as the product's own
    :class:`~mind_mem.hybrid_recall.RecallResults`.
    """

    legs_ran: tuple[str, ...] = ()
    legs_degraded: tuple[str, ...] = ()


@dataclass
class _MindMemState:
    workspace: str
    id_map: dict[str, str]  # block_id -> original doc_id
    probe: PipelineProbe
    #: The constructed hybrid backend on the ``hybrid`` dispatch, else None.
    #: Built once in ``init`` and reused per query: constructing it per call
    #: would re-probe the vector backend (and re-log) on every question, so
    #: the config the probe reported and the config the queries ran under
    #: could not drift apart even in principle.
    hybrid: Any | None = None
    #: The ``recall`` sub-dict, kept for ``resolve_rerank_depth``.
    recall_cfg: dict[str, Any] = field(default_factory=dict)


class MindMemAdapter:
    """Ingest sessions into a real mind-mem store; query real product recall.

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

    A declared ``hybrid`` is the case that has no ``_load_backend`` answer at
    all, and the failure was worse than a mismatch: the loader returned
    ``None`` for the unrecognised value and recall served the Markdown scan,
    while the surrounding report read ``hybrid`` off the config string. That
    dispatch is handled here instead — see :meth:`_probe_hybrid`.
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
        if not isinstance(recall_cfg, dict):
            recall_cfg = {}
        declared = recall_cfg.get("backend", "scan")

        ws = tempfile.mkdtemp(prefix="mm_eval_")
        os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)

        with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)

        id_map = _seed_governed(ws, sessions)

        if declared in _HYBRID_BACKENDS:
            effective, vector_available, notes, extra, hybrid = self._probe_hybrid(ws, cfg)
        else:
            effective, vector_available, notes, extra = self._probe_backend(ws, str(declared))
            hybrid = None

        probe = PipelineProbe(
            adapter=self.name,
            declared_backend=str(declared),
            effective_backend=effective,
            vector_available=vector_available,
            config_sha256=config_sha256(cfg),
            notes=notes,
            extra={"n_sessions": len(sessions), **extra},
        )
        return _MindMemState(
            workspace=ws,
            id_map=id_map,
            probe=probe,
            hybrid=hybrid,
            recall_cfg=dict(recall_cfg),
        )

    # -- probes ------------------------------------------------------------

    def _probe_hybrid(self, ws: str, cfg: dict[str, Any]) -> tuple[str, bool, str, dict[str, Any], Any]:
        """Resolve, and construct, the hybrid dispatch for this workspace.

        Returns ``(effective_backend, vector_available, notes, extra, backend)``.

        Order matters. The FTS index is built **first**, because
        ``HybridBackend._bm25_search_raw`` branches on whether ``recall.db``
        exists: with the index present its lexical arm is ``query_index`` —
        the same FTS5 arm the ``sqlite`` configuration measures — and without
        it the arm silently becomes the Markdown scan. Building first is what
        makes a hybrid run and a sqlite run differ in the dense leg and
        nothing else, which is the only way the comparison between them
        answers the question it is asked.

        ``effective_backend`` is ``"hybrid"`` only when the dense leg can
        actually run: the operator asked for it (``vector_enabled``) *and* the
        backend's own probe says it is serviceable. Otherwise it is
        ``"hybrid_bm25_only"``, which differs from the declared ``"hybrid"``
        and so trips :attr:`PipelineProbe.mismatch` — a hybrid that cannot run
        its dense leg is a BM25 run, and it must not be published under the
        other name.

        ``vector_available`` is the backend's OWN probe (its ``_check_vector``
        result, read through the ``vector_available`` property), not an
        ``importlib`` guess. The two answer different questions: ``find_spec``
        says a module could be imported, the backend says the leg it would
        dispatch is serviceable. The old probe published the first under the
        name of the second. The ``find_spec`` fact keeps its own name in
        ``extra["deps_importable"]`` rather than being dropped.
        """
        notes: list[str] = []
        extra: dict[str, Any] = {}

        try:
            from ..sqlite_index import build_index

            build_index(ws, incremental=False)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"build_index_failed:{type(exc).__name__}")

        extra["bm25_arm"] = self._bm25_arm(ws)
        if extra["bm25_arm"] != "sqlite":
            notes.append("hybrid_bm25_arm_is_scan:no recall.db, lexical arm falls to the markdown scan")

        try:
            from ..hybrid_recall import HybridBackend

            backend = HybridBackend.from_config(cfg)
        except Exception as exc:
            notes.append(f"hybrid_backend_construction_failed:{type(exc).__name__}")
            return "scan", False, "; ".join(notes), extra, None

        vector_available = bool(backend.vector_available)
        vector_enabled = bool(backend.vector_enabled)
        effective = "hybrid" if (vector_enabled and vector_available) else "hybrid_bm25_only"
        if vector_enabled and not vector_available:
            notes.append("vector_requested_but_unavailable:dense leg cannot run")
        elif not vector_enabled:
            notes.append("vector_not_requested:recall.vector_enabled is false")

        if vector_enabled and vector_available:
            indexed, build_notes = self._build_vector_index(ws, backend._config)
            extra["vector_index_blocks"] = indexed
            notes.extend(build_notes)
            if indexed == 0:
                extra["vector_leg_inert"] = True

        extra["deps_importable"] = self._deps_importable()
        extra["vector_enabled"] = vector_enabled
        return effective, vector_available, "; ".join(notes), extra, backend

    @staticmethod
    def _build_vector_index(ws: str, recall_cfg: dict[str, Any]) -> tuple[int, list[str]]:
        """Build the embedding store the dense leg reads, and prove it readable.

        The exact counterpart of the FTS build above, and needed for the same
        reason: an arm dispatches on its store existing. Without it the dense
        leg still *runs* — it constructs the backend, embeds the query and
        searches — but finds no ``index.json``, returns zero candidates and
        records no degradation, so the product's own
        :func:`~mind_mem.recall_attestation.derive_legs` reports ``vector``
        and ``hybrid`` over a fusion that had one arm in it. Measured on the
        5-session smoke workspace before this existed: ``bm25_count: 1,
        vector_count: 0, degraded: false``. A "hybrid" label over that is the
        same false green the probe exists to prevent, one layer down.

        Two details are load-bearing.

        **Which writer.** ``VectorBackend.index`` is used rather than the
        module-level ``recall_vector.rebuild_index``, because those two write
        *different shapes to the same path*: ``_index_local`` writes the dict
        ``{model, dimension, blocks, embeddings}`` that ``_load_local_index``
        reads, while ``rebuild_index`` writes a bare list of ``{_id,
        embedding, text}`` records. A local-provider workspace indexed through
        the latter raises ``AttributeError: 'list' object has no attribute
        'get'`` inside ``_load_local_index`` on the next dense query, which
        ``search_batch`` swallows to ``[]``. That is a product defect this
        harness found by being the first caller to reach the path; it is
        reported, not patched from the benchmark tree.

        **The count is a round trip.** It is read back through the *reader*
        (``_load_local_index``), never taken from the writer's own return, so
        a build that produced a file the search leg cannot parse counts zero
        and says so. Counting what the writer claims it wrote is how the shape
        mismatch above stayed invisible in the first place.

        Only reached when the operator asked for the dense leg AND the backend
        says it is serviceable, so no configuration that ran before this
        existed pays for the embeddings.
        """
        notes: list[str] = []
        try:
            from ..recall_vector import VectorBackend

            vb = VectorBackend(dict(recall_cfg))
            vb.index(ws)
            index = vb._load_local_index(ws)
        except Exception as exc:
            notes.append(f"vector_index_build_failed:{type(exc).__name__}")
            return 0, notes
        if not isinstance(index, dict):
            notes.append(f"vector_index_unreadable:{type(index).__name__} is not the shape the search leg reads")
            return 0, notes
        indexed = len(index.get("blocks") or [])
        if indexed == 0:
            notes.append("vector_index_empty:dense leg will contribute nothing")
        return indexed, notes

    @staticmethod
    def _bm25_arm(ws: str) -> str:
        """Which lexical arm ``HybridBackend`` will dispatch to: sqlite | scan.

        Read off the same predicate the arm itself uses — the existence of
        the resolved ``recall.db`` — rather than off the config, because the
        config is exactly what was wrong.
        """
        try:
            from ..sqlite_index import _db_path

            return "sqlite" if os.path.isfile(_db_path(ws)) else "scan"
        except Exception:  # pragma: no cover - defensive
            return "unknown"

    @staticmethod
    def _deps_importable() -> bool:
        """Whether the embedding dependency is importable. Not availability."""
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:  # pragma: no cover - defensive
            return False

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
                # Name the class. ``_load_backend`` returns an instance for
                # both the vector and the Postgres routes, and one shared
                # label made those two runs read identically in the artifact.
                cls = type(resolved).__name__
                effective = f"{_RECALL_BACKEND_LABELS.get(cls, 'recall_backend')}:{cls}"
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

        return effective, self._deps_importable(), "; ".join(notes), extra

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

    # -- query -------------------------------------------------------------

    def query(self, q: str, state: _MindMemState, k: int) -> list[dict[str, Any]]:
        if state.hybrid is not None:
            return self._query_hybrid(q, state, k)
        from ..recall import recall

        return self._rows(recall(state.workspace, q, limit=k), state, k)

    def _query_hybrid(self, q: str, state: _MindMemState, k: int) -> _HitList:
        """Answer through ``HybridBackend``, and record which legs ran.

        Three things this does that the facade path cannot:

        * ``serving_scope`` — the harness re-initialises the workspace per
          question and drives thousands of queries; without the scope every
          engine call underneath would mint its own attestation row for a
          serve nobody is serving. The scope says this caller owns the serve.
        * ``resolve_rerank_depth`` — the product's own resolution of how many
          fused candidates the reranker may see, so the benchmark reranks over
          the same pool a real request would rather than over ``limit`` (which
          cannot change recall@k by construction).
        * :func:`~mind_mem.recall_attestation.derive_legs` — the **product's**
          deriver, run per question against the run's own recorded state (the
          ``.degraded`` marker and per-hit provenance). It is what turns "the
          config said hybrid" into "the dense leg ran on this question", which
          is a measurement rather than a restatement of the input. The legs go
          onto the probe's ``extra``, which the scorer already serialises
          verbatim into every NDJSON row, so the per-row record needs nothing
          from the writer.
        """
        from ..hybrid_recall import resolve_rerank_depth
        from ..recall import serving_scope
        from ..recall_attestation import derive_legs

        backend = state.hybrid
        if backend is None:  # pragma: no cover - the caller dispatches on this
            raise RuntimeError("hybrid dispatch entered with no hybrid backend on the state")
        with serving_scope():
            results = backend.search(
                q,
                state.workspace,
                limit=k,
                rerank_depth=resolve_rerank_depth(state.recall_cfg, k),
            )
        legs_ran, legs_degraded = derive_legs(
            results,
            vector_requested=bool(backend.vector_enabled),
            vector_available=bool(backend.vector_available),
        )
        out = _HitList(self._rows(results, state, k))
        out.legs_ran = tuple(legs_ran)
        out.legs_degraded = tuple(legs_degraded)
        # ``PipelineProbe`` is frozen; ``extra`` is not, and the scorer holds
        # this exact object per question and serialises it after the query.
        state.probe.extra["legs_ran"] = list(legs_ran)
        state.probe.extra["legs_degraded"] = list(legs_degraded)
        return out

    @staticmethod
    def _rows(hits: Any, state: _MindMemState, k: int) -> list[dict[str, Any]]:
        """Map product hits onto the ``{doc_id, score}`` contract."""
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
