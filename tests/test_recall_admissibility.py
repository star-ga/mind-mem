"""Acceptance gate for recall admissibility — the servability allow-list.

Recall used to decide what it could serve with a *deny-list*: drop the
hits whose status spells ``quarantined``, plus an id-set recomputed off
``memory/IMPORTED.md`` for the legs whose hits carry no status. Two holes
follow from that shape and both were reproduced before this file existed:

1. **Fail-open on unknown statuses.** Anything the deny-list had not been
   told about — a status a future ingest door invents — was served.
2. **Legs that splice raw corpus blocks into the result list.** Graph
   expansion, KG fusion and entity prefetch each resolve a neighbour id
   straight out of the parsed corpus and append the block dict. Those
   dicts carry ``Status`` (capital S) while the funnel filter reads
   ``status``, and the id-set rule only ever looked at the *imported*
   corpus, so a withheld block reached the caller through any of them.

The inversion this file locks in: a block is servable only when its
status is named in :data:`~mind_mem.enums.SERVABLE`, the test runs
*before* fusion, and it runs on every leg.

Everything here is hermetic — no network, no embedder download. The
vector leg is driven by the same deterministic stand-in the rest of the
suite uses (``_vector_search`` monkeypatch); "present in the vector
index" is modelled by that leg *returning* the block, which is precisely
what the fusion layer sees.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Callable, Iterator

import pytest

import mind_mem
from mind_mem.admissibility import is_admissible_status
from mind_mem.enums import SERVABLE, Leg, Status, is_servable
from mind_mem.hybrid_recall import HybridBackend
from mind_mem.recall import recall
from mind_mem.sqlite_index import build_index

# The poison block. Every fixture below routes it in through a different
# leg; the ``_ACTIVE`` control run proves that leg really does deliver it.
QUERY = "pineapple protocol rollout"
SEED = "D-20260829-001"
POISON = "C-20260829-999"
ENTITY_POISON = "PRJ-20260829-777"

#: Text that shares no term with :data:`QUERY`. The splice legs (graph, KG,
#: entity prefetch) give their poison this text so lexical retrieval cannot
#: reach it — whatever those rows prove, they prove about the splice.
UNREACHABLE = "zeta ledger cadence harbour"


# ---------------------------------------------------------------------------
# Workspace construction
# ---------------------------------------------------------------------------


def _block(bid: str, statement: str, status: str, **extra: str) -> str:
    lines = [f"[{bid}]", f"Statement: {statement}", "Date: 2026-08-29", f"Status: {status}"]
    lines.extend(f"{k}: {v}" for k, v in extra.items())
    return "\n".join(lines) + "\n\n---\n\n"


def _write(ws: str, rel: str, text: str) -> None:
    path = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _config(ws: str, *, vector: bool = True, **retrieval: Any) -> dict[str, Any]:
    """Write mind-mem.json and return the ``recall`` sub-config."""
    cfg: dict[str, Any] = {"vector_enabled": vector, "provider": "local", "retrieval": dict(retrieval)}
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump({"recall": cfg}, handle)
    return cfg


def _new_ws(tmp_path: Any, name: str) -> str:
    ws = os.path.join(str(tmp_path), name)
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    return ws


def _seed_ws(tmp_path: Any, name: str, *, poison_status: str, poison_text: str) -> str:
    """One servable seed that matches the query and cross-references the poison."""
    ws = _new_ws(tmp_path, name)
    _write(ws, "decisions/DECISIONS.md", _block(SEED, f"The {QUERY} decision, see {POISON}", "active"))
    _write(ws, "intelligence/CONTRADICTIONS.md", _block(POISON, poison_text, poison_status, IngestTier="external-ingest"))
    return ws


# ---------------------------------------------------------------------------
# Per-leg fixtures — one per Leg member. A Leg with no fixture fails.
#
# Each takes the status to stamp on the poison and returns the served ids, so
# the same fixture provides both the assertion and its positive control: run
# it with ``active`` and the leg MUST deliver the block. Without that control
# a row passes whenever the leg silently failed to fire — which is exactly how
# the entity-prefetch row passed while the leg was leaking (dedup happened to
# eat the block downstream).
# ---------------------------------------------------------------------------


def _served_ids(results: Any) -> list[str]:
    return [str(r.get("_id") or r.get("id") or "") for r in results]


def _no_vector(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_vector_search", lambda *a, **k: [])


def _leg_bm25(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    """BM25-only configuration; the poison IS the query text."""
    ws = _seed_ws(tmp_path, f"bm25-{status}", poison_status=status, poison_text=QUERY)
    cfg = _config(ws, vector=False)
    build_index(ws)
    return _served_ids(HybridBackend(config=cfg).search(QUERY, ws, limit=10))


def _leg_vector(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    """Fused configuration; the poison holds rank 1 of the vector leg."""
    ws = _seed_ws(tmp_path, f"vector-{status}", poison_status=status, poison_text=UNREACHABLE)
    cfg = _config(ws)
    build_index(ws)
    backend = HybridBackend(config=cfg)
    monkeypatch.setattr(backend, "_vector_search", lambda *a, **k: [{"_id": POISON, "score": 0.99, "status": status}])
    return _served_ids(backend.search(QUERY, ws, limit=10))


def _leg_graph(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    """The cross-reference walk splices the poison in from the corpus."""
    ws = _seed_ws(tmp_path, f"graph-{status}", poison_status=status, poison_text=UNREACHABLE)
    cfg = _config(ws, multi_hop={"enabled": True})
    build_index(ws)
    backend = HybridBackend(config=cfg)
    _no_vector(backend, monkeypatch)
    return _served_ids(backend.search(QUERY, ws, limit=10))


def _leg_kg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    """A typed knowledge-graph edge splices the poison in from the corpus."""
    from mind_mem.knowledge_graph import KnowledgeGraph, Predicate, default_db_path

    ws = _seed_ws(tmp_path, f"kg-{status}", poison_status=status, poison_text=UNREACHABLE)
    cfg = _config(ws, kg_fusion={"enabled": True})
    build_index(ws)
    db = default_db_path(ws)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    with KnowledgeGraph(db) as kg:
        kg.add_edge("pineapple", Predicate.RELATED_TO, "protocol", source_block_id=POISON)
    backend = HybridBackend(config=cfg)
    _no_vector(backend, monkeypatch)
    return _served_ids(backend.search(QUERY, ws, limit=10))


def _leg_entity_prefetch(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    """The entity tier prefetches a withheld *entity* block by name."""
    ws = _seed_ws(tmp_path, f"prefetch-{status}", poison_status="active", poison_text=UNREACHABLE)
    _write(ws, "entities/projects.md", _block(ENTITY_POISON, UNREACHABLE, status, Name="Pineapple"))
    cfg = _config(ws, entity_prefetch={"enabled": True})
    build_index(ws)
    backend = HybridBackend(config=cfg)
    _no_vector(backend, monkeypatch)
    return _served_ids(backend.search(QUERY, ws, limit=10))


#: Every :class:`Leg` MUST have a row here. ``list(Leg)`` drives the
#: parametrisation, so adding a member without a fixture fails the suite
#: rather than silently leaving a retrieval path untested.
LEG_FIXTURES: dict[Leg, Callable[[Any, pytest.MonkeyPatch, str], list[str]]] = {
    Leg.BM25: _leg_bm25,
    Leg.VECTOR: _leg_vector,
    Leg.GRAPH: _leg_graph,
    Leg.KG: _leg_kg,
    Leg.ENTITY_PREFETCH: _leg_entity_prefetch,
}

#: The id each leg's fixture routes in.
LEG_POISON: dict[Leg, str] = {leg: (ENTITY_POISON if leg is Leg.ENTITY_PREFETCH else POISON) for leg in Leg}


# ---------------------------------------------------------------------------
# T3 — THE GAP. Parametrised over every leg, each with a positive control.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leg", list(Leg), ids=lambda leg: leg.value)
def test_a_leg_delivers_its_block_when_it_is_servable(leg: Leg, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the fixture really does exercise this leg.

    Without it, a row passes whenever the leg silently failed to fire —
    which is how the entity-prefetch row passed while the leg leaked.
    """
    served = LEG_FIXTURES[leg](tmp_path, monkeypatch, "active")
    assert LEG_POISON[leg] in served, f"leg {leg.value} never delivered the block; the withheld row would be vacuous: {served}"


@pytest.mark.parametrize("leg", list(Leg), ids=lambda leg: leg.value)
def test_a_withheld_block_is_never_served_on_any_leg(leg: Leg, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T3: the same block, the same leg, withheld."""
    served = LEG_FIXTURES[leg](tmp_path, monkeypatch, "quarantined")
    assert LEG_POISON[leg] not in served, f"leg {leg.value} served the withheld block: {served}"


@pytest.mark.parametrize("leg", list(Leg), ids=lambda leg: leg.value)
def test_every_leg_has_a_fixture(leg: Leg) -> None:
    """A new Leg member with no fixture is a retrieval path nobody tests."""
    assert leg in LEG_FIXTURES


def test_the_fused_configuration_withholds_a_block_both_legs_return(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two live legs, the poison at rank 1 of each. Still absent."""
    ws = _seed_ws(tmp_path, "fused", poison_status="quarantined", poison_text=QUERY)
    cfg = _config(ws)
    build_index(ws)
    backend = HybridBackend(config=cfg)
    monkeypatch.setattr(backend, "_vector_search", lambda *a, **k: [{"_id": POISON, "score": 0.99, "status": "quarantined"}])
    assert POISON not in _served_ids(backend.search(QUERY, ws, limit=10))


def test_the_agent_facing_recall_tool_withholds_it(tmp_path: Any) -> None:
    """The MCP surface defaults to ``backend="auto"``, which IS the hybrid path."""
    from unittest.mock import patch

    from mind_mem.mcp.tools import recall as mcp_recall

    ws = _seed_ws(tmp_path, "mcp", poison_status="quarantined", poison_text=UNREACHABLE)
    _config(ws, multi_hop={"enabled": True})
    build_index(ws)
    with patch.dict(os.environ, {**os.environ, "MIND_MEM_WORKSPACE": ws}):
        raw = mcp_recall.recall.__wrapped__(QUERY, limit=10)  # type: ignore[attr-defined]
    served = [str(h.get("id") or h.get("_id")) for h in json.loads(raw).get("results", [])]
    assert POISON not in served, served


# ---------------------------------------------------------------------------
# The allow-list inversion: an unknown status is withheld, not served
# ---------------------------------------------------------------------------


#: Statuses that mean "this has not passed the governance gate", plus the
#: fail-closed case: a status nobody has named.
WITHHELD_STATUSES = ["quarantined", "pending", "a-status-nobody-named", "admitted-by-a-future-door"]

#: Statuses a live corpus holds that recall serves *on purpose*. ``superseded``
#: and friends are demoted by the validity gate, not hidden — the history of a
#: decision is the product — and a Task block spends its life on these.
SERVED_STATUSES = ["superseded", "deprecated", "archived", "rejected", "revoked", "wip", "todo", "doing", "done", "open"]


@pytest.mark.parametrize("status", WITHHELD_STATUSES)
def test_a_status_nobody_named_is_withheld_not_served(status: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deny-list served every status it had never heard of. This inverts it."""
    assert not is_admissible_status(status)
    assert POISON not in _leg_graph(tmp_path, monkeypatch, status)


@pytest.mark.parametrize("status", SERVED_STATUSES)
def test_a_recognised_lifecycle_status_is_still_served(status: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the boundary, and the one that keeps the product.

    Reading ``SERVABLE = {ACTIVE}`` as the *serve* gate rather than the
    *mint* gate deletes task recall and decision history. The allow-list
    is on the admission axis; the lifecycle axis belongs to the ranker.
    """
    assert is_admissible_status(status)
    assert POISON in _leg_graph(tmp_path, monkeypatch, status)


def test_the_withheld_set_is_derived_from_the_admission_table() -> None:
    """A new withheld ingest tier withholds its blocks with no edit here."""
    from mind_mem.admissibility import UNADMITTED
    from mind_mem.enums import INITIAL_STATUS, is_servable

    assert UNADMITTED == frozenset(s.value for s in INITIAL_STATUS.values() if s is not None and not is_servable(s))
    assert UNADMITTED == frozenset({"quarantined", "pending"})


def test_the_recognised_vocabulary_covers_every_source_it_claims() -> None:
    """Drift gate: a status added to one of these and not here is withheld.

    Silently withholding content because a neighbouring module grew a
    status is exactly the failure this whole change is meant to prevent
    in the other direction, so the assembled vocabulary is pinned to its
    sources rather than trusted.
    """
    from mind_mem._recall_constants import VALIDITY_STATUS_DEAD, VALIDITY_STATUS_WIP
    from mind_mem.admissibility import RECOGNISED_STATUSES, UNADMITTED
    from mind_mem.contradiction_detector import COMMITTED_STATUSES
    from mind_mem.enums import TaskStatus

    for source in (VALIDITY_STATUS_DEAD, VALIDITY_STATUS_WIP, COMMITTED_STATUSES, {m.value for m in TaskStatus}):
        missing = {str(s).lower() for s in source} - RECOGNISED_STATUSES - UNADMITTED
        assert not missing, f"status vocabulary drifted: {sorted(missing)}"


def test_active_is_the_only_status_a_governed_write_may_mint() -> None:
    """``SERVABLE`` keeps its step-1 meaning: the *mint* allow-list."""
    assert SERVABLE == frozenset({Status.ACTIVE})
    assert is_servable("active") and is_servable("Active ")
    assert not is_servable(None) and not is_servable(object())
    # ... and serving is the other axis, deliberately wider.
    assert is_admissible_status("superseded") and not is_servable("superseded")


def test_a_release_decision_admits_a_withheld_block(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``admissible := servable OR released`` — the second disjunct is live."""
    ws = _new_ws(tmp_path, "release")
    released = "IMP-20260829-001"
    _write(ws, "decisions/DECISIONS.md", _block(SEED, f"The {QUERY} decision", "active", Releases=released))
    _write(ws, "memory/IMPORTED.md", _block(released, f"{QUERY} imported note", "quarantined"))
    _config(ws)

    assert released in _served_ids(recall(ws, QUERY, limit=10))


def test_a_release_decision_cannot_resurrect_a_superseded_decision(tmp_path: Any) -> None:
    """Releases admit what a withheld ingest tier minted — nothing else."""
    from mind_mem.admissibility import release_ids

    blocks = [
        {"_id": SEED, "Status": "active", "Releases": ["IMP-a", "INBOX-b", "MSG-c", "D-20200101-001"]},
        {"_id": "D-superseded", "Status": "revoked", "Releases": ["IMP-z"]},
    ]
    assert release_ids(blocks) == frozenset({"IMP-a", "INBOX-b", "MSG-c"})


# ---------------------------------------------------------------------------
# T4 — rank invariance
# ---------------------------------------------------------------------------


def test_a_withheld_block_that_would_rank_first_changes_no_rank(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T4: served order is identical with and without the withheld block.

    RRF scores an item at ``sum 1/(k + rank_leg(i))``. Dropping a withheld
    item *after* fusion leaves every admitted item carrying a worse rank
    than it should — so the presence of withheld content is observable
    through its neighbours' ranks. Filtering before fusion closes that.
    """
    filler = [_block(f"D-2026082{n}-10{n}", f"{QUERY} note {n}", "active") for n in range(1, 5)]

    with_poison = str(tmp_path / "with")
    without = str(tmp_path / "without")
    for ws, poison in ((with_poison, True), (without, False)):
        os.makedirs(ws, exist_ok=True)
        for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
            os.makedirs(os.path.join(ws, d), exist_ok=True)
        _write(ws, "decisions/DECISIONS.md", "".join(filler))
        if poison:
            # Exact query text -> rank 1 on the lexical leg.
            _write(ws, "intelligence/CONTRADICTIONS.md", _block(POISON, QUERY, "quarantined"))
        _config(ws)
        build_index(ws)

    def _run(ws: str) -> list[str]:
        backend = HybridBackend(config={"vector_enabled": False})
        return _served_ids(backend.search(QUERY, ws, limit=10))

    assert _run(with_poison) == _run(without)


# ---------------------------------------------------------------------------
# T5 — golden: the served set may only shrink, and only by withheld ids
# ---------------------------------------------------------------------------


def test_the_served_set_only_shrinks_and_only_by_withheld_ids(tmp_path: Any) -> None:
    """T5: an all-active workspace serves the identical ordered set."""
    ws = str(tmp_path)
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    _write(ws, "decisions/DECISIONS.md", "".join(_block(f"D-2026082{n}-10{n}", f"{QUERY} note {n}", "active") for n in range(1, 6)))
    _config(ws)

    before = _served_ids(recall(ws, QUERY, limit=10))
    assert before, "fixture produced no hits — the golden would be vacuous"

    # Adding a withheld block must not perturb the answer at all.
    _write(ws, "memory/INBOX.md", _block("INBOX-20260829-001", QUERY, "quarantined"))
    after = _served_ids(recall(ws, QUERY, limit=10))

    assert after == before


# ---------------------------------------------------------------------------
# T6 — cost: no filesystem probe survives on an import-free workspace
# ---------------------------------------------------------------------------


class _Probed(BaseException):
    """Not an ``Exception``.

    The old filter wrapped its lookups in a bare ``except Exception`` and
    fell back to the status-only rule, so a plain ``AssertionError`` from
    a poisoned probe was swallowed and the cost test passed while the
    probe was still happening. Deriving from ``BaseException`` puts the
    signal outside every defensive handler on the path.
    """


@pytest.fixture()
def _import_free_ws(tmp_path: Any) -> Iterator[str]:
    ws = str(tmp_path)
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    _write(ws, "decisions/DECISIONS.md", _block(SEED, f"The {QUERY} decision", "active"))
    _config(ws)
    yield ws


def test_admissibility_probes_no_filesystem_on_an_import_free_workspace(_import_free_ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T6: the quarantine module's filesystem probes are gone from recall.

    Every probe the old rule made lived behind one of these entry points
    — ``_file_key``/``admitted_import_ids``/``quarantined_import_ids``
    each ``stat`` and then ``parse_file`` a corpus. Poisoning the module
    rather than ``os.stat`` is deliberate: the scan path legitimately
    stats and opens ``decisions/DECISIONS.md`` as a *corpus file*, so a
    path-targeted stat poison could not tell the two uses apart. What is
    provable is that admissibility no longer reaches into this module at
    all — and :func:`test_recall_never_imports_the_quarantine_module`
    proves the stronger version, that it is never even imported.
    """
    import mind_mem.importers.quarantine as q

    def _boom(*a: Any, **k: Any) -> Any:
        raise _Probed("recall probed the filesystem to decide admissibility")

    import mind_mem._recall_core as core
    import mind_mem.admissibility as adm

    for name in ("admitted_import_ids", "quarantined_import_ids"):
        monkeypatch.setattr(q, name, _boom)
    # The disk-backed release lookup is now THE probe. Recall must not
    # reach it while nothing in the corpus is withheld.
    monkeypatch.setattr(adm, "workspace_release_ids", _boom)
    monkeypatch.setattr(core, "workspace_release_ids", _boom)

    assert _served_ids(recall(_import_free_ws, QUERY, limit=10)) == [SEED]


def test_recall_never_imports_the_quarantine_module(_import_free_ws: str) -> None:
    """T6b: the admissibility decision has no import path to the importer."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "from mind_mem.recall import recall;"
        f"recall({_import_free_ws!r}, {QUERY!r}, limit=10);"
        "print('quarantine' if 'mind_mem.importers.quarantine' in sys.modules else 'clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False)
    assert out.stdout.strip().endswith("clean"), out.stdout + out.stderr


# ---------------------------------------------------------------------------
# T7 — an id the index knows and the corpus does not
# ---------------------------------------------------------------------------


def test_an_unresolvable_id_is_dropped_and_counted(tmp_path: Any) -> None:
    """T7: served subset-of resolved subset-of admissible. No exception.

    The KG is the one place an id can genuinely outrun the corpus:
    ``source_block_id`` comes out of the graph database, so an edge can
    name a block the corpus no longer answers for. (The cross-reference
    walk cannot produce one — ``build_xref_graph`` only emits edges
    between ids already in the list it was given — which is why the
    counter lives here and not there.)
    """
    from mind_mem.admissibility import unresolved_count
    from mind_mem.kg_fusion import kg_expand
    from mind_mem.knowledge_graph import KnowledgeGraph, Predicate, default_db_path

    ws = _new_ws(tmp_path, "unresolvable")
    db = default_db_path(ws)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    corpus = [{"_id": SEED, "Statement": QUERY, "Status": "active"}]
    before = unresolved_count()

    with KnowledgeGraph(db) as kg:
        kg.add_edge("pineapple", Predicate.RELATED_TO, "protocol", source_block_id="C-ghost-404")
        out = kg_expand([{"_id": SEED, "score": 1.0}], corpus, kg, QUERY, max_hops=1)

    assert _served_ids(out) == [SEED]
    assert unresolved_count() > before


# ---------------------------------------------------------------------------
# The deny-list is gone
# ---------------------------------------------------------------------------


def test_an_unstated_status_is_servable_however_it_is_spelled() -> None:
    """Absent, empty, and the bare ``Status:`` the parser renders as ``[]``."""
    assert is_admissible_status(None)
    assert is_admissible_status("")
    assert is_admissible_status("   ")
    assert is_admissible_status([])
    # A populated non-string is a status this code cannot read: withheld.
    assert not is_admissible_status(["quarantined"])
    assert not is_admissible_status(object())


def test_a_legacy_block_with_no_status_field_is_still_served(tmp_path: Any) -> None:
    """The realistic unstated case: a block written before the gate existed."""
    ws = _new_ws(tmp_path, "legacy")
    _write(ws, "decisions/DECISIONS.md", f"[{SEED}]\nStatement: The {QUERY} decision\nDate: 2026-08-29\n\n---\n\n")
    _config(ws)
    assert SEED in _served_ids(recall(ws, QUERY, limit=10))


def test_include_pending_survives_the_funnel_as_well_as_the_corpus_filter(tmp_path: Any) -> None:
    """A caller-scoped widening has to reach EVERY place the rule runs.

    Regression: the corpus filter honoured ``include_pending`` and the
    funnel every leg exits through did not, so the flag admitted the
    block and then the next filter took it straight back out. Two
    enforcement points mean two places a per-call exemption can be
    dropped, and only an end-to-end assertion catches the second.
    """
    ws = _new_ws(tmp_path, "pending")
    _write(ws, "intelligence/SIGNALS.md", _block("SIG-20260829-001", f"{QUERY} unreviewed signal", "pending"))
    _config(ws)

    assert _served_ids(recall(ws, QUERY, limit=10)) == []
    assert "SIG-20260829-001" in _served_ids(recall(ws, QUERY, limit=10, include_pending=True))


def test_the_scoring_path_has_no_import_edge_to_the_importer() -> None:
    """The by-construction form of the cost claim.

    A module that cannot import the importer cannot probe its files. This
    is a static check, so it holds for paths no test happens to exercise.
    """
    import ast

    root = pathlib.Path(mind_mem.__file__).parent
    for module in ("_recall_core.py", "admissibility.py", "hybrid_recall.py", "graph_recall.py", "kg_fusion.py"):
        imported: set[str] = set()
        for node in ast.walk(ast.parse((root / module).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        offenders = sorted(name for name in imported if "importer" in name or "ledger" in name)
        assert not offenders, f"{module} imports {offenders}"


def test_the_leg_vocabulary_matches_the_attestation_constants() -> None:
    """One name per leg. Two spellings would be two vocabularies."""
    from mind_mem import recall_attestation as ra

    assert ra.LEG_BM25 == Leg.BM25.value
    assert ra.LEG_VECTOR == Leg.VECTOR.value
    assert ra.LEG_GRAPH == Leg.GRAPH.value
    # ``hybrid`` is a fusion MODE, not a leg, so it has no Leg member.
    assert ra.LEG_HYBRID not in {leg.value for leg in Leg}


def test_the_withheld_id_set_helper_is_deleted() -> None:
    """Rule 2 dies with the second hardcoded corpus path it needed."""
    import mind_mem.importers as importers
    import mind_mem.importers.quarantine as q

    assert not hasattr(q, "withheld_import_ids")
    assert not hasattr(importers, "withheld_import_ids")
    assert "withheld_import_ids" not in q.__all__
    assert "withheld_import_ids" not in importers.__all__
