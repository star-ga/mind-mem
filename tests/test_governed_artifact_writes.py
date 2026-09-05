# Copyright 2026 STARGA, Inc.
"""ROW-7: the derived artefacts are content, and content needs a receipt.

Three write doors landed served content with no admission and no chain
row, and none of them went near ``BlockStore.write_block`` — so none of
the five scans in ``tests/test_governed_write_paths.py`` could see them.
Measured on a fresh workspace at 0e57bcf, evidence chain and hash chain
each moving by **+0**::

    save_truth_page(ws, page)          evidence +0  hash_chain +0
      load_truth_page(ws, 'PRJ-x')     returns the text, USER scope
    add_block_edge(ws, a, b, 'contradicts')
                                       evidence +0  hash_chain +0
      block_lineage(ws, a)             returns the edge, USER scope
    CausalGraph(ws).add_edge(a, b, 'depends_on')
                                       evidence +0  hash_chain +0

Why the 5.0.2 edge governance did not cover them: it closed
``mcp.graph_add_edge`` / ``approve_edge`` /
``graph_ingest.approve_relation_signals``, which write the ``edges``
table under ``admit_edge``. These three write ``entities/compiled/*.md``,
``co_retrieval`` and ``causal_edges`` — different stores, different
doors, and ``entities/compiled/`` is in no registry at all.

WHY THE FIX IS ADMISSION AND NOT CORPUS REGISTRATION
----------------------------------------------------
The ruling offered "register ``entities/compiled/`` in the corpus
registry (or gate compiled_truth behind a flag until then)". Neither was
taken, and the reason is measurable rather than aesthetic:

* ``corpus_registry.discover_corpus_files`` and
  ``MarkdownBlockStore._discover_files`` both list corpus directories
  **one level deep and never descend** (the docstring says so, and
  :func:`test_the_compiled_store_is_not_reachable_from_the_corpus_walk`
  measures it). ``entities/compiled/`` is a subdirectory of ``entities``,
  so registering it means registering a *new corpus directory*, not
  ticking a box.
* I-14 made "what the store reads" equal "what recall serves". So the
  registration would put every compiled page into retrieval — and a
  compiled page is a *synthesis of blocks the corpus already holds*, so
  recall would return the synthesis alongside its own sources. That is a
  ranking change to a shipped product, measured by a benchmark this lane
  does not own, dressed up as a governance fix.
* H1 computes ``verify``'s ``unanchored_blocks`` row as corpus ids minus
  chain-landed ids. Registering the directory turns every page already on
  disk into an unanchored block, so ``verify --strict`` goes red on every
  real workspace until an ``mm anchor`` RESTAMP pass runs. None of that
  buys a single receipt.

**What the route actually taken costs H1: nothing, measured rather than
argued.** ``anchoring.unanchored_report`` over a fresh workspace, before
and after three governed ``save_truth_page`` calls::

    fresh init      unanchored=()  corpus_blocks=0  landed_blocks=0  close_records=0
    3 pages saved   unanchored=()  corpus_blocks=0  landed_blocks=3  close_records=3

Zero unanchored blocks either way, and ``verify(strict=True)``'s
``unanchored_blocks`` row is ``True`` in both. So no ``mm anchor``
migration is required by this change, and no existing workspace goes red.
The only rows that moved at all moved False -> True (``evidence_chain``,
``chain_head_seal``, ``cross_ledger``, ``open_scopes``), because the pages
now put records in the ledgers those rows walk.

Admission and corpus membership are orthogonal, and the product already
says so: a knowledge-graph edge is governed by ``admit_edge`` and is in
no corpus file, and ``session_summarizer.write_summary`` is admitted
while ``summaries/daily/`` is registered nowhere. The thesis is "no
content enters, leaves, or dies without a gate receipt and a chain
record" — it is not "everything is a corpus block". So this lane gives
the three doors receipts and chain rows and changes no retrieval
behaviour, and ``verify --strict`` is untouched on every existing
workspace (:func:`test_registering_the_compiled_store_is_not_required`).

THE SHAPE OF THE FIX
--------------------
Two layers, the same two ``write_block`` has:

1. **A seam that fails closed.** ``compiled_truth._write_compiled_page``,
   ``block_lineage._write_lineage_edge`` and
   ``causal_graph._write_causal_edge`` call ``require_admission`` before
   a byte moves. A future door that forgets the scope raises rather than
   writing.
2. **A static scan that fails the build.** ``_write_path_scan`` grew an
   ARTIFACT scan (scan E) that finds writers into the artefact stores and
   checks each one against :data:`SANCTIONED_ARTIFACT_WRITERS` — the same
   machinery, allowlist and honesty guards as scans A-D, not a parallel
   one.

WHAT IS NOT ADMITTED, AND WHY THAT IS NOT A HOLE
-------------------------------------------------
``retrieval_graph.log_retrieval`` writes ``co_retrieval`` rows too, on
every recall. It is not an assertion door: it cannot name a block the
recall it logs did not return, and it **cannot choose a kind** — its
INSERT names no ``kind`` column and its ON CONFLICT arm does not update
one. Admitting it would put one chain row per query in the ledger. That
argument is only available while the confinement holds, so
:class:`TestTheTelemetrySinkIsNotAnAssertionDoor` pins it.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from typing import Any, Iterator

import _write_path_scan
import pytest
from _write_path_scan import (
    ARTIFACT_DIRS,
    artifact_store_hit,
    artifact_stores_from_source,
    calls_require_admission,
    function_node,
    iter_source_files,
    opens_admission,
    parse,
    scan_artifact_writes,
)

from mind_mem.admission import UngatedWriteError
from mind_mem.block_lineage import add_block_edge, block_lineage, lineage_edge_id
from mind_mem.causal_graph import CausalGraph, causal_edge_id
from mind_mem.compiled_truth import (
    CompiledTruthPage,
    EvidenceEntry,
    _write_compiled_page,
    add_evidence,
    compiled_page_id,
    load_truth_page,
    save_truth_page,
)
from mind_mem.enums import INITIAL_STATUS, TIER_ID_PREFIXES, IngestTier
from mind_mem.governance_gate import (
    ARTIFACT,
    ARTIFACT_ID_PREFIXES,
    OPEN_SCOPE_TIERS,
    SCOPE_BOUND_TIERS,
    GovernanceBypassError,
    evict_gate,
    get_gate,
)
from mind_mem.init_workspace import init as init_workspace

#: ``(file, enclosing qualname) -> "local"``. Every writer into an
#: artefact store, and how it is governed. THIS IS THE INVARIANT — an
#: entry without a justification comment is not reviewable.
#:
#: ``LOCAL`` here means what it means in
#: ``tests/test_governed_write_paths.py``: the function opens its own
#: admission scope. ``SEAM`` means the function is the enforcement point
#: — it calls ``require_admission`` and issues nothing.
LOCAL = "local"
SEAM = "seam"

SANCTIONED_ARTIFACT_WRITERS: dict[tuple[str, str], str] = {
    # The compiled-truth page store. `save_truth_page` is the only
    # function in the product that writes entities/compiled/, and it
    # opens one admit_artifact scope per save keyed on compiled_page_id.
    ("src/mind_mem/compiled_truth.py", "save_truth_page"): LOCAL,
    # …and the seam it writes through, which is where the refusal lives.
    ("src/mind_mem/compiled_truth.py", "_write_compiled_page"): SEAM,
}


@pytest.fixture
def ws(tmp_path: Any) -> Iterator[str]:
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    init_workspace(workspace)
    get_gate(workspace)
    yield workspace
    evict_gate(workspace)


@pytest.fixture(scope="module")
def files() -> tuple[str, ...]:
    return iter_source_files()


def _evidence_rows(workspace: str) -> list[dict[str, Any]]:
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chain_rows(workspace: str) -> int:
    path = os.path.join(workspace, "memory", "hash_chain_v2.db")
    if not os.path.isfile(path):
        return 0
    conn = sqlite3.connect(path)
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        table = "entries" if "entries" in tables else tables[0]
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608 — name from sqlite_master
    finally:
        conn.close()


def _page(entity_id: str = "PRJ-demo", text: str = "mind-mem is a governed memory store.") -> CompiledTruthPage:
    return CompiledTruthPage(
        entity_id=entity_id,
        entity_type="project",
        compiled_section=text,
        evidence_entries=[],
        last_compiled="2026-09-04T00:00:00+00:00",
        version=1,
    )


# ---------------------------------------------------------------------------
# The doors now move both ledgers — the claim ROW-7 says they could not make
# ---------------------------------------------------------------------------


class TestTheThreeDoorsLandAReceipt:
    def test_save_truth_page_moves_both_ledgers(self, ws: str) -> None:
        before_ev, before_hc = len(_evidence_rows(ws)), _chain_rows(ws)
        save_truth_page(ws, _page())
        assert len(_evidence_rows(ws)) > before_ev, "a compiled page landed with no evidence-chain row"
        assert _chain_rows(ws) > before_hc, "a compiled page landed with no hash-chain row"

    def test_the_chain_row_names_the_page_and_hashes_it(self, ws: str) -> None:
        save_truth_page(ws, _page("PRJ-named"))
        rows = _evidence_rows(ws)
        subjects = [str(row.get("target_block_id", "")) for row in rows]
        assert compiled_page_id("PRJ-named") in subjects, f"no chain row names the page; got {subjects}"
        row = next(r for r in rows if r.get("target_block_id") == compiled_page_id("PRJ-named"))
        assert row.get("payload_hash"), "the chain row carries no digest of what was written"
        assert row.get("target_file") == "entities/compiled/PRJ-named.md", "the chain row does not say where the page went"

    def test_add_block_edge_moves_both_ledgers(self, ws: str) -> None:
        before_ev, before_hc = len(_evidence_rows(ws)), _chain_rows(ws)
        add_block_edge(ws, "D-20260101-001", "D-20260101-002", "contradicts")
        assert len(_evidence_rows(ws)) > before_ev, "a typed lineage edge landed with no evidence-chain row"
        assert _chain_rows(ws) > before_hc, "a typed lineage edge landed with no hash-chain row"
        subjects = [str(row.get("target_block_id", "")) for row in _evidence_rows(ws)]
        assert lineage_edge_id("D-20260101-001", "D-20260101-002", "contradicts") in subjects

    def test_causal_add_edge_moves_both_ledgers(self, ws: str) -> None:
        before_ev, before_hc = len(_evidence_rows(ws)), _chain_rows(ws)
        CausalGraph(ws).add_edge("D-20260101-001", "D-20260101-002", "depends_on")
        assert len(_evidence_rows(ws)) > before_ev, "a causal edge landed with no evidence-chain row"
        assert _chain_rows(ws) > before_hc, "a causal edge landed with no hash-chain row"
        subjects = [str(row.get("target_block_id", "")) for row in _evidence_rows(ws)]
        assert causal_edge_id("D-20260101-001", "D-20260101-002", "depends_on") in subjects


# ---------------------------------------------------------------------------
# …without losing the capability. Governing a feature must not disable it.
# ---------------------------------------------------------------------------


class TestTheCapabilitySurvives:
    def test_a_governed_page_still_round_trips(self, ws: str) -> None:
        save_truth_page(ws, _page(text="round trip"))
        loaded = load_truth_page(ws, "PRJ-demo")
        assert loaded is not None
        assert loaded.compiled_section == "round trip"

    def test_evidence_can_still_be_appended_and_resaved(self, ws: str) -> None:
        page = add_evidence(
            _page(),
            EvidenceEntry(
                timestamp="2026-09-04T12:00:00+00:00",
                source="memory/2026-09-04.md",
                observation="5.0.2 governs the compiled store.",
                confidence="high",
                superseded=False,
            ),
        )
        save_truth_page(ws, page)
        loaded = load_truth_page(ws, "PRJ-demo")
        assert loaded is not None and len(loaded.evidence_entries) == 1

    def test_a_lineage_edge_is_still_traversable(self, ws: str) -> None:
        add_block_edge(ws, "D-20260101-001", "D-20260101-002", "cites")
        result = block_lineage(ws, "D-20260101-001", max_depth=2)
        assert [e.block_id for e in result.edges] == ["D-20260101-002"]

    def test_re_asserting_an_edge_is_still_idempotent(self, ws: str) -> None:
        add_block_edge(ws, "D-20260101-001", "D-20260101-002", "cites")
        add_block_edge(ws, "D-20260101-001", "D-20260101-002", "cites")
        result = block_lineage(ws, "D-20260101-001", max_depth=2)
        assert len(result.edges) == 1, "the second assertion duplicated the edge instead of bumping it"

    def test_a_causal_chain_is_still_queryable(self, ws: str) -> None:
        graph = CausalGraph(ws)
        graph.add_edge("D-20260101-002", "D-20260101-001", "depends_on")
        assert [e.target_id for e in graph.dependencies("D-20260101-002")] == ["D-20260101-001"]


# ---------------------------------------------------------------------------
# The seams fail CLOSED — the property a scan cannot give you
# ---------------------------------------------------------------------------


class TestTheSeamsRefuseWithoutAReceipt:
    def test_writing_a_page_outside_a_scope_raises_and_writes_nothing(self, ws: str) -> None:
        target = os.path.join(ws, "entities", "compiled", "rogue.md")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with pytest.raises(UngatedWriteError):
            _write_compiled_page(target, compiled_page_id("rogue"), "unadmitted synthesis")
        assert not os.path.exists(target), "the refusal happened AFTER the bytes landed"

    def test_a_page_receipt_cannot_be_spent_on_another_page(self, ws: str) -> None:
        """The frozen id set, measured rather than asserted."""
        target = os.path.join(ws, "entities", "compiled", "pageB.md")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with pytest.raises(UngatedWriteError):
            with get_gate(ws).admit_artifact(compiled_page_id("pageA"), "a"):
                _write_compiled_page(target, compiled_page_id("pageB"), "b")
        assert not os.path.exists(target)

    def test_the_artefact_scope_refuses_a_corpus_subject(self, ws: str) -> None:
        """An artefact scope named after a block is a block scope in disguise."""
        with pytest.raises(GovernanceBypassError, match="artefact id namespaces"):
            with get_gate(ws).admit_artifact("D-20260101-001", "x"):
                pass  # pragma: no cover — the scope must not open

    def test_a_refused_scope_leaves_no_authorisation_row(self, ws: str) -> None:
        before = len(_evidence_rows(ws))
        with pytest.raises(GovernanceBypassError):
            with get_gate(ws).admit_artifact("D-20260101-001", "x"):
                pass  # pragma: no cover
        assert len(_evidence_rows(ws)) == before, "a subject the gate could not name still minted a record"


# ---------------------------------------------------------------------------
# The tier is confined by its SCOPE, exactly as EDGE_APPROVAL is
# ---------------------------------------------------------------------------


class TestTheArtefactTierIsBoundToItsScope:
    def test_the_tier_carries_and_mints_nothing(self) -> None:
        assert INITIAL_STATUS[IngestTier.DERIVED_ARTIFACT] is None

    def test_an_open_scope_cannot_name_the_tier(self, ws: str) -> None:
        """A carrying row constrains no status, so the scope is the whole rule."""
        assert SCOPE_BOUND_TIERS[ARTIFACT] is IngestTier.DERIVED_ARTIFACT
        assert IngestTier.DERIVED_ARTIFACT not in OPEN_SCOPE_TIERS
        tier = IngestTier.DERIVED_ARTIFACT
        with pytest.raises(GovernanceBypassError), get_gate(ws).admit_block("WRITE", "D-20260101-001", "x", tier=tier):
            pass  # pragma: no cover

    def test_no_artefact_namespace_is_routable_by_the_store(self) -> None:
        """The whole confinement argument, checked instead of asserted.

        An artefact receipt is safe on the block store because no id it
        can cover is an id ``write_block`` can be asked about. That holds
        only while the artefact prefixes name no corpus file.
        """
        from mind_mem.block_store import _BLOCK_PREFIX_MAP

        assert ARTIFACT_ID_PREFIXES, "no artefact namespace at all; the guard would pass over nothing"
        for prefix in ARTIFACT_ID_PREFIXES:
            assert prefix.rstrip("-") not in _BLOCK_PREFIX_MAP, (
                f"{prefix!r} routes to a corpus file, so an artefact receipt is spendable on a block"
            )

    def test_the_artefact_tier_is_not_confined_by_prefix_table(self) -> None:
        """Confinement is the scope's, not the prefix table's — stated once.

        ``TIER_ID_PREFIXES`` buys ``DETECTOR_FINDING`` the right to mint a
        recall-recognised status. An artefact mints no status at all, so
        it has nothing to buy and belongs in neither table; adding a row
        would also break the routability pin in
        ``test_governed_detector_writes.py``, which is the correct
        pressure — this tier writes no corpus file.
        """
        assert IngestTier.DERIVED_ARTIFACT not in TIER_ID_PREFIXES

    def test_the_id_helpers_mint_the_namespaces_the_gate_admits(self) -> None:
        for minted in (
            compiled_page_id("PRJ-x"),
            lineage_edge_id("a", "b", "cites"),
            causal_edge_id("a", "b", "depends_on"),
        ):
            assert minted.startswith(ARTIFACT_ID_PREFIXES), f"{minted!r} is in no namespace admit_artifact will admit"

    def test_the_id_helpers_are_deterministic_and_collision_free(self) -> None:
        assert lineage_edge_id("a", "b", "cites") == lineage_edge_id("a", "b", "cites")
        assert lineage_edge_id("a", "b", "cites") != lineage_edge_id("b", "a", "cites")
        assert lineage_edge_id("a", "b", "cites") != lineage_edge_id("a", "b", "refines")
        assert causal_edge_id("a", "b", "depends_on") != causal_edge_id("a", "b", "supersedes")


# ---------------------------------------------------------------------------
# Corpus registration was NOT the fix — the reasons, measured
# ---------------------------------------------------------------------------


class TestWhyTheCompiledStoreStaysOutOfTheRegistry:
    def test_the_compiled_store_is_not_reachable_from_the_corpus_walk(self, ws: str) -> None:
        """One level, never a walk — so registering it is a real change."""
        from mind_mem.corpus_registry import discover_corpus_files

        save_truth_page(ws, _page("PRJ-hidden"))
        found = {rel for _label, rel in discover_corpus_files(ws)}
        assert "entities/compiled/PRJ-hidden.md" not in found, (
            "the compiled store IS in the corpus walk, so this lane's reasoning about recall and unanchored_blocks is wrong"
        )

    def test_registering_the_compiled_store_is_not_required(self, ws: str) -> None:
        """A governed page is auditable while remaining outside the corpus.

        This is the whole design claim in one assertion: admission and
        corpus membership are orthogonal. The page has a chain row naming
        it and a digest of its bytes, and the store's read set — which
        I-14 made equal to what recall serves — never grew.
        """
        from mind_mem.block_store import MarkdownBlockStore

        before = {str(b.get("_id")) for b in MarkdownBlockStore(ws).get_all(active_only=False)}
        save_truth_page(ws, _page("PRJ-orthogonal"))
        after = {str(b.get("_id")) for b in MarkdownBlockStore(ws).get_all(active_only=False)}
        assert after == before, "governing the page changed what the store reads; recall results would move"
        subjects = [str(row.get("target_block_id", "")) for row in _evidence_rows(ws)]
        assert compiled_page_id("PRJ-orthogonal") in subjects


# ---------------------------------------------------------------------------
# The telemetry sink's exemption, pinned rather than asserted
# ---------------------------------------------------------------------------


class TestTheTelemetrySinkIsNotAnAssertionDoor:
    def test_log_retrieval_cannot_choose_a_kind(self, ws: str) -> None:
        """The property the exemption rests on. If it fails, admit the door.

        Positive control included: the row must EXIST before its kind
        means anything, which is how a "not found" assertion proves
        nothing.
        """
        from mind_mem.block_lineage import ensure_lineage_schema
        from mind_mem.retrieval_graph import graph_db_path, log_retrieval

        # The `kind` column is added by the lineage migration, so the
        # question "what kind did telemetry write" is only askable once
        # lineage has run. Applying it here is the positive control.
        ensure_lineage_schema(ws)
        log_retrieval(ws, "q", [{"_id": "D-20260101-001", "score": 1.0}, {"_id": "D-20260101-002", "score": 0.9}])
        conn = sqlite3.connect(graph_db_path(ws))
        try:
            rows = conn.execute("SELECT mem1_id, mem2_id, kind FROM co_retrieval").fetchall()
        finally:
            conn.close()
        assert rows, "the telemetry write did not happen, so its kind proves nothing"
        assert {row[2] for row in rows} == {"cooccurrence"}, f"log_retrieval wrote a typed edge: {rows}"

    def test_log_retrieval_cannot_retype_an_asserted_edge(self, ws: str) -> None:
        """A `contradicts` edge stays one; telemetry cannot downgrade it."""
        from mind_mem.retrieval_graph import graph_db_path, log_retrieval

        add_block_edge(ws, "D-20260101-001", "D-20260101-002", "contradicts")
        log_retrieval(ws, "q", [{"_id": "D-20260101-001", "score": 1.0}, {"_id": "D-20260101-002", "score": 0.9}])
        conn = sqlite3.connect(graph_db_path(ws))
        try:
            kinds = [
                row[0]
                for row in conn.execute(
                    "SELECT kind FROM co_retrieval WHERE mem1_id = ? AND mem2_id = ?", ("D-20260101-001", "D-20260101-002")
                )  # noqa: E501
            ]
        finally:
            conn.close()
        assert kinds == ["contradicts"], f"the telemetry sink rewrote an asserted edge's kind: {kinds}"

    def test_the_insert_names_no_kind_column(self) -> None:
        """Static twin of the two behavioural pins above.

        The behaviour tests could both pass over a future INSERT that
        names ``kind`` and happens to pass ``'cooccurrence'``. This one
        fails on the edit itself.
        """
        source = open(os.path.join(_write_path_scan.SRC_ROOT, "retrieval_graph.py"), encoding="utf-8").read()
        func = function_node(ast.parse(source), "log_retrieval")
        assert func is not None
        inserts = [
            node.value
            for node in ast.walk(func)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "INSERT INTO co_retrieval" in node.value
        ]
        assert inserts, "the co_retrieval INSERT is gone from log_retrieval; this guard now inspects nothing"
        for sql in inserts:
            assert "kind" not in sql, "log_retrieval names the kind column; it is an assertion door now and must be admitted"


# ---------------------------------------------------------------------------
# Scan E — a NEW ungoverned artefact door fails the build
# ---------------------------------------------------------------------------


class TestScanEIsHonest:
    """Guards first: a checker that inspects nothing must not report a pass."""

    def test_the_scanner_sees_the_whole_package(self, files: tuple[str, ...]) -> None:
        assert len(files) >= 250, f"only {len(files)} source files scanned; the walk is broken, not the tree"

    def test_the_artefact_stores_match_their_modules(self) -> None:
        """The scanner's hand-copied store list, re-derived from source.

        The blind-spot guard the corpus scans already have. A rename of
        ``compiled_truth._COMPILED_DIR`` that this set does not learn
        about is a store the scan silently stops looking at — and it
        still reports a clean tree.
        """
        derived = artifact_stores_from_source()
        assert derived, "the AST reader parsed no artefact store at all; it is broken, not the tree"
        missing = sorted(derived - ARTIFACT_DIRS)
        extra = sorted(ARTIFACT_DIRS - derived)
        assert not missing, f"the modules declare {missing}, which the scanner never learned about"
        assert not extra, f"the scanner scans {extra}, which no module declares"

    def test_the_scanner_finds_the_known_call_site(self, files: tuple[str, ...]) -> None:
        """Positive control on a write that is definitely present."""
        found = {(rel, qual) for rel, qual, _line in scan_artifact_writes(files)}
        assert ("src/mind_mem/compiled_truth.py", "_write_compiled_page") in found, (
            "the artefact-write matcher stopped recognising the compiled-page seam"
        )

    def test_the_matcher_detects_a_synthetic_bypass(self) -> None:
        """Negative control: rogue source the tree does not contain."""
        rogue = (
            "import os\n\n\nclass Rogue:\n    def go(self, ws):\n"
            "        p = os.path.join(ws, 'entities', 'compiled', 'x.md')\n"
            "        with open(p, 'w') as fh:\n            fh.write('mine')\n"
        )
        hits = _write_path_scan.find_artifact_writes(ast.parse(rogue), "synthetic.py", ARTIFACT_DIRS)
        found = [(rel, qual) for rel, qual, _line in hits]
        assert found == [("synthetic.py", "Rogue.go")], f"a rogue compiled-store writer was invisible: {hits}"

    def test_the_matcher_does_not_claim_a_sibling_store(self) -> None:
        """A scanner that cries wolf is one an allowlist grows to silence."""
        assert artifact_store_hit("\x00/entities/compiled/x.md", ARTIFACT_DIRS) == "entities/compiled"
        assert artifact_store_hit("\x00/entities/compiled-archive/x.md", ARTIFACT_DIRS) is None
        assert artifact_store_hit("\x00/entities/compiled/nested/x.md", ARTIFACT_DIRS) is None
        assert artifact_store_hit("\x00/shared/entities/compiled/x.md", ARTIFACT_DIRS) is None


class TestScanE:
    def test_every_artefact_writer_is_sanctioned(self, files: tuple[str, ...]) -> None:
        found = {(rel, qual) for rel, qual, _line in scan_artifact_writes(files)}
        rogue = sorted(found - set(SANCTIONED_ARTIFACT_WRITERS))
        assert not rogue, (
            f"ungoverned writers into an artefact store: {rogue}\n\n"
            "An artefact store holds content that is served straight back to an agent "
            "at USER scope. Open GovernanceGate.admit_artifact around the write and add "
            "the entry to SANCTIONED_ARTIFACT_WRITERS with a justification, or write "
            "outside the store."
        )

    def test_every_sanctioned_writer_is_actually_governed(self, files: tuple[str, ...]) -> None:
        """An allowlist entry promises a scope; this checks the promise."""
        for (rel, qual), how in sorted(SANCTIONED_ARTIFACT_WRITERS.items()):
            path = os.path.join(_write_path_scan.REPO_ROOT, rel.replace("/", os.sep))
            func = function_node(parse(path), qual)
            assert func is not None, f"{rel}::{qual} is in the allowlist and does not exist"
            if how == SEAM:
                assert calls_require_admission(func), f"{rel}::{qual} is pinned as the seam and never calls require_admission"
            else:
                assert opens_admission(func, frozenset({"admit_artifact"})), (
                    f"{rel}::{qual} is allowlisted as opening a scope and opens none"
                )

    def test_every_artefact_seam_requires_a_receipt(self) -> None:
        """The three seams, by name, checked in the source rather than run.

        The behavioural tests above prove each refusal once. This proves
        the refusal is still *written down* in all three, which is what a
        refactor removes silently.
        """
        seams = {
            "compiled_truth.py": "_write_compiled_page",
            "block_lineage.py": "_write_lineage_edge",
            "causal_graph.py": "_write_causal_edge",
        }
        for module, qual in seams.items():
            func = function_node(parse(os.path.join(_write_path_scan.SRC_ROOT, module)), qual)
            assert func is not None, f"{module}::{qual} has gone; the artefact seam has no enforcement point"
            assert calls_require_admission(func), f"{module}::{qual} writes an artefact without calling require_admission"

    def test_the_scan_is_wired_into_the_write_path_suite(self) -> None:
        """Scan E must be the same machinery, not a parallel one."""
        assert hasattr(_write_path_scan, "scan_artifact_writes")
        assert hasattr(_write_path_scan, "find_artifact_writes")
