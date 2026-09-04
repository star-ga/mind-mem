# Copyright 2026 STARGA, Inc.
"""Who may open the scope that mints ACTIVE — an allowlist, not a convention.

``docs/GOVERNED_WRITES.md`` says the apply engine applying an approved
proposal is "the only path to ACTIVE". The tree did not say that: seven
functions opened ``GovernanceGate.admit_proposal``, and nothing structural
bounded the number. ``test_governed_write_paths`` checks that a sanctioned
``write_block`` caller opens **some** admission scope
(``_write_path_scan.opens_admission``); it has never asked *which*, so a new
opener of the broadest scope in the product was a code-review question.

**Why that scope is the one worth bounding.** ``admit_proposal`` is the only
scope that mints :attr:`~mind_mem.enums.IngestTier.PROPOSAL_APPLY`, the only
tier whose :data:`~mind_mem.enums.INITIAL_STATUS` row is ``ACTIVE``; and its
receipt is ``PROPOSAL``-kind, which
:meth:`~mind_mem.admission.AdmissionReceipt.authorizes` answers ``True`` for
**every** id it is asked about. So one opener is ambient authority to write
any block in the corpus at a status recall serves.

**Measured, on 5.0.2 at 2697baf** — with no proposal of that id anywhere::

    find_proposal(ws, "P-19990101-999")     -> (None, None)
    admit_proposal("P-19990101-999", "fabricated") + write_block(Status: active)
    block landed: True   served: True   hash-chain rows: 2

Two chain rows attesting to a review that never happened. The receipt is the
product's core claim, so a forgeable one is the worst defect it can have.
:class:`TestTheScopeIsAmbient` keeps that reproduction green and named,
because it is the fact this allowlist exists to bound — see that class for
what closing it by construction requires, which is a later slice.

Three scans, and the guards that keep them honest:

A   every ``admit_proposal`` call site is in :data:`ADMIT_PROPOSAL_OPENERS`
B   every allowlisted entry still exists (a stale entry weakens the rule)
C   exactly one PRODUCTION opener remains — the apply engine

Guards: a corpus floor (a scan over no files is not a pass), a positive
control on a known-present call site, and a negative control that runs the
matcher over synthetic rogue source and requires it to report the rogue.
"""

from __future__ import annotations

import ast
import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from _write_path_scan import find_scope_openers, iter_source_files, parse, relpath, scan_scope_openers

# ---------------------------------------------------------------------------
# The allowlist. THIS IS THE INVARIANT — no entry without its justification.
# ---------------------------------------------------------------------------

#: The scope this file bounds.
OPENER = "admit_proposal"

#: A shipped door, reachable by an operator, that lands corpus content.
PRODUCTION = "production"

#: Not reachable from a shipped door: a sandbox replay, or a benchmark
#: seeding its own throwaway workspace. Allowlisted, and counted apart, so
#: "how many production openers are there" stays a number scan C can pin.
NON_PRODUCTION = "non-production"

#: ``(file, enclosing qualname) -> PRODUCTION | NON_PRODUCTION``
ADMIT_PROPOSAL_OPENERS: dict[tuple[str, str], str] = {
    # --- THE path. One admission per approved proposal, opened upstream of
    # execute_op so every op the apply writes inherits it. This is the
    # opener docs/GOVERNED_WRITES.md describes, and scan C pins it as the
    # only production one.
    ("src/mind_mem/apply_engine.py", "_apply_proposal_locked"): PRODUCTION,
    # --- the review sandbox. `review_preview._replay` applies a proposal
    # against a COPY of the workspace to show an operator what it would do;
    # the copy is discarded. It opens the real scope because it runs the
    # real apply engine — a preview that took a different write path would
    # be previewing something other than the apply.
    ("src/mind_mem/review_preview.py", "_replay"): NON_PRODUCTION,
    # --- benchmark seeding. Both build a throwaway workspace whose haystack
    # has to be servable or recall retrieves nothing and the benchmark
    # measures zero. Neither is reachable from a shipped door; both are
    # named here rather than exempted so a third one cannot arrive quietly.
    # The narrower BENCH_SEED tier that would let these stop borrowing the
    # apply scope is the deferred half of this slice.
    ("src/mind_mem/bench/ab_seed.py", "write_seed"): NON_PRODUCTION,
    ("src/mind_mem/bench/eval_adapters.py", "_seed_governed"): NON_PRODUCTION,
}

#: Lower bound on the files a scan must have parsed. A matcher that walked
#: an empty tuple reports zero findings, which reads exactly like a clean
#: tree — so the corpus size is asserted before the findings are believed.
MIN_SOURCE_FILES = 200

_REMEDY = """
`admit_proposal` mints the only tier that reaches Status: active, and its
receipt authorises EVERY block id it is asked about. A new opener is
therefore a new way to publish servable content under an authorisation
record naming a proposal.

Either:

  (a) open a narrower scope — `admit_block` / `admit_batch` for ingest,
      `admit_edge` for a knowledge-graph edge; or

  (b) add the entry to ADMIT_PROPOSAL_OPENERS with a one-line justification
      saying why this caller is applying an approved proposal.

This allowlist IS the invariant. Do not add an entry without the comment.
"""


@pytest.fixture(scope="module")
def files() -> tuple[str, ...]:
    return iter_source_files()


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    from mind_mem.init_workspace import init

    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    yield ws


# ---------------------------------------------------------------------------
# Guards — a checker that inspected nothing is not a pass
# ---------------------------------------------------------------------------


class TestTheScanIsHonest:
    @pytest.mark.unit
    def test_the_scan_reads_a_real_corpus(self, files: tuple[str, ...]) -> None:
        assert len(files) >= MIN_SOURCE_FILES, f"only {len(files)} source files scanned; the findings below would mean nothing"

    @pytest.mark.unit
    def test_the_scanner_finds_the_known_opener(self, files: tuple[str, ...]) -> None:
        """Positive control on a call site that is certainly there."""
        found = {(rel, qual) for rel, qual, _ in scan_scope_openers(files, OPENER)}
        assert ("src/mind_mem/apply_engine.py", "_apply_proposal_locked") in found, (
            f"the scanner cannot see the apply engine's own opener; it is matching nothing. Found: {sorted(found)}"
        )

    @pytest.mark.unit
    def test_the_scanner_reports_a_rogue_opener(self, tmp_path: Path) -> None:
        """Negative control: the matcher must be able to FAIL.

        Synthetic source, so this cannot go quiet when the real tree is
        clean. Without it a broken matcher and a clean tree are the same
        green.
        """
        rogue = tmp_path / "rogue.py"
        rogue.write_text(
            "def sneak(ws, store, block):\n    with get_gate(ws).admit_proposal('P-forged', 'x'):\n        store.write_block(block)\n",
            encoding="utf-8",
        )
        hits = find_scope_openers(parse(str(rogue)), "rogue.py", OPENER)
        assert [(rel, qual) for rel, qual, _ in hits] == [("rogue.py", "sneak")]

    @pytest.mark.unit
    def test_the_scanner_ignores_a_different_scope(self, tmp_path: Path) -> None:
        """The matcher is keyed on the method name, not on 'admit' appearing."""
        benign = tmp_path / "benign.py"
        benign.write_text(
            "def fine(ws, store, block):\n"
            "    with get_gate(ws).admit_block(action='WRITE', block_id='D-1', content='x', tier=T):\n"
            "        store.write_block(block)\n",
            encoding="utf-8",
        )
        assert find_scope_openers(parse(str(benign)), "benign.py", OPENER) == []


# ---------------------------------------------------------------------------
# Scans A / B / C
# ---------------------------------------------------------------------------


class TestTheOpenersArePinned:
    @pytest.mark.unit
    def test_no_unlisted_admit_proposal_openers(self, files: tuple[str, ...]) -> None:
        """Scan A — the invariant. An unlisted opener fails the build."""
        unlisted = sorted(
            {(rel, qual, line) for rel, qual, line in scan_scope_openers(files, OPENER) if (rel, qual) not in ADMIT_PROPOSAL_OPENERS}
        )
        if unlisted:
            listing = "\n".join(f"  {rel}:{line}  in  {qual}" for rel, qual, line in unlisted)
            pytest.fail(
                f"UNLISTED PROPOSAL SCOPE — {len(unlisted)} call site(s) to admit_proposal outside the allowlist:\n\n{listing}\n{_REMEDY}"
            )

    @pytest.mark.unit
    def test_every_allowlisted_opener_still_exists(self, files: tuple[str, ...]) -> None:
        """Scan B — a stale entry is a hole the next reader inherits.

        An entry naming a function that has been renamed or deleted would
        silently permit a *different* function to take that name later.
        """
        found = {(rel, qual) for rel, qual, _ in scan_scope_openers(files, OPENER)}
        stale = sorted(entry for entry in ADMIT_PROPOSAL_OPENERS if entry not in found)
        assert not stale, f"allowlist entries that no longer open the scope: {stale}. Prune them."

    @pytest.mark.unit
    def test_exactly_one_production_opener_remains(self, files: tuple[str, ...]) -> None:
        """Scan C — the number docs/GOVERNED_WRITES.md claims, as a number.

        Three of the seven openers this slice found were the knowledge-graph
        doors, which now open ``admit_edge``. What is left in production is
        the apply engine, which is what the document has always said.
        """
        found = {(rel, qual) for rel, qual, _ in scan_scope_openers(files, OPENER)}
        production = sorted(entry for entry in found if ADMIT_PROPOSAL_OPENERS.get(entry) == PRODUCTION)
        assert production == [("src/mind_mem/apply_engine.py", "_apply_proposal_locked")], (
            f"the production openers of the ACTIVE-minting scope are {production}. "
            "docs/GOVERNED_WRITES.md says there is exactly one, and it is the apply engine."
        )

    @pytest.mark.unit
    def test_the_edge_doors_no_longer_open_it(self, files: tuple[str, ...]) -> None:
        """The three doors this slice migrated, named so a revert is loud."""
        found = {rel for rel, _, _ in scan_scope_openers(files, OPENER)}
        migrated = {"src/mind_mem/graph_ingest.py", "src/mind_mem/mcp/tools/graph.py"}
        assert not (found & migrated), (
            f"{sorted(found & migrated)} opened admit_proposal again. An edge needs admit_edge — "
            "a scope covering the edge id it writes, not every id in the corpus."
        )


# ---------------------------------------------------------------------------
# What the allowlist bounds — the reproduction, kept named
# ---------------------------------------------------------------------------


class TestTheScopeIsAmbient:
    """The measured defect the allowlist bounds, rather than closes.

    ``admit_proposal`` takes a proposal **id string** and never resolves
    it. That is documented and, for the apply engine, harmless: the engine
    has already loaded the proposal it is applying. It is what makes the
    scope forgeable for anyone else, and it is why "who may open it" is
    worth a build-failing allowlist while the resolution work is pending.

    **This test asserts the hole is still open.** It is not a blessing: it
    is the honest positive control for the allowlist, so nobody reads the
    scans above as proof that a receipt cannot be forged. Invert it in the
    slice that makes ``admit_proposal`` take a resolved proposal record —
    at that point the call below must raise, and this class's name is what
    tells the next reader where to look.
    """

    @pytest.mark.unit
    def test_a_proposal_id_that_resolves_to_nothing_still_opens_the_scope(self, workspace: str) -> None:
        from mind_mem.apply_engine import find_proposal
        from mind_mem.governance_gate import get_gate
        from mind_mem.storage import get_block_store

        forged = "P-19990101-999"
        assert find_proposal(workspace, forged) == (None, None), "the fixture has a proposal by this id; the probe would prove nothing"

        store = get_block_store(workspace)
        with get_gate(workspace).admit_proposal(forged, "fabricated"):
            store.write_block({"_id": "D-19990101-999", "Statement": "forged", "Status": "active", "Type": "Decision"})

        landed = store.get_by_id("D-19990101-999")
        assert landed is not None and landed["Status"] == "active", (
            "the reproduction no longer reproduces; re-measure before editing this file"
        )

        with sqlite3.connect(os.path.join(workspace, "memory", "hash_chain_v2.db")) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM hash_chain").fetchone()[0]
        assert rows == 2, f"expected the open+close pair the forged scope writes, got {rows}"

    @pytest.mark.unit
    def test_the_receipt_it_mints_covers_every_id(self, workspace: str) -> None:
        """Why one opener matters: the receipt is not bounded by its subject."""
        from mind_mem.admission import PROPOSAL
        from mind_mem.governance_gate import get_gate

        with get_gate(workspace).admit_proposal("P-19990101-998", "[]") as receipt:
            assert receipt.kind == PROPOSAL
            assert receipt.covers == frozenset(), "a proposal receipt names no ids"
            assert receipt.authorizes("D-19990101-001")
            assert receipt.authorizes("anything-at-all")


# ---------------------------------------------------------------------------
# The scans read source, not a runtime — pinned, because it is the point
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_scan_reads_the_source_on_disk(files: tuple[str, ...]) -> None:
    """A monkeypatched gate cannot satisfy an allowlist read off disk."""
    sample = next(p for p in files if relpath(p) == "src/mind_mem/apply_engine.py")
    assert isinstance(parse(sample), ast.Module)
    assert os.path.isfile(sample)
