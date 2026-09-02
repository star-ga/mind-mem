# Copyright 2026 STARGA, Inc.
"""The write invariant: no block reaches the corpus without an admission.

mind-mem's product claim is that every write goes through the governance
gate and lands in an append-only hash chain. Before this file that was a
code-review convention: ``GovernanceGate.admit()`` had two callers while
``BlockStore.write_block`` had thirteen, so the drop-folder ingest and the
agent-message channel wrote ``Status: active`` blocks that were
immediately recallable, with no chain entry naming where they came from.

This file is the invariant. It is static — it reads the source on disk, so
a monkeypatch cannot satisfy it — and it fails the build on a NEW raw
writer rather than leaving the rule to a reviewer's memory.

Five scans:

A   every ``write_block`` CALL is in :data:`SANCTIONED_WRITE_BLOCK_CALLERS`
A2  every sanctioned caller actually runs inside an admission scope
B   every ``write_block`` IMPLEMENTATION calls ``require_admission``
C   the admission contextvar is unreachable outside ``governance_gate``
D   direct corpus-file appends (block minting that skips ``write_block``)

Three guards keep the scans honest, because a checker that silently
matches nothing reports a clean PASS over work it never inspected:
a corpus floor, a positive control on a known-present call site, and a
negative control that runs the matcher against synthetic rogue source.
"""

from __future__ import annotations

import ast

import pytest
from _write_path_scan import (
    ADMIT_OPENERS,
    CORPUS_BASENAMES,
    DELETE_ADMIT_OPENERS,
    corpus_basenames_from_source,
    find_write_block_calls,
    function_node,
    iter_source_files,
    opens_admission,
    parse,
    relpath,
    scan_contextvar_references,
    scan_corpus_writes,
    scan_write_block_calls,
    scan_write_block_defs,
    scan_write_block_status_binding,
)

# ---------------------------------------------------------------------------
# The allowlist. THIS IS THE INVARIANT — do not add an entry without the
# justification comment; a reviewer must be able to tell from this file
# alone why the write is safe.
# ---------------------------------------------------------------------------

#: The caller opens its own admission scope.
LOCAL = "local"

#: The caller IS a ``BlockStore.write_block`` (or a delegating adapter). It
#: enforces the receipt rather than issuing one; its scope belongs to
#: whoever called it.
IMPLEMENTATION = "implementation"

#: The caller opens its own DELETE scope. Distinct from :data:`LOCAL`
#: because the two are checked against different opener sets: a receipt is
#: not transferable between a write and a delete, so a corpus writer that
#: opened only ``admit_delete`` must not satisfy the write-side rule.
DELETE_LOCAL = "delete-local"

#: ``(file, enclosing qualname) -> LOCAL | IMPLEMENTATION | opener qualname``
SANCTIONED_WRITE_BLOCK_CALLERS: dict[tuple[str, str], str] = {
    # --- applying an already-approved proposal. One admission per
    # proposal, opened in _apply_proposal_locked upstream of execute_op
    # (its sole caller), so the ops below inherit it.
    ("src/mind_mem/apply_engine.py", "_op_append_block"): "_apply_proposal_locked",
    ("src/mind_mem/apply_engine.py", "_op_update_field"): "_apply_proposal_locked",
    ("src/mind_mem/apply_engine.py", "_op_append_list_item"): "_apply_proposal_locked",
    ("src/mind_mem/apply_engine.py", "_op_supersede_decision"): "_apply_proposal_locked",
    # --- bulk external ingest. Ungated per-block by design (a 10k-note
    # vault cannot become 10k proposals); bought back with
    # QUARANTINE_STATUS + a batch admission. The reference design.
    ("src/mind_mem/importers/engine.py", "_write_batch"): LOCAL,
    # --- the integrity scanner's own findings (GAP-1). Both detectors
    # funnel through ONE writer, so there is a single place a finding can
    # enter and it is inside an admit_batch scope. It sat in
    # PENDING_CORPUS_WRITERS as "lower severity, internally-derived
    # content"; provenance was never the axis. A `C-` block lands
    # `Status: open`, recall RECOGNISES that status, and the measured
    # result was a served block with all three ledgers at +0 -- reached
    # from `mind-mem-scan`, the third command of the README demo.
    # IngestTier.DETECTOR_FINDING is confined by enums.TIER_ID_PREFIXES to
    # C-/DREF- ids and to the one status it mints, so this scope cannot be
    # spent on anything but a finding.
    ("src/mind_mem/intel_scan.py", "_write_findings"): LOCAL,
    # --- the drop folder: untrusted input by construction. Same bargain
    # as the importer — batch admission, lands quarantined.
    ("src/mind_mem/inbox.py", "ingest_text_file"): LOCAL,
    ("src/mind_mem/inbox.py", "_ingest_pdf"): LOCAL,
    # --- the same drop folder, media half (v4.multi_modal, default OFF).
    # An image or audio drop carries no derivable text, so the operator's
    # SIDECAR file is the content; the media itself is only hashed. One
    # writer for both handlers, opening its own admit_block under
    # EXTERNAL_INGEST exactly as the text and PDF doors do, so a media
    # drop lands quarantined and unrecallable on the same terms.
    ("src/mind_mem/inbox.py", "_write_modal_block"): LOCAL,
    # --- the webhook drop door (`mm ingest-serve`, v4.ingest_serve, default
    # OFF). Anything POSTed to /ingest is untrusted external input, exactly
    # like a file dropped in the inbox, so the drain consumer admits every
    # event under IngestTier.EXTERNAL_INGEST -> Status.QUARANTINED before a
    # byte lands. This is the ONLY function in ingestion_pipeline.py that
    # touches a BlockStore -- every drain path funnels through it, pinned by
    # test_ingestion_pipeline_wiring.TestTheDrainPathIsTheGate.
    ("src/mind_mem/ingestion_pipeline.py", "_write_admitted"): LOCAL,
    # --- agent-to-agent messaging. Admitted per message under
    # IngestTier.AGENT_MESSAGE, which mints Status.QUARANTINED: a peer
    # agent is the standard prompt-injection carrier, so its text is not
    # recallable until a proposal releases it. `mm inbox` still shows it
    # (it enumerates the mailbox by name rather than searching memory).
    ("src/mind_mem/agent_messaging.py", "send_message"): LOCAL,
    # --- plumbing. No new block: Status is inherited from get_by_id, no
    # external content enters. One batch admission per run so a bulk
    # re-stamp of the corpus is not invisible.
    ("src/mind_mem/pipeline_hash.py", "reextract_dirty_blocks"): LOCAL,
    # --- operator-only backend copy. Status preserved from the source
    # corpus, so a quarantined block stays quarantined across backends.
    ("src/mind_mem/mm_cli.py", "_cmd_migrate_store"): LOCAL,
    # --- the A/B benchmark's memory arm. Seeds a THROWAWAY workspace with
    # this repository's own pre-cutoff commit history, under one
    # admit_proposal scope covering the whole seed, exactly as applying one
    # approved proposal covers the blocks it writes. Nothing external
    # enters: the content is the repo's own git log, and the workspace is
    # created and deleted by the harness. It writes ACTIVE blocks on
    # purpose -- a quarantined corpus is not recallable, so the memory arm
    # would silently become a second control arm and the benchmark would
    # report a null result it caused itself.
    ("src/mind_mem/bench/ab_seed.py", "write_seed"): LOCAL,
    # --- one operator approval of one staged relation signal. The signal
    # moves from `pending` (withheld) to `applied` (served), which is a mint
    # of servable content, so the scope is admit_proposal and the block goes
    # through the store like any other applied proposal. It used to be a
    # `re.subn` on SIGNALS.md with no scope at all -- measured: served after,
    # ledgers +0 -- because `write_block` REFUSED every SIG id until the
    # corpus table gave the prefix a row. See
    # tests/test_governed_signal_and_edge.py.
    ("src/mind_mem/graph_ingest.py", "approve_relation_signals"): LOCAL,
    # --- the enforcement point itself, and the three delegating adapters.
    ("src/mind_mem/block_store_postgres_replica.py", "ReplicatedPostgresBlockStore.write_block"): IMPLEMENTATION,
    ("src/mind_mem/storage/sharded_pg.py", "ShardedPostgresBlockStore.write_block"): IMPLEMENTATION,
    # The at-rest-encryption wrapper. Enforces the receipt itself before
    # it unseals the target file, then forwards to the inner store (which
    # enforces again); it mints nothing and opens no scope of its own.
    ("src/mind_mem/block_store_encrypted.py", "EncryptedBlockStore.write_block"): IMPLEMENTATION,
}

#: ``write_block`` definitions exempt from the ``require_admission`` rule.
#: Only the Protocol declaration, whose body is ``...``.
ENFORCEMENT_EXEMPT: frozenset[tuple[str, str]] = frozenset(
    {
        ("src/mind_mem/block_store.py", "BlockStore.write_block"),
    }
)

#: Functions that append block text straight into a corpus file, never
#: touching ``write_block``. A receipt on ``write_block`` cannot cover
#: these, so they are pinned here rather than left unbounded.
SANCTIONED_CORPUS_WRITERS: dict[tuple[str, str], str] = {
    # The sanctioned signal mint behind propose_update. Admits BEFORE the
    # bytes land (it used to admit after, inside a bare except that
    # swallowed the refusal, so a drifted spec blocked nothing).
    ("src/mind_mem/capture.py", "append_signals"): LOCAL,
    # The session summariser's dated artifact, summaries/daily/<date>.md.
    #
    # WHY IT IS ALLOWLISTED RATHER THAN PENDING. It is a DERIVED artifact --
    # topics, file paths and short excerpts computed from a Claude Code
    # transcript -- and its signal side already goes through
    # capture.append_signals (IngestTier.AUTO_CAPTURE -> Status: pending).
    # The summary half used to be the one leg with no admission at all, so
    # it now opens its own admit_block under the SAME tier before a byte
    # lands, and the block it writes declares `Status: pending` read off
    # INITIAL_STATUS rather than spelled as a literal.
    #
    # WHY THE SCANNER WOULD NEVER HAVE CAUGHT IT. `summaries/daily/*.md` is
    # a DATE-named file, so its basename is in neither CORPUS_BASENAMES nor
    # CORPUS_FILES: scan_corpus_writes cannot see this write, and recall
    # does not index the directory. Both facts are true today and neither is
    # a guarantee -- an ingest door whose safety rests on a filename pattern
    # and a directory list kept somewhere else is one refactor from being a
    # leak. Pinned here so the admission is enforced by
    # test_sanctioned_corpus_writers_open_an_admission rather than by that
    # coincidence.
    ("src/mind_mem/session_summarizer.py", "write_summary"): LOCAL,
    # The signal compaction sweep. It does not MINT into SIGNALS.md -- it
    # rewrites the file to drop aged signals -- so it is here as a
    # DELETE_LOCAL: the scanner flags it because it opens SIGNALS.md for
    # writing, and what it opens in return is admit_delete_batch, not a
    # write scope.
    #
    # It sat in PENDING_CORPUS_WRITERS until 5.0.2 with the upgrade path
    # "admit the compaction run", and the measured cost of the wait was
    # exactly what that entry implied: a sweep removed resolved/rejected
    # signals from a file in CORPUS_FILES -- content recall was serving --
    # and left ZERO evidence-chain rows and no deleted_blocks.jsonl entry.
    # One batch scope per sweep now: one authorisation, one removal record
    # with a Merkle root over what actually went. See
    # tests/test_governed_delete_compaction.py.
    ("src/mind_mem/compaction.py", "compact_signals"): DELETE_LOCAL,
}

#: Known-ungoverned corpus writers, pinned so the set cannot grow while
#: each carries its upgrade path. Lower severity than the ingest doors:
#: all three write internally-derived content, none is an external-input
#: channel. Not fixed in this change.
PENDING_CORPUS_WRITERS: frozenset[tuple[str, str]] = frozenset(
    {
        # deferred: appends non-block "## SKILL-..." prose into SIGNALS.md
        # — corpus pollution rather than a block mint. Upgrade path: stop
        # writing to SIGNALS.md and stage a proposal instead.
        ("src/mind_mem/skill_opt/validator.py", "submit_to_governance"),
        # Bench harness: builds a synthetic eval workspace, never the
        # operator's corpus. Pinned so it stays visible, not fixed.
        ("src/mind_mem/bench/eval_adapters.py", "MindMemAdapter.init"),
    }
)

#: Modules with zero production importers. Rerouting them would be wasted
#: work, so the allowlist says so out loud instead of implying they are
#: wired. Pinned by :func:`test_unwired_write_paths_stay_unwired`.
UNWIRED = (
    ("src/mind_mem/storage/sharded_pg.py", "ShardedPostgresBlockStore"),
    ("src/mind_mem/namespaces.py", "SharedLedger.append_fact"),
)

_REMEDY = """
Every block write must be admitted by GovernanceGate first. Either:

  (a) wrap the write in `with get_gate(ws).admit_block(...)` /
      `.admit_batch(...)` and add the entry to
      SANCTIONED_WRITE_BLOCK_CALLERS with a one-line justification; or

  (b) route the content through quarantine + release, exactly as
      importers/engine.py does — the correct answer for anything whose
      content did not originate inside this workspace.

This allowlist IS the invariant. Do not add an entry without the
justification comment.
"""


@pytest.fixture(scope="module")
def files() -> tuple[str, ...]:
    return iter_source_files()


# ---------------------------------------------------------------------------
# Guards — a scan that inspects nothing must not report success
# ---------------------------------------------------------------------------


def test_scanner_sees_the_whole_package(files: tuple[str, ...]) -> None:
    """A broken walk trips here before any allowlist comparison passes."""
    assert len(files) >= 250, f"only {len(files)} source files scanned; the walk is broken, not the tree"


def test_scanner_finds_a_known_call_site(files: tuple[str, ...]) -> None:
    """Positive control: a call site that is definitely present must appear."""
    found = {(rel, qual) for rel, qual, _line in scan_write_block_calls(files)}
    assert ("src/mind_mem/apply_engine.py", "_op_append_block") in found, "the write_block matcher stopped recognising a known call site"


def test_scanner_corpus_basenames_match_the_registry() -> None:
    """The scanner's hand-copied corpus list must equal the real registry.

    Blind-spot guard, not a style check. ``CORPUS_BASENAMES`` is what
    :func:`scan_corpus_writes` looks for, so a file in ``CORPUS_FILES``
    that is missing from it is a recallable corpus file no direct-writer
    scan ever examines — and the scan still reports a clean tree, which
    is the shape of a check that passes because it looked at nothing.

    Measured: ``INGEST.md`` entered ``CORPUS_FILES`` in 5.0.1 and was
    absent here until 5.0.2. Latent rather than exploited (the ``INGEST``
    prefix routes through ``write_block``), and latent only by luck.
    """
    derived = corpus_basenames_from_source()
    assert len(derived) >= 12, f"only {len(derived)} entries parsed out of CORPUS_FILES; the AST reader is broken, not the registry"
    missing = sorted(derived - CORPUS_BASENAMES)
    extra = sorted(CORPUS_BASENAMES - derived)
    assert not missing, f"CORPUS_FILES gained {missing}; the scanner never learned about it, so writers to those files go unscanned"
    assert not extra, f"the scanner scans {extra}, which CORPUS_FILES no longer registers"


def test_matcher_detects_a_synthetic_bypass() -> None:
    """Negative control: the matcher is run against known-bad source.

    Tree-independent. A matcher that silently matches nothing — the way a
    check reports PASS over work it never ran — cannot pass this.
    """
    rogue = "class Rogue:\n    def go(self, store, block):\n        store.write_block(block)\n"
    assert find_write_block_calls(ast.parse(rogue), "synthetic.py") == [("synthetic.py", "Rogue.go", 3)]


# ---------------------------------------------------------------------------
# Scan A / A2 — callers
# ---------------------------------------------------------------------------


def test_no_ungoverned_write_block_callers(files: tuple[str, ...]) -> None:
    unsanctioned = sorted(
        {(rel, qual, line) for rel, qual, line in scan_write_block_calls(files) if (rel, qual) not in SANCTIONED_WRITE_BLOCK_CALLERS}
    )
    if unsanctioned:
        listing = "\n".join(f"  {rel}:{line}  in  {qual}" for rel, qual, line in unsanctioned)
        pytest.fail(
            f"UNGOVERNED WRITE PATH — {len(unsanctioned)} call site(s) to BlockStore.write_block outside the sanctioned set:\n\n{listing}\n{_REMEDY}"  # noqa: E501 - failure message reads better unwrapped
        )


def test_every_sanctioned_caller_opens_an_admission(files: tuple[str, ...]) -> None:
    """An allowlist entry is a promise that a scope is actually opened."""
    by_file: dict[str, ast.Module] = {}
    for path in files:
        by_file[relpath(path)] = parse(path)

    failures: list[str] = []
    for (rel, qual), scope in sorted(SANCTIONED_WRITE_BLOCK_CALLERS.items()):
        if scope == IMPLEMENTATION:
            continue
        tree = by_file.get(rel)
        if tree is None:
            failures.append(f"  {rel} — allowlisted file is gone; prune the entry")
            continue
        opener_qual = qual if scope == LOCAL else scope
        func = function_node(tree, opener_qual)
        if func is None:
            failures.append(f"  {rel}:{opener_qual} — named opener does not exist")
            continue
        if not opens_admission(func):
            failures.append(f"  {rel}:{opener_qual} — writes blocks but opens no admission scope ({'/'.join(sorted(ADMIT_OPENERS))})")
    if failures:
        pytest.fail(
            "SANCTIONED BUT UNADMITTED — the allowlist claims these run under an admission; the source says otherwise:\n\n"
            + "\n".join(failures)
            + "\n"
            + _REMEDY
        )


# ---------------------------------------------------------------------------
# Scan B — the enforcement point
# ---------------------------------------------------------------------------


def test_every_write_block_implementation_requires_a_receipt(files: tuple[str, ...]) -> None:
    missing = [
        (rel, qual, line)
        for rel, qual, line, enforces in scan_write_block_defs(files)
        if not enforces and (rel, qual) not in ENFORCEMENT_EXEMPT
    ]
    if missing:
        listing = "\n".join(f"  {rel}:{line}  {qual}" for rel, qual, line in missing)
        pytest.fail(
            "UNENFORCED WRITE SURFACE — these write_block implementations accept a write with no open admission:\n\n"
            + listing
            + "\n\nEach must begin with require_admission(block_id) so a caller that\nforgot to open a scope raises UngatedWriteError instead of writing.\n"  # noqa: E501 - failure message reads better unwrapped
        )


def test_every_write_block_binds_the_block_status(files: tuple[str, ...]) -> None:
    """The receipt's ingest tier is unenforceable without the block's status.

    ``require_admission`` refuses a servable ``Status`` under a tier that
    cannot mint one — but only if the write surface hands it that status.
    A backend that forgets ``status=`` would keep its receipt check and
    silently lose the tier check, which is exactly the "remembered to
    quarantine" convention the tier table replaces.
    """
    missing = [
        (rel, qual, line)
        for rel, qual, line, binds in scan_write_block_status_binding(files)
        if not binds and (rel, qual) not in ENFORCEMENT_EXEMPT
    ]
    if missing:
        listing = "\n".join(f"  {rel}:{line}  {qual}" for rel, qual, line in missing)
        pytest.fail(
            "UNBOUND STATUS — these write_block implementations admit a write without\n"
            "telling the gate what status it carries, so their receipt's ingest tier\n"
            "cannot be enforced:\n\n" + listing + '\n\nEach must call require_admission(block_id, status=block.get("Status")).\n'
        )


def test_status_binding_scanner_detects_a_synthetic_omission() -> None:
    """Negative control: the binding matcher must reject the unbound form."""
    from _write_path_scan import binds_status_to_require_admission

    bound = ast.parse('def write_block(self, b):\n    require_admission(b["_id"], status=b.get("Status"))\n')
    unbound = ast.parse('def write_block(self, b):\n    require_admission(b["_id"])\n')
    assert binds_status_to_require_admission(bound.body[0]) is True
    assert binds_status_to_require_admission(unbound.body[0]) is False


def test_enforcement_exemptions_are_protocol_stubs_only(files: tuple[str, ...]) -> None:
    """An exemption may only cover a body that does nothing."""
    for rel, qual in sorted(ENFORCEMENT_EXEMPT):
        path = next(p for p in files if relpath(p) == rel)
        func = function_node(parse(path), qual)
        assert func is not None, f"{rel}:{qual} exemption points at nothing"
        body = [stmt for stmt in func.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
        assert all(isinstance(stmt, (ast.Expr, ast.Pass)) for stmt in body), (
            f"{rel}:{qual} is exempt from require_admission but has a real body"
        )


# ---------------------------------------------------------------------------
# Scan C — the contextvar must not be forgeable from elsewhere
# ---------------------------------------------------------------------------


#: The single module allowed to touch the admission contextvar. The receipt
#: machinery lives in its own module rather than in governance_gate, so that is
#: the owner; anywhere else setting this variable could mint a receipt for a
#: write the gate never admitted, which is precisely the forgery this whole
#: invariant exists to prevent.
ADMISSION_CONTEXTVAR_OWNER = "src/mind_mem/admission.py"


def test_admission_contextvar_is_private_to_the_gate(files: tuple[str, ...]) -> None:
    refs = scan_contextvar_references(files, "_active_admission")
    # Guard the guard: if the scanner finds nothing at all it is matching the
    # wrong symbol and would pass vacuously over an unprotected contextvar.
    assert refs, "scan_contextvar_references found no _active_admission references at all -- the scan is not looking at anything"
    outside = [(rel, line) for rel, line in refs if rel != ADMISSION_CONTEXTVAR_OWNER]
    assert not outside, (
        f"the admission contextvar is set or read outside {ADMISSION_CONTEXTVAR_OWNER}, which makes a receipt forgeable: {outside}"
    )


# ---------------------------------------------------------------------------
# Scan D — minting that never touches write_block
# ---------------------------------------------------------------------------


def test_no_unpinned_direct_corpus_writers(files: tuple[str, ...]) -> None:
    known = set(SANCTIONED_CORPUS_WRITERS) | PENDING_CORPUS_WRITERS
    unknown = sorted({(rel, qual, line, name) for rel, qual, line, name in scan_corpus_writes(files) if (rel, qual) not in known})
    if unknown:
        listing = "\n".join(f"  {rel}:{line}  in {qual}  -> {name}" for rel, qual, line, name in unknown)
        pytest.fail(
            "UNPINNED CORPUS WRITER — these append straight into a recallable corpus file, bypassing write_block entirely:\n\n"
            + listing
            + "\n"
            + _REMEDY
        )


def test_sanctioned_corpus_writers_open_an_admission(files: tuple[str, ...]) -> None:
    failures: list[str] = []
    for (rel, qual), scope in sorted(SANCTIONED_CORPUS_WRITERS.items()):
        path = next((p for p in files if relpath(p) == rel), None)
        assert path is not None, f"{rel} is allowlisted but absent"
        openers = DELETE_ADMIT_OPENERS if scope == DELETE_LOCAL else ADMIT_OPENERS
        func = function_node(parse(path), qual if scope in (LOCAL, DELETE_LOCAL) else scope)
        if func is None or not opens_admission(func, openers):
            kind = "delete" if scope == DELETE_LOCAL else "admission"
            failures.append(f"  {rel}:{qual} — rewrites the corpus but opens no {kind} scope")
    assert not failures, "SANCTIONED BUT UNADMITTED (direct corpus write):\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Honesty about dead code — an allowlist entry must not imply "wired"
# ---------------------------------------------------------------------------


def test_unwired_write_paths_stay_unwired(files: tuple[str, ...]) -> None:
    """The allowlist calls these unreachable; prove it rather than assert it.

    ``ShardedPostgresBlockStore`` is not in ``storage._SUPPORTED_BACKENDS``
    and ``SharedLedger.append_fact`` has no caller, so neither can carry
    external content into recall today. If that changes, the entry in
    :data:`UNWIRED` is a lie and this test says so.
    """
    for rel, symbol in UNWIRED:
        leaf = symbol.rsplit(".", 1)[-1]
        callers = [(other, line) for other, line in scan_contextvar_references(files, leaf) if other != rel]
        assert not callers, f"{rel}:{symbol} is documented as having no production importer, but is referenced from {callers}"
