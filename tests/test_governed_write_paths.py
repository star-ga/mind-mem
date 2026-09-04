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

import _write_path_scan
import pytest
from _write_path_scan import (
    ADMIT_OPENERS,
    CORPUS_BASENAMES,
    CORPUS_DIRS,
    DELETE_ADMIT_OPENERS,
    UNRESOLVED,
    corpus_basenames_from_source,
    corpus_dir_hit,
    corpus_dirs_from_source,
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
    # --- the dream cycle's auto-created entities (AUD-06). It used to
    # write entities/<PREFIX>-<slug>.md by hand, which was ungoverned AND
    # unresolvable: measured on a fresh workspace, parse_file found 0
    # blocks in the file, get_by_id answered None and both ledgers stayed
    # at +0, while list_blocks counted it -- so the RepairAction reported
    # `entity_created` for an id that did not exist and GET /status
    # counted a file resolving to nothing. Reached from POST /consolidate
    # {"auto_repair": true} and from the daemon. One admit_block per
    # entity under IngestTier.AUTO_CAPTURE, so the block lands `pending`
    # and recall withholds it until a release admits it.
    ("src/mind_mem/dream_cycle.py", "_create_entity_block"): LOCAL,
    # --- staged skill mutations (AUD-06). It used to append a
    # "## SKILL-<id>" heading to SIGNALS.md; the id it returned is stored
    # by `mm skill-optimize` as the mutation's governance handle and
    # resolved to NOTHING, because a markdown heading is not a block.
    # Same tier and same bargain as the dream cycle: AUTO_CAPTURE,
    # `pending`, withheld until released.
    ("src/mind_mem/skill_opt/validator.py", "submit_to_governance"): LOCAL,
    # --- the auto-capture door itself (R2-03). Twelve callers, the
    # `observe_signal` MCP tool among them. It used to hand-write the
    # `[SIG-...]` text into intelligence/SIGNALS.md with the scope as a
    # CONDITIONAL -- `gate.admit_batch(...) if _gate is not None else
    # nullcontext()`, over a `_get_gate` whose `except Exception: return
    # None` turned any gate failure into a no-op scope. Measured on a
    # fresh workspace with memory/hash_chain_v2.db replaced by a
    # directory: get_gate raised OperationalError, the block landed
    # anyway, both ledgers stayed at +0 and the call returned 1. The
    # allowlist claimed it opened an admission and the scanner agreed,
    # because `admit_batch` appeared textually; `opens_admission` now
    # refuses an opener that is one arm of a conditional expression.
    # One admit_batch per run under IngestTier.AUTO_CAPTURE, so signals
    # land `pending` and recall withholds them until a release admits
    # them, and nothing here can reach a corpus file any more because
    # nothing here opens one.
    ("src/mind_mem/capture.py", "append_signals"): LOCAL,
    # --- the eval harness's synthetic haystack. Not the operator's
    # corpus, which is why it was PENDING rather than a leak; but it spelled
    # `Status: active` straight into a file it opened itself, so the
    # benchmark was measuring a write path the product does not ship.
    # admit_proposal + write_block, exactly as bench/ab_seed does -- and
    # PROPOSAL_APPLY is the only tier that may mint the servable status a
    # retrieval haystack has to carry.
    ("src/mind_mem/bench/eval_adapters.py", "_seed_governed"): LOCAL,
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
    # of servable content, so it runs inside a scope -- admit_edge since
    # 5.0.2, naming the edge id it commits and the signal block id it
    # re-stamps, and nothing else. It was admit_proposal, which authorises
    # every id it is asked about at the tier that mints ACTIVE; one relation
    # approval never needed that reach. It used to be a
    # `re.subn` on SIGNALS.md with no scope at all -- measured: served after,
    # ledgers +0 -- because `write_block` REFUSED every SIG id until the
    # corpus table gave the prefix a row. See
    # tests/test_governed_signal_and_edge.py.
    ("src/mind_mem/graph_ingest.py", "approve_relation_signals"): LOCAL,
    # --- anchoring a pre-gate corpus (H1). `mm anchor --apply` re-writes
    # blocks that are ALREADY in the corpus and that no write scope ever
    # landed, so the chain has no record of them: measured on a fresh
    # workspace, a decision appended to DECISIONS.md by hand was indexed,
    # served by recall, and verified 7/7 green against 0 chain rows and 0
    # evidence rows. One admit_batch per pass under IngestTier.RESTAMP --
    # a CARRYING tier whose INITIAL_STATUS row is None, so it mints no
    # status: an `active` block stays `active` and this scope cannot be
    # spent to escalate anything. The block written is the one `get_by_id`
    # returned, never a synthesised one, and an id the store cannot
    # resolve is reported skipped rather than invented.
    ("src/mind_mem/anchoring.py", "restamp_unanchored"): LOCAL,
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
    # NOT HERE ANY MORE, and the absence is the point: ``capture.
    # append_signals``. It was the sanctioned signal mint behind
    # propose_update, allowlisted because it appended the ``[SIG-...]``
    # text to intelligence/SIGNALS.md itself. Being on this list is what
    # made its fail-open conditional scope survivable -- an entry here
    # only ever promises that SOME admission scope is opened, and a
    # scope opened in one arm of ``X if c else nullcontext()`` satisfied
    # that promise while writing the block regardless. It now mints
    # through ``BlockStore.write_block`` like every other governed door,
    # so it is a scan-A caller (see SANCTIONED_WRITE_BLOCK_CALLERS) and
    # scan D no longer has anything to see: the only ``open`` left in the
    # function is the read for dedup. An entry retired by fixing the
    # writer, not by re-justifying it.
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

#: Known-ungoverned writers to a NAMED corpus file. **Empty, and asserted
#: empty** by :func:`test_no_pending_corpus_writers_remain`.
#:
#: It held two entries until 5.0.2, and both are now closed rather than
#: re-justified:
#:
#: * ``skill_opt/validator.submit_to_governance`` appended a
#:   ``## SKILL-<mutation_id>`` heading to ``SIGNALS.md`` with
#:   ``open(..., "a")``. Its upgrade path here read "stop writing to
#:   SIGNALS.md and stage a proposal instead", and the measured cost of
#:   the wait was worse than "corpus pollution": the id it returned was
#:   recorded by ``mm skill-optimize`` as the mutation's
#:   ``governance_signal`` and resolved to NOTHING, because a markdown
#:   heading is not a block. It now mints a governed ``SIG-`` block at
#:   ``pending``.
#: * ``bench/eval_adapters.MindMemAdapter.init`` wrote its whole synthetic
#:   haystack with ``Status: active`` spelled into the text — a servable
#:   status minted with no admission. It now seeds through
#:   ``admit_proposal`` + ``write_block``, exactly as ``bench/ab_seed``
#:   does.
#:
#: The entry that is NOT here is the point: an empty allowlist is what
#: makes I-1 total at the corpus. Keep it empty. A new hole gets fixed,
#: not listed.
PENDING_CORPUS_WRITERS: frozenset[tuple[str, str]] = frozenset()

#: Ungoverned writers to a corpus DIRECTORY that no basename scan could
#: see, revealed by widening scan D in 5.0.2 (:func:`corpus_dir_hit`).
#:
#: EMPTY, and asserted empty by
#: :func:`test_no_pending_corpus_dir_writers_remain` — the same standard
#: :data:`PENDING_CORPUS_WRITERS` is held to. It was briefly non-empty
#: while two ``intel_scan`` reports were pinned rather than fixed:
#:
#: * ``write_impact`` -> ``intelligence/IMPACT.md``. Measured on a fresh
#:   workspace: it emitted ``[I-YYYYMMDD-###]`` headers, the store parsed
#:   them, ``get_by_id('I-20260902-001')`` RESOLVED and the block was in
#:   ``get_all`` — with the evidence chain at +0 and the hash chain at
#:   +0, one ``Status:`` field away from a servable ungoverned mint.
#: * ``generate_briefing`` -> ``intelligence/BRIEFINGS.md``. Its
#:   ``[2026-W36]`` heading did not parse as a block, so it was an inert
#:   report sitting inside the store's read set — and after I-14 widened
#:   recall to the store's read set, an inert report inside it is one
#:   well-formed heading away from being served.
#:
#: Both now write ``maintenance/derived/``
#: (``intel_scan.DERIVED_DIR``), which no corpus walk reaches. That is
#: the fix by construction: the next report written there cannot re-enter
#: the corpus however it is named, whereas a pinned entry only records
#: that this one did.
PENDING_CORPUS_DIR_WRITERS: frozenset[tuple[str, str]] = frozenset()

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


def test_scanner_corpus_dirs_match_the_registry() -> None:
    """The scanner's hand-copied corpus DIRECTORY list must equal the registry.

    The directory twin of
    :func:`test_scanner_corpus_basenames_match_the_registry`, and it guards
    a wider hole than that one does. ``CORPUS_BASENAMES`` going stale hides
    writers to ONE named file; ``CORPUS_DIRS`` going stale hides writers to
    EVERY ``.md`` in a whole directory, because
    ``MarkdownBlockStore._discover_files`` lists the directory rather than a
    fixed set of names.
    """
    derived = corpus_dirs_from_source()
    assert len(derived) >= 4, f"only {len(derived)} entries parsed out of CORPUS_DIRS; the AST reader is broken, not the registry"
    missing = sorted(derived - CORPUS_DIRS)
    extra = sorted(CORPUS_DIRS - derived)
    assert not missing, f"CORPUS_DIRS gained {missing}; the scanner never learned about it, so every .md written there goes unscanned"
    assert not extra, f"the scanner scans {extra}, which CORPUS_DIRS no longer registers"


def test_no_pending_corpus_writers_remain() -> None:
    """The known-holes list for NAMED corpus files is empty, and stays empty.

    Asserted rather than merely emptied. An allowlist that is empty today
    and unasserted is one commit from being non-empty again, and the whole
    value of I-1 at the corpus is that the allowlist of exceptions is
    *nothing* — a "mostly governed" corpus is an ungoverned one with better
    documentation.
    """
    assert PENDING_CORPUS_WRITERS == frozenset(), (
        f"PENDING_CORPUS_WRITERS is non-empty: {sorted(PENDING_CORPUS_WRITERS)}. "
        "Route the write through GovernanceGate + write_block instead of "
        "pinning it; see the closed entries in this file for two worked examples."
    )


def test_no_pending_corpus_dir_writers_remain() -> None:
    """The known-holes list for corpus DIRECTORIES is empty, and stays empty.

    The directory twin of :func:`test_no_pending_corpus_writers_remain`,
    and it became load-bearing when I-14 made the store's read set equal
    to what recall serves. While the two definitions disagreed, an
    ungoverned ``.md`` written into ``intelligence/`` was "only"
    ``get_by_id``-readable and export-visible. Now anything the store can
    read is also anything recall can serve, so a writer that mints
    unadmitted blocks into a corpus directory mints them straight into
    retrieval — which is why this list is asserted empty rather than
    curated.

    The remedy is never to add an entry. Write the artefact outside the
    corpus directories (``intel_scan.DERIVED_DIR``), or mint it through
    ``GovernanceGate.admit_batch`` + ``write_block`` with a row in
    ``corpus_registry.CORPUS_TABLE``.
    """
    assert PENDING_CORPUS_DIR_WRITERS == frozenset(), (
        f"PENDING_CORPUS_DIR_WRITERS is non-empty: {sorted(PENDING_CORPUS_DIR_WRITERS)}. "
        "A .md written directly into a corpus directory is read by the store AND "
        "served by recall (I-14). Move the file to maintenance/derived/ or admit "
        "the write; see intel_scan.DERIVED_DIR for a worked example."
    )


def test_corpus_dir_scan_sees_a_named_file_written_by_pattern() -> None:
    """Positive control for the widened scan D, on source it owns.

    The rule this pins is the one that made ``dream_cycle`` visible: a
    ``.md`` written directly into a corpus directory under a name no
    registry lists. Held as SYNTHETIC source rather than as an assertion
    about ``dream_cycle`` itself, because the real writer has since been
    fixed — a positive control that disappears the moment the bug is fixed
    is a control that stops controlling anything (the ``HEAD``-pinned
    baseline mistake).
    """
    rogue = (
        "import os\n\n\n"
        "def go(ws, slug):\n"
        "    d = os.path.join(ws, 'entities')\n"
        "    p = os.path.join(d, f'PRJ-{slug}.md')\n"
        "    with open(p, 'w', encoding='utf-8') as f:\n"
        "        f.write('x')\n"
    )
    tree = ast.parse(rogue)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _write_path_scan._write_target(n) is not None]
    assert calls, "the write-mode open matcher stopped recognising open(path, 'w')"
    env = _write_path_scan._assignments(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)))
    templates = _write_path_scan._path_templates(_write_path_scan._write_target(calls[0]), env, {})
    hits = {corpus_dir_hit(t, CORPUS_DIRS) for t in templates}
    assert "entities" in hits, f"the corpus-directory scan no longer sees entities/<name>.md; templates were {sorted(templates)!r}"


def test_corpus_dir_scan_does_not_claim_nested_or_sibling_paths() -> None:
    """Negative control WITH the positive half attached.

    ``corpus_dir_hit`` encodes ``_discover_files``, which calls
    ``os.listdir`` on each corpus directory — one level, never a walk. So a
    ``.md`` one level deeper, and a same-named directory under a different
    parent, are both outside the store's read set. Measured on this tree,
    these are the exact shapes that would otherwise be reported:
    ``intelligence/proposed/EDITS_PROPOSED.md`` (``lint_autofix``,
    ``importers/quarantine``), ``intelligence/applied/<ts>/APPLY_RECEIPT.md``
    (``apply_engine.rollback``) and ``shared/intelligence/LEDGER.md``
    (``namespaces``).

    The first assertion is the positive half: without it, a
    ``corpus_dir_hit`` that returned ``None`` for everything would pass the
    rest of this test while detecting nothing at all.
    """
    root = UNRESOLVED
    assert corpus_dir_hit(f"{root}/entities/PRJ-x.md", CORPUS_DIRS) == "entities"
    assert corpus_dir_hit(f"{root}/intelligence/proposed/EDITS_PROPOSED.md", CORPUS_DIRS) is None
    assert corpus_dir_hit(f"{root}/intelligence/applied/{UNRESOLVED}/APPLY_RECEIPT.md", CORPUS_DIRS) is None
    assert corpus_dir_hit(f"{root}/shared/intelligence/LEDGER.md", CORPUS_DIRS) is None
    assert corpus_dir_hit(f"{root}/intelligence/state/snapshots/S-x.json", CORPUS_DIRS) is None


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


def test_admission_scanner_rejects_a_conditional_scope() -> None:
    """Negative control WITH its positive half: a scope in an ``IfExp`` arm.

    Tree-independent, and held as synthetic source for the reason
    :func:`test_corpus_dir_scan_sees_a_named_file_written_by_pattern`
    gives: the real writer has been fixed, and a control that vanishes
    when the bug is fixed stops controlling anything.

    The first assertion is the positive half — without it a matcher that
    answered ``False`` for everything would pass the rest of this test
    while detecting nothing. The second is the shape that shipped in
    ``capture.append_signals`` and was reported as admitted; the third
    pins the blunt half of the rule (both arms openers still fails),
    which is deliberate: erring toward "ungated" is the only safe
    direction for a checker whose job is to refuse a fail-open path.
    """
    from _write_path_scan import conditional_calls

    unconditional = ast.parse(
        "def go(ws, store, block):\n    with get_gate(ws).admit_batch(action='WRITE'):\n        store.write_block(block)\n"
    ).body[0]
    fail_open = ast.parse(
        "def go(ws, store, block, gate):\n"
        "    scope = gate.admit_batch(action='WRITE') if gate is not None else nullcontext()\n"
        "    with scope, open('x', 'a') as f:\n"
        "        f.write('block')\n"
    ).body[0]
    both_arms = ast.parse(
        "def go(ws, store, block, one):\n"
        "    scope = one.admit_block(action='WRITE') if one else one.admit_batch(action='WRITE')\n"
        "    with scope:\n"
        "        store.write_block(block)\n"
    ).body[0]

    assert opens_admission(unconditional) is True, "the admission matcher stopped recognising an unconditional scope"
    assert opens_admission(fail_open) is False, (
        "an admission scope held in one arm of a conditional expression is reported as opened; "
        "that is the shape that let capture.append_signals write a block with a dead gate"
    )
    assert opens_admission(both_arms) is False, "the conditional rule must not depend on what the other arm holds"
    assert conditional_calls(unconditional) == set(), "conditional_calls flags a call that is in no conditional at all"
    assert conditional_calls(fail_open), "conditional_calls found no call inside the IfExp; the walk is broken"


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
    known = set(SANCTIONED_CORPUS_WRITERS) | PENDING_CORPUS_WRITERS | PENDING_CORPUS_DIR_WRITERS
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
