# Copyright 2026 STARGA, Inc.
""" "The corpus" gets one definition, and every surface derives from it.

Three tables claimed to say which files hold blocks, and they disagreed:

* ``corpus_registry.CORPUS_DIRS`` — what ``MarkdownBlockStore`` could
  *read* (``decisions``, ``tasks``, ``entities``, ``intelligence``);
* ``_recall_constants.CORPUS_FILES`` — what recall and the index
  *served*, including four files under ``memory/``; and
* ``block_store._BLOCK_PREFIX_MAP`` — where a write *landed*, also
  including those four.

So the four untrusted-ingest corpora (``INBOX``, ``MSG``, ``IMP``,
``INGEST`` — the inbox drop folder, agent-to-agent messages, migration
importers and the ingest webhook) were writable and servable and
unreadable. Measured on 5.0.1, with one released ``INBOX-`` block on
disk:

* ``recall`` / ``iter_active_blocks`` returned it;
* ``store.get_by_id`` returned ``None`` and ``store.get_all`` omitted it;
* ``DELETE /memories/INBOX-…`` answered ``404 block not found`` while the
  block sat on disk, because the door pre-checks ``get_by_id``;
* ``POST /clear`` answered ``{"ok": true, "deleted": 1}`` and left it
  behind — a partial purge reported as a whole one; and yet
* ``store.delete_block`` under an ``admit_delete`` scope removed it
  perfectly well.

The four corpora that hold *quarantined external input* — the ones an
operator most needs to be able to purge — were the undeletable ones. The
divergence is the defect; the 404 was a symptom, and a per-door patch
would have left the next door to rediscover it.

The fix is one table: :data:`corpus_registry.CORPUS_TABLE`. The prefix
map, the recall file map and the store's discovery are all derived from
it, so a corpus cannot be added to one surface and forgotten in the
others. This file is the conformance gate over that claim: for **every**
prefix the table routes, a block is written, read, served, counted and
destroyed through every door — and the pinned literal below proves the
derivation did not quietly change what 5.0.1 served.

:class:`TestMutationTwin` restores the pre-fix walk and the pre-fix
divergence and shows these tests going red.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import corpus_registry
from mind_mem._recall_constants import CORPUS_FILES
from mind_mem.block_parser import parse_file
from mind_mem.block_store import _BLOCK_PREFIX_MAP, MarkdownBlockStore, _resolve_block_file
from mind_mem.corpus_registry import BLOCK_PREFIX_MAP, CORPUS_DIRS, CORPUS_RELPATHS, CORPUS_TABLE
from mind_mem.governance_gate import PHASE_REMOVED, evict_gate, get_gate
from mind_mem.http_transport import _handle_clear, _handle_delete_memory
from mind_mem.storage import iter_active_blocks

#: What ``_recall_constants.CORPUS_FILES`` was, verbatim, in 5.0.1 — the
#: literal this module now derives. Pinned because the labels are written
#: onto every block as ``_source_label`` and branched on downstream, and
#: because the walk order decides the order equal-scoring blocks come back
#: in. A derivation that changes either is a behaviour change wearing a
#: refactor's clothes.
CORPUS_FILES_5_0_1: list[tuple[str, str]] = [
    ("decisions", "decisions/DECISIONS.md"),
    ("tasks", "tasks/TASKS.md"),
    ("projects", "entities/projects.md"),
    ("people", "entities/people.md"),
    ("tools", "entities/tools.md"),
    ("incidents", "entities/incidents.md"),
    ("contradictions", "intelligence/CONTRADICTIONS.md"),
    ("drift", "intelligence/DRIFT.md"),
    ("signals", "intelligence/SIGNALS.md"),
    ("messages", "memory/MESSAGES.md"),
    ("inbox", "memory/INBOX.md"),
    ("imported", "memory/IMPORTED.md"),
    ("ingest", "memory/INGEST.md"),
]

#: The prefixes the store could not read before the fix. Named so a
#: regression that drops ``memory/`` from discovery fails with the reason
#: rather than with a count.
UNREADABLE_BEFORE = ("INBOX", "MSG", "IMP", "INGEST")

#: Seed-id prefixes for table rows that carry no block-id prefix of their own
#: (``prefix=None`` — a corpus file recall serves that ``write_block`` cannot
#: route to). The fixture still has to put a block in such a file, so it
#: writes one by hand. Empty of consequence while every row carries a prefix;
#: present so adding a prefix-less row does not silently skip that corpus.
UNMAPPED_ROW_IDS: dict[str, str] = {"drift": "DRIFT", "signals": "SIG"}

#: A block in a ``memory/`` file the table does NOT name — a daily log.
#: ``memory/`` is not walked wholesale, so this must stay invisible: the
#: positive control for it is that ``parse_file`` reads it fine.
DAILY_LOG_REL = "memory/2026-09-02.md"
DAILY_LOG_ID = "D-20260902-777"

#: An ``.md`` sitting directly in a corpus directory under a name the
#: table does not list — an archive split out of ``DECISIONS.md``, an
#: ``entities/`` overflow file, a generated report. The store's walk takes
#: EVERY ``.md`` in a corpus directory, so these were readable by
#: ``get_by_id`` / ``get_all`` / ``export_memory`` / ``GET /memories`` and
#: invisible to recall, which iterated the table alone. Measured on
#: 5.0.2-pre with ``intelligence/BRIEFINGS.md``::
#:
#:     store.get_by_id            True   (control, a table file: True)
#:     store.get_all(active)      True   (control: True)
#:     iter_active_blocks         False  (control: True)
#:     recall(ws, ...)            absent (control: present)
#:
#: One id readable by every enumerating door and unreachable by the one
#: surface a memory product exists to provide.
UNNAMED_BASENAME = "ARCHIVE-2026.md"

#: One distinct id per corpus directory, so a test that passes for
#: ``decisions`` and silently skips ``intelligence`` cannot look green.
#: The ``D-`` prefix is deliberate: the prefix map routes it to
#: ``decisions/DECISIONS.md``, so the delete leg has to fall through to
#: the discovery walk to find it where it actually lives.
UNNAMED_IDS: dict[str, str] = {d: f"D-20260902-9{i:02d}" for i, d in enumerate(CORPUS_DIRS, start=1)}

#: A token that appears only in the unnamed-file blocks, so a recall hit
#: on it cannot come from a seeded table block.
UNNAMED_TOKEN = "zebranomicon"

CLEAR_BODY = {
    "rationale": "operator purge rehearsal for the release",
    "confirm": "yes-i-really-want-to-clear",
}


def _block(bid: str, statement: str, status: str = "active") -> str:
    return f"[{bid}]\nStatement: {statement}\nDate: 2026-09-02\nStatus: {status}\n\n---\n\n"


def _block_dict(bid: str, statement: str, status: str = "active") -> dict[str, Any]:
    return {"_id": bid, "Statement": statement, "Date": "2026-09-02", "Status": status}


def _ids(ws: str, *, active_only: bool = False) -> set[str]:
    return {str(b["_id"]) for b in MarkdownBlockStore(ws).get_all(active_only=active_only) if b.get("_id")}


def _served_ids(ws: str) -> set[str]:
    return {str(b["_id"]) for b in iter_active_blocks(ws) if b.get("_id")}


def _records(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _removed_records(ws: str) -> list[dict[str, Any]]:
    return [r for r in _records(ws) if r.get("metadata", {}).get("delete_phase") == PHASE_REMOVED]


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    """A workspace with one active block in every corpus the table names."""
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    for entry in CORPUS_TABLE:
        seed_prefix = entry.prefix or UNMAPPED_ROW_IDS.get(entry.label, entry.label.upper()[:4])
        bid = f"{seed_prefix}-20260902-001"
        (ws / entry.subdir / entry.filename).write_text(
            _block(bid, f"a block filed in {entry.label}"),
            encoding="utf-8",
        )
    (ws / DAILY_LOG_REL).write_text(_block(DAILY_LOG_ID, "a daily log entry, not a corpus block"), encoding="utf-8")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


# ---------------------------------------------------------------------------
# One definition: the three tables agree because two of them are derived
# ---------------------------------------------------------------------------


def _assert_tables_agree(prefix_map: dict[str, tuple[str, str]], file_map: dict[str, str]) -> None:
    """Every file a write routes to is a file recall scans.

    One direction only, deliberately: a corpus file with no prefix is
    legitimate (``DRIFT.md`` and ``SIGNALS.md`` are written by detectors
    that splice the file, and no id routes there — GAP-1/GAP-2), so
    "scanned but not routed" is allowed. "Routed but not scanned" is the
    hole: content that enters the corpus and is never served, or served
    from a file the store cannot read. The reverse containment — every
    scanned file is *readable by the store* — is
    :func:`test_every_table_row_is_reachable_by_one_of_the_two_discovery_halves`.

    Factored out so :class:`TestMutationTwin` can feed it a diverged pair
    and watch it fail — an agreement check that cannot fail is not one.
    """
    routed = {f"{subdir}/{filename}" for subdir, filename in prefix_map.values()}
    scanned = set(file_map.values())
    unscanned = sorted(routed - scanned)
    assert not unscanned, f"a write routes to {unscanned}, which recall never scans — content that enters and is never served"


def test_the_prefix_map_and_the_recall_file_map_agree() -> None:
    """The GAP-5 divergence, asserted as an invariant rather than fixed once."""
    _assert_tables_agree(_BLOCK_PREFIX_MAP, CORPUS_FILES)


def test_both_tables_are_derived_from_the_one_table() -> None:
    """Identity, not equality: there is no second literal to drift."""
    assert _BLOCK_PREFIX_MAP is BLOCK_PREFIX_MAP
    assert CORPUS_FILES is not None and list(CORPUS_FILES.items()) == [(e.label, e.relpath) for e in CORPUS_TABLE]
    assert BLOCK_PREFIX_MAP == {e.prefix: (e.subdir, e.filename) for e in CORPUS_TABLE if e.prefix}


def test_the_derivation_reproduces_what_5_0_1_served_exactly() -> None:
    """Labels and order are API; the derivation changed neither."""
    assert list(CORPUS_FILES.items()) == CORPUS_FILES_5_0_1


def test_the_mcp_hand_copy_of_the_prefix_map_still_agrees() -> None:
    """``mcp.tools.memory_ops`` keeps its own copy "in lockstep" by hand.

    Derivation cannot reach it from here without an MCP-layer import in
    the storage layer, so the lockstep is guarded rather than removed —
    and guarded whole, not three keys at a time.
    """
    from mind_mem.mcp.tools.memory_ops import _BLOCK_PREFIX_MAP as mcp_map

    assert mcp_map == _BLOCK_PREFIX_MAP, "the MCP door routes ids somewhere the store does not"


def test_every_table_row_is_reachable_by_one_of_the_two_discovery_halves(workspace: str) -> None:
    """CORPUS_FILES ⊆ discover_files(), with every file present on disk."""
    store = MarkdownBlockStore(workspace)
    discovered = {os.path.relpath(p, workspace).replace(os.sep, "/") for p in store.list_blocks()}
    missing = sorted(set(CORPUS_RELPATHS) - discovered)
    assert not missing, f"the store cannot read {missing}, which recall serves"


# ---------------------------------------------------------------------------
# The conformance run: write ⇒ read ⇒ serve ⇒ count ⇒ destroy, per prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", sorted(BLOCK_PREFIX_MAP))
def test_a_written_block_is_readable_servable_and_deletable(workspace: str, prefix: str) -> None:
    """One definition means one answer from every surface, for every prefix."""
    bid = f"{prefix}-20260902-050"
    store = MarkdownBlockStore(workspace)
    gate = get_gate(workspace)

    with gate.admit_proposal(proposal_id=f"P-{prefix}", content="[]"):
        store.write_block(_block_dict(bid, f"a governed {prefix} block"))

    target = _resolve_block_file(workspace, bid)
    assert target is not None and os.path.isfile(target), "the write did not land where the table says"

    fresh = MarkdownBlockStore(workspace)
    assert fresh.get_by_id(bid) is not None, f"{prefix} is writable and unreadable"
    assert bid in _ids(workspace), f"{prefix} is missing from get_all, so /clear cannot count it"
    assert bid in _served_ids(workspace), f"{prefix} is not served, so it entered without being retrievable"

    status, body = _handle_delete_memory(workspace, bid, actor="alice")
    assert status == 200, f"the HTTP door cannot destroy a {prefix} block: {body}"
    assert bid not in _ids(workspace)
    assert MarkdownBlockStore(workspace).get_by_id(bid) is None


@pytest.mark.parametrize("prefix", UNREADABLE_BEFORE)
def test_the_untrusted_ingest_corpora_are_the_ones_this_fixes(workspace: str, prefix: str) -> None:
    """Named separately: these four were readable by recall and by nothing else."""
    bid = f"{prefix}-20260902-001"
    subdir, filename = _BLOCK_PREFIX_MAP[prefix]
    assert subdir == "memory", f"{prefix} moved out of memory/; this test no longer covers the gap it names"
    assert subdir not in CORPUS_DIRS, "memory/ joined CORPUS_DIRS; the union is no longer what makes this work"

    store = MarkdownBlockStore(workspace)
    assert store.get_by_id(bid) is not None
    assert bid in _served_ids(workspace), "positive control: recall served it before the fix too"

    status, _body = _handle_delete_memory(workspace, bid, actor="alice")
    assert status == 200
    assert bid not in _ids(workspace)
    assert bid not in open(os.path.join(workspace, subdir, filename), encoding="utf-8").read()


def test_the_clear_door_counts_and_takes_the_memory_corpora(workspace: str) -> None:
    """A wipe that skipped the inbox was partial and reported as whole."""
    before = _ids(workspace)
    assert len(before) == len(CORPUS_TABLE), "the fixture seeded one distinct id per corpus row; a duplicate would hide a miscount"
    assert {f"{p}-20260902-001" for p in UNREADABLE_BEFORE} <= before, "positive control: the four are in the corpus to be taken"

    status, body = _handle_clear(workspace, CLEAR_BODY, actor="alice")

    assert status == 200
    assert body["deleted"] == len(before), f"the wipe left blocks behind: {sorted(_ids(workspace))}"
    assert _ids(workspace) == set()
    removed = _removed_records(workspace)
    assert len(removed) == 1, "one decision, one removal record"
    assert removed[0]["metadata"]["removed_count"] == len(before)


def test_a_delete_of_a_memory_corpus_block_is_recorded(workspace: str) -> None:
    """The death of an INBOX block reaches the chain like any other."""
    bid = "INBOX-20260902-001"
    _handle_delete_memory(workspace, bid, actor="alice")

    removed = _removed_records(workspace)
    assert len(removed) == 1
    assert removed[0]["metadata"]["removed_count"] == 1
    assert removed[0]["metadata"]["merkle_root"]


# ---------------------------------------------------------------------------
# Positive controls: what the definition still excludes, and why
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# I-14, the other half: the store's read set IS what recall serves
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_unnamed(workspace: str) -> str:
    """The conformance workspace plus one unnamed ``.md`` per corpus dir."""
    for subdir, bid in UNNAMED_IDS.items():
        path = os.path.join(workspace, subdir, UNNAMED_BASENAME)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_block(bid, f"a {UNNAMED_TOKEN} block filed in {subdir}/{UNNAMED_BASENAME}"))
    return workspace


def _discovery_relpaths(ws: str) -> set[str]:
    """Relpaths :func:`discover_corpus_files` yields that exist on disk.

    The existence filter is not a narrowing: the table half is yielded
    unconditionally (that is what makes it a drop-in for the old
    ``CORPUS_FILES.items()``, whose callers all ``isfile``-check), while
    the store's walk can only produce files that are there. Comparing the
    two therefore has to compare the same universe.
    """
    # Through the module attribute, not a from-import binding: the
    # mutation twin replaces the function, and a pin that kept its own
    # reference would keep passing while every consumer was mutated —
    # the shape of a check that cannot fail.
    rows = corpus_registry.discover_corpus_files(ws)
    return {rel for _label, rel in rows if os.path.isfile(os.path.join(ws, *rel.split("/")))}


def _store_relpaths(ws: str) -> set[str]:
    return {os.path.relpath(p, ws).replace(os.sep, "/") for p in MarkdownBlockStore(ws)._discover_files()}


def _assert_discovery_matches_the_store(ws: str) -> None:
    """What the store can READ equals what the retrieval legs discover.

    Factored out so :class:`TestMutationTwin` can drop the walk half from
    the shared function and watch this fail — an equality check that
    cannot fail is not one.
    """
    discovered = _discovery_relpaths(ws)
    store = _store_relpaths(ws)
    assert store, "the store discovered nothing; the comparison below would pass vacuously"
    unserved = sorted(store - discovered)
    unread = sorted(discovered - store)
    assert not unserved, f"the store reads {unserved}, which no retrieval leg discovers — readable, exportable, and never served"
    assert not unread, f"the retrieval legs discover {unread}, which the store cannot read — served, and invisible to every delete door"


def test_discovery_equals_what_the_store_reads(workspace_unnamed: str) -> None:
    """The pin. One discovery function, two consumers, no drift.

    The positive control is the ``unnamed`` assertion: the workspace holds
    files outside the table, so the equality is being tested on the case
    that used to diverge rather than on a table-only corpus where both
    halves trivially agree.
    """
    discovered = _discovery_relpaths(workspace_unnamed)
    unnamed = {rel for rel in discovered if rel.endswith("/" + UNNAMED_BASENAME)}
    assert len(unnamed) == len(CORPUS_DIRS), f"the fixture's unnamed files are not in the corpus at all: {sorted(discovered)}"
    assert not (unnamed & set(CORPUS_RELPATHS)), "the control files are table rows, so they prove nothing about unnamed ones"

    _assert_discovery_matches_the_store(workspace_unnamed)


@pytest.mark.parametrize("subdir", CORPUS_DIRS)
def test_an_unnamed_corpus_dir_block_is_readable_servable_and_deletable(workspace_unnamed: str, subdir: str) -> None:
    """readable ⇔ served-when-active ⇔ deletable, for every corpus dir.

    Parametrised over :data:`CORPUS_DIRS` rather than probing the one
    directory the audit happened to use: the claim is about the rule, and
    ``intelligence/`` and ``entities/`` were the unpinned halves.
    """
    bid = UNNAMED_IDS[subdir]
    rel = f"{subdir}/{UNNAMED_BASENAME}"
    path = os.path.join(workspace_unnamed, subdir, UNNAMED_BASENAME)

    parsed = parse_file(path)
    assert [b["_id"] for b in parsed] == [bid], "the control file is not parseable, so nothing below proves anything"
    assert rel not in CORPUS_RELPATHS, "the file joined the table; this test no longer covers the case it names"

    store = MarkdownBlockStore(workspace_unnamed)
    assert store.get_by_id(bid) is not None, f"{rel} is not readable"
    assert bid in _ids(workspace_unnamed), f"{rel} is missing from get_all, so /clear cannot count it"
    assert bid in _served_ids(workspace_unnamed), f"{rel} is readable and not served — the divergence I-14 forbids"

    status, body = _handle_delete_memory(workspace_unnamed, bid, actor="alice")
    assert status == 200, f"the HTTP door cannot destroy a block in {rel}: {body}"
    assert MarkdownBlockStore(workspace_unnamed).get_by_id(bid) is None
    assert bid not in open(path, encoding="utf-8").read()
    assert bid not in _served_ids(workspace_unnamed)


def test_recall_serves_a_block_from_an_unnamed_corpus_dir_file(workspace_unnamed: str) -> None:
    """The end-to-end half: the retrieval surface itself, not just the walk.

    ``iter_active_blocks`` and ``recall`` are different legs over the same
    definition, and the measured symptom was on ``recall``.
    """
    from mind_mem.recall import recall as run_recall

    hits = run_recall(workspace_unnamed, f"{UNNAMED_TOKEN} block filed", limit=50)
    served = {str(h.get("_id")) for h in hits if h.get("_id")}
    assert served, "recall returned nothing at all; a containment check against it would be vacuous"
    missing = sorted(set(UNNAMED_IDS.values()) - served)
    assert not missing, f"recall does not serve {missing}, which get_by_id resolves"


def test_the_clear_door_takes_the_unnamed_corpus_dir_blocks(workspace_unnamed: str) -> None:
    """A wipe that skipped an archive file was partial and reported as whole."""
    before = _ids(workspace_unnamed)
    assert set(UNNAMED_IDS.values()) <= before, "positive control: the unnamed blocks are in the corpus to be taken"

    status, body = _handle_clear(workspace_unnamed, CLEAR_BODY, actor="alice")

    assert status == 200
    assert body["deleted"] == len(before), f"the wipe left blocks behind: {sorted(_ids(workspace_unnamed))}"
    assert _ids(workspace_unnamed) == set()
    assert _served_ids(workspace_unnamed) == set()


def test_the_derived_reports_are_written_outside_every_corpus_directory(workspace: str) -> None:
    """``intel_scan``'s regenerated reports cannot land in the read set.

    Run, not asserted about: ``write_impact`` really emits
    ``[I-YYYYMMDD-###]`` block headers, and the measured defect was that
    the store parsed them out of ``intelligence/IMPACT.md`` — evidence
    chain +0, hash chain +0, ``get_by_id`` resolving. So the check is that
    a real call leaves the corpus read set unchanged and puts the report
    somewhere no corpus walk descends.
    """
    from mind_mem import intel_scan

    head = intel_scan.DERIVED_DIR.split("/", 1)[0]
    assert head not in CORPUS_DIRS, f"{intel_scan.DERIVED_DIR} is inside a corpus directory; the store would read it"
    for legacy in intel_scan.LEGACY_DERIVED.values():
        assert legacy.split("/", 1)[0] in CORPUS_DIRS, f"{legacy} is not a legacy in-corpus path; this test is checking nothing"

    before_files = _store_relpaths(workspace)
    before_ids = _ids(workspace)
    impacts = [{"decision": "D-20260902-001", "projects": ["PRJ-1"], "tasks": [], "incidents": []}]
    intel_scan.write_impact(impacts, workspace, intel_scan.IntelReport())

    written = intel_scan.derived_path(workspace, "IMPACT.md")
    assert os.path.isfile(written), "write_impact wrote nothing; the assertions below would pass vacuously"
    body = open(written, encoding="utf-8").read()
    assert "[I-" in body, "the report carries no block headers, so it never was the mint this test is about"

    minted = {str(b["_id"]) for b in parse_file(written) if b.get("_id")}
    assert minted, "the report parsed to zero blocks, so the containment checks below are vacuous"

    assert _store_relpaths(workspace) == before_files, "the report landed in a file the store reads"
    assert _ids(workspace) == before_ids, "an ungoverned I- block entered the corpus"
    assert not (minted & _served_ids(workspace)), "recall serves a block minted outside the gate"


def test_the_index_drops_an_unnamed_corpus_file_that_leaves_the_disk(workspace_unnamed: str) -> None:
    """Recovery has to be reachable, not just entry.

    Widening discovery to the directory walk creates a one-way door if
    only the entry side is wired: a table row that vanishes is still
    ENUMERATED (so ``_index_file`` sees a missing path and drops its
    blocks), but an unnamed file that vanishes simply stops being
    discovered — its ``blocks_fts`` rows would then serve a file that no
    longer exists, forever. ``_vanished_indexed_files`` sweeps
    ``file_state`` for exactly that, and this is the round trip: enter the
    index, leave the disk, leave the index.
    """
    from mind_mem import sqlite_index

    bid = UNNAMED_IDS["intelligence"]
    path = os.path.join(workspace_unnamed, "intelligence", UNNAMED_BASENAME)

    sqlite_index.build_index(workspace_unnamed, incremental=False)
    hits = sqlite_index.query_index(workspace_unnamed, UNNAMED_TOKEN, limit=50)
    assert bid in {str(h.get("_id")) for h in hits}, "positive control: the block never entered the index, so its removal proves nothing"

    os.remove(path)
    assert sqlite_index.is_stale(workspace_unnamed), "a vanished unnamed file did not mark the index stale, so nothing would rebuild"
    sqlite_index.build_index(workspace_unnamed, incremental=True)

    after = {str(h.get("_id")) for h in sqlite_index.query_index(workspace_unnamed, UNNAMED_TOKEN, limit=50)}
    assert bid not in after, "the index still serves a block whose file was deleted"
    assert set(UNNAMED_IDS.values()) - {bid} <= after, "the sweep took blocks from files that are still on disk"


def test_a_legacy_in_corpus_briefing_is_carried_over_not_reread(workspace: str) -> None:
    """The move keeps an existing workspace's history and stops feeding the hole.

    Positive control first: the legacy file is real and holds the marker.
    Then one scan writes the derived copy carrying that marker forward,
    and the legacy path is never written again.
    """
    from mind_mem import intel_scan

    legacy = intel_scan.legacy_derived_path(workspace, "BRIEFINGS.md")
    os.makedirs(os.path.dirname(legacy), exist_ok=True)
    with open(legacy, "w", encoding="utf-8") as fh:
        fh.write("# Intelligence Briefings\n\nB-2020-W01 an old briefing\n")
    legacy_bytes = open(legacy, "rb").read()
    assert b"B-2020-W01" in legacy_bytes, "the control file is empty; carrying it over would prove nothing"

    assert intel_scan.read_derived(workspace, "BRIEFINGS.md") == legacy_bytes.decode("utf-8")

    intel_scan.write_derived(workspace, "BRIEFINGS.md", legacy_bytes.decode("utf-8") + "\nB-2026-W36 a new briefing\n")

    derived = intel_scan.derived_path(workspace, "BRIEFINGS.md")
    carried = open(derived, encoding="utf-8").read()
    assert "B-2020-W01" in carried and "B-2026-W36" in carried, "the move lost the workspace's briefing history"
    assert open(legacy, "rb").read() == legacy_bytes, "the legacy in-corpus file was written to"
    assert intel_scan.read_derived(workspace, "BRIEFINGS.md") == carried, "the derived copy does not win once it exists"


def test_a_memory_file_the_table_does_not_name_stays_invisible(workspace: str) -> None:
    """``memory/`` is not walked wholesale — only the rows the table names.

    The positive control is the ``parse_file`` line: the daily log holds a
    well-formed block that the parser reads, so its absence from the store
    is the definition excluding it, not the reader failing on it.
    """
    log_path = os.path.join(workspace, *DAILY_LOG_REL.split("/"))
    parsed = parse_file(log_path)
    assert [b["_id"] for b in parsed] == [DAILY_LOG_ID], "the control file is not parseable, so excluding it proves nothing"

    store = MarkdownBlockStore(workspace)
    discovered = {os.path.relpath(p, workspace).replace(os.sep, "/") for p in store.list_blocks()}
    assert DAILY_LOG_REL not in discovered
    assert store.get_by_id(DAILY_LOG_ID) is None
    assert DAILY_LOG_ID not in _served_ids(workspace), "recall does not serve it either — the surfaces still agree"


def test_the_encrypted_store_inherits_the_definition(workspace: str) -> None:
    """The wrapper delegates discovery, so the fix reaches it for free.

    ``EncryptedBlockStore`` resolves every read through
    ``self.list_blocks()``, which is the Markdown store's. That is the
    point of fixing this at the definition instead of at a door: a store
    nobody edited answers correctly about the inbox corpus.
    """
    from mind_mem.block_store_encrypted import EncryptedBlockStore

    store = EncryptedBlockStore(workspace, passphrase="conformance-passphrase")
    assert store.get_by_id("INBOX-20260902-001") is not None
    assert "INBOX-20260902-001" in {str(b["_id"]) for b in store.get_all() if b.get("_id")}
    assert store.get_by_id(DAILY_LOG_ID) is None, "and it inherits the exclusions too"


def test_a_non_markdown_file_in_a_corpus_dir_is_still_excluded(workspace: str) -> None:
    """The union widened the file set, not the file *types*."""
    readme = os.path.join(workspace, "decisions", "README.txt")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(_block("D-20260902-888", "a block in a .txt file"))

    store = MarkdownBlockStore(workspace)
    assert readme not in store.list_blocks()
    assert store.get_by_id("D-20260902-888") is None


def test_an_explicit_corpus_dirs_argument_still_narrows(workspace: str) -> None:
    """The parameter is a narrowing; the default is the whole corpus."""
    narrowed = MarkdownBlockStore(workspace, corpus_dirs=("decisions", "tasks"))
    assert {os.path.basename(p) for p in narrowed.list_blocks()} == {"DECISIONS.md", "TASKS.md"}
    assert narrowed.get_by_id("INBOX-20260902-001") is None

    whole = MarkdownBlockStore(workspace)
    assert whole.get_by_id("INBOX-20260902-001") is not None, "positive control: the default sees what the narrowing hides"


def test_an_ungated_write_to_a_memory_corpus_still_raises(workspace: str) -> None:
    """Widening the read surface did not widen the write surface."""
    from mind_mem.admission import UngatedWriteError

    store = MarkdownBlockStore(workspace)
    with pytest.raises(UngatedWriteError):
        store.write_block(_block_dict("INBOX-20260902-099", "an ungated inbox write"))
    assert store.get_by_id("INBOX-20260902-099") is None


def test_an_ungated_delete_of_a_memory_corpus_block_still_raises(workspace: str) -> None:
    """Reachable is not unprotected: the same admission check covers it."""
    from mind_mem.admission import UngatedDeleteError

    store = MarkdownBlockStore(workspace)
    assert store.get_by_id("INBOX-20260902-001") is not None, "positive control: the block is here to be refused over"

    with pytest.raises(UngatedDeleteError):
        store.delete_block("INBOX-20260902-001")

    assert MarkdownBlockStore(workspace).get_by_id("INBOX-20260902-001") is not None
    assert _records(workspace) == []


# ---------------------------------------------------------------------------
# Mutation twins
# ---------------------------------------------------------------------------


class TestMutationTwin:
    """Re-diverge the tables and watch every claim above fail."""

    @pytest.mark.parametrize("prefix", UNREADABLE_BEFORE)
    def test_the_corpus_dirs_only_walk_reproduces_the_undeletable_corpora(
        self, workspace: str, monkeypatch: pytest.MonkeyPatch, prefix: str
    ) -> None:
        """The 5.0.1 discovery, run against the same corpus.

        Run for all four ``memory/`` corpora rather than the one that was
        probed by hand: the claim is that the untrusted-ingest corpora
        were undeletable, so all four have to reproduce it.
        """

        def dirs_only(self: MarkdownBlockStore) -> list[str]:
            files: list[str] = []
            for d in self._corpus_dirs:
                dir_path = os.path.join(self._workspace, d)
                if os.path.isdir(dir_path):
                    for fname in sorted(os.listdir(dir_path)):
                        if fname.endswith(".md"):
                            files.append(os.path.join(dir_path, fname))
            return files

        monkeypatch.setattr(MarkdownBlockStore, "_discover_files", dirs_only)

        bid = f"{prefix}-20260902-001"
        subdir, filename = _BLOCK_PREFIX_MAP[prefix]
        path = os.path.join(workspace, subdir, filename)
        assert bid in _served_ids(workspace), "recall still serves it — that half never depended on the store"
        assert MarkdownBlockStore(workspace).get_by_id(bid) is None, "the mutation did not reproduce the blind store"

        # The wipe is the half the blind walk breaks: it takes whatever
        # ``get_all`` returns, and the mutation is what ``get_all`` reads.
        status, body = _handle_clear(workspace, CLEAR_BODY, actor="alice")
        assert status == 200
        assert bid in open(path, encoding="utf-8").read(), "the partial purge, reported as a whole one"
        assert bid in _served_ids(workspace), "still served, after a wipe that reported success"

        # The by-id door is the half that reaches it anyway, because
        # ``delete_block`` resolves an id through the prefix map and not
        # through the discovery walk — which is what ``_handle_clear``'s
        # docstring always claimed. Until 5.0.2 the door contradicted it:
        # an existence pre-check read this blind store *before* the gate
        # and answered 404 for a block sitting in a file it could name.
        # Removing that probe is what makes this a 200.
        status, body = _handle_delete_memory(workspace, bid, actor="alice")
        assert status == 200, f"the by-id door could not reach a block the prefix map resolves: {body}"
        assert bid not in open(path, encoding="utf-8").read(), "200 and still there would be a different bug"

    def test_dropping_the_dir_walk_reproduces_the_readable_but_unserved_blocks(
        self, workspace_unnamed: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restore the table-only discovery and watch the divergence come back.

        The mutation is the pre-fix code: ``discover_corpus_files`` minus
        its ``CORPUS_DIRS`` walk, which is exactly what every retrieval
        leg used to do when it iterated ``CORPUS_FILES``. The store is
        untouched — it never used the shared function — so the two
        surfaces disagree again and the pin has to say so.

        Every module that bound the name is patched, and the list of
        modules is DISCOVERED rather than typed out: a new leg that
        imports the function is mutated automatically instead of quietly
        escaping the twin.
        """
        import sys

        def table_only(workspace: str, corpus_dirs: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
            return [(e.label, e.relpath) for e in CORPUS_TABLE]

        real = corpus_registry.discover_corpus_files
        patched: list[str] = []
        for mod in list(sys.modules.values()):
            name = getattr(mod, "__name__", "")
            if not name.startswith("mind_mem"):
                continue
            if getattr(mod, "discover_corpus_files", None) is real:
                monkeypatch.setattr(mod, "discover_corpus_files", table_only)
                patched.append(name)
        assert "mind_mem.corpus_registry" in patched, "the registry itself was not patched; the lazy importers still see the real walk"
        assert {"mind_mem._recall_core", "mind_mem.sqlite_index"} <= set(patched), (
            f"the recall and index legs were not mutated, so this twin proves nothing: patched={sorted(patched)}"
        )

        with pytest.raises(AssertionError, match="which no retrieval leg discovers"):
            _assert_discovery_matches_the_store(workspace_unnamed)

        for subdir, bid in UNNAMED_IDS.items():
            rel = f"{subdir}/{UNNAMED_BASENAME}"
            store = MarkdownBlockStore(workspace_unnamed)
            assert store.get_by_id(bid) is not None, f"the mutation blinded the store too; {rel} proves nothing"
            assert bid not in _served_ids(workspace_unnamed), f"{rel} is still served — the mutation did not reproduce the pre-fix walk"

    def test_the_dir_walk_half_is_what_makes_the_two_agree(self, workspace_unnamed: str) -> None:
        """Positive control for the twin: unmutated, the same call passes."""
        _assert_discovery_matches_the_store(workspace_unnamed)

    def test_a_prefix_routed_to_an_unscanned_file_fails_the_agreement_check(self) -> None:
        """Add a corpus to one table only — the shape the derivation forbids."""
        diverged = dict(_BLOCK_PREFIX_MAP)
        diverged["AUDIT"] = ("memory", "AUDIT_NOTES.md")

        with pytest.raises(AssertionError, match="which recall never scans"):
            _assert_tables_agree(diverged, dict(CORPUS_FILES))

        _assert_tables_agree(_BLOCK_PREFIX_MAP, dict(CORPUS_FILES))

    def test_dropping_a_row_from_the_recall_map_fails_the_agreement_check(self) -> None:
        """And the mirror image: a scanned file the writer forgot."""
        starved = {label: rel for label, rel in CORPUS_FILES.items() if label != "inbox"}

        with pytest.raises(AssertionError, match="memory/INBOX.md"):
            _assert_tables_agree(_BLOCK_PREFIX_MAP, starved)
