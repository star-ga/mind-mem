"""Central corpus path registry for mind-mem.

All modules should import corpus paths from here instead of hardcoding them.
This is the single source of truth for which directories constitute the corpus.
"""

from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatch
from typing import NamedTuple

# Core corpus directories (order matters for scan priority)
CORPUS_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "intelligence",
)

# Directories that contain active memory blocks
MEMORY_DIRS: tuple[str, ...] = CORPUS_DIRS

# Directories included in backup snapshots
BACKUP_DIRS: tuple[str, ...] = CORPUS_DIRS + ("memory", "summaries", "shared", "agents")

# Directories included in apply-engine rollback snapshots.
#
# v3.2.0 §2.2 atomicity-scope fix: ``maintenance/tracked/`` joins the
# snapshot so behavioural state files (dedup-state.json, compaction
# checkpoints, anything whose presence/absence changes the next
# apply's outcome) are captured + restored atomically with the
# corpus. Append-only files (validation-report.txt, noisy logs) keep
# their prior home under ``maintenance/append-only/`` and are still
# snapshot-excluded so they don't bloat the archive.
SNAPSHOT_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "summaries",
    "memory",
    "maintenance/tracked",
)

# ``maintenance/`` directories that are EXPLICITLY excluded from
# snapshots — append-only outputs where rollback would lose real
# signal. Anything under a path listed here is guaranteed to skip
# both the snapshot capture and the restore walk.
SNAPSHOT_EXCLUDE_DIRS: tuple[str, ...] = (
    "maintenance/append-only",
    "intelligence/applied",
)

# Directories used for cross-reference validation
VALIDATE_DIRS: tuple[str, ...] = (
    "decisions",
    "tasks",
    "entities",
    "summaries",
)

# File extensions recognized as block files
BLOCK_EXTENSIONS: tuple[str, ...] = (".md",)

# ─── The one corpus definition (I-14) ─────────────────────────────────────────
#
# "The corpus" had three definitions that disagreed, and the disagreement was
# the defect. ``CORPUS_DIRS`` said what the store could *read*;
# ``_recall_constants.CORPUS_FILES`` said what recall *served*;
# ``block_store._BLOCK_PREFIX_MAP`` said where a write *landed*. The four
# untrusted-ingest corpora under ``memory/`` (inbox, agent messages, importer
# output, the ingest webhook) were in the last two and absent from the first,
# so a released ``INBOX-`` block was served by recall, invisible to
# ``get_by_id``/``get_all``, and answered ``404 block not found`` by
# ``DELETE /memories/{id}`` — while the store's own ``delete_block`` under a
# scope removed it perfectly well. An id every reader can see and no door can
# destroy is an undeletable record.
#
# One table now, three derived views. A new corpus is one row: the write
# router, the store's read discovery and the recall/index scan all learn about
# it at once, and there is no second place to forget it.
#
# Row order is load-bearing and matches what ``CORPUS_FILES`` shipped in
# 5.0.1: the recall and enumeration walks iterate it in order, so reordering
# rows reorders equal-scoring results. ``tests/test_one_corpus_definition.py``
# pins the derived mapping against that literal.


class CorpusFile(NamedTuple):
    """One corpus file: its recall label, its id prefix, its location.

    ``prefix`` is ``None`` for a corpus file no block id routes to — the
    file is read and served, but nothing can be *written* to it through
    ``write_block`` because no prefix resolves there. Those rows are why
    the prefix map is a projection of this table and not the table
    itself.
    """

    label: str
    prefix: str | None
    subdir: str
    filename: str

    @property
    def relpath(self) -> str:
        """POSIX-relative path inside the workspace."""
        return f"{self.subdir}/{self.filename}"


CORPUS_TABLE: tuple[CorpusFile, ...] = (
    CorpusFile("decisions", "D", "decisions", "DECISIONS.md"),
    CorpusFile("tasks", "T", "tasks", "TASKS.md"),
    CorpusFile("projects", "PRJ", "entities", "projects.md"),
    CorpusFile("people", "PER", "entities", "people.md"),
    CorpusFile("tools", "TOOL", "entities", "tools.md"),
    CorpusFile("incidents", "INC", "entities", "incidents.md"),
    CorpusFile("contradictions", "C", "intelligence", "CONTRADICTIONS.md"),
    # Drift findings. The prefix landed in 5.0.2 (GAP-1): the row was here
    # with ``prefix=None``, so ``write_block`` REFUSED every ``DREF`` id
    # while recall served the file — which is why ``intel_scan.write_drift``
    # appended the block by hand and left all three ledgers at +0. Ids are
    # ``DREF-YYYYMMDD-###`` (not ``DRIFT-``); the prefix is what the
    # detector has always written.
    CorpusFile("drift", "DREF", "intelligence", "DRIFT.md"),
    # Captured signals. The prefix landed in 5.0.2 (GAP-2): the row was here
    # with ``prefix=None``, so ``write_block`` REFUSED every ``SIG`` id while
    # recall served the file — which is why every signal writer spliced
    # SIGNALS.md by hand, the relation-signal approval's regex ``Status:``
    # rewrite (a served status minted outside the gate) included.
    CorpusFile("signals", "SIG", "intelligence", "SIGNALS.md"),
    # v4.0.19: agent-to-agent messaging (`mm send` / `mm inbox`). Scanned so
    # `mm inbox` (which is recall) finds them on the SQLite default, at
    # parity with Postgres.
    CorpusFile("messages", "MSG", "memory", "MESSAGES.md"),
    # v3.9: inbox folder ingestion (text + PDF). Scanning this file is what
    # fixed the pre-existing inbox-invisible-on-SQLite bug — INBOX- blocks
    # were written here and never indexed, so recall returned 0.
    CorpusFile("inbox", "INBOX", "memory", "INBOX.md"),
    # Migration importers (roadmap Group G) (`mm import --from ...`). Scanned
    # so imported content is recallable immediately, at parity with
    # INBOX-/MSG-. A workspace without the file is skipped, as before.
    CorpusFile("imported", "IMP", "memory", "IMPORTED.md"),
    # 5.0.1: the `mm ingest-serve` webhook door. Under memory/ like every
    # other untrusted drop corpus, so a release decision can name these ids
    # (``admissibility._releasable_id_pattern`` derives the releasable set
    # from the prefix map below, which is derived from here). Registered
    # UNGATED on purpose, unlike the door itself: the file is inert until the
    # door writes it, and gating the registry on the flag would make
    # already-RELEASED content vanish the moment an operator turned the door
    # off.
    CorpusFile("ingest", "INGEST", "memory", "INGEST.md"),
)


def _derive_prefix_map() -> dict[str, tuple[str, str]]:
    """Block-id prefix → (subdir, filename), for the rows that have one."""
    out: dict[str, tuple[str, str]] = {}
    for entry in CORPUS_TABLE:
        prefix = entry.prefix
        if prefix is None:
            continue
        if prefix in out:  # pragma: no cover — a duplicate row is a typo, caught at import
            raise ValueError(f"duplicate block-id prefix in CORPUS_TABLE: {prefix!r}")
        out[prefix] = (entry.subdir, entry.filename)
    return out


def _derive_file_map() -> dict[str, str]:
    """Recall label → workspace-relative path, in table order."""
    out: dict[str, str] = {}
    for entry in CORPUS_TABLE:
        if entry.label in out:  # pragma: no cover — duplicate label is a typo
            raise ValueError(f"duplicate corpus label in CORPUS_TABLE: {entry.label!r}")
        out[entry.label] = entry.relpath
    return out


#: Where a block id is written, and — via ``_delete_candidates`` — the first
#: place a delete looks. Re-exported as ``block_store._BLOCK_PREFIX_MAP``.
BLOCK_PREFIX_MAP: dict[str, tuple[str, str]] = _derive_prefix_map()

#: What recall, the index and the corpus enumeration walk. Re-exported as
#: ``_recall_constants.CORPUS_FILES``; the label is written onto every block
#: as ``_source_label``, so labels are API.
CORPUS_FILE_MAP: dict[str, str] = _derive_file_map()

#: Every corpus file the table names, workspace-relative, in table order.
#: ``MarkdownBlockStore._discover_files`` unions these with the
#: ``CORPUS_DIRS`` markdown walk, which is what makes "what the store can
#: read" equal to "what recall serves".
CORPUS_RELPATHS: tuple[str, ...] = tuple(entry.relpath for entry in CORPUS_TABLE)

#: Workspace-relative path -> recall label, for the rows the table names.
#: Derived, like everything else here, so a new row cannot land in one
#: direction and be missing from the other.
_LABEL_BY_RELPATH: dict[str, str] = {entry.relpath: entry.label for entry in CORPUS_TABLE}


def corpus_label_for(rel_path: str) -> str:
    """The ``_source_label`` a corpus file carries.

    A file the table names gets the table's label. A file it does not —
    an ``.md`` sitting directly in a corpus directory under a name
    nobody registered — gets its **directory** name, which is what
    ``v4.block_kinds._LABEL_KIND`` already keys on (``entities`` ->
    ENTITY, ``intelligence`` -> SYNTHESIS) and what
    ``memory_index._block_category`` reports as the category. Directory
    names and table labels are allowed to coincide: a
    ``decisions/ARCHIVE-2025.md`` split out of ``DECISIONS.md`` really is
    a ``decisions`` block, and ``governance._is_decision_block`` should
    count it.

    Total, and never raises: an unrecognised path answers with its first
    segment, so a caller can label a file that has since left the disk
    (which is exactly what the SQLite deletion sweep has to do).
    """
    rel = _normalise(rel_path)
    named = _LABEL_BY_RELPATH.get(rel)
    if named is not None:
        return named
    head = rel.split("/", 1)[0]
    return head or rel


def discover_corpus_files(
    workspace: str,
    corpus_dirs: tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    """``(label, relpath)`` for every file that holds corpus blocks (I-14).

    THE discovery function. ``MarkdownBlockStore._discover_files`` and the
    five retrieval/enumeration legs used to answer this question
    differently, and the difference was a hole rather than a nuance: the
    store lists **every** ``.md`` directly inside :data:`CORPUS_DIRS`,
    while recall, the SQLite index, the vector index and the
    admissibility status map all iterated :data:`CORPUS_FILE_MAP` — the
    table only. So a well-formed active block in a corpus-directory file
    the table does not name (``intelligence/BRIEFINGS.md``, an
    ``entities/`` split, an archive carved out of ``DECISIONS.md``) was
    ``get_by_id``-readable, ``get_all``-listable, ``export_memory``-visible
    and ``GET /memories``-visible, and recall never served it. Measured on
    5.0.1 and again on 5.0.2 before this landed::

        store._discover_files()          intelligence/BRIEFINGS.md  present
        store.get_by_id(probe)           True   (control: True)
        store.get_all(active_only=True)  True   (control: True)
        iter_active_blocks(ws)           False  (control: True)
        recall(ws, ...)                  absent (control: present)

    Two halves, unioned, and the order is load-bearing:

    * **the table**, first and in table order, EVERY row — existence is
      not checked here because :data:`CORPUS_FILE_MAP` never checked it
      either and every caller filters with ``os.path.isfile``. Keeping the
      table half first and whole is what makes this a drop-in for
      ``CORPUS_FILES.items()``: an existing workspace's recall order does
      not move by a single position.
    * **the walk**, after it: each directory in *corpus_dirs* in order,
      each ``.md`` directly inside it in sorted order, skipping anything
      the table already produced. Subdirectories are not descended and
      non-``.md`` files are ignored — the same two rules the store's walk
      applies, so the two sets are equal by construction rather than by
      two lists someone keeps in step.

    *corpus_dirs* is a **narrowing**, exactly as it is on
    ``MarkdownBlockStore``: ``None`` means the whole corpus; an explicit
    tuple restricts the walk to those directories and drops table rows
    that live elsewhere.

    ``tests/test_one_corpus_definition.py`` pins the result equal to
    ``MarkdownBlockStore._discover_files()`` over the same workspace, and
    its mutation twin drops the walk half and watches that pin go red.
    """
    import os  # noqa: PLC0415 — stdlib, and this module is otherwise import-free

    dirs = CORPUS_DIRS if corpus_dirs is None else corpus_dirs
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in CORPUS_TABLE:
        if corpus_dirs is not None and entry.subdir not in corpus_dirs:
            continue
        if entry.relpath in seen:  # pragma: no cover — duplicate relpath is a typo
            continue
        seen.add(entry.relpath)
        out.append((entry.label, entry.relpath))
    for subdir in dirs:
        dir_path = os.path.join(workspace, *subdir.split("/"))
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".md"):
                continue
            rel = f"{subdir}/{fname}"
            if rel in seen:
                continue
            seen.add(rel)
            out.append((corpus_label_for(rel), rel))
    return out


# ─── Ledgers of record ────────────────────────────────────────────────────────
#
# The append-only ledgers are NOT corpus. They are the record *about* the
# corpus, and the two must never move together: a snapshot that captures a
# ledger and a restore that copies it back rewinds the audit chain, so a
# block written after the snapshot disappears with no record that it ever
# existed. Measured on 5.0.1 (fresh workspace, Markdown backend)::
#
#     before snapshot          evidence 1  hash_chain 1
#     after one governed write evidence 2  hash_chain 2   block visible
#     AFTER restore            evidence 1  hash_chain 1   block GONE
#
# The same numbers came back through ``mind-mem-backup restore``. "Hashes
# are never rewritten" was true; "history is never removed" was not.
#
# This registry is the one place that says which paths are ledgers, so the
# rule is a property of the path and not of whoever remembered to check.
# Every snapshot writer asks it before capturing and refuses to build a
# manifest that names one (``block_store._build_manifest``); every restore
# reader asks it before writing and refuses to overwrite one
# (``block_store.restore``, ``backup_restore.restore_workspace``); the
# rollback orphan sweep asks it before removing one
# (``block_store._cleanup_orphans_from_manifest`` — excluding a ledger from
# the manifest without this would make the sweep DELETE it as an orphan,
# which is worse than the rewind it replaces).

#: Exact workspace-relative paths of the append-only ledgers.
#:
#: ``.mind-mem-audit/`` and ``.mind-mem-ledger/`` are not in
#: :data:`SNAPSHOT_DIRS` or :data:`BACKUP_DIRS` today, so those two rows are
#: not load-bearing yet. They are declared here anyway: the moment a
#: directory list grows, the exclusion is already true, rather than being a
#: second edit somebody has to remember.
LEDGER_FILES: tuple[str, ...] = (
    "memory/hash_chain_v2.db",
    # The chain's head seal (5.0.2). Not a ledger of its own — it holds
    # one line — but it is the record that makes the ledger beside it
    # non-truncatable, and a snapshot that captured the chain's seal and
    # put it back would rewind the two together, which is worse than
    # rewinding one: the restored pair agree with each other and the
    # verifier reports a clean chain over the wrong history.
    "memory/hash_chain_v2.head",
    "memory/evidence_chain.jsonl",
    ".mind-mem-audit/chain.jsonl",
    ".mind-mem-ledger/served.jsonl",
)

#: Sidecars and per-run ledger files, matched with :func:`fnmatch.fnmatch`.
#:
#: The SQLite sidecars matter as much as the database: restoring a stale
#: ``hash_chain_v2.db`` next to a live ``-wal`` (or the reverse) is a
#: corrupt chain rather than a rewound one, and corruption reads as
#: tampering to :func:`mind_mem.verify_cli` — an even worse outcome than
#: the rewind.
LEDGER_PATTERNS: tuple[str, ...] = (
    "memory/hash_chain_v2.db-*",
    "memory/evidence_chain.jsonl.*",
    ".mind-mem-ledger/served*.jsonl",
)


class LedgerCaptureError(RuntimeError):
    """A snapshot or archive tried to take a ledger of record with it.

    Raised rather than warned. A snapshot whose manifest names a ledger is
    a rewind waiting for its restore, and the failure has to land on the
    writer that built it — at capture time, when nothing has been lost yet
    — not on the operator who later runs the rollback.
    """


def _normalise(rel_path: str) -> str:
    """Workspace-relative path in canonical forward-slash form.

    Collapses ``\\`` to ``/``, drops a leading ``./`` and any leading or
    trailing ``/`` so ``memory/x``, ``./memory/x`` and ``memory\\x`` are one
    key. Nothing here resolves symlinks or ``..`` — callers hand this
    already-validated manifest entries, and a path that escapes the
    workspace is rejected by ``_safe_child_path`` before it gets here.
    """
    rel = str(rel_path).replace("\\", "/").strip("/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def is_ledger_path(rel_path: str) -> bool:
    """True when *rel_path* names a ledger of record.

    *rel_path* is workspace-relative. Separators are normalised, so a
    Windows-native ``memory\\hash_chain_v2.db`` matches the same row as the
    POSIX spelling — the manifests this is asked about are written on one
    OS and read on another.
    """
    rel = _normalise(rel_path)
    if not rel:
        return False
    if rel in LEDGER_FILES:
        return True
    return any(fnmatch(rel, pattern) for pattern in LEDGER_PATTERNS)


def strip_ledger_paths(paths: Iterable[str]) -> list[str]:
    """*paths* with every ledger path removed, order preserved."""
    return [p for p in paths if not is_ledger_path(p)]


def assert_ledger_free(paths: Iterable[str], *, what: str) -> None:
    """Refuse *paths* if any of them names a ledger of record.

    The choke point. A snapshot writer that forgets to filter its walk
    still cannot produce an artifact that names a ledger, because the
    manifest it has to write goes through here.

    Raises:
        LedgerCaptureError: naming *what* built the list and which paths
            were refused.
    """
    offenders = sorted({_normalise(p) for p in paths if is_ledger_path(p)})
    if offenders:
        raise LedgerCaptureError(
            f"{what} named {len(offenders)} ledger path(s): {offenders}. "
            "The append-only ledgers are the record ABOUT the corpus and are "
            "structurally outside every snapshot, backup and restore — capturing "
            "one makes the next restore rewind the audit chain. Filter the walk "
            "with corpus_registry.is_ledger_path()."
        )


def iter_ledger_paths(workspace: str) -> list[str]:
    """Workspace-relative paths of the ledgers that exist in *workspace*.

    The enumeration lives here rather than in the backup writer so there is
    exactly one answer to "which files are the ledgers" — the writer that
    relocates them and the reader that refuses them are looking at the same
    table, not at two lists that can drift apart.
    """
    import os  # noqa: PLC0415 — stdlib, and this module is otherwise import-free

    found: list[str] = []
    seen: set[str] = set()
    roots = {p.split("/", 1)[0] for p in LEDGER_FILES + LEDGER_PATTERNS}
    for root in sorted(roots):
        root_dir = os.path.join(workspace, root)
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                rel = _normalise(os.path.relpath(os.path.join(dirpath, name), workspace))
                if is_ledger_path(rel) and rel not in seen:
                    seen.add(rel)
                    found.append(rel)
    return sorted(found)
