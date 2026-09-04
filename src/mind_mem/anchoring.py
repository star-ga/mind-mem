# Copyright 2026 STARGA, Inc.
"""Which corpus blocks the write ledger never admitted (5.0.2).

THE DEFECT, measured on a fresh ``init`` workspace. Append a block to
``decisions/DECISIONS.md`` by hand — no receipt, no scope, no chain row —
``Status: active``, then ``build_index``, then
``recall("zebra quantum ledger")``::

    recall ids:                     ['D-20260903-777']
    hash chain rows: 0   evidence rows: 0
    verify ok: True  exit: 0        (7/7 checks green)

The block was served. Nothing in the product asked whether its id had
ever been admitted, because nothing could:
:func:`~mind_mem.admissibility.is_admissible_status` and
``RECOGNISED_STATUSES`` judge the **status string** and only that. The
corpus is human-editable Markdown, so a write that never met the gate is
a boundary the product actually has rather than a hypothetical one.

WHAT THIS MODULE ADDS, and what it deliberately does not. It answers one
question — *which corpus ids appear in no write scope's close record?* —
and it offers the batch that fixes the answer
(:func:`restamp_unanchored`). It does **not** withhold an unanchored
block at serve time. Anchored-only serving is a later release: every
workspace that predates the close record (5.0.2) carries a corpus whose
blocks are all unanchored, and a release that silently stopped serving
them would empty the product rather than govern it. So the shape here is
*report, then offer to anchor* — condemning a pre-gate corpus is what the
re-stamp exists to avoid.

WHERE THE LANDED IDS COME FROM. Not a new source: every write scope
already writes one close record naming the ids
:func:`~mind_mem.admission.require_admission` authorised inside it
(``metadata["landed"]``, see
``GovernanceGate._record_scope_close``). Reading that is what makes this
check a question asked OF the existing ledger rather than a second
opinion about what was written.

THE ONE LIMIT, stated rather than discovered later. A close record caps
its inline id list (``landed_truncated``); ``landed_count`` and
``landed_root`` stay exact at any size, but a Merkle root does not answer
membership. So a scope that landed more ids than the cap contributes its
count and not its ids, and blocks from it can be reported unanchored when
they are not. :class:`AnchorReport` carries ``truncated_scopes`` so the
over-report is visible in the answer instead of being a surprise, and
re-stamping one of those blocks is harmless — it writes a true chain row
for a block that has one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .corpus_registry import discover_corpus_files

__all__ = [
    "AnchorReport",
    "CorpusIds",
    "LandedIds",
    "RestampResult",
    "corpus_block_ids",
    "landed_block_ids",
    "restamp_unanchored",
    "unanchored_report",
]

#: Verb the re-stamp batch is admitted under.
#:
#: ``_ACTION_MAP`` in :mod:`~mind_mem.governance_gate` is an allow-list
#: and refuses a verb it cannot classify, so this is one of the verbs that
#: table already names. ``WRITE`` is the truthful one: the pass really
#: does write the blocks, and it lands as
#: :attr:`~mind_mem.evidence_objects.EvidenceAction.APPLY`. What makes the
#: record say *re-stamp* rather than *fresh ingest* is not the verb but
#: :attr:`~mind_mem.enums.IngestTier.RESTAMP`, which the gate copies into
#: the entry as ``metadata["ingest_tier"]``, plus :data:`ANCHOR_MARKER`.
RESTAMP_VERB = "WRITE"

#: ``metadata`` key marking the admission row this pass mints, so an
#: auditor can tell an anchoring pass from any other re-stamp with one
#: grep instead of by inferring it from the id set.
ANCHOR_MARKER = "anchor_pass"


@dataclass(frozen=True)
class LandedIds:
    """Ids some write scope reported landing, and how complete that is.

    ``truncated_scopes`` is not decoration: a scope that listed its ids
    truncated contributed ``landed_count`` and no ids, so :attr:`ids` is
    a **subset** of what actually landed whenever it is non-zero. A
    consumer that subtracts this set from the corpus must say so.
    """

    ids: frozenset[str]
    close_records: int
    truncated_scopes: int

    @property
    def complete(self) -> bool:
        """True when every close record listed every id it landed."""
        return self.truncated_scopes == 0


@dataclass(frozen=True)
class CorpusIds:
    """Every block id on disk, and proof the walk actually happened.

    ``files_scanned`` exists because "no unanchored blocks" is only
    evidence when something was read. A workspace whose corpus files are
    all absent produces an empty :attr:`ids` that reads exactly like a
    fully-anchored corpus, and the two must not be confused.
    """

    ids: frozenset[str]
    files_scanned: int
    unreadable: tuple[str, ...]


#: The block-store backend whose corpus :func:`corpus_block_ids` can read.
#:
#: The Markdown walk IS the check on the default backend, and it is blind
#: on the other two: ``postgres`` keeps blocks in the database and
#: ``encrypted`` keeps them in ciphertext on disk, so the walk finds a
#: corpus of zero blocks and "all anchored" would be a statement about a
#: search that could not have found anything. Named here so the caller can
#: report the gap instead of the reassurance.
MARKDOWN_BACKEND = "markdown"


@dataclass(frozen=True)
class AnchorReport:
    """Corpus ids no write scope ever reported landing."""

    unanchored: tuple[str, ...]
    corpus_blocks: int
    landed_blocks: int
    files_scanned: int
    close_records: int
    truncated_scopes: int
    unreadable: tuple[str, ...]
    backend: str = MARKDOWN_BACKEND

    @property
    def backend_is_walkable(self) -> bool:
        """True when :func:`corpus_block_ids` can actually see the corpus."""
        return self.backend == MARKDOWN_BACKEND

    @property
    def searched(self) -> bool:
        """True when the corpus walk read at least one file.

        ``not searched`` and ``unanchored == ()`` are the same value and
        two different facts — nothing to report, versus nothing looked
        at. Callers must branch on this before reading the verdict.
        """
        return self.files_scanned > 0

    @property
    def over_reports(self) -> bool:
        """True when :attr:`unanchored` may name an id that IS anchored.

        The truncated-close-record limit from the module docstring. A
        report with this set says "at most these", never "exactly these".
        """
        return self.truncated_scopes > 0


@dataclass(frozen=True)
class RestampResult:
    """What one anchoring pass admitted, and what it could not."""

    anchored: tuple[str, ...]
    skipped: tuple[str, ...]
    dry_run: bool

    @property
    def wrote_nothing(self) -> bool:
        return not self.anchored


# ---------------------------------------------------------------------------
# Reading the two sides
# ---------------------------------------------------------------------------


def _metadata(row: Any) -> Mapping[str, Any]:
    meta = getattr(row, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def landed_block_ids(evidence: Any) -> LandedIds:
    """Ids reported landed by every closed WRITE scope in *evidence*.

    Reads the close record and nothing else. The open admission row names
    what a scope was *authorised* to write, which is the claim the chain
    over-reported before close records existed; ``landed`` names what
    :func:`~mind_mem.admission.require_admission` actually let through.
    Pairing on the close record is therefore the only reading that
    distinguishes an authorised write from a landed one.

    Takes the chain rather than a workspace path for the same reason
    :func:`~mind_mem.governance_gate.unclosed_write_scopes` does: two live
    readers over one JSONL is the fork the gate's ``close()`` exists to
    prevent.
    """
    from .governance_gate import OP_WRITE, PHASE_CLOSED

    ids: set[str] = set()
    closes = 0
    truncated = 0
    for row in evidence.get_latest(n=len(evidence)):
        meta = _metadata(row)
        if meta.get("write_phase") != PHASE_CLOSED or meta.get("operation") != OP_WRITE:
            continue
        closes += 1
        if meta.get("landed_truncated"):
            truncated += 1
        listed = meta.get("landed")
        if isinstance(listed, (list, tuple)):
            ids.update(str(bid) for bid in listed)
    return LandedIds(ids=frozenset(ids), close_records=closes, truncated_scopes=truncated)


def corpus_block_ids(workspace: str) -> CorpusIds:
    """Every block id in every corpus file of *workspace*.

    Walks :func:`~mind_mem.corpus_registry.discover_corpus_files` — THE
    discovery function, so a corpus file the table does not name is still
    walked — and parses each with
    :func:`~mind_mem.block_parser.parse_file`. Read-only: nothing is
    created, so a verifier can call it without leaving a directory behind.

    A file that cannot be read lands in :attr:`CorpusIds.unreadable`
    rather than being skipped silently. A corpus file nobody could parse
    may hold anything, and a walk that quietly dropped it would report
    "no unanchored blocks" over content it never saw.
    """
    from .block_parser import parse_file

    ids: set[str] = set()
    scanned = 0
    unreadable: list[str] = []
    for _label, rel in discover_corpus_files(workspace):
        path = os.path.join(workspace, *rel.split("/"))
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):
            unreadable.append(rel)
            continue
        scanned += 1
        for block in blocks:
            bid = block.get("_id")
            if isinstance(bid, str) and bid:
                ids.add(bid)
    return CorpusIds(ids=frozenset(ids), files_scanned=scanned, unreadable=tuple(unreadable))


def unanchored_report(workspace: str) -> AnchorReport:
    """Corpus ids minus chain-landed ids, with the facts behind both.

    An absent evidence ledger is not an error here and is not treated as
    one: it means *nothing was ever admitted*, so every corpus block is
    unanchored — which is precisely the state the H1 reproduction leaves
    behind and precisely what the caller needs told.
    """
    corpus = corpus_block_ids(workspace)
    landed = _landed_for_workspace(workspace)
    unanchored = tuple(sorted(corpus.ids - landed.ids))
    return AnchorReport(
        unanchored=unanchored,
        corpus_blocks=len(corpus.ids),
        landed_blocks=len(landed.ids),
        files_scanned=corpus.files_scanned,
        close_records=landed.close_records,
        truncated_scopes=landed.truncated_scopes,
        unreadable=corpus.unreadable,
        backend=configured_backend(workspace),
    )


def configured_backend(workspace: str) -> str:
    """``block_store.backend`` for *workspace*, defaulting to Markdown.

    Read straight out of ``mind-mem.json`` with :mod:`json` rather than
    through :func:`mind_mem.storage.get_block_store`, and that is
    deliberate: :mod:`~mind_mem.verify_cli` is stdlib-only and opens no
    store, and constructing one to ask which backend is configured would
    connect to a database in order to answer a question about a config
    file.

    An unreadable or malformed config answers :data:`MARKDOWN_BACKEND` —
    the same default :func:`~mind_mem.storage.get_block_store` applies, so
    the two never disagree about what a missing section means.
    """
    import json  # noqa: PLC0415 — stdlib, kept off this module's import closure

    try:
        with open(os.path.join(workspace, "mind-mem.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return MARKDOWN_BACKEND
    section = config.get("block_store") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        return MARKDOWN_BACKEND
    backend = section.get("backend", MARKDOWN_BACKEND)
    return backend if isinstance(backend, str) and backend else MARKDOWN_BACKEND


def _landed_for_workspace(workspace: str) -> LandedIds:
    """Read the evidence ledger, or report an empty one — creating nothing.

    The ``isfile`` probe is load-bearing in the same way it is in
    :mod:`~mind_mem.verify_cli`: constructing a reader to find out whether
    there is anything to read is asking the question by doing the thing.
    """
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return LandedIds(ids=frozenset(), close_records=0, truncated_scopes=0)

    from .evidence_objects import EvidenceChain

    return landed_block_ids(EvidenceChain(store_path=path))


# ---------------------------------------------------------------------------
# The repair
# ---------------------------------------------------------------------------


def restamp_unanchored(
    workspace: str,
    *,
    actor: str = "",
    limit: Optional[int] = None,
    dry_run: bool = False,
    ids: Optional[Sequence[str]] = None,
) -> RestampResult:
    """Admit a re-stamp batch over the unanchored blocks of *workspace*.

    The answer to "my corpus predates the gate" that is not "delete it".
    One :meth:`~mind_mem.governance_gate.GovernanceGate.admit_batch` scope
    on :attr:`~mind_mem.enums.IngestTier.RESTAMP`, whose
    :data:`~mind_mem.enums.INITIAL_STATUS` row is ``None`` — a *carrying*
    tier: it mints no status, so a block that was ``active`` stays
    ``active`` and an anchoring pass cannot be used to escalate anything.
    What the pass produces is the chain row and the close record the block
    never had.

    **What "unchanged" means, exactly**, because the loose version would
    be wrong. Every field and every value survives — measured on a
    hand-written decision, the parsed block before and after the pass is
    identical key-for-key, ``Status: active`` included. The *file* is not
    byte-identical: the block goes back through the store's own
    serialiser, so the fields come out in the store's canonical order and
    the block gains the trailing ``---`` separator every governed write
    already writes. An operator diffing the Markdown will see that
    reordering and should not read it as content loss.

    One scope, not one per block: a corpus of ten thousand pre-gate blocks
    is one operator decision, and N receipts would leave N chain records
    with nothing saying they were one.

    An empty id set writes **nothing** and opens no scope. A receipt
    covering nothing authorises nothing, and minting one would put a
    decision in the chain that never had a subject —
    ``admit_delete_batch`` already refuses that on the delete side; this
    refuses it here rather than relying on the write side to.

    Args:
        actor: Identity to attribute the anchoring to.
        limit: Anchor at most this many ids (lowest sorted first), so a
            large corpus can be anchored in reviewable passes.
        dry_run: Resolve the ids and return them, writing nothing.
        ids: Anchor exactly these instead of the computed set. For an
            operator who has reviewed the report and wants a subset.
    """
    candidates = tuple(str(i) for i in ids) if ids is not None else unanchored_report(workspace).unanchored
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    if dry_run:
        return RestampResult(anchored=candidates, skipped=(), dry_run=True)
    if not candidates:
        return RestampResult(anchored=(), skipped=(), dry_run=False)

    from .enums import IngestTier
    from .governance_gate import get_gate
    from .storage import get_block_store

    store = get_block_store(workspace)
    resolved: list[tuple[str, dict]] = []
    skipped: list[str] = []
    for bid in candidates:
        block = store.get_by_id(bid) if hasattr(store, "get_by_id") else None
        if block is None:
            # The id parsed out of a corpus file the store cannot resolve.
            # Reported, never re-stamped: writing a block this pass had to
            # invent would anchor content nobody wrote.
            skipped.append(bid)
            continue
        resolved.append((bid, block))
    if not resolved:
        return RestampResult(anchored=(), skipped=tuple(skipped), dry_run=False)

    batch = [bid for bid, _ in resolved]
    anchored: list[str] = []
    with get_gate(workspace).admit_batch(
        action=RESTAMP_VERB,
        batch_id=f"anchor-{len(batch)}",
        block_ids=batch,
        content="\n".join(batch),
        tier=IngestTier.RESTAMP,
        actor=actor or "anchoring",
        metadata={ANCHOR_MARKER: True, "blocks": len(batch)},
    ):
        for bid, block in resolved:
            store.write_block(block)
            anchored.append(bid)
    return RestampResult(anchored=tuple(anchored), skipped=tuple(skipped), dry_run=False)
