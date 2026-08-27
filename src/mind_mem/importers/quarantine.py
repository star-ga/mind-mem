# Copyright 2026 STARGA, Inc.
"""Import quarantine — external ingest is never authoritative on arrival.

``mm import`` is a **bulk** write path: one command lifts an entire
foreign corpus (a markdown vault, another agent's memory directory, a
chat transcript, a ``mem0`` / ``letta`` / ``chroma`` dump) into the
store. Nothing in that content was written by an accountable actor of
this workspace, and nothing in it passed the governance gate.

Two consequences, both deliberate:

**1. Bulk ingest does not go one-proposal-per-block.**
A 10 000-note vault cannot become 10 000 staged proposals — the review
queue would be the migration, and a reviewer facing 10 000 identical
prompts approves them all without reading any. So the *write* is bulk
and ungated, and safety is bought two other ways instead:

* every imported block lands **quarantined**
  (``Status: quarantined`` + ``IngestTier: external-ingest``, on top of
  the ``imported:<system>`` provenance the importers already stamp), so
  it is inert — :func:`mind_mem.recall.recall` will not return it; and
* the import **run** is recorded in the tamper-evident audit chain
  (``.mind-mem-audit/chain.jsonl``) with the system, the source path,
  the batch id and every block id it wrote, so an unreviewed corpus can
  never appear in the store without a chain entry naming where it came
  from.

**2. Promotion out of quarantine is one governed decision.**
Admitting a batch is exactly one judgement — *"do I trust this source?"*
— so it is staged as exactly one ordinary governance proposal in
``intelligence/proposed/EDITS_PROPOSED.md`` and applied through the
existing ``approve_apply`` / :func:`mind_mem.apply_engine.apply_proposal`
gate, with its snapshot, WAL, contradiction check, receipt and rollback.
:func:`propose_import_release` only *stages*; it never writes the corpus.

The applied proposal appends a normal ``[D-...]`` **release decision** to
``decisions/DECISIONS.md`` listing the released block ids. Recall reads
that list (:func:`admitted_import_ids`) and stops filtering those blocks.
Admission is therefore a first-class, recallable, rollback-able corpus
record rather than a silent field flip — rolling the proposal back
re-quarantines the whole batch with no extra machinery.

Why the release is a decision block and not a ``set_status`` op on the
imported block itself: ``IMP-`` blocks live in ``memory/IMPORTED.md``,
outside :data:`mind_mem.corpus_registry.CORPUS_DIRS`, so the apply
engine's block-level ops cannot resolve them through the Markdown
BlockStore (``store.get_by_id`` returns ``None``) — the op would fail at
apply time. The release record lives where the governed path can
actually write it.

Zero external deps — stdlib plus ``apply_engine`` / ``block_parser``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Final, Iterable, Mapping, Sequence

from ..provenance_class import EXTERNAL_INGEST

__all__ = [
    "DECISIONS_FILE",
    "MAX_RELEASE_BLOCKS",
    "QUARANTINE_STATUS",
    "QUARANTINE_TIER",
    "RELEASE_FIELD",
    "RELEASE_PROPOSAL_FILE",
    "ImportQuarantineError",
    "NothingToReleaseError",
    "ReleaseTooLargeError",
    "admitted_import_ids",
    "batch_id_for",
    "build_release_proposal",
    "is_quarantined",
    "propose_import_release",
    "quarantined_import_ids",
    "record_import_in_chain",
    "withheld_import_ids",
    "render_release_proposal_block",
]

#: Status every imported block carries on arrival. Recall filters it.
QUARANTINE_STATUS: Final = "quarantined"

#: Why it is quarantined — the existing provenance-class vocabulary, so
#: the block field and ``classify_provenance`` cannot drift apart.
QUARANTINE_TIER: Final = EXTERNAL_INGEST

#: Block field naming the quarantine tier.
TIER_FIELD: Final = "IngestTier"

#: Block field naming the import batch a block arrived in.
BATCH_FIELD: Final = "ImportBatch"

#: Release-decision field listing the released block ids.
RELEASE_FIELD: Final = "Releases"

#: Release-decision field naming the batch (human handle, one per run).
BATCH_ADMIT_FIELD: Final = "AdmitsImportBatch"

#: Releasing a batch is an edit to the corpus, so it stages in the edit
#: proposal file — the same file ``lint_autofix`` uses.
RELEASE_PROPOSAL_FILE: Final = "intelligence/proposed/EDITS_PROPOSED.md"

#: Where the applied release decision lands.
DECISIONS_FILE: Final = "decisions/DECISIONS.md"

#: Upper bound on block ids in one release proposal. A release is a
#: reviewable artifact; past this the id list stops being readable and
#: the caller should release in explicit batches instead.
MAX_RELEASE_BLOCKS: Final = 500

_PROPOSAL_ID_RE: Final = re.compile(r"^P-(\d{8})-(\d{3})$")
_IMPORT_ID_RE: Final = re.compile(r"^IMP-[a-zA-Z0-9_.-]+$")
_UNSAFE_RE: Final = re.compile(r"[\r\n]+")

#: ``(realpath, mtime_ns, size) -> admitted ids``. One slot: recall calls
#: this repeatedly with the same workspace, and the key carries the file
#: identity so a governance apply invalidates it on the next call.
_ADMITTED_CACHE: dict[tuple[str, int, int], frozenset[str]] = {}

#: Same shape for the withheld set, keyed on BOTH files that define it.
_WITHHELD_CACHE: dict[tuple[tuple[str, int, int], tuple[str, int, int]], frozenset[str]] = {}


class ImportQuarantineError(RuntimeError):
    """Base class for every quarantine-release failure."""


class NothingToReleaseError(ImportQuarantineError):
    """No quarantined block matched the requested ids."""


class ReleaseTooLargeError(ImportQuarantineError):
    """More ids than :data:`MAX_RELEASE_BLOCKS` in a single proposal."""


# ---------------------------------------------------------------------------
# Predicates + batch identity
# ---------------------------------------------------------------------------


def is_quarantined(status: Any) -> bool:
    """True when a block/hit ``Status`` value means *quarantined*."""
    return isinstance(status, str) and status.strip().lower() == QUARANTINE_STATUS


def batch_id_for(system: str, block_ids: Iterable[str]) -> str:
    """Deterministic ``IMPB-<system>-<digest>`` id for one import run.

    Derived from the sorted block ids the run writes, so it is a pure
    function of the content admitted — no clock, no path, no counter.
    An empty run has no batch.
    """
    ordered = sorted(str(b) for b in block_ids)
    if not ordered:
        return ""
    from ..capture import content_hash

    return f"IMPB-{system}-{content_hash(chr(31).join(ordered))}"


# ---------------------------------------------------------------------------
# Audit-chain record for the bulk write
# ---------------------------------------------------------------------------


def record_import_in_chain(
    workspace: str,
    *,
    system: str,
    source_path: str,
    batch: str,
    block_ids: Sequence[str],
    corpus_file: str,
) -> str:
    """Append the tamper-evident record of one bulk import run.

    This is the half of the bulk-ingest bargain that makes an ungated
    write acceptable (see the module docstring): the blocks are inert
    until released, and the run that produced them is chained.

    Returns:
        The ``entry_hash`` of the appended audit entry.

    Raises:
        ImportQuarantineError: the ledger could not be written. The
            caller must surface this — an import that cannot be recorded
            must not be reported as a success.
    """
    from ..audit_chain import AuditChain

    payload = {
        "system": system,
        "source_path": source_path,
        "batch": batch,
        "status": QUARANTINE_STATUS,
        "tier": QUARANTINE_TIER,
        "block_ids": list(block_ids),
    }
    try:
        entry = AuditChain(workspace).append(
            "create_block",
            corpus_file,
            agent=f"importer:{system}",
            reason=(
                f"bulk external ingest ({len(block_ids)} block(s)) quarantined as {QUARANTINE_TIER}; "
                "release requires an approved governance proposal"
            ),
            payload=payload,
            fields_changed=["Status", TIER_FIELD, BATCH_FIELD],
        )
    except (OSError, ValueError) as exc:
        raise ImportQuarantineError(
            f"import could not be recorded in the audit chain ({exc}); the {len(block_ids)} written block(s) stay quarantined and inert"
        ) from exc
    return str(entry.entry_hash)


# ---------------------------------------------------------------------------
# Reading quarantine state
# ---------------------------------------------------------------------------


def quarantined_import_ids(workspace: str, corpus_file: str) -> tuple[str, ...]:
    """Ids of the quarantined blocks currently in *corpus_file*.

    Deterministic file order. Missing/unreadable file yields ``()``.
    """
    path = os.path.join(workspace, corpus_file)
    if not os.path.isfile(path):
        return ()
    from ..block_parser import parse_file

    try:
        blocks = parse_file(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return ()
    return tuple(str(b["_id"]) for b in blocks if b.get("_id") and is_quarantined(b.get("Status")))


def _release_ids(block: Mapping[str, Any]) -> list[str]:
    """Block ids named by a release decision's ``Releases`` field."""
    raw = block.get(RELEASE_FIELD)
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [str(item) for item in raw]
    else:
        return []
    return [c.strip() for c in candidates if _IMPORT_ID_RE.match(c.strip())]


def admitted_import_ids(workspace: str) -> frozenset[str]:
    """Imported block ids released by an **active** release decision.

    The source of truth is ``decisions/DECISIONS.md`` — written only by
    the apply engine, so an id can only appear here via an approved
    proposal. A revoked/superseded release decision stops admitting its
    batch, which is what makes rollback re-quarantine for free.

    Cached on ``(path, mtime_ns, size)``; any governance apply changes
    the file and therefore the key. Never raises: an unreadable
    decisions file admits nothing (fail-closed).
    """
    path = os.path.join(workspace, DECISIONS_FILE)
    try:
        stat = os.stat(path)
    except OSError:
        return frozenset()
    key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)
    cached = _ADMITTED_CACHE.get(key)
    if cached is not None:
        return cached

    from ..block_parser import parse_file

    try:
        blocks = parse_file(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return frozenset()

    admitted: set[str] = set()
    for block in blocks:
        if str(block.get("Status", "")).strip().lower() != "active":
            continue
        admitted.update(_release_ids(block))
    result = frozenset(admitted)
    # Single-slot cache: drop older keys rather than grow unbounded.
    _ADMITTED_CACHE.clear()
    _ADMITTED_CACHE[key] = result
    return result


def _file_key(path: str) -> tuple[str, int, int] | None:
    """``(realpath, mtime_ns, size)`` identity for *path*, or None if absent."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)


def withheld_import_ids(workspace: str, corpus_file: str = "memory/IMPORTED.md") -> frozenset[str]:
    """Imported block ids recall must withhold: quarantined and unadmitted.

    This is the id-set form of the quarantine rule, for retrieval paths
    whose hits do not reliably carry a ``Status`` field (the vector and
    fused legs). It is the exact complement of
    :func:`admitted_import_ids` over what is actually in quarantine.

    Free when there is nothing to withhold: a workspace that has never
    run ``mm import`` has no ``memory/IMPORTED.md``, and this returns the
    empty set after a single ``stat``. Otherwise cached on the identity
    of both files that define the answer, so a governance apply (which
    rewrites ``decisions/DECISIONS.md``) invalidates it.
    """
    corpus_key = _file_key(os.path.join(workspace, corpus_file))
    if corpus_key is None:
        return frozenset()
    decisions_key = _file_key(os.path.join(workspace, DECISIONS_FILE)) or ("", 0, 0)
    key = (corpus_key, decisions_key)
    cached = _WITHHELD_CACHE.get(key)
    if cached is not None:
        return cached
    withheld = frozenset(quarantined_import_ids(workspace, corpus_file)) - admitted_import_ids(workspace)
    _WITHHELD_CACHE.clear()
    _WITHHELD_CACHE[key] = withheld
    return withheld


# ---------------------------------------------------------------------------
# Proposal construction
# ---------------------------------------------------------------------------


def _staged_release_ids(proposal_text: str) -> set[str]:
    """Imported block ids covered by a still-``staged`` release proposal.

    Read straight off the proposal file so two releases of the same
    blocks cannot both sit in the queue. A fingerprint check cannot do
    this job: the generated proposal/decision ids differ between two
    otherwise-identical calls, so the fingerprints differ too.
    """
    from ..block_parser import parse_blocks

    try:
        blocks = parse_blocks(proposal_text)
    except (ValueError, RuntimeError):  # pragma: no cover — defensive
        return set()
    covered: set[str] = set()
    for block in blocks:
        if str(block.get("Status", "")).strip().lower() != "staged":
            continue
        ops = block.get("Ops")
        if not isinstance(ops, list):
            continue
        for op in ops:
            if not isinstance(op, dict) or op.get("file") != DECISIONS_FILE:
                continue
            patch = str(op.get("patch", ""))
            if f"{RELEASE_FIELD}:" not in patch:
                continue
            covered.update(re.findall(r"IMP-[a-zA-Z0-9_.-]+", patch))
    return covered


def _one_line(text: str) -> str:
    """Collapse to a single markdown-safe line (blocks header injection)."""
    return _UNSAFE_RE.sub(" ", text).replace("[", "(").replace("]", ")").strip()[:280]


def _next_id(existing: str, prefix: str, date_compact: str) -> str:
    """Allocate ``<prefix>-<date>-NNN`` one past the highest already used."""
    used = [int(m) for m in re.findall(rf"{prefix}-{re.escape(date_compact)}-(\d{{3}})", existing)]
    nxt = (max(used) + 1) if used else 1
    if nxt > 999:
        raise ImportQuarantineError(f"{prefix} id space exhausted for {date_compact} (999 per day)")
    return f"{prefix}-{date_compact}-{nxt:03d}"


def render_release_decision(
    decision_id: str,
    *,
    system: str,
    batch: str,
    block_ids: Sequence[str],
    rationale: str,
    date: str,
) -> str:
    """Render the ``[D-...]`` release decision the proposal appends.

    Every field is single-line by construction (the id list is a bullet
    list), so the block survives the proposal-file round trip and the
    apply engine's ``store.write_block`` re-render without loss.
    """
    lines = [
        f"[{decision_id}]",
        f"Statement: Admit {len(block_ids)} quarantined imported block(s) from {_one_line(system)} into recall.",
        f"Date: {date}",
        "Status: active",
        "Type: decision",
        f"Rationale: {_one_line(rationale)}",
        f"Source: {QUARANTINE_TIER}",
        f"{BATCH_ADMIT_FIELD}: {_one_line(batch) if batch else 'none'}",
        f"{RELEASE_FIELD}:",
    ]
    lines.extend(f"- {block_id}" for block_id in block_ids)
    return "\n".join(lines)


def build_release_proposal(
    proposal_id: str,
    decision_id: str,
    *,
    system: str,
    batch: str,
    block_ids: Sequence[str],
    rationale: str,
    date: str,
) -> dict[str, Any]:
    """Build the (unwritten) release proposal dict.

    Pure: no filesystem access, no clock. Exposed so a caller — or a
    golden-diff test — can inspect exactly what would be staged.
    """
    from ..apply_engine import compute_fingerprint, validate_proposal

    if not _PROPOSAL_ID_RE.match(proposal_id):
        raise ImportQuarantineError(f"invalid proposal id: {proposal_id!r} (expected P-YYYYMMDD-NNN)")
    if not block_ids:
        raise NothingToReleaseError("a release proposal needs at least one quarantined block id")
    if len(block_ids) > MAX_RELEASE_BLOCKS:
        raise ReleaseTooLargeError(
            f"{len(block_ids)} block ids exceeds MAX_RELEASE_BLOCKS={MAX_RELEASE_BLOCKS}; "
            "release in explicit batches so the proposal stays reviewable"
        )

    patch = render_release_decision(
        decision_id,
        system=system,
        batch=batch,
        block_ids=block_ids,
        rationale=rationale,
        date=date,
    )
    proposal: dict[str, Any] = {
        "ProposalId": proposal_id,
        "Type": "edit",
        "TargetBlock": decision_id,
        # External content becoming recallable is the highest-blast-radius
        # thing this package does; it is never a low-risk edit.
        "Risk": "high",
        "Evidence": [
            _one_line(f"quarantine release: {len(block_ids)} block(s) imported from {system}"),
            _one_line(f"batch: {batch or 'none'}"),
        ],
        "Rollback": "restore_snapshot",
        "Ops": [
            {
                "op": "append_block",
                "file": DECISIONS_FILE,
                "target": decision_id,
                "patch": patch,
            }
        ],
        "FilesTouched": [DECISIONS_FILE],
        "Status": "staged",
        # Sanitised: these land as markdown list items in a governed file.
        "Sources": [_one_line(f"imported:{system}")] + ([_one_line(f"batch:{batch}")] if batch else []),
    }
    proposal["Fingerprint"] = compute_fingerprint(proposal)

    errors = validate_proposal(proposal)
    if errors:  # pragma: no cover — guards against a malformed release op
        raise ImportQuarantineError(f"generated release proposal failed validation: {errors}")
    return proposal


def render_release_proposal_block(proposal: Mapping[str, Any]) -> str:
    """Serialise *proposal* to the canonical proposal-block markdown.

    The ``append_block`` patch is emitted as a ``patch: |`` literal with
    every line indented four spaces — the indent is what stops the
    embedded ``[D-...]`` header from being read as a new block, and the
    block parser strips exactly those four spaces back off.
    """
    op = dict(proposal["Ops"][0])
    patch_lines = "\n".join("    " + line for line in str(op["patch"]).split("\n"))
    evidence = "\n".join(f"- {line}" for line in proposal["Evidence"])
    touched = "\n".join(f"- {line}" for line in proposal["FilesTouched"])
    sources = "\n".join(f"- {line}" for line in proposal["Sources"])
    return (
        f"\n[{proposal['ProposalId']}]\n"
        f"ProposalId: {proposal['ProposalId']}\n"
        f"Type: {proposal['Type']}\n"
        f"TargetBlock: {proposal['TargetBlock']}\n"
        f"Risk: {proposal['Risk']}\n"
        f"Evidence:\n{evidence}\n"
        f"Rollback: {proposal['Rollback']}\n"
        f"Ops:\n"
        f"- op: {op['op']}\n"
        f"  file: {op['file']}\n"
        f"  target: {op['target']}\n"
        f"  patch: |\n{patch_lines}\n"
        f"Fingerprint: {proposal['Fingerprint']}\n"
        f"Status: {proposal['Status']}\n"
        f"FilesTouched:\n{touched}\n"
        f"Sources:\n{sources}\n"
    )


# ---------------------------------------------------------------------------
# Staging (the only write this module performs)
# ---------------------------------------------------------------------------


def propose_import_release(
    workspace: str,
    block_ids: Sequence[str],
    *,
    system: str,
    batch: str = "",
    rationale: str = "",
    corpus_file: str = "memory/IMPORTED.md",
    now: datetime | None = None,
) -> str:
    """Stage a release proposal for *block_ids* and return its proposal id.

    The corpus is not modified. The only file written is
    :data:`RELEASE_PROPOSAL_FILE`; the blocks become recallable only when
    a human runs ``approve_apply(proposal_id, dry_run=False)``, which is
    where the snapshot, the contradiction check, the WAL and the rollback
    receipt live.

    Args:
        workspace: Workspace root.
        block_ids: Imported block ids to release. Ids that are not
            currently quarantined in *corpus_file* are dropped — a
            release can only ever admit what is actually in quarantine.
        system: Source system slug, for the decision statement.
        batch: Import batch id (``ImportBatch`` on the blocks).
        rationale: Why this source is trusted. Recorded on the decision.
        corpus_file: Where the imported blocks live.
        now: Clock override for the id dates (tests/replay).

    Returns:
        The staged proposal id, e.g. ``"P-20260827-001"``.

    Raises:
        NothingToReleaseError: none of *block_ids* is in quarantine.
        ReleaseTooLargeError: more than :data:`MAX_RELEASE_BLOCKS` ids.
        ImportQuarantineError: workspace/proposal-file problems.
    """
    if not workspace or not isinstance(workspace, str):
        raise ImportQuarantineError("workspace must be a non-empty path string")
    ws = os.path.abspath(workspace)
    if not os.path.isdir(ws):
        raise ImportQuarantineError(f"workspace does not exist: {ws}")

    requested = [str(raw).strip() for raw in block_ids]
    in_quarantine = set(quarantined_import_ids(ws, corpus_file))
    already = admitted_import_ids(ws)
    # Preserve caller order, drop duplicates, keep only ids that are in
    # quarantine AND not already admitted by a live release decision —
    # so re-releasing an admitted batch is a refusal, not a second
    # redundant decision block.
    wanted: list[str] = []
    for candidate in requested:
        if candidate in in_quarantine and candidate not in already and candidate not in wanted:
            wanted.append(candidate)
    if not wanted:
        raise NothingToReleaseError(
            f"none of the {len(requested)} requested id(s) is awaiting release in {corpus_file} (already admitted, or never imported)"
        )
    if len(wanted) > MAX_RELEASE_BLOCKS:
        raise ReleaseTooLargeError(
            f"{len(wanted)} block ids exceeds MAX_RELEASE_BLOCKS={MAX_RELEASE_BLOCKS}; "
            "release in explicit batches so the proposal stays reviewable"
        )

    proposal_path = os.path.join(ws, RELEASE_PROPOSAL_FILE)
    if not os.path.isfile(proposal_path):
        raise ImportQuarantineError(f"missing proposal file: {RELEASE_PROPOSAL_FILE} (run mind-mem-init on this workspace)")

    stamp = now or datetime.now()
    date_compact = stamp.strftime("%Y%m%d")
    date_iso = stamp.strftime("%Y-%m-%d")

    decisions_text = ""
    decisions_path = os.path.join(ws, DECISIONS_FILE)
    if os.path.isfile(decisions_path):
        with open(decisions_path, "r", encoding="utf-8") as handle:
            decisions_text = handle.read()

    from ..mind_filelock import FileLock
    from ..observability import get_logger, metrics

    log = get_logger("import_quarantine")

    with FileLock(proposal_path):
        with open(proposal_path, "r", encoding="utf-8") as handle:
            existing = handle.read()
        proposal_id = _next_id(existing, "P", date_compact)
        # Decision ids must not collide with an id already staged in a
        # pending proposal, so both texts feed the allocator.
        decision_id = _next_id(decisions_text + existing, "D", date_compact)
        proposal = build_release_proposal(
            proposal_id,
            decision_id,
            system=system,
            batch=batch,
            block_ids=wanted,
            rationale=rationale or f"External corpus imported from {system}; reviewed before admission.",
            date=date_iso,
        )
        if _staged_release_ids(existing) & set(wanted):
            raise ImportQuarantineError(
                "a release proposal covering one or more of these blocks is already staged; apply or reject it before staging another"
            )
        with open(proposal_path, "a", encoding="utf-8") as handle:
            handle.write(render_release_proposal_block(proposal))

    metrics.inc("import_release_proposals")
    log.info(
        "import_release_staged",
        system=system,
        batch=batch,
        blocks=len(wanted),
        proposal_id=proposal_id,
        decision_id=decision_id,
    )
    return proposal_id
