"""What recall is allowed to serve — the servability allow-list.

Recall used to answer "may I serve this?" with a **deny-list**: drop the
hits whose status spells ``quarantined``, plus an id-set recomputed off
``memory/IMPORTED.md`` for the legs whose hits carry no status. Two holes
follow from that shape:

* it is **fail-open on unknown statuses** — a status nobody had told it
  about (a new ingest door minting ``Status: staged``) was served by
  default;
* the id-set rule named exactly one corpus file, so every *other*
  withheld corpus — the inbox drop folder, agent messages — fell through
  it, and the three legs that splice raw parsed blocks into the result
  list (graph expansion, KG fusion, entity prefetch) leaked through the
  capital-vs-lowercase ``Status`` mismatch as well.

This module inverts it. :func:`is_admissible_status` is an allow-list, so
a status nobody has named — one a future ingest door invents included —
is withheld by default.

**Which axis the allow-list is on.** ``SERVABLE = {ACTIVE}`` in
:mod:`mind_mem.enums` is the allow-list on what a governed write may
*mint*, and it stays exactly that. Serving is a different question, and
reading the mint allow-list as the serve allow-list does not survive
contact with a live corpus: ``superseded`` / ``deprecated`` / ``archived``
/ ``rejected`` blocks are served **and demoted** on purpose
(``validity_gate`` scores them down rather than hiding them, because the
history of a decision is the product), and a Task block spends its whole
life on ``todo`` / ``doing`` / ``done``. A verbatim ``{ACTIVE}`` serve
gate deletes task recall and decision history.

So the withholding rule is on the *admission* axis, where the security
property actually lives:

    withheld(block) := status in UNADMITTED     (never passed the gate)
                    or status not RECOGNISED    (nobody has named it)

:data:`UNADMITTED` is derived from
:data:`~mind_mem.enums.INITIAL_STATUS` — the one table that decides what
an ingest tier mints — so a new withheld tier withholds its blocks the
moment its row exists, with nothing here to update. The second clause is
the fail-closed half and is the whole point: it is what makes a status
nobody has named unservable by default.

**Purity.** The core is a pure function of the blocks or hits it is
handed. No clock, no filesystem, no configuration. The release set is
derived from the decision blocks *already in the list*, which is what
makes the admissibility decision free: the corpus a caller has parsed
already contains its own answer, so an import-free workspace pays no
``stat`` at all (it used to pay one per recall).

One function breaks that purity, deliberately and in its own section:
:func:`workspace_release_ids`. The indexed retrieval paths never load the
Markdown corpus — that is the whole point of having an index — so they
have no decision blocks in hand to read the release set off. It is
resolved lazily, only once a candidate has actually failed the status
allow-list, so the common all-servable case still touches no disk.

**Ordering.** The filter must run *before* fusion. RRF scores an item at
``sum_leg 1/(k + rank_leg(i))``; drop a withheld item after fusion and
every admitted item below it carries a worse rank than it should, so the
presence of withheld content stays observable through its neighbours'
ranks. Filtering the candidates of each leg closes that channel.
"""

from __future__ import annotations

import re
from typing import Any, Final, Iterable, Mapping, Sequence

from .enums import INITIAL_STATUS, TaskStatus, is_servable
from .observability import get_logger, metrics

__all__ = [
    "DECISIONS_FILE",
    "RECOGNISED_STATUSES",
    "UNADMITTED",
    "RELEASE_FIELD",
    "admissible",
    "admit_corpus",
    "admit_leg",
    "count_unresolved",
    "is_admissible_status",
    "live_statuses",
    "release_ids",
    "unresolved_count",
    "with_live_statuses",
    "workspace_release_ids",
]

_log = get_logger("admissibility")

#: Block field on a release decision naming the ids it admits.
RELEASE_FIELD: Final = "Releases"

#: The only file a release decision can be written to. Written by the apply
#: engine alone, so an id can appear here only via an approved proposal.
DECISIONS_FILE: Final = "decisions/DECISIONS.md"

#: Metric incremented when an id reaches a resolution point that no block
#: in the corpus answers. Diagnostic only — an unresolvable id is dropped,
#: and the run is NOT marked degraded.
UNRESOLVED_METRIC: Final = "recall_unresolved_ids"


#: Statuses meaning "this block has not passed the governance gate".
#:
#: Derived from :data:`~mind_mem.enums.INITIAL_STATUS`, never hand-listed:
#: a tier whose row mints a non-servable status contributes that status
#: here automatically, so adding a withheld ingest tier needs no edit in
#: this module.
UNADMITTED: Final[frozenset[str]] = frozenset(
    status.value for status in INITIAL_STATUS.values() if status is not None and not is_servable(status)
)

#: Lifecycle statuses recall recognises. Anything outside this set is
#: withheld — that is the fail-closed clause.
#:
#: Assembled from the vocabularies that already exist in the package so it
#: cannot silently disagree with them; ``tests/test_recall_admissibility``
#: pins each source, so a value added to one of them and not here fails the
#: build rather than being quietly withheld from every caller.
RECOGNISED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        # mint vocabulary — mind_mem.enums
        "active",
        *(member.value for member in TaskStatus),
        # not-yet-done — _recall_constants.VALIDITY_STATUS_WIP
        "wip",
        "in-progress",
        "in_progress",
        # done-with, served demoted — _recall_constants.VALIDITY_STATUS_DEAD
        "deprecated",
        "archived",
        "rejected",
        "superseded",
        # committed work — contradiction_detector.COMMITTED_STATUSES
        "monitoring",
        "completed",
        # proposal lifecycle — apply_engine.VALID_STATUSES
        "staged",
        "applied",
        "deferred",
        "expired",
        "rolled_back",
        # open loops — sqlite_index active_only
        "open",
        # governance outcomes already written into live corpora
        "revoked",
        "resolved",
        "pending-review",
        "draft",
        "planning",
        "roadmap",
        "working",
    }
    - UNADMITTED
)


def is_admissible_status(status: object) -> bool:
    """True when *status* names a block state recall may serve.

    Total and fail-closed over arbitrary input. Spelling is normalised
    (case, surrounding space) because a live corpus really does hold
    ``Active`` beside ``active``, and they are one state.

    An **unstated** status is servable. That is not a hole left open by
    accident: since the governed write paths landed, every write carries
    a receipt whose tier stamps a status, so a block with no status can
    only be one written before the gate existed — a new ingest door
    cannot produce one.

    "Unstated" has three on-disk spellings and they mean one thing: the
    field absent (``None``), the field empty (``""``), and the field
    written as a bare ``Status:`` with nothing after it, which the block
    parser renders as an empty **list** under its list-field convention.
    Treating the third differently from the first two would withhold a
    block for how its author spaced a line.
    """
    if status is None:
        return True
    if not isinstance(status, str):
        # Anything else is unstated only if it is empty; a populated
        # non-string is a status this code cannot read, so it is withheld.
        return isinstance(status, (list, tuple, set, frozenset)) and not status
    normalised = status.strip().lower()
    if not normalised:
        return True
    return normalised in RECOGNISED_STATUSES


def _releasable_id_pattern() -> re.Pattern[str]:
    """Ids a release decision may name: the ingest drop corpora, only.

    Derived from the block-id prefix routing table rather than spelled
    out, so a new ingest corpus under ``memory/`` is releasable the
    moment it is routed. The restriction is the point: a release decision
    is a governed instrument for admitting what a *non-servable ingest
    tier* minted, and it must not double as a way to resurrect a
    superseded decision.
    """
    from .block_store import _BLOCK_PREFIX_MAP

    prefixes = sorted(prefix for prefix, (subdir, _file) in _BLOCK_PREFIX_MAP.items() if subdir == "memory")
    return re.compile(rf"^(?:{'|'.join(prefixes)})-[a-zA-Z0-9_.-]+$")


_RELEASE_ID_RE: re.Pattern[str] | None = None


def _release_id_re() -> re.Pattern[str]:
    global _RELEASE_ID_RE
    if _RELEASE_ID_RE is None:
        _RELEASE_ID_RE = _releasable_id_pattern()
    return _RELEASE_ID_RE


def _block_id(block: Mapping[str, Any]) -> str:
    return str(block.get("_id") or "")


def release_ids(blocks: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Ids admitted by an **active** release decision inside *blocks*.

    The source of truth is ``decisions/DECISIONS.md``, written only by
    the apply engine — so an id can only appear here via an approved
    proposal, and a revoked or superseded release decision stops
    admitting its batch, which is what makes rollback re-quarantine for
    free.

    Reading it off the already-parsed block list rather than off disk is
    what removes the filesystem probe from the recall path. Every corpus
    the retrieval legs load (``CORPUS_FILES`` on the scan path,
    ``CORPUS_DIRS`` on the indexed one) contains the decisions file, so
    the answer travels with the question.
    """
    pattern = _release_id_re()
    released: set[str] = set()
    for block in blocks:
        # An ACTIVE release decision, and only an active one: revoking or
        # superseding it stops admitting its batch, which is what makes a
        # governance rollback re-quarantine for free.
        if not is_servable(block.get("Status")):
            continue
        raw = block.get(RELEASE_FIELD)
        if isinstance(raw, str):
            candidates: Sequence[str] = [raw]
        elif isinstance(raw, list):
            candidates = [str(item) for item in raw]
        else:
            continue
        released.update(c.strip() for c in candidates if pattern.match(c.strip()))
    return frozenset(released)


def _admitted(item: Mapping[str, Any], status_key: str, releases: frozenset[str], allow: frozenset[str]) -> bool:
    """``admissible := recognised-and-admitted status, released, or allowed``."""
    status = item.get(status_key)
    if is_admissible_status(status):
        return True
    if isinstance(status, str) and status.strip().lower() in allow:
        return True
    return _block_id(item) in releases


def admissible(
    blocks: Sequence[Mapping[str, Any]],
    *,
    releases: frozenset[str] | None = None,
    status_key: str = "Status",
    allow: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """The ids in *blocks* recall may serve.

    One pass for the release set, one pass for the answer, both over
    headers the caller has already parsed. No IO.
    """
    admitted = releases if releases is not None else release_ids(blocks)
    return frozenset(bid for block in blocks if (bid := _block_id(block)) and _admitted(block, status_key, admitted, allow))


def admit_corpus(
    blocks: Sequence[Mapping[str, Any]],
    *,
    status_key: str = "Status",
    allow: frozenset[str] = frozenset(),
) -> list[dict]:
    """The servable subset of a parsed corpus, order preserved.

    This is the form the block-splicing legs need. Graph expansion, KG
    fusion and entity prefetch each resolve a neighbour id straight out
    of the corpus they were handed, so filtering the corpus is what makes
    a withheld block *unresolvable* to them — closure by construction,
    rather than a filter each of them has to remember to call.

    *allow* readmits named statuses for one call. The only caller is
    ``recall(include_pending=True)``, where the operator has explicitly
    asked to see unreviewed signals; it is a per-call widening with a
    caller behind it, never a default.
    """
    releases = release_ids(blocks)
    return [dict(b) for b in blocks if _admitted(b, status_key, releases, allow)]


def admit_leg(
    hits: Sequence[Mapping[str, Any]],
    *,
    status_key: str = "status",
    releases: frozenset[str] | None = None,
    allow: frozenset[str] = frozenset(),
    leg: str | None = None,
) -> list[dict]:
    """The servable subset of one leg's candidates, order preserved.

    Runs before fusion. *releases* is optional and resolved lazily by the
    caller: a candidate list that is entirely admissible cannot be
    changed by the release set, so the common case never resolves it.
    """
    admitted = releases or frozenset()
    kept = [dict(h) for h in hits if _admitted(h, status_key, admitted, allow)]
    if len(kept) != len(hits):
        _log.info("recall_withheld", leg=leg or "?", withheld=len(hits) - len(kept))
        metrics.inc("recall_withheld_candidates", len(hits) - len(kept))
    return kept


def count_unresolved(n: int = 1) -> None:
    """Record *n* ids that reached a resolution point no block answers.

    Diagnostic only. An unresolvable id is dropped — ``served`` is a
    subset of ``resolved`` is a subset of ``admissible`` — and the run is
    NOT marked degraded, because an index that has outrun the corpus is
    an ordinary consequence of a block being edited between two writes,
    not a failure of the retrieval.
    """
    if n > 0:
        metrics.inc(UNRESOLVED_METRIC, n)


def unresolved_count() -> int | float:
    """Process-wide count of dropped unresolvable ids."""
    return metrics.get(UNRESOLVED_METRIC)


# ---------------------------------------------------------------------------
# The one lookup that touches disk — for paths with no corpus in hand
# ---------------------------------------------------------------------------

#: Single-slot cache keyed on ``(realpath, mtime_ns, size)`` of the decisions
#: file. Any governance apply rewrites it and therefore changes the key, so a
#: release takes effect on the next recall without a reindex.
_RELEASE_CACHE: dict[tuple[str, int, int], frozenset[str]] = {}


#: Single-slot cache keyed on the identity of every corpus file the live
#: status map was built from, so a workspace whose files have not moved
#: since the last resolution re-reads nothing.
_LIVE_STATUS_CACHE: dict[tuple[tuple[str, int, int], ...], dict[str, str]] = {}


def live_statuses(workspace: str) -> Mapping[str, str]:
    """``id -> live status``, or empty when the index's cache can be trusted.

    The FTS/vector indexes carry ``status`` as a **cache** of block state,
    and the indexed legs hand that cached copy straight to the allow-list.
    A cache is only as good as its freshness, and this one goes stale in
    the **fail-open** direction: ``apply_engine``'s governed ``set_status``
    operation flips an already-indexed ``active`` block to ``quarantined``
    in place, and an operator quarantining a block by editing the Markdown
    does the same. Until something reindexes, every indexed leg still reads
    ``active`` — and serves it. An admission decision read from a stale
    cache is not an admission decision.

    Release is deliberately NOT exposed to this: it is resolved live from
    ``decisions/DECISIONS.md`` by :func:`workspace_release_ids`, so a
    release still takes effect with no reindex and the attested index
    anchor stays stable across one.

    Returns empty — costing one staleness check and no corpus load — when
    either the index does not exist (nothing cached, so the hits came from
    a live parse already) or it is current (the cached column agrees with
    the corpus by construction).
    """
    import os

    from .sqlite_index import DB_REL_PATH, is_stale

    if not os.path.isfile(os.path.join(workspace, DB_REL_PATH)):
        return {}
    try:
        if not is_stale(workspace):
            return {}
    except Exception as exc:  # pragma: no cover — defensive
        # Unreadable index: resolve live rather than trust it. Fail-closed.
        _log.warning("index_staleness_check_failed", error=str(exc))

    from ._recall_constants import CORPUS_FILES

    # CORPUS_FILES, not ``MarkdownBlockStore.list_blocks()``. The two
    # disagree, and the difference is exactly the blocks that matter here:
    # the store enumerates ``CORPUS_DIRS`` (decisions/tasks/entities/
    # intelligence) while CORPUS_FILES also names the three ``memory/``
    # drop corpora — IMPORTED, INBOX, MESSAGES — which is where withheld
    # content lives and which the indexer and the scan leg both read. A
    # status map built off the store would silently never cover them.
    paths = [os.path.join(workspace, rel) for rel in sorted(CORPUS_FILES.values())]
    paths = [path for path in paths if os.path.isfile(path)]
    key = tuple(_file_identity(path) for path in paths)
    cached = _LIVE_STATUS_CACHE.get(key)
    if cached is not None:
        return cached

    statuses = _parse_statuses(paths)
    _LIVE_STATUS_CACHE.clear()
    _LIVE_STATUS_CACHE[key] = statuses
    _log.info("live_statuses_resolved", blocks=len(statuses))
    metrics.inc("recall_live_status_resolutions")
    return statuses


def _parse_statuses(paths: Sequence[str]) -> dict[str, str]:
    """``id -> Status`` over *paths*, skipping whatever will not parse.

    A file that cannot be read contributes nothing rather than raising:
    the caller is deciding admissibility, and an unreadable corpus file
    must not take recall down. Its blocks simply keep whatever status
    the leg already carried, which the allow-list then judges.
    """
    from .block_parser import parse_file

    statuses: dict[str, str] = {}
    for path in paths:
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):  # pragma: no cover
            continue
        for block in blocks:
            bid = _block_id(block)
            if bid:
                raw = block.get("Status")
                statuses[bid] = raw if isinstance(raw, str) else ""
    return statuses


def _file_identity(path: str) -> tuple[str, int, int]:
    """``(realpath, mtime_ns, size)`` — the cache key for one corpus file."""
    import os

    try:
        stat = os.stat(path)
    except OSError:  # pragma: no cover — defensive
        return (path, 0, 0)
    return (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)


def with_live_statuses(
    hits: list[dict],
    overrides: Mapping[str, str],
    *,
    status_key: str = "status",
) -> list[dict]:
    """*hits* with every index-cached status replaced by the live one.

    Returns the input object untouched when *overrides* is empty, so the
    fresh-index path — the common one — copies nothing.

    A hit the live map does not name is left alone rather than dropped:
    resolving it is :func:`admit_leg`'s job, and an id the corpus cannot
    answer for is handled as an unresolvable id, not as a status change.
    """
    if not overrides:
        return hits
    refreshed: list[dict] = []
    for hit in hits:
        bid = _block_id(hit)
        item = dict(hit)
        if bid in overrides:
            item[status_key] = overrides[bid]
        refreshed.append(item)
    return refreshed


def workspace_release_ids(workspace: str) -> frozenset[str]:
    """Release set for an indexed path, read from ``decisions/DECISIONS.md``.

    Call this **only** after a candidate has failed the status allow-list:
    an all-servable candidate list cannot be changed by the answer, so
    resolving it eagerly would be a filesystem probe that buys nothing.

    Never raises. An unreadable decisions file admits nothing, which is
    the fail-closed direction.
    """
    import os

    path = os.path.join(workspace, DECISIONS_FILE)
    try:
        stat = os.stat(path)
    except OSError:
        return frozenset()
    key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)
    cached = _RELEASE_CACHE.get(key)
    if cached is not None:
        return cached

    from .block_parser import parse_file

    try:
        blocks = parse_file(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return frozenset()
    result = release_ids(blocks)
    # Single slot: drop older keys rather than grow unbounded.
    _RELEASE_CACHE.clear()
    _RELEASE_CACHE[key] = result
    return result
