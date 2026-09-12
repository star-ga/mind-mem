"""Stage typed-KG edge proposals from a block at the moment it is applied.

ROADMAP ("Auto-extract edges on the write path (HITL-gated)", called out as "the
single highest-leverage item"): writing a block should *propose* typed KG edges, and
"**Extracted edges land as proposals, never auto-committed** -- same approval gate as
blocks, honoring the Group H wedge guardrail (source-of-truth graph never
self-modifies)."

WHY THE APPLY STEP AND NOT ``propose_update``. The roadmap names propose_update, which
is where a proposal is born -- but a proposal has no block id yet, and
``RelationTriple.__post_init__`` requires ``source_block_id`` "for provenance".
Staging at propose time would mean inventing an id for a block that may never exist:
if the block proposal is later rejected, the edge proposal survives citing a block that
was never written, and an operator can approve it. The apply step is where the id
becomes real, so that is where edges are staged.

WHY ``graph_ingest.backfill`` AND NOT NEW STAGING CODE. backfill already takes an
injected ``extract_fn`` and a ``restrict_to_blocks`` set, already stages to
SIGNALS.md, and its output is already consumed by ``approve_relation_signals`` -- the
operator gate that commits an edge with a real block id inside an ``admit_edge``
scope. Restricting it to one block id IS the write-path feature. A second staging path
would be a second thing to keep in step with that gate, and this codebase has already
paid for two hand-maintained lists of one concept.

The extractor is ``edge_extraction.candidate_edges`` -- deterministic, no model call --
because backfill's DEFAULT extract_fn is the configured extraction model, and leaving
it defaulted would put a network round-trip on every apply.

OFF AND "FOUND NOTHING" ARE DIFFERENT ANSWERS. A caller that cannot tell them apart
reads silence as "this statement has no relations" when the truth is "nobody looked",
so the outcome is a closed enum and the disabled path says DISABLED.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .observability import get_logger

__all__ = [
    "EDGE_FLAG",
    "StageOutcome",
    "StageResult",
    "extract_fn_for_block",
    "stage_edges_for_block",
]

_log = get_logger("write_path_edges")

#: Flag name as an operator writes it in ``mind-mem.json`` (``v4.<flag>``); the probe
#: below passes the bare suffix as a LITERAL rather than deriving it from this constant,
#: so the flag registry's consumer scan can see it. The bare suffix must appear in
#: ``ALL_V4_FLAGS``: ``is_enabled_quiet`` returns False for any
#: flag outside that tuple, so an undeclared flag is not "off by default" -- it is
#: permanently unreachable, and the feature would look shipped while never running.
EDGE_FLAG = "v4.auto_edges_on_write"


class StageOutcome(Enum):
    """Closed outcome set. Each arm is a distinguishable answer, not a shade of none."""

    DISABLED = "disabled"
    NO_BLOCK_ID = "no-block-id"
    BLOCK_NOT_FOUND = "block-not-found"
    NO_CANDIDATES = "no-candidates"
    STAGED = "staged"
    ERROR = "error"


@dataclass(frozen=True)
class StageResult:
    outcome: StageOutcome
    signals_written: int = 0
    detail: str = ""


def _flag_on() -> bool:
    """Silent probe. Never ``is_enabled`` -- that logs on a malformed config, which
    would make a flag-OFF build observably different from one without the feature."""
    try:
        from .v4.feature_flags import is_enabled_quiet

        # The flag name is a LITERAL here on purpose. Computing it
        # (`EDGE_FLAG.split(".", 1)[-1]`) worked at runtime and defeated
        # `v4/flag_registry.py`'s consumer scan, which reads the source for the
        # name -- so the flag was declared WIRED with "0 consumer(s): none" and
        # the registry's own mutation twin caught it. A flag whose consumer no
        # tool can find is indistinguishable from an unwired one, to the
        # registry and to anyone grepping.
        return bool(is_enabled_quiet("auto_edges_on_write"))
    except Exception:  # pragma: no cover — a probe must never raise
        return False


def _backfill(workspace: str, **kwargs: Any) -> dict:
    """Indirection so the staging authority is patchable in tests by name."""
    from .graph_ingest import backfill

    return backfill(workspace, **kwargs)


def extract_fn_for_block(block_id: str) -> Callable[[str], list[dict[str, Any]]]:
    """Build the ``text -> triples`` extractor ``backfill`` injects, for ONE block.

    A real interface mismatch sits here and the closure is how it is closed honestly.
    ``candidate_edges`` is BLOCK-CENTRIC: the block's own ``_id`` is the subject of
    every candidate it emits, so it needs the id, not just the text. backfill's
    contract hands an extractor only ``text`` -- it assumes a general relation
    extractor that finds subject AND object in the sentence. Passing
    ``{"Statement": text}`` with no ``_id`` makes ``candidate_edges`` return ``[]``
    for every input, which would have staged nothing while every metric read clean.

    Closing over the id is sound ONLY because this stages one block at a time:
    ``stage_edges_for_block`` takes a single id and passes ``restrict_to_blocks`` as
    that one id. If a caller ever widened the restriction while reusing this closure,
    every block's edges would be attributed to one subject -- a provenance error that
    an operator approving the edge could not see. Hence one id in, one id out, and no
    parameter to widen.

    ``confidence`` is deliberately absent, so backfill's own default (0.5) applies:
    the extractor is a deterministic pattern match and makes no probabilistic claim.
    Stamping a high confidence would present a phrase match as a judgement.
    """
    from .edge_extraction import candidate_edges

    bid = str(block_id or "").strip()

    def _extract(text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cand in candidate_edges({"_id": bid, "Statement": text}) or []:
            subject = str(cand.get("subject") or "").strip()
            predicate = str(cand.get("predicate") or "").strip()
            obj = str(cand.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            rows.append({"subject": subject, "predicate": predicate, "object": obj})
        return rows

    return _extract


def stage_edges_for_block(workspace: str, block_id: str) -> StageResult:
    """Stage edge PROPOSALS for one just-applied block. Never writes the graph.

    The block's TEXT is not a parameter: ``backfill`` already loads the corpus and
    already knows where a block keeps its text (``excerpt`` / ``content`` /
    ``Statement``). Passing the statement in would put a second copy of that
    knowledge here, to drift from the first.

    Never raises. This runs after the block is already committed, so raising would
    report failure for a completed apply. But a swallowed error would make "writing a
    block proposes typed KG edges" unfalsifiable, so a failure returns ``ERROR`` with
    its reason and logs -- visible, not silent.
    """
    if not _flag_on():
        return StageResult(StageOutcome.DISABLED)

    bid = str(block_id or "").strip()
    if not bid:
        # An edge needs a real source block. Staging against "" would put an empty
        # string where downstream readers expect provenance.
        return StageResult(StageOutcome.NO_BLOCK_ID, detail="no block id to attribute edges to")

    try:
        metrics = _backfill(
            workspace,
            extract_fn=extract_fn_for_block(bid),
            # EXACTLY this block. None would re-scan the whole corpus on every apply;
            # an empty collection would scan nothing while reporting success.
            restrict_to_blocks=[bid],
            # False, or nothing is staged at all -- backfill's default is a dry-run
            # measurement. Still never a direct graph write.
            dry_run=False,
        )
    except Exception as exc:  # noqa: BLE001 — the apply already succeeded
        _log.warning("write_path_edges_failed", block_id=bid, error=str(exc))
        return StageResult(StageOutcome.ERROR, detail=f"{type(exc).__name__}: {exc}")

    metrics = metrics or {}
    # "Scanned nothing" must never read as "found no edges". A block absent from the
    # loaded corpus -- wrong workspace, an id the loader does not index, a store that
    # has not caught up -- produces the same zero as a statement with no relations,
    # and only one of those is a healthy answer.
    if int(metrics.get("blocks_scanned") or 0) == 0:
        return StageResult(
            StageOutcome.BLOCK_NOT_FOUND,
            detail=f"{bid} was not found in the loaded corpus, so nothing was examined",
        )
    written = int(metrics.get("signals_written") or 0)
    if written == 0:
        return StageResult(StageOutcome.NO_CANDIDATES)
    return StageResult(StageOutcome.STAGED, signals_written=written)
