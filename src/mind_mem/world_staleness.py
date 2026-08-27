# Copyright 2026 STARGA, Inc.
"""External grounding — detect that the world outside the corpus moved.

:mod:`mind_mem.lineage_staleness` propagates staleness *within* the
corpus: a block goes stale because another block contradicted it.
Nothing in that path notices that the world a block describes has
changed underneath it. A decision that cites ``src/mind_mem/recall.py``
stays confidently "active" long after the file is deleted.

This module closes that gap. It verifies every external anchor a block
cites (:mod:`mind_mem.world_anchors`) against **local, deterministic**
evidence only:

* filesystem existence for ``path`` anchors,
* a per-language definition grep for ``symbol`` anchors
  (:mod:`mind_mem.world_symbol_probe`),
* ``git rev-parse`` / ``merge-base`` for ``git_ref`` anchors
  (:mod:`mind_mem.world_git_probe`).

No network. No model. Same working tree in, same report out — the
verdict is a function of the world's state, nothing else.

The surface is gated by the ``v4.world_staleness`` feature flag, which
is **OFF by default**. With the flag off, ``scan()`` never calls into
this module and its output is byte-identical to the pre-feature output.
Enable it per workspace in ``mind-mem.json``::

    {
      "v4": {
        "world_staleness": {
          "enabled": true,
          "roots": ["../my-repo"],
          "inline": true,
          "max_ref_drift": 0
        }
      }
    }

Nothing here writes to the corpus. :func:`persist_world_staleness` is an
explicit, opt-in write to the same derived ``block_staleness`` index
table :mod:`mind_mem.lineage_staleness` owns — never to a block. Repairs
stay on the governed ``propose_update`` path.

Stdlib only.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

from .observability import get_logger
from .v4.feature_flags import FeatureDisabledError
from .world_anchors import (
    KIND_GIT_REF,
    KIND_INVALID,
    KIND_PATH,
    KIND_SYMBOL,
    Anchor,
    extract_anchors,
)
from .world_git_probe import (
    GIT_LIVE,
    GIT_MISSING_REF,
    GIT_MOVED,
    GIT_UNVERIFIABLE,
    is_git_repo,
    probe_ref,
)
from .world_staleness_config import (
    DEFAULT_MAX_REPORTED,
    FEATURE_FLAG,
    WorldStalenessConfig,
    is_world_staleness_enabled,
    resolve_world_config,
)
from .world_symbol_probe import probe_symbol

__all__ = [
    "DEAD_STATUSES",
    "DEFAULT_MAX_REPORTED",
    "FEATURE_FLAG",
    "STATUS_INVALID",
    "STATUS_LIVE",
    "STATUS_MISSING_PATH",
    "STATUS_MISSING_REF",
    "STATUS_MISSING_SYMBOL",
    "STATUS_REF_MOVED",
    "STATUS_UNVERIFIABLE",
    "AnchorCheck",
    "BlockLiveness",
    "WorldStalenessConfig",
    "WorldStalenessReport",
    "check_block",
    "check_blocks",
    "is_world_staleness_enabled",
    "persist_world_staleness",
    "resolve_world_config",
    "world_staleness_report",
    "world_staleness_summary",
]

_log = get_logger("world_staleness")

STATUS_LIVE: Final = "live"
STATUS_MISSING_PATH: Final = "missing_path"
STATUS_MISSING_SYMBOL: Final = "missing_symbol"
STATUS_MISSING_REF: Final = GIT_MISSING_REF
STATUS_REF_MOVED: Final = GIT_MOVED
STATUS_UNVERIFIABLE: Final = GIT_UNVERIFIABLE
STATUS_INVALID: Final = KIND_INVALID

#: Statuses that mean "the world moved" — the only ones that make a
#: block stale. ``unverifiable`` and ``invalid`` never do: the first is
#: an absent probe, the second a corpus typo.
DEAD_STATUSES: Final[frozenset[str]] = frozenset({STATUS_MISSING_PATH, STATUS_MISSING_SYMBOL, STATUS_MISSING_REF, STATUS_REF_MOVED})

#: ``source_id`` used when world-staleness penalties are persisted into
#: the shared ``block_staleness`` table. Not a block id by design, so a
#: world-derived penalty is always distinguishable from a lineage one.
WORLD_SOURCE_ID: Final = "__world__"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorCheck:
    """One anchor, verified."""

    anchor: Anchor
    status: str
    detail: str = ""
    root: str = ""
    probe: str = ""

    @property
    def is_dead(self) -> bool:
        return self.status in DEAD_STATUSES

    def as_dict(self) -> dict[str, Any]:
        out = self.anchor.as_dict()
        out["status"] = self.status
        if self.detail:
            out["detail"] = self.detail
        if self.root:
            out["root"] = self.root
        if self.probe:
            out["probe"] = self.probe
        return out


@dataclass(frozen=True)
class BlockLiveness:
    """Every anchor of one block, verified."""

    block_id: str
    checks: tuple[AnchorCheck, ...] = ()

    @property
    def dead(self) -> tuple[AnchorCheck, ...]:
        return tuple(c for c in self.checks if c.is_dead)

    @property
    def invalid(self) -> tuple[AnchorCheck, ...]:
        return tuple(c for c in self.checks if c.status == STATUS_INVALID)

    @property
    def is_stale(self) -> bool:
        """True iff at least one cited anchor no longer holds."""
        return bool(self.dead)


@dataclass(frozen=True)
class WorldStalenessReport:
    """The whole corpus, verified against the world."""

    blocks: tuple[BlockLiveness, ...] = ()
    roots: tuple[str, ...] = ()
    missing_roots: tuple[str, ...] = ()
    blocks_scanned: int = 0
    max_reported: int = DEFAULT_MAX_REPORTED

    @property
    def anchored_blocks(self) -> tuple[BlockLiveness, ...]:
        return tuple(b for b in self.blocks if b.checks)

    @property
    def stale_blocks(self) -> tuple[str, ...]:
        return tuple(sorted(b.block_id for b in self.blocks if b.is_stale))

    @property
    def anchors_checked(self) -> int:
        return sum(len(b.checks) for b in self.blocks)

    def _rows(self, predicate: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for block in self.blocks:
            picked = block.dead if predicate == "dead" else block.invalid
            for check in picked:
                row = {"block_id": block.block_id}
                row.update(check.as_dict())
                rows.append(row)
        rows.sort(key=lambda r: (str(r["block_id"]), str(r["kind"]), str(r["raw"])))
        return rows

    def dead_anchors(self) -> list[dict[str, Any]]:
        """Dead anchors as JSON-safe rows, deterministically ordered."""
        return self._rows("dead")

    def invalid_anchors(self) -> list[dict[str, Any]]:
        """Malformed anchors as JSON-safe rows (corpus defects, not drift)."""
        return self._rows("invalid")

    def as_dict(self) -> dict[str, Any]:
        """Bounded, JSON-safe summary — the shape ``scan()`` embeds."""
        dead = self.dead_anchors()
        invalid = self.invalid_anchors()
        return {
            "roots": list(self.roots),
            "missing_roots": list(self.missing_roots),
            "blocks_scanned": self.blocks_scanned,
            "blocks_with_anchors": len(self.anchored_blocks),
            "anchors_checked": self.anchors_checked,
            "stale_blocks": list(self.stale_blocks),
            "dead_anchor_count": len(dead),
            "dead_anchors": dead[: self.max_reported],
            "invalid_anchor_count": len(invalid),
            "invalid_anchors": invalid[: self.max_reported],
            "truncated": len(dead) > self.max_reported or len(invalid) > self.max_reported,
        }


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def _resolve_in_roots(target: str, roots: Sequence[str]) -> str | None:
    """Return the first root under which *target* exists, else ``None``.

    ``target`` is repo-relative and already free of ``..`` / absolute
    prefixes (:mod:`mind_mem.world_anchors` rejects those); the
    containment assertion below is a second fence, not the first.
    """
    relative = target.replace("\\", os.sep).replace("/", os.sep)
    for root in roots:
        candidate = os.path.normpath(os.path.join(root, relative))
        if not (candidate == root or candidate.startswith(root + os.sep)):
            continue
        if os.path.exists(candidate):
            return root
    return None


def _check_path(anchor: Anchor, config: WorldStalenessConfig) -> AnchorCheck:
    if not config.roots:
        return AnchorCheck(anchor, STATUS_UNVERIFIABLE, detail="no readable root configured")
    root = _resolve_in_roots(anchor.target, config.roots)
    if root is None:
        return AnchorCheck(
            anchor,
            STATUS_MISSING_PATH,
            detail=f"'{anchor.target}' not found under {len(config.roots)} configured root(s)",
        )
    return AnchorCheck(anchor, STATUS_LIVE, root=root)


def _check_symbol(anchor: Anchor, config: WorldStalenessConfig) -> AnchorCheck:
    if not config.roots:
        return AnchorCheck(anchor, STATUS_UNVERIFIABLE, detail="no readable root configured")
    root = _resolve_in_roots(anchor.target, config.roots)
    if root is None:
        return AnchorCheck(
            anchor,
            STATUS_MISSING_PATH,
            detail=f"'{anchor.target}' not found under {len(config.roots)} configured root(s)",
        )
    file_path = os.path.normpath(os.path.join(root, anchor.target.replace("/", os.sep)))
    if not os.path.isfile(file_path):
        return AnchorCheck(anchor, STATUS_UNVERIFIABLE, detail=f"'{anchor.target}' is not a regular file", root=root)
    result = probe_symbol(file_path, anchor.symbol, max_bytes=config.max_file_bytes)
    if not result.readable:
        return AnchorCheck(anchor, STATUS_UNVERIFIABLE, detail=result.detail, root=root, probe=result.strength)
    if result.found:
        return AnchorCheck(anchor, STATUS_LIVE, root=root, probe=result.strength)
    return AnchorCheck(
        anchor,
        STATUS_MISSING_SYMBOL,
        detail=f"symbol '{anchor.symbol}' is no longer defined in '{anchor.target}'",
        root=root,
        probe=result.strength,
    )


def _check_git(anchor: Anchor, config: WorldStalenessConfig) -> AnchorCheck:
    for root in config.roots:
        if not is_git_repo(root):
            continue
        try:
            result = probe_ref(root, anchor.target, max_drift=config.max_ref_drift)
        except ValueError as exc:
            return AnchorCheck(anchor, STATUS_INVALID, detail=str(exc), root=root)
        if result.status == GIT_UNVERIFIABLE:
            continue
        status = STATUS_LIVE if result.status == GIT_LIVE else result.status
        return AnchorCheck(anchor, status, detail=result.detail, root=root)
    return AnchorCheck(anchor, STATUS_UNVERIFIABLE, detail="no git work tree among the configured roots")


def _check_anchor(anchor: Anchor, config: WorldStalenessConfig) -> AnchorCheck:
    if anchor.kind == KIND_INVALID:
        return AnchorCheck(anchor, STATUS_INVALID, detail=anchor.reason)
    if anchor.kind == KIND_PATH:
        return _check_path(anchor, config)
    if anchor.kind == KIND_SYMBOL:
        return _check_symbol(anchor, config)
    if anchor.kind == KIND_GIT_REF:
        return _check_git(anchor, config)
    raise ValueError(f"unknown anchor kind: {anchor.kind!r}")  # pragma: no cover - closed set


def check_block(block: Mapping[str, Any], config: WorldStalenessConfig) -> BlockLiveness:
    """Verify every anchor cited by *block* against the local world.

    A block that cites nothing comes back with no checks and
    ``is_stale=False`` — the zero-false-positive guarantee.
    """
    block_id = str(block.get("_id", "") or block.get("id", ""))
    anchors = extract_anchors(block, inline=config.inline)
    checks = tuple(_check_anchor(a, config) for a in anchors)
    return BlockLiveness(block_id=block_id, checks=checks)


def check_blocks(blocks: Iterable[Mapping[str, Any]], config: WorldStalenessConfig) -> WorldStalenessReport:
    """Verify a whole block set. Pure with respect to the corpus — reads only."""
    liveness = tuple(check_block(b, config) for b in blocks)
    return WorldStalenessReport(
        blocks=liveness,
        roots=config.roots,
        missing_roots=config.missing_roots,
        blocks_scanned=len(liveness),
        max_reported=config.max_reported,
    )


def world_staleness_report(
    workspace: str,
    *,
    blocks: Iterable[Mapping[str, Any]] | None = None,
    config: WorldStalenessConfig | None = None,
) -> WorldStalenessReport:
    """Run the world-liveness check over *workspace*'s active blocks.

    Args:
        workspace: Workspace root.
        blocks:    Block set to check; ``None`` enumerates the configured
                   backend's active blocks via
                   :func:`mind_mem.storage.iter_active_blocks`.
        config:    Pre-resolved config; ``None`` resolves from the
                   workspace.

    Raises:
        FeatureDisabledError: the ``v4.world_staleness`` flag is OFF.
    """
    resolved = config if config is not None else resolve_world_config(workspace)
    if not resolved.enabled:
        raise FeatureDisabledError(
            'mind-mem surface \'world_staleness\' is disabled. Enable via mind-mem.json: "v4": { "world_staleness": { "enabled": true } }'
        )
    if blocks is None:
        from .storage import iter_active_blocks

        blocks = iter_active_blocks(workspace)
    return check_blocks(blocks, resolved)


def world_staleness_summary(workspace: str) -> dict[str, Any]:
    """JSON-safe summary for ``scan()``. Only called when the flag is ON."""
    return world_staleness_report(workspace).as_dict()


# ---------------------------------------------------------------------------
# Optional persistence (derived index only — never the corpus)
# ---------------------------------------------------------------------------


def persist_world_staleness(
    workspace: str,
    report: WorldStalenessReport,
    *,
    score: float = 1.0,
) -> dict[str, float]:
    """Write world-derived penalties into the shared ``block_staleness`` table.

    Opt-in and explicit: ``scan()`` never calls this. It reuses the exact
    table :mod:`mind_mem.lineage_staleness` owns — a derived retrieval
    index, not a block of record — under the reserved
    :data:`WORLD_SOURCE_ID`, so the recall reranker demotes
    world-stale blocks with no new machinery. Corpus repairs still go
    through ``propose_update``.

    Returns the ``{block_id: score}`` map that was written.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")

    stale = report.stale_blocks
    if not stale:
        return {}

    from .lineage_staleness import ensure_block_staleness_schema
    from .retrieval_graph import _connect

    ensure_block_staleness_schema(workspace)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    written = {bid: float(score) for bid in stale if bid}
    if not written:
        return {}
    conn = _connect(workspace)
    try:
        conn.executemany(
            """
            INSERT INTO block_staleness (block_id, source_id, score, decayed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(block_id, source_id) DO UPDATE SET
                score = excluded.score,
                decayed_at = excluded.decayed_at
            """,
            [(bid, WORLD_SOURCE_ID, value, now) for bid, value in sorted(written.items())],
        )
        conn.commit()
    finally:
        conn.close()
    return written
