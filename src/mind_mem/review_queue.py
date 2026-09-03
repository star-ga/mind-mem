# Copyright 2026 STARGA, Inc.
"""The read-only proposal queue behind ``mm review``.

Today an operator approves proposals one at a time with no diff and no
evidence inline, and no surface lists what is even pending: ``scan()``
counts SIGNALS, ``approve_apply`` needs an id it will not give you. That
friction is what kills the product, so this module answers one question
completely — *what is waiting, and what would it do* — and answers it
without touching a byte.

Nothing here writes. Nothing here approves. The queue is assembled from
:mod:`mind_mem.apply_engine`'s own discovery and validation functions so
the listing agrees with what the apply would actually accept, and the
order is proposal-id lexicographic so two runs render identically.

Clocks are injected, never read: :meth:`ReviewItem.age_seconds` takes the
"now" it measures against. A queue that renders differently on two
machines at the same commit is a queue you cannot review from a receipt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

__all__ = [
    "QueueHealth",
    "ReviewItem",
    "ReviewQueueError",
    "load_queue",
    "queue_health",
    "parse_timestamp",
]

#: Statuses a proposal must carry to be reviewable.
REVIEWABLE_STATUSES: frozenset[str] = frozenset({"staged"})

#: Environment variable the MCP ACL reads for scope. Reported, never set.
SCOPE_ENV = "MIND_MEM_SCOPE"

#: The scope ``approve_apply`` / ``reject_proposal`` require.
ADMIN_SCOPE = "admin"


class ReviewQueueError(ValueError):
    """The queue could not be read: bad workspace, bad arguments."""


@dataclass(frozen=True)
class ReviewItem:
    """One pending proposal, with everything needed to decide on it."""

    proposal_id: str
    source_file: str
    proposal_type: str
    target_block: str
    risk: str
    status: str
    created: str
    rollback: str
    fingerprint: str
    evidence: tuple[str, ...] = ()
    files_touched: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    ops: tuple[Mapping[str, Any], ...] = ()
    op_summary: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()

    @property
    def applicable(self) -> bool:
        """True when the apply engine would accept this proposal today."""
        return not self.validation_errors

    def age_seconds(self, *, now_iso: str) -> float | None:
        """Seconds between ``Created`` and *now_iso*, or ``None``.

        ``None`` means the age is unknown — the proposal carries no
        ``Created`` field, or one that does not parse. It never means
        zero: a fabricated age would make the published median a lie.
        """
        created = parse_timestamp(self.created)
        now = parse_timestamp(now_iso)
        if created is None or now is None:
            return None
        return (now - created).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_file": self.source_file,
            "type": self.proposal_type,
            "target_block": self.target_block,
            "risk": self.risk,
            "status": self.status,
            "created": self.created,
            "rollback": self.rollback,
            "fingerprint": self.fingerprint,
            "evidence": list(self.evidence),
            "files_touched": list(self.files_touched),
            "sources": list(self.sources),
            "ops": [dict(op) for op in self.ops],
            "op_summary": list(self.op_summary),
            "validation_errors": list(self.validation_errors),
            "applicable": self.applicable,
        }


@dataclass(frozen=True)
class QueueHealth:
    """Governance state that caps how much of the queue can be applied."""

    governance_mode: str
    backlog_count: int
    backlog_over_limit: bool
    no_touch_ok: bool
    no_touch_reason: str
    scope: str = "user"
    blockers: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "governance_mode": self.governance_mode,
            "scope": self.scope,
            "backlog_count": self.backlog_count,
            "backlog_over_limit": self.backlog_over_limit,
            "no_touch_ok": self.no_touch_ok,
            "no_touch_reason": self.no_touch_reason,
            "blockers": list(self.blockers),
        }


def load_queue(
    workspace: str,
    *,
    limit: int | None = None,
    statuses: Sequence[str] = tuple(sorted(REVIEWABLE_STATUSES)),
) -> tuple[ReviewItem, ...]:
    """Every reviewable proposal in *workspace*, id-lexicographic.

    Args:
        workspace: Workspace root.
        limit: Optional truncation, applied after ordering so the first
            page is stable. Must be positive when given.
        statuses: Proposal statuses to include.

    Raises:
        ReviewQueueError: bad workspace or a non-positive ``limit``.
    """
    root = _require_workspace(workspace)
    if limit is not None and limit <= 0:
        raise ReviewQueueError(f"limit must be positive, got {limit!r}")
    wanted = frozenset(statuses)

    from .apply_engine import PROPOSED_FILES
    from .block_parser import parse_file

    items: list[ReviewItem] = []
    for relative in PROPOSED_FILES:
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        items.extend(_build(block, relative) for block in blocks if str(block.get("Status", "")) in wanted)

    items.sort(key=lambda item: item.proposal_id)
    return tuple(items if limit is None else items[:limit])


def queue_health(workspace: str) -> QueueHealth:
    """Governance gates that stand between the queue and an apply.

    ``governance_mode`` is read through :func:`mind_mem.apply_engine._get_mode`
    rather than from ``memory/intel-state.json`` directly, and that is the
    point: this function reports what the apply engine *will do*, so it has
    to read the file the engine reads. While the engine read
    ``intel-state.json`` and the governance gate read ``mind-mem.json``,
    a health report could truthfully quote one file and be wrong about the
    apply. One reader, one file — the attested one.
    """
    root = _require_workspace(workspace)

    from .apply_engine import _get_mode, check_backlog_limit, check_no_touch_window

    mode = str(_get_mode(root))
    backlog_count, over_limit = check_backlog_limit(root)
    touch_ok, touch_reason = check_no_touch_window(root)

    scope = _current_scope()
    blockers: list[str] = []
    if scope != ADMIN_SCOPE:
        blockers.append(
            f"MCP scope is {scope!r}; approving is an admin capability. "
            "Export MIND_MEM_SCOPE=admin to review. This tool will not elevate it for you."
        )
    if mode == "detect_only":
        blockers.append("governance_mode is detect_only in mind-mem.json — no proposal can be applied until it changes")
    if over_limit:
        blockers.append(f"backlog limit reached ({backlog_count} staged) — the apply engine refuses new applies")
    if not touch_ok:
        blockers.append(f"apply rate limit active — {touch_reason}")
    return QueueHealth(
        governance_mode=mode,
        backlog_count=int(backlog_count),
        backlog_over_limit=bool(over_limit),
        no_touch_ok=bool(touch_ok),
        no_touch_reason=str(touch_reason),
        scope=scope,
        blockers=tuple(blockers),
    )


def _current_scope() -> str:
    """The MCP ACL scope this process holds. Read, never written.

    ``mm review`` reports the scope so a denied approval reads as a
    one-line fix instead of an opaque ACL warning. It deliberately does
    not set it: a review front end that grants itself admin is a
    privilege escalation wearing a usability argument.
    """
    return os.environ.get(SCOPE_ENV, "user")


def parse_timestamp(raw: str) -> datetime | None:
    """Parse an ISO-8601 stamp to an aware UTC datetime, or ``None``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _build(block: Mapping[str, Any], source_file: str) -> ReviewItem:
    """Turn one parsed proposal block into an immutable :class:`ReviewItem`."""
    from .apply_engine import validate_proposal

    ops = tuple(MappingProxyType(dict(op)) for op in block.get("Ops", []) if isinstance(op, Mapping))
    return ReviewItem(
        proposal_id=str(block.get("ProposalId") or block.get("_id") or ""),
        source_file=source_file,
        proposal_type=str(block.get("Type", "")),
        target_block=str(block.get("TargetBlock", "")),
        risk=str(block.get("Risk", "")),
        status=str(block.get("Status", "")),
        created=str(block.get("Created", "")),
        rollback=str(block.get("Rollback", "")),
        fingerprint=str(block.get("Fingerprint", "")),
        evidence=_as_tuple(block.get("Evidence")),
        files_touched=_as_tuple(block.get("FilesTouched")),
        sources=_as_tuple(block.get("Sources")),
        ops=ops,
        op_summary=tuple(_summarise(op) for op in ops),
        validation_errors=tuple(validate_proposal(dict(block))),
    )


def _summarise(op: Mapping[str, Any]) -> str:
    """One display line per op: ``<op> <file>:<target>``."""
    return f"{op.get('op', '?')} {op.get('file', '?')}:{op.get('target', 'eof')}"


def _as_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,) if raw.strip() else ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(entry) for entry in raw if str(entry).strip())
    return ()


def _require_workspace(workspace: str) -> str:
    if not isinstance(workspace, str) or not workspace.strip():
        raise ReviewQueueError("workspace must be a non-empty path string")
    root = os.path.realpath(workspace)
    if not os.path.isdir(root):
        raise ReviewQueueError(f"workspace does not exist: {root}")
    return root
