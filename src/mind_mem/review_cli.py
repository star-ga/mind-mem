# Copyright 2026 STARGA, Inc.
"""``mm review`` — the batch approval surface.

Front end only. Every approval leaves through
:func:`mind_mem.review_batch.governed_approve` and every rejection
through ``governed_reject``, which are the same ``approve_apply`` /
``reject_proposal`` entry points the MCP server exposes. This module
lists, renders, and collects operator decisions; it does not write.

There is no unattended mode and no risk shortcut. ``--approve`` names
proposals explicitly, the interactive session takes one keystroke per
proposal, and nothing else can produce a decision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from .review_batch import BatchReport, ReviewBatchError, ReviewDecision, run_batch
from .review_evidence import gather
from .review_preview import preview_diff
from .review_queue import ReviewItem, ReviewQueueError, load_queue, queue_health
from .review_render import render_detail, render_health, render_queue, render_report
from .review_session import read_keys, review_session

__all__ = ["add_review_parser", "cmd_review"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def add_review_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``mm review`` on *subparsers* and return its parser."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "review",
        help="Batch-review the HITL proposal queue: diff, evidence, approve/reject.",
        description=(
            "List pending proposals with their pre-apply diff, provenance, chain status "
            "and staleness inline, then approve or reject them in a batch. Every approval "
            "routes through the governed approve_apply path; there is no auto-approval."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument("--limit", type=int, default=None, help="Show at most N proposals.")
    parser.add_argument("--show", metavar="PROPOSAL_ID", default="", help="Full detail for one proposal.")
    parser.add_argument(
        "--approve",
        metavar="IDS",
        default="",
        help="Comma-separated proposal ids to approve. Each is named explicitly.",
    )
    parser.add_argument("--reject", metavar="IDS", default="", help="Comma-separated proposal ids to reject.")
    parser.add_argument("--reason", default="", help="Rationale for --reject (required, >= 8 characters).")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Keyboard session: one keystroke per proposal, then commit.",
    )
    parser.set_defaults(func=cmd_review)
    return parser


def cmd_review(args: argparse.Namespace) -> int:
    """Dispatch ``mm review`` to list, show, batch or interactive mode."""
    from .mm_cli import _workspace

    workspace = _workspace()
    now_iso = _now_iso()
    started = time.perf_counter()
    try:
        items = load_queue(workspace, limit=args.limit)
        health = queue_health(workspace)
    except ReviewQueueError as exc:
        print(f"mm review: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.show:
        # Deliberately re-reads the full queue: --limit pages the listing,
        # and paging must never hide a proposal from a direct lookup.
        return _show(workspace, load_queue(workspace), args.show, as_json=args.json)
    if args.approve or args.reject:
        _warn_blockers(health, as_json=args.json)
        return _batch(workspace, args, now_iso=now_iso, started=started)
    if args.interactive:
        _warn_blockers(health, as_json=False)
        return _interactive(workspace, items, now_iso=now_iso, started=started)
    return _list(items, health, now_iso=now_iso, as_json=args.json)


def _warn_blockers(health: Any, *, as_json: bool) -> None:
    """Name the governance gates *before* the operator spends decisions.

    The gates were always computed and only ever rendered in the listing,
    so an operator in ``-i`` pressed thirty keys and then discovered that
    twenty-nine of the applies were rate-limited or scope-denied. A
    blocker reported after the work is an epitaph, not a warning.

    ``mm review`` reports them and works around none of them: a front end
    that quietly lifted a governance rate limit would be the auto-approve
    path wearing a usability argument.
    """
    if as_json or not getattr(health, "blockers", ()):
        return
    print(render_health(health))
    print()


def _list(items: Sequence[ReviewItem], health: Any, *, now_iso: str, as_json: bool) -> int:
    if as_json:
        payload = {
            "queue": [_with_age(item, now_iso) for item in items],
            "health": health.to_dict(),
            "now": now_iso,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_queue(items, health, now_iso=now_iso))
    return EXIT_OK


def _show(workspace: str, items: Sequence[ReviewItem], proposal_id: str, *, as_json: bool) -> int:
    item = next((candidate for candidate in items if candidate.proposal_id == proposal_id), None)
    if item is None:
        print(f"mm review: no pending proposal {proposal_id}", file=sys.stderr)
        return EXIT_FAILED
    preview = preview_diff(workspace, item)
    panel = gather(workspace, item)
    if as_json:
        print(
            json.dumps(
                {"proposal": item.to_dict(), "preview": preview.to_dict(), "evidence": panel.to_dict()},
                indent=2,
                default=str,
            )
        )
    else:
        print(render_detail(item, preview, panel))
    return EXIT_OK


def _batch(workspace: str, args: argparse.Namespace, *, now_iso: str, started: float) -> int:
    """Approve/reject the proposals named on the command line."""
    try:
        decisions = _decisions_from_flags(args)
    except ReviewBatchError as exc:
        print(f"mm review: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return _commit(workspace, decisions, now_iso=now_iso, as_json=args.json, started=started)


def _decisions_from_flags(args: argparse.Namespace) -> tuple[ReviewDecision, ...]:
    """Build decisions from ``--approve`` / ``--reject``. Origin: cli-flag."""
    approvals = tuple(ReviewDecision(pid, "approve", origin="cli-flag") for pid in _split(args.approve))
    rejections = tuple(ReviewDecision(pid, "reject", origin="cli-flag", reason=args.reason) for pid in _split(args.reject))
    return approvals + rejections


def _interactive(workspace: str, items: Sequence[ReviewItem], *, now_iso: str, started: float) -> int:
    decisions = review_session(
        workspace,
        items,
        keys=read_keys(sys.stdin),
        out=sys.stdout,
        reason_prompt=_prompt_reason,
    )
    if not decisions:
        print("No decisions committed.")
        return EXIT_OK
    return _commit(workspace, decisions, now_iso=now_iso, as_json=False, started=started)


def _commit(
    workspace: str,
    decisions: Sequence[ReviewDecision],
    *,
    now_iso: str,
    as_json: bool,
    started: float = 0.0,
) -> int:
    """Run the batch and publish the throughput it actually achieved.

    ``started`` is the perf-counter reading from the top of the
    invocation, so the published rate covers deciding *and* applying —
    the operator's real session, not the apply engine's slice of it.
    """
    try:
        report: BatchReport = run_batch(workspace, decisions, now_iso=now_iso, session_started=started)
    except ReviewBatchError as exc:
        print(f"mm review: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_report(report))
    return EXIT_FAILED if report.failed else EXIT_OK


def _prompt_reason(proposal_id: str) -> str:
    """Ask the operator why. A blank answer drops the rejection."""
    try:
        return input(f"Reason for rejecting {proposal_id} (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _with_age(item: ReviewItem, now_iso: str) -> dict[str, Any]:
    payload = item.to_dict()
    payload["age_seconds"] = item.age_seconds(now_iso=now_iso)
    return payload


def _split(raw: str) -> tuple[str, ...]:
    return tuple(entry.strip() for entry in (raw or "").split(",") if entry.strip())


def _now_iso() -> str:
    """Wall clock, read once per invocation and threaded down explicitly.

    Every module below this one takes the clock as an argument, so the
    listing, the ages and the metric all agree and none of them reads
    the clock behind the caller's back.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
