# Copyright 2026 STARGA, Inc.
"""The pre-apply diff ``mm review`` shows next to each proposal.

The apply engine already generates a diff — but only *after* a
successful apply, into the receipt. An operator deciding whether to
approve has never been able to see what a proposal would do, which is
half of why approving is slow and the other half of why it is risky.

The diff here is produced by the production op executors
(:func:`mind_mem.apply_engine.execute_op`) running against a **sandbox
copy** of the touched files, never by a second implementation of op
semantics: a preview that disagrees with the apply is worse than no
preview at all. The sandbox is a temporary directory outside the
workspace, wired to a Markdown store rooted inside it, so a configured
Postgres or encrypted backend is never reached by a preview.

Nothing here writes to the workspace, and nothing here approves.
"""

from __future__ import annotations

import difflib
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .review_queue import ReviewItem

__all__ = ["PreviewResult", "preview_diff"]

#: Cap on rendered diff lines. A proposal that rewrites a whole corpus
#: file should not scroll the review surface off the screen; the count
#: of elided lines is reported so the truncation is never silent.
MAX_DIFF_LINES = 400


@dataclass(frozen=True)
class PreviewResult:
    """What a proposal would change, or why that cannot be shown."""

    proposal_id: str
    available: bool
    diff_text: str = ""
    reason: str = ""
    files: tuple[str, ...] = ()
    truncated_lines: int = 0
    #: Whether the temporary sandbox was actually gone when the preview
    #: returned. Reported, not assumed: it is the real result of the
    #: removal, so a sandbox that survives is visible instead of silent.
    sandbox_removed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "available": self.available,
            "diff_text": self.diff_text,
            "reason": self.reason,
            "files": list(self.files),
            "truncated_lines": self.truncated_lines,
        }


def preview_diff(workspace: str, item: ReviewItem, *, max_lines: int = MAX_DIFF_LINES) -> PreviewResult:
    """Unified diff of what *item* would change, computed in a sandbox.

    Returns an unavailable :class:`PreviewResult` — never raises — when
    the proposal does not validate, names no files, or names a file that
    is not on disk. A refusal with a reason is reviewable; a traceback
    in the middle of a queue listing is not.
    """
    if item.validation_errors:
        return PreviewResult(item.proposal_id, False, reason=f"proposal is not valid: {item.validation_errors[0]}")

    root = os.path.realpath(workspace)
    targets = _targets(item)
    if not targets:
        return PreviewResult(item.proposal_id, False, reason="proposal names no files to change")

    missing = [rel for rel in targets if not os.path.isfile(os.path.join(root, rel))]
    if missing:
        return PreviewResult(item.proposal_id, False, reason=f"file not found in workspace: {missing[0]}")

    from .governance_gate import evict_gate

    sandbox = tempfile.mkdtemp(prefix="mind-mem-review-")
    try:
        _seed(root, sandbox, targets)
        failure = _replay(sandbox, item)
        if failure:
            result = PreviewResult(item.proposal_id, False, reason=failure)
        else:
            text, elided = _render(root, sandbox, targets, max_lines=max_lines)
            if not text:
                result = PreviewResult(item.proposal_id, False, reason="proposal would change nothing", files=targets)
            else:
                result = PreviewResult(item.proposal_id, True, diff_text=text, files=targets, truncated_lines=elided)
    except Exception as exc:  # noqa: BLE001 — see below; one preview may not end the review
        # Containment is by *kind of failure*, not by a list of exception
        # types. ``execute_op`` catches OSError/ValueError and friends, but
        # the governance gate raises ``GovernanceBypassError`` — a bare
        # ``Exception`` subclass — on spec drift and on an admission that
        # will not resolve in the hash chain. Letting that through ended the
        # whole interactive session and discarded every decision the
        # operator had already staged. A refusal with a named reason is
        # reviewable; a traceback mid-queue destroys work.
        result = PreviewResult(item.proposal_id, False, reason=f"preview failed: {type(exc).__name__}: {exc}")
    finally:
        # Retire the sandbox gate BEFORE deleting the directory under it.
        # ``_replay`` opened it through ``get_gate``, which caches one gate
        # per workspace forever; a sandbox is a workspace that exists for
        # one call, so without this the cache accumulates a gate — with its
        # loaded evidence log — for every preview an operator ever runs, all
        # keyed on temp paths that are already deleted.
        try:
            evict_gate(sandbox)
        except Exception:  # noqa: BLE001 — see below
            # Eviction is best-effort and must not be able to skip the removal
            # this finally exists to guarantee. Unguarded, a raise here would
            # leave the sandbox on disk -- the exact failure the block prevents.
            pass
        removed = _remove_sandbox(sandbox)

    return replace(result, sandbox_removed=removed)


def _remove_sandbox(path: str) -> bool:
    """Remove the preview sandbox, read-only files included.

    Returns whether the sandbox is actually gone, so a cleanup that
    fails is *reported* rather than assumed. ``shutil.rmtree(...,
    ignore_errors=True)`` silently leaves the directory behind, which is
    how a sandbox the code promises to always remove came to survive
    without anything saying so.

    On Windows a read-only file makes ``os.unlink`` raise
    PermissionError and rmtree gives up, and ``shutil.copy2`` in
    :func:`_seed` copies a read-only corpus file's mode into the sandbox
    — so this is reachable from an ordinary workspace. Clearing the
    read-only bit and retrying the one failed entry is the standard
    remedy. The parameter was renamed ``onerror`` -> ``onexc`` in 3.12
    and this package supports 3.10 upward, so both spellings are
    handled. ignore_errors remains the final fallback: a sandbox that
    cannot be removed must not end an operator's review.

    One pass is enough, and that is a property of the chain, not luck.
    A previous version of this function retried four times with
    ``gc.collect()`` and a sleep between attempts, on the theory that
    the sandbox's sqlite handles were "released a moment later" by the
    OS. That theory was wrong. sqlite3's context manager commits but
    does not close, and a ``sqlite3.Connection`` is held alive by its
    own prepared-statement cache, so on CPython 3.11+ refcounting never
    reclaims one — only the cyclic collector does. There was no timing
    race: the ``gc.collect()`` in that loop *was* the close, and the
    sleeps around it were decoration. :meth:`HashChainV2._session` now
    closes every connection it opens, so nothing holds the sandbox open
    (measured: seven descriptors into the sandbox before that change,
    zero after, with the collector disabled). Retrying on top of a
    closed handle would only convert a future regression from a red test
    into a green one.
    """

    def _clear_readonly(func: Any, target: str, _exc: Any) -> None:
        # BOTH the entry and its parent. Unlinking a file needs write
        # permission on the DIRECTORY holding it, so clearing only the file's
        # own bit still fails inside a read-only directory.
        try:
            parent = os.path.dirname(target)
            if parent:
                os.chmod(parent, os.stat(parent).st_mode | stat.S_IWRITE | stat.S_IEXEC)
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD | (stat.S_IEXEC if os.path.isdir(target) else 0))
            func(target)
        except OSError:
            pass

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_clear_readonly)
        else:
            shutil.rmtree(path, onerror=_clear_readonly)
    except OSError:
        shutil.rmtree(path, ignore_errors=True)
    return not os.path.exists(path)


def _targets(item: ReviewItem) -> tuple[str, ...]:
    """Workspace-relative files the proposal's ops touch, deduplicated.

    ``FilesTouched`` is proposal-supplied, and the preview *reads* every
    path it names to build the diff — so a path that escapes the
    workspace is an information-disclosure channel into the review panel
    the operator is trusting. Containment is checked on both separators:
    splitting on ``/`` alone let ``..\\..\\etc`` through, which is a live
    escape on Windows and a latent one everywhere.
    """
    named = list(item.files_touched) + [str(op.get("file", "")) for op in item.ops]
    seen: list[str] = []
    for rel in named:
        clean = rel.strip()
        if clean and clean not in seen and not os.path.isabs(clean) and _is_contained(clean):
            seen.append(clean)
    return tuple(seen)


def _is_contained(relative: str) -> bool:
    """True when *relative* cannot climb out of the workspace root."""
    parts = relative.replace("\\", "/").split("/")
    return ".." not in parts and not any(os.path.isabs(part) for part in parts)


def _seed(root: str, sandbox: str, targets: Sequence[str]) -> None:
    """Copy the touched files into the sandbox, preserving relative paths."""
    for rel in targets:
        destination = os.path.join(sandbox, rel)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), destination)


def _replay(sandbox: str, item: ReviewItem) -> str:
    """Run the proposal's ops inside *sandbox*. Returns "" or a reason.

    The ops run under a real admission opened on the **sandbox** gate,
    exactly as ``apply_engine`` opens one before executing a proposal.
    Opening it on the sandbox and not the workspace is the load-bearing
    part: previewing a proposal must not add a chain entry to the real
    workspace for an apply that never happened. The sandbox chain is
    deleted with the sandbox.
    """
    from .apply_engine import execute_op
    from .block_store import MarkdownBlockStore
    from .governance_gate import get_gate

    store = MarkdownBlockStore(sandbox)
    gate = get_gate(sandbox)
    if gate is None:  # pragma: no cover - gate is always present in a real install
        return "governance gate unavailable; refusing to preview ungated"
    with gate.admit_proposal(
        item.proposal_id,
        _ops_digest_input(item),
        actor="review_preview",
        target_file=item.source_file,
        metadata={"proposal_id": item.proposal_id, "phase": "preview"},
    ):
        for index, op in enumerate(item.ops):
            ok, message = execute_op(sandbox, dict(op), store=store)
            if not ok:
                return f"op[{index}] {op.get('op', '?')} would fail: {message}"
    return ""


def _ops_digest_input(item: ReviewItem) -> str:
    """The same content ``apply_engine`` hashes into its admission."""
    import json

    return json.dumps([dict(op) for op in item.ops], default=str)


def _render(root: str, sandbox: str, targets: Sequence[str], *, max_lines: int) -> tuple[str, int]:
    """Unified diff of workspace vs sandbox, truncated to *max_lines*."""
    chunks: list[str] = []
    for rel in targets:
        before = _lines(os.path.join(root, rel))
        after = _lines(os.path.join(sandbox, rel))
        diff = list(difflib.unified_diff(before, after, fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))
        if diff:
            chunks.append("\n".join(diff))
    text = "\n\n".join(chunks)
    if not text:
        return "", 0
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, 0
    elided = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n... {elided} more diff line(s) elided", elided


def _lines(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()
