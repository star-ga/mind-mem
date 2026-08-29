# Copyright 2026 STARGA, Inc.
"""The keyboard-driven review loop: one keystroke, one decision.

The target is thirty proposals in forty seconds, and the way it is met
is the boring way — the diff, the evidence and the chain status are
already on screen when the operator arrives at a proposal, so the only
thing left to do is press a key.

**One operator keystroke, one decision.** There is no select-all, no
"approve the rest", no risk shortcut. That is not a UX oversight: it is
what makes "a human approved this" true for every applied proposal, and
it costs thirty keystrokes to clear thirty proposals, which is seconds.

The loop reads from an injected key source and writes to an injected
stream, so it is exercised end-to-end in tests without a terminal. It
returns decisions and applies nothing — :func:`mind_mem.review_batch.run_batch`
does the applying, through the governed path.

An exhausted key source returns **no decisions**. A closed pipe is not
consent.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence, TextIO

from .review_batch import ReviewBatchError, ReviewDecision
from .review_evidence import gather
from .review_preview import preview_diff
from .review_queue import ReviewItem
from .review_render import render_detail

__all__ = ["KEY_HELP", "read_keys", "review_session"]

KEY_APPROVE = "a"
KEY_REJECT = "r"
KEY_SKIP = "s"
KEY_SKIP_ALT = " "
KEY_BACK = "b"
KEY_UNDO = "u"
KEY_COMMIT = "c"
KEY_QUIT = "q"

KEY_HELP = "[a] approve  [r] reject  [s]/space skip  [b] back  [u] undo  [c] commit  [q] quit without applying"


def review_session(
    workspace: str,
    items: Sequence[ReviewItem],
    *,
    keys: Iterable[str],
    out: TextIO,
    reason_prompt: Callable[[str], str] | None = None,
) -> tuple[ReviewDecision, ...]:
    """Walk *items*, collecting explicit operator decisions.

    Args:
        workspace: Workspace root, used to render evidence and diffs.
        items: The queue, in listing order.
        keys: Single-character operator actions. Exhaustion discards.
        out: Where the review surface is drawn.
        reason_prompt: Called with a proposal id when the operator
            rejects; must return the rationale. A blank rationale drops
            the decision rather than rejecting without an audit trail.

    Returns:
        The decisions the operator committed, in decision order. Empty
        when the operator quit or the key source ran out.
    """
    if not items:
        out.write("No proposals pending review.\n")
        return ()

    decided: dict[str, ReviewDecision] = {}
    order: list[str] = []
    cursor = 0
    stream: Iterator[str] = iter(keys)
    _draw(workspace, items, cursor, decided, out)

    for key in stream:
        action = key.lower()
        if action == KEY_QUIT:
            out.write("Quit — nothing applied.\n")
            return ()
        if action == KEY_COMMIT:
            return tuple(decided[pid] for pid in order if pid in decided)
        cursor = _handle(workspace, items, cursor, decided, order, action, out, reason_prompt)
        if cursor >= len(items):
            cursor = len(items) - 1
            out.write(f"\nEnd of queue. {len(decided)} decision(s) staged. [c] commit  [q] quit\n")
        else:
            _draw(workspace, items, cursor, decided, out)

    out.write("Input ended before commit — nothing applied.\n")
    return ()


def _handle(
    workspace: str,
    items: Sequence[ReviewItem],
    cursor: int,
    decided: dict[str, ReviewDecision],
    order: list[str],
    action: str,
    out: TextIO,
    reason_prompt: Callable[[str], str] | None,
) -> int:
    """Apply one keystroke to the session state. Returns the new cursor."""
    item = items[cursor]
    if action == KEY_APPROVE:
        _record(decided, order, ReviewDecision(item.proposal_id, "approve", origin="keypress"))
        return cursor + 1
    if action == KEY_REJECT:
        _reject(item, decided, order, out, reason_prompt)
        return cursor + 1
    if action in (KEY_SKIP, KEY_SKIP_ALT):
        _forget(decided, order, item.proposal_id)
        return cursor + 1
    if action == KEY_BACK:
        return max(0, cursor - 1)
    if action == KEY_UNDO:
        if order:
            _forget(decided, order, order[-1])
        return cursor
    out.write(f"Unknown key {action!r}. {KEY_HELP}\n")
    return cursor


def _reject(
    item: ReviewItem,
    decided: dict[str, ReviewDecision],
    order: list[str],
    out: TextIO,
    reason_prompt: Callable[[str], str] | None,
) -> None:
    """Collect a rationale and stage a rejection, or drop it."""
    reason = reason_prompt(item.proposal_id) if reason_prompt else ""
    try:
        _record(decided, order, ReviewDecision(item.proposal_id, "reject", origin="keypress", reason=reason))
    except ReviewBatchError as exc:
        _forget(decided, order, item.proposal_id)
        out.write(f"Rejection dropped: {exc}\n")


def _record(decided: dict[str, ReviewDecision], order: list[str], decision: ReviewDecision) -> None:
    if decision.proposal_id not in decided:
        order.append(decision.proposal_id)
    decided[decision.proposal_id] = decision


def _forget(decided: dict[str, ReviewDecision], order: list[str], proposal_id: str) -> None:
    decided.pop(proposal_id, None)
    if proposal_id in order:
        order.remove(proposal_id)


def _draw(
    workspace: str,
    items: Sequence[ReviewItem],
    cursor: int,
    decided: dict[str, ReviewDecision],
    out: TextIO,
) -> None:
    """Render the proposal under the cursor, with its diff and evidence."""
    item = items[cursor]
    staged = decided.get(item.proposal_id)
    out.write("\n")
    out.write(f"[{cursor + 1}/{len(items)}]  staged decision: {staged.action if staged else '-'}\n")
    out.write(render_detail(item, preview_diff(workspace, item), gather(workspace, item)))
    out.write(f"\n{KEY_HELP}\n")


def read_keys(stream: TextIO) -> Iterator[str]:
    """Yield single keypresses from a terminal, or characters from a pipe.

    Falls back to buffered characters whenever raw mode is unavailable
    (a pipe, a Windows console, a stream with no file descriptor), so
    the session is drivable from a script as well as a keyboard.
    """
    try:
        import termios
        import tty

        descriptor = stream.fileno()
        saved = termios.tcgetattr(descriptor)
    except Exception:  # noqa: BLE001 — no tty: fall back to buffered input
        yield from _buffered(stream)
        return

    try:
        tty.setcbreak(descriptor)
        while True:
            char = stream.read(1)
            if not char:
                return
            yield char
    finally:
        import termios as _termios

        _termios.tcsetattr(descriptor, _termios.TCSADRAIN, saved)


def _buffered(stream: TextIO) -> Iterator[str]:
    for line in stream:
        for char in line.strip() or "\n":
            yield char
