"""A snapshot MANIFEST.json cannot make the restore record name foreign blocks.

Triage of the 5.0.2 code-scanning set (see
``security/code-scanning-triage-2026-09-04.md``). The eight open
``py/path-injection`` alerts (#243-#249, #254) all trace one ``receipt_ts``
flow that two guards already stop, and are false positives -- but auditing
the ``apply_engine`` cluster rather than accepting it surfaced a real
traversal on the same line as #245 that no scanner reported. This file is
the regression test for that defect, and for the ``receipt_ts`` format gate
the false-positive rationale rests on.

``restore_snapshot`` builds its evidence record from the manifest's file
list. ``_block_ids_in_snapshot`` used to join each entry onto the snapshot
directory with a bare ``os.path.join`` and parse whatever came back, while
``BlockStore.restore`` — reading the *same* list — routed every entry
through ``_safe_child_path`` and skipped the ones that escaped.

That asymmetry was the bug. The consequence is not only that ids from
outside the snapshot land in ``reinstated_block_ids``: ``restore_snapshot``
computes

    withdrawn = sorted(set(_live_block_ids(ws)) - set(reinstated))

so an entry pointing at a file outside the snapshot could *subtract* ids
from ``withdrawn`` — the one list the record has to spell out, because a
withdrawn block is recoverable from nothing once the restore lands. A
crafted manifest could therefore hide what a restore destroyed.
"""

from __future__ import annotations

import json
import os

from mind_mem.apply_engine import _block_ids_in_snapshot, _manifest_files
from mind_mem.block_parser import parse_file

FOREIGN_ID = "D-29991231-FOREIGN"
INSIDE_ID = "D-29991231-INSIDE"

#: A block header is ``[<ID>]`` (block_parser.parse_blocks, ``^\[([A-Z]+-[^\]]+)\]``);
#: the bracket text IS the ``_id``. Getting this wrong is what the positive
#: controls below caught on the first run.
_BLOCK = "[{bid}]\nStatement: planted for the containment test\nStatus: active\n\n"


def _write_block(path: str, bid: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_BLOCK.format(bid=bid))


def _make_snapshot(tmp_path, manifest_files: list[str]):
    """A snapshot dir, a sibling file outside it, and the manifest naming both."""
    root = tmp_path / "root"
    snap = root / "intelligence" / "applied" / "20260101-000000"
    snap.mkdir(parents=True)

    # Inside the snapshot: a real block the record SHOULD name.
    _write_block(str(snap / "memory" / "DECISIONS.md"), INSIDE_ID)
    # Outside the snapshot: the file a crafted manifest tries to reach.
    _write_block(str(root / "outside.md"), FOREIGN_ID)

    with open(snap / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump({"files": manifest_files, "cleanup_inventory": {}, "version": 2}, fh)
    return root, snap


def test_traversal_entry_is_not_counted_as_reinstated(tmp_path) -> None:
    escape = "../../../outside.md"
    root, snap = _make_snapshot(tmp_path, ["memory/DECISIONS.md", escape])

    # POSITIVE CONTROL 1 — the foreign file exists and really does parse to
    # FOREIGN_ID. Without this the "not in" assertion below would also pass
    # against an empty file, a bad path, or a parser that returns nothing.
    foreign = root / "outside.md"
    assert foreign.is_file(), "fixture did not write the out-of-snapshot file"
    assert FOREIGN_ID in {b.get("_id") for b in parse_file(str(foreign))}

    # POSITIVE CONTROL 2 — the escaping entry really is reachable by a bare
    # join from the snapshot dir, i.e. this fixture exercises the traversal
    # and is not merely naming a path that does not resolve.
    naive = os.path.join(str(snap), escape.replace("/", os.sep))
    assert os.path.isfile(naive), "fixture does not actually traverse out of the snapshot"

    # POSITIVE CONTROL 3 — the manifest reaches the function under test
    # unfiltered, so a green result cannot come from an empty file list.
    files, source = _manifest_files(str(snap))
    assert source == "manifest"
    assert escape in files

    ids = _block_ids_in_snapshot(str(snap), files)

    assert INSIDE_ID in ids, "the in-snapshot block must still be recorded"
    assert FOREIGN_ID not in ids, "a manifest entry escaping the snapshot was parsed into the restore record"


def test_absolute_manifest_entry_is_refused(tmp_path) -> None:
    """``os.path.join`` DISCARDS the root when the second arg is absolute."""
    root, snap = _make_snapshot(tmp_path, ["memory/DECISIONS.md"])
    foreign = str(root / "outside.md")

    # Positive control: the absolute entry names a real, parseable block, and
    # a bare join would hand it straight back.
    assert os.path.join(str(snap), foreign) == foreign
    assert FOREIGN_ID in {b.get("_id") for b in parse_file(foreign)}

    ids = _block_ids_in_snapshot(str(snap), ["memory/DECISIONS.md", foreign])

    assert INSIDE_ID in ids
    assert FOREIGN_ID not in ids, "an absolute manifest entry was parsed into the restore record"


def test_symlinked_entry_is_refused(tmp_path) -> None:
    """A symlink inside the snapshot pointing out of it is resolved and refused."""
    root, snap = _make_snapshot(tmp_path, ["memory/DECISIONS.md"])
    link = snap / "linked.md"
    try:
        link.symlink_to(root / "outside.md")
    except (OSError, NotImplementedError):  # pragma: no cover - Windows w/o privilege
        import pytest

        pytest.skip("symlinks not permitted on this platform")

    # Positive control: the link resolves to the foreign block.
    assert FOREIGN_ID in {b.get("_id") for b in parse_file(str(link))}

    ids = _block_ids_in_snapshot(str(snap), ["memory/DECISIONS.md", "linked.md"])

    assert INSIDE_ID in ids
    assert FOREIGN_ID not in ids, "a symlink escaping the snapshot was parsed into the restore record"


def test_containment_matches_what_the_restore_will_actually_copy(tmp_path) -> None:
    """The record and the restore must agree on which entries count.

    ``BlockStore.restore`` skips an escaping manifest entry, so the file is
    never put back. The record must not claim it was reinstated — the two
    read the same list and now apply the same guard.
    """
    from mind_mem.block_store import _safe_child_path

    _root, snap = _make_snapshot(tmp_path, ["memory/DECISIONS.md"])
    for escaping in ("../../../outside.md", "../outside.md"):
        try:
            _safe_child_path(str(snap), escaping.replace("/", os.sep))
        except ValueError:
            pass
        else:  # pragma: no cover - guard regression
            raise AssertionError(f"restore's own guard accepted {escaping!r}")
        assert FOREIGN_ID not in _block_ids_in_snapshot(str(snap), [escaping])


class TestReceiptTimestampGate:
    """The gate that makes the ``receipt_ts`` path-injection flows unreachable.

    ``rollback`` is the entry point every open ``py/path-injection`` alert
    traces back to (REST ``/v1/rollback_proposal`` -> ``receipt_ts``). What
    makes those flows unreachable is this format gate, which admits no path
    separator at all, running before ``_safe_resolve``.

    ``$`` also matches immediately before a trailing newline, so the gate
    used to accept ``"20260101-000000\\n"``. That never enabled traversal —
    the character class has no separator — but the gate is the stated reason
    the snapshot path is safe, so it has to accept exactly what it claims.
    """

    def _rollback_stdout(self, ws: str, receipt_ts: str) -> tuple[bool, str]:
        import contextlib
        import io as _io

        from mind_mem.apply_engine import rollback

        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = rollback(ws, receipt_ts)
        return ok, buf.getvalue()

    def test_wellformed_timestamp_passes_the_format_gate(self, tmp_path) -> None:
        """POSITIVE CONTROL: a valid stamp gets PAST the format gate.

        Without this, the rejection assertions below would be satisfied by a
        gate that rejects everything.
        """
        ok, out = self._rollback_stdout(str(tmp_path), "20260101-000000")
        assert ok is False  # no such snapshot in an empty workspace
        assert "Invalid receipt timestamp format" not in out
        assert "Snapshot directory not found" in out, out

    def test_trailing_newline_is_rejected_by_the_format_gate(self, tmp_path) -> None:
        ok, out = self._rollback_stdout(str(tmp_path), "20260101-000000\n")
        assert ok is False
        assert "Invalid receipt timestamp format" in out, out

    def test_separators_are_rejected(self, tmp_path) -> None:
        for bad in ("../../etc/passwd", "20260101-000000/../..", "/etc/passwd"):
            ok, out = self._rollback_stdout(str(tmp_path), bad)
            assert ok is False
            assert "Invalid receipt timestamp format" in out, (bad, out)
