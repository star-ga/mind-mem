"""Proposal ids must never be re-minted over ids already in the file.

``generate_resolution_proposals`` scans the existing proposals file for
the highest ``R-<date>-NNN`` and continues from there — the counter
exists precisely so a second batch on the same day does not collide. But
the scan's ``except OSError: pass`` left the counter at 1 and the write
that follows is an APPEND, so an unreadable-but-present file produced a
second ``R-<date>-001..`` series alongside the first, with nothing
logged. Duplicate proposal ids break everything keyed on
``proposal_id``: apply, rollback, the audit trail.
"""

from __future__ import annotations

import builtins
import os
import re
from pathlib import Path

import pytest

from mind_mem.conflict_resolver import ResolutionStrategy, generate_resolution_proposals


def _resolution(n: int) -> dict:
    return {
        "strategy": ResolutionStrategy.TIMESTAMP,
        "confidence": "medium",
        "winner_id": f"D-20260215-{n:03d}",
        "loser_id": f"D-20260101-{n:03d}",
        "contradiction_id": f"C-20260215-{n:03d}",
        "hash_a": "abc123",
        "hash_b": "def456",
        "rationale": "Newer decision wins",
    }


def _workspace(tmp_path: Path) -> str:
    (tmp_path / "intelligence").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


def _proposed_path(ws: str) -> Path:
    return Path(ws) / "intelligence" / "proposed" / "RESOLUTIONS_PROPOSED.md"


@pytest.mark.unit
def test_a_second_batch_continues_the_counter(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    assert generate_resolution_proposals(ws, [_resolution(1)]) == 1
    assert generate_resolution_proposals(ws, [_resolution(2)]) == 1
    ids = re.findall(r"\[(R-\d{8}-\d{3})\]", _proposed_path(ws).read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids)) == 2


@pytest.mark.unit
def test_an_unreadable_proposals_file_refuses_rather_than_duplicating_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path)
    generate_resolution_proposals(ws, [_resolution(1)])
    target = os.path.abspath(_proposed_path(ws))
    before = _proposed_path(ws).read_text(encoding="utf-8")

    real_open = builtins.open

    def refusing_open(file, mode="r", *args, **kwargs):
        if os.path.abspath(str(file)) == target and "r" in mode and "+" not in mode:
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refusing_open)
    with pytest.raises(OSError, match="refusing to append"):
        generate_resolution_proposals(ws, [_resolution(2)])

    monkeypatch.undo()
    # Nothing was appended, so no id was minted twice.
    after = _proposed_path(ws).read_text(encoding="utf-8")
    assert after == before
    ids = re.findall(r"\[(R-\d{8}-\d{3})\]", after)
    assert len(ids) == len(set(ids)) == 1
