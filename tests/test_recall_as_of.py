"""Tests for the ``recall(..., as_of=)`` time-travel plumb-through (roadmap Group B).

The versioning store (``block_edits``) and ``content_as_of()`` reconstruction
are covered in ``test_v4_block_versioning.py``. This file covers only the
recall-side wiring added to expose ``as_of`` on the recall entrypoint:
``_apply_as_of_projection`` and its routing through ``_apply_post_filters``.

Contract under test:
  * before/at/after each edit, a hit's ``content`` is rewound to the revision
    that was in effect at ``as_of`` (a projection — the result set is unchanged);
  * a block with no recorded edits keeps its current content;
  * when ``self_editing`` is disabled the projection is a graceful no-op
    (current content, no raise) — recall never fails because ``as_of`` was passed;
  * the input hit dicts are never mutated (immutable projection).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mind_mem._recall_core import _apply_as_of_projection, _apply_post_filters
from mind_mem.v4.block_versioning import FLAG as BV_FLAG
from mind_mem.v4.self_editing import EditStatus, ensure_edit_schema

_BEFORE = "2026-01-01T00:00:00+00:00"  # before v2 (alpha era)
_MID = "2026-02-01T00:00:00+00:00"  # after alpha->beta, before beta->gamma
_AFTER = "2026-04-01T00:00:00+00:00"  # after all edits (gamma era)


def _cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: bool) -> Path:
    block = {k: {"enabled": v} for k, v in flags.items()}
    (tmp_path / "mind-mem.json").write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


def _seed_applied_edit(workspace: Path, block_id: str, old: str | None, new: str, approved_at: str) -> None:
    ensure_edit_schema(workspace)
    with sqlite3.connect(workspace / "index.db", timeout=30) as conn:
        conn.execute(
            "INSERT INTO block_edits (block_id, old_content, new_content, reason, "
            "proposed_at, status, approved_at, approver) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (block_id, old, new, "r", approved_at, EditStatus.APPLIED, approved_at, "tester"),
        )
        conn.commit()


def _seed_chain(workspace: Path, block_id: str = "B-1") -> None:
    """v1 'alpha' -> v2 'beta' (2026-01-02) -> v3 'gamma' (2026-03-04)."""
    _seed_applied_edit(workspace, block_id, "alpha", "beta", "2026-01-02T00:00:00+00:00")
    _seed_applied_edit(workspace, block_id, "beta", "gamma", "2026-03-04T00:00:00+00:00")


@pytest.fixture
def ws_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = _cfg(tmp_path, monkeypatch, **{BV_FLAG: True})
    _seed_chain(ws)
    return ws


@pytest.fixture
def ws_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # No history seeded: the disabled-surface guard returns before ever
    # reading the edit store (and seeding would itself require the flag on).
    return _cfg(tmp_path, monkeypatch, **{BV_FLAG: False})


def _hits() -> list[dict]:
    # current live content is the tail of the chain ('gamma').
    return [{"_id": "B-1", "content": "gamma", "score": 1.0}]


@pytest.mark.unit
@pytest.mark.parametrize(("as_of", "expected"), [(_BEFORE, "alpha"), (_MID, "beta"), (_AFTER, "gamma")])
def test_as_of_rewinds_content_to_the_right_revision(ws_on: Path, as_of: str, expected: str) -> None:
    out = _apply_as_of_projection(_hits(), str(ws_on), as_of)
    assert out[0]["content"] == expected
    assert out[0]["valid_as_of"] == as_of
    assert out[0]["score"] == 1.0  # projection leaves score/ordering untouched


@pytest.mark.unit
def test_block_without_history_keeps_current_content(ws_on: Path) -> None:
    hits = [{"_id": "B-NONE", "content": "live", "score": 1.0}]
    out = _apply_as_of_projection(hits, str(ws_on), _BEFORE)
    assert out[0]["content"] == "live"
    assert "valid_as_of" not in out[0]


@pytest.mark.unit
def test_flag_off_is_graceful_noop(ws_off: Path) -> None:
    out = _apply_as_of_projection(_hits(), str(ws_off), _BEFORE)
    assert out[0]["content"] == "gamma"  # current content, no rewind
    assert "valid_as_of" not in out[0]


@pytest.mark.unit
def test_projection_does_not_mutate_input(ws_on: Path) -> None:
    hits = _hits()
    _apply_as_of_projection(hits, str(ws_on), _BEFORE)
    assert hits[0]["content"] == "gamma"
    assert "valid_as_of" not in hits[0]


@pytest.mark.unit
def test_post_filters_routes_as_of(ws_on: Path) -> None:
    out = _apply_post_filters(
        _hits(),
        since=None,
        until=None,
        lifecycle=None,
        event_id=None,
        min_maturity=None,
        limit=10,
        workspace=str(ws_on),
        as_of=_BEFORE,
    )
    assert out[0]["content"] == "alpha"


@pytest.mark.unit
def test_post_filters_as_of_none_is_noop(ws_on: Path) -> None:
    out = _apply_post_filters(
        _hits(),
        since=None,
        until=None,
        lifecycle=None,
        event_id=None,
        min_maturity=None,
        limit=10,
        workspace=str(ws_on),
        as_of=None,
    )
    assert out[0]["content"] == "gamma"
    assert "valid_as_of" not in out[0]
