"""LAST_WRITER_WINS must converge the version vector, not just the log.

Regression: ``resolve_conflict`` upserted the *winner's own* version into
``block_tier_vclock`` through ``MAX(excluded.version, ...)``. Under
LAST_WRITER_WINS the winner is the most recent wall-clock writer, which is
routinely NOT the highest-version agent, so the MAX() kept the leading fork
exactly where it was and the "convergence" was a no-op. Observable damage:

  1. ``detect_conflict`` immediately re-discovered the identical pair, and
     ``_log_conflict`` opened a *second* row for it (the dedupe filter is
     ``resolution IS NULL``, and the first row is now resolved) — so the
     conflict log grew without bound, one row per resolve.
  2. The same loop stamped ``last_seen_at = <resolution time>`` on both
     agents, erasing the only ordering LAST_WRITER_WINS reads. The re-run
     therefore tie-broke on version and named the *other* agent the winner,
     leaving two rows for one pair with contradictory ``resolved_to``.

These tests drive the exact shape from the audit: agent A writes five times,
agent B writes three times but later in wall-clock order.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from mind_mem.v4.federation import (
    MergeStrategy,
    detect_conflict,
    get_version_vector,
    record_agent_write,
    resolve_conflict,
)

BLOCK = "B-lww-convergence"


@pytest.fixture
def fed_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace with the v4 federation flag enabled."""
    cfg = tmp_path / "fed-on.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _diverge(ws: Path, block_id: str = BLOCK) -> None:
    """A reaches version 5; B reaches version 3 but writes LAST."""
    for _ in range(5):
        record_agent_write(ws, block_id, "agent-A")
    # Wall-clock separation so `last_seen_at` orders B after A.
    time.sleep(0.01)
    for _ in range(3):
        record_agent_write(ws, block_id, "agent-B")


def _conflict_rows(ws: Path, block_id: str = BLOCK) -> list[tuple[str, int, str, int, str | None, str | None]]:
    with sqlite3.connect(ws / "index.db") as conn:
        return conn.execute(
            "SELECT left_agent, left_version, right_agent, right_version, resolution, resolved_to "
            "FROM tier_conflict_log WHERE block_id = ? ORDER BY rowid",
            (block_id,),
        ).fetchall()


def _timestamps(ws: Path, block_id: str = BLOCK) -> dict[str, str]:
    with sqlite3.connect(ws / "index.db") as conn:
        rows = conn.execute(
            "SELECT agent_id, last_seen_at FROM block_tier_vclock WHERE block_id = ?",
            (block_id,),
        ).fetchall()
    return {str(agent): str(ts) for agent, ts in rows}


@pytest.mark.unit
def test_last_writer_wins_converges_both_forks(fed_on: Path) -> None:
    """The losing-but-leading fork must be pulled down to a tie.

    Pre-fix the vclock stayed ``{agent-A: 5, agent-B: 3}`` because the
    upsert carried winner_version (3) and ``MAX(3, 5)`` is 5.
    """
    _diverge(fed_on)
    resolution = resolve_conflict(fed_on, BLOCK, MergeStrategy.LAST_WRITER_WINS)
    assert resolution is not None
    # The last wall-clock writer wins even though it has the lower clock.
    assert resolution.winner_agent == "agent-B"
    # `winner_version` is documented as the winning SIDE's own version.
    assert resolution.winner_version == 3

    # ...but the block converges at the pointwise max, so neither fork
    # still looks like an independent writer.
    assert get_version_vector(fed_on, BLOCK) == {"agent-A": 5, "agent-B": 5}
    assert detect_conflict(fed_on, BLOCK) is None


@pytest.mark.unit
def test_resolved_conflict_is_not_relogged_or_re_won(fed_on: Path) -> None:
    """One conflict, one log row, one winner — however often you resolve.

    Pre-fix: the second resolve re-detected the same pair, inserted a
    second open row, and returned ``agent-A`` — contradicting the
    ``resolved_to = 'agent-B'`` already persisted for the same pair.
    """
    _diverge(fed_on)
    first = resolve_conflict(fed_on, BLOCK, MergeStrategy.LAST_WRITER_WINS)
    assert first is not None
    second = resolve_conflict(fed_on, BLOCK, MergeStrategy.LAST_WRITER_WINS)
    assert second is None, "a resolved conflict must not be resolvable a second time"

    rows = _conflict_rows(fed_on)
    assert len(rows) == 1, f"conflict log grew on re-resolution: {rows}"
    assert rows[0][:4] == ("agent-A", 5, "agent-B", 3)
    assert rows[0][4] == MergeStrategy.LAST_WRITER_WINS.value
    assert rows[0][5] == first.winner_agent


@pytest.mark.unit
def test_resolution_preserves_per_agent_last_seen_at(fed_on: Path) -> None:
    """Resolving is not a write: it must not restamp anyone's clock.

    ``last_seen_at`` is the only ordering LAST_WRITER_WINS reads. Pre-fix
    every party was stamped with the resolution time — identical strings —
    which silently degraded the strategy into HIGHER_VERSION.
    """
    _diverge(fed_on)
    before = _timestamps(fed_on)
    assert before["agent-A"] != before["agent-B"]

    resolve_conflict(fed_on, BLOCK, MergeStrategy.LAST_WRITER_WINS)

    after = _timestamps(fed_on)
    assert after == before, "resolve_conflict overwrote the per-agent write timestamps"


@pytest.mark.unit
def test_higher_version_still_converges(fed_on: Path) -> None:
    """Guard the untouched strategy: HIGHER_VERSION already converged
    (its winner IS the max-version agent) and must keep doing so."""
    _diverge(fed_on)
    resolution = resolve_conflict(fed_on, BLOCK, MergeStrategy.HIGHER_VERSION)
    assert resolution is not None
    assert resolution.winner_agent == "agent-A"
    assert resolution.winner_version == 5
    assert get_version_vector(fed_on, BLOCK) == {"agent-A": 5, "agent-B": 5}
    assert detect_conflict(fed_on, BLOCK) is None


@pytest.mark.unit
def test_three_way_merge_converges_above_both_forks(fed_on: Path) -> None:
    """Guard the untouched strategy: a merge mints a fresh version that
    dominates both forks, and the synthetic merge agent joins the vclock
    at that same version rather than becoming a third fork."""
    _diverge(fed_on)
    resolution = resolve_conflict(
        fed_on,
        BLOCK,
        MergeStrategy.THREE_WAY_MERGE,
        merger=lambda _report: b"merged",
    )
    assert resolution is not None
    assert resolution.winner_version == 6
    assert get_version_vector(fed_on, BLOCK) == {
        "agent-A": 6,
        "agent-B": 6,
        resolution.winner_agent: 6,
    }
    assert detect_conflict(fed_on, BLOCK) is None
