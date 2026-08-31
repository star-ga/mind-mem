# Copyright 2026 STARGA, Inc.
"""``TierManager`` must be closeable, or every instance leaks descriptors.

``TierManager.__init__`` builds a ``ConnectionManager`` (memory_tiers.py:218)
and the class has **no** ``close``, no ``__del__`` and no context-manager
protocol. Each instance therefore holds an open SQLite read connection --
three descriptors on disk: the db, its ``-wal`` and its ``-shm`` -- for the
life of the process.

``BlockMetadata`` in the sibling module does the same thing and DOES have a
``close`` (block_metadata.py:321), so this is an omission rather than a
design choice.

The symptom is a full-suite run: ``test_no_descriptors_left_open_on_the_v4_store``
asserts zero descriptors on ``index.db*`` and passes in isolation while
failing in the full run, because by then other tests have left TierManagers
alive holding exactly those descriptors. In a long-running MCP server the
same shape leaks three descriptors per manager, permanently.
"""

from __future__ import annotations

import os

import pytest

from mind_mem.memory_tiers import TierManager


def _db_fds(db_path: str) -> list[str]:
    """Descriptors this process holds on *db_path* and its sidecars.

    Returns [] where /proc is absent, and the caller skips -- so "could not
    measure" is never mistaken for "measured zero".
    """
    out = []
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return out
    for fd in names:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(db_path):
            out.append(target)
    return out


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="/proc/self/fd is Linux-only")
def test_close_releases_every_descriptor(tmp_path) -> None:
    db = str(tmp_path / "tiers.db")
    mgr = TierManager(db)
    mgr.get_tier("D-20260101-001")  # force the read connection open
    assert _db_fds(db), "expected the manager to hold descriptors before close"

    mgr.close()
    assert _db_fds(db) == [], f"descriptors still open after close: {_db_fds(db)}"


@pytest.mark.unit
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="/proc/self/fd is Linux-only")
def test_repeated_managers_do_not_accumulate_descriptors(tmp_path) -> None:
    """The leak is per-INSTANCE, so the bound must hold across many."""
    before = len(os.listdir("/proc/self/fd"))
    for i in range(25):
        db = str(tmp_path / f"t{i}.db")
        mgr = TierManager(db)
        mgr.get_tier("D-20260101-001")
        mgr.close()
    after = len(os.listdir("/proc/self/fd"))
    assert after - before <= 5, f"descriptor count grew {before} -> {after} across 25 closed managers"


@pytest.mark.unit
def test_it_works_as_a_context_manager(tmp_path) -> None:
    db = str(tmp_path / "ctx.db")
    with TierManager(db) as mgr:
        mgr.get_tier("D-20260101-001")
    # Exiting the block must have closed it; a second close must be safe.
    mgr.close()
