# Copyright 2026 STARGA, Inc.
"""The recall index notices every edit to a block file — on the query path too.

``is_stale`` is what decides whether recall re-reads statuses live
(:func:`mind_mem.admissibility.live_statuses`). It keyed a file on
``(size, mtime_ns)`` and skipped the content hash, so two edits stayed
invisible until the next build:

* a same-size rewrite whose ``mtime`` was put back (``os.utime``, ``rsync -t``,
  ``cp -p``, an editor that preserves timestamps) — e.g. ``Status: applied``
  flipped to ``Status: pending`` in place;
* a same-size rewrite landing in the same filesystem timestamp tick as the
  index's own observation of the file, so even the metadata is identical.

A stale admission decision read from a cache is not an admission decision,
and this one goes stale in the fail-OPEN direction. The fix keys the file on
inode + ``ctime_ns`` as well (the kernel stamps ``ctime`` on every write and
``utime`` cannot set it back) and content-hashes only a file whose metadata
is *racily* close to the moment it was observed — so the steady state still
hashes nothing, which the last test measures.
"""

from __future__ import annotations

import os
import time

import pytest

import mind_mem.sqlite_index as sqlite_index
from mind_mem.sqlite_index import _connect, build_index, is_stale

BEFORE = "[D-20261008-001]\nStatement: Route admission through the shared filter\nStatus: applied\n"
AFTER = BEFORE.replace("Status: applied", "Status: pending")
assert len(AFTER) == len(BEFORE)

OLD_NS = 1_600_000_000 * 10**9  # 2020: well outside any racy window


def _workspace(tmp_path) -> tuple[str, str]:
    ws = str(tmp_path)
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(BEFORE)
    return ws, path


def _age_corpus(ws: str) -> None:
    """Push every corpus file's mtime far into the past (no racy window)."""
    for root, _dirs, files in os.walk(ws):
        if ".mind-mem-index" in root:
            continue
        for name in files:
            os.utime(os.path.join(root, name), ns=(OLD_NS, OLD_NS))


def _rewrite_same_size_same_mtime(path: str) -> None:
    before = os.stat(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(AFTER)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_positive_control_unchanged_corpus_is_fresh(tmp_path) -> None:
    ws, _path = _workspace(tmp_path)
    _age_corpus(ws)
    build_index(ws, incremental=False)
    assert is_stale(ws) is False


def test_mtime_restoring_same_size_edit_is_seen(tmp_path, monkeypatch) -> None:
    ws, path = _workspace(tmp_path)
    _age_corpus(ws)
    build_index(ws, incremental=False)
    assert is_stale(ws) is False  # positive control

    _rewrite_same_size_same_mtime(path)
    calls = []
    real = sqlite_index._file_hash
    monkeypatch.setattr(sqlite_index, "_file_hash", lambda p: calls.append(p) or real(p))
    assert is_stale(ws) is True, "a same-size, mtime-restored status flip must make the index stale"
    # The ctime coordinate answers this alone — no content hash needed. On Windows
    # st_ctime is the creation time, not the change time, so only the hash can tell.
    if os.name != "nt":
        assert calls == [], calls


def test_same_tick_edit_is_seen(tmp_path) -> None:
    """Simulate a filesystem whose clock tick swallowed the edit entirely.

    The edit lands, then the recorded metadata is made to match the file's
    current stat exactly — what a coarse-timestamp filesystem produces when
    the write falls in the same tick the index observed the file. Only the
    racy-window content check can see it.
    """
    ws, path = _workspace(tmp_path)
    build_index(ws, incremental=False)  # observed moments after the write: racy

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(AFTER)
    st = os.stat(path)
    conn = _connect(ws)
    try:
        conn.execute(
            "UPDATE file_state SET mtime = ?, mtime_ns = ?, size = ?, ctime_ns = ?, ino = ? WHERE path = 'decisions/DECISIONS.md'",
            (st.st_mtime, st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_ino),
        )
        conn.commit()
    finally:
        conn.close()

    assert is_stale(ws) is True, "a same-tick edit must still be noticed"


def test_rebuild_after_edit_clears_staleness(tmp_path) -> None:
    """Recovery is reachable: the build records the new coordinate."""
    ws, path = _workspace(tmp_path)
    _age_corpus(ws)
    build_index(ws, incremental=False)
    _rewrite_same_size_same_mtime(path)
    assert is_stale(ws) is True
    build_index(ws, incremental=True)
    assert is_stale(ws) is False


def test_steady_state_query_path_hashes_nothing(tmp_path, monkeypatch) -> None:
    """The off path stays O(files) metadata: no content hash when nothing is racy.

    ``ctime`` cannot be set back, so "a corpus last touched well before the
    build" is produced by shrinking the racy window and letting it elapse —
    the same mechanism at test speed, not a bypass of it.
    """
    monkeypatch.setattr(sqlite_index, "_RACY_WINDOW_NS", 100_000_000)  # 100 ms
    ws, _path = _workspace(tmp_path)
    _age_corpus(ws)
    time.sleep(0.25)
    build_index(ws, incremental=False)

    calls = []
    real = sqlite_index._file_hash

    def counting(path: str) -> str:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(sqlite_index, "_file_hash", counting)
    started = time.perf_counter()
    for _ in range(50):
        assert is_stale(ws) is False
    elapsed = time.perf_counter() - started
    assert calls == [], calls
    # Generous bound; the real number is reported by the lane, not asserted tightly.
    assert elapsed < 5.0


@pytest.mark.parametrize("column", ["ctime_ns", "ino", "observed_ns"])
def test_legacy_row_without_new_coordinate_is_rechecked(tmp_path, column) -> None:
    """A pre-migration row (NULL coordinate) cannot vouch for the file."""
    ws, _path = _workspace(tmp_path)
    _age_corpus(ws)
    build_index(ws, incremental=False)
    assert is_stale(ws) is False  # positive control
    conn = _connect(ws)
    try:
        conn.execute(f"UPDATE file_state SET {column} = NULL")  # noqa: S608 — fixed column names
        conn.commit()
    finally:
        conn.close()
    assert is_stale(ws) is True


def test_racy_file_is_hashed_and_only_it(tmp_path, monkeypatch) -> None:
    """Positive control for the cost test: the racy branch really hashes."""
    ws, path = _workspace(tmp_path)
    build_index(ws, incremental=False)  # DECISIONS.md written moments ago: racy

    calls = []
    real = sqlite_index._file_hash
    monkeypatch.setattr(sqlite_index, "_file_hash", lambda p: calls.append(p) or real(p))
    assert is_stale(ws) is False
    assert os.path.normpath(path) in [os.path.normpath(c) for c in calls]
