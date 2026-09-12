"""`parse_file` caches on a STAT, and a changed file is never served stale.

PROFILED 2026-09-11 (`PERF-PLAN.md` P1): over 15 recalls on the live 2,726-block corpus,
`parse_file` -> `parse_blocks` is **8.53s of 10.48s — 81% of recall time**, with 285
`parse_file` calls for 15 recalls. That is 19 corpus files re-read and re-tokenised PER
QUERY, every query, with nothing cached. 2.7 million `re.match` calls fall out of the same
loop.

So the parse is memoised. The whole risk of a parse cache is SERVING STALE CONTENT, and a
memory product that answers from a stale corpus after a governed write is worse than a slow
one — so every test here is about invalidation, not speed:

  * keyed on `(path, mtime_ns, size)` — a STAT, not a read. This codebase already learned
    that rule the expensive way: an off-path probe that re-read and re-parsed config per
    event cost 1000 reads per 1000 flag-off publishes. A cache that re-read the file to
    decide whether to re-read the file would save nothing.
  * a rewrite that changes the SIZE invalidates;
  * a rewrite that keeps the size but changes the mtime invalidates — which is the case a
    size-only key would miss, and an edit that swaps one character for another is exactly
    what a memory corpus does;
  * a returned list is never the cached object, so a caller that mutates its result cannot
    corrupt what the next caller sees. That is the failure a naive memoisation ships with
    and it is invisible until something downstream appends.
"""

from __future__ import annotations

import os
import time

from mind_mem.block_parser import parse_file

BLOCK_A = "[DEC-1]\nStatement: We ship on Friday.\nStatus: active\n\n"
BLOCK_B = "[DEC-1]\nStatement: We ship on Monday.\nStatus: active\n\n"
TWO = BLOCK_A + "[DEC-2]\nStatement: Second block.\nStatus: active\n\n"


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_parsing_the_same_unchanged_file_twice_agrees(tmp_path):
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    first = parse_file(str(f))
    second = parse_file(str(f))
    assert [b["_id"] for b in first] == [b["_id"] for b in second]
    assert first == second


def test_a_SIZE_change_is_seen(tmp_path):
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    assert len(parse_file(str(f))) == 1
    _write(f, TWO)
    assert len(parse_file(str(f))) == 2, "a grown file was served from a stale cache"


def test_a_SAME_SIZE_edit_is_seen(tmp_path):
    """The case a size-only key misses. "Friday" -> "Monday" is the same length, and a
    corpus edit that swaps one word for another of equal length is completely ordinary."""
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    assert "Friday" in parse_file(str(f))[0]["Statement"]
    # Force a distinct mtime: a same-size write inside one filesystem timestamp tick would
    # be indistinguishable, and this test is about the key, not about the clock.
    time.sleep(0.01)
    _write(f, BLOCK_B)
    os.utime(f, ns=(time.time_ns(), time.time_ns()))
    got = parse_file(str(f))[0]["Statement"]
    assert "Monday" in got, f"a same-size edit was served stale: {got!r}"


def test_a_deleted_file_is_not_served_from_cache(tmp_path):
    """A deleted file must NOT answer from the cache. Its existing behaviour is to raise
    FileNotFoundError, and the cache must preserve that rather than quietly returning the
    last good parse — a corpus file that vanished is a fact the caller needs, and serving
    its old contents would hide a deletion."""
    import pytest

    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    assert parse_file(str(f))
    f.unlink()
    with pytest.raises(FileNotFoundError):
        parse_file(str(f))


def test_the_caller_cannot_mutate_what_the_next_caller_sees(tmp_path):
    """The failure naive memoisation ships with. It stays invisible until something
    downstream appends to its result, and then the corruption is in every later read."""
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    first = parse_file(str(f))
    first.append({"_id": "INJECTED"})
    first[0]["Statement"] = "tampered"
    second = parse_file(str(f))
    assert [b["_id"] for b in second] == ["DEC-1"], second
    assert second[0]["Statement"] == "We ship on Friday.", second[0]


def test_two_different_files_do_not_share_an_entry(tmp_path):
    a, b = tmp_path / "A.md", tmp_path / "B.md"
    _write(a, BLOCK_A)
    _write(b, "[DEC-9]\nStatement: Other file.\nStatus: active\n\n")
    assert parse_file(str(a))[0]["_id"] == "DEC-1"
    assert parse_file(str(b))[0]["_id"] == "DEC-9"


def test_strict_mode_is_not_served_from_the_lenient_entry(tmp_path):
    """`strict=True` changes the CONTRACT (it raises instead of skipping), so it must not
    be answered from an entry parsed leniently — a cache that ignored the flag would make
    strict silently permissive, which is the dangerous direction."""
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    lenient = parse_file(str(f))
    strict = parse_file(str(f), strict=True)
    assert [b["_id"] for b in lenient] == [b["_id"] for b in strict]


def test_the_cache_actually_avoids_reparsing(tmp_path, monkeypatch):
    """POSITIVE CONTROL. Every test above passes just as well with no cache at all, so
    without this the file proves only that parsing is correct."""
    import mind_mem.block_parser as bp

    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    parse_file(str(f))  # prime

    calls: list[int] = []
    real = bp.parse_blocks
    monkeypatch.setattr(bp, "parse_blocks", lambda c: calls.append(1) or real(c))
    for _ in range(5):
        parse_file(str(f))
    assert calls == [], f"parse_blocks ran {len(calls)} times on an unchanged file"


def test_an_edit_that_preserves_BOTH_SIZE_AND_MTIME_is_still_seen(tmp_path):
    """THE HOLE MY FIRST VERSION HAD, and the reason the key is a content hash.

    I first keyed on `(mtime_ns, size)` reasoning "a stat, not a read". That rule does not
    transfer here: it comes from an off-path config probe where the READ was the whole
    cost, while here the read is nothing and the PARSE is 81%.

    `tests/test_recall_hot_path_5_0_2.py` already carried a fixture whose docstring names
    this exact class — "an in-place edit that keeps byte size AND st_mtime_ns identical ...
    the one class of change size+mtime cannot see" — and it caught the stat-keyed version
    serving the edit stale. Pinned here too, so the parse cache owns its own guard rather
    than relying on a test in another file to notice.
    """
    f = tmp_path / "DECISIONS.md"
    _write(f, BLOCK_A)
    assert "Friday" in parse_file(str(f))[0]["Statement"]

    before = os.stat(f)
    _write(f, BLOCK_B)                      # same length: "Friday" -> "Monday"
    os.utime(f, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(f)
    # Control on the FIXTURE itself: if these ever differ, the test below would pass for
    # the wrong reason — a cheap key would have caught the change.
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns

    got = parse_file(str(f))[0]["Statement"]
    assert "Monday" in got, f"a size- and mtime-identical edit was served stale: {got!r}"
