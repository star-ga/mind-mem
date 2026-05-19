"""Regression: block_parser must not silently drop corpus past a size cap.

Root cause of a catastrophic, invisible recall loss: parse_file() did
`f.read(MAX_PARSE_SIZE + 1)` with MAX_PARSE_SIZE = 100_000, so any
corpus file larger than 100 KB had every block past the cap silently
discarded. For a persistent-memory product whose files grow unbounded
this is data loss; on a 498 KB LongMemEval haystack only ~11 of 53
sessions were indexed, capping recall_any@5 at ~0.36 vs ~0.92 once
fixed.
"""

from __future__ import annotations

from mind_mem import block_parser
from mind_mem.block_parser import MAX_PARSE_SIZE, parse_file


def _make_blocks(n: int) -> str:
    # ~4 KB per block → 300 blocks ≈ 1.2 MB, far over the old 100 KB cap.
    body = "x " * 2000
    return "\n---\n\n".join(f"[SESSION-s{i}]\nStatement: block {i} {body}\nDate: 2024-01-01\nStatus: active\n" for i in range(n))


def test_large_corpus_parses_all_blocks(tmp_path):
    """A 1.2 MB file (12x the old cap) must yield ALL blocks, not ~25."""
    n = 300
    f = tmp_path / "DECISIONS.md"
    f.write_text(_make_blocks(n), encoding="utf-8")
    assert f.stat().st_size > 1_000_000  # well over the old 100 KB cap

    blocks = parse_file(str(f))
    ids = {b["_id"] for b in blocks}
    assert len(blocks) == n, f"expected {n} blocks, got {len(blocks)} (silent truncation regressed)"
    assert "SESSION-s0" in ids and f"SESSION-s{n - 1}" in ids


def test_dos_guard_is_loud_not_silent(tmp_path, monkeypatch):
    """Over the (now large) guard: truncate at a block boundary AND warn.

    mind-mem's StructuredLogger binds its stderr handler at import and
    sets propagate=False, so caplog/capsys/capfd can't see it reliably;
    assert at the logger call site instead.
    """
    monkeypatch.setattr(block_parser, "MAX_PARSE_SIZE", 50_000)
    warned: list[str] = []
    monkeypatch.setattr(
        block_parser._log,
        "warning",
        lambda event, **kw: warned.append(event),
    )
    f = tmp_path / "DECISIONS.md"
    f.write_text(_make_blocks(40), encoding="utf-8")  # ~160 KB > 50 KB guard

    blocks = parse_file(str(f))

    # Still parses the blocks that fit (not silently emptied)...
    assert len(blocks) >= 1
    # ...and the overflow is loud, not silent.
    assert "block_parser_input_over_max_parse_size" in warned, "oversize corpus truncation must emit a loud warning"


def test_default_guard_is_well_above_real_workspaces():
    """The guard must not sit at a value real memory corpora exceed."""
    assert MAX_PARSE_SIZE >= 50_000_000
