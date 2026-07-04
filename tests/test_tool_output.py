"""Tests for mind_mem.tool_output — the context-offload store (§5).

Proves: (1) a 50k-line log with buried failures keeps ALL failures in the summary,
(2) the handle round-trips the FULL text, (3) the summary is byte-identical across
two runs (deterministic — no LLM/clock/RNG).
"""

from __future__ import annotations

import os
import tempfile

from mind_mem.tool_output import (
    SUMMARIZER_VERSION,
    SummarizerConfig,
    ToolOutputStore,
    make_handle,
    summarize,
)


def _big_log_with_failures() -> tuple[str, list[str]]:
    """A 50k-line synthetic test log with 3 buried, distinct failures."""
    lines = []
    buried = [
        "test tls::handshake::replay_rfc8448 ... FAILED",
        "thread 'main' panicked at 'assertion failed: `(left == right)`', src/ir/verify.rs:88:5",
        "error[E0499]: cannot borrow `*self` as mutable more than once",
    ]
    for i in range(50_000):
        if i == 1_234:
            lines.append(buried[0])
        elif i == 25_000:
            lines.append(buried[1])
        elif i == 48_765:
            lines.append(buried[2])
        else:
            lines.append(f"     Running unittests line {i} ... ok")
    lines.append("test result: FAILED. 244 passed; 3 failed; 0 ignored")
    return "\n".join(lines), buried


def test_all_buried_failures_survive_the_summary():
    text, buried = _big_log_with_failures()
    s = summarize(text, source="cargo test", exit_code=101)
    for fail in buried:
        assert fail in s.summary, f"a failure was DROPPED from the summary: {fail!r}"
    assert "3 failed" in s.summary  # the tally footer is preserved
    assert s.failure_lines >= 4     # 3 buried + the footer
    # the summary is DRAMATICALLY smaller than the input (the whole point)
    assert len(s.summary) < len(text) // 50
    assert s.line_count == 50_001


def test_summary_is_byte_identical_across_runs():
    text, _ = _big_log_with_failures()
    a = summarize(text, source="cargo test", exit_code=101).summary
    b = summarize(text, source="cargo test", exit_code=101).summary
    assert a == b  # deterministic: no clock, no RNG, no LLM


def test_handle_round_trips_full_text():
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"))
        text, _ = _big_log_with_failures()
        r = store.store_and_summarize(text, source="cargo test", exit_code=101)
        assert r.handle.startswith("to-")
        recalled = store.recall_output(r.handle)
        assert recalled == text  # FULL text round-trips byte-for-byte
        assert store.recall_output("to-nonexistent") is None


def test_handle_is_content_addressed_and_idempotent():
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"))
        text = "some output\nline two\n"
        h1 = store.store_and_summarize(text, source="pytest").handle
        h2 = store.store_and_summarize(text, source="pytest").handle
        assert h1 == h2 == make_handle(text, "pytest")  # same input → same handle
        # a different source → a different handle (namespaced)
        assert store.store_and_summarize(text, source="cargo").handle != h1


def test_head_and_tail_windows_preserved():
    text = "\n".join(f"line {i}" for i in range(500))
    s = summarize(text, head=10, tail=10)
    assert "line 0" in s.summary and "line 9" in s.summary      # head
    assert "line 499" in s.summary and "line 490" in s.summary  # tail
    assert "line 250" not in s.summary                          # middle elided
    assert s.dropped_lines > 400


def test_empty_and_tiny_inputs_are_safe():
    assert summarize("").line_count == 0
    tiny = summarize("just one line")
    assert "just one line" in tiny.summary and tiny.dropped_lines == 0


# ── bounded-summary invariant (the load-bearing architectural fix) ──────────────

def test_giant_single_line_cannot_blow_the_summary():
    # a 10 MB minified line (JSON blob / no newlines) must NOT flow whole into the
    # summary — it is per-line capped with an explicit marker.
    text = "x" * (10 * 1024 * 1024)
    s = summarize(text, config=SummarizerConfig(max_line_chars=500))
    assert len(s.summary) < 5000, "summary is unbounded — the whole point is defeated"
    assert "…[+" in s.summary and "chars]" in s.summary  # explicit truncation marker
    assert s.truncated_lines == 1


def test_all_lines_matching_failure_is_bounded_but_not_silent():
    # a log where EVERY line matches a failure pattern must not explode the summary;
    # the display is capped, but the TRUE count is reported and the cap is explicit.
    text = "\n".join(f"error: thing {i} failed" for i in range(10_000))
    s = summarize(text, config=SummarizerConfig(max_failures_shown=50))
    assert s.failure_lines == 10_000          # TRUE count — never understated
    assert s.failures_shown == 50             # display capped
    assert "more failure line(s)" in s.summary  # cap is EXPLICIT, not silent
    assert len(s.summary) < 20_000            # bounded regardless of 10k failures


def test_summary_carries_the_config_version():
    s = summarize("hello", source="x")
    assert f"v{SUMMARIZER_VERSION}/" in s.summary  # reproducibility: version stamped
    assert s.config_hash and len(s.config_hash) == 12


def test_summary_byte_identical_only_within_a_config():
    text = "\n".join(f"line {i}" for i in range(200))
    a = summarize(text, config=SummarizerConfig(head=10))
    b = summarize(text, config=SummarizerConfig(head=10))
    c = summarize(text, config=SummarizerConfig(head=20))
    assert a.summary == b.summary          # same config → byte-identical
    assert a.summary != c.summary          # different config → different (and its hash differs)
    assert a.config_hash != c.config_hash


# ── retention / storage bounds ──────────────────────────────────────────────────

def test_retention_bounds_the_table():
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"), max_rows=5)
        handles = [store.store_and_summarize(f"output number {i}\n", source="cmd").handle
                   for i in range(20)]
        # only the newest 5 survive; the oldest 15 were evicted (bounded table)
        alive = [h for h in handles if store.recall_output(h) is not None]
        assert len(alive) == 5
        assert alive == handles[-5:]  # newest kept, in insertion order


def test_store_cap_truncates_with_explicit_marker():
    with tempfile.TemporaryDirectory() as d:
        store = ToolOutputStore(sqlite_path=os.path.join(d, "t.db"),
                                max_store_bytes=1000)
        big = "A" * 5000 + "\nfinal FAILED line\n"
        r = store.store_and_summarize(big, source="cmd")
        assert r.truncated_store is True
        assert r.stored_bytes <= 1200
        recalled = store.recall_output(r.handle)
        assert "stored blob truncated" in recalled  # explicit, never silent
        # the summary was computed on the FULL text, so the failure is still surfaced
        assert "FAILED" in r.summary
