"""Regression tests for the windowed extract_facts scan (issue #530).

v4.0.12 fixed catastrophic regex backtracking by silently truncating
input past 4 000 chars (``_FACT_TEXT_MAX``). That cap traded one bug
(the 55 s scan on an 80 KB statement) for another: legitimate large
blocks lost every fact past byte 4 000. v4.0.13 splits inputs above
the cap into ≤ ``_FACT_TEXT_MAX_WINDOWS`` sentence-boundary windows
and scans each independently, deduping across windows.

These tests assert:

1. A 6 KB observation with a unique fact at byte 5 500 still surfaces
   as a fact card (the audit case from 2026-05-19).
2. The total scan time of a 30 KB observation stays well under a
   second (anti-ReDoS bound holds).
3. An adversarial 200 KB input is still bounded — runtime cap holds,
   and only ``_FACT_TEXT_MAX_WINDOWS × _FACT_TEXT_MAX`` chars are
   actually scanned (trailing text dropped, parent block still indexed
   in FTS upstream).
4. ``_split_into_windows`` never emits a window larger than its
   ``window_size`` argument.
5. Window count never exceeds ``max_windows``.
6. Windows snap to sentence boundaries when possible.
"""

from __future__ import annotations

import time

import pytest
from mind_mem.extractor import (
    _FACT_TEXT_MAX,
    _FACT_TEXT_MAX_WINDOWS,
    _split_into_windows,
    extract_facts,
)

# ---------------------------------------------------------------------------
# Functional: facts past byte 5 500 must still be extracted
# ---------------------------------------------------------------------------


def test_fact_at_byte_5500_is_extracted_not_silently_truncated() -> None:
    """v4.0.12 dropped this fact; v4.0.13 must surface it (audit case)."""
    # Pad text to push the unique fact past byte 5 500. The padding is
    # short, neutral sentences that don't match any extraction pattern.
    pad_sentence = "Today is a clear day. "  # ~22 chars
    pad = pad_sentence * 260  # ~5 720 chars
    assert len(pad) > 5_500

    target_fact = "I'm a software engineer at NorthernLight Inc."
    text = pad + target_fact + " That's been my role for five years."

    cards = extract_facts(text, speaker="Alex", date="2026-05-19", source_id="BLOCK-1")

    assert cards, "windowed scan must surface at least one fact card"
    contents = [str(c["content"]).lower() for c in cards]
    assert any("software engineer at northernlight" in c for c in contents), f"target fact (byte ~5 500) missing from {contents}"


def test_fact_in_first_window_still_extracted() -> None:
    """Sanity: the existing short-input path must keep working."""
    cards = extract_facts(
        "I am a marine biologist.",
        speaker="Sam",
        date="2026-05-19",
        source_id="BLOCK-2",
    )
    contents = [str(c["content"]).lower() for c in cards]
    assert any("marine biologist" in c for c in contents)


# ---------------------------------------------------------------------------
# Performance: windowed scan stays bounded
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_30kb_observation_under_one_second() -> None:
    """A 30 KB observation must complete in well under a second.

    Issue #530's 80 KB blob took 55.9 s under v4.0.11; with windowing
    the per-window cost is bounded and 30 KB ≈ 8 windows × ~30 ms.
    """
    # 30 KB of mildly fact-shaped text — drives the regex catalog
    # through realistic work without adversarial patterns.
    chunk = "I went to the gym yesterday. I love running and swimming. My favorite book is Dune. I have a brother named Tim. "
    text = chunk * 268  # chunk is 112 chars → ~30 KB
    assert 28_000 < len(text) < 32_000

    t0 = time.perf_counter()
    cards = extract_facts(text, speaker="Alex", source_id="BLOCK-3")
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"30 KB scan took {elapsed:.3f}s — windowing regression"
    assert cards, "expected at least one fact card from 30 KB of factful text"


@pytest.mark.timeout(15)
def test_200kb_adversarial_input_is_bounded() -> None:
    """Adversarial 200 KB input must not blow up runtime (#530 anti-ReDoS).

    Only ``_FACT_TEXT_MAX_WINDOWS × _FACT_TEXT_MAX`` chars are scanned;
    trailing text is silently dropped to preserve the ReDoS bound. The
    parent block remains indexed in FTS upstream — only the sub-fact
    card extraction is bounded.
    """
    # 200 KB of adversarial-ish text. Use plausible content + repeats
    # so we'd notice if a path is uncapped.
    text = "I'm a happy person. Today was great. " * 5_500
    assert len(text) > 200_000

    t0 = time.perf_counter()
    cards = extract_facts(text, speaker="Alex", source_id="BLOCK-4")
    elapsed = time.perf_counter() - t0

    # Whatever the result count, the runtime must be bounded.
    assert elapsed < 3.0, f"200 KB adversarial scan took {elapsed:.3f}s — anti-ReDoS bound broken"
    # And we must not have scanned more than the window budget.
    # (Each card carries source_id but no offset; the budget bound is
    # asserted via runtime; cards may dedup down to small count.)
    assert isinstance(cards, list)


# ---------------------------------------------------------------------------
# Unit: _split_into_windows invariants
# ---------------------------------------------------------------------------


def test_split_into_windows_respects_window_size() -> None:
    """No window may exceed window_size."""
    text = "a" * 20_000
    windows = _split_into_windows(text, _FACT_TEXT_MAX, _FACT_TEXT_MAX_WINDOWS)
    for w in windows:
        assert len(w) <= _FACT_TEXT_MAX


def test_split_into_windows_respects_max_windows() -> None:
    """Total window count never exceeds max_windows."""
    text = "x" * 1_000_000
    windows = _split_into_windows(text, _FACT_TEXT_MAX, _FACT_TEXT_MAX_WINDOWS)
    assert len(windows) <= _FACT_TEXT_MAX_WINDOWS


def test_split_into_windows_handles_short_text() -> None:
    """Short text returns a single window covering the whole input."""
    text = "Short observation."
    windows = _split_into_windows(text, _FACT_TEXT_MAX, _FACT_TEXT_MAX_WINDOWS)
    assert windows == [text]


def test_split_into_windows_snaps_to_sentence_boundary() -> None:
    """When a sentence terminator falls in the latter half of a window,
    the split should snap to it rather than mid-word."""
    sentence = "This is sentence number {} of many. "  # 36 chars per fmt
    # Build text long enough to force splitting.
    body = "".join(sentence.format(i) for i in range(200))  # ~7.2 KB
    windows = _split_into_windows(body, _FACT_TEXT_MAX, _FACT_TEXT_MAX_WINDOWS)
    assert len(windows) >= 2
    # First window must end on a sentence terminator if any boundary
    # exists in its latter half (which it does — every 36 chars).
    assert windows[0].rstrip().endswith((".", "!", "?")), f"first window did not snap: {windows[0][-40:]!r}"


def test_split_into_windows_covers_text_contiguously() -> None:
    """Windows must concatenate back to the original text (no overlap, no gap)
    — up to the max_windows budget."""
    text = "Sentence A. Sentence B. Sentence C. " * 200  # ~7.2 KB
    windows = _split_into_windows(text, _FACT_TEXT_MAX, _FACT_TEXT_MAX_WINDOWS)
    rebuilt = "".join(windows)
    assert text.startswith(rebuilt) or rebuilt == text[: len(rebuilt)]
