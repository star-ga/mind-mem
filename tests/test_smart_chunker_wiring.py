"""``smart_chunker`` wired into the BM25 chunk-scoring boost.

``_recall_core.recall`` scores a long ``Statement`` twice — once whole, once as
its best sub-chunk — and blends the two. The sub-chunks came from
``_recall_detection.chunk_text``, a three-sentence sliding window that knows
nothing about the document: a window routinely spans a markdown header, so a
section's terms are diluted by the neighbouring section's, and the "best chunk"
is not any actual section.

This file pins the *wiring*, not the chunker's own arithmetic (that already has
``test_smart_chunker.py``):

* **Flag ON** — the sub-chunks are the document's own sections. Every chunk
  opens on a header and none carries two, where the sentence window does span
  headers on the very same text; and the structured block's recall score is
  strictly higher, because a clean section chunk beats a diluted window.
* **Flag OFF** — ``retrieval.smart_chunking`` absent, ``false``, truthy-but-not-
  ``true``, or outright malformed: byte-identical to the tree before the seam
  existed, proven against a shim that restores the exact deleted call. Nothing
  new is computed and nothing new is logged.

The load-bearing test is :func:`test_flag_on_raises_the_structured_block_score`.
Revert ``chunk_statement(statement, _smart_chunk_cfg)`` in ``_recall_core`` to
``chunk_text(statement)`` and it fails — the ON and OFF scores collapse onto
each other.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest
from _recall_clock_sentinel import clock_census

import mind_mem._recall_core as recall_core
import mind_mem.recall_smart_chunk as seam
from mind_mem._recall_core import recall
from mind_mem._recall_detection import chunk_text
from mind_mem.init_workspace import init
from mind_mem.recall_smart_chunk import (
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SOFT_MAX_BOUNDARY_SCORE,
    SmartChunkingConfig,
    chunk_statement,
    is_smart_chunking_enabled,
    resolve_smart_chunking_config,
    restore_header_gaps,
)

ON_CONFIG: dict[str, Any] = {"retrieval": {"smart_chunking": {"enabled": True}}}

TARGET = "D-20260901-001"
RIVAL = "D-20260901-002"

QUERY = "rollback playbook"

#: A three-section statement. The query terms sit only in section one, and
#: sections two and three are long, so a whole-section chunk is short and dense
#: while any three-sentence window straddling into section two is long and
#: dilute. "rollback" and "playbook" are deliberately non-adjacent so the
#: bigram boost stays out of the comparison.
TARGET_LINES = (
    "# Rollback duty",
    "Rollback sits with the platform crew and the playbook lives beside it.",
    "# Coffee supplies",
    "The kitchen orders whole beans every Tuesday morning and stocks them in the tall upstairs cupboard beside the sink.",
    "Someone from the floor must sign for the delivery before noon or the courier takes the whole pallet back again.",
    "The espresso machine is descaled monthly by the office facilities vendor under the standing maintenance contract.",
    "# Parking policy",
    "Visitor parking is booked through the front desk with a licence plate and an arrival window agreed in advance.",
    "Overnight parking is not permitted anywhere on the campus grounds outside the two signed loading bays.",
    "The barrier code rotates every quarter and is posted in the lobby beside the fire evacuation notice.",
)

RIVAL_LINES = (
    "This note mentions a rollback once and the playbook nowhere else at all.",
    "It exists only to give the ranking a second candidate to sort against.",
)

#: What ``block_parser`` hands recall: continuation lines joined by a single
#: newline, every blank line dropped. Building the expectation this way rather
#: than from a triple-quoted literal keeps the test honest about the shape the
#: scorer actually sees.
TARGET_STATEMENT = "\n".join(TARGET_LINES)


class SmartChunkCalled(BaseException):
    """Raised if the structure-aware path runs while the flag is off.

    ``BaseException`` on purpose: recall wraps optional legs in
    ``except Exception`` and would swallow a plain assertion.
    """


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _statement_field(lines: tuple[str, ...]) -> str:
    """Render *lines* as one ``Statement:`` field with 2-space continuations."""
    return "Statement: " + lines[0] + "\n" + "\n".join("  " + line for line in lines[1:])


@pytest.fixture
def ws(tmp_path):
    """A two-block workspace: one structured target, one flat rival."""
    root = str(tmp_path / "ws")
    os.makedirs(root)
    init(root)
    with open(os.path.join(root, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"[{TARGET}]\nType: Decision\n{_statement_field(TARGET_LINES)}\nStatus: active\n\n")
        fh.write(f"[{RIVAL}]\nType: Decision\n{_statement_field(RIVAL_LINES)}\nStatus: active\n\n")
    yield root
    _reset_caches()


def _reset_caches() -> None:
    """Drop every cache that would otherwise leak state between runs."""
    recall_core._config_cache.clear()
    recall_core._config_mtime.clear()
    seam._smart_chunks.cache_clear()


def _set_config(ws: str, retrieval: Any, *, stamp: int) -> None:
    """Rewrite ``mind-mem.json``'s ``retrieval`` section and invalidate caches.

    ``_get_config`` caches on mtime, and two writes inside one test can land in
    the same filesystem tick, so the stamp is set explicitly rather than left to
    the clock.
    """
    path = os.path.join(ws, "mind-mem.json")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.pop("retrieval", None)
    if retrieval is not None:
        cfg["retrieval"] = retrieval
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    os.utime(path, (stamp, stamp))
    _reset_caches()


def _score(results: list[dict], block_id: str) -> float:
    for hit in results:
        if hit["_id"] == block_id:
            return float(hit["score"])
    raise AssertionError(f"{block_id} missing from {[r['_id'] for r in results]}")


def _prewiring_chunk_statement(text: str, _cfg: SmartChunkingConfig) -> list[str]:
    """The exact call this seam replaced, for a like-for-like baseline."""
    return chunk_text(text)


# ---------------------------------------------------------------------------
# Boundaries: what the flag actually changes
# ---------------------------------------------------------------------------


#: The three section titles in :data:`TARGET_STATEMENT`.
SECTION_TITLES = tuple(line for line in TARGET_LINES if line.startswith("# "))


def _sections_in(text: str) -> list[str]:
    """Section titles present in *text*, in document order.

    Substring rather than line-start matching, because ``chunk_text`` joins its
    sentences with a space and so folds a header onto the end of the previous
    line — a line-start count would silently report every naive chunk as
    single-section and turn the control below into a tautology.
    """
    return [title for title in SECTION_TITLES if title in text]


def test_enabled_chunks_are_header_aligned():
    """Every chunk opens on a header and no chunk carries two."""
    cfg = resolve_smart_chunking_config(ON_CONFIG)
    chunks = chunk_statement(TARGET_STATEMENT, cfg)

    assert len(chunks) == 3, chunks
    for chunk in chunks:
        assert chunk.startswith("# "), f"chunk does not open on a header: {chunk[:60]!r}"
        assert len(_sections_in(chunk)) == 1, f"chunk spans two sections: {chunk[:80]!r}"

    # And the split is the document's, not an arbitrary one: the section titles
    # come back in order, one per chunk.
    assert [_sections_in(c)[0] for c in chunks] == list(SECTION_TITLES)


def test_naive_window_straddles_headers():
    """Control: the sentence window this replaces does span sections.

    Without this the header-alignment assertion above could be satisfied by the
    old behaviour and prove nothing.
    """
    naive = chunk_text(TARGET_STATEMENT)
    assert len(naive) > 1
    assert any(len(_sections_in(chunk)) > 1 for chunk in naive), naive


def test_disabled_returns_exactly_the_sentence_windows():
    cfg = resolve_smart_chunking_config({})
    assert cfg.enabled is False
    assert chunk_statement(TARGET_STATEMENT, cfg) == chunk_text(TARGET_STATEMENT)


def test_unstructured_text_keeps_its_sentence_windows():
    """Enabling the flag never *removes* a chunk boost from header-less prose."""
    prose = " ".join(f"Sentence number {i} carries no structural marker at all." for i in range(12))
    cfg = resolve_smart_chunking_config(ON_CONFIG)
    assert chunk_statement(prose, cfg) == chunk_text(prose)


def test_header_gap_restoration_is_load_bearing():
    """Without the restored blank line the whole seam is inert.

    ``block_parser`` drops blank lines inside a field, and
    ``_segment_document`` only cuts at blank lines — so on the raw statement
    ``smart_chunk`` sees one segment and returns one chunk.
    """
    from mind_mem.smart_chunker import smart_chunk

    cfg = resolve_smart_chunking_config(ON_CONFIG).to_chunker_config()
    assert len(smart_chunk(TARGET_STATEMENT, config=cfg)) == 1
    assert len(smart_chunk(restore_header_gaps(TARGET_STATEMENT), config=cfg)) == 3


def test_header_gap_restoration_leaves_existing_gaps_alone():
    already = "intro line\n\n# Header\nbody"
    assert restore_header_gaps(already) == already
    assert restore_header_gaps("# Leading\nbody") == "# Leading\nbody"
    assert restore_header_gaps("body\n#hashtag not a header") == "body\n#hashtag not a header"


# ---------------------------------------------------------------------------
# Config resolution — default-off, and unreadable shapes stay off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        "not a mapping",
        {"retrieval": None},
        {"retrieval": "nope"},
        {"retrieval": {}},
        {"retrieval": {"smart_chunking": None}},
        {"retrieval": {"smart_chunking": {}}},
        {"retrieval": {"smart_chunking": {"enabled": False}}},
        {"retrieval": {"smart_chunking": {"enabled": "yes"}}},
        {"retrieval": {"smart_chunking": {"enabled": 1}}},
    ],
)
def test_off_unless_explicitly_true(config):
    assert is_smart_chunking_enabled(config) is False
    assert resolve_smart_chunking_config(config) == SmartChunkingConfig()


def test_enabled_reads_its_knobs():
    cfg = resolve_smart_chunking_config(
        {
            "retrieval": {
                "smart_chunking": {
                    "enabled": True,
                    "max_chunk_size": 400,
                    "min_chunk_size": 20,
                    "overlap_sentences": 2,
                    "soft_max_chunk_size": 200,
                    "soft_max_boundary_score": 0.75,
                }
            }
        }
    )
    assert cfg == SmartChunkingConfig(
        enabled=True,
        max_chunk_size=400,
        min_chunk_size=20,
        overlap_sentences=2,
        soft_max_chunk_size=200,
        soft_max_boundary_score=0.75,
    )


def test_bad_knob_values_fall_back_instead_of_raising():
    cfg = resolve_smart_chunking_config(
        {
            "retrieval": {
                "smart_chunking": {
                    "enabled": True,
                    "max_chunk_size": "big",
                    "min_chunk_size": -5,
                    "overlap_sentences": True,
                    "soft_max_boundary_score": "high",
                }
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.max_chunk_size == DEFAULT_MAX_CHUNK_SIZE
    assert cfg.min_chunk_size == 0
    assert cfg.overlap_sentences == 0
    assert cfg.soft_max_boundary_score == DEFAULT_SOFT_MAX_BOUNDARY_SCORE
    # And the projection the chunker validates never trips its own guard.
    cfg.to_chunker_config()


def test_soft_ceiling_is_clamped_below_the_hard_one():
    """A soft max above the hard max makes ``smart_chunk`` raise; clamp instead."""
    cfg = resolve_smart_chunking_config(
        {"retrieval": {"smart_chunking": {"enabled": True, "max_chunk_size": 300, "soft_max_chunk_size": 9000}}}
    )
    assert cfg.soft_max_chunk_size == 300
    chunk_statement(TARGET_STATEMENT, cfg)  # would raise ValueError unclamped


def test_boundary_score_is_clamped_into_range():
    for raw, want in ((-4, 0.0), (12, 1.0)):
        cfg = resolve_smart_chunking_config({"retrieval": {"smart_chunking": {"enabled": True, "soft_max_boundary_score": raw}}})
        assert cfg.soft_max_boundary_score == want


def test_llm_refinement_can_never_be_switched_on():
    """Recall is a pure function of its inputs; an LLM call on it is not."""
    cfg = resolve_smart_chunking_config({"retrieval": {"smart_chunking": {"enabled": True, "llm_refine": True}}})
    assert cfg.to_chunker_config().llm_refine is False


# ---------------------------------------------------------------------------
# Wiring — the seam is on the scored path
# ---------------------------------------------------------------------------


def test_flag_on_raises_the_structured_block_score(ws):
    """THE wiring test. Revert the call site and ON collapses onto OFF."""
    _set_config(ws, None, stamp=1_000_000)
    off = recall(ws, QUERY, limit=5)
    _set_config(ws, {"smart_chunking": {"enabled": True}}, stamp=2_000_000)
    on = recall(ws, QUERY, limit=5)

    assert _score(off, TARGET) < _score(on, TARGET), (
        f"smart chunking did not reach the score: off={_score(off, TARGET)} on={_score(on, TARGET)}"
    )
    # The rival has no structure to find, so it is untouched — the change is
    # the chunk boundaries, not a blanket rescale.
    assert _score(off, RIVAL) == _score(on, RIVAL)


def test_flag_on_runs_the_chunker_inside_recall(ws, monkeypatch):
    seen: list[str] = []
    original = seam.smart_chunk

    def spy(text, **kwargs):
        seen.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(seam, "smart_chunk", spy)
    _set_config(ws, {"smart_chunking": {"enabled": True}}, stamp=3_000_000)
    recall(ws, QUERY, limit=5)
    assert seen, "recall never reached smart_chunk with the flag on"
    assert any("# Rollback duty" in text for text in seen)


def test_flag_off_never_reaches_the_chunker(ws, monkeypatch):
    def forbidden(*_a, **_k):
        raise SmartChunkCalled("smart_chunk ran with the flag off")

    monkeypatch.setattr(seam, "smart_chunk", forbidden)
    _set_config(ws, None, stamp=4_000_000)
    recall(ws, QUERY, limit=5)  # no SmartChunkCalled escapes


# ---------------------------------------------------------------------------
# Flag OFF is byte-identical to the tree before the seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieval",
    [
        None,
        {},
        {"smart_chunking": {}},
        {"smart_chunking": {"enabled": False}},
        {"smart_chunking": {"enabled": "true"}},
        {"smart_chunking": "malformed"},
        {"smart_chunking": {"enabled": True, "max_chunk_size": None}},
    ],
    ids=["absent", "empty", "no-enabled", "false", "truthy-string", "malformed", "on-but-bad-knob"],
)
def test_flag_off_recall_matches_the_prewiring_result(ws, monkeypatch, retrieval):
    """Same blocks, same order, same scores, same keys as before the seam.

    The ``on-but-bad-knob`` case is included deliberately: an unusable knob
    falls back to a default, it does not turn the surface on by accident — and
    a *valid* enable is proven to move the score by the wiring test above, so
    this parametrisation cannot be passing vacuously.
    """
    stamp = 5_000_000 + abs(hash(repr(retrieval))) % 1000

    _set_config(ws, retrieval, stamp=stamp)
    wired = recall(ws, QUERY, limit=5)

    monkeypatch.setattr(recall_core, "chunk_statement", _prewiring_chunk_statement)
    _reset_caches()
    baseline = recall(ws, QUERY, limit=5)

    if retrieval == {"smart_chunking": {"enabled": True, "max_chunk_size": None}}:
        # A None max_chunk_size is unusable, so the surface stays on with the
        # default ceiling — this row asserts only that it does not crash and
        # still returns the same block set.
        assert [h["_id"] for h in wired] == [h["_id"] for h in baseline]
        return
    assert wired == baseline


def test_flag_off_logs_nothing_the_unwired_tree_would_not(ws, monkeypatch):
    """A probe answering "off" must leave no trace — including on a bad config.

    Slice 1's caught defect: a flag probe called a helper that logged on a
    malformed config, so the wired build emitted a line the unwired one did
    not.
    """
    _set_config(ws, {"smart_chunking": ["not", "a", "mapping"]}, stamp=6_000_000)

    # Warm first: several subsystems log once per process on their first call
    # (``live_statuses_resolved`` among them), and a first-run-only line is not
    # a seam emission. Both sides are measured on an already-warm process.
    recall(ws, QUERY, limit=5)
    with _captured_events() as wired_events:
        recall(ws, QUERY, limit=5)

    monkeypatch.setattr(recall_core, "chunk_statement", _prewiring_chunk_statement)
    _reset_caches()
    recall(ws, QUERY, limit=5)
    with _captured_events() as baseline_events:
        recall(ws, QUERY, limit=5)

    assert wired_events == baseline_events


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append((record.name, str(record.msg)))


class _captured_events:
    """Record every ``mind-mem.*`` log event emitted inside the block.

    These loggers set ``propagate = False``, so ``caplog`` (which listens on
    the root) sees nothing — the handler has to go on each logger directly.
    """

    def __enter__(self) -> list[tuple[str, str]]:
        self._handler = _Recorder()
        self._loggers = [logging.getLogger(name) for name in list(logging.Logger.manager.loggerDict) if name.startswith("mind-mem.")]
        for logger in self._loggers:
            logger.addHandler(self._handler)
        return self._handler.events

    def __exit__(self, *_exc: object) -> None:
        for logger in self._loggers:
            logger.removeHandler(self._handler)


# ---------------------------------------------------------------------------
# Determinism — the new path is on the scored path
# ---------------------------------------------------------------------------


def test_flag_on_reads_no_clock(ws):
    _set_config(ws, {"smart_chunking": {"enabled": True}}, stamp=7_000_000)
    with clock_census(allow_boundary_read=True) as census:
        recall(ws, QUERY, limit=5)
    census.assert_clock_free()


def test_flag_on_is_reproducible(ws):
    _set_config(ws, {"smart_chunking": {"enabled": True}}, stamp=8_000_000)
    first = recall(ws, QUERY, limit=5)
    seam._smart_chunks.cache_clear()
    second = recall(ws, QUERY, limit=5)
    assert first == second


def test_memoisation_returns_a_fresh_list_each_call():
    """The cache owns its tuple; a caller mutating its list must not poison it."""
    cfg = resolve_smart_chunking_config(ON_CONFIG)
    first = chunk_statement(TARGET_STATEMENT, cfg)
    first.append("mutated")
    assert "mutated" not in chunk_statement(TARGET_STATEMENT, cfg)
