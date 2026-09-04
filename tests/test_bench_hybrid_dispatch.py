# Copyright 2026 STARGA, Inc.
"""The benchmark harness can finally reach the HYBRID retrieval path.

Before this, a workspace configured ``recall.backend: "hybrid"`` measured the
Markdown BM25 scan. ``_load_backend`` knows ``scan``/``tfidf``, ``sqlite`` and
``vector``; it logged unknown config *keys* and said nothing about unknown
*values*, so it returned ``None`` for ``"hybrid"`` and recall served the scan
while the scorecard read ``hybrid`` off the config string. Every hybrid number
the harness had produced was a scan number wearing a hybrid label, and the
disclosure line above it was computed from that same string rather than from
anything the run did.

What is asserted here, and why each assertion can fail:

* the ``hybrid`` dispatch really enters :class:`HybridBackend` (spied call
  count), and the non-hybrid dispatches really do not;
* a hybrid that cannot serve its dense leg is labelled ``hybrid_bm25_only``
  and trips the declared-vs-effective tripwire, rather than being published as
  ``hybrid``;
* the legs are the **product's** per-question :func:`derive_legs` output over
  the run's own recorded state, and they reach the NDJSON row;
* the ``sqlite``/``scan`` paths serve a byte-identical ranking to plain
  ``recall()`` -- this is a bench dispatch, and it must not have moved the
  product.

Every state exercised here is one the code can actually reach. The seat that
designed this dispatch found the opposite in the previous positive control: it
fed ``effective_backend: "hybrid"`` to a probe that could never produce that
value, so the control asserted an unreachable state and proved nothing. Where
a test needs the dense leg *unavailable*, it makes ``_check_vector`` answer
False -- which is exactly what that method returns on a box without the
embedding extra -- rather than writing the label it wants to see.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.ranking_identity import (  # noqa: E402
    assert_ranking_unchanged,
    compare_rankings,
    ranking_fingerprint,
)
from mind_mem.bench.eval_adapter import SessionDoc  # noqa: E402
from mind_mem.bench.eval_adapters import MindMemAdapter  # noqa: E402
from mind_mem.hybrid_recall import HybridBackend  # noqa: E402

_HAS_EMBEDDER = importlib.util.find_spec("sentence_transformers") is not None

#: A haystack with one obvious lexical target and four distractors. Small
#: enough to embed in a test, wide enough that a fusion can reorder it.
_DOCS = [
    SessionDoc("s_gold_1", "I adopted a tabby cat named Marmalade in March and she sleeps on the radiator."),
    SessionDoc("s_1", "We discussed the quarterly budget spreadsheet and the travel expense policy."),
    SessionDoc("s_2", "The kitchen renovation needs new tiles and a dishwasher installed by June."),
    SessionDoc("s_3", "My flight to Lisbon leaves at 6am and I still need to print the boarding pass."),
    SessionDoc("s_4", "The team standup moved to 9:30 and the retro is now on Thursdays."),
]
_QUERY = "What is the name of the cat I adopted?"
_BATTERY = (_QUERY, "when does my flight leave", "kitchen tiles and dishwasher")

_SQLITE_CFG: dict[str, Any] = {"recall": {"backend": "sqlite", "knee_cutoff": False, "dedup": {"enabled": False}}}
_HYBRID_CFG: dict[str, Any] = {"recall": {"backend": "hybrid", "knee_cutoff": False, "dedup": {"enabled": False}}}
_HYBRID_VEC_CFG: dict[str, Any] = {
    "recall": {"backend": "hybrid", "vector_enabled": True, "knee_cutoff": False, "dedup": {"enabled": False}}
}


def _adapter_state(config: dict[str, Any]):
    adapter = MindMemAdapter()
    return adapter, adapter.init(list(_DOCS), config)


# ---------------------------------------------------------------------------
# 1. The dispatch itself
# ---------------------------------------------------------------------------


def test_declared_hybrid_enters_hybrid_backend_not_the_recall_facade(monkeypatch: Any) -> None:
    """The whole point: a hybrid config now runs ``HybridBackend.search``.

    Spied on the class, so the assertion is about the call that happened
    rather than about a label. The companion assertion below -- that a
    ``sqlite`` config does NOT enter it -- is what stops this passing for a
    dispatch that simply sent everything to the hybrid backend.
    """
    calls: list[tuple[str, str]] = []
    original = HybridBackend.search

    def spy(self: Any, query: str, workspace: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((query, workspace))
        return original(self, query, workspace, *args, **kwargs)

    monkeypatch.setattr(HybridBackend, "search", spy)

    adapter, state = _adapter_state(_HYBRID_CFG)
    try:
        hits = adapter.query(_QUERY, state, 5)
    finally:
        adapter.teardown(state)

    assert len(calls) == 1, f"expected exactly one HybridBackend.search, got {len(calls)}"
    query_arg, workspace_arg = calls[0]
    # Argument ORDER, not merely arity: ``HybridBackend.search`` is
    # ``(query, workspace)`` -- the mirror of ``RecallBackend.search`` -- and
    # swapping them is a bug this repository has already shipped once
    # (recorded at ``mm_cli.py:1394``). A swap would put the workspace path
    # in the query slot and still "work" by returning nothing.
    assert query_arg == _QUERY
    assert workspace_arg == state.workspace
    assert [h["doc_id"] for h in hits][:1] == ["s_gold_1"]


def test_declared_sqlite_does_not_enter_hybrid_backend(monkeypatch: Any) -> None:
    """The negative half of the dispatch claim, with a positive control.

    ``assert not calls`` on its own passes when the spy was never installed
    correctly. The gold assertion under it proves the query really ran and
    really retrieved, so the empty call list is evidence about the route
    rather than about nothing having happened.
    """
    calls: list[Any] = []
    original = HybridBackend.search

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HybridBackend, "search", spy)

    adapter, state = _adapter_state(_SQLITE_CFG)
    try:
        hits = adapter.query(_QUERY, state, 5)
    finally:
        adapter.teardown(state)

    assert calls == [], "the sqlite dispatch must stay on the recall() facade"
    assert [h["doc_id"] for h in hits][:1] == ["s_gold_1"], "the query did not actually run"


def test_the_hybrid_bm25_arm_is_the_same_fts_arm_as_sqlite() -> None:
    """The like-for-like claim, measured rather than argued.

    A hybrid run and a sqlite run are only comparable if they differ in the
    dense leg and nothing else. That holds because the harness builds the FTS
    index BEFORE constructing the backend, and ``_bm25_search_raw`` branches
    on ``recall.db`` existing: with it present the hybrid lexical arm is
    ``query_index`` -- the same FTS5 arm ``sqlite`` measures -- and without it
    the arm silently becomes the Markdown scan, which ranks differently.

    So with the dense leg off, the two configurations must serve a
    byte-identical ranking. Delete the index build in ``_probe_hybrid`` and
    this goes red, which is what makes it a gate on the build order rather
    than a restatement of it.
    """
    adapter_a, state_a = _adapter_state(_SQLITE_CFG)
    try:
        sqlite_fps = {q: ranking_fingerprint(adapter_a.query(q, state_a, 5), id_field="doc_id") for q in _BATTERY}
    finally:
        adapter_a.teardown(state_a)

    adapter_b, state_b = _adapter_state(_HYBRID_CFG)
    try:
        hybrid_fps = {q: ranking_fingerprint(adapter_b.query(q, state_b, 5), id_field="doc_id") for q in _BATTERY}
    finally:
        adapter_b.teardown(state_b)

    compared = 0
    for query in _BATTERY:
        if not sqlite_fps[query] and not hybrid_fps[query]:
            continue
        assert_ranking_unchanged(sqlite_fps[query], hybrid_fps[query], label=f"{query!r} sqlite vs hybrid-bm25", min_results=1)
        compared += 1
    assert compared, "no query retrieved anything; the comparison was vacuous"


@pytest.mark.skipif(not _HAS_EMBEDDER, reason="dense leg needs the embedding extra")
def test_a_servable_dense_leg_moves_the_ranking() -> None:
    """The dense leg has to change the answer, or measuring it is pointless.

    This is the assertion the whole exercise is for. If a vector-on hybrid
    served exactly what sqlite serves, the fusion contributed nothing and any
    "hybrid beats BM25" claim would be measuring noise. Paired with the
    identity test above -- same lexical arm, different dense leg -- a
    difference here is attributable to the dense leg and to nothing else.
    """
    adapter_a, state_a = _adapter_state(_SQLITE_CFG)
    try:
        sqlite_fp = ranking_fingerprint(adapter_a.query(_QUERY, state_a, 5), id_field="doc_id")
    finally:
        adapter_a.teardown(state_a)

    adapter_b, state_b = _adapter_state(_HYBRID_VEC_CFG)
    try:
        fused_fp = ranking_fingerprint(adapter_b.query(_QUERY, state_b, 5), id_field="doc_id")
    finally:
        adapter_b.teardown(state_b)

    assert sqlite_fp, "sqlite arm served nothing; the comparison would be vacuous"
    assert fused_fp, "fused arm served nothing; the comparison would be vacuous"
    assert compare_rankings(sqlite_fp, fused_fp).moved is True


# ---------------------------------------------------------------------------
# 2. The probe: declared vs effective, and what the dense leg can do
# ---------------------------------------------------------------------------


def test_hybrid_without_a_requested_dense_leg_is_labelled_bm25_only() -> None:
    """``hybrid`` with ``vector_enabled`` off is a BM25 run. Say so.

    ``effective_backend`` differs from the declared ``hybrid``, so
    ``PipelineProbe.mismatch`` fires and the scorer flags the row. Publishing
    this as ``hybrid`` is the claim the seat blocked every SOTA number on.
    """
    adapter, state = _adapter_state(_HYBRID_CFG)
    try:
        probe = state.probe
        assert probe.declared_backend == "hybrid"
        assert probe.effective_backend == "hybrid_bm25_only"
        assert probe.mismatch is True
        assert probe.vector_available is False
        assert probe.extra["vector_enabled"] is False
        # The lexical arm is the SAME FTS5 arm the sqlite configuration
        # measures, which is what makes a hybrid-vs-sqlite comparison
        # differ in the dense leg and nothing else.
        assert probe.extra["bm25_arm"] == "sqlite"
        # find_spec is reported under its own name, never as availability.
        assert probe.extra["deps_importable"] is _HAS_EMBEDDER
    finally:
        adapter.teardown(state)


def test_requested_but_unservable_dense_leg_is_bm25_only_and_degraded(monkeypatch: Any) -> None:
    """A hybrid that asked for the dense leg and cannot run it must not pass.

    The unavailability is produced the way the code produces it -- by
    ``HybridBackend._check_vector`` answering False, which is what it returns
    on a box without the embedding extra -- rather than by writing the label
    the assertion wants. A control that feeds a value the probe cannot emit
    proves nothing; this one exercises a state a real install reaches.
    """
    monkeypatch.setattr(HybridBackend, "_check_vector", lambda self: False)

    adapter, state = _adapter_state(_HYBRID_VEC_CFG)
    try:
        probe = state.probe
        assert probe.declared_backend == "hybrid"
        assert probe.effective_backend == "hybrid_bm25_only"
        assert probe.mismatch is True
        assert probe.vector_available is False
        assert probe.extra["vector_enabled"] is True
        assert "vector_requested_but_unavailable" in probe.notes

        hits = adapter.query(_QUERY, state, 5)
        # The measured per-question record: the dense leg was requested and
        # was not served, so it is DEGRADED, not "ran".
        assert hits.legs_ran == ("bm25",)
        assert hits.legs_degraded == ("vector",)
        assert probe.extra["legs_ran"] == ["bm25"]
        assert probe.extra["legs_degraded"] == ["vector"]
    finally:
        adapter.teardown(state)


def test_legs_are_measured_per_question_and_reach_the_ndjson_row() -> None:
    """``derive_legs`` output lands where the scorer already serialises it.

    ``write_ndjson`` writes ``probe.to_dict()`` per question and the harness
    re-inits per question, so the legs recorded during ``query`` are the legs
    that row carries. Asserted through ``to_dict`` rather than off the dict
    the code mutated, because that is the surface the artifact is built from.
    """
    adapter, state = _adapter_state(_HYBRID_CFG)
    try:
        before = state.probe.to_dict()
        assert "legs_ran" not in before["extra"], "legs must be measured by the query, not predicted by init"
        adapter.query(_QUERY, state, 5)
        after = state.probe.to_dict()
        assert after["extra"]["legs_ran"] == ["bm25"]
        assert after["extra"]["legs_degraded"] == []
    finally:
        adapter.teardown(state)


@pytest.mark.skipif(not _HAS_EMBEDDER, reason="dense leg needs the embedding extra")
def test_a_servable_dense_leg_is_labelled_hybrid_and_actually_fuses_two_arms() -> None:
    """The run the SOTA claim needs: two real arms, measured as two.

    The three assertions are deliberately independent. The label says the
    dispatch resolved to hybrid; ``vector_index_blocks`` says the store the
    dense leg reads exists AND round-trips through the reader that leg uses;
    ``legs_ran`` says the product's own deriver saw a two-leg fusion. A build
    that wrote an index the search leg cannot parse satisfies the first and
    fails the second -- which is how the shape mismatch between
    ``rebuild_index`` and ``_load_local_index`` was found.
    """
    adapter, state = _adapter_state(_HYBRID_VEC_CFG)
    try:
        probe = state.probe
        assert probe.effective_backend == "hybrid"
        assert probe.mismatch is False
        assert probe.vector_available is True
        assert probe.extra["vector_index_blocks"] == len(_DOCS)
        assert "vector_leg_inert" not in probe.extra

        hits = adapter.query(_QUERY, state, 5)
        assert hits.legs_ran == ("bm25", "hybrid", "vector")
        assert hits.legs_degraded == ()
        assert [h["doc_id"] for h in hits][:1] == ["s_gold_1"]
    finally:
        adapter.teardown(state)


# ---------------------------------------------------------------------------
# 3. The non-hybrid paths must be exactly what they were
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [_SQLITE_CFG, {"recall": {"backend": "scan"}}, None])
def test_non_hybrid_adapter_path_is_byte_identical_to_plain_recall(config: dict[str, Any] | None) -> None:
    """The sqlite / scan / default dispatches are still just ``recall()``.

    Compared at ``ranking_identity`` granularity -- every served id in rank
    order paired with the exact score that put it there -- over a battery, not
    one query. Routing any of these through the hybrid backend would move the
    scores (RRF is not BM25) and turn this red.
    """
    from mind_mem.recall import recall

    adapter, state = _adapter_state(config if config is not None else MindMemAdapter.DEFAULT_CONFIG)
    try:
        compared = 0
        for query in _BATTERY:
            through_adapter = ranking_fingerprint(adapter.query(query, state, 5), id_field="doc_id")
            direct = ranking_fingerprint(
                MindMemAdapter._rows(recall(state.workspace, query, limit=5), state, 5),
                id_field="doc_id",
            )
            if not through_adapter and not direct:
                continue
            assert_ranking_unchanged(direct, through_adapter, label=f"{query!r} via adapter", min_results=1)
            compared += 1
        assert compared, "no query in the battery retrieved anything; the comparison was vacuous"
    finally:
        adapter.teardown(state)


def test_recall_backend_instances_are_labelled_by_class(monkeypatch: Any) -> None:
    """``_load_backend`` returns an INSTANCE for the vector / Postgres routes.

    Both used to be recorded as the single opaque string ``recall_backend``,
    so an artifact could not tell a Postgres run from a vector run. The
    mapping is asserted for both known classes and for an unmapped one, which
    must still carry its own name rather than collapsing.
    """
    import mind_mem._recall_core as core

    class _Fake(core.RecallBackend):
        def search(self, workspace: str, query: str, limit: int = 10, active_only: bool = False) -> list[dict]:
            return []

        def index(self, workspace: str) -> None:
            return None

    for cls_name, expected in (
        ("VectorBackend", "vector:VectorBackend"),
        ("PostgresRecallBackend", "postgres:PostgresRecallBackend"),
        ("SomethingElse", "recall_backend:SomethingElse"),
    ):
        instance = type(cls_name, (_Fake,), {})()
        monkeypatch.setattr(core, "_load_backend", lambda _ws, _i=instance: _i)
        effective, _avail, _notes, _extra = MindMemAdapter()._probe_backend("/nonexistent", "vector")
        assert effective == expected


# ---------------------------------------------------------------------------
# 4. The one product line: an unknown backend VALUE is no longer silent
# ---------------------------------------------------------------------------


def _warnings_from_load_backend(tmp_path: Any, backend: Any) -> list[tuple[str, dict[str, Any]]]:
    import json

    import mind_mem._recall_core as core

    ws = tmp_path / f"ws_{backend}"
    ws.mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {"backend": backend}}), encoding="utf-8")

    seen: list[tuple[str, dict[str, Any]]] = []

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            def _record(event: str, **kw: Any) -> None:
                if name == "warning":
                    seen.append((event, kw))

            return _record

    original = core._log
    core._log = _Spy()  # type: ignore[assignment]
    try:
        core._load_backend(str(ws))
    finally:
        core._log = original
    return seen


@pytest.mark.parametrize("backend", ["scan", "tfidf", "sqlite"])
def test_a_known_backend_value_logs_nothing(tmp_path: Any, backend: str) -> None:
    """The OFF path stays silent. Without this the warning is not a signal."""
    events = [e for e, _ in _warnings_from_load_backend(tmp_path, backend)]
    assert "unknown_recall_backend" not in events


def test_an_unknown_backend_value_is_named(tmp_path: Any) -> None:
    """``hybrid`` reached the loader for months and got the scan in silence."""
    seen = _warnings_from_load_backend(tmp_path, "hybrid")
    matching = [kw for event, kw in seen if event == "unknown_recall_backend"]
    assert len(matching) == 1
    assert matching[0]["backend"] == "hybrid"
    assert matching[0]["fallback"] == "bm25_scan"


def test_a_non_string_backend_value_does_not_crash_the_loader(tmp_path: Any) -> None:
    """A hand-edited config can hold a list, and a list is unhashable.

    ``recall_backend not in <frozenset>`` would raise ``TypeError`` on one,
    taking down every recall in the workspace to emit a warning. The type
    guard is load-bearing, not decorative.
    """
    assert _warnings_from_load_backend(tmp_path, ["hybrid"]) is not None


@pytest.mark.parametrize("backend", ["scan", "hybrid", "sqlite"])
def test_the_warning_never_changes_what_the_loader_returns(tmp_path: Any, backend: str) -> None:
    """A log line is not a dispatch. The resolution is what it always was."""
    import json

    import mind_mem._recall_core as core

    ws = tmp_path / f"ret_{backend}"
    ws.mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"recall": {"backend": backend}}), encoding="utf-8")
    resolved = core._load_backend(str(ws))
    assert resolved == ("sqlite" if backend == "sqlite" else None)
