# Copyright 2026 STARGA, Inc.
"""The recall filters decide the candidate pool, not the leftovers of a top-k.

Before this, ``_apply_post_filters`` was handed a list that had already been
cut to ``limit``, so a filter could only ever subtract from the top-k. Measured
on a 230-block workspace whose 30 in-range matches rank below 200 stronger
out-of-range ones: ``since``/``until``/``lifecycle``/``event_id``/
``min_maturity`` each served **1** of the 30, and a query for "what did we
decide before the rewrite" answered with all-but-nothing while the answer sat
in the corpus.

Every assertion here is paired with the proof that its subject EXISTS —
``assert X not in results`` passes beautifully when the seed never wrote an X,
and the same trap catches ``len(in_range_hits) == limit`` when the fixture
quietly wrote no in-range blocks at all. So each test first proves the 30
blocks are on disk (and, for the sqlite leg, in the index), then proves they
rank below the top-k, and only then asserts they come back.

The two properties that make this shippable in a PATCH are pinned here too:

* the push-down is never ENTERED by an unfiltered query — not "is a no-op",
  *not entered* — so the corpus, the SQL, the fetch width and the served slice
  of an unfiltered query are the ones they always were; and
* the push-down accepts exactly what the post-filter it front-runs accepts, so
  moving the filter earlier can never drop a block the old code kept.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date

import pytest

from mind_mem import sqlite_index
from mind_mem._recall_core import (
    _any_filter_set,
    _apply_post_filters,
    _filter_view,
    _in_date_range,
    _prefilter_corpus,
    recall,
)
from mind_mem.block_maturity import apply_min_maturity_filter as _apply_min_maturity_filter
from mind_mem.block_parser import parse_file
from mind_mem.init_workspace import init
from mind_mem.sqlite_index import _date_predicate

_INSTANT = date(2026, 9, 3)
_QUERY = "rewrite decision ranking"
_LIMIT = 10
_IN_RANGE = 30
_DISTRACTORS = 200

#: Every filter the fixture's in-range cohort satisfies and its distractor
#: cohort fails. One dict per recall filter kwarg, so a leg that quietly
#: supports only some of them cannot hide behind the others.
_FILTER_CASES = {
    "since_until": {"since": "2024-01-01", "until": "2024-12-31"},
    "lifecycle": {"lifecycle": "durable"},
    "event_id": {"event_id": "EV-OLD"},
    "min_maturity": {"min_maturity": 0.5},
}

#: The three the sqlite leg answers today. ``lifecycle`` and ``event_id`` are
#: NOT here, and that is a finding rather than an oversight: ``query_index``
#: builds its hit from the ``blocks`` columns plus the provenance passthrough,
#: and ``Lifecycle`` / ``EventId`` are in neither — they never reach the hit,
#: so the funnel judges every sqlite hit as an undeclared "durable" with no
#: event id. Widening the pool cannot fix a field that is not on the hit. The
#: gap predates this work and is left for the passthrough change to close;
#: pinning the list here means that change will show up as this set growing.
_SQLITE_CASES = ("since_until", "min_maturity")

_IN_RANGE_IDS = frozenset(f"D-20240601-{i:06d}" for i in range(_IN_RANGE))


def _seed(ws: str, backend: str) -> None:
    """230 blocks: 200 strong matches that fail every filter, 30 weak ones that pass."""
    os.makedirs(ws, exist_ok=True)
    init(ws)
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    body = ["# DECISIONS\n\n---\n"]
    for i in range(_DISTRACTORS):
        body.append(
            f"\n[D-20260101-{i:06d}]\nDate: 2026-05-14\nStatus: draft\nScope: global\n"
            f"Statement: rewrite decision ranking rewrite decision ranking recorded {i}\n"
            f"Lifecycle: ephemeral\nTags: rewrite, ranking\n"
        )
    for i in range(_IN_RANGE):
        body.append(
            f"\n[D-20240601-{i:06d}]\nDate: 2024-06-11\nStatus: active\nScope: global\n"
            f"Statement: rewrite decision ranking note {i}\nLifecycle: durable\n"
            f"EventId: EV-OLD\nTags: rewrite\n"
        )
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(body))

    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("recall", {})
    cfg["recall"]["backend"] = backend
    # The knee is an adaptive truncation with its own opinion about how many
    # results a query deserves; leaving it on would let it, not the filter,
    # decide the counts these tests assert.
    cfg["recall"]["knee_cutoff"] = False
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    if backend == "sqlite":
        sqlite_index.build_index(ws, incremental=False)


def _assert_seed_is_real(ws: str, backend: str) -> None:
    """The positive control. Nothing below means anything without this."""
    blocks = parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))
    on_disk = {b["_id"] for b in blocks}
    missing = _IN_RANGE_IDS - on_disk
    assert not missing, f"the fixture never wrote {len(missing)} of the in-range blocks"
    assert len(on_disk) == _IN_RANGE + _DISTRACTORS

    sample = next(b for b in blocks if b["_id"] in _IN_RANGE_IDS)
    # …and that they carry the fields every filter under test reads, so a
    # later "the filter returned them" cannot be true by the filter being
    # vacuous on an absent field.
    assert sample["Date"] == "2024-06-11"
    assert sample["Lifecycle"] == "durable"
    assert sample["EventId"] == "EV-OLD"
    assert sample["Status"] == "active"

    if backend == "sqlite":
        conn = sqlite3.connect(sqlite_index._db_path(ws))
        try:
            indexed = {r[0] for r in conn.execute("SELECT id FROM blocks")}
            fts = {r[0] for r in conn.execute("SELECT block_id FROM blocks_fts")}
        finally:
            conn.close()
        assert _IN_RANGE_IDS <= indexed, "in-range blocks are not in the index"
        assert _IN_RANGE_IDS <= fts, "in-range blocks are not in the searchable FTS table"


def _ids(hits) -> list[str]:
    return [h["_id"] for h in hits]


@pytest.fixture(params=("scan", "sqlite"))
def seeded(request, tmp_path) -> tuple[str, str]:
    ws = str(tmp_path / "ws")
    _seed(ws, request.param)
    _assert_seed_is_real(ws, request.param)
    return ws, request.param


# ---------------------------------------------------------------------------
# 1. The defect, and that it is gone
# ---------------------------------------------------------------------------


def test_the_in_range_cohort_ranks_below_the_top_k(seeded) -> None:
    """The precondition the whole file rests on.

    If the weak cohort happened to rank INSIDE the top-k, a post-filter would
    have served it and there would be nothing to fix — every count below would
    pass against the broken code too. So this is asserted, not assumed.
    """
    ws, _backend = seeded
    unfiltered = _ids(recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT))
    assert len(unfiltered) == _LIMIT
    reachable = len(set(unfiltered) & _IN_RANGE_IDS)
    assert reachable < _IN_RANGE, "fixture is degenerate: the filter has nothing to reach for"
    # Measured 1 of 30 at HEAD 6cd37e5. Pinned as a bound, not the exact 1, so
    # a scoring change elsewhere retunes this file instead of breaking it.
    assert reachable <= 2


def test_every_filter_reaches_the_whole_in_range_cohort(seeded) -> None:
    """The fix. A filtered query is answered from blocks that pass the filter."""
    ws, backend = seeded
    cases = _FILTER_CASES if backend == "scan" else {k: _FILTER_CASES[k] for k in _SQLITE_CASES}
    for name, kwargs in cases.items():
        hits = recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT, **kwargs)
        got = _ids(hits)
        assert len(got) == _LIMIT, f"{backend}/{name}: served {len(got)} of a possible {_LIMIT}"
        assert set(got) <= _IN_RANGE_IDS, f"{backend}/{name}: served a block that fails the filter"


def test_a_filtered_query_serves_more_than_the_top_k_ever_held(seeded) -> None:
    """The subtraction property, stated directly.

    A post-filter over an already-cut list can only ever return a SUBSET of
    the unfiltered top-k. That is the shape of the bug, so it is what the test
    denies: the filtered answer must contain blocks the unfiltered top-k does
    not, which is impossible for any implementation that filters after slicing.
    """
    ws, backend = seeded
    unfiltered = set(_ids(recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT)))
    for name in _FILTER_CASES if backend == "scan" else _SQLITE_CASES:
        got = set(_ids(recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT, **_FILTER_CASES[name])))
        new = got - unfiltered
        assert new, f"{backend}/{name}: every hit was already in the unfiltered top-k — still post-slicing"


# ---------------------------------------------------------------------------
# 2. Patch discipline: an unfiltered query does not enter the push-down
# ---------------------------------------------------------------------------


def test_an_unfiltered_query_never_enters_the_corpus_prefilter(seeded, monkeypatch) -> None:
    """Inertness proved by making the path fatal rather than by comparing output.

    "It returns the same thing" is a weaker claim than "it does not run": the
    first survives a pre-filter that happens to keep everything today and
    silently starts costing an O(corpus) pass, or dropping blocks, tomorrow.
    """
    ws, backend = seeded

    def _boom(*_a, **_kw):
        raise AssertionError("the corpus pre-filter ran on an unfiltered query")

    monkeypatch.setattr("mind_mem._recall_core._prefilter_corpus", _boom)
    hits = recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT)
    assert len(hits) == _LIMIT, "positive control: the unfiltered query must still answer"

    # And the control that the trap is armed: with a filter set, it fires.
    if backend == "scan":
        with pytest.raises(AssertionError, match="ran on an unfiltered query"):
            recall(ws, _QUERY, limit=_LIMIT, scoring_instant=_INSTANT, lifecycle="durable")


def _traced_fts_sql(ws: str, **kwargs) -> list[str]:
    """Run ``query_index`` and return the FTS statements SQLite actually saw.

    ``sqlite3.Connection`` is an immutable type, so its ``execute`` cannot be
    patched; the trace callback is the supported seam and it is the better
    one anyway — it reports the EXPANDED statement, bound values and all, so
    these tests read the SQL the engine ran rather than the SQL the caller
    meant. The connection manager hands out one cached read connection per
    workspace, which is the connection ``query_index`` will use.
    """
    conn = sqlite_index._get_conn_manager(ws).get_read_connection()
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        sqlite_index.query_index(ws, _QUERY, scoring_instant=_INSTANT, **kwargs)
    finally:
        conn.set_trace_callback(None)
    fts = [sql for sql in seen if "blocks_fts MATCH" in sql]
    assert fts, "positive control: the FTS statement never ran, so nothing was inspected"
    return fts


def test_an_unfiltered_sqlite_query_emits_the_unwidened_sql(tmp_path) -> None:
    """No date fragment, no over-fetch — the statement and the row budget an
    unfiltered query has always used."""
    ws = str(tmp_path / "ws")
    _seed(ws, "sqlite")
    _assert_seed_is_real(ws, "sqlite")

    sql = _traced_fts_sql(ws, limit=_LIMIT, retrieve_wide_k=200)[0]
    assert "b.date" not in sql, "an unfiltered query must carry no date predicate"
    assert sql.rstrip().endswith("LIMIT 200"), f"unfiltered fetch width changed: {sql!r}"


def test_a_filtered_sqlite_query_pushes_the_date_and_widens_the_fetch(tmp_path) -> None:
    ws = str(tmp_path / "ws")
    _seed(ws, "sqlite")
    _assert_seed_is_real(ws, "sqlite")

    sql = _traced_fts_sql(ws, limit=_LIMIT, retrieve_wide_k=200, since="2024-01-01", until="2024-12-31", return_k=200)[0]
    assert "b.date <> ''" in sql, "an undated block cannot satisfy a bounded query"
    assert "b.date >= '2024-01-01'" in sql
    assert "b.date <= '2024-12-31'" in sql
    assert sql.rstrip().endswith("LIMIT 1000"), f"the fetch did not widen: {sql!r}"


def test_over_fetch_is_bounded(tmp_path) -> None:
    """The widened fetch is a heuristic, so it needs a ceiling, and the
    ceiling needs a test — an unbounded one turns a filtered query into a scan."""
    ws = str(tmp_path / "ws")
    _seed(ws, "sqlite")

    sql = _traced_fts_sql(ws, limit=10, retrieve_wide_k=100_000, return_k=100_000)[0]
    assert sql.rstrip().endswith(f"LIMIT {sqlite_index._FILTERED_FETCH_CAP}")
    # Positive control: without the cap this would have been 500,000.
    assert sqlite_index._FILTERED_FETCH_CAP < 100_000 * sqlite_index._FILTERED_OVERFETCH


# ---------------------------------------------------------------------------
# 3. The push-down decides exactly what the post-filter decides
# ---------------------------------------------------------------------------


_EQUIVALENCE_BLOCKS = [
    {"_id": "A", "Date": "2024-06-11", "Status": "active", "Lifecycle": "durable", "EventId": "EV-OLD"},
    {"_id": "B", "Date": "2026-05-14", "Status": "draft", "Lifecycle": "ephemeral"},
    {"_id": "C", "Status": "active"},  # no Date at all
    {"_id": "D", "Date": "2024-06-11", "Status": "active", "Maturity": 0},  # the falsy-override trap
    {"_id": "E", "Date": "2024-06-11", "Status": "draft", "Maturity": 1.0},
    {"_id": "F", "Date": "", "Status": "active", "Lifecycle": "generated", "EventId": ""},
    {"_id": "G", "Date": "2024-06-11", "Status": "wip", "Lifecycle": "durable", "EventId": "ev-old"},
]


def _hit_like(block: dict) -> dict:
    """The filter-relevant half of a served hit, transcribed from ``recall``.

    Deliberately NOT ``_filter_view``: comparing the push-down against a
    funnel fed by ``_filter_view`` compares ``_filter_view`` with itself, and
    a regression in it would keep both sides agreeing while both were wrong.
    (Measured — mutating ``_filter_view`` to return the raw block left that
    version of this test green.) This transcribes the production payload
    instead: ``status`` always, the four filter fields only when truthy.
    ``test_the_hit_projection_matches_a_real_served_hit`` pins the
    transcription to what ``recall`` actually serves.
    """
    hit = {"_id": block["_id"], "status": block.get("Status", "")}
    for key in ("Date", "Lifecycle", "EventId", "Maturity"):
        if block.get(key):
            hit[key] = block[key]
    return hit


def test_the_hit_projection_matches_a_real_served_hit(tmp_path) -> None:
    """The control on ``_hit_like``: prove it is what recall really serves.

    Without this the equivalence test below is checking the push-down against
    a hand-written fiction.

    It also settles WHICH ``Maturity: 0`` is the divergent one: markdown parses
    the field to the STRING ``"0"``, which is truthy and therefore travels onto
    the hit intact. The divergence lives one source over — a block whose
    ``Maturity`` is a real ``0`` / ``0.0`` (a JSON-backed store, a ``json_blob``
    row), where the truthiness passthrough drops it and ``maturity_score``
    falls back to the composite. That case is block ``D`` below.
    """
    ws = str(tmp_path / "ws")
    _seed(ws, "scan")
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(
            "\n[D-20240601-000999]\nDate: 2024-06-11\nStatus: active\nScope: global\n"
            "Statement: rewrite decision ranking note zero maturity\nLifecycle: durable\n"
            "Maturity: 0\nEventId: EV-OLD\nTags: rewrite\n"
        )
    block = next(b for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md")) if b["_id"] == "D-20240601-000999")
    assert block.get("Maturity") == "0", "positive control: the seed must carry a zero Maturity"

    hits = recall(ws, "zero maturity", limit=5, scoring_instant=_INSTANT)
    served = next((h for h in hits if h["_id"] == "D-20240601-000999"), None)
    assert served is not None, "positive control: the seeded block was never served"

    keys = ("status", "Date", "Lifecycle", "EventId", "Maturity")
    assert {k: served[k] for k in keys if k in served} == {k: v for k, v in _hit_like(block).items() if k in keys}
    # A parsed "0" is a truthy string, so it DOES travel — the transcription
    # has to agree with production there too, not only on the easy fields.
    assert served["Maturity"] == "0"
    assert _filter_view(block) == {k: v for k, v in _hit_like(block).items() if k != "_id"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": "2024-01-01", "until": "2024-12-31"},
        {"lifecycle": "durable"},
        {"min_maturity": 0.5},
    ],
)
def test_the_prefilter_uses_the_hit_projection_not_the_raw_block(kwargs) -> None:
    """Same comparison as below, but with the funnel fed by the independent
    transcription — this is the assertion that goes red if ``_filter_view``
    ever starts handing the raw block through."""
    full = dict(since=None, until=None, lifecycle=None, event_id=None, min_maturity=None)
    full.update(kwargs)
    pushed = [b["_id"] for b in _prefilter_corpus(_EQUIVALENCE_BLOCKS, **full)]  # type: ignore[arg-type]
    hits = [_hit_like(b) for b in _EQUIVALENCE_BLOCKS]
    funnelled = [h["_id"] for h in _apply_post_filters(hits, limit=len(hits), **full)]  # type: ignore[arg-type]
    assert pushed == funnelled, f"push-down and served-hit funnel disagree for {kwargs}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": "2024-01-01", "until": "2024-12-31"},
        {"since": "2024-01-01"},
        {"until": "2024-12-31"},
        {"since": ""},
        {"lifecycle": "durable"},
        {"lifecycle": "ephemeral"},
        {"event_id": "EV-OLD"},
        {"min_maturity": 0.5},
        {"min_maturity": 0.0},
        {"min_maturity": 1.0},
        {"since": "2024-01-01", "lifecycle": "durable", "min_maturity": 0.5},
    ],
)
def test_the_prefilter_keeps_exactly_what_the_funnel_would_have_kept(kwargs) -> None:
    """The anti-narrowing guard.

    Pushing a filter earlier is only safe while it is the SAME filter. The
    two run over different dicts — a corpus block vs the hit built from it —
    and the differences are load-bearing: block ``D`` carries ``Maturity: 0``,
    which reaches ``maturity_score`` as an explicit 0.0 on the block and as an
    absent field on the hit (the hit passthrough tests truthiness). Without
    ``_filter_view`` the push-down would drop ``D`` and the post-filter would
    keep it — a capability quietly lost with every test still green.
    """
    full = dict(since=None, until=None, lifecycle=None, event_id=None, min_maturity=None)
    full.update(kwargs)

    pushed = [b["_id"] for b in _prefilter_corpus(_EQUIVALENCE_BLOCKS, **full)]  # type: ignore[arg-type]
    hits = [{"_id": b["_id"], **_filter_view(b)} for b in _EQUIVALENCE_BLOCKS]
    funnelled = [h["_id"] for h in _apply_post_filters(hits, limit=len(hits), **full)]  # type: ignore[arg-type]

    assert pushed == funnelled, f"push-down and funnel disagree for {kwargs}"


def test_the_falsy_maturity_override_is_the_case_that_needs_the_view() -> None:
    """Positive control for the test above: prove the trap it guards is real.

    Filtering the RAW block rejects D (its explicit numeric ``Maturity`` 0
    wins); filtering the hit built from it keeps D (the 0 is falsy, never
    travels, and the composite scores 0.5). If this ever stops being true the
    equivalence tests above still pass but have stopped testing anything.
    """
    d = next(b for b in _EQUIVALENCE_BLOCKS if b["_id"] == "D")
    assert d["Maturity"] == 0 and not isinstance(d["Maturity"], str), "the trap needs a real numeric zero"
    args = dict(since=None, until=None, lifecycle=None, event_id=None, min_maturity=0.5)
    pushed = [b["_id"] for b in _prefilter_corpus([d], **args)]  # type: ignore[arg-type]
    hit_kept = [h["_id"] for h in _apply_post_filters([_hit_like(d)], limit=1, **args)]  # type: ignore[arg-type]
    assert pushed == ["D"] and hit_kept == ["D"], "the push-down and the served hit must both keep D"

    # …and that they would NOT agree without the projection.
    assert _apply_min_maturity_filter([d], 0.5) == [], "positive control: filtering the raw block is what drops D"

    from mind_mem.block_maturity import maturity_score

    assert maturity_score(d) == 0.0, "raw block: the explicit Maturity 0 wins"
    assert maturity_score(_hit_like(d)) >= 0.5, "hit projection: the falsy override never travels"


@pytest.mark.parametrize("since", [None, "", "2024-01-01", "2026-01-01"])
@pytest.mark.parametrize("until", [None, "", "2024-12-31", "2023-01-01"])
@pytest.mark.parametrize("value", ["", "2024-06-11", "2026-05-14"])
def test_the_sql_date_predicate_transcribes_in_date_range(tmp_path, since, until, value) -> None:
    """The other push-down, checked against the Python it replaces — in SQLite,
    not in an argument about collations."""
    db = str(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE blocks (id TEXT, date TEXT NOT NULL DEFAULT '')")
        conn.execute("INSERT INTO blocks VALUES ('X', ?)", (value,))
        fragment, params = _date_predicate(since, until)
        rows = conn.execute(f"SELECT id FROM blocks b WHERE 1=1{fragment}", params).fetchall()  # nosec B608
    finally:
        conn.close()

    expected = _in_date_range(value or None, since, until) if _any_filter_set(since, until, None, None, None) else True
    assert bool(rows) is expected, f"SQL and Python disagree for date={value!r} since={since!r} until={until!r}"


def test_no_bound_means_no_sql_at_all() -> None:
    """Byte-identity, at the level the SQL string is built."""
    assert _date_predicate(None, None) == ("", ())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"since": "2024-01-01"}, True),
        ({"until": "2024-01-01"}, True),
        ({"lifecycle": "durable"}, True),
        ({"event_id": "EV"}, True),
        ({"min_maturity": 0.0}, True),
        ({"since": ""}, True),
    ],
)
def test_any_filter_set_is_presence_not_truthiness(kwargs, expected) -> None:
    """``min_maturity=0.0`` and ``since=""`` are set filters. Reading them as
    falsy would silently skip the push-down for exactly the two callers most
    likely to be probing the boundary."""
    full = dict(since=None, until=None, lifecycle=None, event_id=None, min_maturity=None)
    full.update(kwargs)
    assert _any_filter_set(**full) is expected  # type: ignore[arg-type]
