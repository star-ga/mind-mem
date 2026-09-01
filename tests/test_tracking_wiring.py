# Copyright 2026 STARGA, Inc.
"""Acceptance gate for the ``tracking`` wiring (5.0.1 restoration slice 5).

``tracking`` shipped in 2.1.0 with four independent pieces and no caller
for any of them, so 5.0.0 deleted it as unreachable. It also shipped with
no tests at all — which is the sharper half of the story, because a
module nothing calls AND nothing exercises has no evidence behind it in
either direction. This file is both: the tests it never had, and the
gate on the three connections that make it reachable.

**Context windows → the pack path.** ``pack_recall_budget(model=...)``
sizes its budget to the target model's REAL context window. The old
lookup answered every unlisted model with a silent 32 000, which is
wrong in both directions — it throws away 84% of a 200 K window, or
overflows a 16 K one — and in neither case did anyone find out. Now an
unverified id is reported as ``model_known: false`` and is NOT clamped:
the caller's number stands, and the not-knowing is visible.

**MRRTracker → measured retrieval quality.** Both halves of a real MRR
were already in the workspace and had never been joined: the signal
ledger stores the ranked block ids a recall returned plus when, and the
calibration store records which ids a caller then accepted. ``index_stats``
now scores one against the other, buckets by ISO week from each signal's
OWN timestamp, and reports the week-over-week delta — plus the
``baseline_mrr`` that ``online_trainer``'s promotion gate is supposed to
compare a candidate against instead of a hand-typed figure.

**PackingQualityMeter → the pack/feedback join.** The pack path knows
what it packed and what each block cost; only ``calibration_feedback``
knows what was referenced. A bounded in-process receipt registry lets one
tell the other, so ``packing_quality.ratio`` is measured rather than
estimated.

Every test here fails if the wiring is removed or the module body is
stubbed — not merely if an import breaks. The flag-OFF group proves the
default build is byte-identical to the one that never had any of it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from mind_mem import tracking
from mind_mem.calibration import CalibrationManager, fingerprint_of, make_query_id, query_fingerprint
from mind_mem.init_workspace import init
from mind_mem.interaction_signals import SignalStore, SignalType
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import calibration as calibration_tools
from mind_mem.mcp.tools import memory_ops as memory_ops_tools
from mind_mem.mcp.tools import recall as recall_tools

# Two instants a full week apart — 7 days always lands in a different ISO
# week, whatever weekday the first one falls on.
WEEK_ONE = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
WEEK_TWO = WEEK_ONE + timedelta(days=7)

# A model id that is deliberately NOT in the verified table.
UNVERIFIED_MODEL = "some-model-we-never-measured"


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _ws(**flags: bool) -> str:
    """An initialised workspace with the named v4 flags explicitly set."""
    workspace = tempfile.mkdtemp(prefix="mm_tracking_")
    init(workspace)
    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if flags:
        config["v4"] = {**config.get("v4", {}), **{name: {"enabled": on} for name, on in flags.items()}}
    else:
        config.pop("v4", None)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return workspace


@pytest.fixture(autouse=True)
def _clean_process_local_state() -> Any:
    """The meter and the receipt registry are process-local singletons."""
    tracking.reset_packing_state()
    yield
    tracking.reset_packing_state()


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray MIND_MEM_CONFIG on the runner must not decide these flags."""
    monkeypatch.delenv("MIND_MEM_CONFIG", raising=False)
    monkeypatch.delenv("MIND_MEM_WORKSPACE", raising=False)


# ---------------------------------------------------------------------------
# Ranked results for the pack tests
# ---------------------------------------------------------------------------


def _hits(n: int = 10, chars: int = 400) -> list[dict]:
    """*n* text results of ~``chars/4`` tokens each, ranked."""
    return [{"id": f"D-2026-{i:03d}", "excerpt": "lorem ipsum " * (chars // 12)} for i in range(n)]


def _pack(ws: str, monkeypatch: pytest.MonkeyPatch, *, results: list[dict] | None = None, **kwargs: Any) -> dict:
    monkeypatch.setattr(
        recall_tools,
        "_recall_impl",
        lambda *a, **k: json.dumps({"results": results if results is not None else _hits()}),
    )
    with use_workspace(ws):
        raw = recall_tools.pack_recall_budget.__wrapped__("anything", **kwargs)  # type: ignore[attr-defined]
    return json.loads(raw)


def _index_stats(ws: str) -> dict:
    with use_workspace(ws):
        return json.loads(memory_ops_tools.index_stats.__wrapped__())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The silent 32 000 default
# ---------------------------------------------------------------------------


class TestUnknownModelIsNotSilentlyDefaulted:
    def test_a_verified_model_reports_known(self) -> None:
        window = tracking.context_window("claude-opus-4-5")
        assert window.known is True
        assert window.tokens == 200_000
        assert window.as_dict()["context_window"] == 200_000

    def test_an_unverified_model_reports_unknown(self) -> None:
        window = tracking.context_window(UNVERIFIED_MODEL)
        assert window.known is False
        assert window.tokens == tracking.UNKNOWN_MODEL_WINDOW
        # The whole point: the number does not travel as if it were looked up.
        assert window.as_dict()["context_window"] is None
        assert window.as_dict()["model_known"] is False

    def test_model_context_window_still_returns_an_int(self) -> None:
        """Back-compat: the old contract was 'an int, always'."""
        assert tracking.model_context_window(UNVERIFIED_MODEL) == tracking.UNKNOWN_MODEL_WINDOW
        assert tracking.model_context_window("") == tracking.UNKNOWN_MODEL_WINDOW
        assert tracking.model_context_window("claude-opus-4-6") == 1_000_000

    def test_is_known_model_separates_the_two(self) -> None:
        assert tracking.is_known_model("GPT-4o") is True  # normalised
        assert tracking.is_known_model(UNVERIFIED_MODEL) is False
        assert "gpt-4o" in tracking.known_models()

    def test_every_listed_window_is_a_positive_int(self) -> None:
        for model in tracking.known_models():
            assert tracking.model_context_window(model) > 0


class TestResolvePackBudget:
    def test_clamps_a_request_over_a_known_window(self) -> None:
        out = tracking.resolve_pack_budget(1_000_000, "deepseek-v4-pro")
        assert out["effective_max_tokens"] == 64_000
        assert out["clamped"] is True
        assert out["requested_max_tokens"] == 1_000_000

    def test_honours_a_request_inside_a_known_window(self) -> None:
        out = tracking.resolve_pack_budget(5_000, "deepseek-v4-pro")
        assert out["effective_max_tokens"] == 5_000
        assert out["clamped"] is False

    def test_never_clamps_to_an_assumed_window(self) -> None:
        """The regression this whole leg exists to prevent."""
        out = tracking.resolve_pack_budget(200_000, UNVERIFIED_MODEL)
        assert out["effective_max_tokens"] == 200_000
        assert out["clamped"] is False
        assert out["model_known"] is False
        assert out["effective_max_tokens"] != tracking.UNKNOWN_MODEL_WINDOW
        assert "note" in out

    def test_no_model_is_no_clamp(self) -> None:
        out = tracking.resolve_pack_budget(999_999, "")
        assert out["effective_max_tokens"] == 999_999
        assert out["clamped"] is False


# ---------------------------------------------------------------------------
# "Working": packing respects REAL model context windows
# ---------------------------------------------------------------------------


class TestPackRespectsRealContextWindows:
    def test_budget_carries_the_real_verified_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws(context_budget=True)
        out = _pack(ws, monkeypatch, max_tokens=1_000_000, model="deepseek-v4-pro")
        assert out["context_budget"]["context_window"] == 64_000
        assert out["context_budget"]["effective_max_tokens"] == 64_000
        assert out["context_budget"]["clamped"] is True
        assert out["context_budget"]["requested_max_tokens"] == 1_000_000
        # The pack really ran under the window, not under the caller's number:
        # `budget` is the ceiling `pack_to_budget` was given.
        assert out["budget"] == 64_000

    def test_the_pre_existing_budget_integer_keeps_its_meaning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`budget` was an int before this slice and must stay one."""
        ws = _ws(context_budget=True)
        out = _pack(ws, monkeypatch, max_tokens=6_000, model="deepseek-v4-pro")
        assert out["budget"] == 6_000
        assert isinstance(out["context_budget"], dict)

    def test_the_clamp_reaches_the_packer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A clamped budget must DROP results, not merely be reported.

        Patching one tiny window into the verified table is the only way to
        make a real context window bite on a fixture-sized result list; the
        code path from tool argument to ``pack_to_budget`` is the shipped
        one.
        """
        monkeypatch.setitem(tracking._CONTEXT_WINDOWS, "tiny-window-model", 300)
        ws = _ws(context_budget=True)
        roomy = _pack(ws, monkeypatch, max_tokens=1_000_000)
        tight = _pack(ws, monkeypatch, max_tokens=1_000_000, model="tiny-window-model")

        assert roomy["included_count"] == 10, "fixture should fit entirely in a huge budget"
        assert tight["context_budget"]["effective_max_tokens"] == 300
        assert tight["context_budget"]["clamped"] is True
        assert tight["context_budget"]["context_window"] == 300
        assert tight["included_count"] < roomy["included_count"]
        assert tight["dropped_count"] > 0
        assert tight["tokens_used"] <= 300

    def test_an_unknown_model_packs_exactly_as_no_model_does(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws(context_budget=True)
        unknown = _pack(ws, monkeypatch, max_tokens=6_000, model=UNVERIFIED_MODEL)
        plain = _pack(ws, monkeypatch, max_tokens=6_000)
        assert unknown["context_budget"]["model_known"] is False
        assert unknown["context_budget"]["clamped"] is False
        for key in ("included_count", "dropped_count", "tokens_used", "included", "budget"):
            assert unknown[key] == plain[key]

    def test_a_request_under_the_window_is_not_inflated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sizing UP to the window would overfill a caller's own ceiling."""
        monkeypatch.setitem(tracking._CONTEXT_WINDOWS, "tiny-window-model", 300)
        ws = _ws(context_budget=True)
        out = _pack(ws, monkeypatch, max_tokens=200, model="tiny-window-model")
        assert out["context_budget"]["effective_max_tokens"] == 200
        assert out["budget"] == 200
        assert out["tokens_used"] <= 200


# ---------------------------------------------------------------------------
# MRR: the ledger join
# ---------------------------------------------------------------------------


def _seed_scoreable(ws: str) -> None:
    """Two weeks of retrieval outcomes, one poor and one good.

    Week 1: the accepted block came back third  → RR = 1/3.
    Week 2: the accepted block came back first  → RR = 1.
    So the delta must be ~ +0.667, and nothing about it depends on when
    the test runs.
    """
    store = SignalStore(os.path.join(ws, "memory", "interaction_signals.jsonl"))
    for moment, prev_query, ranked in (
        (WEEK_ONE, "freight ledger week one", ["D-A", "D-B", "D-GOLD"]),
        (WEEK_TWO, "freight ledger week two", ["D-GOLD", "D-A"]),
    ):
        store.observe(
            session_id="s1",
            previous_query=prev_query,
            new_query=f"{prev_query} again",
            signal_type=SignalType.RE_QUERY,
            similarity=0.9,
            previous_results=ranked,
            timestamp=_iso(moment),
        )
        CalibrationManager(ws).record_feedback(
            query_id=make_query_id(prev_query),
            block_ids_useful=["D-GOLD"],
            block_ids_not_useful=[],
            feedback_type="accepted",
        )


class TestMRRDeltaInIndexStats:
    def test_delta_is_reported_from_real_ledger_data(self) -> None:
        ws = _ws(retrieval_metrics=True)
        _seed_scoreable(ws)
        mrr = _index_stats(ws)["mrr"]
        assert mrr["queries_scored"] == 2
        assert [w["queries"] for w in mrr["weeks"]] == [1, 1]
        assert mrr["weeks"][0]["mean_mrr"] == pytest.approx(1 / 3, abs=1e-6)
        assert mrr["weeks"][1]["mean_mrr"] == pytest.approx(1.0)
        assert mrr["delta"] == pytest.approx(1.0 - 1 / 3, abs=1e-6)
        assert mrr["baseline_mrr"] == pytest.approx(1.0)

    def test_the_series_reads_no_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Determinism, proven by removing the clock rather than asserting it.

        ``_utcnow`` is the module's only clock read. If the wired path ever
        reaches it, this raises instead of quietly producing a series that
        drifts with the calendar.
        """
        ws = _ws(retrieval_metrics=True)
        _seed_scoreable(ws)

        def _forbidden() -> datetime:
            raise AssertionError("the wired MRR path read a clock")

        monkeypatch.setattr(tracking, "_utcnow", _forbidden)
        mrr = _index_stats(ws)["mrr"]
        assert mrr["delta"] == pytest.approx(1.0 - 1 / 3, abs=1e-6)

    def test_replaying_the_same_ledger_gives_the_same_answer(self) -> None:
        ws = _ws(retrieval_metrics=True)
        _seed_scoreable(ws)
        assert _index_stats(ws)["mrr"] == _index_stats(ws)["mrr"]

    def test_unlabelled_signals_are_not_scored_as_misses(self) -> None:
        """No feedback is UNMEASURED, not zero — a metric must not invent."""
        ws = _ws(retrieval_metrics=True)
        SignalStore(os.path.join(ws, "memory", "interaction_signals.jsonl")).observe(
            session_id="s1",
            previous_query="never labelled",
            new_query="never labelled again",
            signal_type=SignalType.RE_QUERY,
            similarity=0.9,
            previous_results=["D-A", "D-B"],
            timestamp=_iso(WEEK_ONE),
        )
        mrr = _index_stats(ws)["mrr"]
        assert mrr["queries_scored"] == 0
        assert mrr["delta"] is None
        assert mrr["baseline_mrr"] is None
        assert mrr["signals_scanned"] == 1
        assert mrr["signals_unlabelled"] == 1

    def test_an_empty_workspace_reports_nothing_measured(self) -> None:
        mrr = _index_stats(_ws(retrieval_metrics=True))["mrr"]
        assert mrr["queries_scored"] == 0
        assert mrr["weeks"] == []

    def test_no_block_content_or_ids_leave_the_metric(self) -> None:
        """Aggregates only — the block-reading rule has nothing to bite on."""
        ws = _ws(retrieval_metrics=True)
        _seed_scoreable(ws)
        blob = json.dumps(_index_stats(ws)["mrr"])
        for block_id in ("D-GOLD", "D-A", "D-B"):
            assert block_id not in blob
        assert "freight ledger" not in blob

    def test_baseline_feeds_the_promotion_gate(self) -> None:
        """The measured MRR ``online_trainer`` compares a candidate against.

        ``index_stats`` now reports a baseline that was *measured* from the
        ledger. Standing it up against the promotion rule is the point: a
        candidate that does not beat it by ``min_improvement`` is refused,
        one that does lands. The weight ledger remains authoritative for the
        swap itself (``ledger_baseline_mrr``); this is the number that
        should be seeding it instead of a figure typed into the call.

        ``verify_load_gate=False`` because these are fixture weight paths,
        not real checkpoints — the rule under test is the MRR comparison.
        """
        from mind_mem.online_trainer import WeightRef, WeightRegistry, promote_candidate

        ws = _ws(retrieval_metrics=True)
        _seed_scoreable(ws)
        baseline = _index_stats(ws)["mrr"]["baseline_mrr"]
        assert baseline == pytest.approx(1.0)

        registry = WeightRegistry()
        registry.set_active(WeightRef("m", "1", "/a", baseline, "t"))
        registry.set_candidate(WeightRef("m", "2", "/b", 0.0, "t"))
        refused = promote_candidate(registry, model_id="m", candidate_mrr=baseline - 0.2, baseline_mrr=baseline, verify_load_gate=False)
        assert refused["promoted"] is False
        assert refused["ledger_baseline_mrr"] == pytest.approx(baseline)

        registry.set_candidate(WeightRef("m", "3", "/c", 0.0, "t"))
        accepted = promote_candidate(registry, model_id="m", candidate_mrr=baseline + 0.5, baseline_mrr=baseline, verify_load_gate=False)
        assert accepted["promoted"] is True


# ---------------------------------------------------------------------------
# Packing quality: the pack → feedback join
# ---------------------------------------------------------------------------


class TestPackingQualityMeterWiring:
    @staticmethod
    def _feedback(ws: str, query_id: str, useful: list[str]) -> dict:
        with use_workspace(ws):
            raw = calibration_tools.calibration_feedback.__wrapped__(  # type: ignore[attr-defined]
                query_id, block_ids_useful=useful, block_ids_not_useful=[]
            )
        return json.loads(raw)

    def test_ratio_is_measured_from_the_pack_receipt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws(retrieval_metrics=True)
        results = [
            {"id": "D-1", "excerpt": "x" * 400},
            {"id": "D-2", "excerpt": "x" * 400},
            {"id": "D-3", "excerpt": "x" * 400},
        ]
        packed = _pack(ws, monkeypatch, results=results, max_tokens=4_000)
        assert packed["included_count"] == 3

        receipt = tracking.default_pack_receipts().get(query_fingerprint("anything"))
        assert receipt is not None, "the pack path left no receipt to join against"

        self._feedback(ws, make_query_id("anything"), ["D-1"])
        quality = _index_stats(ws)["packing_quality"]
        assert quality["events"] == 1
        assert quality["packed_tokens"] == packed["tokens_used"]
        assert quality["referenced_tokens"] == receipt.block_tokens["D-1"]
        assert quality["ratio"] == pytest.approx(round(receipt.block_tokens["D-1"] / packed["tokens_used"], 4))

    def test_feedback_for_a_pack_that_never_happened_records_nothing(self) -> None:
        ws = _ws(retrieval_metrics=True)
        self._feedback(ws, make_query_id("never packed"), ["D-1"])
        assert _index_stats(ws)["packing_quality"]["events"] == 0

    def test_a_block_that_was_not_packed_earns_no_credit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws(retrieval_metrics=True)
        _pack(ws, monkeypatch, results=[{"id": "D-1", "excerpt": "x" * 400}], max_tokens=4_000)
        self._feedback(ws, make_query_id("anything"), ["D-NEVER-PACKED"])
        assert _index_stats(ws)["packing_quality"]["referenced_tokens"] == 0

    def test_the_receipt_registry_is_bounded(self) -> None:
        registry = tracking.PackReceiptRegistry(capacity=3)
        for i in range(10):
            registry.record(tracking.PackReceipt(f"fp{i}", 10, {"D-1": 5}))
        assert len(registry) == 3
        assert registry.get("fp0") is None
        assert registry.get("fp9") is not None

    def test_fingerprint_round_trips_through_a_query_id(self) -> None:
        assert fingerprint_of(make_query_id("hello")) == query_fingerprint("hello")
        assert fingerprint_of("not-an-id") == ""
        assert fingerprint_of("") == ""


# ---------------------------------------------------------------------------
# Flag OFF — indistinguishable from the build that never had this
# ---------------------------------------------------------------------------


class TestFlagOff:
    def test_pack_output_is_unchanged_and_model_is_inert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws()
        plain = _pack(ws, monkeypatch, max_tokens=6_000)
        with_model = _pack(ws, monkeypatch, max_tokens=6_000, model="deepseek-v4-pro")
        assert "context_budget" not in plain
        assert "context_budget" not in with_model
        assert plain == with_model

    def test_a_clamping_model_does_not_clamp_with_the_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(tracking._CONTEXT_WINDOWS, "tiny-window-model", 300)
        ws = _ws()
        out = _pack(ws, monkeypatch, max_tokens=1_000_000, model="tiny-window-model")
        assert out["included_count"] == 10
        assert out["dropped_count"] == 0

    def test_pack_leaves_no_receipt_with_the_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws()
        _pack(ws, monkeypatch, max_tokens=6_000)
        assert len(tracking.default_pack_receipts()) == 0

    def test_index_stats_carries_no_metrics_keys_with_the_flag_off(self) -> None:
        ws = _ws()
        _seed_scoreable(ws)
        stats = _index_stats(ws)
        assert "mrr" not in stats
        assert "packing_quality" not in stats

    def test_feedback_observes_nothing_with_the_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _ws()
        _pack(ws, monkeypatch, results=[{"id": "D-1", "excerpt": "x" * 400}], max_tokens=4_000)
        with use_workspace(ws):
            calibration_tools.calibration_feedback.__wrapped__(  # type: ignore[attr-defined]
                make_query_id("anything"), block_ids_useful=["D-1"], block_ids_not_useful=[]
            )
        assert tracking.default_packing_meter().stats()["events"] == 0

    def test_the_probe_is_silent_even_on_a_broken_config(self) -> None:
        """A flag PROBE that logs makes flag-off observable. It must not."""
        from mind_mem.mcp.tools._helpers import _context_budget_enabled, _retrieval_metrics_enabled
        from mind_mem.v4 import feature_flags

        broken = tempfile.mkdtemp(prefix="mm_tracking_broken_")
        with open(os.path.join(broken, "mind-mem.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json,,,")

        with patch.object(feature_flags, "_log") as log:
            assert _context_budget_enabled(broken) is False
            assert _retrieval_metrics_enabled(broken) is False
        assert log.mock_calls == [], "the OFF probe emitted a log line"

    def test_the_probe_emits_no_log_records_at_all(self, caplog: pytest.LogCaptureFixture) -> None:
        from mind_mem.mcp.tools._helpers import _retrieval_metrics_enabled

        ws = _ws()
        with caplog.at_level(logging.DEBUG):
            assert _retrieval_metrics_enabled(ws) is False
        assert caplog.records == []


# ---------------------------------------------------------------------------
# The module body itself — it had no tests at all
# ---------------------------------------------------------------------------


class TestMRRTracker:
    def test_reciprocal_rank_comes_from_the_incumbent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One metric, one implementation.

        ``MRRTracker`` used to carry a second copy of the reciprocal-rank
        loop that ``evaluate_ab`` scores with. A tracker whose arithmetic
        drifted from the harness would report a delta against a baseline
        the harness never computed.
        """
        monkeypatch.setattr(tracking, "reciprocal_rank", lambda ranked, targets: 0.125)
        assert tracking.MRRTracker().record(["A"], ["A"], at=WEEK_ONE) == 0.125

    def test_empty_relevant_set_is_not_recorded(self) -> None:
        tracker = tracking.MRRTracker()
        assert tracker.record(["A", "B"], [], at=WEEK_ONE) == 0.0
        assert tracker.weeks() == []
        assert tracker.baseline_mrr() is None

    def test_a_miss_is_recorded_as_zero(self) -> None:
        tracker = tracking.MRRTracker()
        assert tracker.record(["A", "B"], ["Z"], at=WEEK_ONE) == 0.0
        assert tracker.scored() == 1
        assert tracker.baseline_mrr() == 0.0

    def test_weeks_are_bucketed_by_the_injected_instant(self) -> None:
        tracker = tracking.MRRTracker()
        tracker.record(["A"], ["A"], at=WEEK_ONE)
        tracker.record(["X", "A"], ["A"], at=WEEK_TWO)
        assert [w["iso_week"] for w in tracker.weeks()] == [
            tracking.iso_week_key(WEEK_ONE),
            tracking.iso_week_key(WEEK_TWO),
        ]
        assert tracker.delta() == pytest.approx(0.5 - 1.0)

    def test_the_window_evicts_oldest_weeks(self) -> None:
        tracker = tracking.MRRTracker(window_weeks=2)
        for i in range(4):
            tracker.record(["A"], ["A"], at=WEEK_ONE + timedelta(days=7 * i))
        assert len(tracker.weeks()) == 2

    def test_delta_needs_two_weeks(self) -> None:
        tracker = tracking.MRRTracker()
        tracker.record(["A"], ["A"], at=WEEK_ONE)
        assert tracker.delta() is None

    def test_events_are_folded_in_chronological_order(self) -> None:
        """Load order must not decide which week the delta compares."""
        late = tracking.MRREvent(("A",), ("A",), WEEK_TWO)
        early = tracking.MRREvent(("X", "A"), ("A",), WEEK_ONE)
        forwards = tracking.mrr_from_events([early, late]).as_dict()
        backwards = tracking.mrr_from_events([late, early]).as_dict()
        assert forwards == backwards
        assert forwards["delta"] == pytest.approx(1.0 - 0.5)


class TestSignalJoin:
    class _Sig:
        def __init__(self, prev: str, new: str, ranked: tuple[str, ...], ts: str) -> None:
            self.previous_query = prev
            self.new_query = new
            self.previous_results = ranked
            self.timestamp = ts

    def _events(self, sigs: list[Any], labels: dict[str, set[str]]) -> list[tracking.MRREvent]:
        return tracking.mrr_events_from_signals(sigs, labels, fingerprint=lambda q: q)

    def test_labels_from_either_phrasing_count(self) -> None:
        sig = self._Sig("old", "new", ("A", "B"), _iso(WEEK_ONE))
        assert self._events([sig], {"new": {"B"}})[0].relevant_ids == ("B",)
        assert self._events([sig], {"old": {"A"}})[0].relevant_ids == ("A",)
        assert self._events([sig], {"old": {"A"}, "new": {"B"}})[0].relevant_ids == ("A", "B")

    def test_a_signal_with_no_ranked_list_is_skipped(self) -> None:
        sig = self._Sig("old", "new", (), _iso(WEEK_ONE))
        assert self._events([sig], {"old": {"A"}}) == []

    def test_an_unparseable_timestamp_is_skipped(self) -> None:
        sig = self._Sig("old", "new", ("A",), "not-a-timestamp")
        assert self._events([sig], {"old": {"A"}}) == []

    def test_parse_signal_timestamp_shapes(self) -> None:
        assert tracking.parse_signal_timestamp("2026-08-10T12:00:00Z") == WEEK_ONE
        assert tracking.parse_signal_timestamp("2026-08-10T12:00:00+00:00") == WEEK_ONE
        assert tracking.parse_signal_timestamp("") is None
        assert tracking.parse_signal_timestamp("nonsense") is None


class TestPackingQualityMeterUnit:
    def test_referenced_is_clamped_to_packed(self) -> None:
        meter = tracking.PackingQualityMeter()
        meter.observe(100, 400)
        assert meter.ratio() == 1.0

    def test_reset_clears_the_counters(self) -> None:
        meter = tracking.PackingQualityMeter()
        meter.observe(100, 40)
        meter.reset()
        assert meter.stats() == {"packed_tokens": 0, "referenced_tokens": 0, "events": 0, "ratio": 0.0}

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            tracking.PackingQualityMeter().observe(1, -1)

    def test_receipt_prices_only_what_was_packed(self) -> None:
        receipt = tracking.pack_receipt_from_included(
            "fp",
            [{"id": "D-1", "_token_cost": 7}, {"block_id": "D-2", "_token_cost": 3}, {"_token_cost": 99}],
            10,
        )
        assert receipt.block_tokens == {"D-1": 7, "D-2": 3}
        assert receipt.referenced_tokens(["D-1", "D-404"]) == 7
        assert receipt.packed_tokens == 10

    def test_a_zero_capacity_registry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            tracking.PackReceiptRegistry(capacity=0)


class TestExtractConventions:
    def test_test_and_error_idioms_are_counted(self) -> None:
        out = tracking.extract_conventions(["def test_alpha(): raise ValueError('x')", "class TestBeta:\n    pass"])
        assert out["test_pattern_hits"] == 2
        assert out["error_handling_hits"] == 1
        assert out["samples_scanned"] == 2

    def test_oversized_samples_are_truncated_not_dropped(self) -> None:
        out = tracking.extract_conventions(["a_b " * 600_000])
        assert out["samples_scanned"] == 1
        assert out["samples_truncated"] == 1

    def test_non_strings_are_ignored(self) -> None:
        out = tracking.extract_conventions([None, 5, "", "snake_case_name"])  # type: ignore[list-item]
        assert out["samples_scanned"] == 1
        assert out["dominant_naming"] == "snake_case"
