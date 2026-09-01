# Copyright 2026 STARGA, Inc.
"""The Model Reliability Score, and the ``memory_health`` seam it hangs on.

``mrs.py`` shipped in 2.6.0 with a scoring core and no caller, and the
only coverage it ever had was five smoke assertions in
``test_v28_completion.py``. This file covers the part that makes it a
feature rather than a library: the corpus collector, the config gate, the
alert routing, and the ``mrs`` section of the ``memory_health`` tool.

Four properties are load-bearing and each has a test that fails loudly if
the wiring is unpicked:

* **The bar.** A p99 breach yields ``score < 100`` with ``p99_ms`` named
  in ``violations`` — asserted both at the pure level and end-to-end
  through ``memory_health``.
* **Flag-off byte identity.** A present-but-disabled ``mrs`` section
  produces the *same bytes* as no section at all, and the module is not
  imported at all on that path.
* **Admission.** The collector reads blocks, so it goes through
  ``admit_corpus``. A quarantined contradiction is not counted, and a
  staleness flag pointing at a withheld block is not counted either —
  each with a positive control, so the assertion cannot pass vacuously.
* **Determinism.** The module reads no clock and no randomness;
  ``computed_at`` is injected.
"""

from __future__ import annotations

import ast
import json
import pathlib
from typing import Any

import pytest

from mind_mem import mrs
from mind_mem.alerting import Alert, AlertRouter, AlertSink
from mind_mem.causal_graph import CausalGraph
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import memory_health

CORPUS = ("decisions", "tasks", "entities", "intelligence", "memory")

#: 95 fast calls and 5 very slow ones. Chosen so p50 (10 ms) and p95
#: (459.5 ms) stay under their thresholds while p99 lands on the slow
#: tail — the breach under test has to be the p99 one specifically, or
#: "p99_ms in violations" would pass for the wrong reason.
FAT_TAIL_MS = [10.0] * 95 + [9000.0] * 5


class _RecordingSink(AlertSink):
    """An alert sink that keeps what it was handed."""

    name = "recording"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


class _RecordingRouter:
    """Stands in for ``AlertRouter`` at the ``memory_health`` seam."""

    def __init__(self) -> None:
        self.fired: list[dict[str, Any]] = []

    def fire(self, *, severity: str, event: str, payload: dict) -> list[bool]:
        self.fired.append({"severity": severity, "event": event, "payload": payload})
        return [True]


class _StubMetrics:
    """The three metric methods ``memory_health`` touches, pinned.

    The live registry accumulates ``mcp_tool_duration_ms`` from every
    other test in the process, so a test asserting an exact score has to
    own its readings rather than inherit whatever ran before it.
    """

    def __init__(self, samples: dict[str, list[float]] | None = None, counters: dict[str, float] | None = None) -> None:
        self._samples = dict(samples or {})
        self._counters = dict(counters or {})

    def samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, ()))

    def get(self, name: str) -> int | float:
        return self._counters.get(name, 0)

    def inc(self, name: str, value: int | float = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value


def _block(block_id: str, statement: str, status: str = "active") -> str:
    return f"[{block_id}]\nStatement: {statement}\nStatus: {status}\nDate: 2026-09-01\n\n---\n"


def _workspace(tmp_path: pathlib.Path, *, config: dict | None = None, files: dict[str, str] | None = None) -> str:
    ws = tmp_path / "ws"
    for sub in CORPUS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps(config if config is not None else {"recall": {"backend": "scan"}}),
        encoding="utf-8",
    )
    for rel, body in (files or {}).items():
        (ws / rel).write_text(body, encoding="utf-8")
    return str(ws)


def _mrs_config(**mrs_section: Any) -> dict:
    return {"recall": {"backend": "scan"}, "mrs": mrs_section}


# ─── the bar: a p99 breach ────────────────────────────────────────────────────


@pytest.mark.unit
def test_p99_breach_scores_below_100_and_names_p99_ms() -> None:
    """The working definition, at the pure level."""
    report = mrs.compute_mrs("retrieval", mrs.latency_slis(FAT_TAIL_MS), computed_at="")

    assert report.violations == ["p99_ms"]
    assert report.score < 100.0
    # p50 and p95 are inside their thresholds, so exactly one of three
    # SLIs is zeroed: 2/3 of the weight survives.
    assert report.score == pytest.approx(100.0 * 2 / 3)


@pytest.mark.unit
def test_memory_health_surfaces_a_p99_breach(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same bar, end to end through the wired tool.

    Fails if the ``memory_health`` call site is removed (no ``mrs`` key),
    and fails if the module body is stubbed (the score and the violation
    list are the ones ``mrs``'s own arithmetic produces).
    """
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=FAT_TAIL_MS, alert=False))

    with use_workspace(ws):
        health = json.loads(memory_health())

    assert "mrs" in health, "memory_health is not wired to mrs"
    assert "p99_ms" in health["mrs"]["violations"]
    assert health["mrs"]["score"] < 100.0
    # An empty corpus scores clean on all three retrieval SLIs, so five
    # of six readings survive.
    assert health["mrs"]["violations"] == ["p99_ms"]
    assert health["mrs"]["score"] == pytest.approx(round(100.0 * 5 / 6, 2))
    assert health["mrs"]["target"] == "retrieval"


@pytest.mark.unit
def test_a_breach_reaches_the_recommendations_and_flips_the_verdict(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=FAT_TAIL_MS, alert=False))

    with use_workspace(ws):
        health = json.loads(memory_health())

    assert any("p99_ms" in rec for rec in health["recommendations"])
    assert health["score"] == "needs_attention"


@pytest.mark.unit
def test_healthy_latency_yields_no_violations(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive control for the breach tests above."""
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=[5.0] * 100, alert=False))

    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["mrs"]["violations"] == []
    assert health["mrs"]["score"] == 100.0


# ─── flag-off ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_flag_off_is_byte_identical_to_no_flag_at_all(tmp_path: pathlib.Path) -> None:
    """A present-but-disabled section changes not one byte of the output."""
    ws = _workspace(tmp_path, config={"recall": {"backend": "scan"}})
    with use_workspace(ws):
        without_section = memory_health()

    pathlib.Path(ws, "mind-mem.json").write_text(
        json.dumps(_mrs_config(enabled=False, latency_ms=FAT_TAIL_MS)),
        encoding="utf-8",
    )
    with use_workspace(ws):
        with_disabled_section = memory_health()

    assert with_disabled_section == without_section
    assert "mrs" not in json.loads(without_section)


@pytest.mark.unit
def test_flag_off_never_calls_the_module(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is a config read and nothing else.

    Every observable entry point is booby-trapped: if the flag-off path
    collected corpus counts, scored a report, or fired an alert, this
    raises instead of returning a dashboard.
    """

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("mrs ran with the flag OFF")

    monkeypatch.setattr(mrs, "workspace_mrs_report", _explode)
    monkeypatch.setattr(mrs, "collect_corpus_counts", _explode)
    monkeypatch.setattr(mrs, "route_mrs_alerts", _explode)
    monkeypatch.setattr("mind_mem.alerting.get_alert_router", _explode)

    ws = _workspace(tmp_path, config=_mrs_config(enabled=False))
    with use_workspace(ws):
        health = json.loads(memory_health())

    assert "mrs" not in health


@pytest.mark.unit
@pytest.mark.parametrize(
    "section",
    [
        {},
        {"enabled": False},
        # The one that matters: plain truthiness reads the STRING "false"
        # as True, which is the direction a switch must never fail in.
        {"enabled": "false"},
        {"enabled": "yes-please"},
        {"enabled": 1},
        "on",
        None,
        ["enabled"],
    ],
)
def test_the_gate_stays_off_for_anything_but_a_literal_true(section: Any) -> None:
    assert mrs.is_mrs_enabled({"mrs": section}) is False
    assert mrs.is_mrs_enabled(None) is False
    assert mrs.is_mrs_enabled({}) is False


@pytest.mark.unit
def test_a_literal_true_is_the_one_thing_that_turns_it_on() -> None:
    """The positive control for the fail-closed gate above."""
    assert mrs.is_mrs_enabled({"mrs": {"enabled": True}}) is True


@pytest.mark.unit
def test_a_stringy_alert_switch_falls_back_rather_than_flipping() -> None:
    assert mrs.resolve_mrs_config({"mrs": {"alert": "false"}})["alert"] is True
    assert mrs.resolve_mrs_config({"mrs": {"alert": False}})["alert"] is False


# ─── admission ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_quarantined_contradiction_is_not_counted(tmp_path: pathlib.Path) -> None:
    """Selecting a status is not filtering on it — so filter on it."""
    withheld = _workspace(
        tmp_path / "withheld",
        files={
            "decisions/DECISIONS.md": _block("D-20260901-001", "a decision"),
            "intelligence/CONTRADICTIONS.md": (
                _block("C-20260901-001", "contradiction one", status="quarantined")
                + _block("C-20260901-002", "contradiction two", status="pending")
                + _block("C-20260901-003", "contradiction three", status="banana")
            ),
        },
    )
    counts = mrs.collect_corpus_counts(withheld)
    assert counts.unresolved_contradictions == 0
    assert counts.servable_blocks == 1

    # Positive control: the identical corpus with admissible statuses is
    # counted in full, so the assertion above cannot pass because the
    # collector simply found nothing.
    admitted = _workspace(
        tmp_path / "admitted",
        files={
            "decisions/DECISIONS.md": _block("D-20260901-001", "a decision"),
            "intelligence/CONTRADICTIONS.md": (
                _block("C-20260901-001", "contradiction one")
                + _block("C-20260901-002", "contradiction two")
                + _block("C-20260901-003", "contradiction three")
            ),
        },
    )
    counts = mrs.collect_corpus_counts(admitted)
    assert counts.unresolved_contradictions == 3
    assert counts.servable_blocks == 4


@pytest.mark.unit
def test_a_staleness_flag_on_a_withheld_block_is_not_counted(tmp_path: pathlib.Path) -> None:
    def _seed(root: pathlib.Path, dependent_status: str) -> str:
        ws = _workspace(
            root,
            files={
                "decisions/DECISIONS.md": (
                    _block("D-20260901-101", "dependent", status=dependent_status) + _block("D-20260901-102", "upstream")
                )
            },
        )
        graph = CausalGraph(ws)
        graph.add_edge("D-20260901-101", "D-20260901-102", "depends_on")
        assert graph.propagate_staleness("D-20260901-102", reason="test") == ["D-20260901-101"]
        return ws

    served = _seed(tmp_path / "served", "active")
    assert mrs.collect_corpus_counts(served).stale_blocks == 1

    withheld = _seed(tmp_path / "withheld", "quarantined")
    counts = mrs.collect_corpus_counts(withheld)
    assert counts.stale_blocks == 0
    assert counts.servable_blocks == 1


@pytest.mark.unit
def test_archived_blocks_are_out_of_the_denominator(tmp_path: pathlib.Path) -> None:
    ws = _workspace(
        tmp_path,
        files={
            "decisions/DECISIONS.md": _block("D-20260901-201", "live"),
            "decisions/DECISIONS_ARCHIVE.md": _block("D-20260901-202", "archived"),
        },
    )
    assert mrs.collect_corpus_counts(ws).servable_blocks == 1


@pytest.mark.unit
def test_collector_survives_a_missing_workspace(tmp_path: pathlib.Path) -> None:
    counts = mrs.collect_corpus_counts(str(tmp_path / "nope"))
    assert counts == mrs.CorpusCounts()


# ─── the corpus SLIs ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_corpus_ratios_are_per_servable_block() -> None:
    counts = mrs.CorpusCounts(servable_blocks=200, stale_blocks=50, unresolved_contradictions=4, drift_items=10)
    by_name = {s.name: s for s in mrs.corpus_retrieval_slis(counts)}

    assert by_name["staleness_ratio"].value == pytest.approx(0.25)
    assert by_name["contradiction_density"].value == pytest.approx(2.0)  # per 100 blocks
    assert by_name["relevance_decay"].value == pytest.approx(0.05)
    # 0.25 > 0.2 and 2.0 > 0.5; relevance_decay sits exactly on its
    # threshold, and "greater than" is the violation rule.
    assert mrs.compute_mrs("t", mrs.corpus_retrieval_slis(counts)).violations == [
        "contradiction_density",
        "staleness_ratio",
    ]


@pytest.mark.unit
def test_observation_window_divides_the_decay_rate() -> None:
    counts = mrs.CorpusCounts(servable_blocks=100, drift_items=10)
    one_day = {s.name: s.value for s in mrs.corpus_retrieval_slis(counts, observation_days=1.0)}
    ten_days = {s.name: s.value for s in mrs.corpus_retrieval_slis(counts, observation_days=10.0)}

    assert one_day["relevance_decay"] == pytest.approx(0.1)
    assert ten_days["relevance_decay"] == pytest.approx(0.01)


@pytest.mark.unit
@pytest.mark.parametrize("days", [0.0, -3.0, None])
def test_a_nonsense_window_falls_back_to_one_day(days: Any) -> None:
    counts = mrs.CorpusCounts(servable_blocks=100, drift_items=10)
    values = {s.name: s.value for s in mrs.corpus_retrieval_slis(counts, observation_days=days)}
    assert values["relevance_decay"] == pytest.approx(0.1)


@pytest.mark.unit
def test_an_empty_corpus_scores_clean_rather_than_dividing_by_zero() -> None:
    slis = mrs.corpus_retrieval_slis(mrs.CorpusCounts())
    assert [s.value for s in slis] == [0.0, 0.0, 0.0]
    assert mrs.compute_mrs("t", slis).score == 100.0


@pytest.mark.unit
def test_error_rate_is_omitted_when_nothing_was_called(tmp_path: pathlib.Path) -> None:
    """No errors out of no requests is not a 0% error rate — it is no reading."""
    ws = _workspace(tmp_path)
    silent = mrs.workspace_mrs_report(ws, latency_ms=[1.0], request_count=0)
    called = mrs.workspace_mrs_report(ws, latency_ms=[1.0], error_count=1, request_count=4)

    assert "error_rate" not in {s.name for s in silent.slis}
    by_name = {s.name: s for s in called.slis}
    assert by_name["error_rate"].value == pytest.approx(0.25)
    assert "error_rate" in called.violations


# ─── SLO spec overlay ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_an_operator_threshold_overrides_the_default() -> None:
    measured = mrs.latency_slis(FAT_TAIL_MS)
    relaxed = mrs.merge_slo_spec(measured, mrs.parse_slo_spec({"slis": [{"name": "p99_ms", "threshold": 20000}]}))

    assert mrs.compute_mrs("t", measured).violations == ["p99_ms"]
    assert mrs.compute_mrs("t", relaxed).violations == []


@pytest.mark.unit
def test_a_spec_entry_without_a_threshold_does_not_disable_the_violation() -> None:
    """``parse_slo_spec`` renders an omitted threshold as None, and None
    means "no threshold" to ``compute_mrs`` — copying it over would let a
    weight-only entry silently switch the SLI off."""
    measured = mrs.latency_slis(FAT_TAIL_MS)
    weighted = mrs.merge_slo_spec(measured, mrs.parse_slo_spec({"slis": [{"name": "p99_ms", "weight": 3.0}]}))

    by_name = {s.name: s for s in weighted}
    assert by_name["p99_ms"].threshold == 1500.0
    assert by_name["p99_ms"].weight == 3.0
    assert mrs.compute_mrs("t", weighted).violations == ["p99_ms"]


@pytest.mark.unit
def test_a_spec_entry_for_an_unmeasured_sli_is_ignored() -> None:
    measured = mrs.latency_slis([1.0])
    merged = mrs.merge_slo_spec(measured, mrs.parse_slo_spec({"slis": [{"name": "cost_per_query", "threshold": 0.5}]}))
    assert {s.name for s in merged} == {s.name for s in measured}


@pytest.mark.unit
def test_the_slo_spec_travels_through_the_config(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    ws = _workspace(
        tmp_path,
        config=_mrs_config(
            enabled=True,
            latency_ms=FAT_TAIL_MS,
            alert=False,
            slo={"slis": [{"name": "p99_ms", "threshold": 20000}]},
        ),
    )
    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["mrs"]["violations"] == []
    assert health["mrs"]["score"] == 100.0


# ─── config resolution ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_malformed_knobs_fall_back_to_defaults() -> None:
    resolved = mrs.resolve_mrs_config(
        {
            "mrs": {
                "enabled": True,
                "observation_days": "seven",
                "alert_below": 500.0,
                "alert_severity": "apocalyptic",
                "latency_ms": "not-a-series",
                "slo": "not-a-mapping",
            }
        }
    )
    assert resolved["observation_days"] == 1.0
    assert resolved["alert_below"] == 100.0
    assert resolved["alert_severity"] == "warning"
    assert resolved["latency_ms"] is None
    assert resolved["slo"] == {}
    assert resolved["latency_metric"] == "mcp_tool_duration_ms"


@pytest.mark.unit
def test_a_literal_latency_series_is_coerced_and_filtered() -> None:
    resolved = mrs.resolve_mrs_config({"mrs": {"latency_ms": [1, 2.5, "three", None, True]}})
    assert resolved["latency_ms"] == [1.0, 2.5]


@pytest.mark.unit
def test_the_latency_metric_name_is_read_from_the_registry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no literal series, the readings come from the metric registry."""
    monkeypatch.setattr(
        "mind_mem.mcp.tools.memory_ops.metrics",
        _StubMetrics(samples={"my_own_latency_ms": FAT_TAIL_MS}),
    )
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_metric="my_own_latency_ms", alert=False))
    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["mrs"]["violations"] == ["p99_ms"]


@pytest.mark.unit
def test_the_metric_registry_hands_back_raw_readings() -> None:
    """``summary()`` cannot answer a percentile question; ``samples()`` can."""
    from mind_mem.observability import Metrics

    registry = Metrics()
    for value in (3.0, 1.0, 2.0):
        registry.observe("probe_ms", value)

    assert registry.samples("probe_ms") == [3.0, 1.0, 2.0]
    assert registry.samples("never_observed") == []
    registry.samples("probe_ms").append(999.0)
    assert registry.samples("probe_ms") == [3.0, 1.0, 2.0]


@pytest.mark.unit
def test_error_counters_reach_the_report(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mind_mem.mcp.tools.memory_ops.metrics",
        _StubMetrics(counters={"mcp_tool_failure": 3, "mcp_tool_success": 1}),
    )
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=[1.0], alert=False))
    with use_workspace(ws):
        health = json.loads(memory_health())

    readings = {s["name"]: s["value"] for s in health["mrs"]["slis"]}
    assert readings["error_rate"] == pytest.approx(0.75)
    assert "error_rate" in health["mrs"]["violations"]


# ─── alert routing ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_breach_is_routed_through_the_alert_router() -> None:
    sink = _RecordingSink()
    router = AlertRouter(sinks=[sink], min_severity="info", workspace="/tmp/ws")
    report = mrs.compute_mrs("retrieval", mrs.latency_slis(FAT_TAIL_MS))

    assert mrs.route_mrs_alerts(report, router=router) == [True]
    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.event == "mrs_degraded"
    assert alert.payload["violations"] == ["p99_ms"]
    assert alert.payload["readings"]["p99_ms"] == 9000.0


@pytest.mark.unit
def test_a_clean_report_fires_nothing() -> None:
    sink = _RecordingSink()
    router = AlertRouter(sinks=[sink], min_severity="info", workspace="/tmp/ws")
    report = mrs.compute_mrs("retrieval", mrs.latency_slis([1.0] * 10))

    assert mrs.route_mrs_alerts(report, router=router) == []
    assert sink.alerts == []


@pytest.mark.unit
def test_the_alert_payload_carries_no_block_text(tmp_path: pathlib.Path) -> None:
    """Sinks write to logs and third-party webhooks. Aggregates only."""
    secret = "canary-fA9-classified-statement"
    ws = _workspace(
        tmp_path,
        files={"intelligence/CONTRADICTIONS.md": _block("C-20260901-901", secret)},
    )
    report = mrs.workspace_mrs_report(ws, latency_ms=FAT_TAIL_MS)
    sink = _RecordingSink()
    router = AlertRouter(sinks=[sink], min_severity="info", workspace=ws)

    mrs.route_mrs_alerts(report, router=router)
    assert secret not in json.dumps(sink.alerts[0].as_dict())


@pytest.mark.unit
def test_memory_health_routes_its_breach(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    router = _RecordingRouter()
    monkeypatch.setattr("mind_mem.alerting.get_alert_router", lambda ws: router)
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=FAT_TAIL_MS, alert_severity="critical"))

    with use_workspace(ws):
        json.loads(memory_health())

    assert len(router.fired) == 1
    assert router.fired[0]["severity"] == "critical"
    assert router.fired[0]["payload"]["violations"] == ["p99_ms"]


@pytest.mark.unit
def test_alert_false_keeps_the_section_and_skips_the_router(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())
    router = _RecordingRouter()
    monkeypatch.setattr("mind_mem.alerting.get_alert_router", lambda ws: router)
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=FAT_TAIL_MS, alert=False))

    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["mrs"]["violations"] == ["p99_ms"]
    assert router.fired == []


@pytest.mark.unit
def test_a_failing_router_does_not_take_the_dashboard_with_it(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mind_mem.mcp.tools.memory_ops.metrics", _StubMetrics())

    def _broken(_ws: str) -> Any:
        raise OSError("no route to alert host")

    monkeypatch.setattr("mind_mem.alerting.get_alert_router", _broken)
    ws = _workspace(tmp_path, config=_mrs_config(enabled=True, latency_ms=FAT_TAIL_MS))

    with use_workspace(ws):
        health = json.loads(memory_health())

    assert health["mrs"]["violations"] == ["p99_ms"]


# ─── determinism ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mrs_reads_no_clock_and_no_randomness() -> None:
    """``computed_at`` is injected, never read.

    A score that quietly stamped itself with ``time.time()`` would not be
    comparable across runs, and the same source of impurity is what puts
    a module on the scored path's forbidden list.
    """
    banned = {"time", "datetime", "random", "secrets", "uuid"}
    tree = ast.parse(pathlib.Path(mrs.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])

    assert not (imported & banned), f"mrs must stay clock-free, found {sorted(imported & banned)}"


@pytest.mark.unit
def test_two_reports_over_one_corpus_are_identical(tmp_path: pathlib.Path) -> None:
    ws = _workspace(tmp_path, files={"decisions/DECISIONS.md": _block("D-20260901-301", "stable")})
    first = mrs.workspace_mrs_report(ws, latency_ms=FAT_TAIL_MS)
    second = mrs.workspace_mrs_report(ws, latency_ms=FAT_TAIL_MS)

    assert first.as_dict() == second.as_dict()
    assert first.computed_at == ""


@pytest.mark.unit
def test_computed_at_is_whatever_the_caller_injected() -> None:
    report = mrs.compute_mrs("t", mrs.latency_slis([1.0]), computed_at="2026-09-01T00:00:00Z")
    assert report.as_dict()["computed_at"] == "2026-09-01T00:00:00Z"


@pytest.mark.unit
def test_injected_counts_skip_the_corpus_read(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon that already collected counts must not pay for them twice."""

    def _explode(_ws: str) -> Any:
        raise AssertionError("collect_corpus_counts ran despite injected counts")

    monkeypatch.setattr(mrs, "collect_corpus_counts", _explode)
    report = mrs.workspace_mrs_report(
        str(tmp_path),
        counts=mrs.CorpusCounts(servable_blocks=10, stale_blocks=9),
    )
    assert "staleness_ratio" in report.violations
