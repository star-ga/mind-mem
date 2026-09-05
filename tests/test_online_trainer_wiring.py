# Copyright 2026 STARGA, Inc.
"""``online_trainer`` is connected — a drain at one end, a governed gate at the other.

``mind_mem.online_trainer`` shipped a signal-to-training-tuple converter nothing
called and a weight registry nothing constructed: a learning loop with neither a
producer nor a consumer. 5.0.0 deleted it as unreachable. It had NO tests at all,
which is why "restore the file" is not the fix — a restored file with no test is
the same unreachable surface with a longer changelog.

Wiring it means two different claims, and this file proves them separately.

**The harvest half is a drain.** The ``dream_cycle`` maintenance pass — which the
daemon already schedules — drains the append-only interaction-signal ledger into
training tuples, advancing a persisted cursor so each signal is consumed once.
The property that matters is the mapping the module was written for: a
``correction`` signal is the user saying the previous results were WRONG, so its
prior results become NEGATIVES. If that inverted, the loop would train the
retriever toward exactly the blocks a user rejected.

**And it is admission-filtered.** A signal's ``previous_results`` are block ids,
and a harvested tuple is a durable artifact naming blocks. So the same question
the slice-2 quarantine bypass asked applies here: does this leg call
``admit_corpus``, or does it merely look at a status? A quarantined block that
becomes a training target is content escaping through the training set while
recall is still faithfully withholding it. The canary is run with positive
controls — the admitted sibling id must survive the same call that drops the
quarantined one, so "absent" cannot mean "nothing was harvested".

**The registry half is a merge, not a second registry.** Promotion semantics now
live in ``model_gate`` beside the load gate, and ``WeightRegistry`` is a facade.
Two things are pinned: a sub-``min_improvement`` promotion is REFUSED and the
refusal is written to a ledger that survives the process (the predecessor's
revert log was a ``deque`` that died with it), and ``online_trainer`` holds no
second copy of the rule — asserted structurally, not by reading the diff.

The last group is the flag-off contract: with ``v4.online_training`` unset the
dream cycle writes the same report bytes, creates no training directory,
``index_stats`` grows no key, the probe emits no log record, and
``online_trainer`` is not imported at all.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import model_gate, online_trainer
from mind_mem.dream_cycle import run_dream_cycle
from mind_mem.init_workspace import init
from mind_mem.interaction_signals import SignalStore, SignalType
from mind_mem.model_gate import (
    EVENT_PROMOTE_REFUSED,
    EVENT_PROMOTED,
    EVENT_REVERTED,
    MIN_IMPROVEMENT_DEFAULT,
    REASON_INSUFFICIENT_IMPROVEMENT,
    REASON_LOAD_GATE_REFUSED,
    REASON_NO_CANDIDATE,
    REASON_NO_ROLLBACK,
    REASON_PROMOTED,
    REASON_REVERTED,
    WeightRef,
    evaluate_promotion,
    promotion_events,
    promotion_stats,
)
from mind_mem.online_trainer import (
    ONLINE_TRAINING_FLAG,
    TrainingLoop,
    TrainingTuple,
    WeightRegistry,
    build_training_tuples,
    cursor_path,
    harvest_stats,
    harvest_tuples,
    promote_candidate,
    queue_path,
    read_cursor,
    run_harvest_job,
    workspace_admitted_ids,
)

ADMITTED_ID = "DEC-001"
WITHHELD_ID = "DEC-002"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Recorder(logging.Handler):
    """Collects every record, from mind-mem's own non-propagating loggers too."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _capture_logs:
    """Attach a recorder to root AND to every live ``mind-mem.*`` logger.

    ``StructuredLogger`` sets ``propagate = False`` on its own logger, so a
    plain ``caplog`` (root only) sees nothing it emits — a silence assertion
    built on caplog alone would pass no matter how loud the code was.
    """

    def __enter__(self) -> _Recorder:
        self.handler = _Recorder()
        self.targets: list[logging.Logger] = [logging.getLogger()]
        manager = logging.Logger.manager
        for name in list(manager.loggerDict):
            if name.startswith("mind-mem") or name.startswith("mind_mem"):
                logger = logging.getLogger(name)
                if not logger.propagate:
                    self.targets.append(logger)
        self.saved = [(t, t.level) for t in self.targets]
        for target in self.targets:
            target.addHandler(self.handler)
            target.setLevel(logging.DEBUG)
        return self.handler

    def __exit__(self, *exc: object) -> None:
        for target, level in self.saved:
            target.removeHandler(self.handler)
            target.setLevel(level)


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, flag: bool) -> str:
    """An initialised workspace whose own config carries the flag.

    The corpus holds two blocks with the SAME topic words: one admitted, one
    quarantined. Every admission assertion below uses the pair, so a green
    result cannot mean "the harvest found nothing".
    """
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    init(ws)

    config_path = os.path.join(ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if flag:
        blob["v4"] = {**blob.get("v4", {}), ONLINE_TRAINING_FLAG: {"enabled": True}}
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    monkeypatch.setenv("MIND_MEM_CONFIG", config_path)

    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(f"\n[{ADMITTED_ID}]\nType: Decision\nStatement: recall scoring is deterministic\nStatus: Active\n\n")
        fh.write(f"[{WITHHELD_ID}]\nType: Decision\nStatement: recall scoring is deterministic\nStatus: quarantined\n\n")
    return ws


def _observe_correction(ws: str, *, results: list[str], session: str = "s1") -> None:
    """Append one CORRECTION signal naming *results* as what the user rejected."""
    store = SignalStore(os.path.join(ws, "memory", "interaction_signals.jsonl"))
    signal = store.observe_pair(
        session_id=session,
        previous_query="how do we score recall",
        new_query="no, i meant how do we score recall latency",
        previous_results=results,
    )
    assert signal is not None and signal.signal_type is SignalType.CORRECTION, "fixture must produce a correction signal"


def _queued(ws: str) -> list[dict[str, Any]]:
    path = queue_path(ws)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the gate registry — and therefore the promotion ledger — at tmp."""
    registry = tmp_path / "gate" / "registry.json"
    monkeypatch.setenv("MIND_MEM_GATE_REGISTRY", str(registry))
    monkeypatch.delenv("MIND_MEM_PROMOTION_LEDGER", raising=False)
    return registry.parent / model_gate.DEFAULT_PROMOTION_FILENAME


@pytest.fixture
def clean_ckpt(tmp_path: Path) -> str:
    """A safetensors-only checkpoint that passes every audit check."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","base_model":"Qwen/Qwen3-8B"}', encoding="utf-8")
    body = b'{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"\x00" * 8)
    return str(root)


@pytest.fixture(autouse=True)
def _reset_flag_warning() -> Iterator[None]:
    from mind_mem.v4 import feature_flags as ff

    ff._last_config_warning = None
    yield
    ff._last_config_warning = None


# ---------------------------------------------------------------------------
# The mapping: a correction's prior results are NEGATIVES
# ---------------------------------------------------------------------------


class TestCorrectionMapping:
    def test_correction_makes_prior_results_negatives(self) -> None:
        """The whole point of the signal taxonomy. Inverting this trains the
        retriever toward the blocks the user explicitly rejected."""
        (tup,) = build_training_tuples(
            [{"signal_type": "correction", "new_query": "q", "previous_results": ["A", "B"]}],
        )
        assert tup.negative_ids == ("A", "B")
        assert tup.positive_ids == ()
        assert tup.weight == 1.25  # an explicit correction is the strongest signal

    def test_re_query_makes_prior_results_positives(self) -> None:
        """The positive control for the test above: the SAME prior results under
        a re-query become POSITIVES, so "negatives" is a decision, not a default."""
        (tup,) = build_training_tuples(
            [{"signal_type": "re_query", "new_query": "q", "previous_results": ["A", "B"]}],
        )
        assert tup.positive_ids == ("A", "B")
        assert tup.negative_ids == ()

    def test_build_is_pure(self) -> None:
        signals = [{"signal_type": "correction", "new_query": "q", "previous_results": ["A"]}]
        assert build_training_tuples(signals) == build_training_tuples(signals)


# ---------------------------------------------------------------------------
# The harvest is admission-filtered
# ---------------------------------------------------------------------------


class TestAdmission:
    def test_admitted_set_comes_from_admit_corpus(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        admitted = workspace_admitted_ids(ws)
        assert ADMITTED_ID in admitted, "positive control: the active block must be admitted"
        assert WITHHELD_ID not in admitted

    def test_quarantined_prior_result_never_becomes_a_negative(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The quarantine canary, with its positive control in the same tuple.

        Both ids were returned to the user and both are named by the signal.
        One is admitted, one is not — and only the admitted one may reach a
        durable training artifact.
        """
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID, WITHHELD_ID])

        report = run_harvest_job(ws)

        assert report["ids_withheld"] == 1
        (row,) = _queued(ws)
        assert row["negative_ids"] == [ADMITTED_ID], "the admitted sibling proves the harvest ran at all"
        assert WITHHELD_ID not in json.dumps(row)

    def test_tuple_with_only_withheld_ids_is_dropped_not_emptied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[WITHHELD_ID])

        report = run_harvest_job(ws)

        assert report["tuples_built"] == 1
        assert report["tuples_kept"] == 0
        assert report["tuples_dropped"] == 1
        assert _queued(ws) == []

    def test_harvest_tuples_requires_the_admitted_set(self) -> None:
        """Fail-closed by signature: there is no default that skips admission."""
        with pytest.raises(TypeError):
            harvest_tuples([{"signal_type": "correction", "new_query": "q", "previous_results": ["A"]}])  # type: ignore[call-arg]

    def test_unknown_id_is_withheld_even_with_a_permissive_corpus(self) -> None:
        kept, counters = harvest_tuples(
            [{"signal_type": "correction", "new_query": "q", "previous_results": ["A", "GHOST"]}],
            admitted_ids=frozenset({"A"}),
        )
        assert [t.negative_ids for t in kept] == [("A",)]
        assert counters["ids_withheld"] == 1


# ---------------------------------------------------------------------------
# The drain: cursor semantics
# ---------------------------------------------------------------------------


class TestDrain:
    def test_signals_are_consumed_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])

        first = run_harvest_job(ws)
        second = run_harvest_job(ws)

        assert first["signals_new"] == 1
        assert second["signals_new"] == 0
        assert len(_queued(ws)) == 1, "a second pass must not duplicate the queue"
        assert read_cursor(ws)["consumed"] == 1

    def test_limit_bounds_one_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        for i in range(3):
            _observe_correction(ws, results=[ADMITTED_ID], session=f"s{i}")

        assert run_harvest_job(ws, limit=2)["signals_new"] == 2
        assert run_harvest_job(ws, limit=2)["signals_new"] == 1

    def test_truncated_ledger_resyncs_instead_of_skipping(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cursor pointing past the end of a replaced ledger must restart,
        not silently declare everything already harvested."""
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID], session="s0")
        run_harvest_job(ws)

        # The ledger is replaced under the cursor (restore from backup, rotation).
        ledger = os.path.join(ws, "memory", "interaction_signals.jsonl")
        os.unlink(ledger)
        _observe_correction(ws, results=[ADMITTED_ID], session="s9")

        report = run_harvest_job(ws)

        assert report["resynced"] is True
        assert report["signals_new"] == 1

    def test_harvest_stats_never_drains(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])

        before = harvest_stats(ws)
        assert before == {"consumed": 0, "signals_total": 1, "signals_pending": 1, "queued_tuples": 0}
        assert harvest_stats(ws) == before, "a read must not advance the cursor"

        run_harvest_job(ws)
        assert harvest_stats(ws) == {
            "consumed": 1,
            "signals_total": 1,
            "signals_pending": 0,
            "queued_tuples": 1,
        }

    def test_report_carries_signal_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])
        assert run_harvest_job(ws)["signal_stats"]["correction"] == 1


# ---------------------------------------------------------------------------
# The wiring: dream_cycle is the caller
# ---------------------------------------------------------------------------


class TestDreamCycleWiring:
    def test_dream_cycle_runs_the_harvest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails if the Pass 4b call site is deleted — nothing else in the
        product drains the signal ledger."""
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])

        report = run_dream_cycle(ws)

        assert report.errors == ()
        assert [row["negative_ids"] for row in _queued(ws)] == [[ADMITTED_ID]]
        assert read_cursor(ws)["consumed"] == 1

    def test_dry_run_harvests_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])

        run_dream_cycle(ws, dry_run=True)

        assert _queued(ws) == []
        assert not os.path.exists(cursor_path(ws))

    def test_a_failing_harvest_does_not_fail_the_cycle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)

        def _boom(*_a: object, **_k: object) -> dict:
            raise RuntimeError("trainer exploded")

        monkeypatch.setattr(online_trainer, "run_harvest_job", _boom)
        report = run_dream_cycle(ws)

        assert any("signal harvest" in e for e in report.errors)
        assert report.timestamp  # the rest of the cycle still produced a report


# ---------------------------------------------------------------------------
# Flag OFF — nothing observable changed
# ---------------------------------------------------------------------------


class TestFlagOff:
    def test_flag_is_registered(self) -> None:
        from mind_mem.v4.feature_flags import ALL_V4_FLAGS, is_enabled

        assert ONLINE_TRAINING_FLAG in ALL_V4_FLAGS
        assert is_enabled(ONLINE_TRAINING_FLAG) is False

    def test_run_harvest_job_is_inert_and_silent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=False)
        _observe_correction(ws, results=[ADMITTED_ID])

        with _capture_logs() as recorder:
            report = run_harvest_job(ws)

        assert report == {"enabled": False}
        assert not os.path.exists(os.path.join(ws, "memory", "training"))
        assert recorder.records == [], "an OFF probe that logs is an observable difference"

    def test_dream_cycle_output_is_byte_identical(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The integrity summary the cycle writes must not gain a byte.

        Both runs are stamped with the same INJECTED instant, so the whole
        summary is compared -- the header line included. It used to drop
        that line and compare the rest, which left the footer's copy of the
        same stamp in the comparison and made the test a race: it passed
        only when both cycles landed inside one wall-clock second. Windows,
        the slowest row, lost that race (CI run 33915724655, index 25:
        ``...at 2026-09-04T20:47:46`` vs ``...:47``), and had been losing it
        on 3.10 for longer.

        The stamp is not dropped, because a summary that stopped carrying
        one would then pass unnoticed. It is pinned instead, and the two
        lines that carry it are asserted to be present, well-formed and to
        name the instant they were given -- which also proves the injection
        is honoured rather than quietly ignored in favour of the clock.
        """
        stamp = datetime(2020, 1, 2, 3, 4, 5)
        stamped = "2020-01-02T03:04:05"

        def _summary(ws: str) -> list[str]:
            files = sorted(Path(ws, "memory").glob("dream-cycle-*.md"))
            assert files, "positive control: the cycle must have written a summary"
            return files[0].read_text(encoding="utf-8").replace(ws, "<WS>").splitlines()

        ws_off = _workspace(tmp_path / "off", monkeypatch, flag=False)
        _observe_correction(ws_off, results=[ADMITTED_ID])
        run_dream_cycle(ws_off, instant=stamp)
        off_lines = _summary(ws_off)

        ws_on = _workspace(tmp_path / "on", monkeypatch, flag=True)
        _observe_correction(ws_on, results=[ADMITTED_ID])
        run_dream_cycle(ws_on, instant=stamp)

        assert off_lines == _summary(ws_on)
        stamped_lines = [ln for ln in off_lines if stamped in ln]
        assert stamped_lines[:1] == [f"# Dream Cycle Report — {stamped}"], "the header stamp is gone or reworded"
        generated_at = f"*Generated by mind-mem dream cycle at {stamped}*"
        assert generated_at in stamped_lines, "the generated-at line is gone, or ignores the instant"
        assert not os.path.exists(os.path.join(ws_off, "memory", "training"))
        assert os.path.exists(queue_path(ws_on)), "positive control: ON really did harvest"

    def test_module_is_not_even_imported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent, not merely inert. Run in a subprocess because this test
        process has imported ``online_trainer`` at module scope."""
        ws_off = _workspace(tmp_path / "off", monkeypatch, flag=False)
        ws_on = _workspace(tmp_path / "on", monkeypatch, flag=True)
        script = (
            "import json, sys\n"
            "from mind_mem.dream_cycle import run_dream_cycle\n"
            "run_dream_cycle(sys.argv[1])\n"
            "print('mind_mem.online_trainer' in sys.modules)\n"
        )

        def _imported(ws: str) -> str:
            env = {**os.environ, "MIND_MEM_CONFIG": os.path.join(ws, "mind-mem.json")}
            out = subprocess.run(
                [sys.executable, "-c", script, ws], capture_output=True, text=True, env=env, check=True, encoding="utf-8", errors="replace"
            )
            return out.stdout.strip().splitlines()[-1]

        assert _imported(ws_off) == "False"
        assert _imported(ws_on) == "True", "positive control: the ON path really does import it"

    def test_index_stats_grows_no_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.mcp.tools.memory_ops import index_stats

        ws = _workspace(tmp_path, monkeypatch, flag=False)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        off = json.loads(index_stats.__wrapped__())
        assert "online_training" not in off
        assert "interaction_signals" in off, "positive control: the neighbouring section is present"

    def test_index_stats_reports_when_on(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.mcp.tools.memory_ops import index_stats

        ws = _workspace(tmp_path, monkeypatch, flag=True)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        _observe_correction(ws, results=[ADMITTED_ID])
        run_harvest_job(ws)

        on = json.loads(index_stats.__wrapped__())
        assert on["online_training"]["queued_tuples"] == 1
        assert on["online_training"]["signals_pending"] == 0
        assert "promotions" in on["online_training"]


# ---------------------------------------------------------------------------
# The promotion rule — refusals are persisted
# ---------------------------------------------------------------------------


class TestPromotionRule:
    def test_no_baseline_promotes(self) -> None:
        assert evaluate_promotion(candidate_mrr=0.1, baseline_mrr=None, min_improvement=0.01) == (
            True,
            REASON_PROMOTED,
        )

    def test_exact_threshold_promotes_and_a_hair_under_refuses(self) -> None:
        ok, _ = evaluate_promotion(candidate_mrr=0.51, baseline_mrr=0.50, min_improvement=0.01)
        assert ok is True
        refused, reason = evaluate_promotion(candidate_mrr=0.5099, baseline_mrr=0.50, min_improvement=0.01)
        assert refused is False
        assert reason == REASON_INSUFFICIENT_IMPROVEMENT

    def test_regression_and_noise_refuse_identically(self) -> None:
        regression = evaluate_promotion(candidate_mrr=0.40, baseline_mrr=0.50, min_improvement=0.01)
        noise = evaluate_promotion(candidate_mrr=0.505, baseline_mrr=0.50, min_improvement=0.01)
        assert regression == noise == (False, REASON_INSUFFICIENT_IMPROVEMENT)

    def test_the_rule_has_exactly_one_implementation(self) -> None:
        """Structural, not a diff review: ``online_trainer`` must contain no
        comparison of its own against ``min_improvement``."""
        source = Path(online_trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = [
            ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare) and "min_improvement" in (ast.get_source_segment(source, node) or "")
        ]
        assert offenders == [], f"second copy of the promotion rule: {offenders}"

    def test_registry_delegates_to_model_gate(self, monkeypatch: pytest.MonkeyPatch, isolated_ledger: Path) -> None:
        calls: list[dict] = []

        def _spy(model_id: str, **kwargs: Any) -> model_gate.PromotionDecision:
            calls.append({"model_id": model_id, **kwargs})
            return model_gate.PromotionDecision(ok=True, reason=REASON_PROMOTED, model_id=model_id)

        monkeypatch.setattr(online_trainer, "promote_weights", _spy)
        assert WeightRegistry().promote("emb", new_mrr=0.9)[0] is True
        assert calls and calls[0]["candidate_mrr"] == 0.9


class TestPromotionLedger:
    def _seed(self, ckpt: str, *, baseline: float = 0.50) -> WeightRegistry:
        registry = WeightRegistry()
        registry.set_active(WeightRef(model_id="emb", version="v1", path=ckpt, base_mrr=baseline, promoted_at="2026-01-01T00:00:00Z"))
        registry.set_candidate(WeightRef(model_id="emb", version="v2", path=ckpt, base_mrr=0.0))
        return registry

    def test_sub_min_improvement_promotion_is_refused_and_logged(self, clean_ckpt: str, isolated_ledger: Path) -> None:
        """The headline: refused, the active weights untouched, and the refusal
        written to a ledger that outlives the process."""
        registry = self._seed(clean_ckpt)

        ok, reason = registry.promote("emb", new_mrr=0.505, now="2026-09-01T00:00:00Z")

        assert ok is False
        assert "0.5050" in reason and "0.5100" in reason
        assert registry.active("emb") is not None and registry.active("emb").version == "v1"
        assert registry.candidate("emb") is not None, "a refused candidate is not consumed"

        # Read it back off disk, through no in-memory state at all.
        persisted = json.loads(isolated_ledger.read_text(encoding="utf-8"))
        (event,) = [e for e in persisted["events"] if e["event"] == EVENT_PROMOTE_REFUSED]
        assert event["reason"] == REASON_INSUFFICIENT_IMPROVEMENT
        assert event["candidate_mrr"] == 0.505
        assert event["baseline_mrr"] == 0.50
        assert event["at"] == "2026-09-01T00:00:00Z"

    def test_promote_candidate_reports_the_refusal(self, clean_ckpt: str, isolated_ledger: Path) -> None:
        registry = self._seed(clean_ckpt)
        decision = promote_candidate(
            registry,
            model_id="emb",
            candidate_mrr=0.505,
            baseline_mrr=0.50,
            min_improvement=MIN_IMPROVEMENT_DEFAULT,
        )
        assert decision["promoted"] is False
        assert decision["reason_code"] == REASON_INSUFFICIENT_IMPROVEMENT

    def test_a_stale_caller_baseline_cannot_talk_a_regression_through(self, clean_ckpt: str, isolated_ledger: Path) -> None:
        """The ledger's baseline is authoritative, not the number the caller
        quotes — otherwise the gate is advisory."""
        self._seed(clean_ckpt, baseline=0.90)
        registry = WeightRegistry()
        decision = promote_candidate(registry, model_id="emb", candidate_mrr=0.60, baseline_mrr=0.10)
        assert decision["promoted"] is False
        assert decision["ledger_baseline_mrr"] == 0.90

    def test_promotion_records_rollback_and_revert_survives_a_reload(self, clean_ckpt: str, isolated_ledger: Path) -> None:
        registry = self._seed(clean_ckpt)

        ok, _ = registry.promote("emb", new_mrr=0.60, now="2026-09-01T00:00:01Z")
        assert ok is True
        assert registry.active("emb").version == "v2"
        assert registry.active("emb").base_mrr == 0.60
        assert registry.rollback("emb").version == "v1"
        assert registry.candidate("emb") is None

        assert registry.revert("emb", "mrr regression in production", now="2026-09-01T00:00:02Z") is True
        assert registry.active("emb").version == "v1"

        fresh = WeightRegistry()  # no shared state with `registry`
        assert [e["event"] for e in promotion_events(limit=10, model_id="emb")] == [EVENT_PROMOTED, EVENT_REVERTED]
        assert fresh.stats()["revert_events"] == 1
        (revert_event,) = [e for e in promotion_events(limit=10) if e["event"] == EVENT_REVERTED]
        assert revert_event["reason"] == REASON_REVERTED
        assert revert_event["detail"] == "mrr regression in production"

    def test_revert_without_a_rollback_is_refused_and_logged(self, isolated_ledger: Path) -> None:
        assert WeightRegistry().revert("emb", "nothing to go back to") is False
        reasons = [e["reason"] for e in promotion_events(limit=10)]
        assert REASON_NO_ROLLBACK in reasons

    def test_missing_candidate_is_refused_and_logged(self, isolated_ledger: Path) -> None:
        ok, _ = WeightRegistry().promote("emb", new_mrr=0.9)
        assert ok is False
        assert [e["reason"] for e in promotion_events(limit=10)] == [REASON_NO_CANDIDATE]

    def test_unaudited_candidate_bytes_are_refused_by_the_load_gate(self, tmp_path: Path, isolated_ledger: Path) -> None:
        """The coupling the merge bought: promotion runs the load gate on the
        candidate's own bytes, so a weight registry write is no longer a way
        around ``gate_check``."""
        registry = WeightRegistry()
        registry.set_candidate(WeightRef(model_id="emb", version="v2", path=str(tmp_path / "ghost"), base_mrr=0.0))

        ok, reason = registry.promote("emb", new_mrr=0.9)

        assert ok is False
        assert "load gate refused" in reason
        assert [e["reason"] for e in promotion_events(limit=10)] == [REASON_LOAD_GATE_REFUSED]

    def test_promotion_stats_carries_no_checkpoint_contents(self, clean_ckpt: str, isolated_ledger: Path) -> None:
        registry = self._seed(clean_ckpt)
        registry.promote("emb", new_mrr=0.60)
        snapshot = promotion_stats()
        assert snapshot["models"] == ["emb"]
        assert snapshot["events_by_kind"] == {EVENT_PROMOTED: 1}
        assert snapshot["rollback_available"] == ["emb"]

    def test_corrupt_ledger_degrades_to_empty(self, isolated_ledger: Path) -> None:
        isolated_ledger.parent.mkdir(parents=True, exist_ok=True)
        isolated_ledger.write_text("{ not json", encoding="utf-8")
        assert model_gate.load_promotion_ledger() == {"models": {}, "events": []}
        # ... and a decision can still be made and recorded over it.
        assert WeightRegistry().promote("emb", new_mrr=0.9)[0] is False
        assert promotion_events(limit=5)


# ---------------------------------------------------------------------------
# TrainingLoop — the caller-supplied gradient step
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def _tuple(self, i: int) -> TrainingTuple:
        return TrainingTuple(query=f"q{i}", positive_ids=("A",), negative_ids=(), signal_type="re_query")

    def test_flushes_whole_batches_only(self) -> None:
        seen: list[int] = []
        loop = TrainingLoop(lambda batch: seen.append(len(batch)) or {}, batch_size=2)

        assert loop.submit([self._tuple(0)]) == 0
        assert seen == []
        assert loop.submit([self._tuple(1), self._tuple(2)]) == 1
        assert seen == [2]
        assert loop.stats()["buffered"] == 1
        assert loop.stats()["steps_run"] == 1

    def test_a_raising_train_step_is_counted_not_propagated(self) -> None:
        def _boom(_batch: list[TrainingTuple]) -> dict:
            raise RuntimeError("cuda oom")

        loop = TrainingLoop(_boom, batch_size=1)
        assert loop.submit([self._tuple(0), self._tuple(1)]) == 0
        assert loop.stats()["errors"] == 2
        assert loop.stats()["buffered"] == 0

    def test_overflow_is_counted(self) -> None:
        loop = TrainingLoop(lambda _b: {}, batch_size=2, buffer_cap=2)
        loop.submit([self._tuple(i) for i in range(5)])
        assert loop.stats()["overflow_dropped"] >= 1

    def test_non_tuples_are_ignored(self) -> None:
        loop = TrainingLoop(lambda _b: {}, batch_size=1)
        assert loop.submit(["not a tuple", None]) == 0  # type: ignore[list-item]
        assert loop.stats()["buffered"] == 0

    @pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": 8, "buffer_cap": 4}])
    def test_invalid_construction_refuses(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TrainingLoop(lambda _b: {}, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry point — the shape cron_runner invokes
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_reports_and_exits_nonzero_when_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=False)
        assert online_trainer.main([ws]) == 1
        assert json.loads(capsys.readouterr().out) == {"enabled": False}

    def test_main_drains_when_on(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        ws = _workspace(tmp_path, monkeypatch, flag=True)
        _observe_correction(ws, results=[ADMITTED_ID])
        assert online_trainer.main([ws]) == 0
        assert json.loads(capsys.readouterr().out)["tuples_kept"] == 1
