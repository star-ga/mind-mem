# Copyright 2026 STARGA, Inc.
"""``event_fanout`` is connected — and the connection is an EXIT, not an entrance.

``mind_mem.event_fanout`` shipped with a publisher protocol, two publishers and
a builder that nothing ever called: a publish-subscribe stream with no
publisher. 5.0.0 deleted it as unreachable. Restoring the file is not the fix;
wiring it is, and wiring it is the part that needs proving, because this module
is the one restored door that points OUTWARD.

That changes what the red-team question is. The other slice-3 modules open a
way IN, so the property to prove is that arriving content lands quarantined and
recall withholds it. Nothing here writes a block — the module imports no store,
no parser and no recall path — so admission is not the risk. The risk is the
mirror image: ``LoggingPublisher`` writes ``payload`` verbatim into the process
log and ``RedisStreamPublisher`` JSON-dumps it onto a stream any subscriber can
read, and BOTH sit outside the ACL and outside the gate. A payload carrying
block text would leak content that quarantine is still faithfully withholding
from recall — the block never becomes servable, and its text gets out anyway.

So the canary here is a LEAK canary, and it is run the same way the quarantine
suite runs its withholding canaries — with positive controls, so a green result
cannot mean "nothing happened":

* the apply really did quote the canary in its own log (so there was something
  to leak),
* the event really was published (so "absent from the payload" is not "absent
  because no event exists"),
* and only then: the canary appears in no payload, on no publisher, in no log
  record this module wrote.

The rest pins the three properties the governance surface depends on: exactly
one ``proposal_applied`` per real apply, a dead publisher cannot fail a
governed write, and with the flag off not even an ``Event`` is constructed.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mind_mem import event_fanout
from mind_mem.event_fanout import (
    EVENT_PROPOSAL_APPLIED,
    EVENT_ROLLBACK_EXECUTED,
    EVENT_TIER_DEMOTED,
    EVENT_TIER_PROMOTED,
    Event,
    register_publisher,
    reset_fanout_cache,
    scrub_payload,
)
from mind_mem.init_workspace import init
from mind_mem.lint import RULE_DUPLICATE_BLOCK

# The canary goes in the STATEMENT of the block the apply rewrites, so it is
# carried by the proposal, by the apply diff and by the apply log — every
# string an over-eager payload could plausibly pick up.
CANARY = "qwvbzntplasker"
CANARY_STATEMENT = f"Root credentials rotate through the {CANARY} channel."

DECISIONS = f"""
[D-20260101-001]
Date: 2026-01-01
Status: active
Scope: global
Statement: {CANARY_STATEMENT}
Rationale: Original ruling.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md

[D-20260104-004]
Date: 2026-01-04
Status: active
Scope: global
Statement: {CANARY_STATEMENT}
Rationale: Restated during onboarding.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md
"""

TWIN = "D-20260104-004"
WINNER = "D-20260101-001"

# Same allowance the lint-wiring suite makes: the validator reports issues on a
# freshly scaffolded workspace for reasons unrelated to this repair. Every
# other gate in the apply pipeline runs for real.
_PRECONDITIONS_PASS = patch(
    "mind_mem.apply_engine.check_preconditions",
    return_value=(True, ["validate: PASS (TOTAL 0 issues)"]),
)


# ---------------------------------------------------------------------------
# A capturing publisher, registered the way a downstream adapter would be
# ---------------------------------------------------------------------------

_CAPTURED: list[Event] = []
_BUILD_FAILURES: list[str] = []


class _CapturingPublisher:
    name = "capture"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def publish(self, event: Event) -> None:
        _CAPTURED.append(event)

    def close(self) -> None:
        return None


class _DeadPublisher:
    """Stands in for Redis with nothing listening on the other end."""

    name = "dead"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        return None

    def publish(self, event: Event) -> None:
        raise ConnectionError("Error 111 connecting to localhost:6379. Connection refused.")

    def close(self) -> None:
        raise ConnectionError("still refused")


def _unbuildable(config: dict[str, Any]) -> Any:
    """A publisher whose constructor blows up — e.g. ``import redis`` missing."""
    _BUILD_FAILURES.append("unbuildable")
    raise ModuleNotFoundError("No module named 'redis'")


register_publisher("capture", _CapturingPublisher)
register_publisher("dead", _DeadPublisher)
register_publisher("unbuildable", _unbuildable)


class _RecordSink(logging.Handler):
    """Captures what this module's own logger actually emitted.

    ``StructuredLogger`` sets ``propagate = False``, so ``caplog`` sees nothing
    from it. Reading the real logger is the point: ``LoggingPublisher`` is the
    default publisher and the log line IS the product behaviour being claimed.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def rendered(self) -> str:
        return json.dumps(
            [{"msg": r.getMessage(), "data": getattr(r, "data", None)} for r in self.records],
            default=str,
        )


@pytest.fixture(autouse=True)
def _clean_fanout_state():
    _CAPTURED.clear()
    _BUILD_FAILURES.clear()
    reset_fanout_cache()
    yield
    _CAPTURED.clear()
    _BUILD_FAILURES.clear()
    reset_fanout_cache()


@pytest.fixture
def sink():
    handler = _RecordSink()
    logger = logging.getLogger("mind-mem.event_fanout")
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _make_ws(tmp_path: Path, name: str = "ws", *, events: Any = None) -> str:
    ws = str(tmp_path / name)
    init(ws)

    config_path = os.path.join(ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["governance_mode"] = "enforce"
    if events is not None:
        config["events"] = events
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = "enforce"
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(DECISIONS)
    return ws


def _capture_events(**extra: Any) -> dict[str, Any]:
    publishers = ["capture", *extra.pop("also", [])]
    return {"enabled": True, "publishers": publishers, **extra}


def _stage_proposal(ws: str) -> str:
    """lint → lint_autofix: a real staged proposal that rewrites a real block."""
    from mind_mem.mcp.tools import lint as lint_tools

    report = json.loads(lint_tools.lint(rule=RULE_DUPLICATE_BLOCK))
    duplicates = [f for f in report["findings"] if f["rule"] == RULE_DUPLICATE_BLOCK]
    assert [f["block_id"] for f in duplicates] == [TWIN], report
    staged = json.loads(lint_tools.lint_autofix(duplicates[0]["finding_id"]))
    assert staged["status"] == "staged", staged
    return str(staged["proposal_id"])


def _status_of(ws: str, block_id: str) -> str:
    from mind_mem.block_parser import parse_file

    blocks = {b["_id"]: b for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))}
    return str(blocks[block_id].get("Status", ""))


def _files_holding_canary(ws: str) -> list[str]:
    """Every workspace file whose text carries the canary, workspace-relative."""
    found: list[str] = []
    for path in sorted(Path(ws).rglob("*")):
        if not path.is_file():
            continue
        try:
            if CANARY in path.read_text(encoding="utf-8", errors="replace"):
                found.append(path.relative_to(ws).as_posix())
        except OSError:  # pragma: no cover — an unreadable file is not a hit
            continue
    return found


@pytest.fixture(autouse=True)
def _isolate_mcp_env(monkeypatch, tmp_path):
    """Deterministic MCP inputs — explicit admin scope, fresh rate budget."""
    from mind_mem.mcp.infra import rate_limit as _rate_limit

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    monkeypatch.delenv("MIND_MEM_CONFIG", raising=False)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    yield
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()


def _enable_lint(ws: str) -> None:
    path = os.path.join(ws, "mind-mem.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("v4", {})["lint"] = {"enabled": True}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _applied_receipt_ts(ws: str) -> str:
    applied = sorted(os.listdir(os.path.join(ws, "intelligence", "applied")))
    stamps = [d for d in applied if re.match(r"^\d{8}-\d{6}$", d)]
    assert stamps, f"no apply receipt directory in {applied}"
    return stamps[-1]


# ---------------------------------------------------------------------------
# The door: proposal_applied
# ---------------------------------------------------------------------------


class TestProposalApplied:
    def test_exactly_one_event_per_real_apply_and_none_on_a_dry_run(self, tmp_path, monkeypatch, sink) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events=_capture_events(also=["logging"]))
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)

        dry = json.loads(governance.approve_apply(proposal_id, dry_run=True))
        assert dry["status"] == "dry_run_passed", dry
        assert [e.kind for e in _CAPTURED] == [], "a dry run must not announce an apply that did not happen"

        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))
        assert applied["status"] == "applied", applied
        assert _status_of(ws, TWIN) == "superseded", "positive control: the apply really changed the corpus"

        applies = [e for e in _CAPTURED if e.kind == EVENT_PROPOSAL_APPLIED]
        assert len(applies) == 1, [e.kind for e in _CAPTURED]
        assert applies[0].payload["proposal_id"] == proposal_id
        assert applies[0].workspace == ws

        # ...and the LoggingPublisher wrote exactly one line for it, which is
        # the operator-visible half of "working".
        lines = [
            r
            for r in sink.records
            if r.getMessage() == "event_fanout" and (getattr(r, "data", None) or {}).get("kind") == EVENT_PROPOSAL_APPLIED
        ]
        assert len(lines) == 1, sink.rendered()

    def test_the_payload_carries_ids_and_hashes_only(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events=_capture_events())
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)
        with _PRECONDITIONS_PASS:
            assert json.loads(governance.approve_apply(proposal_id, dry_run=False))["status"] == "applied"

        payload = _CAPTURED[0].payload
        assert set(payload) == {"proposal_id", "success", "log_digest"}
        assert payload["success"] is True
        assert re.fullmatch(r"[0-9a-f]{16}", payload["log_digest"]), payload


class TestLeakCanary:
    """The security property of THIS door, run with positive controls."""

    def test_block_text_never_reaches_a_payload_a_log_or_the_wire(self, tmp_path, monkeypatch, sink) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events=_capture_events(also=["logging"]))
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)
        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))
        assert applied["status"] == "applied", applied

        # Positive control 1 — there WAS something to leak, and it was inside
        # the apply's own working set: the staged proposal and the corpus file
        # the engine rewrote both quote the canary verbatim.
        carriers = _files_holding_canary(ws)
        assert any(c.endswith("decisions/DECISIONS.md") for c in carriers), carriers
        assert any("/proposed/" in c or "/applied/" in c for c in carriers), carriers
        # Positive control 2 — an event really was published, so an absent
        # canary is absence from a real payload, not absence of a payload.
        assert [e.kind for e in _CAPTURED] == [EVENT_PROPOSAL_APPLIED], [e.kind for e in _CAPTURED]

        for event in _CAPTURED:
            assert CANARY not in json.dumps(event.payload, default=str)
            assert CANARY not in json.dumps(event.to_wire(), default=str)
        assert CANARY not in sink.rendered(), "block text reached the process log through the fan-out"

    def test_an_emit_site_that_tries_to_pass_content_is_refused(self) -> None:
        """The guard is on the TYPE, so a careless future call site cannot leak.

        Dropped, not truncated: the first 128 characters of a Statement is
        still a Statement.
        """
        event = Event(
            kind=EVENT_PROPOSAL_APPLIED,
            payload={"statement": CANARY_STATEMENT, "block_id": TWIN},
            workspace="/tmp/ws",
        )
        assert CANARY not in json.dumps(event.to_wire(), default=str)
        assert event.payload["block_id"] == TWIN
        assert event.payload["_dropped"] == ["statement"]


class TestPayloadDiscipline:
    def test_numbers_pass_under_any_key_because_they_cannot_carry_prose(self) -> None:
        assert scrub_payload({"x": 1, "ratio": 0.5, "ok": True, "nothing": None}) == {
            "x": 1,
            "ratio": 0.5,
            "ok": True,
            "nothing": None,
        }

    def test_text_passes_only_under_an_id_bearing_key(self) -> None:
        clean = scrub_payload({"block_id": "D-1", "from_tier": "SHARED", "note": "a short note"})
        assert clean == {"block_id": "D-1", "from_tier": "SHARED", "_dropped": ["note"]}

    def test_long_multiline_and_nested_values_are_dropped(self) -> None:
        clean = scrub_payload(
            {
                "block_id": "x" * 400,
                "proposal_id": "line one\nline two",
                "meta_id": {"nested": "content"},
                "ids": ["D-1", {"deep": "content"}],
            }
        )
        assert clean == {"_dropped": ["block_id", "ids", "meta_id", "proposal_id"]}

    def test_a_non_mapping_payload_is_refused_wholesale(self) -> None:
        assert scrub_payload(CANARY_STATEMENT) == {"_dropped": ["<non_mapping_payload>"]}
        assert scrub_payload(None) == {"_dropped": ["<non_mapping_payload>"]}

    def test_key_count_is_bounded(self) -> None:
        clean = scrub_payload({f"k{i}": i for i in range(64)})
        assert len([k for k in clean if k != "_dropped"]) == event_fanout._MAX_PAYLOAD_KEYS
        assert len(clean["_dropped"]) == 64 - event_fanout._MAX_PAYLOAD_KEYS


# ---------------------------------------------------------------------------
# Fan-out must never be able to fail a governed write
# ---------------------------------------------------------------------------


class TestFanOutCannotFailAGovernedWrite:
    def test_a_dead_publisher_leaves_the_apply_applied(self, tmp_path, monkeypatch) -> None:
        """Redis down. The apply still succeeds and the corpus still changed."""
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events={"enabled": True, "publishers": ["dead", "capture"]})
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)

        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))

        assert applied["status"] == "applied", applied
        assert applied["success"] is True
        assert _status_of(ws, TWIN) == "superseded"
        assert _status_of(ws, WINNER) == "active"
        # The surviving publisher still got it — one dead leg does not eat the event.
        assert [e.kind for e in _CAPTURED] == [EVENT_PROPOSAL_APPLIED]

    def test_a_publisher_that_cannot_even_be_built_leaves_the_apply_applied(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events={"enabled": True, "publishers": ["unbuildable"]})
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)

        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))

        assert applied["status"] == "applied", applied
        assert _status_of(ws, TWIN) == "superseded"
        assert _BUILD_FAILURES, "positive control: the unbuildable publisher was actually attempted"

    def test_a_corrupt_events_block_leaves_the_apply_applied(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events={"enabled": True, "publishers": "not-a-list"})
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)

        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))
        assert applied["status"] == "applied", applied


# ---------------------------------------------------------------------------
# Flag OFF — byte-identical to the unwired build
# ---------------------------------------------------------------------------


class _NeverConstruct:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("an Event was constructed while events.enabled was off")


class TestFlagOff:
    @pytest.mark.parametrize(
        "events",
        [None, {"enabled": False, "publishers": ["capture"]}, {"publishers": ["capture"]}],
        ids=["absent", "false", "no_enabled_key"],
    )
    def test_nothing_is_published_and_no_event_is_even_constructed(self, tmp_path, monkeypatch, sink, events) -> None:
        """The probe itself must be unobservable: no Event, no log, no publisher.

        Patching ``Event`` to explode is the sharp version of the claim — it
        fails if the flag is read AFTER the payload is built, which is where a
        clock read and a wasted digest would sneak back in.
        """
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events=events)
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)

        monkeypatch.setattr(event_fanout, "Event", _NeverConstruct)
        with _PRECONDITIONS_PASS:
            applied = json.loads(governance.approve_apply(proposal_id, dry_run=False))

        assert applied["status"] == "applied", applied
        assert _status_of(ws, TWIN) == "superseded"
        assert _CAPTURED == []
        assert sink.records == [], sink.rendered()

    def test_the_applied_corpus_matches_a_build_with_no_events_config(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance

        results = {}
        corpora = {}
        for name, events in (("off", {"enabled": False}), ("absent", None)):
            ws = _make_ws(tmp_path, name, events=events)
            _enable_lint(ws)
            monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
            proposal_id = _stage_proposal(ws)
            with _PRECONDITIONS_PASS:
                result = json.loads(governance.approve_apply(proposal_id, dry_run=False))
            # Both carry the workspace path (receipt location); that differs
            # by construction, so normalise it and compare everything else.
            #
            # The receipt directory is stamped ``YYYYMMDD-HHMMSS``, so the two
            # applies below disagree whenever they straddle a second boundary
            # -- which is exactly what happened on a CI runner (204849 vs
            # 204850) while passing locally every time. A byte-identity
            # assertion that includes a wall-clock read is not testing
            # byte-identity, it is testing how fast the machine is.
            result.pop("log", None)
            result["message"] = result["message"].replace(ws, "<ws>")
            result["message"] = re.sub(r"/applied/\d{8}-\d{6}/", "/applied/<stamp>/", result["message"])
            results[name] = result
            corpora[name] = Path(ws, "decisions", "DECISIONS.md").read_text(encoding="utf-8")

        assert results["off"] == results["absent"]
        assert corpora["off"] == corpora["absent"]
        assert _CAPTURED == []


# ---------------------------------------------------------------------------
# The other emit sites
# ---------------------------------------------------------------------------


class TestRollbackExecuted:
    def test_one_event_per_rollback_with_no_reason_text(self, tmp_path, monkeypatch) -> None:
        from mind_mem.mcp.tools import governance

        ws = _make_ws(tmp_path, events=_capture_events())
        _enable_lint(ws)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        proposal_id = _stage_proposal(ws)
        with _PRECONDITIONS_PASS:
            assert json.loads(governance.approve_apply(proposal_id, dry_run=False))["status"] == "applied"

        receipt_ts = _applied_receipt_ts(ws)
        secret_reason = f"reverting because {CANARY} was wrong"
        rolled = json.loads(governance.rollback_proposal(receipt_ts, reason=secret_reason))
        assert rolled["status"] == "rolled_back", rolled

        events = [e for e in _CAPTURED if e.kind == EVENT_ROLLBACK_EXECUTED]
        assert len(events) == 1, [e.kind for e in _CAPTURED]
        payload = events[0].payload
        assert payload["receipt_ts"] == receipt_ts
        assert payload["reason_length"] == len(secret_reason)
        assert CANARY not in json.dumps(payload, default=str), "the rollback reason leaked into the payload"


class TestTierEvents:
    def test_promotion_and_demotion_emit_id_only_payloads(self, tmp_path) -> None:
        from mind_mem.memory_tiers import DemotionReason, MemoryTier, TierManager

        ws = _make_ws(tmp_path, events=_capture_events())
        db_path = os.path.join(ws, "intelligence", "tiers.db")
        with TierManager(db_path, workspace=ws) as mgr:
            mgr._register_block(f"D-{CANARY}", MemoryTier.WORKING)
            assert mgr.promote(f"D-{CANARY}", MemoryTier.SHARED) is True
            assert mgr.demote(f"D-{CANARY}", MemoryTier.WORKING, DemotionReason.STALE) is True
            # A rejected move is not a tier change, so it is not an event.
            assert mgr.promote(f"D-{CANARY}", MemoryTier.VERIFIED) is False

        kinds = [e.kind for e in _CAPTURED]
        assert kinds == [EVENT_TIER_PROMOTED, EVENT_TIER_DEMOTED], kinds
        assert _CAPTURED[0].payload == {
            "block_id": f"D-{CANARY}",
            "from_tier": "WORKING",
            "to_tier": "SHARED",
        }
        assert _CAPTURED[1].payload["reason_code"] == DemotionReason.STALE.value

    def test_a_manager_without_a_workspace_is_exactly_the_old_behaviour(self, tmp_path) -> None:
        """The default is None, so every pre-5.1.0 construction site is inert."""
        from mind_mem.memory_tiers import MemoryTier, TierManager

        ws = _make_ws(tmp_path, events=_capture_events())
        with TierManager(os.path.join(ws, "intelligence", "tiers.db")) as mgr:
            mgr._register_block("D-1", MemoryTier.WORKING)
            assert mgr.promote("D-1", MemoryTier.SHARED) is True
        assert _CAPTURED == []


# ---------------------------------------------------------------------------
# Structural rails
# ---------------------------------------------------------------------------


_SRC = Path(__file__).resolve().parents[1] / "src" / "mind_mem"


def _product_sources() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if p.name != "event_fanout.py"]


class TestStructuralRails:
    def test_every_emit_site_goes_through_the_scrubbing_funnel(self) -> None:
        """No product module may build an Event or a fanout by hand.

        ``emit_event`` is where the flag check, the failure swallowing and the
        payload scrub live. A module that imported ``EventFanout`` and called
        ``publish`` directly would have none of them, which is precisely the
        shape of the bug this door exists to avoid.
        """
        allowed = {"emit_event", "is_fanout_enabled", "reset_fanout_cache", "scrub_payload"}
        importers: list[str] = []
        offenders: list[str] = []
        for path in _product_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or (node.module or "").split(".")[-1] != "event_fanout":
                    continue
                importers.append(path.name)
                bad = sorted(a.name for a in node.names if not a.name.startswith("EVENT_") and a.name not in allowed)
                if bad:
                    offenders.append(f"{path.name}: {bad}")
        assert offenders == [], offenders
        # Positive control: a rail over an empty set proves nothing.
        assert {"governance.py", "memory_tiers.py"} <= set(importers), importers

    def test_the_fan_out_module_reads_and_writes_no_blocks(self) -> None:
        """An egress door that touched the corpus would need the admission gate.

        It does not touch the corpus: payloads are handed to it by the caller,
        so there is no block read to filter and no block write to admit. This
        rail is what keeps that true.
        """
        text = (_SRC / "event_fanout.py").read_text(encoding="utf-8")
        forbidden = (
            "block_store",
            "storage",
            "block_parser",
            "_recall_core",
            "recall",
            "governance_gate",
            "write_block",
        )
        hits = [word for word in forbidden if re.search(rf"^\s*(from|import)\s+.*\b{word}\b", text, re.MULTILINE)]
        assert hits == [], hits

    def test_no_new_mcp_tool_was_added_by_this_slice(self) -> None:
        """This wiring adds no tool, so there is no ACL classification to make.

        Stated as a test rather than a comment: an unclassified tool is
        silently unreachable, and the cheapest way to keep that true is to
        pin that the fan-out surface is emit-only.
        """
        from mind_mem.mcp.infra import acl

        assert not {t for t in acl.ADMIN_TOOLS | acl.USER_TOOLS if "event" in t or "fanout" in t}
