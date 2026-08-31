"""quality_gate rule 6 (``near_duplicate``) must actually execute in the product.

The rule is one of the eight the module docstring advertises, and
"near-duplicates from re-runs" is named there as a production failure mode
it was built for.  It was nonetheless dead: ``recent`` is keyword-only and
defaults to ``None``, and neither call site in the shipped surface —
``mcp/tools/governance.py`` (``propose_update``, the enforcer) nor
``mcp/tools/quality.py`` (``validate_block``, the preview) — ever supplied
a window.  With no window the rule cannot run, so a re-proposal 97%+
identical to one staged minutes earlier was accepted by both.

Every test below fails on the pre-fix tree: without ``recent`` the rule is
reported as skipped and the near-duplicate statement is accepted.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace

# 97%+ similar to _PRIOR below, but not byte-identical, so the
# ContentHash dedupe already in append_signals does not catch it.
_PRIOR = "STARGA pins the recall reranker to the ONNX cross-encoder because the pure-python path missed the latency budget."
_NEAR_DUP = "STARGA pins the recall reranker to the ONNX cross-encoder because the pure-python path missed the latency budgets."
_DISTINCT = "The governance gate refuses every write whose spec binding hash has drifted from the recorded configuration."


def _write_config(ws: str, payload: dict) -> None:
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _seed_signal(ws: str, text: str, *, day: dt.date) -> None:
    """Append a SIGNALS.md block in the exact shape append_signals writes."""
    path = os.path.join(ws, "intelligence", "SIGNALS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = day.strftime("%Y-%m-%d")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n[SIG-{day.strftime('%Y%m%d')}-001]\n")
        fh.write(f"Date: {stamp}\n")
        fh.write("Type: auto-capture-decision\n")
        fh.write("Status: pending\n")
        fh.write(f"Excerpt: {text}\n")
        fh.write("\n---\n")


@pytest.fixture(autouse=True)
def _admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


@pytest.fixture()
def ws(tmp_path) -> str:
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    init(workspace)
    return workspace


# ---------------------------------------------------------------------------
# The window builder
# ---------------------------------------------------------------------------


class TestRecentStatements:
    def test_returns_todays_signal_excerpts(self, ws: str) -> None:
        from mind_mem.mcp.tools.governance import _recent_statements

        today = dt.datetime.now(dt.timezone.utc).date()
        _seed_signal(ws, _PRIOR, day=today)

        window = _recent_statements(ws)
        assert [t for t, _ in window] == [_PRIOR]
        assert window[0][1].tzinfo is not None

    def test_drops_blocks_older_than_the_lookback(self, ws: str) -> None:
        from mind_mem.mcp.tools.governance import _recent_statements

        today = dt.datetime.now(dt.timezone.utc).date()
        _seed_signal(ws, _PRIOR, day=today - dt.timedelta(days=30))

        assert _recent_statements(ws) == []

    def test_missing_signals_file_yields_empty_window_and_never_raises(self, tmp_path) -> None:
        """A probe on the write hot path must not be able to raise."""
        from mind_mem.mcp.tools.governance import _recent_statements

        assert _recent_statements(str(tmp_path / "does-not-exist")) == []

    def test_unreadable_signals_file_yields_empty_window(self, ws: str) -> None:
        from mind_mem.mcp.tools.governance import _recent_statements

        path = os.path.join(ws, "intelligence", "SIGNALS.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.chmod(path, 0o000)
        try:
            if os.access(path, os.R_OK):  # running as root — the chmod is inert
                pytest.skip("cannot make a file unreadable as this user")
            assert _recent_statements(ws) == []
        finally:
            os.chmod(path, 0o600)

    def test_a_raising_parser_degrades_to_an_empty_window(self, ws: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The window is built on the write path: it must never be able to
        break a proposal. A raise here would stop every propose_update —
        strictly worse than the missing rule it exists to restore."""
        import mind_mem.mcp.tools.governance as gov

        _seed_signal(ws, _PRIOR, day=dt.datetime.now(dt.timezone.utc).date())

        def _boom(_text: str) -> list[dict]:
            raise RuntimeError("corpus parser blew up")

        monkeypatch.setattr(gov, "parse_blocks", _boom)

        assert gov._recent_statements(ws) == []
        with use_workspace(ws):
            out = json.loads(
                gov.propose_update(
                    block_type="decision",
                    statement=_NEAR_DUP,
                    rationale="rationale long enough to pass the gate",
                )
            )
        assert out.get("status") == "proposed", out

    def test_tail_truncation_drops_only_the_partial_leading_block(self, ws: str) -> None:
        """A huge SIGNALS.md is read from the tail; no half block enters."""
        import mind_mem.mcp.tools.governance as gov

        today = dt.datetime.now(dt.timezone.utc).date()
        path = os.path.join(ws, "intelligence", "SIGNALS.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n[SIG-19990101-001]\nDate: 1999-01-01\nExcerpt: ancient\n\n---\n")
            fh.write("#" * (gov._RECENT_TAIL_BYTES + 4096))
            fh.write("\n")
        _seed_signal(ws, _PRIOR, day=today)

        window = gov._recent_statements(ws)
        assert [t for t, _ in window] == [_PRIOR]


# ---------------------------------------------------------------------------
# The enforcer: propose_update
# ---------------------------------------------------------------------------


class TestProposeUpdateRunsRule6:
    def _propose(self, ws: str, statement: str) -> dict:
        import mind_mem.mcp.tools.governance as gov

        with use_workspace(ws):
            return json.loads(
                gov.propose_update(
                    block_type="decision",
                    statement=statement,
                    rationale="rationale long enough to pass the gate",
                )
            )

    def test_strict_mode_rejects_a_near_duplicate_of_a_recent_signal(self, ws: str) -> None:
        """Fails pre-fix: with no window the rule cannot fire, so this is accepted."""
        _write_config(ws, {"quality_gate": {"mode": "strict"}})
        _seed_signal(ws, _PRIOR, day=dt.datetime.now(dt.timezone.utc).date())

        out = self._propose(ws, _NEAR_DUP)
        assert out.get("error") == "quality_gate_rejection", out
        assert any(r.startswith("near_duplicate") for r in out["reasons"]), out["reasons"]

    def test_strict_mode_still_accepts_a_distinct_statement(self, ws: str) -> None:
        _write_config(ws, {"quality_gate": {"mode": "strict"}})
        _seed_signal(ws, _PRIOR, day=dt.datetime.now(dt.timezone.utc).date())

        out = self._propose(ws, _DISTINCT)
        assert out.get("status") == "proposed", out

    def test_no_prior_signals_means_the_rule_is_skipped_not_passed(self, ws: str) -> None:
        """An empty window is honest about not having run the rule."""
        from mind_mem.mcp.tools.governance import _recent_statements
        from mind_mem.quality_gate import validate_block

        verdict = validate_block(_NEAR_DUP, recent=_recent_statements(ws))
        assert "near_duplicate" in verdict.skipped_rules
        assert "near_duplicate" not in verdict.checked_rules


# ---------------------------------------------------------------------------
# The preview: mcp.tools.quality.validate_block
# ---------------------------------------------------------------------------


class TestPreviewRunsRule6:
    def test_preview_reports_the_duplicate_the_enforcer_would_reject(self, ws: str) -> None:
        """Fails pre-fix: the preview reported near_duplicate as skipped."""
        import mind_mem.mcp.tools.quality as quality

        _write_config(ws, {"quality_gate": {"mode": "strict"}})
        _seed_signal(ws, _PRIOR, day=dt.datetime.now(dt.timezone.utc).date())

        with use_workspace(ws):
            payload = json.loads(quality.validate_block(_NEAR_DUP))

        assert payload["accept"] is False, payload
        assert "near_duplicate" in payload["checked_rules"], payload
        assert "near_duplicate" not in payload["skipped_rules"], payload
        assert any(r.startswith("near_duplicate") for r in payload["reasons"]), payload

    def test_preview_and_enforcer_agree_on_the_same_statement(self, ws: str) -> None:
        import mind_mem.mcp.tools.governance as gov
        import mind_mem.mcp.tools.quality as quality

        _write_config(ws, {"quality_gate": {"mode": "strict"}})
        _seed_signal(ws, _PRIOR, day=dt.datetime.now(dt.timezone.utc).date())

        with use_workspace(ws):
            preview = json.loads(quality.validate_block(_NEAR_DUP))
            enforced = json.loads(
                gov.propose_update(
                    block_type="decision",
                    statement=_NEAR_DUP,
                    rationale="rationale long enough to pass the gate",
                )
            )

        assert preview["accept"] is False
        assert enforced.get("error") == "quality_gate_rejection"
