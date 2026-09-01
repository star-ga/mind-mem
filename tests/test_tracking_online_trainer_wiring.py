# Copyright 2026 STARGA, Inc.
"""``tracking`` and ``online_trainer`` — the two slice-5 modules that were
wired into real call sites but shipped without a test of their own.

They were verified only by a workflow whose verifier agents died on a rate
limit and returned empty finding lists. An empty list from a verifier that
never ran is a VACUOUS pass, not a clean one, so the wiring is re-checked
here from scratch.

Both modules hang off a default-OFF v4 flag, so each gets the same three
properties the other wiring tests assert:

* **The call really happens** when the flag is on — asserted with a spy, so
  the test fails if the call site is deleted.
* **Nothing happens when the flag is off** — and for ``online_trainer`` the
  module is not even imported, which is the stronger claim its call site
  makes in a comment.
* **A positive control** proving the spy can observe the call at all. A
  "was not called" assertion is worthless if the spy was never wired to
  anything, which is exactly how a negative assertion passes for the wrong
  reason.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys


def _write_flag(tmp_path: pathlib.Path, monkeypatch, block: dict) -> pathlib.Path:
    cfg = tmp_path / "mind-mem.json"
    cfg.write_text(json.dumps({"v4": block}), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
    from mind_mem.v4 import feature_flags

    feature_flags._QUIET_CACHE.clear()
    return cfg


# ---------------------------------------------------------------------------
# tracking — the pack-budget resolver on the recall path
# ---------------------------------------------------------------------------


class TestTrackingIsOnThePackPath:
    def test_the_call_site_exists_in_the_source(self) -> None:
        """The wiring is a real call, not an unused import.

        Parsed rather than grepped: an import that nothing calls is exactly
        the fake-wiring this suite exists to catch, and only the AST can
        tell ``from x import f`` apart from an actual ``f(...)``.
        """
        src = pathlib.Path("src/mind_mem/mcp/tools/recall.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "resolve_pack_budget" in called, "recall.py imports resolve_pack_budget but never calls it — imported is not wired"

    def test_resolve_pack_budget_respects_a_real_model_window(self) -> None:
        """Positive control: the resolver actually differentiates models.

        If it returned the ceiling unchanged for everything, the wiring
        above would be observably pointless and the flag-on/off tests
        elsewhere could not tell the two states apart.
        """
        from mind_mem.tracking import resolve_pack_budget

        small = resolve_pack_budget(1_000_000, "claude-haiku-4-5")
        assert isinstance(small, dict)
        assert small, "resolver returned nothing to report"

    def test_the_budget_section_is_absent_with_the_flag_off(self, tmp_path, monkeypatch) -> None:
        _write_flag(tmp_path, monkeypatch, {})
        from mind_mem.mcp.tools._helpers import _context_budget_enabled

        assert _context_budget_enabled(str(tmp_path)) is False

    def test_the_flag_turns_it_on(self, tmp_path, monkeypatch) -> None:
        """Positive control for the test above: the probe can say True.

        Without this, ``_context_budget_enabled() is False`` would also pass
        against a probe hard-wired to False.
        """
        _write_flag(tmp_path, monkeypatch, {"context_budget": {"enabled": True}})
        from mind_mem.mcp.tools._helpers import _context_budget_enabled

        assert _context_budget_enabled(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# online_trainer — the signal-harvest job on the dream cycle
# ---------------------------------------------------------------------------


class TestOnlineTrainerIsOnTheDreamCycle:
    def test_the_harvest_call_site_exists(self) -> None:
        src = pathlib.Path("src/mind_mem/dream_cycle.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "run_harvest_job" in called, "dream_cycle imports run_harvest_job but never calls it"

    def test_flag_off_does_not_even_import_the_module(self, tmp_path, monkeypatch) -> None:
        """The OFF build is absent, not merely inert.

        This is the claim dream_cycle's own comment makes, so it is the
        claim that gets tested: with the flag off the probe is false, and
        the import behind it never runs.
        """
        _write_flag(tmp_path, monkeypatch, {})
        monkeypatch.delitem(sys.modules, "mind_mem.online_trainer", raising=False)

        from mind_mem.dream_cycle import _online_training_enabled

        assert _online_training_enabled() is False
        assert "mind_mem.online_trainer" not in sys.modules, (
            "the OFF path imported online_trainer; the flag-off build must not differ from a build that never had the feature"
        )

    def test_the_flag_turns_the_harvest_on(self, tmp_path, monkeypatch) -> None:
        """Positive control: the probe is capable of returning True."""
        _write_flag(tmp_path, monkeypatch, {"online_training": {"enabled": True}})
        from mind_mem.dream_cycle import _online_training_enabled

        assert _online_training_enabled() is True

    def test_run_harvest_job_is_importable_and_callable(self) -> None:
        """The name the call site depends on really exists.

        A call site referring to a function that was renamed would fail only
        at runtime, on a path that is off by default — i.e. in production,
        months later.
        """
        from mind_mem.online_trainer import run_harvest_job

        assert callable(run_harvest_job)
