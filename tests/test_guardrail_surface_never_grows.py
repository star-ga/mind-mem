"""Guardrail surfacing must never return more hits than it was given.

``apply_guardrail_surfacing`` sized its output with
``max(len(hits), len(surfaced))``. That was written for the documented empty-page
exception — a constraint must fire even when recall found nothing — but it also
covered every page SHORTER than the number of firing guardrails, which is not an
exception the docstring makes. With the default bound of 3 and three firing
guardrails, ``recall(limit=1)`` came back with three hits: the surfacing step
runs after the pipeline's own ``[:limit]`` truncations, so the caller's page
size and any downstream max-results budget were both overrun.
"""

from __future__ import annotations

from typing import Any

import pytest

from mind_mem.guardrail_surface import apply_guardrail_surfacing
from mind_mem.guardrails import Guardrail, GuardrailContext, GuardrailPolicy, GuardrailTrigger

WS = "/nonexistent-workspace"
CTX = GuardrailContext(tool="Bash")


def _guardrail(bid: str) -> Guardrail:
    return Guardrail(
        block_id=bid,
        statement=f"never do {bid}",
        severity="high",
        # Trigger patterns are stored already normalised by the parser; the
        # matcher normalises only the CONTEXT side, so an unnormalised pattern
        # here would silently never fire and every assertion below would pass
        # vacuously on an untouched hit list. Measured: ``("Bash",)`` matched
        # nothing.
        trigger=GuardrailTrigger(tools=("bash",)),
        source_file="guardrails/GUARDRAILS.md",
        line=1,
        status="active",
        block={"_id": bid, "Statement": f"never do {bid}", "Status": "active"},
    )


@pytest.fixture()
def three_firing(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Three guardrails that fire for CTX, with no workspace on disk.

    The loader is stubbed rather than a corpus written, so the test isolates the
    budget arithmetic from guardrail discovery — the thing under test is how the
    result is SIZED, not which guardrails were found.
    """
    import mind_mem.guardrail_surface as gs

    rails = [_guardrail("GR-20260827-10%d" % n) for n in (1, 2, 3)]
    monkeypatch.setattr(gs, "load_guardrails", lambda _ws, _policy: rails)
    # Prove the fixture actually fires before any test leans on it. A stub that
    # matches nothing turns every length assertion below into a check on an
    # untouched list — the exact way a test passes without exercising anything.
    assert len(gs.guardrail_hits(WS, CTX, GuardrailPolicy())) == 3
    return rails


def _hits(n: int) -> list[dict]:
    return [{"_id": f"D-2026082{i}-001", "score": 1.0 - i / 10} for i in range(n)]


@pytest.mark.parametrize("page", [1, 2, 3, 4, 5])
def test_result_never_exceeds_the_page_it_was_given(three_firing: Any, page: int) -> None:
    hits = _hits(page)
    out = apply_guardrail_surfacing(hits, workspace=WS, context=CTX, policy=GuardrailPolicy())
    assert len(out) == page


def test_a_single_hit_page_is_not_tripled(three_firing: Any) -> None:
    """The concrete repro: three guardrails, ``limit=1``, one hit out."""
    out = apply_guardrail_surfacing(_hits(1), workspace=WS, context=CTX, policy=GuardrailPolicy())
    assert len(out) == 1
    assert out[0]["guardrail"] is True
    assert out[0]["_id"] == "GR-20260827-101", "the most severe / lowest id survives the truncation"


def test_an_empty_page_still_returns_the_constraints(three_firing: Any) -> None:
    """The one documented exception: nothing found is not nothing to say."""
    out = apply_guardrail_surfacing([], workspace=WS, context=CTX, policy=GuardrailPolicy())
    assert [h["_id"] for h in out] == ["GR-20260827-101", "GR-20260827-102", "GR-20260827-103"]


def test_a_roomy_page_keeps_its_length_and_its_survivors(three_firing: Any) -> None:
    """Regression guard on the normal case the old arithmetic already got right."""
    hits = _hits(6)
    out = apply_guardrail_surfacing(hits, workspace=WS, context=CTX, policy=GuardrailPolicy())
    assert len(out) == 6
    assert [h["_id"] for h in out[:3]] == ["GR-20260827-101", "GR-20260827-102", "GR-20260827-103"]
    # Surviving ranked hits keep their relative order, oldest-ranked first.
    assert [h["_id"] for h in out[3:]] == [h["_id"] for h in hits[:3]]


def test_displaced_count_is_never_more_than_the_page_held(three_firing: Any, caplog: Any) -> None:
    """The log line has to survive a budget smaller than the guardrail head.

    ``budget - len(surfaced)`` goes negative on a short page; unclamped it
    reports more displaced hits than the response ever contained.
    """
    import mind_mem.guardrail_surface as gs

    recorded: list[dict] = []
    monkey = gs._log.info
    try:
        gs._log.info = lambda event, **kw: recorded.append({"event": event, **kw})  # type: ignore[method-assign]
        apply_guardrail_surfacing(_hits(1), workspace=WS, context=CTX, policy=GuardrailPolicy())
    finally:
        gs._log.info = monkey  # type: ignore[method-assign]

    assert recorded, "the surfacing step must report what it did"
    assert recorded[0]["displaced"] == 1
