# Copyright 2026 STARGA, Inc.
"""Regression tests for FleetBridge target selection.

The bug these pin down: `query()` selected its targets with
`models or list(self._providers.keys())`. An explicitly empty target list is
falsy, so "query nobody" silently widened to "query everybody". Reached from
`query_excluding()`, that inverted the cross-model critique contract -- when the
exclude set covered the whole fleet, the models that produced the output were
handed their own output to grade.
"""

from __future__ import annotations

import asyncio
import types
from collections.abc import Callable
from typing import Any

import pytest

from mind_mem.skill_opt import fleet_bridge as fb

pytestmark = pytest.mark.unit


class _FakeStatus:
    def __init__(self, value: str = "ok") -> None:
        self.value = value


class _FakeResult:
    def __init__(self, content: str) -> None:
        self.content = content
        self.status = _FakeStatus("ok")
        self.error = ""


class _FakeProvider:
    """Records every prompt it is asked to answer."""

    def __init__(self, api_key: str, model: str, rate_config: Any, timeout_s: float) -> None:
        self.model = model
        self.calls: list[str] = []

    async def request(self, prompt: str) -> _FakeResult:
        self.calls.append(prompt)
        return _FakeResult(f"reply from {self.model}")


class _FakeRateLimitConfig:
    def __init__(self, max_concurrent: int, min_request_spacing_s: float) -> None:
        self.max_concurrent = max_concurrent
        self.min_request_spacing_s = min_request_spacing_s


# The two seats the audit reproduced against: a box where only two provider
# keys resolve, so the test executors ARE the whole fleet.
_TWO_SEATS = ["grok-4.3", "mistral-large-latest"]


@pytest.fixture()
def make_bridge(monkeypatch: pytest.MonkeyPatch) -> Callable[[], fb.FleetBridge]:
    """Builds cold FleetBridges whose real `_ensure_init` makes recording providers."""
    providers_mod = types.SimpleNamespace(
        RateLimitConfig=_FakeRateLimitConfig,
        XAIProvider=_FakeProvider,
        MistralProvider=_FakeProvider,
    )
    keys = {"xai": "test-key", "mistral": "test-key"}
    monkeypatch.setattr(fb, "_load_orchestrator", lambda: (providers_mod, keys))
    return lambda: fb.FleetBridge(models=list(_TWO_SEATS))


@pytest.fixture()
def bridge(make_bridge: Callable[[], fb.FleetBridge]) -> fb.FleetBridge:
    return make_bridge()


def _called_models(bridge: fb.FleetBridge) -> set[str]:
    """Which providers actually had `request()` invoked on them."""
    return {key for key, provider in bridge._providers.items() if provider.calls}


def test_excluding_whole_fleet_queries_nobody(bridge: fb.FleetBridge) -> None:
    """Excluding every model must return nothing, not fall back to everything.

    Old code returned two responses -- one from each excluded model.
    """
    responses = asyncio.run(bridge.query_excluding("critique this", exclude=set(_TWO_SEATS)))

    assert responses == []
    # Anti-vacuity: both providers really were built, so an empty call set is a
    # deliberate skip, not an empty fleet.
    assert set(bridge._providers) == set(_TWO_SEATS)
    # Stronger than checking the return value: the excluded providers were
    # never even asked, so no self-critique was generated or paid for.
    assert _called_models(bridge) == set()


@pytest.mark.parametrize("exclude", [{"grok-4.3"}, {"mistral-large-latest"}, set(_TWO_SEATS)])
def test_excluded_model_never_critiques_itself(make_bridge: Callable[[], fb.FleetBridge], exclude: set[str]) -> None:
    """No response may ever come back from a model in the exclude set."""
    b = make_bridge()

    responses = asyncio.run(b.query_excluding("critique this", exclude=exclude))

    assert {r.model for r in responses}.isdisjoint(exclude), f"self-critique leaked for exclude={exclude}"
    assert _called_models(b).isdisjoint(exclude), f"excluded provider was invoked for exclude={exclude}"
    # Not vacuous: the non-excluded seats really were queried.
    assert _called_models(b) == set(_TWO_SEATS) - exclude


def test_excluding_one_model_still_queries_the_survivor(bridge: fb.FleetBridge) -> None:
    """Guard against over-fixing: a partial exclusion must still query the rest."""
    responses = asyncio.run(bridge.query_excluding("critique this", exclude={"grok-4.3"}))

    assert [r.model for r in responses] == ["mistral-large-latest"]
    assert _called_models(bridge) == {"mistral-large-latest"}


def test_query_with_none_models_queries_whole_fleet(bridge: fb.FleetBridge) -> None:
    """`models=None` keeps meaning "every available model"."""
    responses = asyncio.run(bridge.query("hello", models=None))

    assert {r.model for r in responses} == set(_TWO_SEATS)
    assert all(r.ok for r in responses)


def test_query_with_empty_model_list_queries_nothing(bridge: fb.FleetBridge) -> None:
    """The unit of the fix: an empty list is a real target set, not "unset"."""
    responses = asyncio.run(bridge.query("hello", models=[]))

    assert responses == []
    # Anti-vacuity: the fleet was available and simply not asked.
    assert set(bridge._providers) == set(_TWO_SEATS)
    assert _called_models(bridge) == set()


def test_query_excluding_initializes_providers_first(bridge: fb.FleetBridge) -> None:
    """`query_excluding` reads `_providers`, so it must initialize them itself.

    Without its own `_ensure_init`, the target list would be built from an
    empty provider dict on a cold bridge and the exclusion fix would turn
    "query everyone" into "query no one".
    """
    assert bridge._providers == {}, "bridge must still be cold for this probe"

    responses = asyncio.run(bridge.query_excluding("hello", exclude=set()))

    assert {r.model for r in responses} == set(_TWO_SEATS)


# ---------------------------------------------------------------------------
# Dropped seats are recorded, not swallowed
# ---------------------------------------------------------------------------


def test_dropped_seats_are_recorded_with_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of the three drop paths must be accounted for by name.

    ``_ensure_init`` dropped requested models at three bare ``continue``s
    with no counter, log or returned diff, and no caller could compare
    ``available_models`` against what it asked for — so a run that
    contacted zero models was indistinguishable from one that measured a
    genuinely worthless skill.
    """
    providers_mod = types.SimpleNamespace(
        RateLimitConfig=_FakeRateLimitConfig,
        XAIProvider=_FakeProvider,
        # MistralProvider deliberately absent -> "provider class missing"
    )
    # mistral HAS a key, so it reaches (and fails) the provider-class check;
    # zhipu has none, so it stops at the key check. Two distinct paths.
    keys = {"xai": "test-key", "mistral": "test-key"}
    monkeypatch.setattr(fb, "_load_orchestrator", lambda: (providers_mod, keys))

    b = fb.FleetBridge(models=["grok-4.3", "mistral-large-latest", "glm-5.1", "not-a-real-model"])

    assert b.available_models == ["grok-4.3"]
    unavailable = b.unavailable_models
    assert set(unavailable) == {"mistral-large-latest", "glm-5.1", "not-a-real-model"}
    assert "not in FLEET_MODELS" in unavailable["not-a-real-model"]
    assert "no API key" in unavailable["glm-5.1"]
    assert "MistralProvider" in unavailable["mistral-large-latest"]
    # Together the two properties account for every requested model.
    assert set(b.available_models) | set(unavailable) == set(b._requested_models)


def test_every_seat_dropped_is_visible_not_an_empty_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero contactable seats must be readable as such by the caller."""
    providers_mod = types.SimpleNamespace(RateLimitConfig=_FakeRateLimitConfig)
    monkeypatch.setattr(fb, "_load_orchestrator", lambda: (providers_mod, {}))

    b = fb.FleetBridge(models=list(_TWO_SEATS))

    assert b.available_models == []
    assert len(b.unavailable_models) == len(_TWO_SEATS)
    assert all("no API key" in reason for reason in b.unavailable_models.values())


def test_fully_available_fleet_records_no_drops(bridge: fb.FleetBridge) -> None:
    """Anti-vacuity: a healthy fleet reports an empty unavailable map."""
    assert set(bridge.available_models) == set(_TWO_SEATS)
    assert bridge.unavailable_models == {}
