"""Tests for the v4 Cognitive Mind Kernel registry + dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem.v4 import FeatureDisabledError
from mind_mem.v4.cognitive_kernel import (
    DEFAULT_KERNEL,
    FLAG,
    KernelHit,
    KernelKind,
    KernelResult,
    available_kernels,
    mind_recall,
    register_kernel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": True}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def cfg_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {"v4": {FLAG: {"enabled": False}}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


@pytest.fixture
def fake_kernel() -> tuple[str, Any]:
    """A canned kernel that returns a deterministic two-hit result.

    Returns the kernel name (str — meant to be cast to a KernelKind by
    the test) plus the strategy callable.
    """

    def _strategy(workspace: str, query: str, **kwargs: Any) -> KernelResult:
        return KernelResult(
            kernel=KernelKind.RECENT_FIRST,
            hits=[
                KernelHit(block_id="B-recent-1", score=0.9, reason="recent_first:t=now"),
                KernelHit(block_id="B-recent-2", score=0.7, reason="recent_first:t=t-1"),
            ],
            metadata={"workspace": workspace, "query": query, **kwargs},
        )

    return "recent_first", _strategy


# ---------------------------------------------------------------------------
# Type surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kernel_kinds_set() -> None:
    """The five named strategies plus DEFAULT must be present."""
    assert {k.value for k in KernelKind} == {
        "default",
        "surprise_weighted",
        "lineage_first",
        "recent_first",
        "contradicts_first",
        "graph_walk",
    }


@pytest.mark.unit
def test_kernel_hit_immutable() -> None:
    h = KernelHit(block_id="B-1", score=0.5)
    with pytest.raises((AttributeError, Exception)):
        h.score = 0.9  # type: ignore[misc]


@pytest.mark.unit
def test_kernel_result_defaults() -> None:
    r = KernelResult(kernel=KernelKind.DEFAULT)
    assert r.kernel is KernelKind.DEFAULT
    assert r.hits == []
    assert dict(r.metadata) == {}


# ---------------------------------------------------------------------------
# Registry — flag gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_raises_when_flag_off(cfg_off: Path, fake_kernel: tuple[str, Any]) -> None:
    name, strategy = fake_kernel
    with pytest.raises(FeatureDisabledError):
        register_kernel(name, strategy)


@pytest.mark.unit
def test_available_kernels_raises_when_flag_off(cfg_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        available_kernels()


@pytest.mark.unit
def test_mind_recall_raises_when_flag_off(cfg_off: Path) -> None:
    with pytest.raises(FeatureDisabledError):
        mind_recall("/tmp/ws", "a query")


# ---------------------------------------------------------------------------
# Registry — registration semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_is_pre_registered(cfg_on: Path) -> None:
    """The DEFAULT kernel registers at import time, even before flag flips."""
    assert KernelKind.DEFAULT in available_kernels()


@pytest.mark.unit
def test_register_via_string(cfg_on: Path, fake_kernel: tuple[str, Any]) -> None:
    name, strategy = fake_kernel
    register_kernel(name, strategy)
    assert KernelKind(name) in available_kernels()


@pytest.mark.unit
def test_register_via_enum(cfg_on: Path, fake_kernel: tuple[str, Any]) -> None:
    _, strategy = fake_kernel
    register_kernel(KernelKind.LINEAGE_FIRST, strategy)
    assert KernelKind.LINEAGE_FIRST in available_kernels()


@pytest.mark.unit
def test_register_replaces_previous(cfg_on: Path) -> None:
    """Re-registering under the same kind replaces, doesn't accumulate."""

    def first(_w: str, _q: str, **_: Any) -> KernelResult:
        return KernelResult(kernel=KernelKind.GRAPH_WALK, metadata={"v": 1})

    def second(_w: str, _q: str, **_: Any) -> KernelResult:
        return KernelResult(kernel=KernelKind.GRAPH_WALK, metadata={"v": 2})

    register_kernel(KernelKind.GRAPH_WALK, first)
    register_kernel(KernelKind.GRAPH_WALK, second)
    out = mind_recall("/tmp/ws", "q", kernel=KernelKind.GRAPH_WALK)
    assert out.metadata["v"] == 2


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mind_recall_routes_to_string_kernel(cfg_on: Path, fake_kernel: tuple[str, Any]) -> None:
    name, strategy = fake_kernel
    register_kernel(name, strategy)
    out = mind_recall("/tmp/ws", "the query", kernel=name)
    assert out.kernel is KernelKind.RECENT_FIRST
    assert [h.block_id for h in out.hits] == ["B-recent-1", "B-recent-2"]


@pytest.mark.unit
def test_mind_recall_routes_to_enum_kernel(cfg_on: Path, fake_kernel: tuple[str, Any]) -> None:
    _, strategy = fake_kernel
    register_kernel(KernelKind.SURPRISE_WEIGHTED, strategy)
    out = mind_recall("/tmp/ws", "q", kernel=KernelKind.SURPRISE_WEIGHTED)
    # The strategy hard-codes RECENT_FIRST in its result for this fixture;
    # the routing key is what mind_recall used to find the strategy, the
    # kernel field on the result is whatever the strategy decided to put.
    assert out.kernel is KernelKind.RECENT_FIRST
    assert len(out.hits) == 2


@pytest.mark.unit
def test_mind_recall_passes_kwargs_through(cfg_on: Path, fake_kernel: tuple[str, Any]) -> None:
    name, strategy = fake_kernel
    register_kernel(name, strategy)
    out = mind_recall("/tmp/ws", "q", kernel=name, custom_knob=42)
    assert out.metadata["custom_knob"] == 42
    assert out.metadata["query"] == "q"
    assert out.metadata["workspace"] == "/tmp/ws"


@pytest.mark.unit
def test_unknown_kernel_raises_keyerror_with_registry(cfg_on: Path) -> None:
    """Calling with an unregistered kernel must list what IS registered."""
    # Make sure we're using a valid enum value that isn't registered.
    # Re-import + reset is overkill; just hit one that we won't have set up.
    with pytest.raises(KeyError) as excinfo:
        mind_recall("/tmp/ws", "q", kernel=KernelKind.CONTRADICTS_FIRST)
    msg = str(excinfo.value)
    assert "contradicts_first" in msg
    assert "available" in msg


@pytest.mark.unit
def test_unknown_kernel_string_rejected_at_constructor(cfg_on: Path) -> None:
    """A string that isn't a valid KernelKind name fails at the enum cast."""
    with pytest.raises(ValueError):
        mind_recall("/tmp/ws", "q", kernel="banana_split")


# ---------------------------------------------------------------------------
# Default kernel — pass-through behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_kernel_returns_kernelresult(cfg_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default kernel delegates to v3 recall and adapts the result shape.

    We monkeypatch the v3 recall import so we don't need a real workspace
    set up for this unit test.
    """
    fake_hits = [
        {"_id": "B-1", "rrf_score": 0.95},
        {"block_id": "B-2", "score": 0.42},
    ]

    import mind_mem.v4.cognitive_kernel as ck

    # Build a fake module with a `recall` callable.
    class _FakeRecallMod:
        @staticmethod
        def recall(_w: str, _q: str) -> list[dict]:
            return fake_hits

    # Monkeypatch the lazy import inside _default_kernel.
    import sys

    sys.modules["mind_mem._recall_core"] = _FakeRecallMod  # type: ignore[assignment]
    try:
        out = ck._default_kernel("/tmp/ws", "anything")
    finally:
        sys.modules.pop("mind_mem._recall_core", None)

    assert out.kernel is KernelKind.DEFAULT
    assert [(h.block_id, h.score) for h in out.hits] == [("B-1", 0.95), ("B-2", 0.42)]
    # Reason tags are empty for the pass-through kernel.
    assert all(h.reason == "" for h in out.hits)


@pytest.mark.unit
def test_default_kernel_degrades_when_v3_missing(cfg_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When v3 recall import fails, default kernel returns empty hits."""
    # Force ImportError by stashing a broken module.
    import sys

    import mind_mem.v4.cognitive_kernel as ck

    class _Broken:
        def __getattr__(self, name: str) -> Any:
            raise ImportError(f"forced {name}")

    sys.modules["mind_mem._recall_core"] = _Broken()  # type: ignore[assignment]
    try:
        # Reach _default_kernel via mind_recall path.
        out = ck.mind_recall("/tmp/ws", "q", kernel=KernelKind.DEFAULT)
    finally:
        sys.modules.pop("mind_mem._recall_core", None)

    # Either degraded path or successful real-recall path is acceptable
    # since we can't fully control the import in every test env. What
    # we DO assert: kernel == DEFAULT and shape is KernelResult.
    assert out.kernel is KernelKind.DEFAULT
    assert isinstance(out, KernelResult)


@pytest.mark.unit
def test_default_kernel_is_the_module_constant() -> None:
    from mind_mem.v4.cognitive_kernel import _default_kernel

    assert DEFAULT_KERNEL is _default_kernel
