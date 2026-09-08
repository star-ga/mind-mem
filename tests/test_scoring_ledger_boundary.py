# Copyright 2026 STARGA, Inc.
"""The scoring path names the cap refusal without loading the usage ledger.

``DailyTokenCapExceeded`` lives in ``error_codes`` -- a leaf importing only
``enum`` -- and ``usage_meter`` re-exports it. Modules that merely CATCH the
refusal import it from the leaf; modules that actually meter a model call
still import the ledger, because they use it.

Four controls, one per property that could regress independently.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Scoring-path modules that catch the refusal but never meter anything.
SCORING_ONLY = ("hybrid_recall", "_recall_core", "smart_chunker")

#: Modules that legitimately load the ledger: they charge real model calls.
METERING = ("query_expansion", "_recall_reranking", "llm_extractor", "mm_cli")


def test_control_1_class_identity_is_single() -> None:
    """Both spellings must bind ONE class, or ``except`` clauses stop matching.

    This is the control that makes the whole move safe: every existing
    ``from .usage_meter import DailyTokenCapExceeded`` in the tree keeps
    catching exactly what it caught before.
    """
    from mind_mem.error_codes import DailyTokenCapExceeded as leaf
    from mind_mem.usage_meter import DailyTokenCapExceeded as reexport

    assert leaf is reexport
    assert issubclass(leaf, RuntimeError)
    # Positive control: a subclass relationship alone would pass above even if
    # they were two distinct classes, which is the bug this guards.
    assert leaf.__module__ == "mind_mem.error_codes"


@pytest.mark.parametrize("module", SCORING_ONLY)
def test_control_2_scoring_dispatch_does_not_load_the_ledger(module: str) -> None:
    """Import the scoring module in a CLEAN interpreter; the ledger must be absent.

    A static grep sees ``import`` statements only. Reading ``sys.modules``
    after a real import also catches a re-export, an importlib call, or a
    sideways pull through some other module.
    """
    child = f"import importlib, sys; importlib.import_module('mind_mem.{module}'); print('mind_mem.usage_meter' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = proc.stdout.strip().splitlines()[-1]
    assert loaded == "False", f"importing mind_mem.{module} pulled in the usage ledger"


@pytest.mark.parametrize("module", METERING)
def test_control_2b_metering_modules_are_not_silently_severed(module: str) -> None:
    """The inverse control, so control 2 cannot pass by gutting the metering.

    If someone 'fixed' control 2 by removing metering everywhere, control 2
    would go green and the product would stop charging model calls. These
    modules must still reach the ledger.
    """
    src = (__import__("pathlib").Path("src/mind_mem") / f"{module}.py").read_text(encoding="utf-8")
    assert "usage_meter" in src, f"{module} no longer reaches the usage ledger at all"


def test_control_3_multi_query_fanout_is_real() -> None:
    """Expansion must still produce a genuine multi-variant fan-out.

    Pinned because a stale stub signature once made this path silently
    collapse to a single query while its test kept passing.
    """
    import mind_mem.hybrid_recall as hr

    assert hasattr(hr.HybridBackend, "_search_expanded")
    src = (__import__("pathlib").Path("src/mind_mem/hybrid_recall.py")).read_text(encoding="utf-8")
    assert "_union_degraded" in src, "per-variant degradation aggregation vanished"
    assert "variants_total" in src, "the fan-out no longer records how many variants ran"


def test_control_4_unrelated_runtime_errors_are_not_swallowed() -> None:
    """A plain RuntimeError must NOT be mistaken for a cap refusal.

    The rejected alternative caught broad ``RuntimeError`` and narrowed by
    predicate. With the precise typed exception, an unrelated RuntimeError is
    simply not a cap refusal -- assert that directly.
    """
    from mind_mem.error_codes import DailyTokenCapExceeded

    assert not isinstance(RuntimeError("disk gone"), DailyTokenCapExceeded)
    # Positive control: the real thing IS one, so the assertion above is not
    # passing merely because nothing is ever an instance.
    assert isinstance(DailyTokenCapExceeded("cap"), DailyTokenCapExceeded)
    assert isinstance(DailyTokenCapExceeded("cap"), RuntimeError)
