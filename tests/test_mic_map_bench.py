"""pytest-benchmark suite for ``mind_mem.mic_map``.

Establishes the v3.8.8 throughput floor for the serialization
primitive. Three graph sizes — small (residual block, 7 values),
medium (a transformer-like layer with ~30 values), and large
(deep stack with 200 values) — exercised through both formats.

Only the four ``*Bench`` classes need ``pytest-benchmark`` (declared
in the ``[benchmark]`` extra), so only those are gated on it. Run
them with::

    pytest tests/test_mic_map_bench.py --benchmark-only

Minimum-throughput assertions live in ``TestThroughputFloors`` —
the floors are deliberately conservative so a 2-3× speedup from a
future Cython port is visible. Floors apply to single-core
operation; they bound the *worst* acceptable throughput, not the
*expected* throughput.

``TestThroughputFloors`` and ``TestMemoryCeiling`` use no plugin at
all — ``time.perf_counter`` and ``len()`` — and are NOT gated. The
gate used to be a module-scope ``pytest.importorskip`` that took
them with it, and the result was that all 20 tests in this file ran
nowhere: no CI row installs ``[benchmark]``, and the one workflow
that does selects tests with ``-k "benchmark or perf"``, which
matches none of the ids here (measured: "no tests collected (20
deselected)"). The eight throughput/size tests below had therefore
never executed in CI while the run reported green.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Gate ONLY what actually needs the plugin: the ``benchmark`` fixture. Keeping
# this off module scope is what lets the throughput floors and the memory
# ceiling below run on every matrix row.
requires_pytest_benchmark = pytest.mark.skipif(
    importlib.util.find_spec("pytest_benchmark") is None,
    reason="pytest-benchmark not installed (declared in the [benchmark] extra)",
)

from mind_mem.mic_map import (  # noqa: E402
    Arg,
    Graph,
    Node,
    Param,
    Type,
    emit_mic2,
    emit_micb,
    parse_mic2,
    parse_micb,
)

# ---------------------------------------------------------------------------
# Fixtures — three reference graphs spanning the size range
# ---------------------------------------------------------------------------


def _residual_block() -> Graph:
    return Graph(
        types=[
            Type(dtype="f16", dims=("128", "128")),
            Type(dtype="f16", dims=("128",)),
        ],
        values=[
            Arg(name="X", type_idx=0),
            Param(name="W", type_idx=0),
            Param(name="b", type_idx=1),
            Node(opcode="m", inputs=(0, 1)),
            Node(opcode="+", inputs=(3, 2)),
            Node(opcode="r", inputs=(4,)),
            Node(opcode="+", inputs=(5, 0)),
        ],
        output=6,
    )


def _transformer_layer() -> Graph:
    """Approximate one transformer block: attention + MLP, ~30 values.
    Synthetic but representative — every opcode used in the spec is
    exercised at least once."""
    types = [
        Type(dtype="f16", dims=("B", "S", "D")),
        Type(dtype="f16", dims=("D", "D")),
        Type(dtype="f16", dims=("D",)),
    ]
    values: list = []
    # Inputs / params
    values.append(Arg(name="x", type_idx=0))
    for nm in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wln1", "Wln2"):
        values.append(Param(name=nm, type_idx=1))
    for nm in ("bq", "bk", "bv", "bo", "b1", "b2", "bln1", "bln2"):
        values.append(Param(name=nm, type_idx=2))
    # x_id = 0; Wq..Wln2 = 1..8; bq..bln2 = 9..16
    # Pre-norm
    values.append(Node(opcode="ln", inputs=(0,)))  # 17
    # Q, K, V
    values.append(Node(opcode="m", inputs=(17, 1)))  # 18 = q
    values.append(Node(opcode="m", inputs=(17, 2)))  # 19 = k
    values.append(Node(opcode="m", inputs=(17, 3)))  # 20 = v
    # Attention scores + softmax (axis=-1)
    values.append(Node(opcode="m", inputs=(18, 19)))  # 21
    values.append(Node(opcode="s", inputs=(21,), op_params=(-1,)))  # 22
    values.append(Node(opcode="m", inputs=(22, 20)))  # 23
    values.append(Node(opcode="m", inputs=(23, 4)))  # 24 = attention out
    # Residual + post-norm
    values.append(Node(opcode="+", inputs=(0, 24)))  # 25
    values.append(Node(opcode="ln", inputs=(25,)))  # 26
    # MLP
    values.append(Node(opcode="m", inputs=(26, 5)))  # 27
    values.append(Node(opcode="gelu", inputs=(27,)))  # 28
    values.append(Node(opcode="m", inputs=(28, 6)))  # 29
    # Final residual
    values.append(Node(opcode="+", inputs=(25, 29)))  # 30
    return Graph(types=types, values=values, output=30)


def _deep_stack(n_layers: int = 200) -> Graph:
    """Synthetic deep relu stack: y = relu(relu(...(W·x + b)...))
    with ``n_layers`` of stacked matmul + add + relu. Exercises the
    serializer / parser at the value-table-size limit."""
    types = [
        Type(dtype="f32", dims=("B", "D")),
        Type(dtype="f32", dims=("D", "D")),
        Type(dtype="f32", dims=("D",)),
    ]
    values: list = [Arg(name="x", type_idx=0)]
    for i in range(n_layers):
        values.append(Param(name=f"W{i}", type_idx=1))
        values.append(Param(name=f"b{i}", type_idx=2))
    # n_layers param-pairs after x_id=0; param IDs 1..2*n_layers.
    cur = 0
    for i in range(n_layers):
        w_id = 1 + 2 * i
        b_id = 2 + 2 * i
        mm_id = len(values)
        values.append(Node(opcode="m", inputs=(cur, w_id)))
        add_id = len(values)
        values.append(Node(opcode="+", inputs=(mm_id, b_id)))
        relu_id = len(values)
        values.append(Node(opcode="r", inputs=(add_id,)))
        cur = relu_id
    return Graph(types=types, values=values, output=cur)


@pytest.fixture(scope="module")
def small_graph() -> Graph:
    return _residual_block()


@pytest.fixture(scope="module")
def medium_graph() -> Graph:
    return _transformer_layer()


@pytest.fixture(scope="module")
def large_graph() -> Graph:
    return _deep_stack(200)


# ---------------------------------------------------------------------------
# Benchmarks — emit + parse, both formats, all three sizes
# ---------------------------------------------------------------------------


@requires_pytest_benchmark
class TestEmitMic2Bench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        benchmark(emit_mic2, small_graph)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        benchmark(emit_mic2, medium_graph)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        benchmark(emit_mic2, large_graph)


@requires_pytest_benchmark
class TestEmitMicbBench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        benchmark(emit_micb, small_graph)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        benchmark(emit_micb, medium_graph)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        benchmark(emit_micb, large_graph)


@requires_pytest_benchmark
class TestParseMic2Bench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        text = emit_mic2(small_graph)
        benchmark(parse_mic2, text)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        text = emit_mic2(medium_graph)
        benchmark(parse_mic2, text)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        text = emit_mic2(large_graph)
        benchmark(parse_mic2, text)


@requires_pytest_benchmark
class TestParseMicbBench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        b = emit_micb(small_graph)
        benchmark(parse_micb, b)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        b = emit_micb(medium_graph)
        benchmark(parse_micb, b)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        b = emit_micb(large_graph)
        benchmark(parse_micb, b)


# ---------------------------------------------------------------------------
# Throughput floors — assert pure-Python parser meets a minimum bar
# ---------------------------------------------------------------------------


def _active_tracer() -> str | None:
    """Name the line-level tracer instrumenting this process, or ``None``.

    Three probes, because no single one is both sound and complete — and
    the obvious fourth one is neither:

    * ``sys.gettrace()`` sees coverage.py's C tracer and any debugger.
      Measured on this tree under ``pytest --cov=src``: a
      ``coverage.CTracer`` already installed at module-import time, which
      is when the skip below is decided; ``None`` without ``--cov``.
    * ``sys.monitoring`` (3.12+) is the backend coverage.py uses when
      ``COVERAGE_CORE=sysmon``, and that backend leaves ``sys.gettrace()``
      empty. Reached through ``getattr`` because the 3.10 and 3.11 matrix
      rows have no ``sys.monitoring`` at all.
    * ``coverage.Coverage.current()`` is coverage.py's own answer, and the
      only probe that stays correct if it grows a third backend.

    What this deliberately does **not** test is ``"coverage" in
    sys.modules``. Measured: pytest-cov imports coverage on *every* run,
    instrumented or not, so that clause is true on all fifteen matrix rows
    and would turn the guard below into an unconditional skip rather than
    one conditional on a real environment fact.
    ``TestTheFloorsAreGuardedAgainstTheirOwnInstrumentation`` pins both
    halves of that distinction.
    """
    tracer = sys.gettrace()
    if tracer is not None:
        return f"sys.gettrace() -> {type(tracer).__module__}.{type(tracer).__name__}"
    monitoring = getattr(sys, "monitoring", None)  # 3.12+
    if monitoring is not None:
        tool = monitoring.get_tool(monitoring.COVERAGE_ID)
        if tool is not None:
            return f"sys.monitoring COVERAGE_ID held by {tool!r}"
    current = getattr(getattr(sys.modules.get("coverage"), "Coverage", None), "current", None)
    if current is not None and current() is not None:
        return "coverage.Coverage.current() reports a running Coverage"
    return None


#: Resolved once, at import: that is when the skip is decided, and by then
#: pytest-cov has already installed its tracer.
_TRACER = _active_tracer()


@pytest.mark.skipif(
    _TRACER is not None,
    reason=(f"a wall-clock throughput floor under line tracing measures the tracer, not the parser ({_TRACER})"),
)
class TestThroughputFloors:
    """Worst-acceptable throughput on a single core. Floors are
    intentionally conservative so a CI runner under load doesn't
    flake. The expected pure-Python numbers are well above these
    floors; a future Cython / Rust accelerator should push them
    much higher.

    Skipped while something is tracing this process, because then the
    number under test is the tracer's. Measured on this tree, same box,
    same commit, ``-k TestThroughputFloors``::

        pytest ...                    6 passed
        pytest ... --cov=src          2 failed, 4 passed
                                      parse_micb(small) only 4430/s   (floor 5000)
                                      parse_micb(large) only 46.3/s   (floor 50)

    CI run 33628984458 failed the same two on the one instrumented row
    (ubuntu-3.12, the only row that passes ``--cov``) at 4624/s and
    44.0/s, while the fourteen uninstrumented rows passed them. The floor
    is not lowered and the scan is not narrowed: the other fourteen rows
    still measure it on every push, and the skip states the environment
    fact that makes this row's number meaningless.
    """

    @staticmethod
    def _measure_ops_per_sec(fn, *args, max_seconds: float = 0.5) -> float:
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < max_seconds:
            fn(*args)
            n += 1
        elapsed = time.perf_counter() - t0
        return n / elapsed

    def test_small_emit_micb_above_5k_per_sec(self, small_graph: Graph) -> None:
        ops = self._measure_ops_per_sec(emit_micb, small_graph)
        assert ops > 5000, f"emit_micb(small) only {ops:.0f}/s"

    def test_small_parse_micb_above_5k_per_sec(self, small_graph: Graph) -> None:
        b = emit_micb(small_graph)
        ops = self._measure_ops_per_sec(parse_micb, b)
        assert ops > 5000, f"parse_micb(small) only {ops:.0f}/s"

    def test_medium_emit_micb_above_1k_per_sec(self, medium_graph: Graph) -> None:
        ops = self._measure_ops_per_sec(emit_micb, medium_graph)
        assert ops > 1000, f"emit_micb(medium) only {ops:.0f}/s"

    def test_medium_parse_micb_above_1k_per_sec(self, medium_graph: Graph) -> None:
        b = emit_micb(medium_graph)
        ops = self._measure_ops_per_sec(parse_micb, b)
        assert ops > 1000, f"parse_micb(medium) only {ops:.0f}/s"

    def test_large_emit_micb_above_50_per_sec(self, large_graph: Graph) -> None:
        ops = self._measure_ops_per_sec(emit_micb, large_graph, max_seconds=1.0)
        assert ops > 50, f"emit_micb(large) only {ops:.1f}/s"

    def test_large_parse_micb_above_50_per_sec(self, large_graph: Graph) -> None:
        b = emit_micb(large_graph)
        ops = self._measure_ops_per_sec(parse_micb, b, max_seconds=1.0)
        assert ops > 50, f"parse_micb(large) only {ops:.1f}/s"


# ---------------------------------------------------------------------------
# The guard on the floors above, and the two facts it rests on
# ---------------------------------------------------------------------------

#: Run in a process nobody is instrumenting, to measure the one thing an
#: instrumented process cannot tell you: that importing coverage is not
#: the same as running it. Imports the test module by path, so it is the
#: *shipped* detector under test and not a copy of it.
_IDLE_COVERAGE_PROBE = """\
import sys

sys.path.insert(0, sys.argv[1])
import coverage  # noqa: F401  — imported and never started, as pytest-cov does

import test_mic_map_bench as bench

print("imported=%s tracer=%s" % ("coverage" in sys.modules, bench._active_tracer()))
"""


class TestTheFloorsAreGuardedAgainstTheirOwnInstrumentation:
    """The skip on ``TestThroughputFloors`` is only honest if the thing it
    asks about is real. Two halves, and neither is enough alone: the
    detector must SEE a tracer that is present, and must NOT call an idle
    import one.
    """

    def test_the_detector_sees_a_tracer_that_is_present(self) -> None:
        # Positive control. Before trusting ``_active_tracer() is None``
        # anywhere, prove the method can find one it is standing inside.
        # Works on the instrumented row too, where it is already non-None.
        before = sys.gettrace()
        try:
            sys.settrace(lambda frame, event, arg: None)
            found = _active_tracer()
        finally:
            sys.settrace(before)
        assert found is not None, "the detector cannot see a tracer it is standing in — it can prove nothing"

    def test_an_imported_but_idle_coverage_is_not_reported_as_a_tracer(self) -> None:
        # The clause the detector refuses to use, measured where it can be:
        # a subprocess with coverage imported and nothing tracing.
        proc = subprocess.run(
            [sys.executable, "-c", _IDLE_COVERAGE_PROBE, str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"probe failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        out = proc.stdout.strip().splitlines()[-1]
        # Positive control first: a probe where coverage never imported
        # would report tracer=None for the wrong reason.
        assert "imported=True" in out, f"coverage was not imported in the probe, so it proves nothing: {out!r}"
        assert "tracer=None" in out, (
            f"an imported-but-idle coverage was reported as a tracer: {out!r} — the guard on "
            "TestThroughputFloors would then skip on every matrix row"
        )

    def test_the_floors_carry_the_guard(self) -> None:
        # Wiring, not intent: a detector nothing consults is decoration, and
        # a guard wired to a stale answer is worse than no guard at all.
        marks = [m for m in getattr(TestThroughputFloors, "pytestmark", []) if m.name == "skipif"]
        assert marks, "TestThroughputFloors lost its skipif — the floors are back to measuring the tracer"
        # Deliberately compared against a FRESH call, not against ``_TRACER``.
        # ``_TRACER`` is the value the mark was built from, so comparing the
        # two only asserts that a name equals itself: blinding the detector
        # leaves both sides False and this stays green. Measured — that is
        # exactly what the first version of this assertion did.
        live = _active_tracer()
        assert marks[0].args[0] is (live is not None), (
            f"the floors' skipif no longer tracks the live tracer detector (mark condition={marks[0].args[0]!r}, detector={live!r})"
        )


# ---------------------------------------------------------------------------
# Memory-ceiling check — emitted bytes never exceed input bytes by 10×
# ---------------------------------------------------------------------------


class TestMemoryCeiling:
    """Catches O(n²) blowup. We don't have a precise allocator hook in
    pure Python, but we can assert that the emitted byte length stays
    proportional to the input value count — a sanity bound that
    falsifies the worst-case 'parser allocates an N-element dict per
    value' regressions."""

    def test_emit_micb_size_under_proportional_bound(self, large_graph: Graph) -> None:
        b = emit_micb(large_graph)
        # 200 layers × 3 nodes/layer + ~400 params + small constants
        # → expect well under 100 bytes per value (typical: 5-20).
        assert len(b) < len(large_graph.values) * 100, (
            f"emit_micb produced {len(b)} bytes for {len(large_graph.values)} values — suspiciously large"
        )

    def test_emit_mic2_size_under_proportional_bound(self, large_graph: Graph) -> None:
        s = emit_mic2(large_graph)
        # Text is more verbose: budget 500 bytes per value.
        assert len(s) < len(large_graph.values) * 500
