"""pytest-benchmark suite for ``mind_mem.mic_map``.

Establishes the v3.8.8 throughput floor for the serialization
primitive. Three graph sizes — small (residual block, 7 values),
medium (a transformer-like layer with ~30 values), and large
(deep stack with 200 values) — exercised through both formats.

Skipped unless ``pytest-benchmark`` is installed (declared in the
``[benchmark]`` extras). Run with::

    pytest tests/test_mic_map_bench.py --benchmark-only

Minimum-throughput assertions live in ``TestThroughputFloors`` —
the floors are deliberately conservative so a 2-3× speedup from a
future Cython port is visible. Floors apply to single-core
operation; they bound the *worst* acceptable throughput, not the
*expected* throughput.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pytest_benchmark")

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


class TestEmitMic2Bench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        benchmark(emit_mic2, small_graph)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        benchmark(emit_mic2, medium_graph)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        benchmark(emit_mic2, large_graph)


class TestEmitMicbBench:
    def test_small(self, benchmark, small_graph: Graph) -> None:
        benchmark(emit_micb, small_graph)

    def test_medium(self, benchmark, medium_graph: Graph) -> None:
        benchmark(emit_micb, medium_graph)

    def test_large(self, benchmark, large_graph: Graph) -> None:
        benchmark(emit_micb, large_graph)


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


class TestThroughputFloors:
    """Worst-acceptable throughput on a single core. Floors are
    intentionally conservative so a CI runner under load doesn't
    flake. The expected pure-Python numbers are well above these
    floors; a future Cython / Rust accelerator should push them
    much higher."""

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
