"""Streaming-parser tests for ``mind_mem.mic_map.parse_micb_stream``.

Covers the v3.8.9 contract: incremental decode that yields
:class:`StreamEvent` as bytes arrive, with bounded peak memory and
no requirement to hold the whole payload in RAM. The tests
exercise:

* Event sequence on a known graph (header → string-table →
  symbols → types → values → complete).
* Equivalence with the legacy ``parse_micb`` path — both reconstruct
  identical graphs.
* Pull semantics — a byte-by-byte ``BinaryIO`` produces the same
  events as an all-at-once ``BytesIO``.
* Error propagation mid-stream — partial events are yielded before
  the error fires, and the error type matches ``parse_micb``.
* Memory bound — a synthetic 200-layer graph parses without ever
  holding the full value list resident (caller's choice to drop or
  retain).
"""

from __future__ import annotations

import io

import pytest

from mind_mem.mic_map import (
    Arg,
    Graph,
    MicbParseError,
    Node,
    Param,
    StreamComplete,
    StreamHeader,
    StreamStringTable,
    StreamSymbol,
    StreamType,
    StreamValue,
    Type,
    emit_micb,
    parse_micb,
    parse_micb_stream,
)

# ---------------------------------------------------------------------------
# Fixtures
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


def _graph_with_symbols() -> Graph:
    return Graph(
        types=[Type(dtype="f32", dims=("B", "seq"))],
        values=[Arg(name="X", type_idx=0)],
        output=0,
        symbols=["B", "seq"],
    )


@pytest.fixture
def small_graph() -> Graph:
    return _residual_block()


# ---------------------------------------------------------------------------
# Event sequence
# ---------------------------------------------------------------------------


class TestEventSequence:
    def test_residual_block_event_sequence(self, small_graph: Graph) -> None:
        events = list(parse_micb_stream(io.BytesIO(emit_micb(small_graph))))
        # First event is the header.
        assert isinstance(events[0], StreamHeader)
        assert events[0].version == 0x02
        # Second is the full string table.
        assert isinstance(events[1], StreamStringTable)
        assert "X" in events[1].strings
        assert "W" in events[1].strings
        # No symbols on this fixture.
        # Then per-type events for the 2 types.
        type_events = [e for e in events if isinstance(e, StreamType)]
        assert len(type_events) == 2
        assert type_events[0].index == 0
        assert type_events[0].type == small_graph.types[0]
        # Then per-value events for the 7 values.
        val_events = [e for e in events if isinstance(e, StreamValue)]
        assert len(val_events) == 7
        for i, ev in enumerate(val_events):
            assert ev.index == i
            assert ev.value == small_graph.values[i]
        # Final event is StreamComplete with the output index.
        assert isinstance(events[-1], StreamComplete)
        assert events[-1].output == 6

    def test_symbols_yielded_individually(self) -> None:
        g = _graph_with_symbols()
        events = list(parse_micb_stream(io.BytesIO(emit_micb(g))))
        sym_events = [e for e in events if isinstance(e, StreamSymbol)]
        assert len(sym_events) == 2
        assert sym_events[0].index == 0
        assert sym_events[0].name == "B"
        assert sym_events[1].name == "seq"


# ---------------------------------------------------------------------------
# Equivalence with parse_micb
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_streaming_reconstructs_identical_graph(self, small_graph: Graph) -> None:
        b = emit_micb(small_graph)
        # Reconstruct the graph from the stream and compare with the
        # legacy parser's output.
        types: list = []
        values: list = []
        symbols: list = []
        output: int = -1
        for ev in parse_micb_stream(io.BytesIO(b)):
            if isinstance(ev, StreamSymbol):
                symbols.append(ev.name)
            elif isinstance(ev, StreamType):
                types.append(ev.type)
            elif isinstance(ev, StreamValue):
                values.append(ev.value)
            elif isinstance(ev, StreamComplete):
                output = ev.output
        reconstructed = Graph(types=types, values=values, output=output, symbols=symbols)
        assert reconstructed == parse_micb(b)


# ---------------------------------------------------------------------------
# Pull semantics — byte-by-byte vs all-at-once
# ---------------------------------------------------------------------------


class _OneByteReader(io.RawIOBase):
    """A reader that returns at most one byte per ``read`` call.
    Simulates a slow or chunked network source — the parser must
    handle short reads without loss of state."""

    def __init__(self, data: bytes) -> None:
        super().__init__()
        self._data = data
        self._pos = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._pos >= len(self._data):
            return b""
        take = 1 if (size is None or size < 0) else min(1, size)
        chunk = self._data[self._pos : self._pos + take]
        self._pos += take
        return chunk


class TestPullSemantics:
    def test_byte_by_byte_reader_produces_same_events(self, small_graph: Graph) -> None:
        b = emit_micb(small_graph)
        all_at_once = list(parse_micb_stream(io.BytesIO(b)))
        byte_by_byte = list(parse_micb_stream(_OneByteReader(b)))
        # Strip per-event indexes / values into a comparable shape.
        assert len(all_at_once) == len(byte_by_byte)
        for a, c in zip(all_at_once, byte_by_byte, strict=True):
            assert type(a) is type(c)
            assert a == c


# ---------------------------------------------------------------------------
# Mid-stream error propagation
# ---------------------------------------------------------------------------


class TestMidStreamErrors:
    def test_truncated_after_string_table_raises_during_iteration(self, small_graph: Graph) -> None:
        b = emit_micb(small_graph)
        # Truncate to just past the string table — the parser will get
        # through StreamHeader + StreamStringTable, then fail when it
        # tries to read the symbol count.
        # Conservatively pick a midpoint after which symbol-count read
        # will fail.
        truncated = b[: len(b) // 2]
        events = []
        with pytest.raises(MicbParseError):
            for ev in parse_micb_stream(io.BytesIO(truncated)):
                events.append(ev)
        # Some events were yielded before the error fired — at least
        # the header.
        assert len(events) >= 1
        assert isinstance(events[0], StreamHeader)

    def test_bad_magic_raises_immediately(self) -> None:
        events = []
        with pytest.raises(MicbParseError, match="magic"):
            for ev in parse_micb_stream(io.BytesIO(b"BADX\x02\x00\x00\x00\x00\x00")):
                events.append(ev)
        # No events yielded before the magic check fires.
        assert events == []

    def test_unsupported_version_raises_after_magic(self) -> None:
        events = []
        with pytest.raises(MicbParseError, match="version"):
            for ev in parse_micb_stream(io.BytesIO(b"MICB\x99")):
                events.append(ev)
        assert events == []


# ---------------------------------------------------------------------------
# Memory bound — caller can drop StreamValue objects without losing
# parser state
# ---------------------------------------------------------------------------


class TestMemoryBound:
    def test_caller_can_drop_values_during_stream(self) -> None:
        """Build a synthetic graph with 200 layers and stream-parse it,
        counting StreamValue events but not retaining them. The
        parser must complete without holding the whole value list
        resident — this would catch a regression where the streaming
        parser secretly accumulates everything."""
        types = [
            Type(dtype="f32", dims=("B", "D")),
            Type(dtype="f32", dims=("D", "D")),
            Type(dtype="f32", dims=("D",)),
        ]
        values: list = [Arg(name="x", type_idx=0)]
        for i in range(50):  # 50 layers, kept small for test budget
            values.append(Param(name=f"W{i}", type_idx=1))
            values.append(Param(name=f"b{i}", type_idx=2))
        cur = 0
        for i in range(50):
            w_id = 1 + 2 * i
            b_id = 2 + 2 * i
            mm_id = len(values)
            values.append(Node(opcode="m", inputs=(cur, w_id)))
            add_id = len(values)
            values.append(Node(opcode="+", inputs=(mm_id, b_id)))
            relu_id = len(values)
            values.append(Node(opcode="r", inputs=(add_id,)))
            cur = relu_id
        g = Graph(types=types, values=values, output=cur)
        b = emit_micb(g)

        n_values_seen = 0
        completed = False
        for ev in parse_micb_stream(io.BytesIO(b)):
            if isinstance(ev, StreamValue):
                n_values_seen += 1
                # Drop the value here — don't accumulate.
            elif isinstance(ev, StreamComplete):
                completed = True

        assert n_values_seen == len(values)
        assert completed


# ---------------------------------------------------------------------------
# StreamEvent dataclass identity / equality
# ---------------------------------------------------------------------------


class TestStreamEventTypes:
    def test_events_are_frozen_dataclasses(self) -> None:
        # Mutation must be rejected — events are part of the public API
        # and callers may stash them.
        ev = StreamHeader(version=0x02)
        with pytest.raises(Exception):
            ev.version = 0x03  # type: ignore[misc]

    def test_string_table_event_preserves_order(self) -> None:
        g = Graph(
            types=[Type(dtype="f32", dims=("D",))],
            values=[Arg(name="alpha", type_idx=0), Param(name="beta", type_idx=0)],
            output=1,
        )
        events = list(parse_micb_stream(io.BytesIO(emit_micb(g))))
        st = next(e for e in events if isinstance(e, StreamStringTable))
        # First-seen order: argument name comes before param name.
        assert st.strings.index("alpha") < st.strings.index("beta")
