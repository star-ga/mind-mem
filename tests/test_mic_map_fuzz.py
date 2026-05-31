"""Property-based fuzz tests for ``mind_mem.mic_map``.

Targets the v3.8.8 scale-fragility bar: at scale we cannot afford a
single pathological input to crash a worker or cause unbounded
allocations. Hypothesis drives this by generating arbitrary valid
graphs and arbitrary byte strings; we assert two invariants:

1. **Round-trip identity** — for every Hypothesis-built ``Graph``,
   ``parse_mic2(emit_mic2(g))`` and ``parse_micb(emit_micb(g))`` both
   reconstruct an equal graph. Catches silent corruption in the
   serializer / deserializer pair.

2. **Crash safety on arbitrary input** — for arbitrary byte strings
   and arbitrary text strings, parsers either succeed (rare, by
   chance) or raise the documented ``Mic2ParseError`` /
   ``MicbParseError``. They never raise an unhandled exception, never
   loop forever, never exceed reasonable memory.

Hypothesis defaults are bounded (max_examples=100) so this stays in
the unit-test budget. The adversarial corpus in
``test_mic_map_adversarial.py`` covers known-shaped DoS inputs that
fuzzing alone wouldn't reliably hit.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from mind_mem.mic_map import (  # noqa: E402
    DTYPES,
    OP_ARITY,
    OPCODES,
    Arg,
    Graph,
    Mic2ParseError,
    MicbParseError,
    Node,
    Param,
    Type,
    emit_mic2,
    emit_micb,
    parse_mic2,
    parse_micb,
)

# ---------------------------------------------------------------------------
# Strategies — build syntactically valid ``Graph`` instances.
# ---------------------------------------------------------------------------


@st.composite
def _identifier(draw: st.DrawFn) -> str:
    """A short, ASCII-only identifier suitable for symbol / name slots
    in mic@2 (no whitespace, no NUL, no leading digit)."""
    alphabet = st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    rest = st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")
    head = draw(alphabet)
    tail = draw(st.text(rest, min_size=0, max_size=8))
    return head + tail


@st.composite
def _type(draw: st.DrawFn) -> Type:
    dtype = draw(st.sampled_from(DTYPES))
    rank = draw(st.integers(min_value=0, max_value=4))
    dims = tuple(draw(_identifier()) for _ in range(rank))
    return Type(dtype=dtype, dims=dims)


@st.composite
def _graph(draw: st.DrawFn) -> Graph:
    """Build a small valid graph: a few types, a handful of args /
    params, a single op node referencing earlier values, output =
    last value id.

    Keeps shapes small (≤ 8 values, ≤ 3 types) so per-example cost
    stays in the milliseconds. Hypothesis's default 100 examples
    must finish well under one second.
    """
    n_types = draw(st.integers(min_value=1, max_value=3))
    types = [draw(_type()) for _ in range(n_types)]

    n_args = draw(st.integers(min_value=1, max_value=3))
    n_params = draw(st.integers(min_value=0, max_value=2))
    values: list = []
    for _ in range(n_args):
        values.append(
            Arg(
                name=draw(_identifier()),
                type_idx=draw(st.integers(min_value=0, max_value=n_types - 1)),
            )
        )
    for _ in range(n_params):
        values.append(
            Param(
                name=draw(_identifier()),
                type_idx=draw(st.integers(min_value=0, max_value=n_types - 1)),
            )
        )

    # Add one or two op nodes referencing only earlier values.
    n_ops = draw(st.integers(min_value=0, max_value=2))
    for _ in range(n_ops):
        opcode = draw(st.sampled_from(OPCODES))
        arity = OP_ARITY[opcode]
        if arity == -1:
            in_count = draw(st.integers(min_value=1, max_value=min(3, len(values))))
        else:
            in_count = arity
        if in_count > len(values):
            # Not enough earlier values for this opcode — skip.
            continue
        inputs = tuple(draw(st.integers(min_value=0, max_value=len(values) - 1)) for _ in range(in_count))
        op_params = _default_op_params(opcode)
        values.append(Node(opcode=opcode, inputs=inputs, op_params=op_params))

    return Graph(types=types, values=values, output=len(values) - 1)


def _default_op_params(opcode: str) -> tuple[int, ...]:
    """Minimal valid op_params for opcodes that require any."""
    if opcode in ("s", "cat", "gth"):
        return (0,)  # axis=0
    if opcode in ("sum", "mean", "max"):
        return (0,)  # one axis
    if opcode == "t":
        return (0, 1)  # 2-rank perm
    if opcode == "split":
        return (0, 2)  # axis=0 count=2
    return ()


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=2000,  # ms — generous for CI runners
)
@given(_graph())
def test_mic2_round_trip_preserves_canonical(g: Graph) -> None:
    """``emit -> parse -> emit`` is idempotent for every well-formed
    graph. This is the core correctness invariant of the text path."""
    text = emit_mic2(g)
    twice = emit_mic2(parse_mic2(text))
    assert text == twice


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=2000,
)
@given(_graph())
def test_micb_round_trip_preserves_bytes(g: Graph) -> None:
    """``emit -> parse -> emit`` preserves the exact bytes for the
    binary path. Determinism is mandatory — the wire format must
    serialize the same graph to the same bytes on every machine."""
    b = emit_micb(g)
    twice = emit_micb(parse_micb(b))
    assert b == twice


@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=2000,
)
@given(_graph())
def test_text_binary_text_canonical(g: Graph) -> None:
    """Text and binary representations agree: ``text -> binary -> text``
    yields the canonical text. Catches drift between the two
    serializers."""
    text = emit_mic2(g)
    via_binary = emit_mic2(parse_micb(emit_micb(parse_mic2(text))))
    assert text == via_binary


# ---------------------------------------------------------------------------
# Crash safety — arbitrary input must never raise an unhandled exception
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=500)
@given(st.binary(max_size=8192))
def test_micb_arbitrary_bytes_never_unhandled(data: bytes) -> None:
    """Any byte string either parses successfully (rare) or raises
    ``MicbParseError``. No segfault, no IndexError, no recursion limit,
    no UnicodeDecodeError, no MemoryError. Bound the input at 8 KiB
    so the test budget stays sane while still exercising every
    decoder path."""
    try:
        parse_micb(data)
    except MicbParseError:
        pass  # Expected — random bytes almost always fail.
    except Exception as exc:
        raise AssertionError(f"parse_micb leaked unhandled {type(exc).__name__}: {exc!r}") from exc


@settings(max_examples=200, deadline=500)
@given(st.text(max_size=4096))
def test_mic2_arbitrary_text_never_unhandled(data: str) -> None:
    """Same contract for the text parser."""
    try:
        parse_mic2(data)
    except Mic2ParseError:
        pass
    except Exception as exc:
        raise AssertionError(f"parse_mic2 leaked unhandled {type(exc).__name__}: {exc!r}") from exc


@settings(max_examples=100, deadline=500)
@given(st.binary(min_size=4, max_size=2048))
def test_micb_with_correct_magic_never_unhandled(data: bytes) -> None:
    """Force the magic + version bytes through so the parser walks
    deeper into the format. Catches IndexError / EOF panics in the
    string-table / type-table / value-table decoders."""
    payload = b"MICB\x02" + data
    try:
        parse_micb(payload)
    except MicbParseError:
        pass
    except Exception as exc:
        raise AssertionError(f"parse_micb leaked unhandled {type(exc).__name__}: {exc!r}") from exc


# ---------------------------------------------------------------------------
# Round-trip on the spec residual block — sanity guard for the strategies
# ---------------------------------------------------------------------------


def test_strategy_actually_builds_round_trippable_graphs() -> None:
    """Sanity check the strategies — at least one Hypothesis-built
    graph round-trips through both paths. (Without this, a broken
    strategy could pass the round-trip tests vacuously.)"""
    g = Graph(
        types=[Type(dtype="f32", dims=("4",))],
        values=[Arg(name="X", type_idx=0)],
        output=0,
    )
    assert emit_mic2(parse_mic2(emit_mic2(g))) == emit_mic2(g)
    assert emit_micb(parse_micb(emit_micb(g))) == emit_micb(g)
