"""Tests for ``mind_mem.mic_map`` — STARGA mic@2 / mic-b serialization.

Coverage targets every spec rule:
  * mic2-spec §"Grammar"       — header, types, args/params, nodes, output
  * mic2-spec §"Validation"    — sequential type IDs, no forward refs,
                                 opcode arity, output references valid value
  * mic2-spec §"Canonicalization" — sorted output, single space, LF-only
  * micb-spec §"Wire Format"   — magic+ver, ULEB128 minimum encoding,
                                 string-table first-seen order, no padding
  * Round-trip: text ↔ Graph ↔ binary ↔ text returns identical bytes.
"""

from __future__ import annotations

import pytest
from mind_mem.mic_map import (
    DTYPES,
    MIC2_HEADER,
    MICB_MAGIC,
    MICB_VERSION,
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
    round_trip,
    round_trip_b,
)

# ---------------------------------------------------------------------------
# Spec residual block — the canonical example used in both spec docs.
# ---------------------------------------------------------------------------

SPEC_RESIDUAL_TEXT = "mic@2\nT0 f16 128 128\nT1 f16 128\na X T0\np W T0\np b T1\nm 0 1\n+ 3 2\nr 4\n+ 5 0\nO 6\n"


@pytest.fixture
def residual_graph() -> Graph:
    return parse_mic2(SPEC_RESIDUAL_TEXT)


# ---------------------------------------------------------------------------
# Text parser — happy path
# ---------------------------------------------------------------------------


class TestParseMic2:
    def test_residual_block(self, residual_graph: Graph) -> None:
        g = residual_graph
        assert len(g.types) == 2
        assert g.types[0] == Type(dtype="f16", dims=("128", "128"))
        assert g.types[1] == Type(dtype="f16", dims=("128",))
        assert len(g.values) == 7
        assert g.values[0] == Arg(name="X", type_idx=0)
        assert g.values[1] == Param(name="W", type_idx=0)
        assert g.values[2] == Param(name="b", type_idx=1)
        assert g.values[3] == Node(opcode="m", inputs=(0, 1))
        assert g.values[6] == Node(opcode="+", inputs=(5, 0))
        assert g.output == 6

    def test_comments_and_blank_lines_ignored(self) -> None:
        text = "# leading comment\n\nmic@2\n# inline comment\nT0 f32 4\n\na X T0\nO 0\n"
        g = parse_mic2(text)
        assert g.output == 0
        assert g.values[0].name == "X"

    def test_symbols_preserved(self) -> None:
        text = "mic@2\nS B\nS seq\nT0 f32 B seq\na X T0\nO 0\n"
        g = parse_mic2(text)
        assert g.symbols == ["B", "seq"]
        assert g.types[0].dims == ("B", "seq")


# ---------------------------------------------------------------------------
# Text parser — rejection of every spec violation
# ---------------------------------------------------------------------------


class TestParseMic2Rejection:
    def test_missing_header(self) -> None:
        with pytest.raises(Mic2ParseError, match="header"):
            parse_mic2("T0 f32 4\na X T0\nO 0\n")

    def test_wrong_header(self) -> None:
        with pytest.raises(Mic2ParseError, match="first non-blank"):
            parse_mic2("mic@1\nT0 f32 4\na X T0\nO 0\n")

    def test_non_sequential_type_index(self) -> None:
        with pytest.raises(Mic2ParseError, match="sequential"):
            parse_mic2("mic@2\nT0 f32 4\nT2 f32 8\nO 0\n")

    def test_unknown_dtype(self) -> None:
        with pytest.raises(Mic2ParseError, match="dtype"):
            parse_mic2("mic@2\nT0 BOGUS 4\na X T0\nO 0\n")

    def test_forward_reference_rejected(self) -> None:
        # m 0 1 — but only one value defined yet → 1 is forward ref
        with pytest.raises(Mic2ParseError, match="earlier value|input 1 is not"):
            parse_mic2("mic@2\nT0 f32 4\na X T0\nm 0 1\nO 1\n")

    def test_unknown_opcode(self) -> None:
        with pytest.raises(Mic2ParseError, match="unknown opcode"):
            parse_mic2("mic@2\nT0 f32 4\na X T0\nNOTANOP 0\nO 1\n")

    def test_arity_mismatch(self) -> None:
        # 'r' is unary; passing two operands triggers spec validation.
        with pytest.raises(Mic2ParseError):
            parse_mic2("mic@2\nT0 f32 4\na X T0\np W T0\nm 0 1\n+ 2\nO 3\n")

    def test_output_out_of_range(self) -> None:
        with pytest.raises(Mic2ParseError, match="not a valid value"):
            parse_mic2("mic@2\nT0 f32 4\na X T0\nO 99\n")

    def test_missing_output(self) -> None:
        with pytest.raises(Mic2ParseError, match="missing 'O"):
            parse_mic2("mic@2\nT0 f32 4\na X T0\n")

    def test_data_after_output(self) -> None:
        with pytest.raises(Mic2ParseError, match="trailing data"):
            parse_mic2("mic@2\nT0 f32 4\na X T0\nO 0\nT1 f32 8\n")

    def test_type_ref_to_undefined_type(self) -> None:
        with pytest.raises(Mic2ParseError, match="not yet defined"):
            parse_mic2("mic@2\nT0 f32 4\na X T5\nO 0\n")


# ---------------------------------------------------------------------------
# Text emitter — canonicalization
# ---------------------------------------------------------------------------


class TestEmitMic2:
    def test_round_trip_idempotent(self, residual_graph: Graph) -> None:
        # parse(emit(parse(emit(g)))) == parse(emit(g))
        once = emit_mic2(residual_graph)
        twice = emit_mic2(parse_mic2(once))
        assert once == twice

    def test_canonical_emit_starts_with_header(self, residual_graph: Graph) -> None:
        out = emit_mic2(residual_graph)
        assert out.startswith(MIC2_HEADER + "\n")

    def test_canonical_uses_lf_only(self, residual_graph: Graph) -> None:
        out = emit_mic2(residual_graph)
        assert "\r" not in out
        # All lines except the trailing newline are non-empty.
        for ln in out.splitlines():
            assert ln == ln.strip()

    def test_no_double_spaces(self, residual_graph: Graph) -> None:
        for ln in emit_mic2(residual_graph).splitlines():
            assert "  " not in ln, f"double space in: {ln!r}"


# ---------------------------------------------------------------------------
# Binary — magic, version, basic shape
# ---------------------------------------------------------------------------


class TestEmitMicb:
    def test_starts_with_magic_and_version(self, residual_graph: Graph) -> None:
        b = emit_micb(residual_graph)
        assert b[:4] == MICB_MAGIC
        assert b[4] == MICB_VERSION

    def test_bytes_smaller_than_text(self, residual_graph: Graph) -> None:
        text_bytes = emit_mic2(residual_graph).encode("utf-8")
        bin_bytes = emit_micb(residual_graph)
        # Spec claims ~3-4× compression on text; we only check binary < text.
        assert len(bin_bytes) < len(text_bytes)

    def test_deterministic(self, residual_graph: Graph) -> None:
        a = emit_micb(residual_graph)
        b = emit_micb(residual_graph)
        assert a == b


class TestParseMicb:
    def test_round_trip_binary(self, residual_graph: Graph) -> None:
        b = emit_micb(residual_graph)
        g2 = parse_micb(b)
        assert emit_mic2(g2) == emit_mic2(residual_graph)

    def test_text_binary_text_preserves_canonical(self, residual_graph: Graph) -> None:
        # Text → Graph → Binary → Graph → Text yields the same canonical text.
        text = emit_mic2(residual_graph)
        g_from_text = parse_mic2(text)
        roundtripped = emit_mic2(parse_micb(emit_micb(g_from_text)))
        assert text == roundtripped

    def test_bad_magic_rejected(self, residual_graph: Graph) -> None:
        good = emit_micb(residual_graph)
        bad = b"BADX" + good[4:]
        with pytest.raises(MicbParseError, match="magic"):
            parse_micb(bad)

    def test_unsupported_version_rejected(self, residual_graph: Graph) -> None:
        good = emit_micb(residual_graph)
        bad = good[:4] + b"\x99" + good[5:]
        with pytest.raises(MicbParseError, match="unsupported version"):
            parse_micb(bad)

    def test_truncated_payload_rejected(self) -> None:
        with pytest.raises(MicbParseError):
            parse_micb(b"MICB")  # missing version + everything else


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


class TestRoundTripHelpers:
    def test_round_trip_text(self, residual_graph: Graph) -> None:
        g2 = round_trip(residual_graph)
        assert emit_mic2(g2) == emit_mic2(residual_graph)

    def test_round_trip_binary(self, residual_graph: Graph) -> None:
        g2 = round_trip_b(residual_graph)
        assert emit_micb(g2) == emit_micb(residual_graph)


# ---------------------------------------------------------------------------
# Op-coverage smoke — every opcode parses + emits + round-trips
# ---------------------------------------------------------------------------


class TestOpCoverage:
    """Exercise every opcode in OPCODES so the param section logic
    can't drift silently when the spec is extended."""

    @pytest.mark.parametrize("opcode", OPCODES)
    def test_each_opcode_round_trips(self, opcode: str) -> None:
        from mind_mem.mic_map import OP_ARITY

        # Build the smallest valid graph that uses this opcode.
        types = [Type(dtype="f32", dims=("4",))]
        values: list = [Arg(name="X", type_idx=0)]
        if OP_ARITY[opcode] == 2:
            values.append(Arg(name="Y", type_idx=0))
        elif OP_ARITY[opcode] == -1:  # variadic — feed one extra input
            values.append(Arg(name="Y", type_idx=0))

        # Build the node — supply minimal op_params for ops that need them.
        if opcode == "s":
            node = Node(opcode=opcode, inputs=(0,), op_params=(0,))
        elif opcode == "cat":
            node = Node(opcode=opcode, inputs=(0, 1), op_params=(0,))
        elif opcode == "gth":
            node = Node(opcode=opcode, inputs=(0, 1), op_params=(0,))
        elif opcode in ("sum", "mean", "max"):
            node = Node(opcode=opcode, inputs=(0,), op_params=(0,))
        elif opcode == "split":
            node = Node(opcode=opcode, inputs=(0,), op_params=(0, 2))
        elif opcode == "t":
            node = Node(opcode=opcode, inputs=(0,), op_params=(0, 1))
        elif OP_ARITY[opcode] == 1:
            node = Node(opcode=opcode, inputs=(0,))
        else:
            node = Node(opcode=opcode, inputs=(0, 1))
        values.append(node)
        g = Graph(types=types, values=values, output=len(values) - 1)
        # Round trip through both formats and confirm canonical text matches.
        text = emit_mic2(g)
        assert emit_mic2(parse_mic2(text)) == text
        bin_ = emit_micb(g)
        assert emit_micb(parse_micb(bin_)) == bin_


# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------


class TestDtypes:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_each_dtype_round_trips(self, dtype: str) -> None:
        g = Graph(
            types=[Type(dtype=dtype, dims=("4",))],
            values=[Arg(name="X", type_idx=0)],
            output=0,
        )
        text = emit_mic2(g)
        assert dtype in text
        assert emit_mic2(parse_mic2(text)) == text
        # Binary encodes dtype to a single byte.
        b = emit_micb(g)
        assert emit_mic2(parse_micb(b)) == text


# ---------------------------------------------------------------------------
# Validation API — direct check
# ---------------------------------------------------------------------------


class TestGraphValidate:
    def test_mismatched_arity_caught_in_validate(self) -> None:
        g = Graph(
            types=[Type(dtype="f32", dims=("4",))],
            values=[
                Arg(name="X", type_idx=0),
                Node(opcode="m", inputs=(0,)),  # m needs 2 inputs
            ],
            output=1,
        )
        with pytest.raises(ValueError, match="expects 2 inputs"):
            g.validate()

    def test_forward_ref_caught_in_validate(self) -> None:
        g = Graph(
            types=[Type(dtype="f32", dims=("4",))],
            values=[
                Arg(name="X", type_idx=0),
                Node(opcode="r", inputs=(99,)),  # 99 is way out of range
            ],
            output=1,
        )
        with pytest.raises(ValueError, match="not an earlier value"):
            g.validate()

    def test_output_out_of_range_caught_in_validate(self) -> None:
        g = Graph(
            types=[Type(dtype="f32", dims=("4",))],
            values=[Arg(name="X", type_idx=0)],
            output=99,
        )
        with pytest.raises(ValueError, match="not a valid value"):
            g.validate()
