"""Adversarial corpus for ``mind_mem.mic_map`` parsers.

Hand-crafted DoS-shaped inputs that fuzzing alone wouldn't reliably
hit. Every test asserts two things:

1. The parser raises the documented ``Mic2ParseError`` /
   ``MicbParseError`` (not a leaked exception of any other type).
2. The parser returns within a small fixed time budget — no
   pathological input should let the parser do more than a few ms
   of work before deciding the input is rotten.

Time-budget assertions are intentionally generous (50ms) because CI
runners vary; the goal is to catch O(n²) or unbounded-loop
regressions, not to benchmark precise latency.

The corpus covers the named DoS vectors from the spec security
checklist (``mic2-spec.md`` §"Security Limits" and ``micb-spec.md``
§"Wire Format Robustness"):

* Varint bombs (ULEB128 padded with continuation bytes)
* Length-prefix overflow (string-table claims a billion strings)
* Truncation at every offset
* Magic / version mismatches
* Tag-byte fuzzing (unknown value tags, unknown opcodes, unknown
  dtypes)
* Forward-reference exploits (claim impossible value IDs)
* Output-out-of-range
* Empty payloads at every layer

The 10 MiB / 100k value / 32 dim / 64 KiB string limits in
``mic_map.py`` are the load-bearing defence — these tests prove
they actually fire.
"""

from __future__ import annotations

import time

import pytest
from mind_mem.mic_map import (
    MAX_DIM_COUNT,
    MAX_INPUT_BYTES,
    MAX_LINE_COUNT,
    MAX_STRING_COUNT,
    MAX_STRING_LEN,
    MAX_VALUE_COUNT,
    MICB_MAGIC,
    Mic2ParseError,
    MicbParseError,
    parse_mic2,
    parse_micb,
)

# ---------------------------------------------------------------------------
# Time-budget helper
# ---------------------------------------------------------------------------

# Generous CI-friendly ceiling. A correct parser rejects pathological
# input in microseconds; if any test ever takes >500ms it's a bug.
TIME_BUDGET_S = 0.5


def _bounded_call(fn, *args, **kwargs):
    """Call ``fn`` and assert wall-clock time stays under
    ``TIME_BUDGET_S``. Returns either the result or raises the same
    exception ``fn`` raised."""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - t0
        assert elapsed < TIME_BUDGET_S, f"parse took {elapsed * 1000:.1f}ms (budget {TIME_BUDGET_S * 1000:.0f}ms)"
    return result


# ---------------------------------------------------------------------------
# Varint bombs — minimum-encoding enforcement
# ---------------------------------------------------------------------------


class TestVarintBombs:
    def test_uleb128_padded_to_70_bits_rejected(self) -> None:
        # 11 continuation bytes — exceeds the 70-bit limit baked into
        # the ULEB128 decoder. Without the limit, an attacker could
        # send arbitrarily-many bytes per varint.
        bomb = b"\x80" * 11 + b"\x00"
        payload = MICB_MAGIC + b"\x02" + bomb
        with pytest.raises(MicbParseError, match="too long"):
            _bounded_call(parse_micb, payload)

    def test_megabyte_of_continuation_bytes_short_circuits(self) -> None:
        # 1 MiB of \x80 bytes — every byte would advance shift by 7;
        # the 70-bit cap kicks in after ~10 bytes and aborts.
        bomb = b"\x80" * (1024 * 1024) + b"\x01"
        payload = MICB_MAGIC + b"\x02" + bomb
        with pytest.raises(MicbParseError):
            _bounded_call(parse_micb, payload)


# ---------------------------------------------------------------------------
# Length-prefix overflow — claim more elements than the spec allows
# ---------------------------------------------------------------------------


class TestLengthPrefixOverflow:
    @staticmethod
    def _uleb128(n: int) -> bytes:
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b | 0x80)
            else:
                out.append(b)
                return bytes(out)

    def test_string_count_exceeds_max(self) -> None:
        payload = MICB_MAGIC + b"\x02" + self._uleb128(MAX_STRING_COUNT + 1)
        with pytest.raises(MicbParseError, match="string count"):
            _bounded_call(parse_micb, payload)

    def test_value_count_exceeds_max(self) -> None:
        # Empty string + symbol + type tables, then a value count
        # that exceeds the cap. Should be caught before the parser
        # tries to allocate.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + self._uleb128(0)  # 0 strings
            + self._uleb128(0)  # 0 symbols
            + self._uleb128(0)  # 0 types
            + self._uleb128(MAX_VALUE_COUNT + 1)  # too many values
        )
        with pytest.raises(MicbParseError, match="value count"):
            _bounded_call(parse_micb, payload)

    def test_string_length_exceeds_max(self) -> None:
        payload = (
            MICB_MAGIC
            + b"\x02"
            + self._uleb128(1)  # 1 string
            + self._uleb128(MAX_STRING_LEN + 1)  # too long
        )
        with pytest.raises(MicbParseError, match="string length"):
            _bounded_call(parse_micb, payload)

    def test_type_rank_exceeds_max(self) -> None:
        payload = (
            MICB_MAGIC
            + b"\x02"
            + self._uleb128(0)  # strings
            + self._uleb128(0)  # symbols
            + self._uleb128(1)  # 1 type
            + b"\x01"  # dtype f32
            + self._uleb128(MAX_DIM_COUNT + 1)  # too many dims
        )
        with pytest.raises(MicbParseError, match="rank"):
            _bounded_call(parse_micb, payload)

    def test_total_input_above_10mb_rejected(self) -> None:
        # MAX_INPUT_BYTES is 10 MiB by default. We test just past the
        # boundary, not orders of magnitude above, to keep the test
        # cheap.
        payload = b"X" * (MAX_INPUT_BYTES + 1)
        with pytest.raises(MicbParseError, match="exceeds"):
            _bounded_call(parse_micb, payload)


# ---------------------------------------------------------------------------
# Truncation at every layer
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_truncated_at_magic(self) -> None:
        with pytest.raises(MicbParseError, match="shorter than magic"):
            _bounded_call(parse_micb, b"MIC")

    def test_truncated_just_after_magic(self) -> None:
        with pytest.raises(MicbParseError, match="shorter than magic"):
            _bounded_call(parse_micb, b"MICB")

    def test_truncated_after_string_count(self) -> None:
        # Claims 1 string but no length follows.
        payload = MICB_MAGIC + b"\x02" + b"\x01"
        with pytest.raises(MicbParseError, match="EOF"):
            _bounded_call(parse_micb, payload)

    def test_truncated_string_payload(self) -> None:
        # 1 string of length 5, only 2 bytes follow.
        payload = MICB_MAGIC + b"\x02" + b"\x01" + b"\x05" + b"hi"
        with pytest.raises(MicbParseError, match="EOF in string table"):
            _bounded_call(parse_micb, payload)


# ---------------------------------------------------------------------------
# Magic / version
# ---------------------------------------------------------------------------


class TestMagicVersion:
    def test_unknown_magic(self) -> None:
        with pytest.raises(MicbParseError, match="magic"):
            _bounded_call(parse_micb, b"BADX\x02")

    def test_unsupported_version(self) -> None:
        with pytest.raises(MicbParseError, match="version"):
            _bounded_call(parse_micb, MICB_MAGIC + b"\x99")

    def test_version_zero(self) -> None:
        with pytest.raises(MicbParseError, match="version"):
            _bounded_call(parse_micb, MICB_MAGIC + b"\x00")


# ---------------------------------------------------------------------------
# Tag-byte fuzzing — unknown value tags, opcodes, dtypes
# ---------------------------------------------------------------------------


class TestUnknownTags:
    def test_unknown_value_tag(self) -> None:
        # Value tag must be 0 (Arg), 1 (Param), or 2 (Node).
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x00"  # 0 strings
            + b"\x00"  # 0 symbols
            + b"\x00"  # 0 types
            + b"\x01"  # 1 value
            + b"\xff"  # bogus tag
        )
        with pytest.raises(MicbParseError, match="value tag"):
            _bounded_call(parse_micb, payload)

    def test_unknown_dtype_byte(self) -> None:
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x00"  # 0 strings
            + b"\x00"  # 0 symbols
            + b"\x01"  # 1 type
            + b"\xff"  # bogus dtype byte
            + b"\x00"  # rank 0
        )
        with pytest.raises(MicbParseError, match="dtype"):
            _bounded_call(parse_micb, payload)

    def test_unknown_opcode_byte(self) -> None:
        # 1 type, 1 arg, then a node with an unknown opcode.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x01\x01X"  # 1 string "X"
            + b"\x00"  # 0 symbols
            + b"\x01\x01\x00"  # 1 type: f32, rank 0
            + b"\x02"  # 2 values
            + b"\x00\x00\x00"  # value 0: Arg name=0 type=0
            + b"\x02"  # value 1: Node tag
            + b"\xff"  # unknown opcode byte
        )
        with pytest.raises(MicbParseError, match="opcode"):
            _bounded_call(parse_micb, payload)


# ---------------------------------------------------------------------------
# Forward-reference / out-of-range
# ---------------------------------------------------------------------------


class TestOutOfRange:
    def test_arg_string_index_out_of_range(self) -> None:
        # 1 value (Arg) referencing string index 99 with 0 strings present.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x00"  # 0 strings
            + b"\x00"  # 0 symbols
            + b"\x01\x01\x00"  # 1 type: f32, rank 0
            + b"\x01"  # 1 value
            + b"\x00"  # tag Arg
            + b"\x63"  # name str_idx = 99 (uleb128)
            + b"\x00"  # type_idx = 0
        )
        with pytest.raises(MicbParseError, match="str_idx"):
            _bounded_call(parse_micb, payload)

    def test_arg_type_index_out_of_range(self) -> None:
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x01\x01X"  # 1 string "X"
            + b"\x00"  # 0 symbols
            + b"\x00"  # 0 types
            + b"\x01"  # 1 value
            + b"\x00"  # tag Arg
            + b"\x00"  # name str_idx = 0
            + b"\x05"  # type_idx = 5 (no types defined)
        )
        with pytest.raises(MicbParseError, match="type_idx"):
            _bounded_call(parse_micb, payload)

    def test_output_index_out_of_range(self) -> None:
        # Tail uleb128 = output index. Value table empty → output 0
        # is invalid.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x00"  # 0 strings
            + b"\x00"  # 0 symbols
            + b"\x00"  # 0 types
            + b"\x00"  # 0 values
            + b"\x00"  # output = 0 (no values present)
        )
        with pytest.raises(MicbParseError, match="output"):
            _bounded_call(parse_micb, payload)


# ---------------------------------------------------------------------------
# Text-mode adversarial inputs
# ---------------------------------------------------------------------------


class TestTextDoS:
    def test_input_above_10mb_rejected(self) -> None:
        with pytest.raises(Mic2ParseError, match="exceeds"):
            _bounded_call(parse_mic2, "X" * (MAX_INPUT_BYTES + 1))

    def test_too_many_lines_rejected(self) -> None:
        # Just over the line cap — synthesise the smallest payload
        # that hits the cap. Empty lines count.
        payload = "\n" * (MAX_LINE_COUNT + 1)
        with pytest.raises(Mic2ParseError, match="exceeds"):
            _bounded_call(parse_mic2, payload)

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(Mic2ParseError, match="header"):
            _bounded_call(parse_mic2, "")

    def test_only_comments_rejected(self) -> None:
        text = "# nothing but comments\n# more comments\n"
        with pytest.raises(Mic2ParseError, match="header"):
            _bounded_call(parse_mic2, text)


# ---------------------------------------------------------------------------
# Empty / zero-element edge cases
# ---------------------------------------------------------------------------


class TestEmptyAndZero:
    def test_zero_value_graph_invalid_output(self) -> None:
        # A graph with no values is meaningless — output index can't
        # point at anything.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x00\x00\x00\x00"  # 0 strings, 0 symbols, 0 types, 0 values
            + b"\x00"  # output 0
        )
        with pytest.raises(MicbParseError, match="output"):
            _bounded_call(parse_micb, payload)

    def test_invalid_utf8_in_string_table(self) -> None:
        # \xff is never valid as the first byte of a UTF-8 sequence.
        payload = (
            MICB_MAGIC
            + b"\x02"
            + b"\x01"  # 1 string
            + b"\x02"  # length 2
            + b"\xff\xff"  # invalid UTF-8
        )
        with pytest.raises(MicbParseError, match="UTF-8"):
            _bounded_call(parse_micb, payload)
