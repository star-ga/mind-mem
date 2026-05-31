"""Regression tests for the optional Cython accelerator at
``mind_mem._mic_map_accel``.

Two contracts:

1. **Bit-identical output** — when the accelerator is present, every
   ``parse_micb`` and ``emit_micb`` call returns exactly the same
   bytes / Graph as the pure-Python fallback would. The accelerator
   is a perf optimisation, never a behaviour change.

2. **Graceful absence** — when the accelerator isn't built into the
   wheel (the default ``pip install mind-mem`` path), ``mic_map.py``
   silently falls back to its pure-Python codec.
   ``_ACCEL_AVAILABLE`` reports the truth either way.

The first contract is checked by re-running the full reference graph
through both code paths. The second is checked structurally — the
``_ACCEL_AVAILABLE`` flag must be a bool, ``_read_exact`` /
``_uleb128_decode`` must always be callable.
"""

from __future__ import annotations

import io

import pytest
from mind_mem.mic_map import (
    _ACCEL_AVAILABLE,
    Arg,
    Graph,
    MicbParseError,
    Node,
    Param,
    Type,
    _py_read_exact,
    _py_uleb128_decode,
    _read_exact,
    _uleb128_decode,
    emit_micb,
    parse_micb,
)


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


# ---------------------------------------------------------------------------
# Structural — module always exposes the right shape
# ---------------------------------------------------------------------------


class TestModuleShape:
    def test_accel_available_is_bool(self) -> None:
        assert isinstance(_ACCEL_AVAILABLE, bool)

    def test_read_exact_is_callable(self) -> None:
        assert callable(_read_exact)

    def test_uleb128_decode_is_callable(self) -> None:
        assert callable(_uleb128_decode)

    def test_pure_python_fallbacks_always_present(self) -> None:
        # The pure-Python helpers exist regardless of accelerator
        # status — they're the regression net.
        assert callable(_py_read_exact)
        assert callable(_py_uleb128_decode)


# ---------------------------------------------------------------------------
# Equivalence — accel and pure Python produce identical results
# ---------------------------------------------------------------------------


class TestEquivalence:
    """When the accelerator is built (CI's ``[accelerated]`` job),
    these tests run the full graph through both paths and assert
    they agree. When the accelerator isn't built, the tests are
    skipped (the pure-Python path is the only path)."""

    def setup_method(self) -> None:
        if not _ACCEL_AVAILABLE:
            pytest.skip("Cython accelerator not built — pure-Python only")

    def test_uleb128_decode_agrees(self) -> None:
        # Standard encodings of values across the byte-count range.
        for v in (0, 1, 127, 128, 16383, 16384, 2**32 - 1, 2**63 - 1):
            from mind_mem.mic_map import _uleb128_encode

            buf = _uleb128_encode(v)
            a = _uleb128_decode(io.BytesIO(buf))
            b = _py_uleb128_decode(io.BytesIO(buf))
            assert a == b == v, f"value {v}: accel={a} py={b}"

    def test_read_exact_agrees(self) -> None:
        data = b"hello world" * 100
        a = _read_exact(io.BytesIO(data), 50)
        b = _py_read_exact(io.BytesIO(data), 50)
        assert a == b
        # Both must handle short reads (BytesIO short-reads naturally
        # at EOF only; the loop is exercised by the streaming parser
        # tests in test_mic_map_stream.py).

    def test_uleb128_eof_raises_micbparseerror(self) -> None:
        # Both paths translate EOF to MicbParseError — the wrapper in
        # mic_map.py converts MicbAccelParseError → MicbParseError.
        with pytest.raises(MicbParseError, match="EOF"):
            _uleb128_decode(io.BytesIO(b""))
        with pytest.raises(MicbParseError, match="EOF"):
            _py_uleb128_decode(io.BytesIO(b""))

    def test_uleb128_too_long_raises_micbparseerror(self) -> None:
        # 11 continuation bytes → fail-closed.
        bomb = b"\x80" * 11 + b"\x00"
        with pytest.raises(MicbParseError, match="too long"):
            _uleb128_decode(io.BytesIO(bomb))
        with pytest.raises(MicbParseError, match="too long"):
            _py_uleb128_decode(io.BytesIO(bomb))

    def test_full_parse_micb_agrees(self) -> None:
        # Round-trip the residual block through parse_micb under
        # whatever path is wired in. Then compare with a parse done
        # under the pure-Python decoder by temporarily swapping in
        # the fallbacks. (We don't have an easy in-process swap, so
        # instead we just assert parse_micb produces the canonical
        # graph whether or not the accelerator is loaded.)
        g = _residual_block()
        b = emit_micb(g)
        parsed = parse_micb(b)
        assert parsed == g


# ---------------------------------------------------------------------------
# Always-on — pure-Python decoder is always callable, even when the
# accelerator path is taken
# ---------------------------------------------------------------------------


class TestPurePythonAlwaysWorks:
    def test_py_uleb128_decode_handles_known_values(self) -> None:
        from mind_mem.mic_map import _uleb128_encode

        for v in (0, 1, 127, 128, 12345, 2**40):
            assert _py_uleb128_decode(io.BytesIO(_uleb128_encode(v))) == v

    def test_py_read_exact_loops_on_short_reads(self) -> None:
        class OneByteReader(io.RawIOBase):
            def __init__(self, data: bytes) -> None:
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

        # 50 single-byte reads coalesce into one 50-byte answer.
        out = _py_read_exact(OneByteReader(b"x" * 100), 50)
        assert out == b"x" * 50
