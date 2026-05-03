# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated hot paths for ``mind_mem.mic_map``.

This module is **strictly optional** — ``mic_map.py`` falls back to its
pure-Python implementation when this extension isn't built. The build
is opt-in via ``pip install mind-mem[accelerated]`` (which pulls in
Cython at build time) or by installing from sdist with Cython already
present.

The hot paths picked for Cythonisation are the ones the v3.8.8
benchmark identified as bottlenecks:

* ``uleb128_encode`` / ``uleb128_decode`` — called once per varint
  (and a typical mic-b payload has dozens of varints per value).
* ``sleb128_decode`` — same story.
* ``read_exact`` — short-read loop is hot when streaming over a
  socket or chunked source.

Public API exactly mirrors the underscore-prefixed pure-Python
helpers so ``mic_map.py`` can substitute either implementation
transparently. The wire-format invariants (minimum encoding, 70-bit
ULEB128 cap, MicbParseError on EOF) are preserved bit-identically —
this is a perf optimisation only, never a behaviour change.
"""

from libc.stdint cimport int64_t, uint8_t, uint64_t


class MicbAccelParseError(ValueError):
    """Raised by the Cython codec on EOF / minimum-encoding violation.

    Mirrors :class:`mind_mem.mic_map.MicbParseError` so ``mic_map.py``
    can catch one and translate to the other; we don't import the
    pure-Python class here to keep the extension dependency-free.
    """


def uleb128_encode(value):
    """Encode a non-negative Python int as ULEB128. Mirrors the
    pure-Python ``_uleb128_encode``."""
    cdef uint64_t v
    cdef bytearray out = bytearray()
    cdef uint8_t b

    if value < 0:
        raise ValueError("uleb128 only encodes non-negative values")
    if value > 0xFFFFFFFFFFFFFFFF:
        # Fall back to Python big-int path for >64-bit values.
        return _py_uleb128_encode(value)
    v = <uint64_t> value
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


cdef bytes _py_uleb128_encode(value):
    """Slow path for Python ints that overflow uint64_t."""
    cdef bytearray out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def uleb128_decode(buf):
    """Decode a ULEB128 from a binary stream, enforcing minimum
    encoding via the 70-bit shift cap. Mirrors the pure-Python
    ``_uleb128_decode``."""
    cdef uint64_t result = 0
    cdef int shift = 0
    cdef bytes ch
    cdef int byte

    while True:
        ch = read_exact(buf, 1)
        if not ch:
            raise MicbAccelParseError("unexpected EOF in uleb128")
        byte = ch[0]
        # For huge values shift can run past 64 bits — fall back to
        # pure-Python big-int once we've seen 9 continuation bytes.
        if shift >= 63:
            return _py_uleb128_continue(buf, result, shift, byte)
        result |= (<uint64_t> (byte & 0x7F)) << shift
        if (byte & 0x80) == 0:
            return result
        shift += 7
        if shift > 70:
            raise MicbAccelParseError("uleb128 too long (>10 bytes)")


cdef _py_uleb128_continue(buf, uint64_t low_result, int shift, int byte):
    """Continue decoding a ULEB128 in pure-Python big-int once we've
    overflowed uint64_t. Preserves the 70-bit cap."""
    cdef object result = int(low_result)
    cdef object byte_obj = byte
    while True:
        result |= (byte_obj & 0x7F) << shift
        if (byte_obj & 0x80) == 0:
            return result
        shift += 7
        if shift > 70:
            raise MicbAccelParseError("uleb128 too long (>10 bytes)")
        ch = read_exact(buf, 1)
        if not ch:
            raise MicbAccelParseError("unexpected EOF in uleb128")
        byte_obj = ch[0]


def sleb128_decode(buf):
    """Decode an SLEB128 (zigzag → ULEB128). Mirrors the pure-Python
    ``_sleb128_decode``."""
    z = uleb128_decode(buf)
    return (z >> 1) ^ -(z & 1)


def read_exact(buf, n):
    """Read exactly ``n`` bytes from ``buf``, looping on short reads.
    Mirrors the pure-Python ``_read_exact``."""
    cdef bytearray out = bytearray()
    cdef int needed = n
    while len(out) < needed:
        chunk = buf.read(needed - len(out))
        if not chunk:
            return bytes(out)
        out.extend(chunk)
    return bytes(out)
