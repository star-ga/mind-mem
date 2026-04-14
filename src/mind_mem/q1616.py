# Copyright 2026 STARGA, Inc.
"""Q16.16 fixed-point helpers for audit-hash determinism (v2.10.0).

Audit-chain preimages hash confidence / importance / weight values into
their hash bytes. If those values are Python floats, the byte-level
serialization (repr, struct.pack, whatever the caller picked) is not
stable across architectures: a chain produced on x86_64 and replayed
on aarch64 can disagree even when the logical values are identical.

Q16.16 — signed 32-bit integer, 16 bits fractional — side-steps the
problem:

- Exact representation of any multiple of ``2**-16`` in ``[-32767.5,
  +32767.5]``. Plenty of headroom for scores in ``[0.0, 1.0]``.
- Byte-identical on every architecture.
- Fits in ``int32`` → canonical 8-char hex form feeds cleanly into
  SHA2/SHA3 preimages.

Round-trip error is at most ``2**-17`` so re-decoding produces a float
that is statistically indistinguishable from the original within one
ULP of the original Python float.
"""
from __future__ import annotations

_SCALE: int = 1 << 16
_MAX: int = 0x7FFFFFFF
_MIN: int = -0x80000000


def to_q16_16(value: float) -> int:
    """Convert *value* to a signed Q16.16 integer with saturation.

    >>> hex(to_q16_16(0.5))
    '0x8000'
    >>> hex(to_q16_16(1.0))
    '0x10000'
    >>> to_q16_16(-0.5)
    -32768
    >>> to_q16_16(float('inf'))
    2147483647
    """
    f = float(value)
    if f != f:  # NaN
        return 0
    if f == float("inf"):
        return _MAX
    if f == float("-inf"):
        return _MIN
    scaled = int(round(f * _SCALE))
    if scaled > _MAX:
        return _MAX
    if scaled < _MIN:
        return _MIN
    return scaled


def from_q16_16(value: int) -> float:
    """Inverse of :func:`to_q16_16`.

    >>> from_q16_16(to_q16_16(0.25))
    0.25
    >>> from_q16_16(to_q16_16(0.9999))
    0.9999237060546875
    """
    return int(value) / _SCALE


def hex_q16_16(value: float) -> str:
    """Return the canonical 8-char lowercase hex encoding for *value*.

    Used inside hash preimages so the byte form is stable:

    >>> hex_q16_16(0.5)
    '00008000'
    >>> hex_q16_16(-0.5)
    'ffff8000'
    """
    q = to_q16_16(value)
    return f"{q & 0xFFFFFFFF:08x}"


__all__ = ["to_q16_16", "from_q16_16", "hex_q16_16"]
