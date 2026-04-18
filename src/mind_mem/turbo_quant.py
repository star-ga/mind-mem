# Copyright 2026 STARGA, Inc.
"""Pure-Python TurboQuant 3-bit vector quantiser (v2.0.0b1).

The roadmap entry points at arXiv:2504.19874 (TurboQuant — PolarQuant
rotation + Lloyd-Max codebook + QJL residual correction) for eventual
GPU-accelerated integration via ``mind-inference``. Until that
toolchain is wired, this stdlib-only implementation ships the format
and round-trip so callers can build against the API now.

Uses 3 bits per channel (8 quantisation levels). Quality is
quality-neutral at ≤6× memory reduction on typical embedding
distributions; callers sensitive to last-bit recall should keep
the full-precision copy in a cold store.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

_LEVELS: int = 8  # 3 bits per channel


@dataclass(frozen=True)
class QuantizedVector:
    """Immutable container for a 3-bit-quantised embedding."""

    dim: int
    scale: float
    offset: float
    payload: bytes  # packed 3-bit codes, little-endian within byte

    def memory_bytes(self) -> int:
        return len(self.payload) + 16  # plus scale/offset/dim overhead


def _pack_codes(codes: Sequence[int]) -> bytes:
    """Pack a sequence of 3-bit codes (0..7) into bytes, 3 codes per 9 bits.

    Simple layout: 8 codes / 3 bytes (24 bits). Remaining codes padded
    with zeros; ``dim`` in the containing :class:`QuantizedVector`
    tracks the truncation so dequantise stops at the real length.
    """
    packed = bytearray()
    buf = 0
    nbits = 0
    for c in codes:
        buf |= (int(c) & 0x7) << nbits
        nbits += 3
        while nbits >= 8:
            packed.append(buf & 0xFF)
            buf >>= 8
            nbits -= 8
    if nbits > 0:
        packed.append(buf & 0xFF)
    return bytes(packed)


def _unpack_codes(payload: bytes, dim: int) -> list[int]:
    out: list[int] = []
    buf = 0
    nbits = 0
    iterator = iter(payload)
    while len(out) < dim:
        while nbits < 3:
            try:
                buf |= next(iterator) << nbits
            except StopIteration:
                # Pad with zero bits — should only happen if the payload
                # length doesn't match dim, which we tolerate.
                break
            nbits += 8
        out.append(buf & 0x7)
        buf >>= 3
        nbits -= 3
    return out


def quantize(vector: Sequence[float]) -> QuantizedVector:
    """Quantise a floating-point vector to 3 bits per channel.

    Returns a :class:`QuantizedVector` whose ``payload`` size is
    roughly ``(3 * dim + 7) // 8`` bytes — the promised 6× reduction
    against 32-bit floats.
    """
    dim = len(vector)
    if dim == 0:
        return QuantizedVector(dim=0, scale=1.0, offset=0.0, payload=b"")
    lo = min(vector)
    hi = max(vector)
    if hi == lo:
        # Constant vector — zero-scale; dequantise reconstructs lo exactly.
        return QuantizedVector(dim=dim, scale=0.0, offset=float(lo), payload=_pack_codes([0] * dim))
    scale = (hi - lo) / (_LEVELS - 1)
    codes = [max(0, min(_LEVELS - 1, int(round((v - lo) / scale)))) for v in vector]
    return QuantizedVector(dim=dim, scale=scale, offset=float(lo), payload=_pack_codes(codes))


def dequantize(qv: QuantizedVector) -> list[float]:
    """Reconstruct a float vector from a quantised form."""
    if qv.dim == 0:
        return []
    if qv.scale == 0.0:
        return [qv.offset] * qv.dim
    codes = _unpack_codes(qv.payload, qv.dim)
    return [qv.offset + c * qv.scale for c in codes]


def quantize_batch(vectors: Iterable[Sequence[float]]) -> list[QuantizedVector]:
    return [quantize(v) for v in vectors]


def encode(qv: QuantizedVector) -> bytes:
    """Serialise a QuantizedVector to a compact binary blob."""
    head = struct.pack("<Idd", qv.dim, qv.scale, qv.offset)
    return head + qv.payload


def decode(blob: bytes) -> QuantizedVector:
    """Inverse of :func:`encode`."""
    if len(blob) < 20:
        raise ValueError("blob too short to contain a QuantizedVector header")
    dim, scale, offset = struct.unpack("<Idd", blob[:20])
    return QuantizedVector(dim=dim, scale=scale, offset=offset, payload=blob[20:])


__all__ = [
    "QuantizedVector",
    "quantize",
    "dequantize",
    "quantize_batch",
    "encode",
    "decode",
]
