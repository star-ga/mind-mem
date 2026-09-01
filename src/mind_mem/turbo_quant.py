# Copyright 2026 STARGA, Inc.
"""Pure-Python 3-bit scalar vector quantiser (v2.0.0b1).

**What this actually is:** a per-vector uniform min-max scalar quantiser
at 3 bits per channel (8 levels), stdlib-only, with the dimension, scale
and offset carried in the blob header. It is fully self-describing and
deterministic on round-trip: a blob decodes with no external state.

That self-description is the whole point, and it is what separates this
from ``v4/pq``. Product quantisation trains KMeans codebooks and reaches
far higher compression, but a PQ code is meaningless without the codebook
that produced it. So the two are different layers, not rivals:

* ``v4/pq``      -- the quantiser on the RETRIEVAL path (ANN distance).
* ``turbo_quant`` -- the embedding blob codec for SELF-CONTAINED artifacts:
  export bundles, cold-tier storage, federation transport. Anywhere a
  vector must decode without shipping a codebook beside it.

**Invariant: this module never touches the recall path.** Two quantisers
on one retrieval path would be a fork; keeping this one at the codec
layer is what stops that.

Roughly 6x memory reduction on typical embedding distributions. Callers
sensitive to last-bit recall should keep the full-precision copy in a
cold store.

deferred: the name is aspirational -- the format is a placeholder for a
rotation + learned-codebook + residual-correction scheme (see the private
research notes, not cited here). Upgrade path: keep the header, swap the
channel encoder, bump the format byte.
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
