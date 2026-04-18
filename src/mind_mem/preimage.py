# Copyright 2026 STARGA, Inc.
"""Versioned, NUL-separated hash preimages (v2.10.0).

Before v2.10.0 several hash-preimage builders concatenated field
values with ad-hoc separators (often none). A field value containing
the separator character — or two adjacent fields where the boundary
isn't unambiguous — could craft collisions without touching the
logical payload.

This module ships a single canonical builder:

    preimage("EV_v1", evidence_id, timestamp, action, actor, ...)

Rules:

1. The first slot is a **version tag** — mandatory, ascii. Different
   tags for different preimage classes (``EV_v1``, ``AUDIT_v1``, ...)
   prevent cross-class collision even when bodies coincide.
2. Every subsequent field is rendered to ``bytes`` deterministically:
   - ``bytes``   → as-is
   - ``str``     → ``utf-8`` encode
   - ``int``     → canonical decimal ascii (no leading zero, signed)
   - ``float``   → Q16.16 hex via :mod:`q1616` (cross-arch stable)
   - ``None``    → empty bytes (rendered as ``\\x00\\x00``; the leading
                   NUL is the separator, the trailing NUL from the
                   *next* separator closes the slot)
   - anything else raises ``TypeError``
3. Fields are joined with a single NUL byte (``\\x00``).
4. Field values containing ``\\x00`` are rejected with ``ValueError``
   so the separator is unambiguous.

Consumers use :func:`preimage` to build the bytes and then hash with
whatever digest they prefer (SHA-256, SHA3-512, etc.).
"""

from __future__ import annotations

from typing import Union

from .q1616 import hex_q16_16

_SEP = b"\x00"
Value = Union[bytes, bytearray, memoryview, str, int, float, None]


def _render(value: Value) -> bytes:
    """Canonical byte form of a single preimage field.

    ``None`` is intentionally rejected rather than rendered as empty
    bytes — otherwise ``None`` and ``""`` hash identically (both empty
    slots), which lets an attacker swap ``target_file=""`` for
    ``target_file=None`` in a JSON payload without invalidating the
    digest. Callers that want "absent" must pass ``""`` explicitly.
    """
    if value is None:
        raise ValueError('preimage(): None is not a valid field value — convert to "" or a sentinel string first')
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, str):
        data = value.encode("utf-8")
    elif isinstance(value, bool):
        # bool is-a int in Python; handle before int branch so
        # True/False render as the canonical "t"/"f" — DISTINCT from
        # int 1/0 rendering so a bool↔int swap invalidates the hash.
        data = b"t" if value else b"f"
    elif isinstance(value, int):
        data = str(int(value)).encode("ascii")
    elif isinstance(value, float):
        if value != value:  # NaN — would otherwise Q16.16 → 0, which
            # collides with float 0.0. Reject upfront so callers hand
            # us finite floats (or explicitly coerce NaN to a sentinel).
            raise ValueError("preimage(): NaN is not hashable — convert to a sentinel float first")
        data = hex_q16_16(value).encode("ascii")
    else:
        raise TypeError(f"preimage(): unsupported field type {type(value).__name__!r}")
    if _SEP in data:
        raise ValueError("preimage(): field value contains NUL byte; cannot build a NUL-separated preimage around it")
    return data


def preimage(tag: str, *fields: Value) -> bytes:
    """Build a tagged, NUL-separated hash preimage.

    The returned bytes always start with ``tag + \\x00`` and every
    field is followed by a trailing ``\\x00`` so boundaries are
    unambiguous even when a field is empty.

    Args:
        tag: Fixed-string version tag (e.g. ``EV_v1``). Must not contain
             NUL.
        *fields: Values for each preimage slot, in order.

    Raises:
        ValueError: The tag is empty/contains NUL, or a field contains NUL.
        TypeError:  A field is of an unsupported type.
    """
    if not tag:
        raise ValueError("preimage(): tag must be non-empty")
    tag_bytes = tag.encode("ascii")
    if _SEP in tag_bytes:
        raise ValueError("preimage(): tag must not contain NUL")
    parts = [tag_bytes]
    for f in fields:
        parts.append(_render(f))
    # Trailing separator closes the final slot.
    return _SEP.join(parts) + _SEP


__all__ = ["preimage"]
