# Copyright 2026 STARGA, Inc.
"""Tests for q1616.py + preimage.py (v2.10.0 audit-integrity helpers)."""

from __future__ import annotations

import pytest

from mind_mem.preimage import preimage
from mind_mem.q1616 import from_q16_16, hex_q16_16, to_q16_16

# ---------------------------------------------------------------------------
# Q16.16
# ---------------------------------------------------------------------------


class TestQ1616:
    def test_zero(self) -> None:
        assert to_q16_16(0.0) == 0
        assert from_q16_16(0) == 0.0

    def test_exact_binary_halves(self) -> None:
        # Halves are exactly representable in Q16.16.
        for f in (0.0, 0.5, 0.25, 0.125, 0.0625):
            assert from_q16_16(to_q16_16(f)) == f

    def test_unit_one(self) -> None:
        # 1.0 → 0x10000
        assert to_q16_16(1.0) == 1 << 16
        assert from_q16_16(1 << 16) == 1.0

    def test_negative(self) -> None:
        assert to_q16_16(-0.5) == -(1 << 15)
        assert from_q16_16(-(1 << 15)) == -0.5

    def test_saturates_to_max(self) -> None:
        # +inf saturates to 0x7fffffff.
        assert to_q16_16(float("inf")) == 0x7FFFFFFF
        # A value well past the max also saturates.
        assert to_q16_16(1e10) == 0x7FFFFFFF

    def test_saturates_to_min(self) -> None:
        assert to_q16_16(float("-inf")) == -0x80000000
        assert to_q16_16(-1e10) == -0x80000000

    def test_nan_is_zero(self) -> None:
        # NaN is coerced to 0 — deterministic, avoids poison in preimages.
        assert to_q16_16(float("nan")) == 0

    def test_hex_canonical(self) -> None:
        # Canonical 8-char lowercase hex, two's complement for negatives.
        assert hex_q16_16(0.0) == "00000000"
        assert hex_q16_16(0.5) == "00008000"
        assert hex_q16_16(1.0) == "00010000"
        assert hex_q16_16(-0.5) == "ffff8000"
        assert hex_q16_16(-1.0) == "ffff0000"

    def test_round_trip_uniformity(self) -> None:
        # Every 1/65536 step round-trips exactly.
        for k in range(-100, 100):
            f = k / (1 << 16)
            assert from_q16_16(to_q16_16(f)) == f

    def test_cross_architecture_determinism(self) -> None:
        # Unlike repr(0.9999), the hex encoding is byte-identical on
        # every architecture. We assert the byte sequence matches a
        # baseline.
        baseline = {0.1: "0000199a", 0.9: "0000e666", 1.5: "00018000"}
        for value, expected in baseline.items():
            assert hex_q16_16(value) == expected


# ---------------------------------------------------------------------------
# preimage
# ---------------------------------------------------------------------------


class TestPreimage:
    def test_empty_tag_rejected(self) -> None:
        with pytest.raises(ValueError):
            preimage("", "a", "b")

    def test_nul_in_tag_rejected(self) -> None:
        with pytest.raises(ValueError):
            preimage("bad\x00tag", "a")

    def test_nul_in_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            preimage("T_v1", "hello\x00world")

    def test_unsupported_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            preimage("T_v1", object())

    def test_string_field(self) -> None:
        # tag \x00 field \x00
        assert preimage("T_v1", "hello") == b"T_v1\x00hello\x00"

    def test_bytes_field(self) -> None:
        assert preimage("T_v1", b"hello") == b"T_v1\x00hello\x00"

    def test_int_field(self) -> None:
        # int → canonical decimal ascii
        assert preimage("T_v1", 42) == b"T_v1\x0042\x00"
        assert preimage("T_v1", -7) == b"T_v1\x00-7\x00"

    def test_float_field_q16_16(self) -> None:
        # float → Q16.16 hex, no repr() noise
        assert preimage("T_v1", 0.5) == b"T_v1\x0000008000\x00"
        assert preimage("T_v1", -0.5) == b"T_v1\x00ffff8000\x00"

    def test_none_rejected(self) -> None:
        # None rejected — would otherwise collide with empty string.
        with pytest.raises(ValueError):
            preimage("T_v1", None, "x")

    def test_nan_rejected(self) -> None:
        # NaN rejected — would otherwise Q16.16-collide with 0.0.
        with pytest.raises(ValueError):
            preimage("T_v1", float("nan"))

    def test_bool_vs_int_distinct(self) -> None:
        # bool rendered "t"/"f" — distinct from int 1/0 so a
        # bool↔int swap invalidates the digest.
        assert preimage("T_v1", True, False) == b"T_v1\x00t\x00f\x00"
        assert preimage("T_v1", 1, 0) == b"T_v1\x001\x000\x00"
        assert preimage("T_v1", True) != preimage("T_v1", 1)
        assert preimage("T_v1", False) != preimage("T_v1", 0)

    def test_collision_resistance_against_pipe_separator(self) -> None:
        # Classic attack: field A = "a|b", field B = "c" collides with
        # A = "a", B = "b|c" under a "|"-separator scheme. NUL-sep
        # rejects both because fields can't contain NUL.
        # Positive case: distinct field values → distinct preimages.
        p1 = preimage("T_v1", "a", "b|c")
        p2 = preimage("T_v1", "a|b", "c")
        assert p1 != p2

    def test_version_tag_isolates_domains(self) -> None:
        # Same field values under different tags → different preimages.
        assert preimage("EV_v1", "x", "y") != preimage("AUDIT_v1", "x", "y")

    def test_trailing_separator_closes_final_slot(self) -> None:
        # preimage() always ends in \x00 so a bare suffix is unambiguous.
        assert preimage("T_v1", "a").endswith(b"\x00")

    def test_empty_string_vs_absent_distinguished(self) -> None:
        # "" renders to an empty slot; None is rejected. This means
        # `preimage(tag, "")` and `preimage(tag, None)` can't both
        # exist — closing the None/"" collision hole Gemini flagged.
        p_empty = preimage("T_v1", "")
        assert p_empty == b"T_v1\x00\x00"
        with pytest.raises(ValueError):
            preimage("T_v1", None)
