# Copyright 2026 STARGA, Inc.
"""The governance receipt_ts gate must anchor at end-of-string, not end-of-line.

`$` matches before a trailing newline; `\\Z` does not. apply_engine.py already
uses `\\Z` for the identical value one frame later, so this was an inconsistency
between two gates on the same input rather than a style preference.

It matters because on the gRPC route the MCP tool is called as
``fn(**request.args)`` with no Pydantic model and no length pin, which makes this
regex the FIRST gate the value meets rather than the second.
"""

import re

import pytest


def _gate_pattern():
    from mind_mem.mcp.tools import governance

    src = __import__("inspect").getsource(governance)
    m = re.search(r're\.match\(r"(\^\\d\{8\}-\\d\{6\}(?:\$|\\Z))"', src)
    assert m, "the receipt_ts gate regex was not found; this test is anchored on it"
    return m.group(1)


class TestTheGateAnchorsAtEndOfString:
    def test_a_trailing_newline_is_rejected(self):
        assert re.match(_gate_pattern(), "20260906-123456\n") is None, "a trailing newline slipped through the receipt_ts gate"

    def test_the_legitimate_value_still_passes(self):
        """Positive control: the rejection above is about the newline."""
        assert re.match(_gate_pattern(), "20260906-123456") is not None

    @pytest.mark.parametrize("bad", ["2026090-123456", "20260906-12345", "20260906_123456", ""])
    def test_malformed_values_are_still_rejected(self, bad):
        assert re.match(_gate_pattern(), bad) is None

    def test_it_agrees_with_the_engine_side_gate(self):
        """The two gates on the same value must not disagree."""
        from mind_mem import apply_engine

        engine_src = __import__("inspect").getsource(apply_engine)
        assert r'r"^\d{8}-\d{6}\Z"' in engine_src, "engine-side anchor changed; re-check both"
        assert _gate_pattern().endswith(r"\Z"), "the MCP gate is looser than the engine gate"
