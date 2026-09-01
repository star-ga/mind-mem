# Copyright 2026 STARGA, Inc.
"""``error_codes`` on the MCP boundary, via ``error_envelope``.

The module shipped a full code taxonomy and never had a caller. It is wired
ADDITIVELY, which is the whole constraint: ``error`` keeps the exact string
it always carried, so no existing reader changes, and ``code`` is new
information beside it. A client that wants to branch on a failure should not
have to pattern-match English prose that changes whenever someone improves
the wording.

The typed exceptions in ``mind_mem.errors`` are untouched — they remain the
in-process contract. This is only the wire form, for the boundary an
exception cannot cross.
"""

from __future__ import annotations

import ast
import json
import pathlib

from mind_mem.error_codes import ErrorCode, error_category, error_severity
from mind_mem.mcp.tools._helpers import error_envelope


class TestEnvelopeIsAdditive:
    def test_without_a_code_the_envelope_is_exactly_what_it_always_was(self) -> None:
        """The no-code path must not gain a single key.

        This is the compatibility guarantee: every call site not yet
        classified keeps emitting today's bytes.
        """
        assert json.loads(error_envelope("boom")) == {"error": "boom"}

    def test_with_a_code_the_message_is_untouched(self) -> None:
        out = json.loads(error_envelope("boom", ErrorCode.RECALL_QUERY_TOO_LONG))
        assert out["error"] == "boom", "the human message must survive verbatim"
        assert out["code"] == "MM-4003"
        assert out["error_category"] == error_category(ErrorCode.RECALL_QUERY_TOO_LONG).value
        assert out["error_severity"] == error_severity(ErrorCode.RECALL_QUERY_TOO_LONG).value

    def test_extra_fields_survive(self) -> None:
        out = json.loads(error_envelope("boom", None, hint="try fewer tokens"))
        assert out["hint"] == "try fewer tokens"


class TestRecallUsesIt:
    def test_the_call_site_is_a_call_not_an_import(self) -> None:
        """Parsed, not grepped — an unused import is the fake-wiring case."""
        tree = ast.parse(pathlib.Path("src/mind_mem/mcp/tools/recall.py").read_text(encoding="utf-8"))
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "error_envelope" in called

    def test_an_over_long_query_is_refused_with_its_code(self) -> None:
        """End-to-end through the shipped tool, not the helper in isolation."""
        from mind_mem.mcp.tools.recall import _MAX_QUERY_LEN, _recall_impl

        out = json.loads(_recall_impl("x" * (_MAX_QUERY_LEN + 1)))
        assert out["code"] == "MM-4003"
        assert "characters" in out["error"]

    def test_positive_control_a_normal_query_is_not_an_error(self) -> None:
        """Without this, the assertion above could pass because EVERY call
        returns an error envelope."""
        from mind_mem.mcp.tools.recall import _MAX_QUERY_LEN

        assert _MAX_QUERY_LEN > 10, "fixture assumption"
        # A short query must not be refused by the length guard; whatever
        # else it returns, it must not carry the too-long code.
        from mind_mem.mcp.tools.recall import _recall_impl

        out = json.loads(_recall_impl("short query"))
        assert out.get("code") != "MM-4003"
