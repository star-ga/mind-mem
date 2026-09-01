# Copyright 2026 STARGA, Inc.
"""Regression tests for the doc tool-count gate (scripts/count_mcp_tools.py).

The gate exists because doc claims drift from the shipped surface. These
tests exist because the GATE ITSELF drifted: it reported success over
files it had silently excused.
"""

from __future__ import annotations


class TestVersionExemptionIsScopedToTheClaim:
    """A version stamp must not silence a claim it does not describe.

    `--check-docs` reported "all tool-count claims agree with 96" while TWO
    live docs said 89. The exemption for historical claims ("19 MCP tools at
    v1.x") was applied per LINE, so this sentence was excused entirely:

        "MIND-Mem exposes 89 MCP tools for integration with ... Zed (v3.1.0+)."

    The `v3.1.0+` qualifies Zed's editor support and sits 137 characters from
    the count. A gate that reports success over a file it silently excused is
    worse than no gate, so proximity is now required.
    """

    def test_a_distant_version_does_not_exempt_a_live_claim(self):

        from scripts.count_mcp_tools import _CLAIM_RE, _version_qualifies

        line = "MIND-Mem exposes 89 MCP tools for integration with editors like Zed (v3.1.0+)."
        m = _CLAIM_RE.search(line)
        assert m is not None, "the claim regex must still match the sentence"
        assert not _version_qualifies(line, m), "a version 137 chars away must not exempt the claim"

    def test_an_adjacent_version_still_exempts_a_historical_claim(self):
        from scripts.count_mcp_tools import _CLAIM_RE, _version_qualifies

        line = "Shipped 19 MCP tools at v1.x, before the governance surface landed."
        m = _CLAIM_RE.search(line)
        assert m is not None
        assert _version_qualifies(line, m), "an adjacent version must still mark the claim historical"

    def test_the_live_docs_are_clean_under_the_narrowed_gate(self):
        """The narrowed gate found two stale docs; they must stay fixed."""
        from scripts.count_mcp_tools import check_docs, count_tools

        assert check_docs(count_tools()) == []


class TestAModifierDoesNotHideACount:
    """"89 distinct tools" is a claim, and the gate used to miss it.

    `_CLAIM_RE` requires the count and the noun to be adjacent (or separated
    only by "MCP"), so a single modifier hid real drift: README,
    `docs/api-reference.md` and `docs/claude-desktop-setup.md` all said
    "89 distinct tools / 90 registrations" long after the surface reached 98 --
    README four lines under its own "### MCP Server (98 tools, 8 resources)"
    heading -- and `--check-docs` reported agreement.
    """

    def test_the_distinct_spelling_is_a_gated_claim(self):
        from scripts.count_mcp_tools import _DISTINCT_RE

        m = _DISTINCT_RE.search("server with 89 distinct tools and 8 read-only resources")
        assert m is not None, "the modifier spelling must be gated"
        assert m.group(1) == "89"

    def test_the_widening_stays_narrow(self):
        """Any-modifier matching would flag "17 AI development tools", which
        counts editors, not MCP tools. Only "distinct" is admitted."""
        from scripts.count_mcp_tools import _DISTINCT_RE

        assert _DISTINCT_RE.search("wired into 17 AI development tools") is None
