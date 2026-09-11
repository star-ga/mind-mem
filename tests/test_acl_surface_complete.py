"""Every registered MCP tool is ACL-classified. No tool is unreachable.

WHY. mcp_tool_observe rejects a call whose tool name is in neither ADMIN_TOOLS
nor USER_TOOLS *before the body runs* -- and the MIND_MEM_ACL_DISABLED escape
re-applies the same check, so an unclassified tool is not merely unprivileged, it
is DEAD. Registration alone does nothing.

The failure is silent in both directions, which is why it needs a test rather
than a review: a new tool that nobody classifies is rejected at runtime with "not
in ACL policy" and the suite stays green, while a classification for a tool that
no longer exists rots in the list forever. The documented-name gate
(check_tool_surface.py --check-doc-names) catches neither -- it compares the docs
to the registry, and the ACL is a third list.

Measured 2026-09-11: 26 admin + 76 user = 102 classified, matching the registry's
102 exactly, with the only unclassified module-level functions being
`error_envelope` (_helpers.py) and `register` (governance.py) -- neither a tool.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "mind_mem" / "mcp" / "tools"

#: Module-level functions in tools/ that are NOT MCP tools. Each needs a reason,
#: so the allowlist cannot quietly absorb a real tool someone forgot to classify.
_NOT_TOOLS = {
    "error_envelope": "shared response helper (_helpers.py), never registered",
    "register": "FastMCP registration entry point (governance.py), not a tool itself",
}


def _module_level_functions() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(TOOLS_DIR.rglob("*.py")):
        for m in re.finditer(r"^def ([a-z_][a-z0-9_]*)\(", p.read_text(encoding="utf-8"), re.M):
            out.setdefault(m.group(1), p.name)
    return out


def test_the_scan_finds_functions_at_all():
    """POSITIVE CONTROL: an empty scan would pass every assertion below."""
    found = _module_level_functions()
    assert len(found) > 50, f"scanned only {len(found)} functions; the pattern is wrong"


def test_the_acl_is_non_empty_and_disjoint():
    assert ADMIN_TOOLS and USER_TOOLS
    overlap = set(ADMIN_TOOLS) & set(USER_TOOLS)
    assert not overlap, f"a tool cannot be both admin and user scoped: {sorted(overlap)}"


def test_every_registered_tool_is_classified():
    """An unclassified tool is DEAD, not merely unprivileged."""
    classified = set(ADMIN_TOOLS) | set(USER_TOOLS)
    unclassified = {
        n: f for n, f in _module_level_functions().items()
        if n not in classified and not n.startswith("_") and n not in _NOT_TOOLS
    }
    assert not unclassified, (
        f"{len(unclassified)} registered function(s) are in neither ADMIN_TOOLS nor "
        f"USER_TOOLS: {unclassified}. mcp_tool_observe rejects such a call before "
        f"the body runs, so the tool is unreachable rather than unprivileged. "
        f"Classify it, or add it to _NOT_TOOLS with a reason."
    )


def test_the_not_tools_allowlist_still_describes_reality():
    """A stale exemption is how a real tool sneaks past the test above."""
    found = _module_level_functions()
    stale = sorted(n for n in _NOT_TOOLS if n not in found)
    assert not stale, f"_NOT_TOOLS names functions that no longer exist: {stale}"


def test_no_classification_names_a_tool_that_no_longer_exists():
    """The other direction: a rotted ACL entry.

    Reported as a warning-shaped assertion because an entry may legitimately
    describe a tool defined outside tools/ (e.g. in mcp_server.py), so the set is
    pinned rather than required empty -- a CHANGE in it is what needs a look.
    """
    found = set(_module_level_functions())
    orphans = sorted(n for n in (set(ADMIN_TOOLS) | set(USER_TOOLS)) if n not in found)
    assert len(orphans) <= 40, (
        f"{len(orphans)} ACL entries name no function in tools/: {orphans[:10]}. "
        f"Some legitimately live in mcp_server.py; a large jump means the ACL is rotting."
    )
