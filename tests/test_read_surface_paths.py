# Copyright 2026 STARGA, Inc.
"""Every raw corpus read in the tool layer, structurally — the other instrument.

``test_read_surface_admission.py`` sweeps behaviour: it calls each registered
tool with a canary corpus and asserts the withheld blocks never come back. That
is the strong check, and it has one blind spot it cannot close from the inside —
**it only sees the argument sets somebody wrote down**. A branch no invocation
reaches is a branch the canary never enters, and a read surface added inside one
would be green from the day it shipped.

So this file asks a different question, over the source rather than the output:
*which functions in the tool layer read blocks straight out of the corpus, and
what makes each of them safe?* Every call to a raw block reader must sit in a
function that also calls an admission seam, or be named in :data:`ALLOWLIST`
with the reason it does not need one. A new raw read fails the build until
somebody answers that question — which is the same contract
``test_read_surface_classification`` imposes on a new tool, one layer down.

The two instruments are deliberately independent and neither subsumes the other:

* the sweep catches a leak the source reads as safe (a helper that admits on one
  branch and not another),
* this catches a leak the sweep never reaches (a parameter combination or a
  backend nobody swept).

Pure AST, no ``mind_mem`` import, so the invariant is checked against the source
on disk rather than against whatever a runtime happens to expose — the same
reason ``tests/_write_path_scan.py`` is written that way for the write side.

**Scope, stated rather than implied:** ``src/mind_mem/mcp/tools`` — the tool
layer. A helper in another module that reads raw and is called from here is out
of this scanner's reach; the behavioural sweep is what covers that today, and
widening this scanner to the whole package is the follow-on.
"""

from __future__ import annotations

import ast
import os

import pytest

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mind_mem", "mcp", "tools")


# ---------------------------------------------------------------------------
# The two vocabularies
# ---------------------------------------------------------------------------

#: Calls that hand back block dicts with no admission decision attached. Every
#: one of these returns pending and quarantined blocks along with the servable
#: ones, which is correct for the function itself -- ``parse_file`` parsing
#: everything is the whole point -- and is exactly why the CALLER has to decide.
RAW_READERS: frozenset[str] = frozenset(
    {
        "parse_file",  # block_parser: every block in a corpus file
        "parse_blocks",  # block_parser: every block in a string
        "parse_text",  # block_parser: same, other spelling
        "get_active",  # block_parser: 'active' is narrower than admissible, not equal to it
        "iter_blocks",  # storage: every block in the workspace
        "load_blocks",  # storage: same, eager
        "get_all",  # BlockStore.get_all(active_only=False)
        "read_block",  # BlockStore single-block read
        "list_blocks",  # BlockStore enumeration
        "query_index",  # sqlite_index: rows whose cached status goes stale fail-OPEN
        "fts_query",  # the name query_index is imported under in recall.py
    }
)

#: The admission seams. A function that calls one of these has taken the egress
#: decision rather than skipped it. ``iter_active_blocks`` is here because it is
#: an admitting READER: it yields the active set only, so it cannot emit a
#: withheld block in the first place.
ADMITTERS: frozenset[str] = frozenset(
    {
        "admit_read",
        "admit_read_one",
        "admit_corpus",
        "admit_leg",
        "admissible",
        "_withhold_inadmissible",
        "iter_active_blocks",
    }
)


# ---------------------------------------------------------------------------
# The committed allowlist. A raw read with no admission, and why.
# ---------------------------------------------------------------------------

#: qualname -> the reason this function may read raw. Every entry is a claim
#: somebody made on purpose; the behavioural sweep is what keeps them honest,
#: because each of these functions is reached by at least one swept tool.
ALLOWLIST: dict[str, str] = {
    "governance._recent_statements": (
        "Write path, not a read surface: it builds the 24h dedupe window the quality gate compares a "
        "PROPOSED statement against. The verdict reports a similarity ratio and never the matched text "
        "(quality_gate._near_duplicate -> _fail('near_duplicate', f'... ratio={ratio:.3f} ...')), so no "
        "byte of a withheld block reaches the proposer."
    ),
    "governance.scan": (
        "Aggregate counts only -- totals, active counts, drift and signal tallies. It reads the corpus "
        "wide on purpose: a governance scan that could not see quarantined blocks could not report that "
        "they need review. Its servable enumeration goes through iter_active_blocks."
    ),
    "memory_ops._store_block_is_active": "A predicate over a block the caller already holds; returns a bool and no content.",
    "memory_ops.index_stats": "Counts indexed vs. corpus blocks. Numbers, never block text or ids.",
    "memory_ops.memory_health": "Health tallies -- block counts, drift counts, signal counts, index freshness. Numbers, never block text.",
    "memory_ops._resolve_block_for_read": (
        "Resolution only: it says whether the bytes EXIST and where, and hands them to its caller, which "
        "applies admit_read_one. Deliberately split that way so there is ONE egress decision in get_block "
        "rather than one at each of the three resolution sites. Pinned as a RESOLVER below, so every "
        "in-module caller is checked for that admission."
    ),
}

#: Allowlisted helpers that return raw blocks to a caller. For these the reason
#: is "my caller admits", which is only true if every caller actually does --
#: so it is checked rather than asserted in prose.
RESOLVERS: frozenset[str] = frozenset({"memory_ops._resolve_block_for_read"})


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def scan_source(source: str, module: str) -> dict[str, dict[str, list]]:
    """Per-function raw reads and admission calls in one module's source.

    Returns ``{qualname: {"raw": [(name, lineno)], "admit": [name], "calls": [name]}}``.
    Takes source text, not a path, so the tests below can point it at a fixture
    and prove the scanner can actually see a leak.
    """
    tree = ast.parse(source, filename=f"{module}.py")
    found: dict[str, dict[str, list]] = {}

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, stack + [child.name])
                continue
            if isinstance(child, ast.Call):
                name = _called_name(child)
                qual = f"{module}." + ".".join(stack) if stack else f"{module}.<module>"
                record = found.setdefault(qual, {"raw": [], "admit": [], "calls": []})
                if name is not None:
                    record["calls"].append(name)
                if name in RAW_READERS:
                    record["raw"].append((name, child.lineno))
                elif name in ADMITTERS:
                    record["admit"].append(name)
            visit(child, stack)

    visit(tree, [])
    return found


def scan_tool_layer() -> dict[str, dict[str, list]]:
    """The whole tool layer, module by module, deterministic order."""
    merged: dict[str, dict[str, list]] = {}
    for filename in sorted(os.listdir(TOOLS_DIR)):
        if not filename.endswith(".py"):
            continue
        with open(os.path.join(TOOLS_DIR, filename), encoding="utf-8") as handle:
            merged.update(scan_source(handle.read(), filename[:-3]))
    return merged


def unadmitted(scan: dict[str, dict[str, list]]) -> dict[str, list]:
    """Functions that read raw with no admission seam anywhere up their nesting.

    An enclosing function's admission counts: a closure inside a function that
    already admitted is covered by that decision.
    """
    admitting = {qual for qual, record in scan.items() if record["admit"]}
    out: dict[str, list] = {}
    for qual, record in scan.items():
        if not record["raw"]:
            continue
        if any(qual == other or qual.startswith(other + ".") for other in admitting):
            continue
        out[qual] = record["raw"]
    return out


# ---------------------------------------------------------------------------
# Non-vacuity — the scan has to have found something
# ---------------------------------------------------------------------------


def test_the_scan_actually_reaches_the_tool_layer() -> None:
    """A scanner that matched nothing would pass every assertion below.

    Absence of a finding is only evidence when the search ran, so the shape of
    the corpus it searched is asserted before anything is concluded from it.
    """
    scan = scan_tool_layer()
    raw_sites = sum(len(record["raw"]) for record in scan.values())
    admitting = {qual for qual, record in scan.items() if record["admit"]}
    modules = {qual.split(".", 1)[0] for qual in scan}
    assert raw_sites >= 20, f"only {raw_sites} raw read sites found; the scanner's reader vocabulary has probably gone stale"
    assert len(admitting) >= 5, f"only {len(admitting)} admitting functions found: {sorted(admitting)}"
    assert len(modules) >= 15, f"the scan reached only {len(modules)} tool modules: {sorted(modules)}"


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


def test_every_raw_read_is_admitted_or_allowlisted() -> None:
    """A raw corpus read either admits, or is a committed decision with a reason."""
    offenders = {qual: sites for qual, sites in unadmitted(scan_tool_layer()).items() if qual not in ALLOWLIST}
    assert not offenders, (
        f"functions in the tool layer that read blocks with no admission seam and no ALLOWLIST entry: {offenders}. "
        f"Route the rows through mind_mem.admission.admit_read (or admit_read_one), or add the function to "
        f"ALLOWLIST with the reason its raw read cannot serve withheld content."
    )


def test_the_allowlist_names_no_function_that_vanished() -> None:
    """Both directions, so a renamed function cannot linger as a stale excuse."""
    scan = scan_tool_layer()
    ghosts = sorted(qual for qual in ALLOWLIST if qual not in scan)
    assert not ghosts, f"ALLOWLIST names functions that no longer exist: {ghosts}"
    no_longer_raw = sorted(qual for qual in ALLOWLIST if not scan[qual]["raw"])
    assert not no_longer_raw, f"ALLOWLIST excuses functions that no longer read raw; delete the row: {no_longer_raw}"


def test_every_allowlist_entry_states_a_reason() -> None:
    thin = sorted(qual for qual, reason in ALLOWLIST.items() if len(reason.strip()) < 40)
    assert not thin, f"ALLOWLIST entries with no real reason: {thin}"


def test_every_resolver_has_only_admitting_callers() -> None:
    """A resolver's "my caller admits" is a claim about the call graph, so check the graph.

    ``_resolve_block_for_read`` returns an unadmitted block on purpose -- the
    single egress decision lives in ``get_block``. That is safe exactly as long
    as no second caller appears which forgets it, which is the "imported is not
    wired" failure read backwards: a helper is only as governed as its callers.
    """
    scan = scan_tool_layer()
    violations: dict[str, list[str]] = {}
    for resolver in sorted(RESOLVERS):
        module, _, short = resolver.partition(".")
        callers = [qual for qual, record in scan.items() if qual.startswith(module + ".") and qual != resolver and short in record["calls"]]
        assert callers, f"{resolver} has no caller in {module}; the resolver rule would be vacuous"
        ungoverned = [qual for qual in callers if not scan[qual]["admit"]]
        if ungoverned:
            violations[resolver] = sorted(ungoverned)
    assert not violations, f"resolvers whose callers do not apply an admission seam: {violations}"


# ---------------------------------------------------------------------------
# Tests-of-the-test — the scanner has to be able to see both answers
# ---------------------------------------------------------------------------

_LEAK_FIXTURE = '''
def serve_everything(ws):
    """A tool handler that reads the corpus and returns it raw."""
    blocks = parse_file(ws + "/decisions/DECISIONS.md")
    return json.dumps(blocks)
'''

_ADMITTED_FIXTURE = """
def serve_admitted(ws):
    blocks = parse_file(ws + "/decisions/DECISIONS.md")
    decision = admit_read(blocks, workspace=ws, surface="fixture")
    return json.dumps(decision.admitted)
"""


def test_the_scanner_flags_a_raw_read() -> None:
    """Positive control: the exact shape get_block had must be caught."""
    flagged = unadmitted(scan_source(_LEAK_FIXTURE, "fixture"))
    assert "fixture.serve_everything" in flagged, f"the scanner did not see the leak fixture: {flagged}"
    assert flagged["fixture.serve_everything"] == [("parse_file", 4)]


def test_the_scanner_clears_an_admitted_read() -> None:
    """Negative control: routing through the seam must actually clear it.

    Without this the scanner could be flagging every function unconditionally
    and the test above would still be green.
    """
    assert not unadmitted(scan_source(_ADMITTED_FIXTURE, "fixture"))


def test_the_tripwire_fails_on_an_unallowlisted_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: an unclassified raw read reaches the assertion as a failure."""
    import sys

    real = scan_tool_layer()
    monkeypatch.setattr(
        sys.modules[__name__],
        "scan_tool_layer",
        lambda: {**real, **scan_source(_LEAK_FIXTURE, "fixture")},
    )
    with pytest.raises(AssertionError, match="serve_everything"):
        test_every_raw_read_is_admitted_or_allowlisted()


def test_the_resolver_rule_fails_on_an_ungoverned_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    real = scan_tool_layer()
    poisoned = dict(real)
    poisoned["memory_ops.some_new_tool"] = {"raw": [], "admit": [], "calls": ["_resolve_block_for_read"]}
    monkeypatch.setattr(sys.modules[__name__], "scan_tool_layer", lambda: poisoned)
    with pytest.raises(AssertionError, match="some_new_tool"):
        test_every_resolver_has_only_admitting_callers()
