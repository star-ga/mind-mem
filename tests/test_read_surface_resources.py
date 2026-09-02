# Copyright 2026 STARGA, Inc.
"""The OTHER read surface: MCP resources, enumerated and classified.

``test_read_surface_classification`` widened the tripwire from one module to the
whole **tool** registry. This file closes the axis that widening did not cover.

``mcp/server.py`` calls ``_resources.register(mcp)`` one line before it registers
any tool, and that call wires eight ``mind-mem://`` URIs onto the same server. A
client reads them exactly the way it calls a tool. They are not tools, so
``count_mcp_tools._tool_names`` -- which collects ``mcp.tool(fn)`` arguments --
cannot see one of them, and neither can any assertion built on it. Eight read
surfaces, zero tripwire coverage, for the same structural reason the recall-only
tripwire missed ``get_block``: the enumeration and the surface were different
sets, and nothing checked that.

**Measured, 2026-09-02.** Seeded a workspace with a quarantined canary in each
backing file and read all eight:

    get_decisions           withheld served: NO    (get_active -> active only)
    get_tasks               withheld served: YES
    get_entities(projects)  withheld served: YES
    get_signals             withheld served: YES
    get_contradictions      withheld served: YES
    get_recall              withheld served: NO    (recall admits)
    get_health              withheld served: NO    (counts)
    get_ledger              withheld served: NO    (chain rows)

So four resources serve quarantined block content verbatim -- the ``get_block``
defect, on the surface next door. ``src/mind_mem/mcp/resources.py`` is outside
this lane's write scope, so this file does not fix them; it makes them
**impossible to lose**:

* every registered resource must be classified, both directions, so a ninth
  resource fails the build until somebody says what it serves;
* the resources that DO admit are gated properly -- a canary that ever appears
  in one of them is a failure, today;
* the four that do not are pinned by EXACT EQUALITY, so the set cannot grow
  quietly, and the day one is fixed this file goes red and the row must be
  deleted. A ratchet that only tightens, in both directions.

Nothing here is an argument that the four are acceptable. They are a defect with
a name, a measurement and a test that will not let go of it.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

CONTENT = "content"
NO_CONTENT = "no-content"

#: Improbable tokens, one per backing file, so a hit names the file it came from.
ACTIVE_CANARIES: dict[str, str] = {
    "decisions": "zqxACTIVEcanary",
    "tasks": "zqxACTIVEtask",
    "projects": "zqxACTIVEproj",
}
WITHHELD_CANARIES: dict[str, str] = {
    "decisions": "zqxQUARANTINEcanary",
    "tasks": "zqxQUARANTINEtask",
    "projects": "zqxQUARANTINEproj",
    "signals": "zqxQUARANTINEsignal",
    "contradictions": "zqxQUARANTINEcontra",
}

#: rel path -> (block id, status, canary) rows seeded into it.
SEED: dict[str, tuple[tuple[str, str, str], ...]] = {
    "decisions/DECISIONS.md": (
        ("D-20260101-001", "active", ACTIVE_CANARIES["decisions"]),
        ("D-20260103-001", "quarantined", WITHHELD_CANARIES["decisions"]),
    ),
    "tasks/TASKS.md": (
        ("T-20260101-001", "active", ACTIVE_CANARIES["tasks"]),
        ("T-20260103-001", "quarantined", WITHHELD_CANARIES["tasks"]),
    ),
    "entities/projects.md": (
        ("P-20260101-001", "active", ACTIVE_CANARIES["projects"]),
        ("P-20260103-001", "quarantined", WITHHELD_CANARIES["projects"]),
    ),
    "intelligence/SIGNALS.md": (("S-20260103-001", "quarantined", WITHHELD_CANARIES["signals"]),),
    "intelligence/CONTRADICTIONS.md": (("C-20260103-001", "quarantined", WITHHELD_CANARIES["contradictions"]),),
}

#: function name -> verdict. Resources are classified on the same two-state
#: vocabulary the tool table uses, and for the same reason: a surface nobody
#: classified is a surface nobody checked.
CLASSIFICATION: dict[str, str] = {
    "get_decisions": CONTENT,
    "get_tasks": CONTENT,
    "get_entities": CONTENT,
    "get_signals": CONTENT,
    "get_contradictions": CONTENT,
    "get_recall": CONTENT,
    "get_health": NO_CONTENT,
    "get_ledger": NO_CONTENT,
}

#: How the sweep reads each resource, and which withheld canary it would leak.
INVOCATIONS: dict[str, tuple[tuple, str | None, str | None]] = {
    # name: (args, withheld canary to look for, active canary that proves reach)
    "get_decisions": ((), WITHHELD_CANARIES["decisions"], ACTIVE_CANARIES["decisions"]),
    "get_tasks": ((), WITHHELD_CANARIES["tasks"], ACTIVE_CANARIES["tasks"]),
    "get_entities": (("projects",), WITHHELD_CANARIES["projects"], ACTIVE_CANARIES["projects"]),
    "get_signals": ((), WITHHELD_CANARIES["signals"], None),
    "get_contradictions": ((), WITHHELD_CANARIES["contradictions"], None),
    "get_recall": (("architecture",), WITHHELD_CANARIES["decisions"], ACTIVE_CANARIES["decisions"]),
    "get_health": ((), WITHHELD_CANARIES["decisions"], None),
    "get_ledger": ((), WITHHELD_CANARIES["decisions"], None),
}

#: The four that serve withheld block content today, with the fix each needs.
#: Pinned by exact equality below. This is a defect register, not an allowlist:
#: an entry here is a known leak somebody must close, and closing one breaks
#: this file until the row is removed.
KNOWN_UNADMITTED: dict[str, str] = {
    "get_tasks": (
        "parse_file(TASKS.md) returned verbatim. Fix: route the parsed blocks through admission.admit_read before _blocks_to_json."
    ),
    "get_entities": ("_read_file('entities/<type>.md') returns the raw Markdown, every block in it. Fix: parse, admit_read, re-render."),
    "get_signals": (
        "_read_file('intelligence/SIGNALS.md') verbatim. Signals are pending BY DESIGN, so the fix is a review surface "
        "with a scope check, not a silent filter."
    ),
    "get_contradictions": "_read_file('intelligence/CONTRADICTIONS.md') verbatim. Same shape as signals.",
}


# ---------------------------------------------------------------------------
# Discovery — the registry, not a hand-written list
# ---------------------------------------------------------------------------


def registered_resources() -> dict[str, str]:
    """``{function name: uri template}`` for every registered MCP resource.

    Read off a live ``register(_Probe())`` -- the same trick the tool tripwire
    uses to cross-check its static discovery. ``resources.py`` has exactly one
    ``register`` and no decorators, so there is no static/live gap to bridge.
    """
    from mind_mem.mcp import resources

    found: dict[str, str] = {}

    class _Probe:
        def resource(self, uri):  # noqa: ANN001, ANN202 - FastMCP's shape
            def bind(fn):  # noqa: ANN001, ANN202
                found[getattr(fn, "__name__", str(fn))] = uri
                return fn

            return bind

    resources.register(_Probe())
    return found


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------


def _render(block_id: str, status: str, canary: str) -> str:
    return (
        f"[{block_id}]\n"
        f"Date: 2026-01-01\n"
        f"Status: {status}\n"
        f"Scope: global\n"
        f"Statement: The {canary} architecture decision governs frost telemetry.\n"
        f"Rationale: {canary} rationale for the frost telemetry architecture rollout.\n"
        f"Tags: architecture, frost\n"
        f"Sources: -\n"
        f"Supersedes: -\n\n"
    )


@pytest.fixture(scope="module")
def workspace() -> str:
    from mind_mem import sqlite_index
    from mind_mem.init_workspace import init

    ws = tempfile.mkdtemp(prefix="mm_resources_")
    shutil.rmtree(ws)
    os.makedirs(ws)
    init(ws)
    for rel, rows in SEED.items():
        path = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n" + "".join(_render(bid, status, canary) for bid, status, canary in rows))
    sqlite_index.build_index(ws)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _read(name: str, workspace: str) -> str:
    from mind_mem.mcp import resources
    from mind_mem.mcp.infra.workspace import use_workspace

    args = INVOCATIONS[name][0]
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    os.environ["MIND_MEM_CONFIG"] = os.path.join(workspace, "mind-mem.json")
    with use_workspace(workspace):
        try:
            result = getattr(resources, name)(*args)
        except Exception as exc:  # noqa: BLE001 - a raising resource is still an answer
            return f"{type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else json.dumps(result, default=str)


@pytest.fixture(scope="module")
def served(workspace: str) -> dict[str, str]:
    return {name: _read(name, workspace) for name in sorted(INVOCATIONS)}


# ---------------------------------------------------------------------------
# Positive controls — before any conclusion is drawn from an absence
# ---------------------------------------------------------------------------


def test_the_withheld_canaries_are_really_in_the_backing_files(workspace: str) -> None:
    """Every "canary absent" assertion below is vacuous without this."""
    from mind_mem.block_parser import parse_file

    for rel, rows in SEED.items():
        path = os.path.join(workspace, rel)
        assert os.path.isfile(path), f"seed failed: {rel} was not written"
        text = open(path, encoding="utf-8").read()
        for block_id, status, canary in rows:
            assert canary in text, f"seed failed: {canary} is not in {rel}"
        parsed = {b["_id"]: b.get("Status") for b in parse_file(path) if b.get("_id")}
        for block_id, status, _canary in rows:
            assert parsed.get(block_id) == status, f"seed failed: {block_id} parsed as {parsed.get(block_id)!r}, wanted {status!r}"


def test_every_content_resource_actually_reaches_the_corpus(served: dict[str, str]) -> None:
    """A resource that served nothing would pass the leak check for free.

    Only the three with an active canary can prove reach positively; the two
    review files hold withheld blocks only, by their nature, so their reach is
    proven by the leak measurement itself.
    """
    for name, (_args, _withheld, active) in sorted(INVOCATIONS.items()):
        if active is None:
            continue
        assert active in served[name], f"{name} did not serve the ACTIVE canary; its leak result would mean nothing: {served[name][:200]}"


# ---------------------------------------------------------------------------
# Tripwire — the registry and the table must agree, both ways
# ---------------------------------------------------------------------------


def test_every_registered_resource_is_classified() -> None:
    registry = registered_resources()
    unclassified = sorted(set(registry) - set(CLASSIFICATION))
    assert not unclassified, (
        f"registered MCP resources with no read-surface classification: {unclassified}. "
        f"Classify each as '{CONTENT}' or '{NO_CONTENT}' and give it an INVOCATIONS row so the canary sweep reads it."
    )
    ghosts = sorted(set(CLASSIFICATION) - set(registry))
    assert not ghosts, f"CLASSIFICATION names resources that are not registered: {ghosts}"
    unswept = sorted(set(registry) - set(INVOCATIONS))
    assert not unswept, f"registered resources the sweep never reads: {unswept}"


def test_the_resource_registry_is_not_visible_to_the_tool_tripwire() -> None:
    """Why this file exists, asserted rather than explained.

    If the tool discovery ever DID see these, two enumerations would cover the
    same surface and this file could be folded into the other one. Until then
    the gap is real and the assertion documents it.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from count_mcp_tools import _tool_names, _tool_source_files

    tools: set[str] = set()
    for path in _tool_source_files():
        tools.update(_tool_names(path))
    overlap = sorted(set(registered_resources()) & tools)
    assert not overlap, f"resources now visible to the tool tripwire: {overlap} -- fold this file into the tool classification"


# ---------------------------------------------------------------------------
# The gate — admitted resources must not leak, and the defect set cannot grow
# ---------------------------------------------------------------------------


def measured_leaks(served: dict[str, str]) -> set[str]:
    return {name for name, (_args, withheld, _active) in INVOCATIONS.items() if withheld and withheld in served[name]}


def test_no_admitted_resource_serves_withheld_content(served: dict[str, str]) -> None:
    """The live gate: every resource NOT in the defect register must be clean."""
    leaking = sorted(measured_leaks(served) - set(KNOWN_UNADMITTED))
    assert not leaking, (
        f"MCP resources serving quarantined block content with no entry in KNOWN_UNADMITTED: {leaking}. "
        f"Route them through mind_mem.admission.admit_read."
    )


def test_the_unadmitted_resource_set_is_exactly_what_is_pinned(served: dict[str, str]) -> None:
    """A ratchet, both ways.

    Grew: a new leak appeared and must be fixed, not recorded.
    Shrank: a leak was FIXED -- delete its row, and the register gets smaller by
    one. Either way somebody looks. A defect register that silently disagrees
    with the code is how the recall-only tripwire stayed green for two leaks.
    """
    measured = measured_leaks(served)
    assert measured == set(KNOWN_UNADMITTED), (
        f"the pinned set of unadmitted resources no longer matches measurement. "
        f"newly leaking: {sorted(measured - set(KNOWN_UNADMITTED))}; "
        f"no longer leaking (delete the row): {sorted(set(KNOWN_UNADMITTED) - measured)}"
    )


def test_every_pinned_defect_names_its_fix() -> None:
    thin = sorted(name for name, fix in KNOWN_UNADMITTED.items() if len(fix.strip()) < 40)
    assert not thin, f"KNOWN_UNADMITTED entries with no fix described: {thin}"


def test_the_admitting_resources_are_admitting_for_a_reason(served: dict[str, str]) -> None:
    """The clean four are clean by construction, and the source says how.

    ``get_decisions`` filters to active, ``get_recall`` goes through the recall
    engine, and the two summaries return numbers. Asserting the mechanism as
    well as the outcome means a refactor that keeps the canary out by accident
    (an empty result, a swallowed exception) does not read as admission.
    """
    import inspect

    from mind_mem.mcp import resources

    assert "get_active(" in inspect.getsource(resources.get_decisions)
    recall_source = inspect.getsource(resources.get_recall)
    assert "recall_engine(" in recall_source or "fts_query(" in recall_source
    assert ACTIVE_CANARIES["decisions"] not in served["get_health"], "get_health started serving block text; reclassify it"
    assert ACTIVE_CANARIES["decisions"] not in served["get_ledger"], "get_ledger started serving block text; reclassify it"


# ---------------------------------------------------------------------------
# Tests-of-the-test
# ---------------------------------------------------------------------------


def test_the_tripwire_fails_on_an_unclassified_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    real = registered_resources()
    monkeypatch.setattr(sys.modules[__name__], "registered_resources", lambda: {**real, "get_secrets": "mind-mem://secrets"})
    with pytest.raises(AssertionError, match="get_secrets"):
        test_every_registered_resource_is_classified()


def test_the_gate_fails_when_a_leak_is_not_pinned(served: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``get_tasks`` from the register: the live gate must catch it."""
    import sys

    monkeypatch.setattr(sys.modules[__name__], "KNOWN_UNADMITTED", {k: v for k, v in KNOWN_UNADMITTED.items() if k != "get_tasks"})
    with pytest.raises(AssertionError, match="get_tasks"):
        test_no_admitted_resource_serves_withheld_content(served)


def test_the_ratchet_fails_when_a_pinned_defect_is_fixed(served: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The direction that matters when somebody does the work.

    A register that quietly kept a fixed row would let the next reader believe
    a closed defect is still open.
    """
    import sys

    monkeypatch.setattr(sys.modules[__name__], "KNOWN_UNADMITTED", {**KNOWN_UNADMITTED, "get_health": "a defect that is not real"})
    with pytest.raises(AssertionError, match="get_health"):
        test_the_unadmitted_resource_set_is_exactly_what_is_pinned(served)
