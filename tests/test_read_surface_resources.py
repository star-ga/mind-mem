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

**Measured, 2026-09-02, before the fix.** Seeded a workspace with a quarantined
canary in each backing file and read all eight:

    get_decisions           withheld served: NO    (get_active -> active only)
    get_tasks               withheld served: YES
    get_entities(projects)  withheld served: YES
    get_signals             withheld served: YES
    get_contradictions      withheld served: YES
    get_recall              withheld served: NO    (recall admits)
    get_health              withheld served: NO    (counts)
    get_ledger              withheld served: YES   <- not in the first count

``get_ledger`` is the one this file got wrong, and the way it got it wrong is
worth keeping. It was classified ``no-content`` on the strength of the
*decisions* canary never appearing in it. That is true of every file except
one, so it measured nothing about the ledger:
``namespaces.SharedLedger.append_fact`` writes ``[FACT-...]`` blocks with a
free-text ``Text:`` field and a ``Status:`` line, and ``_read_file`` served the
file whole. Seeded with a quarantined ``FACT`` block it served it. **A negative
assertion aimed at the wrong file is not a negative result** -- every resource
now carries a canary from its OWN backing file, and an active canary as the
positive control that it still reaches the corpus at all.

**Measured 2026-09-02, after the fix:** all eight read ``NO``, all six
content resources still read their own active canary ``YES``.

The five leaks are closed in ``src/mind_mem/mcp/resources.py``: the raw file
reader is gone from that module, every corpus row a resource returns comes
through ``_admitted_corpus`` -> ``admission.admit_read``, and signals -- pending
BY DESIGN, on a surface with no scope check to review them behind -- answer with
the admitted subset plus the withheld count, naming the ACL-classified tools as
the review path.

What this file holds:

* every registered resource must be classified, both directions, so a ninth
  resource fails the build until somebody says what it serves;
* every content resource is swept with its own canary pair -- a withheld block
  appearing in one is a failure, today;
* :data:`KNOWN_UNADMITTED` is EMPTY and pinned by exact equality, so the defect
  register cannot grow back quietly;
* the mechanism is asserted, not just the outcome, so a refactor that keeps the
  canary out by accident (an empty result, a swallowed exception) does not read
  as admission.
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
    # The three review corpora hold withheld blocks as their normal content, so
    # before the fix their reach was proven by the leak itself. With the leak
    # closed that proof is gone, and an admitted block is the only thing that
    # can replace it: each of these is a status the gate SERVES, seeded into a
    # file whose usual content it withholds.
    "signals": "zqxAPPLIEDsignal",
    "contradictions": "zqxOPENcontra",
    "ledger": "zqxREVIEWfact",
}
#: Admissible and NOT active. ``get_decisions`` narrows past it and admission
#: does not, which is what makes it the control for telling the two apart.
SUPERSEDED_CANARY: str = "zqxSUPERSEDEDcanary"

WITHHELD_CANARIES: dict[str, str] = {
    "decisions": "zqxQUARANTINEcanary",
    "tasks": "zqxQUARANTINEtask",
    "projects": "zqxQUARANTINEproj",
    "signals": "zqxPENDINGsignal",
    "contradictions": "zqxQUARANTINEcontra",
    "ledger": "zqxQUARANTINEfact",
}

#: rel path -> (block id, status, canary) rows seeded into it.
SEED: dict[str, tuple[tuple[str, str, str], ...]] = {
    "decisions/DECISIONS.md": (
        ("D-20260101-001", "active", ACTIVE_CANARIES["decisions"]),
        ("D-20260102-001", "superseded", SUPERSEDED_CANARY),
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
    # ``pending`` rather than ``quarantined``: it is what ``capture`` actually
    # writes, and it is withheld for the other reason -- INITIAL_STATUS mints it
    # for AUTO_CAPTURE, so admissibility.UNADMITTED carries it.
    "intelligence/SIGNALS.md": (
        ("SIG-20260101-001", "applied", ACTIVE_CANARIES["signals"]),
        ("SIG-20260103-001", "pending", WITHHELD_CANARIES["signals"]),
    ),
    "intelligence/CONTRADICTIONS.md": (
        ("C-20260101-001", "open", ACTIVE_CANARIES["contradictions"]),
        ("C-20260103-001", "quarantined", WITHHELD_CANARIES["contradictions"]),
    ),
    # The ledger is corpus content too -- see the module docstring.
    "shared/intelligence/LEDGER.md": (
        ("FACT-20260101-001", "pending-review", ACTIVE_CANARIES["ledger"]),
        ("FACT-20260103-001", "quarantined", WITHHELD_CANARIES["ledger"]),
    ),
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
    # Reclassified 2026-09-02: measured serving a quarantined FACT block. The
    # old ``no-content`` row was read off the wrong canary.
    "get_ledger": CONTENT,
}

#: How the sweep reads each resource, and which withheld canary it would leak.
INVOCATIONS: dict[str, tuple[tuple, str | None, str | None]] = {
    # name: (args, withheld canary to look for, active canary that proves reach)
    "get_decisions": ((), WITHHELD_CANARIES["decisions"], ACTIVE_CANARIES["decisions"]),
    "get_tasks": ((), WITHHELD_CANARIES["tasks"], ACTIVE_CANARIES["tasks"]),
    "get_entities": (("projects",), WITHHELD_CANARIES["projects"], ACTIVE_CANARIES["projects"]),
    "get_signals": ((), WITHHELD_CANARIES["signals"], ACTIVE_CANARIES["signals"]),
    "get_contradictions": ((), WITHHELD_CANARIES["contradictions"], ACTIVE_CANARIES["contradictions"]),
    "get_recall": (("architecture",), WITHHELD_CANARIES["decisions"], ACTIVE_CANARIES["decisions"]),
    "get_health": ((), WITHHELD_CANARIES["decisions"], None),
    "get_ledger": ((), WITHHELD_CANARIES["ledger"], ACTIVE_CANARIES["ledger"]),
}

#: Resources that serve withheld block content, with the fix each needs.
#: **Empty, and pinned by exact equality below.** The five rows that were here
#: (get_tasks, get_entities, get_signals, get_contradictions, and get_ledger,
#: which was never written down because it was measured against the wrong
#: canary) were closed in ``mcp/resources.py`` on 2026-09-02.
#:
#: A defect register, never an allowlist: an entry means a known leak somebody
#: must close, adding one is a decision a reviewer sees, and the exact-equality
#: ratchet below fails the build both when a leak appears and when a pinned one
#: is fixed without deleting its row.
KNOWN_UNADMITTED: dict[str, str] = {}


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


#: Prose ``init`` writes above the blocks. Nothing parses it, so a resource
#: that serves it is serving FILE BYTES rather than admitted blocks -- which is
#: what four of these did. ``init`` writes the first four itself;
#: ``init_multi_agent_workspace`` writes the ledger's, and the fixture seeds it
#: because a plain ``init`` leaves the ledger absent.
FILE_PREAMBLE: dict[str, str] = {
    "tasks/TASKS.md": "Schema: [T-YYYYMMDD-###]",
    "entities/projects.md": "Schema: PRJ-slug",
    "intelligence/SIGNALS.md": "Auto-generated by capture.py",
    "intelligence/CONTRADICTIONS.md": "Contradiction log: conflicting decisions",
    "shared/intelligence/LEDGER.md": "Cross-agent facts pending review",
}

#: rel path -> what the fixture writes when ``init`` did not create the file.
SEED_HEADER: dict[str, str] = {
    "shared/intelligence/LEDGER.md": "# Shared Fact Ledger\n\nCross-agent facts pending review.\n\n",
}


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
        if not os.path.isfile(path) and rel in SEED_HEADER:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(SEED_HEADER[rel])
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


def test_the_seeded_canaries_are_really_in_the_backing_files(workspace: str) -> None:
    """Every "canary absent" assertion below is vacuous without this.

    Both directions: the withheld canaries must be on disk with the status
    that withholds them, and the active ones with a status the gate serves,
    or "reaches the corpus" would be measuring the seed rather than the code.
    """
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

    Before the fix the review corpora (signals, contradictions, ledger) had no
    active canary and their reach was proven by the leak itself -- which is
    precisely the proof the fix destroys. Each now seeds a block the gate
    SERVES alongside the one it withholds, so "the canary is absent" keeps
    meaning "admission dropped it" rather than "the resource read nothing".
    """
    unproven = sorted(name for name, verdict in CLASSIFICATION.items() if verdict == CONTENT and INVOCATIONS[name][2] is None)
    assert not unproven, f"content resources with no ACTIVE canary -- their leak result would be unfalsifiable: {unproven}"
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
    """The live gate: every resource NOT in the defect register must be clean.

    The register is empty, so this reads: no registered MCP resource serves a
    quarantined or pending block. ``test_the_gate_fails_when_a_resource_
    regresses_to_serving_the_file`` is the proof it can still fail.
    """
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


def test_every_content_resource_reaches_the_corpus_through_the_admission_seam(served: dict[str, str]) -> None:
    """The mechanism, not just the outcome.

    A refactor that keeps the canary out by accident -- an empty result, a
    swallowed exception, a filter somebody wrote a second time -- would pass
    the sweep above and be exactly the drift the seam exists to prevent. So
    every content resource must be seen calling it: ``_admitted_corpus`` for
    the five that read a corpus file, the recall engine for the one that does
    not. ``get_decisions`` additionally narrows to active, which is a
    *caller's* filter and never the governance decision.
    """
    import inspect

    from mind_mem.mcp import resources

    for name, verdict in sorted(CLASSIFICATION.items()):
        if verdict != CONTENT or name == "get_recall":
            continue
        source = inspect.getsource(getattr(resources, name))
        assert "_admitted_corpus(" in source, f"{name} returns corpus content without going through the admission seam:\n{source}"

    recall_source = inspect.getsource(resources.get_recall)
    assert "recall_engine(" in recall_source or "fts_query(" in recall_source
    assert ACTIVE_CANARIES["decisions"] not in served["get_health"], "get_health started serving block text; reclassify it"


def test_the_decisions_narrowing_is_a_caller_filter_and_not_the_gate(workspace: str, served: dict[str, str]) -> None:
    """``get_active`` narrows; it must not be mistaken for the admission gate.

    The two answers differ on exactly one kind of block: ``superseded`` is
    withheld by the narrowing and SERVED by admission (recall demotes decision
    history rather than hiding it -- ``admissibility`` says so at length). So
    the seeded superseded block is the control that tells them apart: admission
    admits it, this resource does not show it, and the day somebody replaces
    ``refine=get_active`` with "admission handles it" the block appears here
    and this test says which of the two moved.
    """
    from mind_mem.admission import admit_read
    from mind_mem.block_parser import parse_file

    blocks = parse_file(os.path.join(workspace, "decisions/DECISIONS.md"))
    seeded = {b.get("_id") for b in blocks}
    assert "D-20260102-001" in seeded, "positive control failed: the superseded block was never seeded"
    admitted = {b.get("_id") for b in admit_read(blocks, workspace=workspace, surface="test_control").admitted}
    assert "D-20260102-001" in admitted, (
        "positive control failed: admission withholds the superseded block, so it cannot measure the narrowing"
    )

    assert SUPERSEDED_CANARY not in served["get_decisions"], "get_decisions stopped narrowing to active"
    assert ACTIVE_CANARIES["decisions"] in served["get_decisions"], "get_decisions narrowed past the active block too"


def test_no_resource_can_return_raw_corpus_bytes() -> None:
    """By construction, not by remembering.

    Four of the five leaks were one call: ``_read_file`` returns the file
    whole, and it was one import away from every body in the module. The
    import is gone, so the convenient way to serve a corpus no longer exists
    in this module and a new resource has to go through the seam -- or write
    its own ``open()`` in plain sight of review, which is what the last
    assertion watches for.

    Three checks rather than one grep: the module namespace (a top-level
    import), a call in the source (a function-local import used once), and
    a direct ``open`` in a content body. Prose may still name ``_read_file``
    -- explaining why it is gone is worth more than the extra strictness.
    """
    import inspect

    from mind_mem.mcp import resources

    assert not hasattr(resources, "_read_file"), (
        "the raw file reader is imported into resources.py again; a resource can serve corpus bytes"
    )

    module_source = inspect.getsource(resources)
    assert "_read_file(" not in module_source, "resources.py calls the raw file reader; corpus bytes can reach a client unadmitted"

    for name, verdict in sorted(CLASSIFICATION.items()):
        if verdict != CONTENT:
            continue
        body = inspect.getsource(getattr(resources, name))
        assert "open(" not in body, f"{name} reads a file directly instead of going through the admission seam:\n{body}"


def test_no_resource_echoes_the_prose_its_backing_file_starts_with(workspace: str, served: dict[str, str]) -> None:
    """The bytes-vs-blocks distinction, measured rather than argued.

    ``init`` writes a header above the blocks in every corpus file. Nothing
    parses it, so it can only reach a client one way: the resource returned
    the file rather than its admitted blocks. Its presence on disk is the
    positive control -- without that check this asserts the absence of a
    string that was never there.
    """
    reads: dict[str, str] = {
        "tasks/TASKS.md": "get_tasks",
        "entities/projects.md": "get_entities",
        "intelligence/SIGNALS.md": "get_signals",
        "intelligence/CONTRADICTIONS.md": "get_contradictions",
        "shared/intelligence/LEDGER.md": "get_ledger",
    }
    for rel, name in sorted(reads.items()):
        marker = FILE_PREAMBLE[rel]
        on_disk = open(os.path.join(workspace, rel), encoding="utf-8").read()
        assert marker in on_disk, f"positive control failed: {marker!r} is not in {rel}, so its absence downstream proves nothing"
        assert marker not in served[name], f"{name} echoed the {rel} preamble -- it is serving file bytes, not admitted blocks"


def test_signals_reports_the_backlog_it_may_not_serve(served: dict[str, str]) -> None:
    """Withheld BY DESIGN has to be visibly withheld, not silently empty.

    ``capture`` writes ``Status: pending`` and admission withholds it, so this
    resource is empty on a live workspace. An empty list would read as "no
    signals" -- the answer that is wrong in the direction nobody checks. The
    envelope carries the count instead, and names the surface that can review
    them: a resource has no scope check (``register`` binds these functions
    with none of the ACL wrapping ``mcp_tool_observe`` gives a tool), so the
    pending blocks are not servable here at any scope.
    """
    envelope = json.loads(served["get_signals"])
    assert envelope["withheld_count"] == 1, f"the pending signal was not counted: {envelope}"
    served_ids = {block.get("_id") for block in envelope["signals"]}
    assert served_ids == {"SIG-20260101-001"}, f"signals served something other than the admitted block: {served_ids}"
    assert "signal_stats" in envelope["note"], "the envelope does not name the review surface a caller must use instead"


def test_the_ledger_serves_blocks_and_not_the_file(served: dict[str, str]) -> None:
    """The row this file got wrong, pinned as behaviour.

    ``get_ledger`` was classified ``no-content`` against the decisions canary
    -- a file it never reads. It serves ``[FACT-...]`` blocks, and it served
    the quarantined one.
    """
    blocks = json.loads(served["get_ledger"])
    assert isinstance(blocks, list) and blocks, f"get_ledger returned no blocks: {served['get_ledger'][:200]}"
    assert {b.get("_id") for b in blocks} == {"FACT-20260101-001"}, f"get_ledger served the wrong block set: {blocks}"


# ---------------------------------------------------------------------------
# Tests-of-the-test
# ---------------------------------------------------------------------------


def test_the_tripwire_fails_on_an_unclassified_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    real = registered_resources()
    monkeypatch.setattr(sys.modules[__name__], "registered_resources", lambda: {**real, "get_secrets": "mind-mem://secrets"})
    with pytest.raises(AssertionError, match="get_secrets"):
        test_every_registered_resource_is_classified()


def test_the_gate_fails_when_a_resource_regresses_to_serving_the_file(
    workspace: str, served: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation control, and the reason this is not the old test.

    The old version dropped ``get_tasks`` from the defect register and watched
    the gate catch the pinned leak. With the register EMPTY there is no row to
    drop, and a gate whose only proof of life was a row it no longer has is a
    gate nobody has seen fail. So put the DEFECT back instead -- literally the
    2026-09-01 body of ``get_tasks``, parse and return -- and require the gate
    to name it.
    """
    from mind_mem.block_parser import parse_file
    from mind_mem.mcp import resources
    from mind_mem.mcp.infra.workspace import _workspace

    def regressed_get_tasks() -> str:
        ws = _workspace()
        path = os.path.join(ws, "tasks", "TASKS.md")
        if not os.path.isfile(path):
            return json.dumps([])
        return json.dumps(parse_file(path), indent=2, default=str)

    monkeypatch.setattr(resources, "get_tasks", regressed_get_tasks)
    regressed = {**served, "get_tasks": _read("get_tasks", workspace)}
    assert WITHHELD_CANARIES["tasks"] in regressed["get_tasks"], "the regressed body did not reproduce the leak; the control proves nothing"
    with pytest.raises(AssertionError, match="get_tasks"):
        test_no_admitted_resource_serves_withheld_content(regressed)
    with pytest.raises(AssertionError, match="get_tasks"):
        test_the_unadmitted_resource_set_is_exactly_what_is_pinned(regressed)


def test_the_fix_description_check_fails_on_a_thin_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KNOWN_UNADMITTED`` is empty, so its content check needs a control.

    An assertion over an empty mapping passes without reading anything. This
    is what keeps it a rule rather than a formality for whoever adds the next
    row.
    """
    import sys

    monkeypatch.setattr(sys.modules[__name__], "KNOWN_UNADMITTED", {"get_tasks": "leaks"})
    with pytest.raises(AssertionError, match="get_tasks"):
        test_every_pinned_defect_names_its_fix()


def test_the_ratchet_fails_when_a_pinned_defect_is_fixed(served: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The direction that matters when somebody does the work.

    A register that quietly kept a fixed row would let the next reader believe
    a closed defect is still open.
    """
    import sys

    monkeypatch.setattr(sys.modules[__name__], "KNOWN_UNADMITTED", {**KNOWN_UNADMITTED, "get_health": "a defect that is not real"})
    with pytest.raises(AssertionError, match="get_health"):
        test_the_unadmitted_resource_set_is_exactly_what_is_pinned(served)
