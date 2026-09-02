"""Every read surface, swept with a three-status canary.

Two leaks motivated this file, both found by probe rather than by the suite:

* ``get_block`` is a USER-scope tool and it served quarantined block content
  verbatim to anyone who named the id, while ``recall`` withheld the same block.
* ``export_memory`` exported quarantined content, and said nothing about it.

Neither was exotic. Both were invisible because the read-surface tripwire that
was supposed to catch this class enumerated ONE module (``mcp/tools/recall.py``)
out of the twenty-odd that register tools. So the fix is not two patches: it is
the sweep below, which calls **every registered tool** against a corpus holding
one ``active``, one ``pending`` and one ``quarantined`` canary and asserts the
withheld two never come back.

What makes it a proof rather than 102 green error strings:

1. **Positive control on the seed.** The withheld blocks are shown to exist, on
   disk and in the index, by a raw read that bypasses every gate. An assertion
   that a token is absent is worth nothing until something proves the token was
   there to find.
2. **Positive control on the sweep.** Every tool classified ``content`` must
   actually REACH the corpus -- return the active canary or the active block id
   -- and the measured reach set is asserted equal to the ``content`` set. A
   tool that answered "permission denied" or "flag disabled" would otherwise
   pass the canary check while checking nothing.
3. **The sweep can see a leak.** A fixture tool that returns raw block text is
   run through the same helper and must be caught.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
from typing import Any

import pytest
from test_read_surface_classification import (
    ACTIVE_ID,
    ID_DISCLOSING,
    PENDING_ID,
    QUARANTINED_ID,
    SWEEP_FLAGS,
    TOOL_INVOCATIONS,
    WS,
    content_tools,
    registered_tools,
    tool_module,
)

# Improbable tokens: any hit is the seeded block and nothing else.
CANARIES: dict[str, str] = {
    "active": "zqxACTIVEcanary",
    "pending": "zqxPENDINGcanary",
    "quarantined": "zqxQUARANTINEcanary",
}
WITHHELD_CANARIES = (CANARIES["pending"], CANARIES["quarantined"])
SEEDED = ((ACTIVE_ID, "active"), (PENDING_ID, "pending"), (QUARANTINED_ID, "quarantined"))

#: Both ACL scopes. An admin-scope tool that refuses at user scope has not been
#: checked, and ``export_memory`` -- one of the two leaks -- is admin-only.
SCOPES = ("user", "admin")


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------


def _render(block_id: str, status: str, canary: str) -> str:
    """One canary block.

    The statement names ``architecture`` deliberately: the category distiller
    files a block by keyword, and a corpus that lands entirely in
    ``uncategorized`` never exercises ``category_summary``'s topic match. That
    weaker seed is how the category leak stayed invisible on the first pass.
    """
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


def _seed(workspace: str) -> None:
    """One active, one pending, one quarantined canary block, plus the index.

    Written to the corpus directly. Going through an ingest door would be a
    test of that door; what this file needs is a corpus in a known state, with
    the withheld blocks unambiguously present.
    """
    from mind_mem import sqlite_index
    from mind_mem.init_workspace import init

    init(workspace)
    body = "".join(_render(bid, status, CANARIES[status]) for bid, status in SEEDED)
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + body)

    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("v4", {})
    for flag in SWEEP_FLAGS:
        config["v4"][flag] = {"enabled": True}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)

    sqlite_index.build_index(workspace)

    # Derived artifacts are part of the read surface. ``reindex`` runs the
    # category distiller, which copies every block's statement -- withheld ones
    # included -- into ``categories/<name>.md``, and ``category_summary``
    # serves those files. A seed without them sweeps a workspace where that
    # whole path is inert.
    from mind_mem.category_distiller import CategoryDistiller

    CategoryDistiller().distill(workspace)


@pytest.fixture(scope="module")
def seed_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Seeded once; every invocation gets its own copy.

    A shared workspace would let one tool's writes (``delete_memory_item``,
    ``compact``, ``encrypt_file``) change what the next tool sees, and a sweep
    whose corpus depends on alphabetical order proves whatever it happens to.
    """
    template = str(tmp_path_factory.mktemp("read_surface_seed") / "ws")
    _seed(template)
    return template


def _fresh(template: str) -> str:
    target = tempfile.mkdtemp(prefix="mm_sweep_")
    shutil.rmtree(target)
    shutil.copytree(template, target)
    return target


# ---------------------------------------------------------------------------
# Positive control on the seed itself
# ---------------------------------------------------------------------------


def test_the_withheld_canaries_really_are_in_the_corpus(seed_template: str) -> None:
    """Without this, every "canary absent" assertion below is vacuous.

    Read raw: parse the Markdown, and read the index's own rows. Both are
    pre-admission views, so they see what the gates are meant to withhold.
    """
    from mind_mem.block_parser import parse_file

    blocks = {b["_id"]: b for b in parse_file(os.path.join(seed_template, "decisions", "DECISIONS.md")) if b.get("_id")}
    for block_id, status in SEEDED:
        assert block_id in blocks, f"seed failed: {block_id} is not in the corpus"
        assert blocks[block_id]["Status"] == status
        assert CANARIES[status] in json.dumps(blocks[block_id]), f"seed failed: {block_id} carries no canary"

    import sqlite3

    from mind_mem.sqlite_index import DB_REL_PATH

    conn = sqlite3.connect(os.path.join(seed_template, DB_REL_PATH))
    try:
        rows = dict(conn.execute("SELECT id, status FROM blocks WHERE id IN (?,?,?)", (ACTIVE_ID, PENDING_ID, QUARANTINED_ID)).fetchall())
    finally:
        conn.close()
    assert rows == {ACTIVE_ID: "active", PENDING_ID: "pending", QUARANTINED_ID: "quarantined"}, f"index does not hold the seed: {rows}"


def test_the_derived_category_files_carry_the_withheld_canaries(seed_template: str) -> None:
    """Positive control for the derived-artifact leg of the sweep.

    ``category_summary`` reads ``categories/*.md``, not the corpus. If the
    distiller had filed these blocks somewhere the topic match never reaches
    -- or had filtered them itself -- the category sweep would be green over a
    file with nothing in it to leak.
    """
    path = os.path.join(seed_template, "categories", "architecture.md")
    assert os.path.isfile(path), f"seed failed: no architecture category file in {os.listdir(os.path.join(seed_template, 'categories'))}"
    text = open(path, encoding="utf-8").read()
    for status in ("active", "pending", "quarantined"):
        assert CANARIES[status] in text, f"the distiller did not copy the {status} block into the category file"


def test_the_seeded_statuses_are_the_ones_admission_withholds() -> None:
    """The seed must exercise the gate, not sit on the servable side of it."""
    from mind_mem.admissibility import is_admissible_status

    assert is_admissible_status("active")
    assert not is_admissible_status("pending")
    assert not is_admissible_status("quarantined")


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def _call(tool: str, kwargs: dict, workspace: str, scope: str) -> str:
    """Invoke one tool and return everything the caller would see, as text.

    An exception is part of the answer: a traceback quoting the block it failed
    on is as much a leak as a JSON payload, so it is captured and swept too.
    """
    from mind_mem.mcp.infra import rate_limit
    from mind_mem.mcp.infra.workspace import use_workspace

    # The MCP surface rate-limits per client, and a sweep of 100+ tools blows
    # through 120 calls/minute long before it finishes. A rate-limited call
    # returns a refusal envelope, which carries no canary and would therefore
    # read as a clean pass -- 100 green rows measuring the rate limiter. Reset
    # it per call so what the sweep observes is the tool.
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()

    module = importlib.import_module(tool_module(tool))
    fn = getattr(module, tool)
    resolved = {k: (workspace if v == WS else v) for k, v in kwargs.items()}
    previous = os.environ.get("MIND_MEM_SCOPE")
    os.environ["MIND_MEM_SCOPE"] = scope
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    os.environ["MIND_MEM_CONFIG"] = os.path.join(workspace, "mind-mem.json")
    try:
        with use_workspace(workspace):
            result = fn(**resolved)
    except Exception as exc:  # noqa: BLE001 - a raising tool still gets swept
        return f"{type(exc).__name__}: {exc}"
    finally:
        if previous is None:
            os.environ.pop("MIND_MEM_SCOPE", None)
        else:
            os.environ["MIND_MEM_SCOPE"] = previous
    return result if isinstance(result, str) else json.dumps(result, default=str)


@pytest.fixture(scope="module")
def sweep(seed_template: str) -> dict[str, dict[str, Any]]:
    """Call every registered tool, in both scopes, each in its own workspace."""
    results: dict[str, dict[str, Any]] = {}
    for tool in sorted(TOOL_INVOCATIONS):
        outputs: list[tuple[str, dict, str]] = []
        for scope in SCOPES:
            for kwargs in TOOL_INVOCATIONS[tool]:
                workspace = _fresh(seed_template)
                try:
                    outputs.append((scope, kwargs, _call(tool, kwargs, workspace, scope)))
                finally:
                    shutil.rmtree(workspace, ignore_errors=True)
        blob = "\n".join(text for _, _, text in outputs)
        # Naming a withheld block's ID is a weaker disclosure than serving its
        # text, and a different one: it says the block exists. Measured only
        # over invocations that did NOT pass that id in, so an echo of the
        # caller's own argument never counts.
        unprompted = "\n".join(text for _, kwargs, text in outputs if not any(wid in str(kwargs) for wid in (PENDING_ID, QUARANTINED_ID)))
        results[tool] = {
            "outputs": outputs,
            "leaked": sorted(status for status in ("pending", "quarantined") if CANARIES[status] in blob),
            "reached": CANARIES["active"] in blob,
            "names_withheld_ids": any(wid in unprompted for wid in (PENDING_ID, QUARANTINED_ID)),
            "id_only": ACTIVE_ID in blob and CANARIES["active"] not in blob,
        }
    return results


@pytest.mark.parametrize("tool", sorted(TOOL_INVOCATIONS))
def test_no_registered_tool_serves_withheld_content(tool: str, sweep: dict[str, dict[str, Any]]) -> None:
    """The sweep. Every tool, every scope, no withheld canary in the answer."""
    result = sweep[tool]
    if not result["leaked"]:
        return
    offending = [(scope, kwargs, text[:400]) for scope, kwargs, text in result["outputs"] if any(c in text for c in WITHHELD_CANARIES)]
    raise AssertionError(f"{tool} served withheld block content ({result['leaked']}): {offending}")


def test_the_sweep_reached_exactly_the_content_tools(sweep: dict[str, dict[str, Any]]) -> None:
    """Positive control on the sweep, and the check that keeps the table true.

    ``reached`` means the tool's response carried the ACTIVE canary or the
    active block id -- and no invocation passes that id in, so reaching is not
    an echo. Asserting SET EQUALITY (not containment) makes the classification
    a measured fact in both directions: a tool that starts returning block
    content joins this set and fails until it is classified ``content``, and a
    ``content`` tool that quietly stops reaching (a flag default flipped, an
    optional dependency vanished) fails rather than degrading into a canary
    check over an error string.
    """
    reached = {tool for tool, result in sweep.items() if result["reached"]}
    declared = content_tools()
    assert reached == declared, (
        f"read-surface classification disagrees with measured behaviour. "
        f"reached the corpus but classified 'no-content': {sorted(reached - declared)}; "
        f"classified 'content' but never reached the corpus: {sorted(declared - reached)}"
    )


def test_only_the_pinned_tools_name_a_withheld_block_id(sweep: dict[str, dict[str, Any]]) -> None:
    """The weaker disclosure, pinned rather than ignored.

    None of these serves withheld TEXT -- the sweep above covers that. They
    name ids they were not given, which tells a caller a withheld block
    exists. That is defensible for a maintenance surface and indefensible as
    a silent default, so the set is committed and equality is asserted: a new
    tool that starts naming withheld blocks fails here and someone decides.
    """
    naming = {tool for tool, result in sweep.items() if result["names_withheld_ids"]}
    assert naming == set(ID_DISCLOSING), (
        f"tools naming withheld block ids changed. new: {sorted(naming - set(ID_DISCLOSING))}; "
        f"no longer: {sorted(set(ID_DISCLOSING) - naming)}"
    )


def test_the_sweep_called_every_registered_tool(sweep: dict[str, dict[str, Any]]) -> None:
    assert set(sweep) == registered_tools(), f"swept {len(sweep)} of {len(registered_tools())} registered tools"


def test_the_sweep_catches_a_tool_that_leaks(seed_template: str) -> None:
    """Test-of-the-test: the same helper, pointed at a deliberate leak.

    A fixture tool that reads the corpus and returns it raw -- which is what
    ``get_block`` did -- must be caught by the identical assertion the sweep
    uses. If it is not, the 102 green rows above mean nothing.
    """
    from mind_mem.block_parser import parse_file

    workspace = _fresh(seed_template)
    try:
        raw = json.dumps(parse_file(os.path.join(workspace, "decisions", "DECISIONS.md")), default=str)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    leaked = sorted(status for status in ("pending", "quarantined") if CANARIES[status] in raw)
    assert leaked == ["pending", "quarantined"], "the leak fixture did not leak; the sweep's detector is untested"


# ---------------------------------------------------------------------------
# S2 -- get_block
# ---------------------------------------------------------------------------


def _get_block(workspace: str, block_id: str, scope: str = "user") -> dict:
    return json.loads(_call("get_block", {"block_id": block_id}, workspace, scope))


@pytest.mark.parametrize("scope", SCOPES)
def test_get_block_serves_an_admitted_block(seed_template: str, scope: str) -> None:
    """Positive control: the tool still works, so withholding is not a stub."""
    workspace = _fresh(seed_template)
    try:
        payload = _get_block(workspace, ACTIVE_ID, scope)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    assert payload["found"] is True
    assert CANARIES["active"] in json.dumps(payload)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("block_id,status", [(PENDING_ID, "pending"), (QUARANTINED_ID, "quarantined")])
def test_get_block_withholds(seed_template: str, scope: str, block_id: str, status: str) -> None:
    """The leak, pinned. Quarantine is about content, so scope does not widen it."""
    workspace = _fresh(seed_template)
    try:
        payload = _get_block(workspace, block_id, scope)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    blob = json.dumps(payload)
    assert CANARIES[status] not in blob, f"get_block served {status} content at {scope} scope: {blob[:400]}"
    assert payload["found"] is False
    assert payload["withheld"] is True
    assert "block" not in payload, "the envelope still carries the block body"
    assert status not in blob, "the refusal names the status, which is a channel of its own"


def test_get_block_tells_withheld_apart_from_absent(seed_template: str) -> None:
    """Two refusals, two answers. "Not found" for a block that exists is a lie."""
    workspace = _fresh(seed_template)
    try:
        withheld = _get_block(workspace, QUARANTINED_ID, "user")
        absent = _get_block(workspace, "D-19990101-999", "user")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    assert withheld["found"] is False and withheld["withheld"] is True
    assert absent["found"] is False and "withheld" not in absent


def test_get_block_withholds_when_only_the_index_knows_the_status(seed_template: str) -> None:
    """The stale-cache direction: a block quarantined AFTER it was indexed.

    ``admit_read_one`` refreshes from the corpus when the index is stale, which
    is the fail-open direction the index goes stale in. Flip the corpus without
    reindexing and the tool must follow the corpus, not the cache.
    """
    workspace = _fresh(seed_template)
    try:
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        flipped = text.replace(
            f"[{ACTIVE_ID}]\nDate: 2026-01-01\nStatus: active",
            f"[{ACTIVE_ID}]\nDate: 2026-01-01\nStatus: quarantined",
        )
        assert flipped != text, "the corpus rewrite did not match; the test would prove nothing"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(flipped)
        payload = _get_block(workspace, ACTIVE_ID, "user")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    assert payload["found"] is False and payload.get("withheld") is True
    assert CANARIES["active"] not in json.dumps(payload)


# ---------------------------------------------------------------------------
# S3 -- export_memory
# ---------------------------------------------------------------------------


def _export(workspace: str, **kwargs: Any) -> dict:
    return json.loads(_call("export_memory", kwargs, workspace, "admin"))


def test_export_memory_withholds_and_says_how_much(seed_template: str) -> None:
    """Exports the admitted set, and reports the refusal as a count.

    Silently short is the failure mode that matters: an operator who does not
    know two blocks were held back reads the export as the whole corpus.
    """
    workspace = _fresh(seed_template)
    try:
        payload = _export(workspace)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    blob = json.dumps(payload)
    for canary in WITHHELD_CANARIES:
        assert canary not in blob, f"export_memory exported withheld content: {blob[:400]}"
    assert CANARIES["active"] in blob, "positive control: the admitted block was not exported either"
    assert payload["withheld_count"] == 2, f"withheld_count should be 2, got {payload.get('withheld_count')}"
    exported = [json.loads(line) for line in payload["data"].splitlines() if line]
    assert payload["block_count"] == len(exported)
    assert {b["_id"] for b in exported if b["_id"].startswith("D-2026010")} == {ACTIVE_ID}


def test_export_memory_counts_withholding_separately_from_truncation(seed_template: str) -> None:
    """Two different refusals must not collapse into one number.

    ``max_blocks`` is a size cap; admission is a governance decision. An
    operator seeing a short export needs to know which one shortened it, so
    admission runs first and the cap applies to what survived it.
    """
    workspace = _fresh(seed_template)
    try:
        payload = _export(workspace, max_blocks=1)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    assert payload["withheld_count"] == 2
    assert payload["block_count"] == 1
    for canary in WITHHELD_CANARIES:
        assert canary not in json.dumps(payload)


def test_export_memory_has_no_bypass_parameter() -> None:
    """The decision, enforced rather than documented.

    A full-fidelity copy of the corpus is ``snapshot()``. Re-adding an
    ``include_withheld``-shaped argument here would put the leak back behind a
    keyword, so the signature itself is pinned.
    """
    import inspect

    from mind_mem.mcp.tools.memory_ops import export_memory

    params = set(inspect.signature(export_memory).parameters)
    assert params == {"format", "include_metadata", "max_blocks"}, f"export_memory grew a parameter: {sorted(params)}"


def test_export_memory_admits_through_the_shared_seam() -> None:
    """Not a hand-rolled status check -- the same function recall uses.

    Rule 1 of wiring discipline: a leg that reads blocks calls the shared
    admission filter. A local ``if status != 'active'`` would drift from it,
    and drift is how the release-set readmission path would be lost.
    """
    import inspect

    from mind_mem.mcp.tools import memory_ops

    source = inspect.getsource(memory_ops.export_memory)
    assert "admit_read(" in source, "export_memory does not call the shared read-admission seam"
    assert memory_ops.admit_read.__module__ == "mind_mem.admission"
