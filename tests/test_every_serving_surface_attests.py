"""Every door proves what it served — not one of them.

Round 1 bound the recall attestation and the served-set ledger to a single
caller, the MCP recall handler. Measured against a seeded workspace with the
ledger on: ``_recall_impl`` produced a record and a row; ``_handle_query`` (the
HTTP ``POST /query`` route), the ``mind-mem-recall`` CLI, the axis
orchestrator, the guardrail surface, the chat evidence path and the library
entry every one of those calls all served block content and recorded nothing.
"mind-mem can prove what it served" was therefore a property of one handler
rather than of the product, which is the same shape as the read-surface
tripwire that enumerated one module out of twenty.

So the obligation moved onto the entry every consumer already imports
(:func:`mind_mem.recall.recall`), and this file is what keeps it there. Three
layers, because any one of them alone is a signpost rather than a gate:

**Behaviour.** Every registered ``content`` tool, the HTTP query route and the
CLI are invoked against a seeded corpus, and each must leave exactly one ledger
row naming exactly the ids it served. Guarded against vacuity throughout: the
seed is shown to be reachable (the tool returns the seeded id), and the row
count is shown to MOVE rather than merely to be non-zero.

**Structure.** A static walk over ``src/mind_mem`` finds every module that
reaches the ranking function, in any spelling, and requires each to be
classified. Reaching ``_recall_core`` — the engine, which cannot attest —
is allowed only for the entry itself and for a pinned, shrinking residual.

**The guard, guarded.** The scanner is fed source that does reach the engine
and must say so; the leg suppression is exercised on a worker thread with a
positive control proving the same call from a door DOES record.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import mind_mem
from mind_mem.recall_digests import served_set_digest
from mind_mem.served_ledger import (
    LEDGER_ATTESTATION_KEYS,
    LEDGER_DISABLED,
    LEDGER_RELPATH,
    SERVED_ROW_HASH_KEY,
    SERVED_SEQ_KEY,
    ledger_enabled,
    read_served_runs,
    row_hash,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from test_read_surface_classification import (  # noqa: E402
    ACTIVE_ID,
    TOOL_INVOCATIONS,
    WS,
    content_tools,
    tool_module,
)

_PKG_ROOT = pathlib.Path(mind_mem.__file__).resolve().parent
_SRC_ROOT = _PKG_ROOT.parent


# ---------------------------------------------------------------------------
# The classification. Every door, every leg, and the residual that is neither.
# ---------------------------------------------------------------------------

#: The serving entry itself: the one module allowed to reach the raw engine as
#: a matter of course, because reaching it and then attesting is its whole job.
ENTRY = "entry"

#: Serves an answer to somebody. Must reach the engine THROUGH the entry, so
#: the record and the row happen whether or not the author remembered them.
DOOR = "door"

#: Calls the entry from inside another recall — never on its own account. A row
#: from a leg names a candidate set nobody was handed, so the entry suppresses
#: it (:data:`mind_mem.recall.LEG_MODULES`).
LEG = "leg"

#: Reaches ``_recall_core`` directly and therefore serves unattested. A pinned
#: residual, not a permission: each entry names the one-line change that closes
#: it, and :func:`test_no_new_module_reaches_the_engine_unattested` fails the
#: build on a new one.
UNATTESTED = "unattested"

ROLES: dict[str, str] = {
    "mind_mem.recall": ENTRY,
    # Doors, all of them already importing the entry.
    "mind_mem.axis_recall": DOOR,
    "mind_mem.chat_memory": DOOR,
    "mind_mem.http_transport": DOOR,
    "mind_mem.mcp.resources": DOOR,
    "mind_mem.mcp.tools.guardrails": DOOR,
    "mind_mem.mcp.tools.recall": DOOR,
    "mind_mem.mm_cli": DOOR,
    # Doors whose caller is not a person. They run a real recall and receive a
    # real ranking, so their runs are recorded like any other — deliberately,
    # rather than exempted for being internal. An operator reading the ledger
    # sees contradiction scans and benchmark arms alongside client traffic;
    # that is the honest shape, and hiding them would mean the ledger answered
    # "what was served" with "what was served to somebody I considered a user".
    "mind_mem.bench.ab_arms": DOOR,
    "mind_mem.bench.eval_adapters": DOOR,
    "mind_mem.contradiction_detector": DOOR,
    # Legs: reached only from inside a recall that is already underway.
    "mind_mem.hybrid_recall": LEG,
    "mind_mem.sqlite_index": LEG,
    # The residual. Both want the same one-line change — import the ranking
    # function from ``mind_mem.recall`` instead of ``mind_mem._recall_core`` —
    # and both live in files owned by another change, so they are pinned here
    # rather than quietly left out of the sweep.
    "mind_mem.walkthrough": UNATTESTED,
    "mind_mem.v4.cognitive_kernel": UNATTESTED,
}

#: Registered ``content`` tools that serve by running the recall engine. Each
#: must leave a row. Derived nowhere — written down, and checked against
#: :func:`content_tools` in both directions below, so a new content tool fails
#: the build until somebody decides which list it belongs in.
ENGINE_SERVED: frozenset[str] = frozenset(
    {
        "agent_inject",
        "chat_with_memory",
        "hybrid_search",
        "pack_recall_budget",
        "prefetch",
        "recall",
        "recall_with_axis",
        "recall_with_guardrails",
        "recall_with_persona",
    }
)

#: Registered ``content`` tools that serve block content WITHOUT running the
#: engine — a direct block read, a corpus dump, a distilled category file. They
#: are doors and they owe the same proof, but the record they owe is not a
#: recall attestation and the change is not in this one's reach. Pinned, so the
#: set cannot grow silently; every name here is an open item, not an exemption.
STORE_SERVED: frozenset[str] = frozenset({"category_summary", "export_memory", "get_block"})


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------

_SEED_BLOCK = (
    f"[{ACTIVE_ID}]\n"
    "Date: 2026-01-01\n"
    "Status: active\n"
    "Scope: global\n"
    "Statement: The zqxACTIVEcanary architecture decision governs frost telemetry.\n"
    "Rationale: zqxACTIVEcanary rationale for the frost telemetry architecture rollout.\n"
    "Tags: architecture, frost\n"
    "Sources: -\n"
    "Supersedes: -\n\n"
)


def _seed(workspace: str) -> None:
    """One active block, indexed and distilled. No ledger configuration at all.

    Deliberately no ``served_ledger`` section: the default is ON since 5.0.2,
    and a fixture that switched it on would make every assertion below a test
    of the fixture rather than of the default.
    """
    from mind_mem import sqlite_index
    from mind_mem.category_distiller import CategoryDistiller
    from mind_mem.init_workspace import init

    init(workspace)
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + _SEED_BLOCK)
    sqlite_index.build_index(workspace)
    CategoryDistiller().distill(workspace)


@pytest.fixture(scope="module")
def seed_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    template = str(tmp_path_factory.mktemp("attest_seed") / "ws")
    _seed(template)
    return template


def _fresh(template: str) -> str:
    target = tempfile.mkdtemp(prefix="mm_attest_")
    shutil.rmtree(target)
    shutil.copytree(template, target)
    return target


def test_the_seed_is_reachable_and_the_ledger_is_on_by_default(seed_template: str) -> None:
    """Positive control for everything below.

    Two ways this file could be green over nothing: a corpus the tools never
    reach (every "one row" assertion would fail loudly, but every "no extra
    row" one would pass vacuously), and a workspace where the ledger is off
    (in which case zero rows everywhere is the *correct* answer and the sweep
    proves nothing). Both are checked here, before anything else runs.
    """
    ws = _fresh(seed_template)
    try:
        assert ledger_enabled(ws) is True, "the fixture wrote no served_ledger section — the default must carry it"
        assert read_served_runs(ws) == (), "a freshly seeded workspace already has rows"
        from mind_mem.recall import recall

        hits = recall(ws, "architecture decision", limit=10)
        assert [h["_id"] for h in hits] == [ACTIVE_ID], f"the seed is not reachable: {hits}"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Layer 1 — behaviour: every door leaves exactly one row naming what it served
# ---------------------------------------------------------------------------


def _invoke(tool: str, kwargs: dict, workspace: str) -> str:
    """Call one MCP tool exactly as the read-surface sweep does."""
    from mind_mem.mcp.infra import rate_limit
    from mind_mem.mcp.infra.workspace import use_workspace

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    module = importlib.import_module(tool_module(tool))
    fn = getattr(module, tool)
    resolved = {k: (v.replace(WS, workspace) if isinstance(v, str) else v) for k, v in kwargs.items()}
    previous = os.environ.get("MIND_MEM_SCOPE")
    os.environ["MIND_MEM_SCOPE"] = "admin"
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    os.environ["MIND_MEM_CONFIG"] = os.path.join(workspace, "mind-mem.json")
    try:
        with use_workspace(workspace):
            result = fn(**resolved)
    finally:
        if previous is None:
            os.environ.pop("MIND_MEM_SCOPE", None)
        else:
            os.environ["MIND_MEM_SCOPE"] = previous
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _served_ids_in(text: str) -> list[str] | None:
    """The ids a response's ``results`` ranking exposes, or ``None``.

    ``None`` means *this response does not expose ids*, which is a real answer
    and not a failure to parse: the ``brief`` persona projection strips ``_id``
    off every hit, so its response carries content with no id anywhere in it.
    A row is still recorded for that run — the ledger is derived from the
    engine's output, not from the projection — but the response cannot be
    cross-checked against it, and pretending it can (by comparing the row's ids
    to a list of empty strings) would report a projection as a mismatch.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    hits = [hit for hit in results if isinstance(hit, dict)]
    ids = [str(hit.get("_id", "")) for hit in hits]
    if any(not block_id for block_id in ids):
        return None
    return ids


def test_the_content_classification_is_partitioned_and_complete() -> None:
    """The two lists must together be exactly the registry's ``content`` set.

    Both directions. A new content tool that nobody classified fails here
    rather than being quietly absent from the sweep — which is precisely how
    the previous generation of this check enumerated one module and stayed
    green over the other five sixths of the surface.
    """
    assert ENGINE_SERVED & STORE_SERVED == frozenset(), "a tool cannot be in both lists"
    assert (ENGINE_SERVED | STORE_SERVED) == content_tools(), (
        f"unclassified: {sorted(content_tools() - ENGINE_SERVED - STORE_SERVED)}; "
        f"stale: {sorted((ENGINE_SERVED | STORE_SERVED) - content_tools())}"
    )


@pytest.fixture(scope="module")
def tool_sweep(seed_template: str) -> dict[str, list[dict[str, Any]]]:
    """Every ``content`` tool, every argument set, each in its own workspace.

    One sweep, read by the four properties below, because calling a hundred-odd
    tool invocations once per assertion is the difference between a gate that
    runs and a gate somebody deletes. A private workspace per invocation so a
    tool's own writes cannot change what the next one sees.
    """
    sweep: dict[str, list[dict[str, Any]]] = {}
    for tool in sorted(ENGINE_SERVED | STORE_SERVED):
        calls: list[dict[str, Any]] = []
        for kwargs in TOOL_INVOCATIONS[tool]:
            ws = _fresh(seed_template)
            try:
                try:
                    text = _invoke(tool, kwargs, ws)
                except Exception as exc:  # noqa: BLE001 — a raising tool is still swept
                    text = f"{type(exc).__name__}: {exc}"
                rows = read_served_runs(ws)
                calls.append(
                    {
                        "kwargs": kwargs,
                        "text": text,
                        "rows": rows,
                        "reached": ACTIVE_ID in text,
                        "returned": _served_ids_in(text),
                    }
                )
            finally:
                shutil.rmtree(ws, ignore_errors=True)
        sweep[tool] = calls
    return sweep


@pytest.mark.parametrize("tool", sorted(ENGINE_SERVED))
def test_no_engine_served_tool_reaches_the_corpus_without_a_row(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """The finding itself: content out, nothing recorded.

    Measured per invocation rather than per tool, because a tool can serve on
    one argument set and legitimately serve nothing on another — ``recall``
    dispatches ten modes and two of them (``classify``, ``diagnostics``) answer
    without touching a block. What is not allowed is content leaving with no
    row behind it.
    """
    offenders = [call["kwargs"] for call in tool_sweep[tool] if call["reached"] and not call["rows"]]
    assert not offenders, f"{tool} served block content and recorded nothing: {offenders}"


@pytest.mark.parametrize("tool", sorted(ENGINE_SERVED))
def test_no_engine_served_tool_records_one_answer_twice(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """At most one row per call — the other way the ledger stops being joinable.

    A door that fuses several engine passes, or falls back through a second
    backend, can mint a row per leg. Rows naming candidate sets nobody was
    handed are not a surplus, they are a different answer to "what was served".
    """
    offenders = [(call["kwargs"], len(call["rows"])) for call in tool_sweep[tool] if len(call["rows"]) > 1]
    assert not offenders, f"{tool} recorded more than one row for a single call: {offenders}"


@pytest.mark.parametrize("tool", sorted(ENGINE_SERVED))
def test_every_engine_served_tool_records_the_ids_it_returned(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """The row is the answer, not an approximation of it.

    Where the response exposes its ranking (a ``results`` list of hits), the
    row's ids must BE that ranking, in that order. Where it does not, every
    recorded id must at least appear in what the caller received. And the
    record the response carries, when it carries one, must commit to the same
    served set as the row — one run, one digest, never two opinions.
    """
    for call in tool_sweep[tool]:
        rows = call["rows"]
        if not rows:
            continue
        row = rows[-1]
        assert row.served_digest == served_set_digest(row.ids), f"{tool}{call['kwargs']}: row digest does not match its own ids"
        returned = call["returned"]
        if returned:
            assert list(row.ids) == returned, f"{tool}{call['kwargs']} served {returned} but recorded {list(row.ids)}"
        elif call["reached"]:
            for block_id in row.ids:
                assert block_id in call["text"], f"{tool}{call['kwargs']} recorded {block_id}, which is not in what it returned"
        try:
            payload = json.loads(call["text"])
        except (TypeError, ValueError):
            continue
        attestation = payload.get("attestation") if isinstance(payload, dict) else None
        if attestation is not None:
            assert attestation["results_digest"] == row.served_digest, f"{tool}{call['kwargs']}: record and row disagree on the served set"


@pytest.mark.parametrize("tool", sorted(ENGINE_SERVED))
def test_every_engine_served_tool_actually_reached_the_corpus(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """The positive control, without which the three checks above are free.

    A tool that answered "permission denied", "flag disabled" or a binding
    error never served anything, so "no unrecorded content" is true of it for
    the wrong reason. Each tool must reach the seeded block on at least one of
    its argument sets, and record a row on that call.
    """
    reached = [call for call in tool_sweep[tool] if call["reached"]]
    assert reached, f"{tool} never reached the corpus on any invocation — its rows prove nothing"
    assert any(call["rows"] for call in reached), f"{tool} reached the corpus but never recorded"


def test_at_least_one_swept_response_exposes_the_ids_it_served(tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """The control on :func:`_served_ids_in`'s ``None`` branch.

    That branch says "this response does not expose ids", and a helper that
    returned ``None`` for *every* response would silently downgrade the
    strongest assertion in this file — row ids EQUAL returned ids — into the
    weak containment fallback, everywhere, with nothing going red. So at least
    one swept response must come back with real ids.
    """
    exposed = {tool: call["returned"] for tool, calls in tool_sweep.items() for call in calls if call["returned"]}
    assert exposed, "no swept response exposed a served id — the equality branch never runs"


@pytest.mark.parametrize("tool", sorted(STORE_SERVED))
def test_the_store_read_surfaces_are_the_named_residual(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """A ratchet over what is NOT yet closed, so it cannot be forgotten.

    These three serve block content without running the engine, so they record
    nothing today. That is the open item; pinning it means the day one of them
    starts recording, this test fails and somebody moves it into
    :data:`ENGINE_SERVED` deliberately rather than by accident.
    """
    calls = tool_sweep[tool]
    assert any(call["reached"] for call in calls), f"{tool} did not reach the corpus — this pin measures nothing"
    assert not any(call["rows"] for call in calls), f"{tool} now records a served run — move it to ENGINE_SERVED"


def _attestation_in(text: str) -> dict[str, Any] | None:
    """The ``attestation`` object a response carries, or ``None``."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    attestation = payload.get("attestation")
    return attestation if isinstance(attestation, dict) else None


@pytest.mark.parametrize("tool", sorted(ENGINE_SERVED))
def test_every_attesting_tool_publishes_which_row_it_wrote(tool: str, tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """A record that is silent about its row is a proof-shaped object.

    The measured defect: with the ledger directory unwritable, a recall
    returned its hits and an attestation while the ledger stayed empty, and
    nothing in the record said so — the caller was handed a proof of a run no
    verifier could find. So every attesting surface publishes all three of
    :data:`~mind_mem.served_ledger.LEDGER_ATTESTATION_KEYS`, and where a row
    WAS written the published seq and hash must be that row's.
    """
    for call in tool_sweep[tool]:
        attestation = _attestation_in(call["text"])
        if attestation is None:
            continue
        missing = [key for key in LEDGER_ATTESTATION_KEYS if key not in attestation]
        assert not missing, f"{tool}{call['kwargs']}: attestation is silent about its ledger row — missing {missing}"
        rows = call["rows"]
        if rows:
            assert attestation[SERVED_SEQ_KEY] == rows[-1].seq, f"{tool}{call['kwargs']}: published seq is not the row it wrote"
            assert attestation[SERVED_ROW_HASH_KEY] == row_hash(rows[-1]), f"{tool}{call['kwargs']}: published hash is not the row it wrote"
        else:
            claimed = attestation[SERVED_SEQ_KEY]
            assert claimed is None, f"{tool}{call['kwargs']}: claims row {claimed} with an empty ledger"


def test_at_least_one_swept_response_carries_an_attestation(tool_sweep: dict[str, list[dict[str, Any]]]) -> None:
    """The vacuity control for the sweep above.

    Every one of its assertions is inside ``if attestation is None: continue``.
    A helper that never found one would make the whole parametrisation green
    over nothing at all.
    """
    carried = {tool for tool, calls in tool_sweep.items() for call in calls if _attestation_in(call["text"])}
    assert carried, "no swept response carried an attestation — the ledger-key sweep never ran a single assertion"


def _break_the_ledger(ws: str) -> None:
    """A plain FILE where the ledger directory must be.

    Not a permission bit: CI containers run as root, for whom the bit is
    advisory, and Windows does not honour it at all. This fails identically
    for every user on every platform.
    """
    directory = pathlib.Path(ws) / pathlib.Path(LEDGER_RELPATH).parent
    shutil.rmtree(directory, ignore_errors=True)
    directory.write_text("not a directory\n", encoding="utf-8")


def test_an_unwritable_ledger_yields_a_null_seq_on_every_attesting_surface(seed_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The three code paths that mint a published attestation, all broken the same way.

    ``mind_mem.recall.attest_and_record`` is what the HTTP route, the CLI, the
    axis orchestrator, the guardrail surface, the chat path and the library
    entry all publish through; ``mcp.tools.recall._record_served_run`` is the
    second and last. Driving the entry, a door over it, and the MCP handler
    covers both, and each must answer "no row, and here is why" rather than
    handing back a record that merely omits the question.

    Each assertion is paired with its positive control on an intact workspace,
    so a surface that stopped attesting entirely fails here rather than passing
    for having nothing to say.
    """
    from mind_mem import http_transport
    from mind_mem.recall import recall

    def _entry(ws: str) -> dict[str, Any] | None:
        return recall(ws, "architecture decision", limit=5).attestation

    def _http(ws: str) -> dict[str, Any] | None:
        _status, payload = http_transport._handle_query(ws, {"query": "architecture decision", "limit": 5})
        return payload.get("attestation")

    def _mcp(ws: str) -> dict[str, Any] | None:
        import mind_mem.mcp.tools.recall as mcp_recall

        monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
        envelope = json.loads(mcp_recall._recall_impl("architecture decision", limit=5))
        return envelope.get("attestation")

    for name, surface in (("entry", _entry), ("http", _http), ("mcp", _mcp)):
        healthy = _fresh(seed_template)
        broken = _fresh(seed_template)
        try:
            control = surface(healthy)
            assert control is not None, f"{name} published no attestation at all — its null-seq check would be vacuous"
            assert control[SERVED_SEQ_KEY] == 0, f"{name} did not record a row on an intact workspace: {control}"
            assert control[SERVED_ROW_HASH_KEY] == row_hash(read_served_runs(healthy)[-1])

            _break_the_ledger(broken)
            record = surface(broken)
            assert record is not None, f"{name} stopped attesting when the ledger broke"
            assert record[SERVED_SEQ_KEY] is None, f"{name} claims row {record[SERVED_SEQ_KEY]} with no ledger to hold it"
            assert record[SERVED_ROW_HASH_KEY] is None, f"{name} published a row hash for a row it never wrote"
            assert record["ledger_error"], f"{name} published no reason for the missing row"
            assert record["ledger_error"] != LEDGER_DISABLED, f"{name} reported a broken ledger as an opted-out one"
        finally:
            shutil.rmtree(healthy, ignore_errors=True)
            shutil.rmtree(broken, ignore_errors=True)


def test_the_http_query_route_attests_and_records(seed_template: str) -> None:
    """``POST /query`` — the route that served content and recorded nothing."""
    from mind_mem import http_transport

    ws = _fresh(seed_template)
    try:
        status, payload = http_transport._handle_query(ws, {"query": "architecture decision", "limit": 5})
        assert status == 200, payload
        served = [hit["_id"] for hit in payload["results"]]
        assert served == [ACTIVE_ID], f"the route did not reach the corpus: {payload}"

        rows = read_served_runs(ws)
        assert len(rows) == 1, f"the route left {len(rows)} ledger rows, expected 1"
        assert list(rows[0].ids) == served
        attestation = payload["attestation"]
        assert attestation is not None, "the route served without a record"
        assert attestation["results_digest"] == rows[0].served_digest
        assert attestation["results_digest"] == served_set_digest(served)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_the_http_query_route_on_the_raw_engine_would_be_caught(seed_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation twin: put the route back on the engine and this must go red.

    The route reached ``_recall_core.recall`` directly until 5.0.2. Rather than
    trusting that the assertion above would have caught it, the pre-fix wiring
    is reinstated — the entry is replaced by the raw engine it wraps, which is
    exactly what importing ``_recall_core`` gave the route — and the check is
    shown to fail. A gate nobody has watched fail is a gate nobody has tested.
    """
    from mind_mem import _recall_core, http_transport

    monkeypatch.setattr("mind_mem.recall.recall", _recall_core.recall)
    ws = _fresh(seed_template)
    try:
        status, payload = http_transport._handle_query(ws, {"query": "architecture decision", "limit": 5})
        assert status == 200, payload
        assert [hit["_id"] for hit in payload["results"]] == [ACTIVE_ID], "the mutant did not reach the corpus"
        assert payload["attestation"] is None, "the mutation did not take — the route still attested"
        assert read_served_runs(ws) == (), "the mutation did not take — the route still recorded"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_the_recall_cli_attests_and_records(seed_template: str) -> None:
    """``mind-mem-recall`` — a door with a human on the other side of it.

    Run as a subprocess, through the console-script entry point's own module,
    so what is measured is the shipped CLI rather than a function this test
    chose to call.
    """
    ws = _fresh(seed_template)
    try:
        env = {**os.environ, "PYTHONPATH": str(_SRC_ROOT)}
        env.pop("MIND_MEM_WORKSPACE", None)
        proc = subprocess.run(
            [sys.executable, "-m", "mind_mem.recall", "-q", "architecture decision", "-w", ws, "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        served = [hit["_id"] for hit in json.loads(proc.stdout)]
        assert served == [ACTIVE_ID], f"the CLI did not reach the corpus: {proc.stdout[:400]}"

        rows = read_served_runs(ws)
        assert len(rows) == 1, f"the CLI left {len(rows)} ledger rows, expected 1"
        assert list(rows[0].ids) == served
        assert rows[0].served_digest == served_set_digest(served)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_the_mm_cli_attests_and_records(seed_template: str) -> None:
    """``mm recall`` — the second CLI, and a door in its own right.

    Two shipped command-line entry points reach the engine, and covering one
    of them would repeat this finding's own mistake at a smaller scale.
    """
    ws = _fresh(seed_template)
    try:
        env = {**os.environ, "PYTHONPATH": str(_SRC_ROOT), "MIND_MEM_WORKSPACE": ws}
        proc = subprocess.run(
            [sys.executable, "-m", "mind_mem.mm_cli", "recall", "architecture decision"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert ACTIVE_ID in proc.stdout, f"the CLI did not reach the corpus: {proc.stdout[:400]}"

        rows = read_served_runs(ws)
        assert len(rows) == 1, f"mm recall left {len(rows)} ledger rows, expected 1"
        assert list(rows[0].ids) == [ACTIVE_ID]
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Layer 2 — structure: nothing reaches the engine without a declared role
# ---------------------------------------------------------------------------


def _module_name(path: pathlib.Path) -> str:
    parts = list(path.resolve().relative_to(_SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _ranking_imports(source: str, module: str) -> set[str]:
    """Which ranking-function providers *source* reaches, in ANY spelling.

    Returns a subset of ``{"mind_mem.recall", "mind_mem._recall_core"}``.
    Normalised across relative and absolute forms, and laziness-blind — a
    function-local import is exactly the form a new bypass arrives in, and it
    is invisible to both the loader and ``sys.modules``.
    """
    package = module.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name in ("mind_mem.recall", "mind_mem._recall_core")}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                kept = max(0, len(package) - (node.level - 1))
                base = ".".join([*package[:kept], *(node.module.split(".") if node.module else [])])
            if base in ("mind_mem.recall", "mind_mem._recall_core") and any(a.name == "recall" for a in node.names):
                found.add(base)
            elif base == "mind_mem":
                found |= {f"mind_mem.{a.name}" for a in node.names if a.name in ("recall", "_recall_core")}
    return found


def engine_reachers() -> dict[str, set[str]]:
    """Every first-party module that reaches the ranking function."""
    out: dict[str, set[str]] = {}
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        module = _module_name(path)
        if module == "mind_mem._recall_core":
            continue  # the engine defines it; it does not reach it
        reached = _ranking_imports(path.read_text(encoding="utf-8"), module)
        if reached:
            out[module] = reached
    return out


def test_every_module_that_reaches_the_ranking_function_has_a_role() -> None:
    """Both directions, so the table can neither miss one nor keep a ghost."""
    reachers = engine_reachers()
    assert reachers, "the scanner found nothing — it is not scanning"
    assert set(reachers) == set(ROLES), f"unclassified: {sorted(set(reachers) - set(ROLES))}; stale: {sorted(set(ROLES) - set(reachers))}"


def test_no_new_module_reaches_the_engine_unattested() -> None:
    """The tripwire the finding asked for, in its enforceable form.

    Importing the ranking function from ``mind_mem._recall_core`` bypasses the
    attestation by construction, because the engine is the half that cannot
    attest. Exactly two modules still do it and both are pinned above; the
    entry itself is the third and it is the point. Anything else fails here.
    """
    offenders = {module: sorted(reached) for module, reached in engine_reachers().items() if "mind_mem._recall_core" in reached}
    allowed = {module for module, role in ROLES.items() if role in (ENTRY, UNATTESTED)}
    assert set(offenders) == allowed, (
        f"new unattested engine importer(s): {sorted(set(offenders) - allowed)}; "
        f"closed (delete the pin): {sorted(allowed - set(offenders))}"
    )


def test_every_door_reaches_the_engine_only_through_the_entry() -> None:
    """A door that imported the engine would serve without a record."""
    reachers = engine_reachers()
    for module, role in ROLES.items():
        if role not in (DOOR, LEG):
            continue
        assert reachers[module] == {"mind_mem.recall"}, f"{module} is a {role} but reaches {sorted(reachers[module])}"


def test_the_declared_legs_are_the_suppressed_legs() -> None:
    """The table and the runtime suppression are one list, not two.

    ``LEG_MODULES`` is what the entry actually consults; this table is what a
    reader is told. Two copies of a rule is how the rule stops being true.
    """
    from mind_mem.recall import LEG_MODULES

    assert {module for module, role in ROLES.items() if role == LEG} == set(LEG_MODULES)


#: Every spelling that reaches the engine, paired with a module the relative
#: forms actually resolve FROM. The pairing is the point: ``from ._recall_core
#: import recall`` names the engine inside ``mind_mem.walkthrough`` and names a
#: module that does not exist inside ``mind_mem.mcp.tools.recall``, so a check
#: that fed every form to one arbitrary module would be testing its own choice
#: of module rather than the scanner.
_ENGINE_IMPORT_FORMS = (
    ("from mind_mem._recall_core import recall", "mind_mem.probe"),
    ("from mind_mem._recall_core import recall as _r", "mind_mem.mcp.tools.probe"),
    ("from ._recall_core import recall", "mind_mem.probe"),
    ("from .. import _recall_core", "mind_mem.mcp.probe"),
    ("import mind_mem._recall_core", "mind_mem.mcp.tools.probe"),
    ("from mind_mem import _recall_core", "mind_mem.probe"),
)


@pytest.mark.parametrize("form,module", _ENGINE_IMPORT_FORMS, ids=[f for f, _ in _ENGINE_IMPORT_FORMS])
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_the_scanner_sees_every_spelling_of_the_engine_import(form: str, module: str, lazy: bool) -> None:
    """The guard, guarded — a PRESENCE assertion under an absence check.

    Every structural test above asserts that a set is empty or equal, and an
    empty set is what a broken walker reports for free. Fed source that really
    does reach the engine, in each spelling, the scanner must say so. A walker
    that quietly recognises five forms out of six passes every absence check
    ever written and stops the sixth author from nothing.
    """
    source = f"def f():\n    {form}\n    return 1\n" if lazy else f"{form}\n"
    assert "mind_mem._recall_core" in _ranking_imports(source, module), f"the scanner cannot see `{form}` in {module}"


def test_a_relative_import_resolves_against_its_own_package() -> None:
    """``from ._recall_core import x`` means a different module in a subpackage.

    Normalisation that ignored the importing module's package would report
    ``mind_mem.mcp.tools._recall_core`` as the engine — firing on a module that
    does not exist — or miss the one that does.
    """
    assert _ranking_imports("from ._recall_core import recall\n", "mind_mem.mcp.tools.probe") == set()
    assert _ranking_imports("from ._recall_core import recall\n", "mind_mem.probe") == {"mind_mem._recall_core"}
    assert _ranking_imports("from .recall import recall\n", "mind_mem.probe") == {"mind_mem.recall"}


def test_the_scanner_does_not_cry_wolf() -> None:
    """A symbol is not the ranking function, and a stdlib import is neither."""
    assert _ranking_imports("from mind_mem.recall import context_pack\n", "mind_mem.probe") == set()
    assert _ranking_imports("from mind_mem.recall_digests import served_set_digest\n", "mind_mem.probe") == set()
    assert _ranking_imports("import json\n", "mind_mem.probe") == set()


# ---------------------------------------------------------------------------
# Layer 3 — the leg suppression, on the thread where it is hardest
# ---------------------------------------------------------------------------


def test_a_retrieval_leg_records_nothing_even_on_a_worker_thread(seed_template: str) -> None:
    """The half of the guard the serving scope cannot cover.

    ``hybrid_recall``'s BM25 arm falls back into the serving entry when the FTS
    database is missing, and it does so on a pool worker when query expansion
    produces more than one variant. The scope is thread state and a worker
    starts with a fresh one — measured, not assumed — so a leg on a worker
    would mint its own row for a candidate set nobody was handed.

    The positive control is the whole test: the SAME call shape from a door,
    on a worker thread, must still record. Without it this would pass on a
    workspace where nothing can be recorded at all.
    """
    from mind_mem.hybrid_recall import HybridBackend
    from mind_mem.recall import recall, serving_scope
    from mind_mem.sqlite_index import _db_path

    ws = _fresh(seed_template)
    try:
        os.remove(_db_path(ws))  # force the BM25 arm onto its recall fallback
        backend = HybridBackend.from_config({})

        with serving_scope():
            with ThreadPoolExecutor(max_workers=2) as pool:
                legs = list(pool.map(lambda q: backend._bm25_search_raw(q, ws, limit=5), ("architecture", "frost telemetry")))
        assert any(hits for hits in legs), "the legs returned nothing — the suppression check is vacuous"
        assert read_served_runs(ws) == (), "a retrieval leg on a worker thread minted a ledger row"

        with ThreadPoolExecutor(max_workers=1) as pool:
            served = pool.submit(lambda: recall(ws, "architecture decision", limit=5)).result()
        assert [hit["_id"] for hit in served] == [ACTIVE_ID]
        assert len(read_served_runs(ws)) == 1, "positive control failed: a door on a worker thread recorded nothing"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_a_nested_engine_call_under_an_open_scope_records_nothing(seed_template: str) -> None:
    """The other half: a door that claims the serve owns the only row."""
    from mind_mem.recall import recall, serving_scope

    ws = _fresh(seed_template)
    try:
        with serving_scope():
            inner = recall(ws, "architecture decision", limit=5)
        assert [hit["_id"] for hit in inner] == [ACTIVE_ID], "the nested call returned nothing to suppress"
        assert inner.attestation is None
        assert read_served_runs(ws) == ()

        outer = recall(ws, "architecture decision", limit=5)
        assert outer.attestation is not None, "positive control failed: an unscoped call recorded nothing"
        assert len(read_served_runs(ws)) == 1
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_the_serving_entry_still_reads_the_clock_exactly_once(seed_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding the proof must not add a second "now".

    Recall's determinism claim rests on exactly one clock read per run, at the
    boundary. A record derived after the ranking, with its own defaulted
    instant, is a SECOND read — and across a UTC midnight the two disagree, so
    the record names a day the ranking never scored against and replays to a
    different answer. The entry therefore resolves the instant itself and hands
    the same date to the engine and to the record.

    Counted rather than argued, with the pre-fix shape as the positive control:
    letting the record default its own instant reads the clock twice.
    """
    from mind_mem import _recall_core, scoring_instant
    from mind_mem.recall import attest_and_record, recall

    reads: list[int] = []
    real = scoring_instant._read_utc_today
    monkeypatch.setattr(scoring_instant, "_read_utc_today", lambda: (reads.append(1), real())[1])

    ws = _fresh(seed_template)
    try:
        reads.clear()
        engine_hits = _recall_core.recall(ws, "architecture decision", limit=5)
        assert engine_hits, "the engine returned nothing — the count would be of an empty run"
        assert len(reads) == 1, f"the engine alone read the clock {len(reads)} times"

        reads.clear()
        served = recall(ws, "architecture decision", limit=5)
        assert served.attestation is not None, "nothing was attested — there is no second read to look for"
        assert len(reads) == 1, f"the serving entry read the clock {len(reads)} times, not once"

        # Positive control: the shape where the record defaults its own instant.
        reads.clear()
        hits = _recall_core.recall(ws, "architecture decision", limit=5)
        attest_and_record(ws, "architecture decision", hits)
        assert len(reads) == 2, "the control did not double-read — this test cannot see the defect it guards"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_an_exception_inside_a_scope_does_not_leave_it_open(seed_template: str) -> None:
    """A raising leg must not turn every later serve into a silent one."""
    from mind_mem.recall import in_serving_scope, recall, serving_scope

    ws = _fresh(seed_template)
    try:
        with pytest.raises(RuntimeError):
            with serving_scope():
                raise RuntimeError("a leg blew up")
        assert in_serving_scope() is False
        assert recall(ws, "architecture decision", limit=5).attestation is not None
        assert len(read_served_runs(ws)) == 1
    finally:
        shutil.rmtree(ws, ignore_errors=True)
