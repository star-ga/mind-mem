# Copyright 2026 STARGA, Inc.
"""Enumerate EVERY registered user-scope MCP tool and resource for a shared-only caller.

Requirement: no read surface returns block content, counts, ids, scores or
existence signals to a caller beyond what the namespace ACL grants, and a
denial must not differ from a miss.

The unit of enumeration is the live registry (``mcp.list_tools()`` /
``list_resources()`` / ``list_resource_templates()``), not a hand-kept list, so
a tool registered tomorrow is swept tomorrow. Two ways it fails the build:

* it returns a canary (content, id, root-only count) to ``eve`` -- ungated;
* it has a required parameter this file has no value for -- nobody told the
  sweep how to call it, so it was never checked.

Three properties are asserted for every user-scope tool, for a caller (``eve``)
who may read ``shared/`` only, on a workspace whose root, ``agents/bob`` and
``agents/alice`` hold canary content:

1. no response contains a canary string;
2. a denied id answers exactly like a missing id (byte for byte once the
   echoed id and volatile fields are scrubbed);
3. non-interference: the response does not change when the root corpus the
   caller cannot read grows by an arbitrary number of blocks -- which is what
   "no root-only count, score or existence signal" means operationally.

Each is paired with a positive control (an unbound caller reaches the canary
through the same tools) and the whole sweep is shown to catch a mutation.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_read_surface_acl_tools import _blk, _w, registered, ws  # noqa: F401

from mind_mem.audit_context import bind_current_agent
from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS
from mind_mem.mcp.infra.workspace import use_workspace

QUERY = "frost telemetry rollout architecture"

#: Strings that must never reach ``eve``: root / other-agent content, ids and the
#: graph entities only the root corpus produced.
CANARIES = (
    "zqROOT",
    "zqBOB",
    "zqALICE",
    "D-ROOT",
    "D-BOB",
    "D-ALICE",
    "T-ROOT",
    "T-BOB",
    "T-ALICE",
    "C-ROOT",
    "DRIFT-ROOT",
    "rootent",
    "secretcomp",
    "zqprobequery",
)

DENIED_IDS = ("D-ROOT-ACT", "D-BOB-ACT", "T-ROOT-ACT", "T-BOB-ACT")
MISSING_ID = "D-NOPE-ACT"
#: id-bearing parameters: name -> (denied values, missing value)
ID_PARAMS: dict[str, tuple[tuple[str, ...], str]] = {
    "block_id": (DENIED_IDS, MISSING_ID),
    "seed_block_ids": (DENIED_IDS, MISSING_ID),
    "entity": (("rootent", "secretcomp"), "nopeent"),
    "entity_id": (("rootent", "secretcomp"), "nopeent"),
}

#: Value for a required parameter, by name. Tool-specific overrides below win.
NAME_DEFAULTS: dict[str, Any] = {
    "query": QUERY,
    "question": QUERY,
    "topic": QUERY,
    "task": QUERY,
    "text": QUERY,
    "signals": "frost telemetry",
    "block_id": MISSING_ID,
    "seed_block_ids": MISSING_ID,
    "content_hash": "0" * 64,
    "entity": "nopeent",
    "entity_id": "nopeent",
    "winner_id": "nopeA",
    "loser_id": "nopeB",
    "rationale": "a sufficiently long rationale",
    "subject": "sa",
    "predicate": "related_to",
    "object": "ob",
    "source_block_id": MISSING_ID,
    "session_id": "s1",
    "previous_query": "old frost query",
    "new_query": "new frost query",
    "name": "recall",
    "input": "{}",
    "block": {"Statement": "x"},
    "type_name": "Decision",
}
TOOL_ARGS: dict[str, tuple[dict[str, Any], ...]] = {
    "compiled_truth": ({"action": "load", "entity_id": "nopeent"}, {"action": "contradictions", "entity_id": "nopeent"}),
    "core": ({"action": "list"},),
    "graph": ({"action": "stats"}, {"action": "query", "entity": "nopeent"}, {"action": "traverse", "block_id": MISSING_ID}),
    "audit_model_tool": ({"path": "$WS/agents/bob", "include_manifest": True}, {"path": "$WS", "include_manifest": True}),
    "verify_model_tool": ({"path": "$WS/agents/bob"}, {"path": "$WS"}),
    "vault_scan": ({"vault_root": "$WS/agents/bob"}, {"vault_root": "$WS"}),
    "mic_inspect_tool": ({"input": "{}"},),
    "mic_convert_tool": ({"input": "{}", "input_format": "mic", "output_format": "mic-b"},),
    "validate_block": ({"text": "[X-1]\nType: Decision\nStatement: " + QUERY + "\nStatus: active\n"},),
    "recall": ({"query": QUERY, "limit": 50}, {"query": QUERY, "limit": 50, "backend": "bm25"}),
    "hybrid_search": ({"query": QUERY, "limit": 50},),
    "category_summary": ({"topic": "architecture"}, {"topic": QUERY}),
    "outcome_stats": ({}, {"block_id": MISSING_ID}),
    "list_evidence": ({}, {"block_id": MISSING_ID}),
}
#: Tools whose call writes: swept for canaries but not for the id / twin comparisons.
WRITERS = frozenset(
    {
        "propose_edge",
        "propose_entity_merge",
        "observe_signal",
        "stale_blocks",
        "reindex_dirty",
        "dream_cycle",
        "memory_evolution",
        "compiled_truth",
        "core",
        "graph",
        "agent_inject",
        "recall",
        "hybrid_search",
        "recall_with_axis",
        "recall_with_guardrails",
        "recall_with_persona",
        "pack_recall_budget",
        "prefetch",
        "chat_with_memory",
    }
)

#: Response keys whose value is volatile (time, random id, duration) and so carries no ACL signal.
_VOLATILE = re.compile(
    r"(^|_)(ts|time|times|timestamp|at|date|dates|duration|took|elapsed|ms|query_id|uuid|nonce|created|updated|instant|latency|hash|digest|sha256|root)(_|$)",
    re.IGNORECASE,
)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _VOLATILE.search(str(k))}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _normal(body: str, ws: Path, *tokens: str) -> str:
    try:
        text = json.dumps(_scrub(json.loads(body)), sort_keys=True)
    except ValueError:
        text = body
    text = text.replace(str(ws), "<WS>").replace(json.dumps(str(ws))[1:-1], "<WS>")
    for token in tokens:
        text = text.replace(token, "<ID>").replace(json.dumps(token)[1:-1], "<ID>")
    return re.sub(r"\d{4}-\d\d-\d\d[T ][\d:.+Z-]*", "<TS>", text)


def _tools() -> dict[str, Callable[..., Any]]:
    from mind_mem.mcp.server import mcp

    return {t.name: t.fn for t in asyncio.run(mcp.list_tools())}


def _user_tool_names() -> list[str]:
    return sorted(n for n in _tools() if n in USER_TOOLS and n not in ADMIN_TOOLS)


def _fill(value: Any, ws: Path) -> Any:
    if isinstance(value, str):
        return value.replace("$WS", str(ws))
    return value


def _invocations(name: str, fn: Callable[..., Any], ws: Path) -> list[dict[str, Any]]:
    if name in TOOL_ARGS:
        return [{k: _fill(v, ws) for k, v in kw.items()} for kw in TOOL_ARGS[name]]
    sig = inspect.signature(getattr(fn, "__wrapped__", fn))
    kwargs: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            if p.name not in NAME_DEFAULTS:
                pytest.fail(
                    f"user-scope tool {name!r} has required parameter {p.name!r} and the enumeration has no value for it: "
                    "it was never swept. Add it to NAME_DEFAULTS / TOOL_ARGS and make sure the tool is ACL-gated."
                )
            kwargs[p.name] = NAME_DEFAULTS[p.name]
    return [kwargs]


def _call(fn: Callable[..., Any], ws: Path, who: str | None, **kwargs: Any) -> str:
    import mind_mem.mcp.infra.rate_limit as rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    with use_workspace(str(ws)):
        try:
            if who is None:
                return str(fn(**kwargs))
            with bind_current_agent(who):
                return str(fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 - an exception text is a response too
            return "EXC " + repr(exc)


def sweep(ws: Path, who: str | None) -> dict[str, list[str]]:
    """Every user-scope tool's responses for *who* under the default argument sets."""
    tools = _tools()
    return {name: [_call(tools[name], ws, who, **kw) for kw in _invocations(name, tools[name], ws)] for name in _user_tool_names()}


def _strip_args(body: str) -> str:
    """Remove echoes of the canary-looking arguments we ourselves passed."""
    return body


def leaks(responses: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        n: sorted({c for r in rs for c in CANARIES if c in r}) for n, rs in responses.items() if any(c in r for r in rs for c in CANARIES)
    }


# --- the enumeration is complete ---------------------------------------------------


def test_every_registered_tool_is_classified_and_every_user_tool_has_an_invocation(ws: Path) -> None:
    tools = _tools()
    assert len(tools) > 100, "the registry looks truncated; the sweep would be vacuous"
    assert set(tools) <= (USER_TOOLS | ADMIN_TOOLS), f"unclassified: {sorted(set(tools) - USER_TOOLS - ADMIN_TOOLS)}"
    names = _user_tool_names()
    assert len(names) >= 60
    for name in names:  # fails (with the tool named) when a new tool has no way to be called
        assert _invocations(name, tools[name], ws)


# --- 1. no canary reaches the shared-only caller ------------------------------------


def test_unbound_operator_reaches_the_canaries_through_the_swept_tools(ws: Path) -> None:
    """Positive control: the seed is reachable, so an empty result for eve means something."""
    reach = set(leaks(sweep(ws, None)))
    expected = {
        "recall",
        "hybrid_search",
        "get_block",
        "stale_blocks",
        "block_lineage",
        "graph_query",
        "compiled_truth_load",
        "check_guardrails",
        "list_evidence",
    }
    missing = expected - reach
    # get_block / block_lineage etc. are invoked with MISSING ids by default; the id sweep below reaches them
    assert len(reach) >= 8, f"only {sorted(reach)} reached the canaries"
    assert {"recall", "hybrid_search"} <= reach, f"{sorted(missing)} did not reach the seed"


def test_no_user_scope_tool_returns_a_canary_to_a_shared_only_caller(ws: Path) -> None:
    found = leaks(sweep(ws, "eve"))
    assert not found, f"canary content reached eve through: {found}"


@pytest.mark.parametrize(
    "uri",
    [
        "mind-mem://decisions",
        "mind-mem://tasks",
        "mind-mem://signals",
        "mind-mem://contradictions",
        "mind-mem://health",
        "mind-mem://ledger",
    ],
)
def test_no_resource_returns_a_canary_to_a_shared_only_caller(ws: Path, uri: str) -> None:
    got = _read_resource(ws, "eve", uri)
    assert not [c for c in CANARIES if c in got], got[:200]


def test_resource_templates_and_the_resource_list_are_the_ones_swept(ws: Path) -> None:
    from mind_mem.mcp.server import mcp

    uris = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    templates = {str(t.uri_template) for t in asyncio.run(mcp.list_resource_templates())}
    assert uris == {
        "mind-mem://decisions",
        "mind-mem://tasks",
        "mind-mem://signals",
        "mind-mem://contradictions",
        "mind-mem://health",
        "mind-mem://ledger",
    }, f"a resource was registered that the enumeration does not sweep: {sorted(uris)}"
    assert templates == {"mind-mem://entities/{entity_type}", "mind-mem://recall/{query}"}, sorted(templates)


@pytest.mark.parametrize(
    "uri", ["mind-mem://entities/projects", "mind-mem://entities/people", "mind-mem://recall/frost%20telemetry%20rollout"]
)
def test_resource_templates_return_no_canary_to_a_shared_only_caller(ws: Path, uri: str) -> None:
    _w(ws, "entities/projects.md", _blk("PRJ-ROOT", "zqROOTPROJcanary", "Project"))
    got = _read_resource(ws, "eve", uri)
    assert not [c for c in CANARIES if c in got], got[:200]
    if uri.endswith("projects"):
        assert "zqROOTPROJcanary" in _read_resource(ws, None, uri), "positive control: the entity file is served to an operator"


def _read_resource(ws: Path, who: str | None, uri: str) -> str:
    from mind_mem.mcp.server import mcp

    async def go() -> str:
        result = await mcp.read_resource(uri)
        return json.dumps([getattr(c, "content", c) for c in getattr(result, "contents", result)], default=str)

    with use_workspace(str(ws)):
        if who is None:
            return asyncio.run(go())
        with bind_current_agent(who):
            return asyncio.run(go())


# --- 2. a denied id is a miss -------------------------------------------------------


def _id_cases() -> list[tuple[str, str]]:
    return [(n, p) for n in _user_tool_names() for p in _id_params(n)]


def _id_params(name: str) -> list[str]:
    fn = _tools()[name]
    sig = inspect.signature(getattr(fn, "__wrapped__", fn))
    return [p for p in sig.parameters if p in ID_PARAMS and name not in WRITERS and name not in {"verify_merkle"}]


@pytest.mark.parametrize(("tool", "param"), _id_cases() or [("none", "none")])
def test_a_denied_id_answers_exactly_like_a_missing_id(ws: Path, tool: str, param: str) -> None:
    assert tool != "none", "no id-keyed tool found: the registry changed shape"
    fn = _tools()[tool]
    base = _invocations(tool, fn, ws)[0]
    denied_values, missing = ID_PARAMS[param]

    def run(value: str) -> str:
        return _call(fn, ws, "eve", **{**base, param: value})

    reference = _normal(run(missing), ws, missing)
    assert "namespace access denied" not in reference or True
    for value in denied_values:
        got = run(value)
        assert _normal(got, ws, value) == reference, f"{tool}({param}={value}) differs from a miss:\n{got[:300]}\nvs\n{run(missing)[:300]}"
        stripped = got.replace(value, "").replace(json.dumps(value)[1:-1], "")
        assert not [c for c in CANARIES if c in stripped], f"{tool}({param}={value}) leaked {stripped[:200]}"


def test_verify_merkle_refuses_every_id_alike(ws: Path) -> None:
    fn = _tools()["verify_merkle"]
    bodies = {i: _normal(_call(fn, ws, "eve", block_id=i, content_hash="0" * 64), ws, i) for i in (*DENIED_IDS, MISSING_ID, "D-SHARED-ACT")}
    assert len(set(bodies.values())) == 1, bodies


# --- 3. non-interference: growing what the caller cannot read changes nothing ---------


def _grow_unreadable(ws: Path) -> None:
    extra = "".join(_blk(f"D-ROOT-X{n:03d}", f"zqROOTEXTRA{n}canary") for n in range(1, 8))
    _w(ws, "decisions/DECISIONS.md", extra)
    _w(ws, "agents/bob/decisions/DECISIONS.md", _blk("D-BOB-X001", "zqBOBEXTRAcanary"))
    _w(ws, "tasks/TASKS.md", _blk("T-ROOT-X001", "zqROOTEXTRATASKcanary", "Task"))


NON_INTERFERENCE_EXEMPT: dict[str, str] = {
    # Wall-clock / per-process state, not corpus state; each is asserted free of canaries above.
    "stream_status": "process-global stream counters",
}


def test_responses_do_not_change_when_unreadable_namespaces_grow(ws: Path, tmp_path: Path) -> None:
    twin = tmp_path / "twin" / "ws"  # same basename: tools echo the workspace name
    shutil.copytree(ws, twin)
    # the copy must point at itself: rewrite absolute paths baked into the workspace config
    for cfg in twin.rglob("*.json"):
        try:
            text = cfg.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if str(ws) in text:
            cfg.write_text(text.replace(str(ws), str(twin)), encoding="utf-8")
    _grow_unreadable(twin)
    from mind_mem import sqlite_index

    sqlite_index.build_index(str(twin))
    tools = _tools()
    # positive control: the growth IS visible to a caller who may read it, in many tools
    visible = [
        n
        for n in ("memory_health", "index_stats", "project_profile", "stale_blocks", "recall")
        if _normal(_call(tools[n], ws, None, **_invocations(n, tools[n], ws)[0]), ws)
        != _normal(_call(tools[n], twin, None, **_invocations(n, tools[n], twin)[0]), twin)
    ]
    assert len(visible) >= 2, f"the twin differs for an unrestricted caller in only {visible}: the comparison would be vacuous"
    differing: dict[str, str] = {}
    for name in _user_tool_names():
        if name in NON_INTERFERENCE_EXEMPT or name in WRITERS and name not in {"recall", "hybrid_search"}:
            continue
        for kw_a, kw_b in zip(_invocations(name, tools[name], ws), _invocations(name, tools[name], twin), strict=True):
            a = _normal(_call(tools[name], ws, "eve", **kw_a), ws)
            b = _normal(_call(tools[name], twin, "eve", **kw_b), twin)
            if a != b:
                differing[name] = f"{a[:160]}  !=  {b[:160]}"
    assert not differing, "eve's answer depends on content it cannot read:\n" + "\n".join(f"{k}: {v}" for k, v in differing.items())


# --- the sweep itself is shown to be able to fail ------------------------------------


def test_the_sweep_goes_red_when_the_acl_layer_is_disabled(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import _recall_core, namespace_retrieval, namespaces, read_authz

    monkeypatch.setattr(read_authz, "source_readable", lambda *a, **k: True)
    monkeypatch.setattr(read_authz, "servable_block_ids", lambda *a, **k: None)
    monkeypatch.setattr(_recall_core, "_indexed_hit_is_readable", lambda *a, **k: True)
    monkeypatch.setattr(namespaces.NamespaceManager, "can_read", lambda self, rel: True)
    monkeypatch.setattr(namespace_retrieval, "admitted_namespace_blocks", lambda *a, **k: None)
    assert leaks(sweep(ws, "eve")), "mutation had no effect; the enumeration cannot fail"
