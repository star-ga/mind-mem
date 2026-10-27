# Copyright 2026 STARGA, Inc.
"""No user-scope read surface returns an absolute host path.

Requirement, verbatim: never return absolute host paths to user scope.

Two halves:

* the model tools echo a path exactly as the caller supplied it, in every
  reply and every error, so ``~/key.pub`` never comes back as ``/home/x/key.pub``
  and a trailing-slash or ``//`` variant is not normalised;
* a sweep calls EVERY tool in ``USER_TOOLS`` (enumerated from the registry, so
  a new tool is covered automatically) and every ``mind-mem://`` resource, as an
  unbound caller and as a shared-only bound caller, on a workspace under a
  distinctive absolute path, and requires that neither that path nor ``$HOME``
  appears in any response.

Every negative assertion has a positive control: the same call, as a caller who
is granted the content, really ran and really returned data.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# Fixtures shared with the tool-level read-surface file.
from test_read_surface_acl_tools import call, registered, ws  # noqa: F401

from mind_mem.audit_context import bind_current_agent
from mind_mem.mcp.infra.acl import USER_TOOLS
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.path_redact import redact_host_paths, workspace_label

DENIED = "namespace access denied"

# --- 1. the model tools echo the caller's own spelling --------------------------

SPELLINGS = ("~/zq-absent/no-such-key.pub", "relative/zq-absent.pub", "/zq-absent//double//slash.pub", "/zq-absent/trailing/")


@pytest.fixture
def home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path_factory.mktemp("zq-home-distinctive")
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.mark.parametrize("who", [None, "eve"], ids=["unbound", "shared-only-bound"])
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_verify_model_tool_echoes_the_pubkey_path_as_supplied(
    ws: Path, home: Path, call: Callable[..., str], who: str | None, spelling: str
) -> None:
    body = call("verify_model_tool", who, path=str(ws / "shared"), pubkey_path=spelling)
    err = json.loads(body)["error"]
    assert err == f"pubkey_path not found: {spelling}", err
    assert str(home) not in body and str(ws) not in body.replace(str(ws / "shared"), "")


@pytest.mark.parametrize("who", [None, "eve"], ids=["unbound", "shared-only-bound"])
def test_verify_model_tool_wrong_length_error_does_not_resolve_the_key_path(
    ws: Path, home: Path, call: Callable[..., str], who: str | None
) -> None:
    (home / "k.pub").write_bytes(b"short")  # positive control: the file really exists
    body = call("verify_model_tool", who, path=str(ws / "shared"), pubkey_path="~/k.pub")
    if who is not None:
        # a bound caller may not read the home directory: it must read as absent
        assert json.loads(body)["error"] == "pubkey_path not found: ~/k.pub"
        return
    err = json.loads(body)["error"]
    assert "got 5" in err, "the unbound caller must really reach the length check"
    assert err.endswith("(~/k.pub)") and str(home) not in body


@pytest.mark.parametrize("who", [None, "eve"], ids=["unbound", "shared-only-bound"])
@pytest.mark.parametrize("variant", ["", "/", "//", "/."])
def test_model_tool_errors_keep_the_supplied_model_path(
    ws: Path, home: Path, call: Callable[..., str], who: str | None, variant: str
) -> None:
    gone = str(ws / "shared" / "zq-no-such-checkpoint") + variant
    for tool in ("audit_model_tool", "verify_model_tool"):
        body = call(tool, who, path=gone)
        assert gone in body, f"{tool} must echo {gone!r} unnormalised: {body}"
        assert str(home) not in body
    # a file given where a directory is expected keeps the caller's spelling too
    a_file = str(ws / "shared" / "decisions" / "DECISIONS.md")
    for tool in ("audit_model_tool", "verify_model_tool"):
        body = call(tool, who, path=a_file + variant if variant in ("/", "//") else a_file)
        assert str(ws.parent) not in body.replace(a_file, ""), f"{tool}: {body}"


def test_model_tools_reply_with_the_supplied_directory_when_it_works(ws: Path, call: Callable[..., str]) -> None:
    supplied = str(ws / "shared") + "//"
    got = json.loads(call("audit_model_tool", None, path=supplied))
    assert got["file_count"] >= 1, "positive control: the audit really ran"
    assert got["model_path"] == supplied


def test_sign_model_tool_echoes_key_file_as_supplied(
    ws: Path, home: Path, registered: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")  # sign_model_tool is admin-scope
    with use_workspace(str(ws)):
        body = registered["sign_model_tool"](path=str(ws / "shared"), key_file="~/zq-no-such.sk")
    assert json.loads(body)["error"] == "key_file not found: ~/zq-no-such.sk" and str(home) not in body
    (home / "k.sk").write_bytes(b"x")  # positive control: exists, wrong length
    with use_workspace(str(ws)):
        body = registered["sign_model_tool"](path=str(ws / "shared"), key_file="~/k.sk")
    assert "got 1" in body and str(home) not in body and "(~/k.sk)" in body


def test_sign_model_tool_replies_name_sidecars_relative_to_the_supplied_path(
    ws: Path, home: Path, registered: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    from mind_mem.model_signing import generate_keypair

    sk, _pk = generate_keypair()
    (home / "k.sk").write_bytes(sk)
    ckpt = ws / "shared" / "zq-ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    supplied = str(ckpt) + "//"
    with use_workspace(str(ws)):
        got = json.loads(registered["sign_model_tool"](path=supplied, key_file="~/k.sk"))
    assert got["ok"] is True and got["files_signed"] >= 1, got
    for field in ("manifest_path", "signature_path", "pubkey_path"):
        assert got[field].startswith(supplied), (field, got[field])
    assert str(home) not in json.dumps(got)


# --- 2. the shared redaction helper ---------------------------------------------


def test_redact_helper_rewrites_workspace_and_home_and_keeps_supplied_spelling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "zq-root" / "ws"
    root.mkdir(parents=True)
    h = tmp_path / "zq-h"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    text = f"in {root}/decisions/D.md and {h}/x and {root} end"
    out = redact_host_paths(text, workspace=str(root))
    assert str(root) not in out and str(h) not in out and "zq-root" not in out
    assert "decisions/D.md" in out and out.count("~") == 1
    assert workspace_label(str(root)) == "ws"
    # supplied spelling wins over any resolved form
    out = redact_host_paths(f"not found in {root.resolve()}", supplied={str(root.resolve()): "~/m//"})
    assert out == "not found in ~/m//"
    # a sibling directory that merely shares a prefix is NOT rewritten
    sib = f"{root}-other/x"
    assert redact_host_paths(sib, workspace=str(root)).endswith("-other/x") is True
    assert redact_host_paths("", workspace=str(root)) == ""


def test_redact_helper_is_the_only_redaction_the_tools_use() -> None:
    import mind_mem.mcp.tools.lint as lint_mod
    import mind_mem.mcp.tools.model as model_mod

    for mod in (lint_mod, model_mod):
        assert "redact_host_paths" in inspect.getsource(mod) or "workspace_label" in inspect.getsource(mod), mod.__name__


# --- 3. lint --------------------------------------------------------------------


def _enable_lint(ws: Path, enabled: bool = True) -> None:
    path = ws / "mind-mem.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cfg.setdefault("v4", {})["lint"] = {"enabled": enabled}
    path.write_text(json.dumps(cfg), encoding="utf-8")


@pytest.mark.parametrize("who", [None, "eve", "ops"])
def test_lint_never_returns_the_absolute_workspace(ws: Path, call: Callable[..., str], who: str | None) -> None:
    _enable_lint(ws, False)
    disabled = call("lint", who)
    assert "disabled" in disabled, "positive control: the disabled answer was reached"
    assert str(ws) not in disabled and str(ws.parent) not in disabled
    _enable_lint(ws)
    body = call("lint", who)
    if who == "eve":
        assert DENIED in body
    else:
        got = json.loads(body)
        assert "findings" in got, "positive control: the lint really ran"
        assert got["workspace"] == workspace_label(str(ws)) and not got["workspace"].startswith("/")
    assert str(ws) not in body and str(ws.parent) not in body


def test_lint_autofix_disabled_payload_has_no_path(ws: Path, registered: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")  # lint_autofix is admin-scope
    _enable_lint(ws, False)
    with use_workspace(str(ws)):
        body = registered["lint_autofix"](finding_id="LF-00000000")
    assert "disabled" in body and str(ws.parent) not in body


# --- 4. the sweep ---------------------------------------------------------------

_PATHY = {"path", "vault_root", "model_path", "directory", "dir", "root"}
_GOOD = {
    "block_id": "D-SHARED-ACT",
    "query": "frost telemetry",
    "text": "frost telemetry",
    "entity": "sharedent",
    "topic": "architecture",
    "source_block_id": "D-SHARED-ACT",
    "winner_id": "sharedent",
    "loser_id": "sharedcomp",
    "rationale": "a sufficiently long rationale",
    "filename": "zq-absent-1.0.mmcore",
    "name": "zq",
    "kind": "decision",
    "namespace": "shared",
    "content": "Statement: zq\n",
    "message": "frost telemetry",
    "question": "frost telemetry",
    "subject": "sharedent",
    "predicate": "related_to",
    "object": "sharedcomp",
    "action": "list",
}


def _args_for(fn: Callable[..., Any], ws: Path, *, hostile: bool) -> dict[str, Any]:
    """Required arguments for *fn*; ``hostile`` points every path-like one at a missing host path."""
    out: dict[str, Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        pathy = pname in _PATHY or pname.endswith("_path") or pname.endswith("_file") or pname.endswith("_dir")
        if pathy and (hostile or param.default is inspect.Parameter.empty):
            # relative to the cwd (the workspace): a caller's own absolute spelling is echoed by design
            out[pname] = "shared/zq-absent" if hostile else "shared"
            if pname in ("pubkey_path", "key_file"):
                out[pname] = "~/zq-absent.key"
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if pname in _GOOD:
            out[pname] = _GOOD[pname]
            continue
        ann = str(param.annotation)
        out[pname] = (
            1
            if "int" in ann
            else 1.0
            if "float" in ann
            else False
            if "bool" in ann
            else []
            if "list" in ann
            else {}
            if "dict" in ann
            else "zq"
        )
    return out


def _resource_uris(ws: Path) -> list[str]:
    from mind_mem.mcp.server import mcp

    uris = [str(r.uri) for r in asyncio.run(mcp.list_resources())]
    for t in asyncio.run(mcp.list_resource_templates()):
        uris.append(t.uri_template.replace("{entity_type}", "projects").replace("{query}", "frost%20telemetry"))
    return uris


def _read_resource(uri: str) -> str:
    from mind_mem.mcp.server import mcp

    result = asyncio.run(mcp.read_resource(uri))
    return "\n".join(str(getattr(c, "content", c)) for c in result.contents)


def _sweep(ws: Path, registered: dict[str, Callable[..., Any]], who: str | None) -> dict[str, str]:
    from mind_mem.mcp.infra import rate_limit

    seen: dict[str, str] = {}

    def fresh() -> None:
        # 140+ calls would otherwise trip the per-client limiter and turn the
        # tail of the sweep into "Rate limit exceeded" -- a vacuous pass.
        rate_limit._rate_limiters.clear()

    def run() -> None:
        for name in sorted(USER_TOOLS):
            fn = registered.get(name)
            if fn is None:
                continue
            # dispatcher tools take an ``action``: sweep every read branch of it
            actions = ("list", "stats", "query", "traverse", "load") if "action" in inspect.signature(fn).parameters else (None,)
            for hostile in (False, True):
                for action in actions:
                    label = f"{name}[{'hostile' if hostile else 'plain'}{'/' + action if action else ''}]"
                    fresh()
                    kwargs = _args_for(fn, ws, hostile=hostile)
                    if action:
                        kwargs["action"] = action
                    try:
                        seen[label] = str(fn(**kwargs))
                    except Exception as exc:  # noqa: BLE001 - an exception text is still a response
                        seen[label] = f"EXC {type(exc).__name__}: {exc}"
        for uri in _resource_uris(ws):
            fresh()
            try:
                seen[f"resource {uri}"] = _read_resource(uri)
            except Exception as exc:  # noqa: BLE001
                seen[f"resource {uri}"] = f"EXC {type(exc).__name__}: {exc}"

    with use_workspace(str(ws)):
        if who is None:
            run()
        else:
            with bind_current_agent(who):
                run()
    return seen


@pytest.fixture
def sweep_env(ws: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(ws)
    _enable_lint(ws)
    monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", str(ws))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    return ws


def test_the_sweep_enumerates_the_registry_not_a_hand_list(registered: dict[str, Callable[..., Any]]) -> None:
    missing = sorted(USER_TOOLS - set(registered))
    # consolidated dispatchers etc. may be registered under the same names; none may silently drop out
    assert len(USER_TOOLS & set(registered)) >= 50, f"the sweep would cover too few tools; unregistered: {missing}"
    uris = _resource_uris(Path("/unused"))
    assert len(uris) >= 8


@pytest.mark.parametrize("who", [None, "eve"], ids=["unbound-user-scope", "shared-only-bound"])
def test_no_user_scope_tool_or_resource_returns_an_absolute_host_path(
    sweep_env: Path, home: Path, registered: dict[str, Callable[..., Any]], who: str | None
) -> None:
    ws = sweep_env
    seen = _sweep(ws, registered, who)
    # positive controls: the sweep really reached content, not a wall of errors
    limited = sorted(k for k, b in seen.items() if "Rate limit exceeded" in b)
    assert not limited, f"the sweep was rate limited, so these responses prove nothing: {limited}"
    joined = "\n".join(seen.values())
    assert "zqSHAREDACTIVEcanary" in joined or "D-SHARED-ACT" in joined, "the sweep never read the seeded corpus"
    assert sum(1 for b in seen.values() if not b.startswith("EXC")) >= 40
    markers = {str(ws), os.path.realpath(ws), str(ws.parent), str(home), os.path.realpath(home)}
    hits = {label: [m for m in markers if m in body] for label, body in seen.items()}
    hits = {k: v for k, v in hits.items() if v}
    assert not hits, f"absolute host paths in user-scope responses: {hits}"


def test_the_sweep_can_fail_when_a_tool_leaks(
    sweep_env: Path, registered: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: a tool that echoes the workspace must be caught by the same comparison."""
    ws = sweep_env
    leaky = dict(registered)
    leaky["lint"] = lambda **_kw: json.dumps({"workspace": str(ws)})
    seen = _sweep(ws, leaky, None)
    assert str(ws) in seen["lint[plain]"]


def test_vault_scan_refusal_echoes_the_supplied_root(sweep_env: Path, call: Callable[..., str]) -> None:
    ws = sweep_env
    control = call("vault_scan", None, vault_root="shared")
    assert "DECISIONS.md" in control and str(ws) not in control, "positive control: the scan ran"
    for who in (None, "eve"):
        gone = call("vault_scan", who, vault_root="shared//zq-absent/")
        assert json.loads(gone)["error"].endswith("vault root not found: shared//zq-absent/") or "zq-absent" in gone
        assert str(ws) not in gone
    denied = call("vault_scan", "eve", vault_root="agents/bob/")
    assert json.loads(denied)["error"] == "vault root not found: agents/bob/", "denied reads like absent, in the caller's own spelling"
