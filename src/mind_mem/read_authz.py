# Copyright 2026 STARGA, Inc.
"""One namespace-ACL decision for every surface that reads the corpus.

The MCP tool path decides "may THIS caller read bytes from THIS source" with
:class:`~mind_mem.namespaces.NamespaceManager` plus the realpath check in
:func:`mind_mem._recall_core._indexed_hit_is_readable`. Two other read
surfaces -- the REST ``/v1`` endpoints and the ``mind-mem://`` MCP resources --
used to skip it: REST never bound the authenticated principal around the tool
body (the dependency that authenticates runs in a different threadpool context
from the endpoint), so the tool saw an unbound caller and answered workspace-wide;
resources read fixed root-level corpus files without asking at all.

This module is the seam both now go through, so a surface cannot be "filtered
by whoever remembered". Status admission (quarantined / pending) is the
separate, equally shared :func:`mind_mem.admission.admit_read`; a surface needs
both.

Semantics, identical to the tool path:

* an UNBOUND caller (no transport identity: stdio, a local operator) keeps
  workspace-level access;
* a BOUND caller reads a source only if its ACL grants the lexical path AND the
  resolved real path;
* a caller whose identity cannot be established is DENIED, never promoted to
  unbound.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .audit_context import UNATTRIBUTED


class CallerUnresolved(Exception):
    """The caller's identity could not be established; the read must be refused."""


def caller_principal() -> str | None:
    """The authenticated namespace principal, ``None`` for an unbound caller.

    Raises:
        CallerUnresolved: the authentication context exists but cannot be
            inspected. Treating that as "unbound" would widen a read to the
            whole workspace after an authentication failure.
    """
    from .mcp.infra.acl import authenticated_agent_id

    try:
        principal = authenticated_agent_id()
    except ValueError as exc:
        raise CallerUnresolved("authentication context unavailable") from exc
    if principal in {None, "", UNATTRIBUTED}:
        return None
    return principal


def is_bound(principal: str | None) -> bool:
    return principal not in {None, "", UNATTRIBUTED}


def source_readable(workspace: str, rel_source: str, principal: str | None, *, check_realpath: bool = True) -> bool:
    """May *principal* read corpus bytes from workspace-relative *rel_source*?"""
    from ._recall_core import _indexed_hit_is_readable
    from .namespaces import InvalidAgentIdError, NamespaceManager

    agent = principal if is_bound(principal) else None
    try:
        manager = NamespaceManager(workspace, agent_id=agent)
    except InvalidAgentIdError:
        return False
    source = rel_source.replace("\\", "/")
    if not manager.can_read(source):
        return False
    if agent is None:
        return True
    return _indexed_hit_is_readable(workspace, {"file": source}, manager, check_realpath=check_realpath)


def block_source(block: dict[str, Any]) -> str | None:
    """The workspace-relative source a block claims, else ``None``."""
    for key in ("_source_file", "_source", "file"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def readable_blocks(
    workspace: str, blocks: Iterable[dict[str, Any]], principal: str | None, *, default_source: str | None = None
) -> list[dict[str, Any]]:
    """Blocks of *blocks* whose source *principal* may read. Unresolvable -> dropped."""
    if not is_bound(principal):
        return list(blocks)
    kept: list[dict[str, Any]] = []
    for block in blocks:
        source = block_source(block) or default_source
        if not source:
            continue
        if os.path.isabs(source):
            real = os.path.realpath(source)
            root = os.path.realpath(workspace)
            if not real.startswith(root + os.sep):
                continue
            source = os.path.relpath(real, root)
        if source_readable(workspace, source, principal):
            kept.append(block)
    return kept


@dataclass(frozen=True)
class ReadScope:
    """What one caller may read, resolved once per tool call.

    ``servable`` is the set of block ids the caller's namespace ACL grants AND
    the admission filter admits (the same funnel ranked recall uses), or
    ``None`` for an unbound caller, which keeps workspace-level access.
    ``whole`` is true when the caller's view IS the workspace's view: an
    unbound caller, or a bound one that can read every admissible block. A
    tool that returns a count, a ranking or a health figure over the whole
    corpus is only honest for such a caller; for any other it must filter by
    id or refuse.
    """

    workspace: str
    principal: str | None
    servable: frozenset[str] | None
    whole: bool

    @property
    def restricted(self) -> bool:
        return self.servable is not None

    def allows(self, block_id: object) -> bool:
        """May the caller see this block id? Non-string ids are never allowed."""
        if self.servable is None:
            return True
        return isinstance(block_id, str) and block_id in self.servable

    def keep(self, rows: Iterable[Any], key: Any = None) -> list[Any]:
        """The rows whose id (``key(row)``, or the row itself) the caller may see."""
        pick = key if key is not None else (lambda row: row)
        return [row for row in rows if self.allows(pick(row))]


def servable_block_ids(workspace: str, principal: str | None) -> frozenset[str] | None:
    """Ids *principal* may read, ``None`` for an unbound (unrestricted) caller."""
    if not is_bound(principal):
        return None
    from .namespace_retrieval import admitted_namespace_blocks

    return frozenset(admitted_namespace_blocks(workspace, principal) or ())


def read_scope(workspace: str) -> ReadScope:
    """Resolve the current caller's :class:`ReadScope`.

    Raises:
        CallerUnresolved: the caller's identity cannot be established; the
            read must be refused, never promoted to unbound.
    """
    principal = caller_principal()
    servable = servable_block_ids(workspace, principal)
    if servable is None:
        return ReadScope(workspace, principal, None, True)
    return ReadScope(workspace, principal, servable, whole_view(workspace, principal))


def whole_view(workspace: str, principal: str | None) -> bool:
    """True when *principal* can READ every namespace source of the workspace.

    "Whole" is a statement about the ACL, not about which blocks happen to be
    admitted today: the root corpus, ``shared``, every ``agents/*`` directory
    and every declared custom namespace must each be readable. Deriving it
    from admissible blocks made a caller "whole" whenever the namespaces it
    could not read held only quarantined / pending blocks, which then served
    it every workspace-wide aggregate (query log, calibration, index stats).
    Unbound callers are whole by definition.
    """
    if not is_bound(principal):
        return True
    return all(source_readable(workspace, src, principal) for src in _namespace_sources(workspace))


def _namespace_sources(workspace: str) -> list[str]:
    """One probe per namespace root plus every corpus file registered under it."""
    from .corpus_registry import discover_corpus_files

    root = os.path.realpath(workspace)
    sources: list[str] = ["decisions/DECISIONS.md"]
    for label, rel in discover_corpus_files(root):
        sources.append(rel.replace(os.sep, "/"))
    for ns in _other_namespace_roots(workspace):
        sources.append(f"{ns}/decisions/DECISIONS.md")
        ns_path = os.path.join(root, *ns.split("/"))
        if os.path.islink(ns_path) or not os.path.isdir(ns_path):
            continue
        for _label, rel in discover_corpus_files(ns_path):
            sources.append(f"{ns}/{rel.replace(os.sep, '/')}")
    return list(dict.fromkeys(sources))


def readable_namespace_roots(workspace: str, principal: str | None) -> tuple[str, ...]:
    """Namespace roots *principal* can read: ``""`` is the workspace root.

    A compact fingerprint of what a caller could have drawn free text from.
    Recorded on a staged proposal so another viewer is shown it only when that
    viewer can read at least everything the proposer could.
    """
    probes = {"": "decisions/DECISIONS.md"}
    for ns in _other_namespace_roots(workspace):
        probes[ns] = f"{ns}/decisions/DECISIONS.md"
    return tuple(ns for ns, src in probes.items() if source_readable(workspace, src, principal))


def roots_readable(workspace: str, roots: Iterable[object], principal: str | None) -> bool:
    """True when *principal* can read every namespace root in *roots*.

    A malformed entry is unreadable: the check fails closed.
    """
    if not is_bound(principal):
        return True
    for ns in roots:
        if not isinstance(ns, str):
            return False
        src = f"{ns}/decisions/DECISIONS.md" if ns else "decisions/DECISIONS.md"
        if not source_readable(workspace, src, principal):
            return False
    return True


#: Upper bound on files examined when proving a whole directory is readable.
MAX_ACL_WALK_FILES = 200_000


def path_readable(
    workspace: str,
    path: str,
    principal: str | None,
    *,
    want_file: bool = False,
    extra_roots: Iterable[str] = (),
    outside_ok: bool = False,
) -> bool:
    """May *principal* look at filesystem *path* (and, for a directory, everything under it)?

    For tools that take an arbitrary host path (model checkpoints, key files,
    vault roots). An unbound caller keeps today's behaviour. A bound caller may
    touch a path only if it is (a) under one of *extra_roots* -- directories an
    operator designated in configuration, (b) inside the workspace with every
    file under it readable by the namespace ACL, or (c) outside the workspace
    when *outside_ok* (the caller already passed an operator allowlist).
    Anything else is refused, and the caller must answer a refusal exactly as it
    answers a path that does not exist.
    """
    if not is_bound(principal):
        return True
    ws = os.path.realpath(workspace)
    real = os.path.realpath(os.path.expanduser(path))
    for allowed in extra_roots:
        allowed = os.path.realpath(allowed)
        if real == allowed or real.startswith(allowed + os.sep):
            return True
    if real != ws and not real.startswith(ws + os.sep):
        return outside_ok
    if real == ws:
        rel = ""
    else:
        rel = os.path.relpath(real, ws).replace(os.sep, "/")
    if os.path.isfile(real):
        return bool(rel) and source_readable(ws, rel, principal)
    if want_file:
        return False
    if not source_readable(ws, (rel + "/" if rel else "") + ".dir", principal):
        return False
    seen = 0
    for dirpath, _dirs, files in os.walk(real, followlinks=False):
        for name in files:
            seen += 1
            if seen > MAX_ACL_WALK_FILES:
                return False
            member = os.path.relpath(os.path.join(dirpath, name), ws).replace(os.sep, "/")
            if not source_readable(ws, member, principal):
                return False
    return True


def _other_namespace_roots(workspace: str) -> list[str]:
    """Namespace roots besides the workspace root: shared, declared custom, every agent dir."""
    from .namespace_retrieval import declared_custom_namespaces
    from .namespaces import InvalidAgentIdError, _validate_agent_id
    from .storage import _load_workspace_config

    roots = ["shared", *declared_custom_namespaces(_load_workspace_config(workspace, quiet=True))]
    agents = os.path.join(os.path.realpath(workspace), "agents")
    if not os.path.islink(agents) and os.path.isdir(agents):
        for name in sorted(os.listdir(agents)):
            try:
                _validate_agent_id(name)
            except InvalidAgentIdError:
                continue
            roots.append("agents/" + name)
    return list(dict.fromkeys(roots))


def readable_sources(workspace: str, sources: Iterable[str], principal: str | None) -> tuple[str, ...]:
    """The workspace-relative *sources* that *principal* may read, order preserved."""
    return tuple(src for src in sources if source_readable(workspace, src, principal))


#: Refusal text for a workspace-level transport. Deliberately content-free.
WORKSPACE_LEVEL_TRANSPORT_REFUSAL = "namespace ACL is configured; this transport has no per-caller identity and is refused"


def namespace_acl_configured(workspace: str) -> bool:
    """True when the workspace declares per-agent namespace restrictions.

    A transport that authenticates with one flat credential (the stdlib HTTP
    transport, gRPC) has no principal to put in front of
    :class:`~mind_mem.namespaces.NamespaceManager`; it reads as an unbound,
    workspace-level caller. That is a coherent contract only while no
    per-agent restriction exists. Once ``mind-mem-acl.json`` declares agents
    (or ``default_policy: deny``) such a transport would silently return what
    the MCP tool path denies, so it must refuse instead.

    An ACL file that exists but cannot be parsed counts as configured:
    "I could not tell" is not "there is no restriction".
    """
    import json

    path = os.path.join(workspace, "mind-mem-acl.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            acl = json.load(handle)
    except (OSError, ValueError):
        return True
    if not isinstance(acl, dict):
        return True
    agents = acl.get("agents")
    if isinstance(agents, dict) and agents:
        return True
    return acl.get("default_policy") == "deny"
