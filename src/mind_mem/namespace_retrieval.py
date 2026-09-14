"""Configuration and enforcement for namespace retrieval properties.

This module is deliberately retrieval-only.  Namespace ACLs and admission remain
owned by :mod:`mind_mem.namespaces` and :mod:`mind_mem.admissibility`; these
helpers only decide whether an already ACL-visible block may enter search and
how a bounded behaviour context is added to a pack.
"""

from __future__ import annotations

import fnmatch
import math
import os
from collections.abc import Mapping
from typing import Any

REACHABILITY_SEARCHABLE = "searchable"
REACHABILITY_DIRECT_ONLY = "direct-only"
REACHABILITY_ALWAYS_INJECTED = "always-injected"
REACHABILITIES = frozenset(
    {REACHABILITY_SEARCHABLE, REACHABILITY_DIRECT_ONLY, REACHABILITY_ALWAYS_INJECTED}
)
FLOOR_NONE = "none"
FLOOR_INHERIT_GLOBAL = "inherit-global"
DEFAULT_DECLARATION = {"reachability": REACHABILITY_SEARCHABLE, "floor": FLOOR_INHERIT_GLOBAL}
_MAX_ALWAYS_ITEMS = 32


def _properties(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the validated-shaped retrieval declarations, or an empty map."""
    if not isinstance(config, Mapping):
        return {}
    recall = config.get("recall")
    if not isinstance(recall, Mapping):
        return {}
    props = recall.get("namespace_properties")
    if props is not None and not isinstance(props, Mapping):
        raise ValueError("recall.namespace_properties must be an object")
    return props if isinstance(props, Mapping) else {}


def namespace_for_path(path: object) -> str:
    """Map a workspace-relative source path to its namespace identity."""
    if not isinstance(path, str):
        return "workspace"
    cleaned = path.replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("/") or ".." in cleaned.split("/"):
        return "workspace"
    parts = [part for part in cleaned.split("/") if part]
    if not parts:
        return "workspace"
    if parts[0] == "agents" and len(parts) >= 2:
        return f"agents/{parts[1]}"
    if parts[0] in {
        "decisions",
        "tasks",
        "entities",
        "intelligence",
        "memory",
        "summaries",
        "maintenance",
        ".mind-mem-index",
    }:
        return "workspace"
    # Additional top-level roots (for example ``always`` or ``direct``)
    # are explicit namespace identities rather than workspace corpus files.
    return parts[0]


def declaration_for(config: Mapping[str, Any] | None, namespace: str) -> dict[str, Any]:
    """Resolve one exact or glob declaration, defaulting to searchable."""
    props = _properties(config)
    defaults = props.get("defaults")
    out: dict[str, Any] = dict(DEFAULT_DECLARATION)
    if isinstance(defaults, Mapping):
        _validate_declaration(defaults, "defaults")
        _merge_valid(out, defaults)
    elif defaults is not None:
        raise ValueError("namespace_properties.defaults must be an object")
    for name, candidate in props.items():
        if name == "defaults":
            continue
        if not isinstance(name, str) or not isinstance(candidate, Mapping):
            raise ValueError("namespace_properties declarations must map names to objects")
        _validate_declaration(candidate, name)
    # Exact declarations win.  Otherwise the first matching glob is used in
    # insertion order, so a config remains deterministic and reviewable.
    chosen: Mapping[str, Any] | None = None
    exact = props.get(namespace)
    if isinstance(exact, Mapping):
        chosen = exact
    else:
        for pattern, candidate in props.items():
            if pattern in {"defaults", namespace} or not isinstance(candidate, Mapping):
                continue
            if isinstance(pattern, str) and fnmatch.fnmatchcase(namespace, pattern):
                chosen = candidate
                break
    if chosen is not None:
        _merge_valid(out, chosen)
        out["_configured"] = True
    else:
        out["_configured"] = False
    return out


def _merge_valid(target: dict[str, Any], candidate: Mapping[str, Any]) -> None:
    reachability = candidate.get("reachability")
    if isinstance(reachability, str) and reachability in REACHABILITIES:
        target["reachability"] = reachability
    floor = candidate.get("floor")
    if floor == FLOOR_NONE or floor == FLOOR_INHERIT_GLOBAL:
        target["floor"] = floor
    elif isinstance(floor, (int, float)) and not isinstance(floor, bool) and math.isfinite(float(floor)):
        evidence = candidate.get("evidence")
        # Numeric floors are accepted only with a human-readable measurement
        # reference.  This is a declaration gate, not a tuning guess.
        if isinstance(evidence, (str, Mapping)) and bool(evidence):
            value = float(floor)
            if 0.0 <= value <= 1_000_000.0:
                target["floor"] = value
                target["evidence"] = evidence
    for key in ("max_items", "content_type"):
        if key in candidate:
            target[key] = candidate[key]


def _validate_declaration(candidate: Mapping[str, Any], name: str) -> None:
    """Reject malformed declarations instead of silently using a weaker one."""
    if "reachability" in candidate and candidate["reachability"] not in REACHABILITIES:
        raise ValueError(f"namespace_properties.{name}.reachability is invalid")
    if "floor" in candidate:
        floor = candidate["floor"]
        if floor != FLOOR_NONE and floor != FLOOR_INHERIT_GLOBAL:
            if (
                isinstance(floor, bool)
                or not isinstance(floor, (int, float))
                or not math.isfinite(float(floor))
                or not 0.0 <= float(floor) <= 1_000_000.0
            ):
                raise ValueError(f"namespace_properties.{name}.floor is invalid")
            if not isinstance(candidate.get("evidence"), (str, Mapping)) or not candidate.get("evidence"):
                raise ValueError(f"namespace_properties.{name}.floor requires non-empty evidence")
    if "max_items" in candidate:
        value = candidate["max_items"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_ALWAYS_ITEMS:
            raise ValueError(f"namespace_properties.{name}.max_items must be an integer in [1, 32]")
    if "content_type" in candidate and not isinstance(candidate["content_type"], str):
        raise ValueError(f"namespace_properties.{name}.content_type must be a string")


def _global_floor(config: Mapping[str, Any] | None) -> float:
    recall = config.get("recall") if isinstance(config, Mapping) else None
    value = recall.get("min_score", 0.0) if isinstance(recall, Mapping) else 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def namespace_search_allowed(path: object, config: Mapping[str, Any] | None) -> bool:
    """Whether a source path may enter ranked search."""
    reachability = declaration_for(config, namespace_for_path(path))["reachability"]
    return reachability not in {REACHABILITY_DIRECT_ONLY, REACHABILITY_ALWAYS_INJECTED}


def filter_search_hits(hits: list[dict[str, Any]], config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Remove direct-only/always-injected namespaces and apply declared floors."""
    # Preserve the historical unconfigured pipeline byte-for-byte.  In
    # particular, do not require backend source metadata or add marker keys
    # unless a namespace declaration actually opts into this surface.
    if not _properties(config):
        return hits
    out: list[dict[str, Any]] = []
    global_floor = _global_floor(config)
    for hit in hits:
        source = hit.get("_source_file") or hit.get("file")
        if not isinstance(source, str) or not source.strip():
            # A backend result without a source cannot be bound to a declared
            # namespace.  Serving it would let forged/default metadata bypass
            # a direct-only declaration.
            continue
        declaration = declaration_for(config, namespace_for_path(source))
        if declaration["reachability"] in {REACHABILITY_DIRECT_ONLY, REACHABILITY_ALWAYS_INJECTED}:
            continue
        floor = declaration.get("floor", FLOOR_INHERIT_GLOBAL)
        floor_override = floor == FLOOR_NONE or isinstance(floor, (int, float))
        hit["_namespace_floor_override"] = floor_override
        hit["_namespace_floor_none"] = floor == FLOOR_NONE
        threshold = global_floor if floor == FLOOR_INHERIT_GLOBAL else None if floor == FLOOR_NONE else float(floor)
        if threshold is not None:
            try:
                if float(hit.get("score", 0.0)) < threshold:
                    continue
            except (TypeError, ValueError):
                continue
        out.append(hit)
    return out


def always_injected_hits(
    workspace: str,
    config: Mapping[str, Any] | None,
    *,
    agent_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read bounded, admitted behaviour blocks from configured always namespaces."""
    from .admissibility import admit_corpus
    from .block_parser import parse_file
    from .block_store import MarkdownBlockStore
    from .namespaces import NamespaceManager

    selected: list[dict[str, Any]] = []
    selected_identities: set[tuple[str, str, str]] = set()
    remaining_global = _MAX_ALWAYS_ITEMS
    acl = NamespaceManager(workspace, agent_id=agent_id)
    declarations = _properties(config)
    import glob

    for namespace, raw in declarations.items():
        if remaining_global <= 0:
            break
        if namespace == "defaults" or not isinstance(namespace, str) or not isinstance(raw, Mapping):
            continue
        declaration = declaration_for(config, namespace)
        if declaration["reachability"] != REACHABILITY_ALWAYS_INJECTED:
            continue
        if declaration.get("content_type") != "behavior":
            continue
        try:
            cap = int(declaration.get("max_items", 0))
        except (TypeError, ValueError):
            cap = 0
        if cap < 1 or cap > _MAX_ALWAYS_ITEMS:
            continue
        normalized = namespace.replace("\\", "/").strip("/")
        if not normalized or normalized in {".", ".."} or normalized.startswith("../") or "/../" in f"/{normalized}/":
            continue
        root = os.path.realpath(workspace)
        if os.path.isabs(normalized) or ".." in normalized.split("/"):
            continue
        candidates = (
            [os.path.join(root, normalized)]
            if not any(ch in normalized for ch in "*?[")
            else sorted(glob.glob(os.path.join(root, normalized)))
        )
        matched = 0
        for candidate in candidates:
            namespace_root = os.path.realpath(candidate)
            if not namespace_root.startswith(root + os.sep) or not os.path.isdir(namespace_root) or os.path.islink(candidate):
                continue
            actual_namespace = os.path.relpath(namespace_root, root).replace(os.sep, "/")
            # Apply the same namespace ACL before reading a configured source.
            # The workspace-level manager permits all in-workspace paths; an
            # agent-scoped caller therefore cannot use an always declaration as
            # a second, wider discovery mechanism.
            if not acl.can_read(actual_namespace):
                continue
            try:
                store = MarkdownBlockStore(namespace_root)
                blocks: list[dict[str, Any]] = []
                for source_path in store.list_blocks():
                    rel_source = os.path.relpath(source_path, root).replace(os.sep, "/")
                    blocks.extend({**block, "_source_file": rel_source} for block in parse_file(source_path))
                blocks = admit_corpus(blocks)
            except (OSError, ValueError):
                continue
            # Bind the namespace prefix before the shared revocation check;
            # otherwise a duplicate credential ID in the root corpus could
            # decide the status of this explicitly configured source.
            from .content_lifecycle import content_identity, filter_revoked_credentials

            for block in blocks:
                source = block.get("_source_file") or ""
                source = str(source).replace("\\", "/").strip("/")
                block["_source_file"] = (
                    source
                    if source == actual_namespace or source.startswith(actual_namespace + "/")
                    else f"{actual_namespace}/{source}"
                )
            blocks = filter_revoked_credentials(blocks, workspace)
            namespace_count = 0
            for block in blocks:
                if remaining_global <= 0:
                    break
                kind = str(block.get("Type", block.get("type", ""))).strip().lower()
                if kind not in {"behavior", "behaviour"}:
                    continue
                block_id = block.get("_id") or block.get("id")
                if not isinstance(block_id, str) or not block_id:
                    continue
                identity = content_identity(block)
                if identity is None or identity in selected_identities:
                    continue
                if not acl.can_read(str(block.get("_source_file") or "")):
                    continue
                excerpt = block.get("Statement") or block.get("Summary") or block.get("Description") or ""
                selected.append(
                    {
                        "_id": block_id,
                        "type": "Behavior",
                        "score": 0.0,
                        "excerpt": str(excerpt),
                        "file": str(block.get("_source_file") or actual_namespace).rstrip("/"),
                        "line": int(block.get("_line", 0) or 0),
                        "status": str(block.get("Status", "") or ""),
                        "_namespace_reachability": REACHABILITY_ALWAYS_INJECTED,
                    }
                )
                selected_identities.add(identity)
                namespace_count += 1
                remaining_global -= 1
                if namespace_count >= cap:
                    break
            matched += min(namespace_count, cap)
        if matched == 0:
            continue
    cap_total = 0
    for name in declarations:
        if not isinstance(name, str) or name == "defaults":
            continue
        resolved = declaration_for(config, name)
        if resolved["reachability"] != REACHABILITY_ALWAYS_INJECTED:
            continue
        try:
            declared_cap = int(resolved.get("max_items", 0) or 0)
        except (TypeError, ValueError):
            declared_cap = 0
        if 1 <= declared_cap <= _MAX_ALWAYS_ITEMS:
            normalized = name.replace("\\", "/").strip("/")
            matches = (
                [normalized]
                if not any(ch in normalized for ch in "*?[")
                else [
                    os.path.relpath(p, os.path.realpath(workspace)).replace(os.sep, "/")
                    for p in glob.glob(os.path.join(os.path.realpath(workspace), normalized))
                    if os.path.isdir(p)
                ]
            )
            cap_total += declared_cap * len(matches)
    return selected, {"count": len(selected), "cap": min(cap_total, _MAX_ALWAYS_ITEMS), "content_type": "behavior"}


__all__ = [
    "REACHABILITY_SEARCHABLE",
    "REACHABILITY_DIRECT_ONLY",
    "REACHABILITY_ALWAYS_INJECTED",
    "FLOOR_NONE",
    "FLOOR_INHERIT_GLOBAL",
    "declaration_for",
    "namespace_for_path",
    "filter_search_hits",
    "namespace_search_allowed",
    "always_injected_hits",
]
