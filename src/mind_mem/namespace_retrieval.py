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
from typing import Any, cast

REACHABILITY_SEARCHABLE = "searchable"
REACHABILITY_DIRECT_ONLY = "direct-only"
REACHABILITY_ALWAYS_INJECTED = "always-injected"
REACHABILITIES = frozenset({REACHABILITY_SEARCHABLE, REACHABILITY_DIRECT_ONLY, REACHABILITY_ALWAYS_INJECTED})
FLOOR_NONE = "none"
FLOOR_INHERIT_GLOBAL = "inherit-global"
DEFAULT_DECLARATION = {"reachability": REACHABILITY_SEARCHABLE, "floor": FLOOR_INHERIT_GLOBAL}
_MAX_ALWAYS_ITEMS = 32
_BUILTIN_NAMESPACE_DECLARATIONS = frozenset({"defaults", "workspace", "shared", "agents/*"})


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


def namespace_for_path(path: object) -> str | None:
    """Map a canonical workspace-relative source path to its namespace.

    Invalid source metadata is unresolved rather than silently promoted to the
    workspace namespace.  That distinction matters when an opted-in policy
    declares an agent namespace as direct-only.
    """
    if not isinstance(path, str):
        return None
    cleaned = path.replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or "\x00" in cleaned:
        return None
    if cleaned.startswith("/") or (len(cleaned) >= 2 and cleaned[0].isalpha() and cleaned[1] == ":") or ".." in cleaned.split("/"):
        return None
    parts = cleaned.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
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


def declared_custom_namespaces(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return explicitly declared, top-level custom namespace roots.

    The standard workspace/shared/agent roots have dedicated discovery
    rules.  A custom root is discoverable only when its exact name appears in
    ``recall.namespace_properties``; arbitrary directories and glob patterns
    are never promoted into the corpus by this helper.
    """
    props = _properties(config)
    roots: list[str] = []
    for name, candidate in props.items():
        if name in _BUILTIN_NAMESPACE_DECLARATIONS:
            continue
        if not isinstance(name, str) or not isinstance(candidate, Mapping):
            continue
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or any(ch in name for ch in "*?[")
            or name.startswith(".")
            or "\x00" in name
        ):
            continue
        _validate_declaration(candidate, name)
        roots.append(name)
    return tuple(sorted(roots))


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
    if "reachability" in candidate and (not isinstance(candidate["reachability"], str) or candidate["reachability"] not in REACHABILITIES):
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
    if not _properties(config):
        # Preserve the historical unconfigured pipeline, including its
        # treatment of backend paths that predate source metadata validation.
        return True
    namespace = namespace_for_path(path)
    if namespace is None:
        return False
    reachability = declaration_for(config, namespace)["reachability"]
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
        namespace = namespace_for_path(source)
        if namespace is None:
            # A declared policy cannot safely classify an absolute, traversal,
            # empty, or otherwise malformed source claim.
            continue
        declaration = declaration_for(config, namespace)
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
    processed_namespaces: set[str] = set()
    effective_caps: dict[str, int] = {}
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
            # Resolve the expanded directory again.  An exact declaration is
            # authoritative over a matching wildcard and must be able to
            # suppress that wildcard's always-injected read.
            declaration = declaration_for(config, actual_namespace)
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
            if actual_namespace in processed_namespaces:
                continue
            # Apply the same namespace ACL before reading a configured source.
            # The workspace-level manager permits all in-workspace paths; an
            # agent-scoped caller therefore cannot use an always declaration as
            # a second, wider discovery mechanism.
            if not acl.can_read(actual_namespace):
                processed_namespaces.add(actual_namespace)
                continue
            processed_namespaces.add(actual_namespace)
            effective_caps[actual_namespace] = cap
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
                    source if source == actual_namespace or source.startswith(actual_namespace + "/") else f"{actual_namespace}/{source}"
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
    cap_total = sum(effective_caps.values())
    return selected, {"count": len(selected), "cap": min(cap_total, _MAX_ALWAYS_ITEMS), "content_type": "behavior"}


def admitted_namespace_blocks(workspace: str, agent_id: str | None) -> dict[str, dict[str, Any]] | None:
    """Return canonical live blocks visible to a bound principal.

    This is deliberately below the MCP layer so Python callers such as chat
    can bind evidence without importing an optional transport.  A duplicate
    block ID across visible sources is unresolved and omitted: choosing one
    source would let an ID-only extension seam attach the wrong content.
    Unbound callers receive ``None`` and retain the historical content-only
    admission path.
    """
    if not agent_id:
        return None

    from .admissibility import admit_corpus
    from .block_parser import parse_file
    from .corpus_registry import discover_corpus_files
    from .namespaces import NamespaceManager
    from .storage import _MARKDOWN_BACKENDS, _backend_name, _corpus_parse_fn

    manager = NamespaceManager(workspace, agent_id=agent_id)
    backend = _backend_name(workspace)
    if backend not in _MARKDOWN_BACKENDS:
        from .compliance.export import load_admitted_blocks

        admitted, _withheld = load_admitted_blocks(workspace)
    else:
        from ._recall_constants import CORPUS_FILES

        workspace_real = os.path.realpath(workspace)
        prefix = workspace_real + os.sep
        paths: list[str] = [rel for _label, rel in discover_corpus_files(workspace)]
        for _label, rel in CORPUS_FILES.items():
            for candidate in manager.resolve_corpus_paths(rel, policy="read"):
                candidate_rel = os.path.relpath(candidate, workspace_real).replace(os.sep, "/")
                paths.append(candidate_rel)

        try:
            custom_namespaces = declared_custom_namespaces(_load_config_for_namespace(workspace))
        except ValueError:
            custom_namespaces = ()
        custom_roots = set(custom_namespaces)
        for namespace in custom_namespaces:
            for _label, local_rel in discover_corpus_files(os.path.join(workspace_real, namespace)):
                paths.append(os.path.join(namespace, local_rel).replace(os.sep, "/"))

        # ``read`` may name a concrete file or a corpus subdirectory rather
        # than a namespace root. Resolve those grants against the finite
        # corpus registry as well; never recurse through an arbitrary ACL
        # path. Top-level custom roots still require their retrieval
        # declaration before they enter this set.
        known_roots = {
            "shared",
            "agents",
            "decisions",
            "tasks",
            "entities",
            "intelligence",
            "memory",
            "summaries",
            "maintenance",
            ".mind-mem-index",
            *custom_roots,
        }
        read_grants = manager._agent_policy.get("read", [])
        if isinstance(read_grants, list):
            for grant in read_grants:
                if not isinstance(grant, str) or any(mark in grant for mark in "*?["):
                    continue
                normalized = grant.replace("\\", "/").strip("/")
                if not normalized or normalized.split("/", 1)[0] not in known_roots:
                    continue
                if normalized.endswith(".md"):
                    paths.append(normalized)
                    continue
                for corpus_rel in CORPUS_FILES.values():
                    candidate = os.path.join(normalized, corpus_rel).replace(os.sep, "/")
                    paths.append(candidate)
                    if "/" in corpus_rel:
                        subdir, filename = corpus_rel.rsplit("/", 1)
                        if normalized.endswith("/" + subdir) or normalized == subdir:
                            paths.append(os.path.join(normalized, filename).replace(os.sep, "/"))

        unique_paths = list(dict.fromkeys(paths))
        try:
            reader = _corpus_parse_fn(workspace, backend, sources=unique_paths)
        except (OSError, ValueError):
            return {}
        blocks: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for rel_path in unique_paths:
            if rel_path in seen_paths or not manager.can_read(rel_path):
                continue
            top_level = rel_path.split("/", 1)[0]
            if top_level not in known_roots:
                continue
            seen_paths.add(rel_path)
            candidate = os.path.realpath(os.path.join(workspace_real, rel_path))
            if not candidate.startswith(prefix) or not os.path.isfile(candidate):
                continue
            try:
                parsed = reader(candidate) if reader is not parse_file else parse_file(candidate)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            for block in parsed:
                block["_source_file"] = rel_path
                blocks.append(block)
        admitted = admit_corpus(blocks, workspace=workspace)

    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for block in admitted:
        block_id = block.get("_id")
        source = block.get("_source_file") or block.get("_source") or block.get("file")
        if not isinstance(block_id, str) or not block_id or not isinstance(source, str) or not manager.can_read(source):
            continue
        if block_id in by_id:
            duplicates.add(block_id)
        else:
            by_id[block_id] = block
    for block_id in duplicates:
        by_id.pop(block_id, None)
    return by_id


def _load_config_for_namespace(workspace: str) -> dict[str, Any]:
    """Read workspace config without importing MCP configuration helpers."""
    from .storage import _load_workspace_config

    return cast(dict[str, Any], _load_workspace_config(workspace, quiet=True))


__all__ = [
    "REACHABILITY_SEARCHABLE",
    "REACHABILITY_DIRECT_ONLY",
    "REACHABILITY_ALWAYS_INJECTED",
    "FLOOR_NONE",
    "FLOOR_INHERIT_GLOBAL",
    "declaration_for",
    "declared_custom_namespaces",
    "namespace_for_path",
    "filter_search_hits",
    "namespace_search_allowed",
    "always_injected_hits",
    "admitted_namespace_blocks",
]
