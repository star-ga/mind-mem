"""Context-core MCP tools — ``.mmcore`` bundle lifecycle.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, core domain). Four tools wrap the
:class:`CoreRegistry` singleton for building, loading, unloading,
and listing portable block + knowledge-graph archives.

A fifth, :func:`export_core`, is the outbound half of the same lifecycle:
it renders a ``.mmcore`` bundle into a static interchange format
(:mod:`mind_mem.core_export`) so a consumer that does not understand the
archive can read it. It is gated on the v4 ``core_export`` flag and does
nothing at all while that flag is OFF.

Export is one-way on purpose. The inbound direction — reading a foreign
OKF bundle back in — is NOT a tool here, because a tool that wrote
foreign concepts into the corpus would be a write path around the HITL
gate. It goes through the importer instead
(``mm import --from okf <bundle>``), which lands every concept
quarantined and unservable until a governed release proposal is applied.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _core_dir, _core_registry, _kg_path

#: v4 flag gating :func:`export_core`. OFF -> the tool is inert.
CORE_EXPORT_FLAG = "core_export"

#: What :func:`export_core` returns while that flag is OFF.
CORE_EXPORT_DISABLED = f"export_core requires the v4 '{CORE_EXPORT_FLAG}' flag (mind-mem.json: v4.{CORE_EXPORT_FLAG}.enabled = true)"

#: Static formats :func:`export_core` can render a core into.
EXPORT_FORMATS: tuple[str, ...] = ("okf", "jsonld", "markdown")

#: Subdirectory of ``memory/cores/`` that exports land in.
EXPORT_SUBDIR = "exports"

_UNSAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@mcp_tool_observe
def build_core(namespace: str, version: str, filename: str = "") -> str:
    """Build a .mmcore bundle from the active workspace's blocks.

    Snapshots the current block index + knowledge graph into a portable
    `.mmcore` archive. Downstream callers can load it into another
    mind-mem instance via `load_core`.

    Args:
        namespace: Identifier used to prefix blocks when loaded.
        version: Caller-facing semver recorded in the manifest.
        filename: Optional output filename (defaults to
            ``<namespace>-<version>.mmcore`` under ``memory/cores/``).

    Returns:
        JSON envelope with the bundle path and manifest summary.
    """
    from mind_mem.context_core import build_core as _build_core
    from mind_mem.knowledge_graph import KnowledgeGraph

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    if not isinstance(version, str) or not version.strip():
        return json.dumps({"error": "version must be a non-empty string"})

    blocks: list[dict] = []
    try:
        from mind_mem.sqlite_index import merkle_leaves as _leaves

        for bid, content_hash in _leaves(ws):
            blocks.append({"_id": bid, "content_hash": content_hash})
    except (ImportError, AttributeError):
        pass

    edges: list[dict] = []
    kg_file = _kg_path(ws)
    if os.path.isfile(kg_file):
        kg = KnowledgeGraph(kg_file)
        try:
            fallback_edges: list[Any] = []
            for e in kg.edges_from("__all__") if False else fallback_edges:
                edges.append(e.as_dict())
            rows = kg._conn.execute(
                "SELECT subject, predicate, object, source_block_id, confidence, valid_from, valid_until, metadata FROM edges"
            ).fetchall()
            for row in rows:
                edges.append(
                    {
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "source_block_id": row["source_block_id"],
                        "confidence": row["confidence"],
                        "valid_from": row["valid_from"],
                        "valid_until": row["valid_until"],
                        "metadata": row["metadata"],
                    }
                )
        finally:
            kg.close()

    out_name = filename.strip() or f"{namespace.strip()}-{version.strip()}.mmcore"
    if any(ch in out_name for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})
    out_path = os.path.join(_core_dir(ws), out_name)

    try:
        manifest = _build_core(
            out_path,
            namespace=namespace,
            version=version,
            blocks=blocks,
            edges=edges,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        {"path": out_path, "manifest": manifest.as_dict(), "_schema_version": "1.0"},
        indent=2,
    )


@mcp_tool_observe
def load_core(filename: str, verify: bool = True) -> str:
    """Load a .mmcore bundle from the workspace's cores/ directory.

    Args:
        filename: Core filename relative to ``memory/cores/``.
        verify: Recompute and compare the content hash (default True).
    """
    from mind_mem.context_core import CoreLoadError

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(filename, str) or not filename.strip():
        return json.dumps({"error": "filename must be a non-empty string"})
    if any(ch in filename for ch in "/\\"):
        return json.dumps({"error": "filename must not contain path separators"})

    path = os.path.join(_core_dir(ws), filename.strip())
    try:
        loaded = _core_registry().load(path, verify=verify)
    except CoreLoadError as exc:
        return json.dumps({"error": str(exc)})
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {
            "loaded": True,
            "namespace": loaded.manifest.namespace,
            "blocks": loaded.block_count(),
            "edges": loaded.edge_count(),
            "content_hash": loaded.manifest.content_hash,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def unload_core(namespace: str) -> str:
    """Unload a previously-loaded core by namespace."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    if not isinstance(namespace, str) or not namespace.strip():
        return json.dumps({"error": "namespace must be a non-empty string"})
    ok = _core_registry().unload(namespace.strip())
    return json.dumps({"unloaded": bool(ok)})


@mcp_tool_observe
def list_cores() -> str:
    """List every currently-loaded .mmcore bundle (namespace + stats)."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    return json.dumps({"cores": _core_registry().stats(), "_schema_version": "1.0"}, indent=2)


def _core_export_enabled() -> bool:
    """True iff the v4 ``core_export`` flag is ON, without emitting anything.

    ``feature_flags.is_enabled`` warns (``v4_config_unreadable``) on a
    malformed config, so using it here would make a flag-OFF build log a
    line the pre-wiring build never logged. See ``is_enabled_quiet``.
    """
    from mind_mem.v4.feature_flags import is_enabled_quiet

    return is_enabled_quiet(CORE_EXPORT_FLAG)


@mcp_tool_observe
def export_core(name: str, format: str = "okf") -> str:  # noqa: A002 - MCP-facing arg name
    """Render a .mmcore bundle into a static interchange format.

    Requires the v4 ``core_export`` flag; returns an error and touches
    nothing while it is OFF.

    The output is a directory under ``memory/cores/exports/`` named
    ``<core-stem>-<content-hash>``, so re-exporting the same core
    overwrites its own files and a *different* core can never leave a
    stale concept file behind for a later re-import to pick up.

    Deterministic: every format is derived from the bundle alone (the
    manifest's own ``built_at`` is used, never the current clock), so the
    same core exports byte-identically on any machine.

    Args:
        name: Core filename relative to ``memory/cores/`` (the path
            ``build_core`` returned, without directories).
        format: ``okf`` (default) writes a conformant Open Knowledge
            Format bundle — one markdown concept per block plus
            ``index.md``/``log.md`` — which
            ``mm import --from okf <dir>`` can read back in through the
            quarantine gate. ``jsonld`` writes ``core.jsonld``;
            ``markdown`` writes ``core.md`` for human review.

    Returns:
        JSON envelope with the export directory, the files written, and
        the source manifest summary.
    """
    if not _core_export_enabled():
        return json.dumps({"error": CORE_EXPORT_DISABLED})

    from mind_mem.context_core import CoreLoadError
    from mind_mem.context_core import load_core as _load_core
    from mind_mem.core_export import export_to_jsonld, export_to_markdown, write_okf_bundle

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(name, str) or not name.strip():
        return json.dumps({"error": "name must be a non-empty string"})
    if any(ch in name for ch in "/\\"):
        return json.dumps({"error": "name must not contain path separators"})
    if not isinstance(format, str) or format.strip().lower() not in EXPORT_FORMATS:
        return json.dumps({"error": f"format must be one of {', '.join(EXPORT_FORMATS)}"})
    fmt = format.strip().lower()

    path = os.path.join(_core_dir(ws), name.strip())
    try:
        core = _load_core(path, verify=True)
    except CoreLoadError as exc:
        return json.dumps({"error": str(exc)})
    except OSError as exc:
        return json.dumps({"error": f"cannot read core {name.strip()!r}: {exc}"})

    manifest = core.manifest.as_dict()
    stem = _UNSAFE_STEM_RE.sub("-", name.strip().removesuffix(".mmcore")).strip("-") or "core"
    out_dir = os.path.join(_core_dir(ws), EXPORT_SUBDIR, f"{stem}-{manifest['content_hash'][:16]}")

    try:
        if fmt == "okf":
            write_okf_bundle(core, out_dir)
        else:
            os.makedirs(out_dir, exist_ok=True)
            filename = "core.jsonld" if fmt == "jsonld" else "core.md"
            body = json.dumps(export_to_jsonld(core), indent=2) if fmt == "jsonld" else export_to_markdown(core)
            with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as handle:
                handle.write(body)
    except OSError as exc:
        return json.dumps({"error": f"cannot write export to {out_dir}: {exc}"})

    return json.dumps(
        {
            "format": fmt,
            "path": out_dir,
            "files": sorted(os.listdir(out_dir)),
            "blocks": core.block_count(),
            "edges": core.edge_count(),
            "manifest": manifest,
            "_schema_version": "1.0",
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the core tools onto *mcp*."""
    mcp.tool(build_core)
    mcp.tool(load_core)
    mcp.tool(export_core)
    mcp.tool(unload_core)
    mcp.tool(list_cores)
