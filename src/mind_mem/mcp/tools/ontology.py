"""Ontology MCP tools — ``ontology_load`` + ``ontology_validate``.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, ontology domain). Both tools operate on the lazy-init
:class:`OntologyRegistry` singleton shared via
``mcp.tools._helpers._ontology_registry`` (which preloads the
in-box ``software_engineering_ontology`` so
``ontology_validate`` works on a fresh workspace without a
separate ``ontology_load`` step).
"""

from __future__ import annotations

import json

from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _ontology_registry


@mcp_tool_observe
def ontology_load(spec: str, make_active: bool = False) -> str:
    """Load an ontology from an inline JSON spec.

    The spec must be a JSON object with ``version`` and ``types``
    fields (see ``Ontology.from_dict``). Pass ``make_active=True`` to
    promote the loaded ontology to the default used by
    ``ontology_validate``.
    """
    from mind_mem.ontology import Ontology

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(spec, str) or not spec.strip():
        return json.dumps({"error": "spec must be a non-empty JSON string"})
    if len(spec) > 1_048_576:
        return json.dumps({"error": "spec must be ≤1 MiB"})
    try:
        data = json.loads(spec)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"spec is not valid JSON: {exc}"})
    if not isinstance(data, dict):
        return json.dumps({"error": "spec must decode to a JSON object"})
    try:
        ont = Ontology.from_dict(data)
    except (ValueError, KeyError, TypeError) as exc:
        return json.dumps({"error": f"invalid ontology: {exc}"})

    _ontology_registry().load(ont, make_active=bool(make_active))
    return json.dumps(
        {
            "loaded": True,
            "version": ont.version,
            "types": ont.type_names(),
            "active": bool(make_active) or _ontology_registry().active().version == ont.version,
            "_schema_version": "1.0",
        },
        indent=2,
    )


@mcp_tool_observe
def ontology_validate(block: str, type_name: str, strict: bool = True) -> str:
    """Validate *block* (JSON object) against the active ontology.

    Returns ``{valid: bool, errors: [str]}`` — an empty ``errors``
    list means the block satisfies the type's effective schema
    (including inherited parent properties).
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not isinstance(block, str) or not block.strip():
        return json.dumps({"error": "block must be a non-empty JSON string"})
    if len(block) > 1_048_576:
        return json.dumps({"error": "block must be ≤1 MiB"})
    try:
        block_obj = json.loads(block)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"block is not valid JSON: {exc}"})
    if not isinstance(block_obj, dict):
        return json.dumps({"error": "block must decode to a JSON object"})
    if not isinstance(type_name, str) or not type_name.strip():
        return json.dumps({"error": "type_name must be a non-empty string"})

    ont = _ontology_registry().active()
    if ont is None:
        return json.dumps({"error": "no active ontology; call ontology_load first"})
    errors = ont.validate(type_name, block_obj, strict=bool(strict))
    return json.dumps(
        {
            "valid": len(errors) == 0,
            "errors": errors,
            "type": type_name,
            "ontology_version": ont.version,
            "_schema_version": "1.0",
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the ontology tools onto *mcp*."""
    mcp.tool(ontology_load)
    mcp.tool(ontology_validate)
