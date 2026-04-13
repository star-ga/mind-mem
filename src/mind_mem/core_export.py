# Copyright 2026 STARGA, Inc.
"""Static-format export + diffing for .mmcore bundles (v2.3.0).

Two pieces:

- :func:`export_to_jsonld` / :func:`export_to_markdown` — produce
  static representations suitable for interop with tools that don't
  understand the .mmcore archive format.
- :func:`diff_cores` — compute a structured delta between two
  LoadedCore objects so operators can ship incremental updates
  without re-shipping the full bundle.

RDF/Turtle export is out of scope — needs `rdflib` — but JSON-LD
covers the same interchange need for most consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .context_core import LoadedCore


# ---------------------------------------------------------------------------
# JSON-LD export
# ---------------------------------------------------------------------------


_JSONLD_CONTEXT: dict[str, Any] = {
    "@vocab": "https://schema.star.ga/mindmem#",
    "id": "@id",
    "type": "@type",
    "blocks": {"@id": "https://schema.star.ga/mindmem#block"},
    "edges": {"@id": "https://schema.star.ga/mindmem#edge"},
}


def export_to_jsonld(core: LoadedCore) -> dict[str, Any]:
    """Return a JSON-LD representation of a loaded core."""
    manifest = core.manifest.as_dict()
    return {
        "@context": _JSONLD_CONTEXT,
        "@type": "ContextCore",
        "id": f"urn:mindmem:{manifest['namespace']}:{manifest['version']}",
        "manifest": manifest,
        "blocks": [
            {"@type": "Block", **b} for b in core.blocks
        ],
        "edges": [
            {"@type": "Edge", **e} for e in core.edges
        ],
    }


def export_to_markdown(core: LoadedCore) -> str:
    """Flatten a core into a single markdown document for human review."""
    manifest = core.manifest
    lines: list[str] = [
        f"# Context Core — {manifest.namespace} v{manifest.version}",
        "",
        f"- Built at: {manifest.built_at}",
        f"- Blocks: {manifest.block_count}",
        f"- Edges: {manifest.edge_count}",
        f"- Content hash: `{manifest.content_hash}`",
        "",
        "## Blocks",
        "",
    ]
    for block in core.blocks:
        block_id = block.get("_id") or block.get("id") or "?"
        block_type = block.get("type", "block")
        lines.append(f"### {block_type} — {block_id}")
        text = (
            block.get("text")
            or block.get("excerpt")
            or block.get("statement")
            or block.get("content")
            or ""
        )
        if text:
            lines.append("")
            lines.append(str(text).strip())
        lines.append("")

    if core.edges:
        lines.extend(["## Graph edges", ""])
        for edge in core.edges:
            lines.append(
                f"- {edge.get('subject')} — **{edge.get('predicate')}** → {edge.get('object')}"
            )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Core diff / rollback helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreDiff:
    added_blocks: list[dict]
    removed_blocks: list[dict]
    changed_blocks: list[dict]  # {id, from, to}
    added_edges: list[dict]
    removed_edges: list[dict]

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_blocks": self.added_blocks,
            "removed_blocks": self.removed_blocks,
            "changed_blocks": self.changed_blocks,
            "added_edges": self.added_edges,
            "removed_edges": self.removed_edges,
            "is_empty": self.is_empty,
        }

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.added_blocks,
                self.removed_blocks,
                self.changed_blocks,
                self.added_edges,
                self.removed_edges,
            )
        )


def _block_id(block: Mapping[str, Any]) -> str:
    for k in ("_id", "id", "block_id"):
        v = block.get(k)
        if v:
            return str(v)
    return ""


def _edge_key(edge: Mapping[str, Any]) -> tuple:
    return (
        str(edge.get("subject", "")),
        str(edge.get("predicate", "")),
        str(edge.get("object", "")),
        str(edge.get("source_block_id", "")),
    )


def diff_cores(old: LoadedCore, new: LoadedCore) -> CoreDiff:
    """Compute a structured delta between two loaded cores."""
    old_blocks = {_block_id(b): dict(b) for b in old.blocks if _block_id(b)}
    new_blocks = {_block_id(b): dict(b) for b in new.blocks if _block_id(b)}
    added_block_ids = [bid for bid in new_blocks if bid not in old_blocks]
    removed_block_ids = [bid for bid in old_blocks if bid not in new_blocks]
    changed: list[dict] = []
    for bid, nb in new_blocks.items():
        ob = old_blocks.get(bid)
        if ob is not None and ob != nb:
            changed.append({"id": bid, "from": ob, "to": nb})

    old_edges = {_edge_key(e): dict(e) for e in old.edges}
    new_edges = {_edge_key(e): dict(e) for e in new.edges}
    added_edges = [e for k, e in new_edges.items() if k not in old_edges]
    removed_edges = [e for k, e in old_edges.items() if k not in new_edges]

    return CoreDiff(
        added_blocks=[new_blocks[bid] for bid in added_block_ids],
        removed_blocks=[old_blocks[bid] for bid in removed_block_ids],
        changed_blocks=changed,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )


def apply_diff_rollback(current: LoadedCore, diff: CoreDiff) -> LoadedCore:
    """Reverse *diff* to reconstruct the prior core state.

    Removes blocks/edges the diff added, restores blocks/edges it
    removed, reverts changed blocks to their ``from`` value. Returns a
    new :class:`LoadedCore` with manifest carried over unchanged —
    the caller is expected to re-sign / re-anchor as needed.
    """
    added_ids = {_block_id(b) for b in diff.added_blocks}
    blocks = [b for b in current.blocks if _block_id(b) not in added_ids]
    for b in diff.removed_blocks:
        blocks.append(b)
    for change in diff.changed_blocks:
        block_id = change.get("id")
        if not block_id:
            continue
        blocks = [b for b in blocks if _block_id(b) != block_id]
        blocks.append(change.get("from", {}))

    added_edge_keys = {_edge_key(e) for e in diff.added_edges}
    edges = [e for e in current.edges if _edge_key(e) not in added_edge_keys]
    edges.extend(diff.removed_edges)

    return LoadedCore(
        manifest=current.manifest,
        blocks=blocks,
        edges=edges,
        retrieval_policies=current.retrieval_policies,
        ontology=current.ontology,
    )


__all__ = [
    "CoreDiff",
    "export_to_jsonld",
    "export_to_markdown",
    "diff_cores",
    "apply_diff_rollback",
]
