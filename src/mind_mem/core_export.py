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

from dataclasses import dataclass
from typing import Any, Mapping

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
        "blocks": [{"@type": "Block", **b} for b in core.blocks],
        "edges": [{"@type": "Edge", **e} for e in core.edges],
    }


# ---------------------------------------------------------------------------
# OKF export (interop envelope)
# ---------------------------------------------------------------------------

OKF_VERSION = "0.1"

# Map mind-mem block fields onto OKF unit fields. Mind-mem capitalizes its
# field keys (the `^[A-Z][A-Za-z]+:` grammar); OKF uses lowercase. We probe
# each candidate in order and take the first present.
_OKF_TITLE_KEYS = ("Title", "Name", "Summary")
_OKF_DESC_KEYS = ("Statement", "Excerpt", "Description", "text", "content")
_OKF_RESOURCE_KEYS = ("Resource", "resource")
_OKF_TIMESTAMP_KEYS = ("Date", "timestamp", "built_at")
_OKF_TAG_KEYS = ("Tags", "tags")


def _first(block: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = block.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _block_to_okf_unit(block: Mapping[str, Any]) -> dict[str, Any]:
    """Project a mind-mem block onto a lossy OKF knowledge unit.

    Only the OKF-conformant surface is emitted. Mind-mem's differentiated
    fields (governance status, evidence chain, contradiction links,
    retrieval scores) are intentionally dropped — OKF is an interop
    envelope, not a faithful round-trip of the moat.
    """
    unit: dict[str, Any] = {"type": block.get("type", "block")}
    bid = _block_id(block)
    if bid:
        unit["id"] = bid
    title = _first(block, _OKF_TITLE_KEYS)
    if title is not None:
        unit["title"] = str(title)
    desc = _first(block, _OKF_DESC_KEYS)
    if desc is not None:
        unit["description"] = str(desc).strip()
    resource = _first(block, _OKF_RESOURCE_KEYS)
    if resource is not None:
        unit["resource"] = str(resource)
    ts = _first(block, _OKF_TIMESTAMP_KEYS)
    if ts is not None:
        unit["timestamp"] = str(ts)
    tags = _first(block, _OKF_TAG_KEYS)
    if tags is not None:
        unit["tags"] = tags if isinstance(tags, list) else [tags]
    return unit


def export_to_okf(core: LoadedCore) -> dict[str, Any]:
    """Return an OKF-conformant export envelope for a loaded core.

    OKF = Open Knowledge Format (Apache-2.0,
    github.com/GoogleCloudPlatform/knowledge-catalog). This is an
    *envelope only*: blocks become OKF knowledge units and graph edges
    become typed relations. Mind-mem's governance, contradiction
    handling, retrieval ranking, and evidence chain are deliberately not
    represented (OKF's "notable absences") — the export is lossy by
    design so the moat stays above the format.
    """
    manifest = core.manifest.as_dict()
    return {
        "okf_version": OKF_VERSION,
        "source": "mind-mem",
        "id": f"urn:mindmem:{manifest['namespace']}:{manifest['version']}",
        "manifest": manifest,
        "units": [_block_to_okf_unit(b) for b in core.blocks],
        "relations": [
            {
                "subject": e.get("subject"),
                "predicate": e.get("predicate"),
                "object": e.get("object"),
            }
            for e in core.edges
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
        text = block.get("text") or block.get("excerpt") or block.get("statement") or block.get("content") or ""
        if text:
            lines.append("")
            lines.append(str(text).strip())
        lines.append("")

    if core.edges:
        lines.extend(["## Graph edges", ""])
        for edge in core.edges:
            lines.append(f"- {edge.get('subject')} — **{edge.get('predicate')}** → {edge.get('object')}")
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
