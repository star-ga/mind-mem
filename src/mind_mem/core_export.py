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

import re
from dataclasses import dataclass
from pathlib import Path
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
# OKF export — Open Knowledge Format (Apache-2.0,
# github.com/GoogleCloudPlatform/knowledge-catalog)
#
# Two surfaces:
#   - export_to_okf()        -> a JSON *envelope* (units + relations) for
#                               programmatic interop. Not a conformant
#                               OKF bundle; named honestly.
#   - write_okf_bundle()     -> a real OKF bundle on disk: one markdown
#                               file per concept with YAML frontmatter +
#                               body (Citations section), plus index.md.
#                               This is the form OKF consumers read.
#
# Both are lossy by construction: only the allow-listed OKF surface is
# emitted, so mind-mem's governance/contradiction/retrieval/evidence
# layers ("notable absences" in OKF) can never leak into the format.
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
_OKF_CITATION_KEYS = ("Sources", "Citations", "sources", "citations")

# The OKF-conformant surface of an emitted unit. Anything outside this set
# is part of the moat and must never appear in OKF output. Asserted in tests.
_OKF_UNIT_FIELDS = frozenset({"id", "type", "title", "description", "resource", "timestamp", "tags"})

# mind-mem block-id prefix -> OKF concept `type`. OKF requires a non-empty
# `type`; the real type lives in the `_id` prefix, not on the build path.
_ID_PREFIX_TYPE = {
    "D-": "decision",
    "T-": "task",
    "INC-": "incident",
    "PRJ-": "project",
    "PER-": "person",
    "TOOL-": "tool",
    "DREF-": "drift",
    "SIG-": "signal",
    "P-": "proposal",
}


def _first(block: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = block.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _okf_type(block: Mapping[str, Any]) -> str:
    """Resolve a non-empty OKF `type` (required field).

    Prefer an explicit ``type``; otherwise derive it from the ``_id``
    prefix (``D-`` -> decision, ``PRJ-`` -> project, …). Falls back to
    ``"concept"`` so the required field is never empty or a misleading
    ``"block"`` default that masks missing data.
    """
    t = block.get("type")
    if isinstance(t, str) and t and t != "block":
        return t
    bid = _block_id(block)
    for prefix, mapped in _ID_PREFIX_TYPE.items():
        if bid.startswith(prefix):
            return mapped
    # `t` is empty or the masking "block" default with no resolvable prefix.
    return "concept"


def _okf_timestamp(raw: Any) -> str:
    """Coerce a mind-mem timestamp to ISO-8601 datetime (OKF convention).

    mind-mem stores dates as ``YYYY-MM-DD``; OKF wants an ISO-8601
    datetime. A bare date is widened to midnight UTC; anything already
    carrying a time/zone is passed through untouched.
    """
    s = str(raw).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return f"{s}T00:00:00Z"
    return s


def _citations(block: Mapping[str, Any]) -> list[str]:
    raw = _first(block, _OKF_CITATION_KEYS)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    # comma/newline-separated string
    parts = re.split(r"[\n,]", str(raw))
    return [p.strip() for p in parts if p.strip()]


def _block_to_okf_unit(block: Mapping[str, Any]) -> dict[str, Any]:
    """Project a mind-mem block onto a lossy OKF knowledge unit.

    Only the OKF-conformant surface is emitted. Mind-mem's differentiated
    fields (governance status, evidence chain, contradiction links,
    retrieval scores) are intentionally dropped — OKF is an interop
    envelope, not a faithful round-trip of the moat.
    """
    unit: dict[str, Any] = {"type": _okf_type(block)}
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
        unit["timestamp"] = _okf_timestamp(ts)
    tags = _first(block, _OKF_TAG_KEYS)
    if tags is not None:
        unit["tags"] = tags if isinstance(tags, list) else [tags]
    return unit


def export_to_okf(core: LoadedCore) -> dict[str, Any]:
    """Return an OKF interop *envelope* (JSON) for a loaded core.

    This is a programmatic envelope — units + relations + manifest — not
    a conformant OKF bundle on disk. For the directory form an OKF
    consumer can read, use :func:`write_okf_bundle`. Both are lossy by
    design (OKF's "notable absences"): governance, contradiction
    handling, retrieval ranking, and the evidence chain are deliberately
    not represented, so the moat stays strictly above the format.
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


# ---------------------------------------------------------------------------
# OKF bundle writer (the real on-disk conformant form)
# ---------------------------------------------------------------------------

# YAML scalars that need quoting to stay unambiguous. We hand-emit a tiny,
# deterministic frontmatter subset (string/list-of-string) rather than pull
# in PyYAML; OKF frontmatter for our units is exactly that shape.
_YAML_SAFE = re.compile(r"^[A-Za-z0-9 ._:/+@#-]+$")


def _yaml_scalar(value: str) -> str:
    s = str(value)
    if s and _YAML_SAFE.fullmatch(s) and not s.startswith((" ", "-")):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _concept_filename(unit: Mapping[str, Any]) -> str:
    """Stable, filesystem-safe ``<concept>.md`` name; the concept id is the
    path minus the suffix, per OKF."""
    raw = str(unit.get("id") or unit.get("title") or unit.get("type") or "concept")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "concept"
    return f"{safe}.md"


def _render_okf_unit(unit: Mapping[str, Any], edges: list[Mapping[str, Any]]) -> str:
    """Render one OKF concept file: YAML frontmatter + markdown body.

    Relationships originating at this concept become bundle-relative
    markdown links in a ``# Relationships`` section (OKF represents edges
    as in-body links). Citations land in the conventional ``# Citations``
    heading.
    """
    fm: list[str] = ["---"]
    # `type` first — it is the only required field.
    fm.append(f"type: {_yaml_scalar(unit['type'])}")
    for key in ("title", "description", "resource", "timestamp"):
        if key in unit:
            fm.append(f"{key}: {_yaml_scalar(unit[key])}")
    if unit.get("tags"):
        fm.append("tags:")
        fm.extend(f"  - {_yaml_scalar(t)}" for t in unit["tags"])
    fm.append("---")

    body: list[str] = [""]
    title = unit.get("title") or unit.get("id") or unit["type"]
    body.append(f"# {title}")
    if unit.get("description"):
        body.extend(["", str(unit["description"])])

    if edges:
        body.extend(["", "# Relationships", ""])
        for e in edges:
            obj = str(e.get("object", ""))
            pred = str(e.get("predicate", "related"))
            target = re.sub(r"[^A-Za-z0-9._-]+", "-", obj).strip("-") or "concept"
            body.append(f"- {pred}: [{obj}](./{target}.md)")

    citations = unit.get("_citations") or []
    if citations:
        body.extend(["", "# Citations", ""])
        body.extend(f"- {c}" for c in citations)

    return "\n".join(fm + body).rstrip() + "\n"


def write_okf_bundle(core: LoadedCore, out_dir: str | Path) -> Path:
    """Write a real OKF bundle to *out_dir* and return its path.

    Produces a directory of one markdown file per concept (YAML
    frontmatter + body), an ``index.md`` listing every concept, and a
    ``log.md`` recording the producer. This is the form Google's catalog
    and other OKF consumers actually read — the JSON
    :func:`export_to_okf` envelope is not.

    Lossy by design: only the allow-listed OKF surface is emitted; the
    moat never reaches disk.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = core.manifest.as_dict()

    # Group edges by subject so each concept carries its outgoing links.
    edges_by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for e in core.edges:
        edges_by_subject.setdefault(str(e.get("subject", "")), []).append(e)

    index_rows: list[str] = []
    for block in core.blocks:
        unit = _block_to_okf_unit(block)
        unit_with_cites = dict(unit)
        unit_with_cites["_citations"] = _citations(block)  # body-only, not frontmatter
        fname = _concept_filename(unit)
        subj = str(unit.get("id", ""))
        (out / fname).write_text(
            _render_okf_unit(unit_with_cites, edges_by_subject.get(subj, [])),
            encoding="utf-8",
        )
        label = unit.get("title") or unit.get("id") or unit["type"]
        index_rows.append(f"- [{label}](./{fname}) — {unit['type']}")

    (out / "index.md").write_text(
        "---\n"
        "type: bundle\n"
        f"title: {_yaml_scalar(manifest['namespace'] + ' v' + str(manifest['version']))}\n"
        f"timestamp: {_okf_timestamp(manifest.get('built_at', ''))}\n"
        "---\n\n"
        f"# {manifest['namespace']} v{manifest['version']}\n\n" + "\n".join(index_rows) + "\n",
        encoding="utf-8",
    )
    (out / "log.md").write_text(
        f"# Provenance log\n\n- producer: mind-mem (OKF {OKF_VERSION})\n"
        f"- namespace: {manifest['namespace']}\n"
        f"- version: {manifest['version']}\n"
        f"- concepts: {len(core.blocks)}\n",
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# OKF bundle importer (consume OKF concepts into mind-mem blocks)
# ---------------------------------------------------------------------------

# OKF frontmatter keys -> capitalized mind-mem field keys (the parser grammar
# requires `^[A-Z][A-Za-z]+:`, so lowercase OKF keys are uppercased on ingest).
_OKF_IMPORT_FIELD = {
    "title": "Title",
    "description": "Statement",
    "resource": "Resource",
    "timestamp": "Date",
    "tags": "Tags",
}

# Reverse of _ID_PREFIX_TYPE: a unit `type` with no id gets a synthesized
# prefix so it parses as the right block kind.
_TYPE_ID_PREFIX = {v: k for k, v in _ID_PREFIX_TYPE.items()}


def _parse_okf_frontmatter(text: str) -> dict[str, Any]:
    """Parse the tiny YAML subset our bundle writer emits (string scalars +
    list-of-string under a key). Tolerant of files other producers wrote in
    the same shape; ignores body content."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm: dict[str, Any] = {}
    cur_list_key: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if cur_list_key is not None and line.lstrip().startswith("- "):
            fm.setdefault(cur_list_key, []).append(_unquote(line.lstrip()[2:].strip()))
            continue
        cur_list_key = None
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            cur_list_key = key  # a list follows
            fm[key] = []
        else:
            fm[key] = _unquote(val)
    return fm


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return s


def import_okf_bundle(bundle_dir: str | Path) -> list[dict[str, Any]]:
    """Read an OKF bundle directory and return mind-mem block dicts.

    Inverse of :func:`write_okf_bundle`: each concept ``.md`` (except the
    bundle's ``index.md`` / ``log.md``) becomes a block with capitalized
    field keys so it satisfies mind-mem's `^[A-Z][A-Za-z]+:` grammar. The
    OKF ``type`` is preserved both as ``type`` and reflected into the
    ``_id`` prefix when the source carried no id. Governance/evidence are
    *not* reconstructed — imported blocks enter through the normal
    HITL-gated path, never as trusted source-of-truth.
    """
    root = Path(bundle_dir)
    blocks: list[dict[str, Any]] = []
    for md in sorted(root.rglob("*.md")):
        if md.name in ("index.md", "log.md"):
            continue
        fm = _parse_okf_frontmatter(md.read_text(encoding="utf-8"))
        if not fm.get("type"):
            continue
        concept_id = str(md.relative_to(root).with_suffix(""))
        block: dict[str, Any] = {"type": str(fm["type"])}
        for okf_key, mm_key in _OKF_IMPORT_FIELD.items():
            if okf_key in fm and fm[okf_key] not in (None, "", []):
                block[mm_key] = fm[okf_key]
        # Preserve / synthesize an id so the block keys a graph node.
        prefix = _TYPE_ID_PREFIX.get(block["type"], "")
        bid = concept_id
        if prefix and not any(bid.startswith(p) for p in _ID_PREFIX_TYPE):
            bid = f"{prefix}{concept_id}"
        block["_id"] = bid
        blocks.append(block)
    return blocks


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
    "OKF_VERSION",
    "export_to_jsonld",
    "export_to_okf",
    "write_okf_bundle",
    "import_okf_bundle",
    "export_to_markdown",
    "diff_cores",
    "apply_diff_rollback",
]
