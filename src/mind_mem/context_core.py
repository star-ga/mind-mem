# Copyright 2026 STARGA, Inc.
"""Context Cores (v2.3.0) — portable memory bundles.

A context core is a single ``.mmcore`` archive that carries a filtered
snapshot of a workspace's memory plus the retrieval policies needed to
use it. The format is intentionally simple: a ``.tar.gz`` of a manifest
JSON, an optional block JSONL, an optional graph edges JSONL, and
optional ontology / retrieval-policy dumps.

Design goals:

- **Deterministic** — two builds over the same inputs produce the same
  tarball (sorted entries, fixed mtimes, stable JSON key ordering).
- **Immutable once published** — the manifest records a content hash
  (SHA-256) over the canonical entry stream so tampering is detectable.
- **Namespace isolation** — each core carries a namespace string that
  downstream consumers can prefix when loading multiple cores into the
  same mind-mem instance.
- **Zero new dependencies** — pure stdlib (tarfile, hashlib, json).

Edge / full-graph sync and ``.mmcore`` → RDF / JSON-LD exporters are
deferred; this release lands the format + build/load round-trip.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Format versioning
# ---------------------------------------------------------------------------


CORE_FORMAT_VERSION: str = "1.0"

# Entries we recognise inside a .mmcore archive. Adding a new entry is a
# minor-version bump; removing or changing one is a major-version bump.
_BLOCKS_ENTRY = "blocks.jsonl"
_EDGES_ENTRY = "graph_edges.jsonl"
_RETRIEVAL_ENTRY = "retrieval_policies.json"
_ONTOLOGY_ENTRY = "ontology.json"
_MANIFEST_ENTRY = "manifest.json"

# Known entries. Anything else in the archive is rejected by `load_core`
# so an attacker cannot hide payload inside unknown filenames or fill
# the archive with garbage that the loader would otherwise still
# materialise into memory.
_KNOWN_ENTRIES: frozenset[str] = frozenset(
    {
        _BLOCKS_ENTRY,
        _EDGES_ENTRY,
        _RETRIEVAL_ENTRY,
        _ONTOLOGY_ENTRY,
        _MANIFEST_ENTRY,
    }
)

# Default DoS guards for load_core. Callers with legitimately huge
# bundles pass higher limits via the `max_entry_bytes` / `max_entries`
# kwargs.
_DEFAULT_MAX_ENTRY_BYTES: int = 256 * 1024 * 1024  # 256 MiB
_DEFAULT_MAX_ENTRIES: int = len(_KNOWN_ENTRIES) + 2  # room for future additions


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreManifest:
    """Immutable manifest describing a ``.mmcore`` bundle."""

    namespace: str
    version: str  # caller-facing semver; NOT the format version
    format_version: str
    built_at: str
    block_count: int
    edge_count: int
    has_retrieval_policies: bool
    has_ontology: bool
    content_hash: str
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "version": self.version,
            "format_version": self.format_version,
            "built_at": self.built_at,
            "block_count": self.block_count,
            "edge_count": self.edge_count,
            "has_retrieval_policies": self.has_retrieval_policies,
            "has_ontology": self.has_ontology,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoreManifest":
        return cls(
            namespace=str(data["namespace"]),
            version=str(data["version"]),
            format_version=str(data["format_version"]),
            built_at=str(data["built_at"]),
            block_count=int(data["block_count"]),
            edge_count=int(data["edge_count"]),
            has_retrieval_policies=bool(data["has_retrieval_policies"]),
            has_ontology=bool(data["has_ontology"]),
            content_hash=str(data["content_hash"]),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Encode records as JSONL with sorted keys (for deterministic hashing)."""
    out = io.StringIO()
    for rec in records:
        out.write(json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str))
        out.write("\n")
    return out.getvalue().encode("utf-8")


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    # Namespaces are used as filename prefixes downstream; restrict to a
    # safe alphabet so they can't contain path separators or shell meta.
    bad = set('/\\:*?"<>|\n\r\t')
    if any(ch in bad for ch in namespace):
        raise ValueError(f"namespace must not contain path separators or control chars, got {namespace!r}")
    if len(namespace) > 128:
        raise ValueError("namespace must be ≤128 chars")
    return namespace.strip()


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------


def build_core(
    output_path: str,
    *,
    namespace: str,
    version: str,
    blocks: Iterable[Mapping[str, Any]] = (),
    edges: Iterable[Mapping[str, Any]] = (),
    retrieval_policies: Optional[Mapping[str, Any]] = None,
    ontology: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    built_at: Optional[str] = None,
) -> CoreManifest:
    """Create a ``.mmcore`` archive at *output_path*.

    Entries are written in a fixed order with zeroed mtimes so two
    builds over the same inputs produce byte-identical tarballs. The
    content hash embedded in the manifest covers every entry *before*
    the manifest itself, so tampering (re-extracting and re-tarring with
    modified blocks) is detectable by any downstream loader.
    """
    namespace = _validate_namespace(namespace)
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be a non-empty string")
    version = version.strip()

    block_list = list(blocks)
    edge_list = list(edges)

    # Materialise the canonical bodies once so we can both hash and
    # write them in a single pass.
    payloads: list[tuple[str, bytes]] = []
    if block_list:
        payloads.append((_BLOCKS_ENTRY, _canonical_jsonl(block_list)))
    if edge_list:
        payloads.append((_EDGES_ENTRY, _canonical_jsonl(edge_list)))
    if retrieval_policies is not None:
        payloads.append((_RETRIEVAL_ENTRY, _canonical_json(retrieval_policies)))
    if ontology is not None:
        payloads.append((_ONTOLOGY_ENTRY, _canonical_json(ontology)))
    payloads.sort(key=lambda kv: kv[0])

    content_hash = _sha256(p[1] for p in payloads)
    if built_at is None:
        built_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        # Validate explicit timestamp so a reproducible-build CI doesn't
        # accidentally ship a malformed manifest.
        if not isinstance(built_at, str) or not built_at.strip():
            raise ValueError("built_at must be a non-empty ISO 8601 string")

    manifest = CoreManifest(
        namespace=namespace,
        version=version,
        format_version=CORE_FORMAT_VERSION,
        built_at=built_at,
        block_count=len(block_list),
        edge_count=len(edge_list),
        has_retrieval_policies=retrieval_policies is not None,
        has_ontology=ontology is not None,
        content_hash=content_hash,
        metadata=dict(metadata or {}),
    )
    manifest_bytes = _canonical_json(manifest.as_dict())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Build the uncompressed tar in memory first so we can hand-wrap it
    # in a gzip stream with filename="" and mtime=0. tarfile's own
    # w:gz mode would leak the OUTPUT filename into the gzip header,
    # which breaks byte-for-byte reproducibility across build sites.
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:

        def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
            info = tarfile.TarInfo(name=name)
            info.size = size
            info.mtime = 0
            info.mode = 0o644
            # Pin uid/gid/uname/gname explicitly so the tar header does
            # not pick up per-builder defaults (different Python builds
            # sometimes differ). This is what makes the bundle a
            # SOURCE_DATE_EPOCH-style reproducible artifact.
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            return info

        for name, body in payloads:
            tf.addfile(_tarinfo(name, len(body)), io.BytesIO(body))
        tf.addfile(_tarinfo(_MANIFEST_ENTRY, len(manifest_bytes)), io.BytesIO(manifest_bytes))
    tar_bytes = tar_buf.getvalue()

    with open(output_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, filename="", mtime=0) as gz:
            gz.write(tar_bytes)
    return manifest


class CoreLoadError(Exception):
    """Raised when a ``.mmcore`` archive cannot be parsed or fails integrity checks."""


@dataclass
class LoadedCore:
    """Result of loading a ``.mmcore`` archive into memory."""

    manifest: CoreManifest
    blocks: list[dict]
    edges: list[dict]
    retrieval_policies: Optional[dict]
    ontology: Optional[dict]

    def block_count(self) -> int:
        return len(self.blocks)

    def edge_count(self) -> int:
        return len(self.edges)


def load_core(
    path: str,
    *,
    verify: bool = True,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_entry_bytes: int = _DEFAULT_MAX_ENTRY_BYTES,
) -> LoadedCore:
    """Load and (by default) verify a ``.mmcore`` archive.

    ``verify=True`` recomputes the content hash from the archive's
    payload entries and compares it to the manifest's. A mismatch raises
    :class:`CoreLoadError`.

    Unknown archive entries (anything outside the small fixed set of
    known filenames) are rejected rather than silently loaded, and each
    entry is capped at ``max_entry_bytes`` to defuse tar-bomb DoS. The
    archive must carry ≤ ``max_entries`` files total.
    """
    if not os.path.isfile(path):
        raise CoreLoadError(f"core file not found: {path}")

    try:
        tf = tarfile.open(path, "r:gz")
    except tarfile.TarError as exc:
        raise CoreLoadError(f"cannot open core archive: {exc}") from exc

    try:
        members = [m for m in tf.getmembers() if m.isfile()]
        if len(members) > max_entries:
            raise CoreLoadError(f"core archive has too many entries ({len(members)} > {max_entries})")
        names = [m.name for m in members]
        for name in names:
            if name not in _KNOWN_ENTRIES:
                raise CoreLoadError(f"core archive has unknown entry: {name!r}")
        if _MANIFEST_ENTRY not in names:
            raise CoreLoadError("core archive is missing manifest.json")

        entries: dict[str, bytes] = {}
        for info in members:
            if info.size > max_entry_bytes:
                raise CoreLoadError(f"core entry {info.name!r} exceeds {max_entry_bytes} bytes ({info.size})")
            reader = tf.extractfile(info)
            if reader is None:
                continue
            entries[info.name] = reader.read(max_entry_bytes + 1)
            if len(entries[info.name]) > max_entry_bytes:
                raise CoreLoadError(f"core entry {info.name!r} exceeds {max_entry_bytes} bytes on read (possible size-header lie)")
    finally:
        tf.close()

    try:
        manifest = CoreManifest.from_dict(json.loads(entries[_MANIFEST_ENTRY].decode("utf-8", errors="replace")))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise CoreLoadError(f"manifest.json is malformed: {exc}") from exc

    if manifest.format_version != CORE_FORMAT_VERSION:
        raise CoreLoadError(f"unsupported core format: manifest says {manifest.format_version!r}, loader expects {CORE_FORMAT_VERSION!r}")

    payload_names = sorted(n for n in entries if n != _MANIFEST_ENTRY)
    if verify:
        recomputed = _sha256(entries[n] for n in payload_names)
        if recomputed != manifest.content_hash:
            raise CoreLoadError(
                f"content hash mismatch: archive payload has been modified (expected {manifest.content_hash[:16]}…, got {recomputed[:16]}…)"
            )

    blocks = _parse_jsonl(entries.get(_BLOCKS_ENTRY))
    edges = _parse_jsonl(entries.get(_EDGES_ENTRY))
    retrieval = _parse_json(entries.get(_RETRIEVAL_ENTRY))
    ontology = _parse_json(entries.get(_ONTOLOGY_ENTRY))

    # Manifest ↔ content reconciliation flags a subtle class of
    # tampering where someone hand-rebuilds the archive with consistent
    # hashes but edits the block/edge counts. Gated on `verify` so
    # callers debugging a known-tampered archive can still load it.
    if verify:
        if len(blocks) != manifest.block_count:
            raise CoreLoadError(f"block count mismatch: manifest={manifest.block_count}, archive={len(blocks)}")
        if len(edges) != manifest.edge_count:
            raise CoreLoadError(f"edge count mismatch: manifest={manifest.edge_count}, archive={len(edges)}")
        if manifest.has_retrieval_policies != (retrieval is not None):
            raise CoreLoadError("has_retrieval_policies flag mismatch")
        if manifest.has_ontology != (ontology is not None):
            raise CoreLoadError("has_ontology flag mismatch")

    return LoadedCore(
        manifest=manifest,
        blocks=blocks,
        edges=edges,
        retrieval_policies=retrieval,
        ontology=ontology,
    )


def _parse_jsonl(body: Optional[bytes]) -> list[dict]:
    if not body:
        return []
    out: list[dict] = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _parse_json(body: Optional[bytes]) -> Optional[dict]:
    if not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Registry — mount / unmount loaded cores in a single mind-mem process
# ---------------------------------------------------------------------------


class CoreRegistry:
    """Process-local registry of mounted cores, keyed by namespace.

    Intended for ephemeral use inside an MCP server: load a core,
    consult it while serving requests, unload when done. Persistence is
    the caller's concern — the registry holds references only.
    """

    def __init__(self, *, max_cores: int = 32) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be >= 1")
        self._max = int(max_cores)
        self._cores: dict[str, LoadedCore] = {}

    def load(self, path: str, *, verify: bool = True) -> LoadedCore:
        core = load_core(path, verify=verify)
        if len(self._cores) >= self._max and core.manifest.namespace not in self._cores:
            raise RuntimeError(f"core registry full (max={self._max}); unload before loading more")
        self._cores[core.manifest.namespace] = core
        return core

    def unload(self, namespace: str) -> bool:
        return self._cores.pop(namespace, None) is not None

    def get(self, namespace: str) -> Optional[LoadedCore]:
        return self._cores.get(namespace)

    def namespaces(self) -> list[str]:
        return sorted(self._cores.keys())

    def stats(self) -> list[dict]:
        return [
            {
                "namespace": ns,
                "version": c.manifest.version,
                "blocks": c.block_count(),
                "edges": c.edge_count(),
                "content_hash": c.manifest.content_hash,
            }
            for ns, c in sorted(self._cores.items())
        ]


__all__ = [
    "CORE_FORMAT_VERSION",
    "CoreManifest",
    "LoadedCore",
    "CoreLoadError",
    "CoreRegistry",
    "build_core",
    "load_core",
]
