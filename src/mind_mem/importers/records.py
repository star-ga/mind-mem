# Copyright 2026 STARGA, Inc.
"""Value types + errors for the migration importers (roadmap Group G).

Both types are frozen dataclasses — the import pipeline is a pure
transform from a foreign dump to a tuple of :class:`ImportRecord`, and
:class:`ImportResult` is the immutable receipt the CLI prints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "ImporterError",
    "ImportParseError",
    "UnsupportedSystemError",
    "ImportRecord",
    "ImportResult",
]


class ImporterError(Exception):
    """Base class for every importer failure."""


class ImportParseError(ImporterError):
    """The dump file is unreadable, malformed, or not in the claimed format."""


class UnsupportedSystemError(ImporterError):
    """The requested source system has no file-based importer.

    Raised for the endpoint-backed systems (pinecone / weaviate / qdrant)
    which are explicitly deferred, and for entirely unknown names.
    """


@dataclass(frozen=True)
class ImportRecord:
    """One memory unit lifted out of a foreign dump.

    Attributes:
        system: Source system slug (``chroma`` / ``mem0`` / ``letta``).
        external_id: Identifier the source system used. Stable across
            re-exports for every format we support, which is what makes
            the derived block id — and therefore the import — idempotent.
        text: The memory content itself. Never empty (empty records are
            dropped by the parsers before an ``ImportRecord`` is built).
        metadata: Flattened scalar metadata, string keys and values only.
        created_at: Source timestamp verbatim, or ``""`` when absent.
        links: Ordered, de-duplicated names this record links out to —
            ``[[wikilink]]`` targets for the note-tree formats, empty for
            every other source. Preserved as a real ``Links`` block field
            and, only behind the default-OFF ``link_edges`` flag,
            materialized as lineage edges.
    """

    system: str
    external_id: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    created_at: str = ""
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportResult:
    """Immutable receipt for one ``mm import`` run.

    Attributes:
        batch: Deterministic import-batch id stamped on every block the
            run wrote (``""`` when the run wrote nothing). The handle an
            operator passes to the release proposal.
        status: The status the blocks landed with. Always
            ``"quarantined"`` — imported content is never authoritative
            on arrival, so the receipt says so out loud rather than
            letting a caller assume the blocks are live.
    """

    system: str
    source_path: str
    parsed: int
    imported: int
    skipped_existing: int
    skipped_near_duplicate: int
    block_ids: tuple[str, ...]
    dry_run: bool = False
    linked_edges: int = 0
    batch: str = ""
    status: str = "quarantined"

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form (what ``mm import`` prints).

        ``linked_edges`` appears only when the default-OFF ``link_edges``
        flag actually materialized edges, so the flag-off receipt keeps
        the shape it had before that flag existed.
        """
        payload: dict[str, Any] = {
            "system": self.system,
            "source_path": self.source_path,
            "parsed": self.parsed,
            "imported": self.imported,
            "skipped_existing": self.skipped_existing,
            "skipped_near_duplicate": self.skipped_near_duplicate,
            "block_ids": list(self.block_ids),
            "dry_run": self.dry_run,
            "batch": self.batch,
            "status": self.status,
        }
        if self.linked_edges:
            payload["linked_edges"] = self.linked_edges
        return payload
