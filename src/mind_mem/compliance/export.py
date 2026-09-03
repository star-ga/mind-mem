# Copyright 2026 STARGA, Inc.
"""``mm export`` — a deterministic bundle over the *admitted* corpus.

A dump and a piece of evidence differ in one property: run the export
twice over an unchanged corpus and the bytes are identical. Without that,
two parties holding "the same" export cannot tell whether a difference
means the corpus changed or the exporter did, and the bundle stops being
something either of them can attest to. Everything here follows from it —
sorted directory walks, sorted record order, sorted JSON keys, no clock
anywhere in the output, and a content digest over the record section that
the envelope carries so a truncated bundle cannot pass as a short one.

**Admission, not selection.** Every record comes from
:func:`~mind_mem.admissibility.admit_corpus`, the one predicate the read
paths share, so a quarantined or pending block is not merely filtered out
of the bundle — it was never in the set the bundle was built from. The
envelope reports ``withheld_count`` rather than staying silent about it:
an export that quietly drops a third of a corpus is a misleading
document, and the honest form says how much was withheld without saying
what.

**Three policies**, registered by construction — an
:class:`ExportPolicy` enters the registry as it is created, so a policy
that exists is a policy that can be named on the command line, and two
policies cannot share a name:

``full``
    Every admitted block, every field, verbatim.
``redacted``
    The same set, with every string value run through the pre-write
    detector chain. Findings are counted in the envelope.
``metadata-only``
    Ids, status, dates, tags and the five provenance fields, plus a
    digest of the content that was left out — enough to prove what
    existed and who wrote it, with none of what it said.

``--since`` filters on the block's own date (``Date``, then ``Updated``,
then ``Created``). A block with no readable date is **excluded** and
counted in ``undated_excluded``: it cannot be shown to fall inside the
window, and quietly including it would make the window a suggestion.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..v4.feature_flags import FeatureDisabledError, is_enabled_for_workspace
from .detectors import Detector, resolve_detectors
from .redaction import MODE_REDACT, RedactionResult, redact

__all__ = [
    "BUNDLE_SCHEMA",
    "COMPLIANCE_EXPORT_FLAG",
    "DATE_FIELDS",
    "ExportBundle",
    "ExportPolicy",
    "FORMATS",
    "UnknownExportPolicyError",
    "build_bundle",
    "policy_names",
    "render_bundle",
    "resolve_export_policy",
]

#: The declared flag this module consumes.
COMPLIANCE_EXPORT_FLAG = "compliance_export"

#: Bundle schema tag. Bump only for a breaking envelope change; a reader
#: that does not recognise it must refuse the bundle rather than guess.
BUNDLE_SCHEMA = "mind-mem/compliance-export/1"

#: Serialisations. Both are deterministic; ``jsonl`` is the machine form
#: and carries the envelope as its first line.
FORMATS: tuple[str, ...] = ("jsonl", "markdown")

#: Block fields consulted for ``--since``, in order of preference.
DATE_FIELDS: tuple[str, ...] = ("Date", "Updated", "Created")

#: Fields ``metadata-only`` keeps.
_METADATA_FIELDS: tuple[str, ...] = (
    "Status",
    "Date",
    "Updated",
    "Created",
    "Tags",
    "ActorId",
    "ActorRole",
    "SessionId",
    "ToolId",
    "Purpose",
)


class UnknownExportPolicyError(ValueError):
    """A policy name no registered policy answers to."""


_POLICIES: dict[str, "ExportPolicy"] = {}


@dataclass(frozen=True)
class ExportPolicy:
    """One named view over the admitted corpus.

    Registers itself on construction. There is no separate ``register()``
    to forget, and a duplicate name raises rather than replacing the
    policy already holding it — a CLI whose ``--policy`` silently resolved
    to the second of two definitions would be the worst kind of
    compliance bug, correct-looking and wrong.
    """

    name: str
    description: str
    redacts: bool = False
    metadata_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an export policy needs a name")
        existing = _POLICIES.get(self.name)
        if existing is not None and existing != self:
            raise ValueError(f"export policy {self.name!r} is already registered with different behaviour")
        _POLICIES[self.name] = self


FULL = ExportPolicy(name="full", description="Every admitted block, every field, verbatim.")
REDACTED = ExportPolicy(
    name="redacted", description="Every admitted block with string values run through the detector chain.", redacts=True
)
METADATA_ONLY = ExportPolicy(
    name="metadata-only",
    description="Ids, status, dates, tags and provenance only, plus a digest of the withheld content.",
    metadata_only=True,
)


def policy_names() -> tuple[str, ...]:
    """Every registered policy name, sorted."""
    return tuple(sorted(_POLICIES))


def resolve_export_policy(name: str) -> ExportPolicy:
    """One policy by name, or a refusal that lists the real ones."""
    try:
        return _POLICIES[name]
    except KeyError:
        raise UnknownExportPolicyError(f"unknown export policy {name!r}; registered: {', '.join(policy_names())}") from None


@dataclass(frozen=True)
class ExportBundle:
    """The envelope plus the records, before serialisation."""

    envelope: dict[str, Any]
    records: tuple[dict[str, Any], ...] = ()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _block_date(block: Mapping[str, Any]) -> Optional[date]:
    for name in DATE_FIELDS:
        raw = block.get(name)
        if not isinstance(raw, str):
            continue
        try:
            return date.fromisoformat(raw.strip()[:10])
        except ValueError:
            continue
    return None


def load_admitted_blocks(workspace: str) -> tuple[list[dict], int]:
    """``(admitted blocks, withheld count)`` over the governed corpus.

    Walks the four governed directories in sorted order, including
    ``*_ARCHIVE.md``: an archived block is still something the corpus
    holds, and a compliance export that silently omits it is answering a
    narrower question than the one it was asked. What it does *not*
    include is anything the admission gate withholds, which is the
    security boundary and is counted rather than hidden.
    """
    from ..admissibility import admit_corpus
    from ..block_parser import parse_file
    from ..corpus_registry import CORPUS_DIRS

    root = os.path.abspath(workspace)
    parsed: list[dict] = []
    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(root, subdir)
        if not os.path.isdir(dir_path):
            continue
        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(".md"):
                continue
            try:
                blocks = parse_file(os.path.join(dir_path, filename))
            except (OSError, ValueError):
                continue
            source = f"{subdir}/{filename}"
            for block in blocks:
                block["_source"] = source
            parsed.extend(blocks)

    admitted = admit_corpus(parsed)
    return admitted, len(parsed) - len(admitted)


def _public_fields(block: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if not k.startswith("_")}


def _content_digest(fields: Mapping[str, Any]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(canonical.encode("utf-8"))


def _redact_value(value: Any, detectors: Sequence[Detector], sink: list[RedactionResult]) -> Any:
    if isinstance(value, str):
        result = redact(value, mode=MODE_REDACT, detectors=detectors)
        sink.append(result)
        return result.text
    if isinstance(value, list):
        return [_redact_value(item, detectors, sink) for item in value]
    if isinstance(value, dict):
        return {k: _redact_value(value[k], detectors, sink) for k in sorted(value)}
    return value


def _record(block: Mapping[str, Any], policy: ExportPolicy, detectors: Sequence[Detector], sink: list[RedactionResult]) -> dict[str, Any]:
    fields = _public_fields(block)
    if policy.metadata_only:
        kept = {name: fields[name] for name in _METADATA_FIELDS if name in fields}
        kept["ContentSha256"] = _content_digest(fields)
        fields = kept
    elif policy.redacts:
        fields = {k: _redact_value(fields[k], detectors, sink) for k in sorted(fields)}
    return {
        "id": str(block.get("_id") or ""),
        "source": str(block.get("_source") or ""),
        "fields": {k: fields[k] for k in sorted(fields)},
    }


def build_bundle(
    workspace: str,
    *,
    policy: str = "full",
    since: Optional[date] = None,
    fmt: str = "jsonl",
    detectors: Sequence[Detector] | None = None,
    require_flag: bool = True,
) -> ExportBundle:
    """Build the bundle for *workspace* under *policy*.

    Refuses unless ``v4.compliance_export`` is on for the workspace. The
    flag is the whole surface's door, probed once here rather than per
    block, so with it off this build does exactly what the build without
    the feature did: nothing.
    """
    if require_flag and not is_enabled_for_workspace(workspace, COMPLIANCE_EXPORT_FLAG):
        raise FeatureDisabledError(
            f"mind-mem v4 surface '{COMPLIANCE_EXPORT_FLAG}' is disabled. "
            f'Enable via mind-mem.json: "v4": {{ "{COMPLIANCE_EXPORT_FLAG}": {{ "enabled": true }} }}'
        )
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; expected one of {list(FORMATS)}")

    chosen = resolve_export_policy(policy)
    chain = resolve_detectors(None) if detectors is None else tuple(detectors)
    admitted, withheld = load_admitted_blocks(workspace)

    kept: list[Mapping[str, Any]] = []
    undated = 0
    for block in admitted:
        if since is not None:
            when = _block_date(block)
            if when is None:
                undated += 1
                continue
            if when < since:
                continue
        kept.append(block)

    sink: list[RedactionResult] = []
    records = [_record(block, chosen, chain, sink) for block in kept]
    records.sort(key=lambda r: (r["source"], r["id"]))

    envelope: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "policy": chosen.name,
        "format": fmt,
        "since": since.isoformat() if since is not None else None,
        "block_count": len(records),
        "withheld_count": withheld,
        "undated_excluded": undated,
    }
    if chosen.redacts:
        envelope["redaction"] = {
            "mode": MODE_REDACT,
            "detectors": [d.name for d in chain],
            "finding_count": sum(len(r.findings) for r in sink),
        }
    envelope["content_sha256"] = _sha256(_records_bytes(records, fmt))
    return ExportBundle(envelope=envelope, records=tuple(records))


def _json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _records_bytes(records: Iterable[Mapping[str, Any]], fmt: str) -> bytes:
    if fmt == "markdown":
        return "".join(_markdown_record(r) for r in records).encode("utf-8")
    return "".join(f"{_json_line(r)}\n" for r in records).encode("utf-8")


def _markdown_record(record: Mapping[str, Any]) -> str:
    lines = [f"## {record['source']} :: {record['id']}", ""]
    fields = record["fields"]
    for key in sorted(fields):
        lines.append(f"- {key}: {_json_line(fields[key]) if not isinstance(fields[key], str) else fields[key]}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_bundle(bundle: ExportBundle) -> bytes:
    """Serialise *bundle*. Same bundle in, same bytes out, always.

    The format is read off the envelope rather than taken as an argument,
    because ``content_sha256`` was computed over the record section *in
    that format*: rendering one format under another envelope's digest
    would produce a bundle that fails its own integrity check.
    """
    fmt = str(bundle.envelope.get("format", ""))
    if fmt not in FORMATS:
        raise ValueError(f"bundle envelope declares format {fmt!r}; expected one of {list(FORMATS)}")
    body = _records_bytes(bundle.records, fmt)
    if fmt == "markdown":
        head = ["# mind-mem compliance export", ""]
        for key in sorted(bundle.envelope):
            head.append(
                f"- {key}: {_json_line(bundle.envelope[key]) if not isinstance(bundle.envelope[key], str) else bundle.envelope[key]}"
            )
        head.append("")
        return ("\n".join(head) + "\n").encode("utf-8") + body
    return f"{_json_line(bundle.envelope)}\n".encode("utf-8") + body
