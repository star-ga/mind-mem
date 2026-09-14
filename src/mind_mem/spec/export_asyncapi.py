# Copyright 2026 STARGA, Inc.
"""Export and drift-check the outbound Redis Streams event contract.

The product's asynchronous surface is deliberately small: when the opt-in
``redis`` publisher is enabled, :class:`mind_mem.event_fanout.RedisStreamPublisher`
appends one JSON string in the Redis Streams ``data`` field.  This module
documents that actual wire shape as AsyncAPI 3.0 and provides a stdlib-only
drift and wire validator.  It does not describe a consumer service, SSE,
webhooks, or delivery guarantees the publisher does not implement.

Commands::

    python3 -m mind_mem.spec.export_asyncapi --write
    python3 -m mind_mem.spec.export_asyncapi --check
    python3 -m mind_mem.spec.export_asyncapi --write --output /tmp/asyncapi.json
    python3 -m mind_mem.spec.export_asyncapi --write --stdout

The committed artifact is ``sdk/spec/asyncapi.json``.  The structural check
compares the complete document, including the source-observed event kinds;
there is no version-only exception because this is a transport contract.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mind_mem.event_fanout import (
    EVENT_AUDIT_CHAIN_VERIFIED,
    EVENT_BLOCK_PROMOTED,
    EVENT_CONTRADICTION_DETECTED,
    EVENT_PROPOSAL_APPLIED,
    EVENT_ROLLBACK_EXECUTED,
    EVENT_SNAPSHOT_CREATED,
    EVENT_TIER_DEMOTED,
    EVENT_TIER_PROMOTED,
    scrub_payload,
)
from mind_mem.spec._paths import default_artifact
from mind_mem.spec._paths import source_root as resolve_source_root

_DEFAULT_ARTIFACT = default_artifact(__file__, "asyncapi.json")
SPEC_PATH = _DEFAULT_ARTIFACT
DEFAULT_STREAM = "mind-mem:events"
_WIRE_KEYS = frozenset({"kind", "payload", "workspace", "ts_wall"})
_OBSERVED_EVENT_KINDS = (
    EVENT_CONTRADICTION_DETECTED,
    EVENT_PROPOSAL_APPLIED,
    EVENT_ROLLBACK_EXECUTED,
    EVENT_TIER_DEMOTED,
    EVENT_TIER_PROMOTED,
)
_CANONICAL_EVENT_KINDS = (
    EVENT_AUDIT_CHAIN_VERIFIED,
    EVENT_BLOCK_PROMOTED,
    EVENT_CONTRADICTION_DETECTED,
    EVENT_PROPOSAL_APPLIED,
    EVENT_ROLLBACK_EXECUTED,
    EVENT_SNAPSHOT_CREATED,
    EVENT_TIER_DEMOTED,
    EVENT_TIER_PROMOTED,
)
_EVENT_CONSTANTS = {
    name: value
    for name, value in {
        "EVENT_AUDIT_CHAIN_VERIFIED": EVENT_AUDIT_CHAIN_VERIFIED,
        "EVENT_BLOCK_PROMOTED": EVENT_BLOCK_PROMOTED,
        "EVENT_CONTRADICTION_DETECTED": EVENT_CONTRADICTION_DETECTED,
        "EVENT_PROPOSAL_APPLIED": EVENT_PROPOSAL_APPLIED,
        "EVENT_ROLLBACK_EXECUTED": EVENT_ROLLBACK_EXECUTED,
        "EVENT_SNAPSHOT_CREATED": EVENT_SNAPSHOT_CREATED,
        "EVENT_TIER_DEMOTED": EVENT_TIER_DEMOTED,
        "EVENT_TIER_PROMOTED": EVENT_TIER_PROMOTED,
    }.items()
}


def observed_event_kinds(source_root: Path | None = None) -> tuple[str, ...]:
    """Return literal event kinds used by source ``emit_event`` calls.

    Constants alone are not an emitter inventory: the fanout module exports
    eight canonical names, while only the literal calls in product source are
    observed production emitters.  AST parsing keeps the check independent of
    comments and avoids treating documentation or tests as production wiring.
    """
    root = resolve_source_root(__file__) if source_root is None else source_root
    if not root.is_dir():
        raise RuntimeError(f"event source root is missing or not a directory: {root}")
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if path.name == "event_fanout.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except OSError as exc:
            raise RuntimeError(f"cannot read event source {path}") from exc
        except SyntaxError as exc:
            raise RuntimeError(f"event source is not parseable: {path}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            is_emit = isinstance(function, ast.Name) and function.id == "emit_event"
            if not is_emit or len(node.args) < 2:
                continue
            kind = node.args[1]
            if isinstance(kind, ast.Constant) and isinstance(kind.value, str) and kind.value:
                found.add(kind.value)
            elif isinstance(kind, ast.Name) and kind.id in _EVENT_CONSTANTS:
                found.add(_EVENT_CONSTANTS[kind.id])
    return tuple(sorted(found))


def _event_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["kind", "payload", "workspace", "ts_wall"],
        "additionalProperties": False,
        "properties": {
            "kind": {
                "type": "string",
                "minLength": 1,
                "description": "Extensible event kind; canonical and observed sets are listed in x-mind-mem.",
            },
            "payload": {
                "type": "object",
                "description": "Scrubbed scalar/id/hash metadata; block prose is dropped before publishing.",
                "additionalProperties": True,
            },
            "workspace": {"type": ["string", "null"]},
            "ts_wall": {"type": "number"},
        },
    }


def build_live_spec(source_root: Path | None = None) -> dict[str, Any]:
    """Build the AsyncAPI document from the live publisher contract."""
    observed = list(observed_event_kinds(source_root))
    return {
        "asyncapi": "3.0.0",
        "info": {
            "title": "mind-mem outbound events",
            "version": "1",
            "description": (
                "Opt-in outbound Redis Streams notification contract. "
                "The publisher appends a JSON event to one stream; failures are swallowed."
            ),
        },
        "defaultContentType": "application/json",
        "channels": {
            "mindMemEvents": {
                "address": DEFAULT_STREAM,
                "messages": {
                    "event": {"$ref": "#/components/messages/MindMemEvent"},
                },
                "bindings": {
                    "x-redis-stream": {
                        "type": "redis-stream",
                        "stream": DEFAULT_STREAM,
                        "field": "data",
                        "description": (
                            "Redis Streams XADD record. The data field is a JSON string containing the event. "
                            "This extension documents Redis Streams because standard Pub/Sub bindings do not."
                        ),
                    }
                },
            }
        },
        "operations": {
            "publishMindMemEvent": {
                "action": "send",
                "channel": {"$ref": "#/channels/mindMemEvents"},
                "messages": [{"$ref": "#/channels/mindMemEvents/messages/event"}],
                "description": (
                    "Best-effort outbound append from the configured publisher. No consumer group, retry, "
                    "acknowledgement, ordering, or at-least-once guarantee is part of this contract."
                ),
            }
        },
        "components": {
            "messages": {
                "MindMemEvent": {
                    "name": "MindMemEvent",
                    "title": "mind-mem event wire body",
                    "contentType": "application/json",
                    "payload": {"$ref": "#/components/schemas/MindMemEvent"},
                }
            },
            "schemas": {"MindMemEvent": _event_schema()},
        },
        "x-mind-mem": {
            "transport": "redis-stream-xadd",
            "stream": DEFAULT_STREAM,
            "record_field": "data",
            "delivery": "best-effort outbound append; publish failures are swallowed and not retried",
            "canonical_event_kinds": list(_CANONICAL_EVENT_KINDS),
            "observed_source_event_kinds": observed,
            "event_kind_policy": "extensible string; canonical names are advisory taxonomy",
            "payload_policy": "scrub_payload permits bounded ids, hashes, enums, numbers and booleans; prose is dropped",
            "consumer_surface": "none shipped by this module",
        },
    }


def canonical_json(spec: dict[str, Any]) -> str:
    """Serialise a spec deterministically for reviewable diffs."""
    return json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_committed_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the committed artifact and require a JSON object."""
    resolved = SPEC_PATH if path is None else path
    if resolved is None:
        raise FileNotFoundError("no checkout artifact is available; pass --input or --path")
    loaded: Any = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{resolved} does not contain a JSON object")
    return loaded


def structural_diff(committed: dict[str, Any], live: dict[str, Any]) -> str:
    """Return a unified diff, or an empty string when the documents agree."""
    left = canonical_json(committed)
    right = canonical_json(live)
    if left == right:
        return ""
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile="sdk/spec/asyncapi.json (committed)",
            tofile="mind_mem event publisher (live)",
        )
    )


def write_spec(path: Path | None = None) -> str:
    """Write the live contract and return its canonical text."""
    resolved = SPEC_PATH if path is None else path
    if resolved is None:
        raise RuntimeError("no checkout artifact is available; pass --output, --path, or --stdout")
    text = canonical_json(build_live_spec())
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return text


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _payload_is_scrubbed(payload: dict[str, Any]) -> bool:
    for key, value in payload.items():
        if key == "_dropped":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                return False
            continue
        if scrub_payload({key: value}) != {key: value}:
            return False
    return True


def validate_wire_fields(fields: Any) -> dict[str, Any]:
    """Validate and decode one captured Redis Streams record.

    ``fields`` is the mapping passed to ``RedisStreamPublisher.xadd``.  The
    returned object is a fresh decoded wire body.  This validator intentionally
    accepts the publisher's extensible event kinds while enforcing its exact
    envelope and payload-scrubbing boundary.
    """
    if not isinstance(fields, dict) or set(fields) != {"data"} or not isinstance(fields["data"], str):
        raise ValueError("Redis event record must contain exactly one string data field")
    try:
        body = json.loads(fields["data"], object_pairs_hook=_no_duplicate_pairs)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Redis event data is not valid unique-key JSON") from exc
    if not isinstance(body, dict) or set(body) != _WIRE_KEYS:
        raise ValueError("event wire body has unexpected keys")
    if not isinstance(body["kind"], str) or not body["kind"]:
        raise ValueError("event kind must be a non-empty string")
    if not isinstance(body["payload"], dict) or not _payload_is_scrubbed(body["payload"]):
        raise ValueError("event payload is not scrubbed according to the publisher policy")
    if body["workspace"] is not None and not isinstance(body["workspace"], str):
        raise ValueError("event workspace must be a string or null")
    if isinstance(body["ts_wall"], bool) or not isinstance(body["ts_wall"], (int, float)) or not math.isfinite(body["ts_wall"]):
        raise ValueError("event ts_wall must be a finite number")
    return copy.deepcopy(body)


def validate_wire_record(record: Any, *, expected_stream: str = DEFAULT_STREAM) -> dict[str, Any]:
    """Validate a captured ``xadd`` call, including its stream address."""
    if not isinstance(record, Mapping) or set(record) != {"stream", "fields"}:
        raise ValueError("captured Redis record must contain stream and fields")
    if record["stream"] != expected_stream:
        raise ValueError("captured Redis record targets an unexpected stream")
    return validate_wire_fields(record["fields"])


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or drift-check sdk/spec/asyncapi.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the committed artifact")
    mode.add_argument("--check", action="store_true", help="fail when the artifact has drifted")
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument("--input", type=Path, help="artifact to verify")
    artifact.add_argument("--output", type=Path, help="artifact to write")
    artifact.add_argument("--path", type=Path, help="artifact path (input for --check, output for --write)")
    parser.add_argument("--stdout", action="store_true", help="write the generated artifact to stdout (with --write)")
    args = parser.parse_args(argv)
    if args.stdout and not args.write:
        parser.error("--stdout requires --write")
    if args.write and args.input is not None:
        parser.error("--input is only valid with --check")
    if args.check and args.output is not None:
        parser.error("--output is only valid with --write")
    if args.write:
        if args.stdout:
            sys.stdout.write(canonical_json(build_live_spec()))
            return 0
        target = args.output or args.path
        try:
            write_spec(target)
        except (OSError, RuntimeError) as exc:
            print(f"CANNOT WRITE SPEC: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {target or SPEC_PATH}")
        return 0
    target = args.input or args.path
    try:
        committed = load_committed_spec(target)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        shown = target or SPEC_PATH or "<no checkout artifact>"
        print(f"MISSING OR INVALID: {shown}: {exc}\nPass --input/--path to verify an artifact", file=sys.stderr)
        return 1
    try:
        diff = structural_diff(committed, build_live_spec())
    except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
        print(f"CANNOT BUILD LIVE SPEC: {exc}", file=sys.stderr)
        return 1
    if diff:
        sys.stderr.write(diff)
        print(
            "\nDRIFT: sdk/spec/asyncapi.json no longer matches the live event publisher.\n"
            f"Regenerate with: python3 {Path(__file__)} --write --output <path>",
            file=sys.stderr,
        )
        return 1
    print(f"ok: {target or SPEC_PATH} matches the live outbound event contract")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
