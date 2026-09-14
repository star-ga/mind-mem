# Copyright 2026 STARGA, Inc.
"""Bounded local export and offline verification for served-ledger evidence.

This module is the RE.1/RE.2 local adapter.  It captures the existing served
ledger and its head sidecar under the ledger's append lock, then releases the
lock before doing parsing and package serialization.  The package is a local
audit artifact: it proves only the captured chain's local consistency.  It
contains no query text, source text, scoring, payment, signature, or external
witness claim.

The package keeps the exact JSONL and head bytes rather than minting a second
row or digest protocol.  ``served_ledger.decode_row`` and ``row_hash`` remain
the owners of V1/V2 schemas and row-chain hashing.  A caller can retain the
returned manifest digest separately; without that independent retention, a
self-consistent package is still only local inspection.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .recall_digests import served_set_digest
from .served_ledger import (
    GENESIS_ROW_HASH,
    HEAD_RELPATH,
    LEDGER_RELPATH,
    ServedRun,
    ServedRunV2,
    _append_lock,
    decode_row,
    ledger_path,
    row_hash,
    run_id,
)

__all__ = [
    "DEFAULT_MAX_LEDGER_BYTES",
    "DEFAULT_MAX_PACKAGE_BYTES",
    "DEFAULT_MAX_ROWS",
    "RECEIPT_SCHEMA",
    "ReceiptError",
    "ReceiptUnavailable",
    "export_receipt",
    "verify_receipt",
    "write_receipt",
]

RECEIPT_SCHEMA = "mind-mem/retrieval-receipt/1"
_PROFILE = "local-served-ledger-v1"
_DEFAULT_MAX_MB = 8
DEFAULT_MAX_LEDGER_BYTES = _DEFAULT_MAX_MB * 1024 * 1024
DEFAULT_MAX_PACKAGE_BYTES = 12 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
_REQUIRED_KEYS = frozenset({"schema", "profile", "manifest", "ledger_b64", "head_b64", "manifest_sha256"})
_MANIFEST_KEYS = frozenset(
    {
        "ledger_relpath",
        "head_relpath",
        "ledger_bytes",
        "ledger_rows",
        "ledger_sha256",
        "ledger_identity",
        "head_present",
        "head_bytes",
        "head_sha256",
        "head_identity",
        "scope",
        "portable_identity",
        "disclosure_profile",
    }
)


class ReceiptError(ValueError):
    """The receipt is malformed, unsupported, or fails local integrity checks."""


class ReceiptUnavailable(ReceiptError):
    """The source ledger cannot provide a non-empty immutable snapshot."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_stable(path: Path, *, limit: int, optional: bool = False) -> tuple[bytes | None, dict[str, int] | None]:
    """Read one file and prove its identity did not move during the read."""
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > limit:
                raise ReceiptUnavailable(f"{path.name} exceeds the {limit}-byte snapshot limit")
            raw = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
    except FileNotFoundError:
        if optional:
            return None, None
        raise ReceiptUnavailable(f"required evidence file is absent: {path}") from None
    except (OSError, ValueError) as exc:
        raise ReceiptUnavailable(f"cannot read evidence file {path}: {exc}") from exc
    if len(raw) > limit:
        raise ReceiptUnavailable(f"{path.name} exceeds the {limit}-byte snapshot limit")
    if _stat_identity(before) != _stat_identity(after):
        raise ReceiptUnavailable(f"evidence file changed while captured: {path}")
    return raw, _stat_identity(after)


def _decode_rows(raw: bytes, *, max_rows: int) -> tuple[dict[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReceiptError(f"ledger is not UTF-8: {exc}") from exc
    lines = text.splitlines()
    if len(lines) > max_rows:
        raise ReceiptError(f"ledger has more than the {max_rows}-row snapshot limit")
    if not lines:
        raise ReceiptUnavailable("served ledger is empty; no receipt evidence is available")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            raise ReceiptError(f"ledger row {index} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReceiptError(f"ledger row {index} is malformed JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ReceiptError(f"ledger row {index} is not a JSON object")
        rows.append(value)
    return tuple(rows)


def _check_rows(rows: tuple[dict[str, Any], ...], head: bytes | None) -> tuple[bool, str, int]:
    """Check row schemas, derived fields, links, and the head sidecar."""
    decoded: list[ServedRun | ServedRunV2] = []
    for index, raw in enumerate(rows):
        try:
            row = decode_row(raw)
        except (TypeError, ValueError, KeyError) as exc:
            return False, f"row {index} is malformed: {exc}", index
        if row.seq != index:
            return False, f"row {index}: seq {row.seq} out of order", index
        if served_set_digest(row.ids) != row.served_digest:
            return False, f"row {index}: served_digest does not match ids", index
        expected_run = run_id(
            query_hash=row.query_hash,
            served_digest=row.served_digest,
            pipeline_hash=row.pipeline_hash,
        )
        if row.run_id != expected_run:
            return False, f"row {index}: run_id is not derived from the row", index
        if index == 0:
            expected_prev = GENESIS_ROW_HASH
        else:
            expected_prev = row_hash(decoded[-1])
        if row.prev_row_hash != expected_prev:
            return False, f"row {index}: predecessor link does not match", index
        decoded.append(row)

    expected_head = row_hash(decoded[-1])
    if head is None:
        return False, "head sidecar is absent for a non-empty ledger", len(rows) - 1
    try:
        head_text = head.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        return False, f"head sidecar is not UTF-8: {exc}", len(rows) - 1
    if head_text != expected_head:
        return False, "head sidecar does not match the captured tail", len(rows) - 1
    return True, "", len(rows)


def _manifest_payload(package: Mapping[str, Any]) -> dict[str, Any]:
    return {key: package[key] for key in sorted(package) if key != "manifest_sha256"}


def _package_bytes(package: Mapping[str, Any]) -> bytes:
    return _canonical(package)


def export_receipt(
    workspace: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_LEDGER_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> bytes:
    """Capture a bounded, immutable local served-ledger package.

    The ledger lock covers both source reads.  Parsing, verification and JSON
    serialization happen after release, so a long package cannot block an
    ordinary append.  A missing, empty, unreadable, malformed, or over-limit
    source raises an explicit error; no empty package is treated as proof.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    ledger = Path(ledger_path(workspace))
    head_path = Path(workspace) / HEAD_RELPATH
    with _append_lock(workspace):
        ledger_raw, ledger_identity = _read_stable(ledger, limit=max_bytes)
        head_raw, head_identity = _read_stable(head_path, limit=4096, optional=True)
    assert ledger_raw is not None
    rows = _decode_rows(ledger_raw, max_rows=max_rows)
    if head_raw is None:
        raise ReceiptError("head sidecar is absent for a non-empty ledger")
    ok, reason, _ = _check_rows(rows, head_raw)
    if not ok:
        raise ReceiptError(f"captured served ledger is not locally consistent: {reason}")

    manifest = {
        "ledger_relpath": LEDGER_RELPATH,
        "head_relpath": HEAD_RELPATH,
        "ledger_bytes": len(ledger_raw),
        "ledger_rows": len(rows),
        "ledger_sha256": _sha256(ledger_raw),
        "ledger_identity": ledger_identity,
        "head_present": True,
        "head_bytes": len(head_raw),
        "head_sha256": _sha256(head_raw),
        "head_identity": head_identity,
        "scope": "local",
        "portable_identity": "unavailable",
        "disclosure_profile": "local-audit-v1",
    }
    package: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "profile": _PROFILE,
        "manifest": manifest,
        "ledger_b64": base64.b64encode(ledger_raw).decode("ascii"),
        "head_b64": base64.b64encode(head_raw).decode("ascii"),
    }
    package["manifest_sha256"] = _sha256(_package_bytes(package))
    output = _package_bytes(package)
    if len(output) > DEFAULT_MAX_PACKAGE_BYTES:
        raise ReceiptUnavailable(f"receipt package exceeds the {DEFAULT_MAX_PACKAGE_BYTES}-byte package limit")
    return output


def _load_package(package: bytes | bytearray | str | Path | Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if isinstance(package, Mapping):
        value: Any = dict(package)
        raw_size = len(_package_bytes(value))
    elif isinstance(package, Path):
        try:
            if package.stat().st_size > DEFAULT_MAX_PACKAGE_BYTES:
                return None, {"status": "unavailable", "reason": "receipt package exceeds size limit"}
            raw = package.read_bytes()
        except OSError as exc:
            return None, {"status": "unavailable", "reason": f"cannot read receipt package: {exc}"}
        raw_size = len(raw)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, {"status": "malformed", "reason": f"receipt package is not valid UTF-8 JSON: {exc}"}
    else:
        raw = package.encode("utf-8") if isinstance(package, str) else bytes(package)
        raw_size = len(raw)
        if raw_size > DEFAULT_MAX_PACKAGE_BYTES:
            return None, {"status": "unavailable", "reason": "receipt package exceeds size limit"}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, {"status": "malformed", "reason": f"receipt package is not valid UTF-8 JSON: {exc}"}
    if raw_size > DEFAULT_MAX_PACKAGE_BYTES:
        return None, {"status": "unavailable", "reason": "receipt package exceeds size limit"}
    if not isinstance(value, dict):
        return None, {"status": "malformed", "reason": "receipt package must be a JSON object"}
    return value, {"status": "loaded"}


def _decode_b64(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise ReceiptError(f"{name} must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ReceiptError(f"{name} is not valid base64") from exc


def verify_receipt(
    package: bytes | bytearray | str | Path | Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a receipt offline and return individual check results.

    ``locally_consistent`` means the captured bytes form an intact V1/V2
    served-ledger chain.  ``expected_manifest_sha256`` is optional external
    retention; without it the report deliberately says ``issuer_trust`` is
    unknown and never upgrades local consistency to an external claim.
    """
    value, loaded = _load_package(package)
    if value is None:
        return {**loaded, "scope": "local", "checks": {}}
    checks: dict[str, Any] = {}
    unknown = set(value) - _REQUIRED_KEYS
    missing = _REQUIRED_KEYS - set(value)
    if unknown or missing:
        return {
            "status": "malformed",
            "scope": "local",
            "checks": {"package_shape": False},
            "reason": f"package keys mismatch; missing={sorted(missing)} unknown={sorted(unknown)}",
        }
    if value.get("schema") != RECEIPT_SCHEMA:
        return {"status": "unsupported", "scope": "local", "checks": {"schema": False}, "reason": "unsupported receipt schema"}
    if value.get("profile") != _PROFILE:
        return {"status": "unsupported", "scope": "local", "checks": {"profile": False}, "reason": "unsupported receipt profile"}
    checks["schema"] = True
    checks["profile"] = True

    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        return {
            "status": "malformed",
            "scope": "local",
            "checks": {**checks, "manifest_shape": False},
            "reason": "manifest keys do not match the receipt schema",
        }
    checks["manifest_shape"] = True
    if (
        manifest.get("scope") != "local"
        or manifest.get("portable_identity") != "unavailable"
        or manifest.get("disclosure_profile") != "local-audit-v1"
    ):
        return {
            "status": "malformed",
            "scope": "local",
            "checks": {**checks, "manifest_scope": False},
            "reason": "receipt does not declare the supported local disclosure scope",
        }
    checks["manifest_scope"] = True
    expected_hash = value.get("manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return {"status": "malformed", "scope": "local", "checks": {**checks, "manifest": False}, "reason": "manifest_sha256 is malformed"}
    actual_hash = _sha256(_package_bytes(_manifest_payload(value)))
    checks["manifest"] = actual_hash == expected_hash
    if expected_manifest_sha256 is not None:
        checks["retained_manifest"] = actual_hash == expected_manifest_sha256
        if not checks["retained_manifest"]:
            return {
                "status": "integrity_failed",
                "scope": "local",
                "checks": checks,
                "reason": "package differs from the retained manifest digest",
            }
    try:
        ledger_raw = _decode_b64(value["ledger_b64"], "ledger_b64")
        head_raw = _decode_b64(value["head_b64"], "head_b64")
    except ReceiptError as exc:
        return {"status": "malformed", "scope": "local", "checks": {**checks, "payload_encoding": False}, "reason": str(exc)}
    checks["payload_encoding"] = True

    checks["ledger_digest"] = _sha256(ledger_raw) == manifest["ledger_sha256"] and len(ledger_raw) == manifest["ledger_bytes"]
    checks["head_digest"] = _sha256(head_raw) == manifest["head_sha256"] and len(head_raw) == manifest["head_bytes"]
    if not checks["ledger_digest"] or not checks["head_digest"] or not checks["manifest"]:
        return {"status": "integrity_failed", "scope": "local", "checks": checks, "reason": "captured bytes do not match the manifest"}
    try:
        rows = _decode_rows(ledger_raw, max_rows=DEFAULT_MAX_ROWS)
    except ReceiptError as exc:
        return {"status": "integrity_failed", "scope": "local", "checks": {**checks, "rows": False}, "reason": str(exc)}
    checks["rows"] = len(rows) == manifest["ledger_rows"]
    if not checks["rows"]:
        return {
            "status": "integrity_failed",
            "scope": "local",
            "checks": checks,
            "reason": "captured row count does not match the manifest",
        }
    chain_ok, reason, rows_checked = _check_rows(rows, head_raw)
    checks["local_chain"] = chain_ok
    report = {
        "status": "locally_consistent" if chain_ok else "integrity_failed",
        "scope": "local",
        "checks": checks,
        "rows_checked": rows_checked,
        "issuer_trust": "unknown",
        "external_anchor": "unknown",
    }
    if not chain_ok:
        report["reason"] = reason
    return report


def write_receipt(
    workspace: str | Path,
    destination: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_LEDGER_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> int:
    """Write an already captured package to an operator-selected path."""
    payload = export_receipt(workspace, max_bytes=max_bytes, max_rows=max_rows)
    Path(destination).write_bytes(payload)
    return len(payload)
