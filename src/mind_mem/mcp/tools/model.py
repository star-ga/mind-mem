"""Model audit / signing MCP tools — wraps ``mind_mem.model_audit``,
``mind_mem.model_signing``, and ``mind_mem.model_provenance``.

Three tools, each mirroring the equivalent ``mm`` CLI subcommand:

* ``audit_model_tool`` — static security scan of a local checkpoint.
* ``sign_model_tool``  — Ed25519 manifest signing.
* ``verify_model_tool`` — verify a previously-signed checkpoint.

Path-escape guards: every ``path`` argument is rejected if it contains
NULs (filesystem-mismatch attempt) or is an empty string. Concrete
filesystem checks (existence, directory-ness) are delegated to the
underlying functions which already raise structured errors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


def _reject_bad_path(path: str, field: str = "path") -> str | None:
    """Return a JSON error envelope if ``path`` is empty or contains NULs."""
    if not isinstance(path, str) or not path.strip():
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "error": f"{field} must be a non-empty string"},
            indent=2,
        )
    if "\x00" in path:
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "error": f"{field} contains NUL byte"},
            indent=2,
        )
    return None


@mcp_tool_observe
def audit_model_tool(
    path: str,
    allow_publisher: list[str] | None = None,
    include_manifest: bool = False,
) -> str:
    """Static security scan of a local model checkpoint.

    Runs the full seven-check audit (remote-code hooks, no .py files,
    weight format, pickle opcodes, tokenizer injection, safetensors
    header, provenance) and returns a JSON envelope identical to the
    one ``mm audit-model --json`` emits, plus an ``ok`` flag.

    Args:
        path: Filesystem path to a local model directory (HF checkpoint
            layout). Remote / HF-hub IDs are not accepted — agents must
            download first.
        allow_publisher: Optional list of HF org slugs to extend the
            default provenance allowlist. Use for internal fine-tune
            orgs that aren't in the canonical publisher list.
        include_manifest: When True, the response includes the
            per-file SHA-256 manifest. Off by default because manifests
            on multi-GB checkpoints are large.

    Returns:
        JSON object with ``ok`` (bool — passed all checks),
        ``checks`` (list of ``{name, passed, detail, evidence}``),
        ``file_count``, ``total_bytes``, ``model_path``, and (when
        requested) ``manifest`` (mapping ``relpath -> sha256``).
    """
    err = _reject_bad_path(path)
    if err:
        return err

    extra = tuple(allow_publisher) if allow_publisher else None

    try:
        from mind_mem.model_audit import audit_model

        report = audit_model(path, allow_extra_publishers=extra)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": str(exc)},
            indent=2,
        )

    out = report.to_dict()
    if not include_manifest:
        out.pop("manifest", None)
    out["_schema_version"] = MCP_SCHEMA_VERSION
    out["ok"] = report.passed

    metrics.inc("mcp_audit_model")
    _log.info(
        "mcp_audit_model",
        path=path,
        passed=report.passed,
        file_count=report.file_count,
    )
    return json.dumps(out, indent=2, default=str)


@mcp_tool_observe
def sign_model_tool(
    path: str,
    key_file: str = "",
    generate_key_prefix: str = "",
    write_sidecars: bool = True,
) -> str:
    """Sign every file in a local model checkpoint with Ed25519.

    Exactly one of ``key_file`` or ``generate_key_prefix`` must be
    provided — refusing to sign with an unrecorded ephemeral key is
    intentional (the operator could never verify against it again).

    Args:
        path: Filesystem path to a local model directory.
        key_file: Path to a raw 32-byte Ed25519 private key file.
        generate_key_prefix: When set, generate a fresh keypair, write
            ``<prefix>.sk`` (mode 0600) + ``<prefix>.pub`` to disk, and
            sign with it. Mutually exclusive with ``key_file``.
        write_sidecars: When True (default) write
            ``MODEL_MANIFEST.txt`` / ``.sig`` / ``MODEL_PUBKEY.pub``
            next to the checkpoint root.

    Returns:
        JSON object with ``ok`` (bool), ``manifest_sha256``,
        ``signature_hex``, ``public_key_hex``, ``files_signed`` and
        the three sidecar paths (or ``null`` when ``write_sidecars``
        is False).
    """
    err = _reject_bad_path(path)
    if err:
        return err

    if (not key_file and not generate_key_prefix) or (key_file and generate_key_prefix):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "ok": False,
                "error": "exactly one of key_file or generate_key_prefix must be provided",
            },
            indent=2,
        )

    from mind_mem.model_signing import (
        ED25519_PRIVATE_KEY_BYTES,
        generate_keypair,
        sign_model,
    )

    if generate_key_prefix:
        sk, pk = generate_keypair()
        sk_path = Path(os.path.expanduser(generate_key_prefix + ".sk"))
        pk_path = Path(os.path.expanduser(generate_key_prefix + ".pub"))
        sk_path.write_bytes(sk)
        try:
            os.chmod(sk_path, 0o600)
        except OSError:
            pass  # Windows / non-POSIX FS — best effort
        pk_path.write_bytes(pk)
    else:
        sk_path = Path(os.path.expanduser(key_file))
        if not sk_path.is_file():
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": f"key_file not found: {sk_path}",
                },
                indent=2,
            )
        sk = sk_path.read_bytes()
        if len(sk) != ED25519_PRIVATE_KEY_BYTES:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": (f"key_file must be {ED25519_PRIVATE_KEY_BYTES} raw bytes, got {len(sk)} ({sk_path})"),
                },
                indent=2,
            )

    try:
        result = sign_model(path, sk, write_sidecars=write_sidecars)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": str(exc)},
            indent=2,
        )

    metrics.inc("mcp_sign_model")
    _log.info(
        "mcp_sign_model",
        path=path,
        files_signed=len(result.manifest_text.splitlines()),
        manifest_sha256=result.manifest_sha256,
    )
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "ok": True,
            "manifest_sha256": result.manifest_sha256,
            "signature_hex": result.signature.hex(),
            "public_key_hex": result.public_key.hex(),
            "files_signed": len(result.manifest_text.splitlines()),
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            "signature_path": str(result.signature_path) if result.signature_path else None,
            "pubkey_path": str(result.pubkey_path) if result.pubkey_path else None,
        },
        indent=2,
    )


@mcp_tool_observe
def verify_model_tool(path: str, pubkey_path: str = "") -> str:
    """Verify a previously-signed checkpoint.

    Args:
        path: Filesystem path to a local model directory that was
            previously signed with ``sign_model_tool`` (or ``mm
            sign-model``).
        pubkey_path: Optional path to a raw 32-byte Ed25519 public key
            file. When omitted, the ``MODEL_PUBKEY.pub`` sidecar in the
            checkpoint directory is used; when provided, the explicit
            key takes precedence (useful for centrally-pinned trust
            roots).

    Returns:
        JSON object with ``ok`` (bool), ``manifest_sha256``,
        ``error_kind`` (``manifest_mismatch`` / ``bad_signature`` /
        ``missing_file`` / ``null``), and ``error_detail``.
    """
    err = _reject_bad_path(path)
    if err:
        return err

    pk: bytes | None = None
    if pubkey_path:
        from mind_mem.model_signing import ED25519_PUBLIC_KEY_BYTES

        p = Path(os.path.expanduser(pubkey_path))
        if not p.is_file():
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": f"pubkey_path not found: {p}",
                },
                indent=2,
            )
        pk = p.read_bytes()
        if len(pk) != ED25519_PUBLIC_KEY_BYTES:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": (f"pubkey_path must be {ED25519_PUBLIC_KEY_BYTES} raw bytes, got {len(pk)} ({p})"),
                },
                indent=2,
            )

    from mind_mem.model_signing import verify_model

    try:
        result = verify_model(path, public_key=pk)
    except NotADirectoryError as exc:
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": str(exc)},
            indent=2,
        )

    metrics.inc("mcp_verify_model")
    _log.info(
        "mcp_verify_model",
        path=path,
        passed=result.passed,
        error_kind=result.error_kind,
    )
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "ok": result.passed,
            "manifest_sha256": result.manifest_sha256,
            "error_kind": result.error_kind,
            "error_detail": result.error_detail,
        },
        indent=2,
    )


def register(mcp) -> None:
    """Wire the three model tools onto *mcp*."""
    mcp.tool(audit_model_tool)
    mcp.tool(sign_model_tool)
    mcp.tool(verify_model_tool)
