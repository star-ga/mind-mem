"""At-rest encryption MCP tools — ``encrypt_file`` / ``decrypt_file``.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, encryption domain). Both tools require
``MIND_MEM_ENCRYPTION_PASSPHRASE`` to be set in the server
environment and are admin-scope — never exposed to user tokens.
The ``_encryption_passphrase`` resolver and the
``_safe_vault_path`` guard move with them because they are only
used by this pair.

Registration: ``register(mcp)`` wires the ``@mcp.tool`` decorator
onto each function after construction. The
``@mcp_tool_observe`` wrapping is applied at definition time so
direct ``mcp_server.encrypt_file(...)`` calls still go through
the rate-limit + ACL + timing gates.
"""

from __future__ import annotations

import base64
import json
import os

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ..infra.workspace import _workspace


def _encryption_passphrase() -> str | None:
    """Fetch the at-rest encryption passphrase from env. None = disabled."""
    raw = os.environ.get("MIND_MEM_ENCRYPTION_PASSPHRASE", "").strip()
    return raw or None


def _safe_vault_path(ws: str, candidate: str) -> str:
    """Resolve *candidate* against *ws* and reject path-escapes."""
    resolved = os.path.realpath(candidate)
    ws_abs = os.path.realpath(ws)
    try:
        common = os.path.commonpath([resolved, ws_abs])
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {candidate}") from exc
    if common != ws_abs:
        raise ValueError(f"path escapes workspace: {candidate}")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


@mcp_tool_observe
def encrypt_file(file_path: str) -> str:
    """Encrypt a single workspace file at rest.

    Requires ``MIND_MEM_ENCRYPTION_PASSPHRASE`` to be set in the
    server environment. Files already encrypted are no-ops.
    Admin-scope tool — never exposed to user tokens.

    Args:
        file_path: Absolute path to the plaintext file.

    Returns:
        JSON status envelope.
    """
    passphrase = _encryption_passphrase()
    if not passphrase:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "MIND_MEM_ENCRYPTION_PASSPHRASE is not set",
            }
        )
    if not isinstance(file_path, str) or not file_path.strip():
        return json.dumps({"error": "file_path must be a non-empty string"})
    ws = _workspace()
    try:
        safe_path = _safe_vault_path(ws, file_path)
    except Exception as exc:
        return json.dumps({"error": f"path rejected: {exc}"})
    try:
        from mind_mem.encryption import EncryptionManager

        EncryptionManager(ws, passphrase).encrypt_file(safe_path)
    except Exception as exc:
        return json.dumps({"error": f"encrypt failed: {exc}"})
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "encrypted": True,
            "path": safe_path,
        }
    )


@mcp_tool_observe
def decrypt_file(file_path: str) -> str:
    """Return plaintext bytes (base64-encoded) for an encrypted file.

    Does not modify the on-disk ciphertext. Admin-scope tool.

    Args:
        file_path: Absolute path to the encrypted file.

    Returns:
        JSON with ``plaintext_b64`` field on success.
    """
    passphrase = _encryption_passphrase()
    if not passphrase:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "MIND_MEM_ENCRYPTION_PASSPHRASE is not set",
            }
        )
    if not isinstance(file_path, str) or not file_path.strip():
        return json.dumps({"error": "file_path must be a non-empty string"})
    ws = _workspace()
    try:
        safe_path = _safe_vault_path(ws, file_path)
    except Exception as exc:
        return json.dumps({"error": f"path rejected: {exc}"})
    try:
        from mind_mem.encryption import EncryptionManager

        plaintext = EncryptionManager(ws, passphrase).decrypt_file(safe_path)
    except Exception as exc:
        return json.dumps({"error": f"decrypt failed: {exc}"})
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "path": safe_path,
            "plaintext_b64": base64.b64encode(plaintext).decode("ascii"),
        }
    )


def register(mcp) -> None:
    """Wire the encryption tools onto *mcp*."""
    mcp.tool(encrypt_file)
    mcp.tool(decrypt_file)
