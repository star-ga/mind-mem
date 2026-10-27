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

import contextlib
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from mind_mem.path_redact import redact_host_paths, supplied_pairs

from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import mcp_tool_observe
from ._helpers import get_logger, metrics

_log = get_logger("mcp_server")


def _write_private_key(sk_path: Path, sk: bytes) -> None:
    """Write a raw private key that is owner-only from the first byte.

    ``Path.write_bytes`` opens with the default ``0666 & ~umask``, so under
    the common 022 umask the signing key exists world-readable for the
    width of the chmod that follows it. Create it ``0600`` in the same
    syscall instead, then confirm the mode that actually landed: an
    ``O_CREAT`` mode applies only to a newly created file, and a mount with
    a fixed fmask (or a squashing NFS export) ignores it outright. A key
    whose permissions cannot be restricted is removed rather than kept —
    reporting a successful signing while leaving a 0644 private key on a
    shared host is the state worth refusing.

    Raises:
        OSError: The key could not be written, or could not be restricted
            to owner-only. Nothing is left on disk in that case.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(sk_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(sk)
            fh.flush()
            if os.name != "posix":
                return
            mode = os.fstat(fh.fileno()).st_mode & 0o777
            if not mode & 0o077:
                return
            try:
                os.fchmod(fh.fileno(), 0o600)
            except OSError as exc:
                raise PermissionError(f"cannot restrict private key {sk_path} to owner-only: {exc}") from exc
            mode = os.fstat(fh.fileno()).st_mode & 0o777
            if mode & 0o077:
                raise PermissionError(f"private key {sk_path} landed with mode {mode:04o}; this filesystem will not make it owner-only")
    except BaseException:
        # Never leave a private key on disk at a mode we could not secure.
        with contextlib.suppress(OSError):
            os.unlink(sk_path)
        raise


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


def _bound_principal() -> tuple[bool, str | None]:
    """``(refuse, principal)``: refuse when the caller cannot be established."""
    from mind_mem.read_authz import CallerUnresolved, caller_principal, is_bound

    try:
        principal = caller_principal()
    except CallerUnresolved:
        return True, None
    return False, principal if is_bound(principal) else None


def _configured_model_dirs(ws: str) -> tuple[str, ...]:
    """Absolute, resolved ``model_dirs`` from the workspace config (operator-designated)."""
    from mind_mem.storage import _load_workspace_config

    try:
        raw = _load_workspace_config(ws, quiet=True).get("model_dirs")
    except Exception:  # config unreadable: no extra directories
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(os.path.realpath(os.path.expanduser(d)) for d in raw if isinstance(d, str) and d.strip())


def _model_path_allowed(path: str, *, want_file: bool = False) -> bool:
    """May the current caller look at *path* at all?

    An unbound caller (stdio operator) keeps today's behaviour. A bound caller
    may only touch a configured ``model_dirs`` directory or a path inside the
    workspace its namespace ACL lets it read; see
    :func:`mind_mem.read_authz.path_readable`. An unresolvable caller is refused.
    """
    refuse, principal = _bound_principal()
    if refuse:
        return False
    if principal is None:
        return True
    from mind_mem.mcp.infra.workspace import _workspace
    from mind_mem.read_authz import path_readable

    ws = _workspace()
    return path_readable(ws, path, principal, want_file=want_file, extra_roots=_configured_model_dirs(ws))


def _display_path(path: str) -> str:
    """The path exactly as the caller supplied it: never normalised, expanded or resolved."""
    return path


def _echo(text: str, *supplied: str) -> str:
    """Library-produced *text* with every resolved form of a supplied path put back as the caller spelled it."""
    return redact_host_paths(text, supplied=supplied_pairs(*supplied))


def _echo_obj(obj: Any, *supplied: str) -> Any:
    """:func:`_echo` over every string leaf of a JSON-shaped value."""
    if isinstance(obj, str):
        return _echo(obj, *supplied)
    if isinstance(obj, dict):
        return {k: _echo_obj(v, *supplied) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_echo_obj(v, *supplied) for v in obj]
    return obj


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

    # A bound caller reaches only paths its ACL grants; every other path --
    # unreadable, outside the workspace, absent -- gets the "not found" body.
    # File names, hashes, counts and sizes of an unreadable namespace are
    # content, and "missing vs present" is an existence oracle.
    _refused, _principal = _bound_principal()
    masked = _refused or _principal is not None
    if masked and not _model_path_allowed(path):
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": f"model path not found: {_display_path(path)}"},
            indent=2,
        )

    try:
        from mind_mem.model_audit import audit_model

        report = audit_model(path, allow_extra_publishers=extra)
    except (FileNotFoundError, NotADirectoryError) as exc:
        shown = _display_path(path)
        msg = f"model path not found: {shown}" if isinstance(exc, FileNotFoundError) else f"model path is not a directory: {shown}"
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": msg},
            indent=2,
        )

    out = _echo_obj(report.to_dict(), path)
    out["model_path"] = _display_path(path)
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
            ``<prefix>.sk`` (created mode 0600, never wider for an
            instant) + ``<prefix>.pub`` to disk, and sign with it. If the
            filesystem will not hold the key at owner-only permissions the
            key is deleted and the call fails — it never reports success
            over a world-readable signing key. Mutually exclusive with
            ``key_file``.
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
        try:
            _write_private_key(sk_path, sk)
        except OSError as exc:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": f"could not write private key securely: {exc}",
                },
                indent=2,
            )
        pk_path.write_bytes(pk)
    else:
        sk_path = Path(os.path.expanduser(key_file))
        if not sk_path.is_file():
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": f"key_file not found: {key_file}",
                },
                indent=2,
            )
        sk = sk_path.read_bytes()
        if len(sk) != ED25519_PRIVATE_KEY_BYTES:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": (f"key_file must be {ED25519_PRIVATE_KEY_BYTES} raw bytes, got {len(sk)} ({key_file})"),
                },
                indent=2,
            )

    try:
        result = sign_model(path, sk, write_sidecars=write_sidecars)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return json.dumps(
            {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": _echo(str(exc), path)},
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
            "manifest_path": _echo(str(result.manifest_path), path) if result.manifest_path else None,
            "signature_path": _echo(str(result.signature_path), path) if result.signature_path else None,
            "pubkey_path": _echo(str(result.pubkey_path), path) if result.pubkey_path else None,
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

    _refused, _principal = _bound_principal()
    masked = _refused or _principal is not None
    pk: bytes | None = None
    if pubkey_path:
        from mind_mem.model_signing import ED25519_PUBLIC_KEY_BYTES

        p = Path(os.path.expanduser(pubkey_path))
        # A bound caller's key file is subject to the same ACL as its
        # checkpoint; an unreadable one answers exactly like an absent one.
        if (masked and not _model_path_allowed(pubkey_path, want_file=True)) or not p.is_file():
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": f"pubkey_path not found: {pubkey_path}",
                },
                indent=2,
            )
        pk = p.read_bytes()
        if len(pk) != ED25519_PUBLIC_KEY_BYTES:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "ok": False,
                    "error": (f"pubkey_path must be {ED25519_PUBLIC_KEY_BYTES} raw bytes, got {len(pk)} ({pubkey_path})"),
                },
                indent=2,
            )

    from mind_mem.model_signing import MANIFEST_FILENAME, VerifyResult, verify_model

    if masked and not _model_path_allowed(path):
        # Same result an absent checkpoint produces, with the caller's own path.
        result = VerifyResult(
            passed=False,
            manifest_sha256="",
            error_kind="missing_file",
            error_detail=f"{MANIFEST_FILENAME} not found in {_display_path(path)}",
        )
    else:
        try:
            result = verify_model(path, public_key=pk)
        except NotADirectoryError:
            msg = f"checkpoint root not a directory: {_display_path(path)}"
            return json.dumps(
                {"_schema_version": MCP_SCHEMA_VERSION, "ok": False, "error": msg},
                indent=2,
            )
        if result.error_detail:
            result = dataclasses.replace(result, error_detail=_echo(result.error_detail, path, pubkey_path))

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
