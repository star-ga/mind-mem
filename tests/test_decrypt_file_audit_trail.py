"""Regression test for the `decrypt_file` forensic audit trail
(N-08, roadmap v4.0.15).

Every successful ``decrypt_file`` invocation appends a JSON line to
``memory/decrypted_files.jsonl`` carrying timestamp + path + actor +
mode. The trail is append-only; pair with the T-007 operator runbook
at ``docs/append-only-audit-logs.md`` for the OS-level
append-only attribute (``chattr +a`` / ``chflags uappnd``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def encrypted_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a workspace + encrypt a real file so decrypt_file has work."""
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    (ws / "decisions").mkdir()
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "test-passphrase")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", str(ws))
    # decrypt_file is admin-scope; we need MIND_MEM_SCOPE=admin to bypass
    # the ACL gate in tests.
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    # Write + encrypt a real file via the EncryptionManager.
    target = ws / "decisions" / "secret.md"
    target.write_text("highly confidential statement")
    from mind_mem.encryption import EncryptionManager

    EncryptionManager(str(ws), "test-passphrase").encrypt_file(str(target))
    return ws, target


def test_decrypt_file_appends_jsonl_record(
    encrypted_workspace: tuple[Path, Path],
) -> None:
    """A successful decrypt_file call must append one line to
    memory/decrypted_files.jsonl carrying ts + path + actor + mode."""
    from mind_mem.mcp.tools.encryption import decrypt_file

    ws, target = encrypted_workspace
    log = ws / "memory" / "decrypted_files.jsonl"
    assert not log.exists()  # not created until first decrypt

    out = decrypt_file(str(target))
    parsed = json.loads(out)
    assert "plaintext_b64" in parsed, f"decrypt failed: {parsed!r}"

    assert log.exists(), "decrypted_files.jsonl not created"
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # Required fields per the doc.
    assert "ts" in record
    assert record["ts"].endswith("Z")  # ISO-8601 UTC marker
    assert record["path"] == str(target)
    assert "actor" in record
    assert record["mode"] == "read"


def test_decrypt_file_appends_one_record_per_call(
    encrypted_workspace: tuple[Path, Path],
) -> None:
    """Each call appends a fresh line — never rewrites or truncates."""
    from mind_mem.mcp.tools.encryption import decrypt_file

    ws, target = encrypted_workspace
    log = ws / "memory" / "decrypted_files.jsonl"

    for _ in range(3):
        decrypt_file(str(target))

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 3
    # All three must parse and carry distinct timestamps (or at least
    # be valid JSON — timestamps may collide at second granularity in
    # a tight loop).
    for line in lines:
        rec = json.loads(line)
        assert rec["path"] == str(target)
        assert rec["mode"] == "read"


def test_decrypt_file_failure_does_not_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When decrypt_file fails (bad path / missing passphrase), the
    audit trail must NOT carry a line — only successful decrypts are
    forensically interesting (a failure is observable in the error log)."""
    ws = tmp_path / "ws"
    (ws / "memory").mkdir(parents=True)
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "test-passphrase")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", str(ws))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")

    from mind_mem.mcp.tools.encryption import decrypt_file

    # Path doesn't exist → decrypt failure.
    out = decrypt_file(str(ws / "decisions" / "missing.md"))
    parsed = json.loads(out)
    assert "error" in parsed

    log = ws / "memory" / "decrypted_files.jsonl"
    # No successful decrypt happened → the audit file should not exist
    # (or exist empty).
    assert not log.exists() or log.read_text().strip() == ""


def test_decrypt_file_audit_actor_defaults_to_anonymous(
    encrypted_workspace: tuple[Path, Path],
) -> None:
    """In the absence of an explicit agent-id ContextVar, the audit
    record carries 'anonymous' as the actor (best-effort attribution
    — the ContextVar is populated by the REST + MCP layer; bare
    function calls fall through to 'anonymous'). When the layer that
    populates it is added (deferred audit attribution item), this
    test gains a 'with actor=…' counterpart."""
    from mind_mem.mcp.tools.encryption import decrypt_file

    ws, target = encrypted_workspace
    decrypt_file(str(target))

    log = ws / "memory" / "decrypted_files.jsonl"
    record = json.loads(log.read_text().strip().splitlines()[0])
    assert record["actor"] == "anonymous"
