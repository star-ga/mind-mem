"""Backend-aware enumeration on the ``encrypted`` block-store backend.

``block_store.backend = "encrypted"`` is a supported production config
(``mind-mem-init --backend encrypted`` writes it) whose blocks of record
are the ordinary Markdown corpus files — rewritten in place as
ciphertext by ``encrypt_workspace`` / the ``encrypt_file`` MCP tool.

Regression: :func:`mind_mem.storage.iter_active_blocks` read those files
with the plain ``parse_file``, which decodes ciphertext with
``errors="replace"``, finds no ``[ID]`` header and returns **zero blocks
without raising**. Everything routed through that primitive (reindex,
scan, drift, dream-cycle, export, workspace health) then ran on an empty
corpus and reported success on a workspace full of blocks. These tests
pin the decrypting read path, the source tags / ``#429`` rule the walk
must keep, and the refusal to report an empty corpus when the key is
absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.block_parser import parse_file
from mind_mem.block_store_encrypted import encrypt_workspace
from mind_mem.encryption import _MAGIC
from mind_mem.storage import iter_active_blocks, iter_blocks

_PASSPHRASE = "encrypted-backend-regression-passphrase"


def _block(bid: str, statement: str, status: str = "active") -> str:
    return f"[{bid}]\nStatement: {statement}\nStatus: {status}\nDate: 2026-06-13\n\n---\n"


def _encrypted_workspace(ws: Path) -> None:
    """Scaffold a workspace configured for the ``encrypted`` backend."""
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    config = {"recall": {"backend": "bm25"}, "block_store": {"backend": "encrypted"}}
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")


def _is_ciphertext(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(_MAGIC)) == _MAGIC


def test_encrypted_corpus_blocks_stay_visible_after_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented encrypt migration must not empty the active corpus."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    _encrypted_workspace(ws)
    decisions = ws / "decisions" / "DECISIONS.md"
    decisions.write_text(_block("D-20260613-201", "encrypted blocks stay visible"), encoding="utf-8")

    assert {b["_id"] for b in iter_active_blocks(str(ws))} == {"D-20260613-201"}

    assert encrypt_workspace(str(ws))["encrypted"] >= 1

    # Mechanism: the bytes on disk really are ciphertext, and the plain
    # reader really cannot see the block — so the only way the block can
    # come back below is that the enumeration decrypted it.
    assert _is_ciphertext(decisions)
    assert parse_file(str(decisions)) == []

    assert {b["_id"] for b in iter_active_blocks(str(ws))} == {"D-20260613-201"}


def test_encrypted_corpus_keeps_source_tags_and_pending_signal_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decryption happens inside the corpus walk, not instead of it.

    Routing the encrypted backend through ``get_block_store().get_all()``
    would also return blocks, but would drop ``_source_file`` /
    ``_source_label`` (which reindex writes as the block's path) and the
    ``#429`` pending-signal exclusion. Both must survive encryption.
    """
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    _encrypted_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(
        _block("D-20260613-202", "an active call") + _block("D-20260613-203", "an archived call", "archived"),
        encoding="utf-8",
    )
    (ws / "intelligence" / "SIGNALS.md").write_text(
        _block("SIG-20260613-201", "unreviewed", "pending") + _block("SIG-20260613-202", "reviewed"),
        encoding="utf-8",
    )
    encrypt_workspace(str(ws))

    blocks = iter_active_blocks(str(ws))
    by_id = {b["_id"]: b for b in blocks}

    assert set(by_id) == {"D-20260613-202", "SIG-20260613-202"}
    assert by_id["D-20260613-202"]["_source_file"] == "decisions/DECISIONS.md"
    assert by_id["D-20260613-202"]["_source_label"] == "decisions"
    assert by_id["SIG-20260613-202"]["_source_file"] == "intelligence/SIGNALS.md"
    assert by_id["SIG-20260613-202"]["_source_label"] == "signals"

    # active_only=False (the mailbox shape) decrypts too, and skips both
    # status filters rather than the decryption.
    assert {b["_id"] for b in iter_blocks(str(ws), active_only=False)} == {
        "D-20260613-202",
        "D-20260613-203",
        "SIG-20260613-201",
        "SIG-20260613-202",
    }


def test_partially_encrypted_corpus_reads_both_halves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``memory/`` is outside the encrypted dirs, so mixed corpora are normal."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    _encrypted_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(_block("D-20260613-204", "encrypted half"), encoding="utf-8")
    (ws / "memory" / "INBOX.md").write_text(_block("INBOX-20260613-201", "plaintext half"), encoding="utf-8")
    encrypt_workspace(str(ws))

    assert _is_ciphertext(ws / "decisions" / "DECISIONS.md")
    assert not _is_ciphertext(ws / "memory" / "INBOX.md")
    assert {b["_id"] for b in iter_active_blocks(str(ws))} == {"D-20260613-204", "INBOX-20260613-201"}


def test_encrypted_corpus_without_passphrase_refuses_to_report_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No key + ciphertext on disk is unreadable, never an empty corpus."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    _encrypted_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(_block("D-20260613-205", "still here"), encoding="utf-8")
    encrypt_workspace(str(ws))

    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE", raising=False)
    with pytest.raises(ValueError, match="MIND_MEM_ENCRYPTION_PASSPHRASE"):
        iter_active_blocks(str(ws))


def test_encrypted_backend_before_migration_needs_no_passphrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured-but-not-yet-encrypted workspace keeps reading plainly.

    Guard against over-correcting into "backend=encrypted always demands
    a key": the corpus here is plaintext, so the plain reader is right
    and no PBKDF2 key derivation should be forced on a read.
    """
    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE", raising=False)
    ws = tmp_path / "ws"
    _encrypted_workspace(ws)
    (ws / "decisions" / "DECISIONS.md").write_text(_block("D-20260613-206", "not migrated yet"), encoding="utf-8")

    assert {b["_id"] for b in iter_active_blocks(str(ws))} == {"D-20260613-206"}
    # No key material was minted just to read plaintext.
    assert not (ws / ".mind-mem-keys").exists()
