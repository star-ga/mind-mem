"""Drift detection over an ``encrypted`` block-store backend.

``block_store.backend = "encrypted"`` is corpus-resident: its blocks of
record are the ordinary Markdown corpus files, rewritten in place as
ciphertext by ``encrypt_workspace`` / the ``encrypt_file`` tool. Drift
detection took the corpus branch for it and then read
``decisions/DECISIONS.md`` with the plain ``parse_file``, which decodes
ciphertext with ``errors="replace"``, finds no ``[ID]`` header and returns
zero blocks **without raising**. ``DriftDetector.scan()`` saw fewer than
two blocks and reported a clean scan over a corpus full of decisions, with
nothing in the output to tell "encrypted" apart from "empty".

These tests pin the three halves of the fix: the sealed corpus is read
through the decrypting reader, a corpus that cannot be opened refuses
instead of reporting no drift, and the plaintext default path still reads
straight off disk without paying for (or minting) any key material.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.block_parser import parse_file
from mind_mem.block_store import BlockStoreError
from mind_mem.block_store_encrypted import encrypt_workspace
from mind_mem.drift_detector import DriftDetector
from mind_mem.encryption import _MAGIC

_PASSPHRASE = "drift-encrypted-corpus-regression-passphrase"

# Two near-identical decision statements drift detection must pair.
_DRIFT_A = "Use PostgreSQL for all persistent data storage in production"
_DRIFT_B = "Use MongoDB for all persistent data storage in production"


def _workspace(ws: Path, backend: str = "encrypted") -> Path:
    """Scaffold a workspace configured for *backend* with two drifting decisions."""
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    config = {"recall": {"backend": "scan"}, "block_store": {"backend": backend}}
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    decisions = ws / "decisions" / "DECISIONS.md"
    decisions.write_text(
        f"\n[D-20260301-001]\nDate: 2026-03-01\nSubject: {_DRIFT_A}\nStatus: active\n\n"
        f"---\n\n[D-20260302-001]\nDate: 2026-03-02\nSubject: {_DRIFT_B}\nStatus: active\n",
        encoding="utf-8",
    )
    return decisions


def _is_ciphertext(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(_MAGIC)) == _MAGIC


def _pair(signals) -> set[str]:
    return {signals[0].block_a_id, signals[0].block_b_id}


def test_drift_survives_the_documented_encrypt_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Encrypting a workspace must not turn a drifting corpus into a clean scan."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    decisions = _workspace(ws)

    before = DriftDetector(str(ws), similarity_threshold=0.2).scan()
    assert _pair(before) == {"D-20260301-001", "D-20260302-001"}

    assert encrypt_workspace(str(ws))["encrypted"] >= 1

    # Mechanism: the bytes on disk really are ciphertext and the plain
    # reader really cannot see the blocks, so the only way the pair can
    # come back below is that the load decrypted the file.
    assert _is_ciphertext(decisions)
    assert parse_file(str(decisions)) == []

    detector = DriftDetector(str(ws), similarity_threshold=0.2)
    assert {b["_id"] for b in detector._load_blocks()} == {"D-20260301-001", "D-20260302-001"}
    after = detector.scan()
    assert _pair(after) == {"D-20260301-001", "D-20260302-001"}


def test_sealed_corpus_without_passphrase_refuses_instead_of_scanning_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No key for a sealed corpus is "cannot read", never "no drift"."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    decisions = _workspace(ws)
    assert encrypt_workspace(str(ws))["encrypted"] >= 1
    assert _is_ciphertext(decisions)

    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE", raising=False)
    with pytest.raises(ValueError, match="MIND_MEM_ENCRYPTION_PASSPHRASE"):
        DriftDetector(str(ws), similarity_threshold=0.2).scan()


def test_wrong_passphrase_refuses_instead_of_scanning_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rotated/wrong passphrase fails the MAC check loudly, not silently."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE)
    ws = tmp_path / "ws"
    decisions = _workspace(ws)
    assert encrypt_workspace(str(ws))["encrypted"] >= 1
    assert _is_ciphertext(decisions)

    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASSPHRASE + "-rotated")
    with pytest.raises(BlockStoreError, match="decrypt"):
        DriftDetector(str(ws), similarity_threshold=0.2).scan()


def test_plaintext_corpus_reads_straight_off_disk_and_mints_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A not-yet-migrated corpus keeps the plain reader — and pays no KDF cost.

    Guards the other direction of the fix: routing every corpus-resident
    workspace through the decrypting reader would derive a PBKDF2 key and
    mint ``.mind-mem-keys/salt`` on a read of a plaintext file, and would
    fail outright when no passphrase is set.
    """
    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE", raising=False)
    ws = tmp_path / "ws"
    decisions = _workspace(ws)
    assert not _is_ciphertext(decisions)

    signals = DriftDetector(str(ws), similarity_threshold=0.2).scan()
    assert _pair(signals) == {"D-20260301-001", "D-20260302-001"}
    assert not os.path.exists(ws / ".mind-mem-keys")


def test_markdown_default_path_is_unchanged(tmp_path: Path) -> None:
    """The zero-config Markdown / SQLite default still reads the same file."""
    ws = tmp_path / "ws"
    _workspace(ws, backend="markdown")
    signals = DriftDetector(str(ws), similarity_threshold=0.2).scan()
    assert _pair(signals) == {"D-20260301-001", "D-20260302-001"}
    assert not os.path.exists(ws / ".mind-mem-keys")
