"""Regression controls for ledger refusal across path aliases."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest

from mind_mem import backup_restore
from mind_mem.apply_engine import restore_snapshot
from mind_mem.backup_restore import backup_workspace, restore_workspace
from mind_mem.block_store import _carry_ledgers_into
from mind_mem.corpus_registry import is_ledger_path, is_ledger_target


def _workspace(root: Path) -> Path:
    for name in ("decisions", "tasks", "entities", "intelligence", "memory", "summaries"):
        (root / name).mkdir(parents=True)
    (root / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "markdown"}}), encoding="utf-8")
    (root / "decisions" / "DECISIONS.md").write_text("# Decisions\n\n", encoding="utf-8")
    (root / "tasks" / "TASKS.md").write_text("# Tasks\n\n", encoding="utf-8")
    return root


def _governed_write(workspace: Path) -> None:
    from mind_mem.enums import IngestTier
    from mind_mem.governance_gate import get_gate
    from mind_mem.storage import get_block_store

    gate = get_gate(str(workspace))
    with gate.admit_block("WRITE", "D-ALIAS-001", "before", tier=IngestTier.EXTERNAL_INGEST):
        get_block_store(str(workspace)).write_block(
            {"_id": "D-ALIAS-001", "Statement": "before", "Status": "quarantined", "Date": "2026-09-09"}
        )


def _archive_with_members(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_ledger_path_aliases_are_one_registry_identity(tmp_path: Path) -> None:
    for spelling in (
        "memory/./hash_chain_v2.db",
        "memory//hash_chain_v2.db",
        "memory/sub/../hash_chain_v2.db",
        "MEMORY\\HASH_CHAIN_V2.DB",
    ):
        assert is_ledger_path(spelling), spelling


def test_archive_alias_is_refused_and_nonledger_member_restores(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    chain = workspace / "memory" / "hash_chain_v2.db"
    before_evidence = evidence.read_bytes()
    before_chain = chain.read_bytes()

    archive = tmp_path / "alias.tar.gz"
    _archive_with_members(
        archive,
        {
            "decisions/DECISIONS.md": b"# Decisions\n\nrestored\n",
            "memory/./evidence_chain.jsonl": b"STALE-EVIDENCE-MARKER\n",
            "memory//hash_chain_v2.db": b"STALE-DB-MARKER\n",
        },
    )

    result = restore_workspace(str(workspace), str(archive), force=True)

    assert "restored" in (workspace / "decisions" / "DECISIONS.md").read_text(encoding="utf-8")
    assert result["refused_ledgers"] == 2
    after_evidence = evidence.read_bytes()
    assert after_evidence.startswith(before_evidence)
    assert b"STALE-EVIDENCE-MARKER" not in after_evidence
    assert b"STALE-DB-MARKER" not in chain.read_bytes()
    assert chain.read_bytes() != before_chain  # the restore admission is recorded, not overwritten


def test_existing_symlink_alias_cannot_overwrite_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    before = evidence.read_bytes()
    alias = workspace / "memory" / "operator-note.jsonl"
    alias.symlink_to("evidence_chain.jsonl")
    assert is_ledger_target(str(workspace), "memory/operator-note.jsonl")

    archive = tmp_path / "symlink-alias.tar.gz"
    _archive_with_members(archive, {"memory/operator-note.jsonl": b"STALE-SYMLINK-MARKER\n"})
    result = restore_workspace(str(workspace), str(archive), force=True)

    assert result["refused_ledgers"] == 1
    assert evidence.read_bytes().startswith(before)
    assert b"STALE-SYMLINK-MARKER" not in evidence.read_bytes()
    assert alias.is_symlink()


def test_backup_excludes_symlink_alias_but_relocates_canonical_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _governed_write(workspace)
    alias = workspace / "memory" / "operator-note.jsonl"
    alias.symlink_to("evidence_chain.jsonl")
    archive = tmp_path / "backup.tar.gz"

    backup_workspace(str(workspace), str(archive))

    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "memory/operator-note.jsonl" not in names
    assert "ledger-archive/memory/evidence_chain.jsonl" in names
    assert "memory/evidence_chain.jsonl" not in names


def test_snapshot_and_manifest_restore_refuse_aliases(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    before = evidence.read_bytes()
    alias = workspace / "memory" / "operator-note.jsonl"
    alias.symlink_to("evidence_chain.jsonl")

    from mind_mem.storage import get_block_store

    snapshot = tmp_path / "snapshot"
    get_block_store(str(workspace)).snapshot(str(snapshot))
    manifest = json.loads((snapshot / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    assert "decisions/DECISIONS.md" in manifest
    assert all("operator-note" not in name for name in manifest)
    assert all(not is_ledger_path(name) for name in manifest)

    # A pre-5.0.2 style manifest can name the symlink alias. The store-level
    # restore has its own destination guard and must refuse it before copy2.
    malicious = tmp_path / "malicious-snapshot"
    (malicious / "memory").mkdir(parents=True)
    (malicious / "memory" / "operator-note.jsonl").write_bytes(b"STALE-SNAPSHOT-MARKER\n")
    (malicious / "MANIFEST.json").write_text(json.dumps({"files": ["memory/./operator-note.jsonl"], "version": 2}), encoding="utf-8")
    restore_snapshot(str(workspace), str(malicious))
    assert evidence.read_bytes().startswith(before)
    assert b"STALE-SNAPSHOT-MARKER" not in evidence.read_bytes()


@pytest.mark.parametrize(
    "canonical_rel",
    ("memory/evidence_chain.jsonl", ".mind-mem-ledger/served.jsonl", "memory/hash_chain_v2.db-wal"),
)
def test_hardlink_aliases_of_each_ledger_family_are_refused(tmp_path: Path, canonical_rel: str) -> None:
    workspace = _workspace(tmp_path / canonical_rel.replace("/", "_").replace(".", "dot"))
    _governed_write(workspace)
    canonical = workspace / canonical_rel
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if not canonical.exists():
        canonical.write_bytes((canonical_rel + " original\n").encode())
    keepalive = None
    if canonical_rel == "memory/hash_chain_v2.db-wal":
        # Keep a real SQLite WAL sidecar live while the restore admission
        # opens its own connection; otherwise SQLite may remove an empty
        # synthetic sidecar before the destination guard observes it.
        keepalive = sqlite3.connect(workspace / "memory" / "hash_chain_v2.db")
        keepalive.execute("PRAGMA journal_mode=WAL")
        keepalive.execute("UPDATE hash_chain SET action = action WHERE rowid = 1")
        keepalive.commit()
    before = canonical.read_bytes()
    alias = workspace / "memory" / ("hardlink-" + canonical.name)
    os.link(canonical, alias)
    assert is_ledger_target(str(workspace), "memory/" + alias.name)

    archive = tmp_path / (canonical.name + ".tar.gz")
    _archive_with_members(archive, {"memory/" + alias.name: b"STALE-HARDLINK-MARKER\n"})
    result = restore_workspace(str(workspace), str(archive), force=True)

    assert result["refused_ledgers"] == 1
    after = canonical.read_bytes()
    if canonical_rel == "memory/evidence_chain.jsonl":
        assert after.startswith(before)
    elif canonical_rel == "memory/hash_chain_v2.db-wal":
        # The live admission may checkpoint/update SQLite's WAL, so byte
        # identity is not a valid assertion for this sidecar. The archive's
        # marker must still be absent and the canonical/alias inode relation
        # must remain intact.
        assert (canonical.stat().st_dev, canonical.stat().st_ino) == (alias.stat().st_dev, alias.stat().st_ino)
    else:
        assert after == before
    assert b"STALE-HARDLINK-MARKER" not in after
    if keepalive is not None:
        keepalive.close()


def test_ordinary_hardlink_remains_a_restorable_nonledger_file(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "ordinary")
    ordinary = workspace / "decisions" / "ordinary.md"
    ordinary.write_bytes(b"ordinary original\n")
    alias = workspace / "memory" / "ordinary-alias.md"
    os.link(ordinary, alias)
    assert not is_ledger_target(str(workspace), "memory/ordinary-alias.md")

    archive = tmp_path / "ordinary.tar.gz"
    _archive_with_members(archive, {"memory/ordinary-alias.md": b"ordinary replacement\n"})
    result = restore_workspace(str(workspace), str(archive), force=True)

    assert result["refused_ledgers"] == 0
    assert result["restored"] == 1
    assert ordinary.read_bytes() == b"ordinary replacement\n"


def test_hardlink_refusal_mutant_reopens_the_ledger_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.evidence_objects import EvidenceChainCompromisedError

    workspace = _workspace(tmp_path / "mutant")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    alias = workspace / "memory" / "hardlink-mutant"
    os.link(evidence, alias)
    archive = tmp_path / "mutant.tar.gz"
    _archive_with_members(archive, {"memory/hardlink-mutant": b"STALE-HARDLINK-MARKER\n"})
    monkeypatch.setattr(backup_restore, "is_ledger_target", lambda _workspace, rel: is_ledger_path(rel))

    with pytest.raises(EvidenceChainCompromisedError):
        restore_workspace(str(workspace), str(archive), force=True)
    assert b"STALE-HARDLINK-MARKER" in evidence.read_bytes()


def test_hardlink_alias_is_excluded_from_capture_and_legacy_carry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "capture")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    alias = workspace / "memory" / "hardlink-capture"
    os.link(evidence, alias)

    backup = tmp_path / "capture.tar.gz"
    backup_workspace(str(workspace), str(backup))
    with tarfile.open(backup) as archive:
        names = archive.getnames()
    assert "memory/hardlink-capture" not in names
    assert "ledger-archive/memory/evidence_chain.jsonl" in names

    snapshot = tmp_path / "snapshot"
    from mind_mem.storage import get_block_store

    get_block_store(str(workspace)).snapshot(str(snapshot))
    manifest = json.loads((snapshot / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    assert "memory/hardlink-capture" not in manifest

    staged = tmp_path / "staged-memory"
    staged.mkdir()
    _carry_ledgers_into(str(workspace), str(workspace / "memory"), str(staged))
    assert (staged / "evidence_chain.jsonl").is_file()
    assert not (staged / "hardlink-capture").exists()


def test_legacy_carry_refuses_unreadable_ledger_before_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace(tmp_path / "unreadable")
    _governed_write(workspace)
    evidence = workspace / "memory" / "evidence_chain.jsonl"
    before = evidence.read_bytes()
    staged = tmp_path / "unreadable-staged"
    staged.mkdir()
    original_stat = os.stat

    def denied_stat(path, *args, **kwargs):
        if os.fspath(path) == str(evidence):
            raise PermissionError("synthetic ledger stat denial")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", denied_stat)
        with pytest.raises(PermissionError, match="synthetic ledger stat denial"):
            _carry_ledgers_into(str(workspace), str(workspace / "memory"), str(staged))
    assert evidence.read_bytes() == before


def test_legacy_carry_preserves_all_canonical_names_sharing_an_inode(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "canonical-links")
    memory = workspace / "memory"
    evidence = memory / "evidence_chain.jsonl"
    evidence.write_bytes(b"retained original bytes\n")
    sidecar = memory / "evidence_chain.jsonl.retained"
    os.link(evidence, sidecar)
    staged = tmp_path / "canonical-staged"
    staged.mkdir()

    _carry_ledgers_into(str(workspace), str(memory), str(staged))

    assert (staged / evidence.name).read_bytes() == evidence.read_bytes()
    assert (staged / sidecar.name).read_bytes() == evidence.read_bytes()
