# Copyright 2026 STARGA, Inc.
"""``EncryptedBlockStore`` must be a whole BlockStore, not the read half.

The wrapper documented itself as exposing "the same public surface as
MarkdownBlockStore" and both factories ``cast(BlockStore, ...)`` to assert
it, but it implemented only ``get_all`` / ``get_by_id`` / ``search`` /
``list_blocks``. With ``block_store.backend = "encrypted"`` the apply
engine resolves this object and calls ``snapshot()`` before it applies a
proposal, so the first governed write against an encrypted workspace died
on ``AttributeError`` — and a later ``write_block`` would have escaped
``execute_op``'s ``(OSError, ValueError, KeyError, IndexError)`` handler
mid-transaction, past the rollback.

The other half of the fix is that the missing methods cannot simply
forward: the inner store's read-modify-write reads the target as UTF-8
text, which ciphertext is not. ``test_unsealing_is_what_makes_the_write_work``
is the mechanism probe — it disables the unseal and shows the delegated
write failing, so nothing here can be passing for some other reason.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem.admission import UngatedWriteError
from mind_mem.block_store import BlockStore, BlockStoreError, MarkdownBlockStore
from mind_mem.block_store_encrypted import EncryptedBlockStore, encrypt_workspace
from mind_mem.encryption import _MAGIC
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import get_gate
from mind_mem.mind_filelock import LockTimeout

_PASS = "test-passphrase-for-unit-tests"

#: Every method the ``BlockStore`` Protocol declares, plus the file-list
#: cache hook the Markdown backend exposes and ``write_block`` calls.
_REQUIRED_SURFACE = (
    "get_all",
    "get_by_id",
    "search",
    "list_blocks",
    "write_block",
    "delete_block",
    "snapshot",
    "restore",
    "diff",
    "lock",
    "invalidate_cache",
)


def _block(bid: str, rationale: str = "seed") -> str:
    return f"[{bid}]\nDate: 2026-04-13\nStatus: active\nRationale: {rationale}\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace whose DECISIONS.md is real ciphertext on disk."""
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASS)
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(_block("D-20260410-001"), encoding="utf-8")
    assert encrypt_workspace(str(tmp_path))["encrypted"] == 1
    assert _head(tmp_path / "decisions" / "DECISIONS.md") == _MAGIC
    return tmp_path


@pytest.fixture
def store(workspace: Path) -> EncryptedBlockStore:
    return EncryptedBlockStore(str(workspace), passphrase=_PASS)


def _head(path: Path) -> bytes:
    return path.read_bytes()[: len(_MAGIC)]


def _decisions(workspace: Path) -> Path:
    return workspace / "decisions" / "DECISIONS.md"


@contextlib.contextmanager
def _admitted(workspace: Path, proposal_id: str = "P-TEST") -> Iterator[None]:
    """A real proposal-apply admission, exactly as the apply engine opens one."""
    with get_gate(str(workspace)).admit_proposal(proposal_id=proposal_id, content="[]", actor="pytest"):
        yield


def _write_message(workspace: Path, store: EncryptedBlockStore) -> None:
    """Write one MSG block — its target file (memory/MESSAGES.md) starts absent."""
    with get_gate(str(workspace)).admit_block(action="WRITE", block_id="MSG-20260410-001", content="hello", tier=IngestTier.AGENT_MESSAGE):
        store.write_block({"_id": "MSG-20260410-001", "Date": "2026-04-13", "Status": "quarantined", "Rationale": "hello"})


# ---------------------------------------------------------------------------
# The surface itself
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_implements_every_block_store_method(self, store: EncryptedBlockStore) -> None:
        """The claim the factory's ``cast(BlockStore, ...)`` already makes."""
        missing = [name for name in _REQUIRED_SURFACE if not callable(getattr(store, name, None))]
        assert missing == [], f"EncryptedBlockStore is cast to BlockStore but is missing {missing}"

    def test_satisfies_the_runtime_checkable_protocol(self, store: EncryptedBlockStore) -> None:
        assert isinstance(store, BlockStore)

    def test_apply_engine_can_snapshot_an_encrypted_workspace(self, workspace: Path) -> None:
        """The exact reported failure: ``create_snapshot`` -> ``AttributeError``."""
        (workspace / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": "encrypted"}}), encoding="utf-8")
        from mind_mem.apply_engine import _store_for, create_snapshot

        assert isinstance(_store_for(str(workspace)), EncryptedBlockStore)
        snap_dir = create_snapshot(str(workspace), "20260420-000000")
        assert os.path.isfile(os.path.join(snap_dir, "MANIFEST.json"))


# ---------------------------------------------------------------------------
# Write surface — the unseal is load-bearing
# ---------------------------------------------------------------------------


class TestWriteSurface:
    def test_write_block_keeps_the_file_encrypted_and_readable(self, workspace: Path, store: EncryptedBlockStore) -> None:
        with _admitted(workspace):
            written = store.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "second"})
        assert written == "D-20260410-002"
        assert _head(_decisions(workspace)) == _MAGIC, "write left the corpus file in plaintext"
        assert sorted(b["_id"] for b in store.get_all()) == ["D-20260410-001", "D-20260410-002"]

    def test_unsealing_is_what_makes_the_write_work(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """Mechanism probe: with the unseal disabled the delegated write fails.

        A passing ``test_write_block_keeps_the_file_encrypted_and_readable``
        would prove nothing if the inner store could write through
        ciphertext on its own. It cannot — it reads the target with a
        strict UTF-8 decode — and this pins that, so the first test can
        only be passing because of ``_decrypted_target``.
        """

        @contextlib.contextmanager
        def _no_unseal(_block_id: str) -> Iterator[None]:
            yield

        before = _decisions(workspace).read_bytes()
        object.__setattr__(store, "_decrypted_target", _no_unseal)
        with _admitted(workspace), pytest.raises(UnicodeDecodeError):
            store.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "second"})
        assert _decisions(workspace).read_bytes() == before

    def test_file_is_resealed_when_the_inner_write_raises(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """A failed write must not leave the corpus decrypted on disk."""
        inner = MarkdownBlockStore(str(workspace))

        def _boom(_block: dict[str, Any]) -> str:
            raise RuntimeError("inner store exploded")

        object.__setattr__(inner, "write_block", _boom)
        wrapper = EncryptedBlockStore(str(workspace), passphrase=_PASS, inner=inner)
        with _admitted(workspace), pytest.raises(RuntimeError, match="exploded"):
            wrapper.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "x"})
        assert _head(_decisions(workspace)) == _MAGIC

    def test_ungated_write_is_refused_before_the_file_is_unsealed(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """No admission open → refused, and the ciphertext is never opened."""
        with pytest.raises(UngatedWriteError):
            store.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "x"})
        assert _head(_decisions(workspace)) == _MAGIC

    def test_write_block_without_an_id_is_rejected(self, store: EncryptedBlockStore) -> None:
        with pytest.raises(ValueError, match="_id"):
            store.write_block({"Status": "active"})

    def test_delete_block_keeps_the_file_encrypted(self, workspace: Path, store: EncryptedBlockStore) -> None:
        with _admitted(workspace):
            store.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "second"})
        assert store.delete_block("D-20260410-002") is True
        assert _head(_decisions(workspace)) == _MAGIC
        assert [b["_id"] for b in store.get_all()] == ["D-20260410-001"]

    def test_a_file_this_store_creates_is_born_encrypted(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """memory/MESSAGES.md does not exist yet — in a sealed corpus it is born sealed."""
        messages = workspace / "memory" / "MESSAGES.md"
        assert not messages.exists()
        _write_message(workspace, store)
        assert _head(messages) == _MAGIC

    def test_a_new_file_stays_plaintext_in_an_unmigrated_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Writing a block must not be the thing that seals a workspace.

        Configured for encryption but never migrated: the corpus is all
        plaintext, so a reader that opens these files directly still
        works. A file created here must not silently become the one
        ciphertext file that reader goes quiet on.
        """
        monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASS)
        (tmp_path / "decisions").mkdir()
        (tmp_path / "decisions" / "DECISIONS.md").write_text(_block("D-20260410-001"), encoding="utf-8")
        store = EncryptedBlockStore(str(tmp_path), passphrase=_PASS)

        _write_message(tmp_path, store)
        assert _head(tmp_path / "memory" / "MESSAGES.md") != _MAGIC
        assert _head(tmp_path / "decisions" / "DECISIONS.md") != _MAGIC

    def test_a_plaintext_file_stays_plaintext(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """Rollout rule: a file the operator has not migrated is not migrated by a write."""
        tasks = workspace / "tasks" / "TASKS.md"
        tasks.parent.mkdir()
        tasks.write_text(_block("T-20260410-001"), encoding="utf-8")
        store.invalidate_cache()
        with _admitted(workspace):
            store.write_block({"_id": "T-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "t2"})
        assert _head(tasks) != _MAGIC
        assert sorted(b["_id"] for b in store.get_all() if b["_id"].startswith("T-")) == ["T-20260410-001", "T-20260410-002"]


# ---------------------------------------------------------------------------
# Snapshot / lock surface
# ---------------------------------------------------------------------------


class TestSnapshotAndLockSurface:
    def test_snapshot_restore_round_trips_ciphertext(self, workspace: Path, store: EncryptedBlockStore) -> None:
        snap_dir = str(workspace / "intelligence" / "applied" / "snap")
        manifest = store.snapshot(snap_dir)
        assert "decisions/DECISIONS.md" in manifest["files"]
        assert _head(Path(snap_dir) / "decisions" / "DECISIONS.md") == _MAGIC, "snapshot stored the corpus in plaintext"

        with _admitted(workspace):
            store.write_block({"_id": "D-20260410-002", "Date": "2026-04-13", "Status": "active", "Rationale": "second"})
        assert store.diff(snap_dir) == ["decisions/DECISIONS.md"]

        store.restore(snap_dir)
        store.invalidate_cache()
        assert [b["_id"] for b in store.get_all()] == ["D-20260410-001"]
        assert _head(_decisions(workspace)) == _MAGIC

    def test_diff_is_empty_against_an_untouched_workspace(self, workspace: Path, store: EncryptedBlockStore) -> None:
        snap_dir = str(workspace / "intelligence" / "applied" / "snap")
        store.snapshot(snap_dir)
        assert store.diff(snap_dir) == []

    def test_lock_is_exclusive(self, store: EncryptedBlockStore) -> None:
        with store.lock(timeout=5):
            with pytest.raises(LockTimeout):
                store.lock(blocking=False).acquire()

    def test_invalidate_cache_reaches_the_inner_store(self, workspace: Path, store: EncryptedBlockStore) -> None:
        before = store.list_blocks()
        (workspace / "tasks").mkdir()
        (workspace / "tasks" / "TASKS.md").write_text(_block("T-20260410-001"), encoding="utf-8")
        assert store.list_blocks() == before, "precondition: the inner store caches its file list"
        store.invalidate_cache()
        assert len(store.list_blocks()) == len(before) + 1


# ---------------------------------------------------------------------------
# Read errors are errors, not an empty corpus
# ---------------------------------------------------------------------------


class TestUnreadableIsNotEmpty:
    def test_wrong_passphrase_raises_instead_of_reporting_no_blocks(self, workspace: Path) -> None:
        """A rotated or mistyped passphrase must not look like an empty workspace."""
        wrong = EncryptedBlockStore(str(workspace), passphrase="a-completely-different-passphrase")
        with pytest.raises(BlockStoreError) as excinfo:
            wrong.get_all()
        assert "MIND_MEM_ENCRYPTION_PASSPHRASE" in str(excinfo.value)

    def test_the_error_is_not_a_valueerror(self, workspace: Path) -> None:
        """The corpus walk skips one unreadable file on ValueError/OSError.

        A passphrase failure is not one bad file, it is the whole
        workspace, so it must not be catchable by that per-file skip.
        """
        wrong = EncryptedBlockStore(str(workspace), passphrase="a-completely-different-passphrase")
        with pytest.raises(BlockStoreError):
            try:
                wrong.get_all()
            except (OSError, UnicodeDecodeError, ValueError) as exc:  # pragma: no cover - the point is that this never fires
                pytest.fail(f"the per-file skip would swallow this: {exc!r}")

    def test_get_by_id_also_raises(self, workspace: Path) -> None:
        wrong = EncryptedBlockStore(str(workspace), passphrase="a-completely-different-passphrase")
        with pytest.raises(BlockStoreError):
            wrong.get_by_id("D-20260410-001")

    def test_an_unopenable_corpus_entry_is_not_reported_as_empty(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """``list_blocks`` lists any ``*.md`` name; one that cannot be opened is a fault."""
        (workspace / "tasks").mkdir()
        (workspace / "tasks" / "TASKS.md").mkdir()  # a directory wearing a corpus file's name
        store.invalidate_cache()
        with pytest.raises(OSError):
            store.get_all()

    def test_a_file_deleted_mid_walk_is_skipped(self, workspace: Path, store: EncryptedBlockStore) -> None:
        """The one swallow that is correct: the file is genuinely gone."""
        assert store._parse_maybe_encrypted(str(workspace / "decisions" / "MISSING.md")) == []
