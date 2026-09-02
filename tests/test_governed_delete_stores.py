# Copyright 2026 STARGA, Inc.
"""DIE is governed: no block leaves a store without a receipt and a record.

The measured 5.0.1 defect, reproduced by live probe before this file
existed: an ungated ``write_block`` raised ``UngatedWriteError``, while an
ungated ``delete_block`` **returned ``True`` and the block was gone** — no
admission check, no evidence record, no chain entry, in all five stores.
Three doors reached it (the ADMIN ``delete_memory_item`` tool, ``DELETE
/memories/{id}``, and ``POST /clear``).

This file is the store half of the fix: one conformance suite over every
``BlockStore`` implementation in the tree, asserting the same four
properties of each.

1. **An ungated delete raises.** With a *positive control in the same
   test*: the block is shown present before the call and still present
   after it, so the refusal cannot be an artefact of a store that never
   had the block or of a method that always raises.
2. **An admitted delete still works.** Without this the suite would pass
   just as well against a ``delete_block`` that was simply broken.
3. **A write receipt cannot be spent on a delete**, and a delete receipt
   cannot be spent on a write. Coverage and operation are separate
   checks and both are enforced.
4. **Probing resistance.** The admission check runs *before* the target
   is resolved, so an id that is not there returns ``False`` inside a
   covering scope while any id with no scope open raises. Existence never
   leaks through the refusal.

Plus a structural gate (:class:`TestEveryStoreIsWired`) that reads the
source rather than the behaviour, so a *sixth* store added later fails
the build naming its own file instead of shipping an ungoverned delete.
It carries its own positive control: a fixture module with the call
removed must be reported, or the scan proves nothing.

Doubles are used for the driver layer only — psycopg's pool, and the
inner stores of the two wrappers. Every double that stands in for a
``BlockStore`` implements the *whole* contract (check first, then
``record_removal`` on success), because a wrapper test whose inner store
did not would be measuring the double instead of the wrapper.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from mind_mem.admission import (
    UngatedDeleteError,
    UngatedWriteError,
    require_admission,
    require_delete_admission,
)
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.block_store_encrypted import EncryptedBlockStore, encrypt_workspace
from mind_mem.enums import IngestTier
from mind_mem.governance_gate import PHASE_ADMITTED, PHASE_REMOVED, evict_gate, get_gate
from mind_mem.storage.sharded_pg import ShardConfig, ShardedPostgresBlockStore, ShardRouter

SRC = Path(__file__).resolve().parents[1] / "src" / "mind_mem"

_PASS = "governed-delete-conformance-passphrase"

#: Id the Markdown backend routes into ``decisions/DECISIONS.md``.
SEED_ID = "D-20260901-001"

#: A second id, for the "covered but absent" probe.
ABSENT_ID = "D-20260901-404"


# ---------------------------------------------------------------------------
# Evidence-chain helpers
# ---------------------------------------------------------------------------


def _records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase(ws: str, phase: str) -> list[dict]:
    return [r for r in _records(ws) if r.get("metadata", {}).get("delete_phase") == phase]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Conforming doubles for the layers a unit test cannot reach
# ---------------------------------------------------------------------------


class _ConformingStore:
    """An in-memory ``BlockStore`` that obeys the delete contract.

    Stands in for the *inner* store of a wrapper (replica, shard). It
    checks admission before resolving the target and reports the removed
    text back, exactly as ``MarkdownBlockStore`` and
    ``PostgresBlockStore`` do — so a wrapper test measures the wrapper,
    not a permissive fake.
    """

    def __init__(self, name: str = "inner") -> None:
        self.name = name
        self.rows: dict[str, str] = {}
        #: Every id this store was *asked* about, so a wrapper test can
        #: prove the inner store was never contacted at all.
        self.attempts: list[str] = []

    # -- write surface (enough for the tests that seed through it) -----
    def write_block(self, block: dict[str, Any], **_: Any) -> str:
        bid = str(block.get("_id") or "")
        require_admission(bid, status=block.get("Status"))
        self.rows[bid] = json.dumps(block, sort_keys=True)
        return bid

    def delete_block(self, block_id: str, **_: Any) -> bool:
        self.attempts.append(str(block_id))
        receipt = require_delete_admission(str(block_id))
        removed = self.rows.pop(str(block_id), None)
        if removed is None:
            return False
        receipt.record_removal(str(block_id), removed)
        return True

    # -- read surface --------------------------------------------------
    def get_by_id(self, block_id: str) -> dict[str, Any] | None:
        raw = self.rows.get(str(block_id))
        return json.loads(raw) if raw is not None else None

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [json.loads(v) for v in self.rows.values()]

    def list_blocks(self) -> list[str]:
        return sorted(self.rows)

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return []


class _FakeCursor:
    """The two statements ``PostgresBlockStore.delete_block`` issues."""

    def __init__(self, rows: dict[str, str], journal: list[tuple[str, str]]) -> None:
        self._rows = rows
        self._journal = journal
        self._last: tuple[str, str] | None = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def execute(self, sql: Any, params: tuple = ()) -> None:
        # psycopg composes SQL into a ``Composed`` object, not a str; its
        # repr carries the literal fragments, which is all this double
        # needs to tell the two statements apart.
        sql = sql if isinstance(sql, str) else str(sql)
        if "DELETE FROM" in sql:
            bid = str(params[0])
            self._last = (bid, self._rows.pop(bid)) if bid in self._rows else None
        elif "INSERT INTO" in sql:
            self._journal.append((str(params[0]), str(params[1])))
            self._last = None
        else:  # pragma: no cover - the delete path issues nothing else
            raise AssertionError(f"unexpected SQL in the delete path: {sql!r}")

    def fetchone(self) -> tuple[str, str] | None:
        return self._last


class _FakeConnection:
    def __init__(self, rows: dict[str, str], journal: list[tuple[str, str]]) -> None:
        self._rows = rows
        self._journal = journal

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows, self._journal)


class _FakePool:
    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.journal: list[tuple[str, str]] = []

    def connection(self) -> _FakeConnection:
        return _FakeConnection(self.rows, self.journal)


# ---------------------------------------------------------------------------
# The five cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One store under test, with the three operations the suite needs."""

    name: str
    store: Any
    workspace: str
    seed: Callable[[str, str], None]
    present: Callable[[str], bool]
    #: Whether this store writes the removal into the chain itself. The
    #: two wrappers deliberately do not — they check and delegate, so the
    #: inner store writes exactly one record however the delete arrived.
    records_removal: bool = True
    #: Content this case actually stored, keyed by id, for the hash check.
    stored: dict[str, str] = field(default_factory=dict)


def _fresh_workspace(tmp_path: Path, name: str) -> str:
    ws = tmp_path / name
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    return str(ws)


def _markdown_case(tmp_path: Path) -> Case:
    ws = _fresh_workspace(tmp_path, "markdown")
    store = MarkdownBlockStore(ws)
    case = Case(name="markdown", store=store, workspace=ws, seed=lambda *_: None, present=lambda _b: False)

    def seed(bid: str, statement: str) -> None:
        text = f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: active\n\n---\n\n"
        path = os.path.join(ws, "decisions", "DECISIONS.md")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
        store.invalidate_cache()
        case.stored[bid] = statement

    case.seed = seed
    case.present = lambda bid: store.get_by_id(bid) is not None
    return case


def _encrypted_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    ws = _fresh_workspace(tmp_path, "encrypted")
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", _PASS)
    inner = MarkdownBlockStore(ws)
    case = Case(name="encrypted", store=None, workspace=ws, seed=lambda *_: None, present=lambda _b: False)

    def seed(bid: str, statement: str) -> None:
        """Append while the corpus is still plaintext; ``_seal`` locks it after."""
        text = f"[{bid}]\nStatement: {statement}\nDate: 2026-09-01\nStatus: active\n\n---\n\n"
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
            handle.write(text)
        inner.invalidate_cache()
        case.stored[bid] = statement

    def _seal() -> None:
        """Encrypt the seeded corpus, then hand the case the real wrapper.

        The store under test is therefore a wrapper over ciphertext on
        disk — the arrangement that makes the ordering matter: the
        wrapper unseals the file for the duration of the operation, so a
        check that ran after the unseal would let an ungated caller put
        plaintext on disk before being refused.
        """
        assert encrypt_workspace(ws)["encrypted"] >= 1
        sealed_store = EncryptedBlockStore(ws, passphrase=_PASS)
        case.store = sealed_store
        case.present = lambda bid: sealed_store.get_by_id(bid) is not None

    case.seed = seed
    case.store = inner  # replaced by _seal()
    setattr(case, "_seal", _seal)
    return case


def _postgres_case(tmp_path: Path) -> Case:
    from mind_mem.block_store_postgres import PostgresBlockStore

    ws = _fresh_workspace(tmp_path, "postgres")
    store = PostgresBlockStore(dsn="postgresql://unused", schema="mm_conformance", workspace=ws)
    pool = _FakePool()
    # Drive the real ``delete_block`` body against a stand-in driver: the
    # code under test is the admission check and the removal report, not
    # psycopg. Everything below the pool is the double; everything above
    # it is production code.
    store._schema_ready = True  # type: ignore[attr-defined]
    store._pool = pool  # type: ignore[attr-defined]
    case = Case(name="postgres", store=store, workspace=ws, seed=lambda *_: None, present=lambda _b: False)

    def seed(bid: str, statement: str) -> None:
        pool.rows[bid] = statement
        case.stored[bid] = statement

    case.seed = seed
    case.present = lambda bid: bid in pool.rows
    return case


def _replica_case(tmp_path: Path) -> Case:
    from unittest.mock import patch

    ws = _fresh_workspace(tmp_path, "replica")
    inner = _ConformingStore("primary")
    with patch("mind_mem.block_store_postgres_replica.PostgresBlockStore") as mock_cls:
        mock_cls.side_effect = lambda dsn, **_: inner
        from mind_mem.block_store_postgres_replica import ReplicatedPostgresBlockStore

        store = ReplicatedPostgresBlockStore(primary_dsn="pg://primary", replica_dsns=["pg://replica"])
    case = Case(name="replica", store=store, workspace=ws, seed=lambda *_: None, present=lambda _b: False)

    def seed(bid: str, statement: str) -> None:
        inner.rows[bid] = statement
        case.stored[bid] = statement

    case.seed = seed
    case.present = lambda bid: bid in inner.rows
    return case


def _sharded_case(tmp_path: Path) -> Case:
    ws = _fresh_workspace(tmp_path, "sharded")
    shards = [ShardConfig(index=i, dsn=f"pg://shard{i}") for i in range(3)]
    router = ShardRouter(shards=shards)
    stores = {i: _ConformingStore(f"shard{i}") for i in range(3)}
    store = ShardedPostgresBlockStore(router, stores)
    case = Case(name="sharded", store=store, workspace=ws, seed=lambda *_: None, present=lambda _b: False)

    def seed(bid: str, statement: str) -> None:
        # Land it on a shard the fan-out has to look for.
        stores[len(stores) - 1].rows[bid] = statement
        case.stored[bid] = statement

    case.seed = seed
    case.present = lambda bid: any(bid in s.rows for s in stores.values())
    return case


CASE_NAMES = ["markdown", "encrypted", "postgres", "replica", "sharded"]


@pytest.fixture(params=CASE_NAMES)
def case(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Case]:
    """One store per parameter, seeded with :data:`SEED_ID`."""
    name = request.param
    if name == "markdown":
        built = _markdown_case(tmp_path)
        built.seed(SEED_ID, "the block under test")
    elif name == "encrypted":
        built = _encrypted_case(tmp_path, monkeypatch)
        built.seed(SEED_ID, "the block under test")
        getattr(built, "_seal")()
    elif name == "postgres":
        built = _postgres_case(tmp_path)
        built.seed(SEED_ID, "the block under test")
        built.records_removal = True
    elif name == "replica":
        built = _replica_case(tmp_path)
        built.seed(SEED_ID, "the block under test")
        built.records_removal = True
    else:
        built = _sharded_case(tmp_path)
        built.seed(SEED_ID, "the block under test")
        built.records_removal = True
    try:
        yield built
    finally:
        evict_gate(built.workspace)


# ===========================================================================
# 1 — an ungated delete raises, and the block survives it
# ===========================================================================


def test_an_ungated_delete_raises_and_the_block_survives(case: Case) -> None:
    """The measured defect, per store.

    Positive control on both sides of the refusal: the block is proven
    present *before* the ungated call and proven present *after* it. A
    test that only asserted the raise would pass against a store that
    never held the block, and one that only asserted absence-of-deletion
    would pass against a store whose delete never worked at all.
    """
    assert case.present(SEED_ID), f"{case.name}: fixture never seeded the block; the rest of this test would be vacuous"

    with pytest.raises(UngatedDeleteError) as excinfo:
        case.store.delete_block(SEED_ID)
    assert "no governance admission is open" in str(excinfo.value)

    assert case.present(SEED_ID), f"{case.name}: the refused delete removed the block anyway"


def test_the_refusal_is_catchable_as_the_write_error(case: Case) -> None:
    """A handler written before deletes were gated still catches this."""
    with pytest.raises(UngatedWriteError):
        case.store.delete_block(SEED_ID)


# ===========================================================================
# 2 — an admitted delete still works (the positive control for all of §1)
# ===========================================================================


def test_an_admitted_delete_removes_the_block(case: Case) -> None:
    assert case.present(SEED_ID), f"{case.name}: fixture never seeded the block"

    gate = get_gate(case.workspace)
    with gate.admit_delete(SEED_ID, rationale="conformance suite removal", actor="pytest"):
        assert case.store.delete_block(SEED_ID) is True

    assert not case.present(SEED_ID), f"{case.name}: the admitted delete reported success without removing anything"


# ===========================================================================
# 3 — the receipt is not transferable between operations
# ===========================================================================


def test_a_write_receipt_does_not_authorize_a_delete(case: Case) -> None:
    gate = get_gate(case.workspace)
    with gate.admit_block("WRITE", SEED_ID, "body", tier=IngestTier.EXTERNAL_INGEST):
        # Positive control: the receipt is real and does authorise its write.
        assert require_admission(SEED_ID, status="quarantined") is not None
        with pytest.raises(UngatedDeleteError) as excinfo:
            case.store.delete_block(SEED_ID)
    assert "authorises a write, not a delete" in str(excinfo.value)
    assert case.present(SEED_ID)


def test_a_scope_for_another_block_does_not_authorize_this_one(case: Case) -> None:
    """Coverage is checked as well as operation."""
    gate = get_gate(case.workspace)
    with gate.admit_delete(ABSENT_ID, rationale="conformance suite removal"):
        # Positive control: the scope authorises the id it names.
        assert require_delete_admission(ABSENT_ID) is not None
        with pytest.raises(UngatedDeleteError):
            case.store.delete_block(SEED_ID)
    assert case.present(SEED_ID)


# ===========================================================================
# 4 — probing resistance: authorisation decides, existence does not
# ===========================================================================


def test_a_covered_but_absent_id_returns_false_rather_than_raising(case: Case) -> None:
    """Inside a covering scope a missing id is a miss, not a refusal.

    Paired with :func:`test_an_ungated_delete_raises_and_the_block_survives`
    this is the whole probing-resistance property: the two outcomes are
    told apart by whether the caller was authorised, never by whether the
    block was there.
    """
    assert not case.present(ABSENT_ID)
    gate = get_gate(case.workspace)
    with gate.admit_delete(ABSENT_ID, rationale="conformance suite removal"):
        assert case.store.delete_block(ABSENT_ID) is False


def test_an_ungated_delete_of_an_absent_id_fails_the_same_way(case: Case) -> None:
    """…and with no scope open, a missing id raises exactly like a present one."""
    assert not case.present(ABSENT_ID)
    with pytest.raises(UngatedDeleteError):
        case.store.delete_block(ABSENT_ID)


# ===========================================================================
# 5 — the chain record a delete leaves
# ===========================================================================


def test_the_delete_leaves_an_authorisation_and_a_removal_record(case: Case) -> None:
    """Two records per scope, naming the operation, the actor and the target."""
    gate = get_gate(case.workspace)
    with gate.admit_delete(SEED_ID, rationale="conformance suite removal", actor="alice") as receipt:
        entry_id = receipt.entry_id
        assert case.store.delete_block(SEED_ID) is True

    admitted = _phase(case.workspace, PHASE_ADMITTED)
    assert len(admitted) == 1, f"{case.name}: expected exactly one authorisation record"
    assert admitted[0]["actor"] == "alice"
    assert admitted[0]["target_block_id"] == SEED_ID
    assert admitted[0]["metadata"]["operation"] == "delete"
    assert admitted[0]["metadata"]["rationale"] == "conformance suite removal"

    removed = _phase(case.workspace, PHASE_REMOVED)
    assert len(removed) == 1, f"{case.name}: a completed delete must leave exactly one removal record"
    assert removed[0]["metadata"]["admission_entry_id"] == entry_id
    assert removed[0]["metadata"]["removed_count"] == 1
    assert removed[0]["metadata"]["scope_outcome"] == "ok"
    assert removed[0]["actor"] == "alice"


def test_a_refused_delete_leaves_no_record_at_all(case: Case) -> None:
    """Nothing was authorised and nothing died, so the chain stays silent."""
    with pytest.raises(UngatedDeleteError):
        case.store.delete_block(SEED_ID)
    assert _records(case.workspace) == []


def test_the_removal_record_hashes_what_was_actually_removed(tmp_path: Path) -> None:
    """The Markdown store's record covers the exact text it took out.

    Cross-checked against the recovery journal rather than a string this
    test rebuilt: the journal is written by the store from the same
    ``removed`` slice, so agreeing with it proves the hash covers the
    real removal and not a re-render that happens to match.
    """
    case = _markdown_case(tmp_path)
    case.seed(SEED_ID, "content whose hash must reach the chain")
    try:
        gate = get_gate(case.workspace)
        with gate.admit_delete(SEED_ID, rationale="conformance suite removal"):
            assert case.store.delete_block(SEED_ID) is True

        journal_path = os.path.join(case.workspace, "memory", "deleted_blocks.jsonl")
        entries = [json.loads(line) for line in Path(journal_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(entries) == 1
        assert "content whose hash must reach the chain" in entries[0]["content"]

        removed = _phase(case.workspace, PHASE_REMOVED)
        assert len(removed) == 1
        assert removed[0]["payload_hash"] == _sha256(entries[0]["content"])
    finally:
        evict_gate(case.workspace)


def test_the_encrypted_wrapper_refuses_before_it_unseals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ungated delete must not unseal the corpus on its way to "no".

    ``EncryptedBlockStore.delete_block`` writes the corpus file back out
    decrypted for the duration of the operation. Leaving the refusal to
    the inner store would unseal first and refuse second, putting the
    plaintext on disk for the length of the call — a window that a
    second process, a backup, or a crash can see. The check therefore
    runs in the wrapper, before ``_decrypted_target``.

    The *unseal itself* is what this test observes, not the file
    afterwards: ``_decrypted_target`` re-seals on the way out even when
    the body raises, so a post-hoc file check would pass whether or not
    the plaintext was ever written. Positive control: the same spy shows
    the unseal happening for the admitted delete two lines later.
    """
    case = _encrypted_case(tmp_path, monkeypatch)
    case.seed(SEED_ID, "plaintext that must not reach disk through a refusal")
    getattr(case, "_seal")()
    store = case.store

    unseals: list[str] = []
    real_target = store._decrypted_target

    @contextlib.contextmanager
    def spying_target(block_id: str) -> Iterator[None]:
        unseals.append(str(block_id))
        with real_target(block_id):
            yield

    monkeypatch.setattr(store, "_decrypted_target", spying_target)

    try:
        with pytest.raises(UngatedDeleteError):
            store.delete_block(SEED_ID)
        assert unseals == [], "the refused delete unsealed the corpus before refusing"

        gate = get_gate(case.workspace)
        with gate.admit_delete(SEED_ID, rationale="conformance suite removal"):
            assert store.delete_block(SEED_ID) is True
        assert unseals == [SEED_ID], "positive control: an admitted delete must still unseal"
    finally:
        evict_gate(case.workspace)


def test_the_replica_refuses_before_it_reaches_the_primary(tmp_path: Path) -> None:
    """The adapter checks first; the primary is never contacted.

    Enforced on the adapter as well as the primary because a caller
    holding only the replica must not get a laxer mutation surface than
    one holding the primary — and because a refusal that depends on a
    round-trip is a refusal that costs one. Positive control: the same
    counter shows the primary being reached for the admitted delete.
    """
    case = _replica_case(tmp_path)
    case.seed(SEED_ID, "the block behind the adapter")
    inner = case.store._primary
    try:
        with pytest.raises(UngatedDeleteError):
            case.store.delete_block(SEED_ID)
        assert inner.attempts == [], "the refused delete was forwarded to the primary anyway"

        gate = get_gate(case.workspace)
        with gate.admit_delete(SEED_ID, rationale="conformance suite removal"):
            assert case.store.delete_block(SEED_ID) is True
        assert inner.attempts == [SEED_ID], "positive control: an admitted delete must reach the primary"
    finally:
        evict_gate(case.workspace)


def test_the_sharded_wrapper_refuses_before_the_fan_out(tmp_path: Path) -> None:
    """No shard is contacted by an ungated delete.

    Without a namespace the delete fans out across every shard until one
    reports a removal, so leaving the check to the shards would put an
    unauthorised probe on each of them in turn. Checking before the loop
    also means a router that resolves to no shard refuses rather than
    returning a quiet ``False``.
    """
    case = _sharded_case(tmp_path)
    case.seed(SEED_ID, "the block on the last shard")
    shards = list(case.store._stores.values())
    try:
        with pytest.raises(UngatedDeleteError):
            case.store.delete_block(SEED_ID, tenant_id="acme")
        assert [a for s in shards for a in s.attempts] == [], "the refused delete probed the shards anyway"

        gate = get_gate(case.workspace)
        with gate.admit_delete(SEED_ID, rationale="conformance suite removal"):
            assert case.store.delete_block(SEED_ID, tenant_id="acme") is True
        assert [a for s in shards for a in s.attempts], "positive control: an admitted delete must reach a shard"
    finally:
        evict_gate(case.workspace)


# ===========================================================================
# 6 — the structural gate: a sixth store cannot ship ungoverned
# ===========================================================================


def _delete_impls(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "delete_block"]


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_ellipsis(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis


def _is_declaration_only(fn: ast.FunctionDef) -> bool:
    """True for a Protocol/ABC stub: a docstring and an ellipsis, nothing else."""
    body = [stmt for stmt in fn.body if not _is_docstring(stmt)]
    return all(isinstance(stmt, ast.Pass) or _is_ellipsis(stmt) for stmt in body)


def _guards_first(fn: ast.FunctionDef) -> bool:
    """True when the first executable statement calls ``require_delete_admission``."""
    for stmt in fn.body:
        if _is_docstring(stmt):
            continue
        call = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        return bool(call is not None and isinstance(call.func, ast.Name) and call.func.id == "require_delete_admission")
    return False


def _ungoverned_deletes(root: Path) -> list[str]:
    """Every ``delete_block`` in *root* that does not check admission first."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken source file fails elsewhere
            continue
        for fn in _delete_impls(tree):
            if _is_declaration_only(fn):
                continue
            if not _guards_first(fn):
                offenders.append(f"{path.relative_to(root)}:{fn.lineno}")
    return offenders


class TestEveryStoreIsWired:
    def test_no_delete_block_in_src_skips_admission(self) -> None:
        """The build-time gate. A new store fails here, naming its own file."""
        offenders = _ungoverned_deletes(SRC)
        assert offenders == [], (
            "these delete_block implementations do not call require_delete_admission "
            f"as their first statement: {offenders}. A delete that skips the check "
            "removes content with no receipt and no chain record."
        )

    def test_the_scan_actually_finds_the_five_stores(self) -> None:
        """A scan that matched nothing would pass the test above vacuously."""
        found = {
            str(path.relative_to(SRC))
            for path in sorted(SRC.rglob("*.py"))
            for fn in _delete_impls(ast.parse(path.read_text(encoding="utf-8")))
            if not _is_declaration_only(fn)
        }
        expected = {
            "block_store.py",
            "block_store_encrypted.py",
            "block_store_postgres.py",
            "block_store_postgres_replica.py",
            os.path.join("storage", "sharded_pg.py"),
        }
        assert expected <= found, f"the delete scan lost sight of {sorted(expected - found)}"

    def test_the_scan_reports_an_unguarded_implementation(self, tmp_path: Path) -> None:
        """Test-of-the-test: the checker must be able to see a violation.

        Without this, a scan broken into always returning ``[]`` would
        read as a clean bill of health forever.
        """
        fake = tmp_path / "pkg"
        fake.mkdir()
        (fake / "rogue_store.py").write_text(
            "class RogueStore:\n"
            "    def delete_block(self, block_id: str) -> bool:\n"
            '        """Removes a block with no admission check."""\n'
            "        self.rows.pop(block_id, None)\n"
            "        return True\n",
            encoding="utf-8",
        )
        (fake / "good_store.py").write_text(
            "from mind_mem.admission import require_delete_admission\n\n\n"
            "class GoodStore:\n"
            "    def delete_block(self, block_id: str) -> bool:\n"
            "        receipt = require_delete_admission(block_id)\n"
            "        removed = self.rows.pop(block_id, None)\n"
            "        if removed is None:\n"
            "            return False\n"
            "        receipt.record_removal(block_id, removed)\n"
            "        return True\n",
            encoding="utf-8",
        )
        offenders = _ungoverned_deletes(fake)
        assert [o.split(":")[0] for o in offenders] == ["rogue_store.py"]
