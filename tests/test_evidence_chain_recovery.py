# Copyright 2026 STARGA, Inc.
"""A forked evidence chain gets a way back that forges nothing.

``EvidenceChain`` refuses to append to a store whose history did not load
intact, and that refusal is correct: appending would root a second chain
at the genesis hash behind the untrusted tail. But refusal alone leaves a
workspace with no governed writes at all, and the damage in the field is
already done — a store carrying 606 records loaded zero of them because
an older release, one with no such refusal, restarted the chain at
genesis on every append it made after the first break.

The way back is not repair. ``_freeze_and_raise`` states the constraint
outright — "repairing the history by rewriting hashes is never this
code's decision" — so recovery seals instead: it archives the damaged
file byte-for-byte, and starts a new chain whose first record is an
anchor naming that archive and its sha256. The archive still fails to
verify, forever, and the new segment verifies from its own genesis.

These tests pin every part of that: the census is honest about which
breaks are genesis restarts and which are forks from a stale head, the
archive is a faithful copy, the anchor's recorded digest is the archive's
real digest, no stored hash is ever rewritten, a clean chain is refused,
and nothing happens without an explicit confirmation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess

import pytest

from mind_mem.cross_ledger import reconcile
from mind_mem.evidence_objects import (
    _GENESIS_HASH,
    EvidenceAction,
    EvidenceChain,
    EvidenceChainCompromisedError,
    EvidenceObject,
)
from mind_mem.evidence_recovery import (
    ARCHIVE_INFIX,
    ARCHIVE_MISMATCH,
    ARCHIVE_MISSING,
    ARCHIVE_OK,
    ARCHIVE_UNREADABLE,
    BREAK_FORK_FROM_STALE_HEAD,
    BREAK_GENESIS_RESTART,
    COMPANION_ENTRIES_KEY,
    COMPANION_PRESENT_KEY,
    COMPANION_TAIL_KEY,
    RECOVERY_VERB,
    RECOVERY_VERB_KEY,
    ChainRecoveryRefused,
    recover_chain,
    survey_chain_file,
    verify_archives,
)
from mind_mem.hash_chain_v2 import GENESIS_HASH as COMPANION_GENESIS_HASH
from mind_mem.hash_chain_v2 import HashChainV2, head_path
from mind_mem.verify_cli import EXIT_EVIDENCE, verify_workspace

# ---------------------------------------------------------------------------
# Fixtures — chains damaged the way the field damaged them
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emulate_windows_readonly_unlink(monkeypatch) -> None:
    """Make Linux exercise Windows' refusal to unlink a read-only file."""
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        candidate = os.fspath(path)
        if ARCHIVE_INFIX in os.path.basename(candidate):
            mode = os.stat(candidate, follow_symlinks=False).st_mode
            if not mode & stat.S_IWRITE:
                raise PermissionError(f"read-only Windows file: {candidate}")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)


def _seed(store: str, n: int = 3) -> EvidenceChain:
    """Write *n* correctly linked records and return the live chain."""
    chain = EvidenceChain(store_path=store)
    for i in range(n):
        chain.create(
            action=EvidenceAction.APPLY,
            actor="seed",
            target_block_id=f"B-{i:03d}",
            target_file="decisions/DECISIONS.md",
            payload=b"payload",
        )
    return chain


def _raw_append(store: str, previous_hash: str, block_id: str) -> EvidenceObject:
    """Append a self-consistent record linked to *previous_hash*.

    This is what a release with no fork refusal does: it mints a valid
    record — its own hash checks out — and links it to whatever it
    believed the tail was. Every record written this way verifies
    individually; only the linkage between them is wrong, which is
    exactly the shape of the damaged store in the field.
    """
    ev = EvidenceChain()._forge(
        previous_hash=previous_hash,
        action=EvidenceAction.APPLY,
        actor="stale-writer",
        target_block_id=block_id,
        target_file="decisions/DECISIONS.md",
        payload_hash=hashlib.sha256(b"payload").hexdigest(),
        metadata={"evidence_schema": "v3.1"},
        confidence=1.0,
    )
    with open(store, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")
    return ev


def _genesis_restart_store(tmp_path) -> str:
    """Three good records, then one that restarts the chain at genesis."""
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    _seed(store, 3)
    _raw_append(store, _GENESIS_HASH, "B-restart")
    return store


def _stale_head_store(tmp_path) -> str:
    """Four good records, then one linked to the *third* record's hash."""
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    chain = _seed(store, 4)
    stale = chain._entries[2].evidence_hash
    _raw_append(store, stale, "B-fork")
    return store


def _clean_store(tmp_path) -> str:
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    _seed(store, 4)
    return store


def _seed_companion(store: str, n: int = 3, prefix: str = "retained") -> tuple[str, str]:
    db_path = os.path.join(os.path.dirname(store), "hash_chain_v2.db")
    chain = HashChainV2(db_path)
    for index in range(n):
        chain.append(f"{prefix}-{index}", "WRITE", f"content-{prefix}-{index}")
    tail = chain.get_latest(1)[0].entry_hash if n else COMPANION_GENESIS_HASH
    return db_path, tail


def _rewrite_anchor_metadata(store: str, mutate) -> None:
    """Replace the one-record recovery segment with a valid changed anchor."""
    original = EvidenceChain(store_path=store).get_latest(1)[0]
    metadata = dict(original.metadata)
    mutate(metadata)
    rewritten = EvidenceChain()._forge(
        previous_hash=original.previous_hash,
        action=original.action,
        actor=original.actor,
        target_block_id=original.target_block_id,
        target_file=original.target_file,
        payload_hash=original.payload_hash,
        metadata=metadata,
        confidence=original.confidence,
    )
    with open(store, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(rewritten.to_dict(), separators=(",", ":")) + "\n")
    assert EvidenceChain(store_path=store).verify_chain() == (True, [])


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


def test_survey_classifies_a_genesis_restart(tmp_path):
    store = _genesis_restart_store(tmp_path)
    survey = survey_chain_file(store)

    assert survey.records == 4
    assert survey.is_damaged
    assert [b.kind for b in survey.breaks] == [BREAK_GENESIS_RESTART]
    assert survey.breaks[0].line == 4
    assert survey.breaks[0].found_previous == _GENESIS_HASH
    assert survey.census[BREAK_GENESIS_RESTART] == 1


def test_survey_classifies_a_fork_from_a_stale_head(tmp_path):
    store = _stale_head_store(tmp_path)
    survey = survey_chain_file(store)

    assert survey.records == 5
    assert [b.kind for b in survey.breaks] == [BREAK_FORK_FROM_STALE_HEAD]
    assert survey.breaks[0].line == 5
    assert survey.census[BREAK_FORK_FROM_STALE_HEAD] == 1


def test_survey_of_a_clean_chain_reports_no_damage(tmp_path):
    survey = survey_chain_file(_clean_store(tmp_path))

    assert survey.records == 4
    assert survey.breaks == ()
    assert not survey.is_damaged


def test_survey_reads_a_store_the_loader_cannot(tmp_path):
    """The whole point: the loader stops at break one, the survey does not."""
    store = _genesis_restart_store(tmp_path)
    _raw_append(store, _GENESIS_HASH, "B-restart-2")

    assert len(EvidenceChain(store_path=store)) == 0
    assert survey_chain_file(store).records == 5
    assert len(survey_chain_file(store).breaks) == 2


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_recovery_refuses_a_chain_that_verifies_clean(tmp_path):
    store = _clean_store(tmp_path)
    before = _sha256_file(store)

    with pytest.raises(ChainRecoveryRefused, match="verifies clean"):
        recover_chain(store, actor="operator", confirm=True)

    assert _sha256_file(store) == before
    assert "evidence_chain.jsonl" in os.listdir(os.path.dirname(store))
    assert not [n for n in os.listdir(os.path.dirname(store)) if ".damaged-" in n]


def test_recovery_requires_explicit_confirmation(tmp_path):
    store = _genesis_restart_store(tmp_path)
    before = _sha256_file(store)

    with pytest.raises(ChainRecoveryRefused, match="confirm"):
        recover_chain(store, actor="operator")

    assert _sha256_file(store) == before
    assert not [n for n in os.listdir(os.path.dirname(store)) if ".damaged-" in n]


def test_recovery_refuses_a_store_that_does_not_exist(tmp_path):
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    os.makedirs(os.path.dirname(store))

    with pytest.raises(ChainRecoveryRefused):
        recover_chain(store, actor="operator", confirm=True)


def test_recovery_refuses_a_mismatched_reviewed_digest_without_mutation(tmp_path):
    store = _genesis_restart_store(tmp_path)
    before = _sha256_file(store)
    directory_before = sorted(os.listdir(os.path.dirname(store)))

    with pytest.raises(ChainRecoveryRefused, match="in-lock survey"):
        recover_chain(store, actor="operator", confirm=True, expected_sha256="0" * 64)

    assert _sha256_file(store) == before
    assert sorted(os.listdir(os.path.dirname(store))) == directory_before


def test_recovery_accepts_the_exact_reviewed_digest(tmp_path):
    store = _genesis_restart_store(tmp_path)
    expected = _sha256_file(store)

    result = recover_chain(store, actor="operator", confirm=True, expected_sha256=expected.upper())

    assert result.archive_sha256 == expected


def test_recovery_refuses_a_corrupt_companion_chain_without_mutation(tmp_path):
    store = _genesis_restart_store(tmp_path)
    db_path, _tail = _seed_companion(store, n=3)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("UPDATE hash_chain SET content_hash = 'tampered' WHERE rowid = 2")
        connection.commit()
    finally:
        connection.close()
    before = _sha256_file(store)
    companion_before = _sha256_file(db_path)

    with pytest.raises(ChainRecoveryRefused, match="own chain breaks"):
        recover_chain(store, actor="operator", confirm=True, expected_sha256=before)

    assert _sha256_file(store) == before
    assert _sha256_file(db_path) == companion_before
    assert not [name for name in os.listdir(os.path.dirname(store)) if ".damaged-" in name or name.endswith(".reanchor-pending")]


def test_recovery_aborts_when_a_writer_appends_mid_operation(tmp_path, monkeypatch):
    """A record that landed after the archive was taken must not be destroyed.

    The release that caused this damage takes no append lock, so a locked
    recovery can still have the store move under it. Simulated by appending
    from inside the anchor-minting step — the last point before the replace.
    """
    import mind_mem.evidence_recovery as recovery

    _emulate_windows_readonly_unlink(monkeypatch)
    store = _genesis_restart_store(tmp_path)
    real_metadata = recovery._anchor_metadata

    def append_then_build(*args, **kwargs):
        _raw_append(store, _GENESIS_HASH, "B-raced")
        return real_metadata(*args, **kwargs)

    monkeypatch.setattr(recovery, "_anchor_metadata", append_then_build)

    with pytest.raises(ChainRecoveryRefused, match="changed while the recovery ran"):
        recover_chain(store, actor="operator", confirm=True)

    # The raced record is still in the store, unharmed, and the abort left
    # nothing behind — no pending segment, and no archive that would be a
    # prefix of the history rather than the history.
    records = [json.loads(line) for line in open(store, encoding="utf-8") if line.strip()]
    assert len(records) == 5
    assert records[-1]["target_block_id"] == "B-raced"
    leftovers = os.listdir(os.path.dirname(store))
    assert not [n for n in leftovers if ".damaged-" in n or n.endswith(".reanchor-pending")], leftovers

    # Positive control: with no racing writer the same store recovers, so the
    # emptiness above is the abort unwinding rather than recovery never
    # having anything to leave.
    monkeypatch.setattr(recovery, "_anchor_metadata", real_metadata)
    result = recover_chain(store, actor="operator", confirm=True)
    assert os.path.isfile(result.archive_path)
    assert result.archived_records == 5


def test_recovery_aborts_when_the_companion_chain_moves_mid_operation(tmp_path, monkeypatch):
    import mind_mem.evidence_recovery as recovery

    _emulate_windows_readonly_unlink(monkeypatch)
    store = _genesis_restart_store(tmp_path)
    db_path, _tail = _seed_companion(store, n=2)
    before = _sha256_file(store)
    real_metadata = recovery._anchor_metadata

    def append_companion_then_build(*args, **kwargs):
        HashChainV2(db_path).append("raced", "WRITE", "raced content")
        return real_metadata(*args, **kwargs)

    monkeypatch.setattr(recovery, "_anchor_metadata", append_companion_then_build)

    with pytest.raises(ChainRecoveryRefused, match="companion"):
        recover_chain(store, actor="operator", confirm=True)

    assert _sha256_file(store) == before
    assert HashChainV2(db_path).length == 3, "the raced companion row was preserved"
    leftovers = os.listdir(os.path.dirname(store))
    assert not [name for name in leftovers if ".damaged-" in name or name.endswith(".reanchor-pending")]


def test_recovery_preserves_a_changed_partial_archive_on_race_abort(tmp_path, monkeypatch):
    import mind_mem.evidence_recovery as recovery

    store = _genesis_restart_store(tmp_path)
    before = _sha256_file(store)
    real_metadata = recovery._anchor_metadata

    def change_archive_then_append(*args, **kwargs):
        directory = os.path.dirname(store)
        names = [name for name in os.listdir(directory) if ARCHIVE_INFIX in name]
        assert len(names) == 1
        archive = os.path.join(directory, names[0])
        os.chmod(archive, 0o600)
        with open(archive, "ab") as handle:
            handle.write(b"changed by another actor")
        os.chmod(archive, 0o400)
        _raw_append(store, _GENESIS_HASH, "B-raced")
        return real_metadata(*args, **kwargs)

    monkeypatch.setattr(recovery, "_anchor_metadata", change_archive_then_append)

    with pytest.raises(OSError, match="refusing to remove changed recovery archive"):
        recover_chain(store, actor="operator", confirm=True)

    assert _sha256_file(store) != before, "the concurrent store append must survive"
    archives = [os.path.join(os.path.dirname(store), name) for name in os.listdir(os.path.dirname(store)) if ARCHIVE_INFIX in name]
    assert len(archives) == 1, "a changed path must not be deleted as this operation's disposable output"
    assert not os.stat(archives[0]).st_mode & stat.S_IWRITE, "the retained archive must remain read-only"


def test_recovery_restores_readonly_mode_if_partial_archive_unlink_fails(tmp_path, monkeypatch):
    import mind_mem.evidence_recovery as recovery

    store = _genesis_restart_store(tmp_path)
    real_metadata = recovery._anchor_metadata
    real_unlink = os.unlink

    def append_then_build(*args, **kwargs):
        _raw_append(store, _GENESIS_HASH, "B-raced")
        return real_metadata(*args, **kwargs)

    def reject_archive_unlink(path, *args, **kwargs):
        if ARCHIVE_INFIX in os.path.basename(os.fspath(path)):
            raise PermissionError("simulated delete failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(recovery, "_anchor_metadata", append_then_build)
    monkeypatch.setattr(os, "unlink", reject_archive_unlink)

    with pytest.raises(PermissionError, match="simulated delete failure"):
        recover_chain(store, actor="operator", confirm=True)

    archives = [os.path.join(os.path.dirname(store), name) for name in os.listdir(os.path.dirname(store)) if ARCHIVE_INFIX in name]
    assert len(archives) == 1
    assert not os.stat(archives[0]).st_mode & stat.S_IWRITE, "failed cleanup must not leave a sealed archive writable"


def test_recovery_refuses_to_overwrite_an_existing_archive(tmp_path, monkeypatch):
    store = _genesis_restart_store(tmp_path)
    result = recover_chain(store, actor="operator", confirm=True)
    stamp = os.path.basename(result.archive_path).rsplit("-", 1)[-1]
    # Freeze the clock so the second recovery picks the same archive name.
    monkeypatch.setattr("mind_mem.evidence_recovery._archive_stamp", lambda: stamp)
    _raw_append(store, _GENESIS_HASH, "B-damage-again")

    with pytest.raises(ChainRecoveryRefused, match="already"):
        recover_chain(store, actor="operator", confirm=True)


# ---------------------------------------------------------------------------
# Recovery — the two damage shapes seen in the field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_store,kind",
    [(_genesis_restart_store, BREAK_GENESIS_RESTART), (_stale_head_store, BREAK_FORK_FROM_STALE_HEAD)],
)
def test_recovered_chain_appends_again(tmp_path, make_store, kind):
    store = make_store(tmp_path)
    assert EvidenceChain(store_path=store).integrity_compromised

    result = recover_chain(store, actor="operator", reason="field recovery", confirm=True)
    assert result.census[kind] >= 1

    chain = EvidenceChain(store_path=store)
    assert not chain.integrity_compromised
    assert len(chain) == 1
    ev = chain.create(
        action=EvidenceAction.APPLY,
        actor="post-recovery",
        target_block_id="B-new",
        target_file="decisions/DECISIONS.md",
        payload=b"after",
    )
    assert ev.previous_hash == result.anchor.evidence_hash
    ok, broken = EvidenceChain(store_path=store).verify_chain()
    assert ok, broken


def test_archive_is_byte_identical_to_the_original(tmp_path):
    store = _genesis_restart_store(tmp_path)
    original = str(tmp_path / "original.jsonl")
    with open(store, "rb") as src, open(original, "wb") as dst:
        dst.write(src.read())

    result = recover_chain(store, actor="operator", confirm=True)

    assert subprocess.run(["cmp", "-s", original, result.archive_path]).returncode == 0
    assert _sha256_file(result.archive_path) == _sha256_file(original)


def test_anchor_records_the_real_archive_digest(tmp_path):
    store = _genesis_restart_store(tmp_path)
    result = recover_chain(store, actor="operator", confirm=True)

    on_disk = _sha256_file(result.archive_path)
    assert result.archive_sha256 == on_disk
    assert result.anchor.metadata["archived_sha256"] == on_disk
    # The digest is also the record's payload_hash, so it is covered by the
    # evidence hash through the ordinary preimage rather than by a side
    # channel this module would have to be trusted about.
    assert result.anchor.payload_hash == on_disk
    assert EvidenceChain(store_path=store).verify(result.anchor)


def test_anchor_carries_the_break_census(tmp_path):
    store = _genesis_restart_store(tmp_path)
    _raw_append(store, _GENESIS_HASH, "B-restart-2")
    result = recover_chain(store, actor="operator", confirm=True)

    meta = result.anchor.metadata
    assert meta[RECOVERY_VERB_KEY] == RECOVERY_VERB
    assert meta["archived_records"] == 5
    assert meta["archived_break_count"] == 2
    assert meta["archived_break_census"][BREAK_GENESIS_RESTART] == 2
    assert meta["archived_first_break_line"] == 4
    assert meta["archived_last_break_line"] == 5
    assert meta["archived_chain"] == os.path.basename(result.archive_path)


def test_anchor_explicitly_denies_continuity_and_hashes_both_claims(tmp_path):
    store = _genesis_restart_store(tmp_path)
    result = recover_chain(store, actor="operator", confirm=True)
    meta = result.anchor.metadata

    assert meta["continues_predecessor_chain"] is False
    assert meta["predecessor_trust_restored"] is False
    assert EvidenceChain(store_path=store).verify(result.anchor)

    changed = json.loads(json.dumps(result.anchor.to_dict()))
    changed["metadata"]["continues_predecessor_chain"] = True
    assert not EvidenceChain().verify(EvidenceObject.from_dict(changed))


def test_anchor_binds_the_retained_companion_hash_chain(tmp_path):
    store = _genesis_restart_store(tmp_path)
    _db_path, tail = _seed_companion(store, n=3)

    result = recover_chain(store, actor="operator", confirm=True)
    meta = result.anchor.metadata

    assert meta[COMPANION_PRESENT_KEY] is True
    assert meta[COMPANION_ENTRIES_KEY] == 3
    assert meta[COMPANION_TAIL_KEY] == tail
    verdict = reconcile(str(tmp_path))
    assert verdict.ok, verdict.reasons
    assert verdict.recovery_baseline_present is True
    assert verdict.recovery_baseline_entries == 3
    assert verdict.recovery_baseline_tail == tail


@pytest.mark.parametrize("damage", ["missing", "truncated", "replaced"])
def test_reconciliation_rejects_missing_or_changed_retained_history(tmp_path, damage):
    store = _genesis_restart_store(tmp_path)
    db_path, _tail = _seed_companion(store, n=3)
    recover_chain(store, actor="operator", confirm=True)

    if os.path.isfile(head_path(db_path)):
        os.remove(head_path(db_path))
    if damage == "missing":
        os.remove(db_path)
    elif damage == "truncated":
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("DELETE FROM hash_chain WHERE rowid = (SELECT MAX(rowid) FROM hash_chain)")
            connection.commit()
        finally:
            connection.close()
    else:
        os.remove(db_path)
        _seed_companion(store, n=3, prefix="replacement")

    verdict = reconcile(str(tmp_path))
    assert not verdict.ok
    assert any("recovery anchor" in reason or "retained hash-chain prefix" in reason for reason in verdict.reasons)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda meta: meta.pop(COMPANION_ENTRIES_KEY),
        lambda meta: meta.__setitem__(COMPANION_PRESENT_KEY, "yes"),
        lambda meta: meta.__setitem__(COMPANION_ENTRIES_KEY, "3"),
        lambda meta: meta.__setitem__(COMPANION_ENTRIES_KEY, True),
        lambda meta: meta.__setitem__(COMPANION_ENTRIES_KEY, -1),
        lambda meta: meta.__setitem__(COMPANION_TAIL_KEY, None),
    ],
    ids=["missing", "boolean-type", "count-type", "boolean-count", "negative-count", "tail-type"],
)
def test_reconciliation_rejects_a_malformed_recovery_baseline(tmp_path, mutate):
    store = _genesis_restart_store(tmp_path)
    recover_chain(store, actor="operator", confirm=True)
    _rewrite_anchor_metadata(store, mutate)

    verdict = reconcile(str(tmp_path))

    assert verdict.checked
    assert not verdict.ok
    assert "a recovery anchor carries a missing or malformed companion hash-chain baseline" in verdict.reasons


def test_no_stored_hash_is_rewritten(tmp_path):
    store = _genesis_restart_store(tmp_path)
    before = [json.loads(line) for line in open(store, encoding="utf-8") if line.strip()]
    # Positive control: comparing two empty lists would pass without proving
    # anything, so pin that there really are records to preserve.
    assert len(before) == 4
    assert all(r["evidence_hash"] for r in before)

    result = recover_chain(store, actor="operator", confirm=True)

    after = [json.loads(line) for line in open(result.archive_path, encoding="utf-8") if line.strip()]
    assert [(r["evidence_id"], r["evidence_hash"], r["previous_hash"]) for r in after] == [
        (r["evidence_id"], r["evidence_hash"], r["previous_hash"]) for r in before
    ]

    # And none of the archived hashes reappears as a *stored* hash in the new
    # segment: the anchor links from genesis, so nothing was re-parented.
    archived_hashes = {r["evidence_hash"] for r in before}
    new = [json.loads(line) for line in open(store, encoding="utf-8") if line.strip()]
    assert len(new) == 1
    assert new[0]["previous_hash"] == _GENESIS_HASH
    assert new[0]["evidence_hash"] not in archived_hashes


def test_the_break_does_not_disappear(tmp_path):
    """New segment verifies from its own genesis; the archive never will."""
    store = _genesis_restart_store(tmp_path)
    result = recover_chain(store, actor="operator", confirm=True)

    new_chain = EvidenceChain(store_path=store)
    ok, broken = new_chain.verify_chain()
    assert ok and not broken
    assert new_chain.get_latest(1)[0].previous_hash == _GENESIS_HASH

    archived = EvidenceChain(store_path=result.archive_path)
    assert archived.integrity_compromised
    assert len(archived) == 0
    archived_ok, archived_broken = archived.verify_chain()
    assert not archived_ok
    assert archived_broken == ["load_integrity_compromised"]


def test_the_archive_head_hash_is_pinned_by_the_anchor(tmp_path):
    store = _stale_head_store(tmp_path)
    last = json.loads([ln for ln in open(store, encoding="utf-8") if ln.strip()][-1])
    result = recover_chain(store, actor="operator", confirm=True)

    assert result.anchor.metadata["archived_head_hash"] == last["evidence_hash"]


# ---------------------------------------------------------------------------
# Recovery is explicit — never a side effect of reading
# ---------------------------------------------------------------------------


def test_a_failed_load_never_recovers_by_itself(tmp_path):
    store = _genesis_restart_store(tmp_path)
    directory = os.path.dirname(store)

    for _ in range(3):
        chain = EvidenceChain(store_path=store)
        assert chain.integrity_compromised
        with pytest.raises(EvidenceChainCompromisedError):
            chain.create(
                action=EvidenceAction.APPLY,
                actor="caller",
                target_block_id="B-x",
                target_file="decisions/DECISIONS.md",
            )

    assert not [n for n in os.listdir(directory) if ".damaged-" in n]

    # Positive control: an absence proves nothing unless the thing can
    # appear. The explicit call does create exactly that file, so the
    # emptiness above is about reading, not about a name nothing ever uses.
    recover_chain(store, actor="operator", confirm=True)
    assert len([n for n in os.listdir(directory) if ".damaged-" in n]) == 1


def test_chain_method_recovers_and_unfreezes_in_place(tmp_path):
    store = _genesis_restart_store(tmp_path)
    chain = EvidenceChain(store_path=store)
    assert chain.integrity_compromised

    survey = chain.survey_damage()
    assert survey.census[BREAK_GENESIS_RESTART] == 1

    with pytest.raises(ChainRecoveryRefused):
        chain.recover_by_reanchor(actor="operator")

    result = chain.recover_by_reanchor(actor="operator", confirm=True)
    assert not chain.integrity_compromised
    assert chain.load_failure is None
    assert len(chain) == 1

    ev = chain.create(
        action=EvidenceAction.APPLY,
        actor="post-recovery",
        target_block_id="B-new",
        target_file="decisions/DECISIONS.md",
    )
    assert ev.previous_hash == result.anchor.evidence_hash
    ok, broken = EvidenceChain(store_path=store).verify_chain()
    assert ok, broken


def test_the_archive_is_left_read_only(tmp_path):
    store = _genesis_restart_store(tmp_path)
    result = recover_chain(store, actor="operator", confirm=True)

    mode = os.stat(result.archive_path).st_mode & 0o777
    assert mode & 0o222 == 0, oct(mode)


# ---------------------------------------------------------------------------
# Reading the anchor's claim BACK
#
# Recovery records an archive's sha256 in a tamper-evident record. Until
# ``verify_archives`` existed nothing ever read it back: an adversarial review
# deleted the archive after a recovery and found ``verify_chain()`` returning
# ``(True, [])`` and a survey reporting "intact", because the anchor named a
# file that was gone and no code looked. Recorded evidence with no check behind
# it is a note, not tamper-evidence. These tests are that check.
# ---------------------------------------------------------------------------


def _recovered(tmp_path) -> tuple[str, str]:
    """A recovered store plus the path of the archive its anchor attests."""
    store = _genesis_restart_store(tmp_path)
    recover_chain(store, actor="test", reason="archive-check fixture", confirm=True)
    directory = os.path.dirname(store)
    archive = next(os.path.join(directory, f) for f in os.listdir(directory) if ARCHIVE_INFIX in f)
    return store, archive


def _rewrite_archive(path: str, data: bytes) -> None:
    """Archives are chmod 0444 by recovery, so a test edit must re-open them."""
    os.chmod(path, 0o600)
    with open(path, "wb") as handle:
        handle.write(data)


def test_an_untouched_archive_verifies(tmp_path):
    store, _archive = _recovered(tmp_path)
    checks = verify_archives(store)
    assert len(checks) == 1, checks
    assert checks[0].status == ARCHIVE_OK
    assert checks[0].ok


def test_a_deleted_archive_is_reported_missing(tmp_path):
    """The exact case that previously reported 'intact'."""
    store, archive = _recovered(tmp_path)
    assert verify_archives(store)[0].status == ARCHIVE_OK, "positive control"
    os.chmod(archive, 0o600)
    os.remove(archive)
    check = verify_archives(store)[0]
    assert check.status == ARCHIVE_MISSING, check
    assert not check.ok


def test_an_archive_edited_in_place_is_reported_mismatch(tmp_path):
    """Same size, different bytes — the edit a size check alone would miss."""
    store, archive = _recovered(tmp_path)
    with open(archive, "rb") as handle:
        data = handle.read()
    _rewrite_archive(archive, data[:-1] + (b"X" if data[-1:] != b"X" else b"Y"))
    check = verify_archives(store)[0]
    assert check.status == ARCHIVE_MISMATCH, check
    assert check.expected_bytes == check.actual_bytes, "the size is deliberately unchanged"
    assert "edited in place" in check.detail


def test_a_truncated_archive_is_reported_mismatch(tmp_path):
    store, archive = _recovered(tmp_path)
    with open(archive, "rb") as handle:
        data = handle.read()
    _rewrite_archive(archive, data[:-1])
    check = verify_archives(store)[0]
    assert check.status == ARCHIVE_MISMATCH, check
    assert "size changed" in check.detail


def test_restoring_the_archive_verifies_again(tmp_path):
    """The check tracks the bytes, not a one-way latch."""
    store, archive = _recovered(tmp_path)
    with open(archive, "rb") as handle:
        data = handle.read()
    _rewrite_archive(archive, data[:-1])
    assert verify_archives(store)[0].status == ARCHIVE_MISMATCH
    _rewrite_archive(archive, data)
    assert verify_archives(store)[0].status == ARCHIVE_OK


def test_a_store_with_no_anchors_reports_nothing_rather_than_passing(tmp_path):
    """An empty result is not a pass, and callers must not read it as one."""
    store = _clean_store(tmp_path)
    assert verify_archives(store) == ()


def test_workspace_verifier_discloses_archive_absence(tmp_path):
    _clean_store(tmp_path)

    report = verify_workspace(str(tmp_path))

    assert report.checks["evidence_archives"] is True
    assert report.details["evidence_archives"] == {"archives": 0, "statuses": {}, "archive_names": []}
    assert any("no recovery anchor" in message for message in report.messages)


def test_workspace_verifier_checks_an_attested_archive(tmp_path):
    store, archive = _recovered(tmp_path)

    report = verify_workspace(str(tmp_path))

    assert report.checks["evidence_archives"] is True
    assert report.details["evidence_archives"]["archives"] == 1
    assert report.details["evidence_archives"]["statuses"] == {ARCHIVE_OK: 1}
    assert report.details["evidence_archives"]["archive_names"] == [os.path.basename(archive)]
    assert report.checks["evidence_chain"] is True
    assert os.path.samefile(store, tmp_path / "memory" / "evidence_chain.jsonl")


@pytest.mark.parametrize("damage", ["missing", "mismatch"])
def test_workspace_verifier_fails_on_a_missing_or_mismatched_archive(tmp_path, damage):
    _store, archive = _recovered(tmp_path)
    os.chmod(archive, 0o600)
    if damage == "missing":
        os.remove(archive)
        expected_status = ARCHIVE_MISSING
    else:
        with open(archive, "ab") as handle:
            handle.write(b"tamper")
        expected_status = ARCHIVE_MISMATCH

    report = verify_workspace(str(tmp_path))

    assert report.checks["evidence_archives"] is False
    assert report.details["evidence_archives"]["statuses"] == {expected_status: 1}
    assert report.exit_code == EXIT_EVIDENCE


def test_workspace_verifier_fails_when_an_archive_cannot_be_read(tmp_path, monkeypatch):
    _store, archive = _recovered(tmp_path)
    real_open = open

    def deny_archive(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(archive):
            raise PermissionError("synthetic archive read refusal")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", deny_archive)
    report = verify_workspace(str(tmp_path))

    assert report.checks["evidence_archives"] is False
    assert report.details["evidence_archives"]["statuses"] == {ARCHIVE_UNREADABLE: 1}
    assert report.exit_code == EXIT_EVIDENCE


def test_workspace_verifier_fails_closed_on_malformed_archive_claim(tmp_path):
    store, _archive = _recovered(tmp_path)
    rows = [json.loads(line) for line in open(store, encoding="utf-8") if line.strip()]
    rows[0]["metadata"]["archived_bytes"] = "not-an-integer"
    with open(store, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[0]) + "\n")

    report = verify_workspace(str(tmp_path))

    assert report.checks["evidence_chain"] is False
    assert report.checks["evidence_archives"] is False
    assert report.details["evidence_archives"]["statuses"] == {ARCHIVE_MISMATCH: 1}
    assert report.exit_code == EXIT_EVIDENCE


def test_the_check_resolves_by_basename_only(tmp_path):
    """A rewritten anchor must not be able to redirect the check at another file.

    The anchor records a basename; honouring a path from the record would let
    an edited record point the verifier at a file of its choosing.
    """
    store, archive = _recovered(tmp_path)
    decoy = os.path.join(tmp_path, "elsewhere.jsonl")
    with open(archive, "rb") as handle:
        os.makedirs(os.path.dirname(decoy), exist_ok=True) if os.path.dirname(decoy) else None
        with open(decoy, "wb") as out:
            out.write(handle.read())
    rows = []
    with open(store, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("metadata") or {}
            if meta.get(RECOVERY_VERB_KEY) == RECOVERY_VERB:
                meta["archived_chain"] = decoy  # an absolute path, not a basename
            rows.append(record)
    with open(store, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record) + "\n")
    os.chmod(archive, 0o600)
    os.remove(archive)
    check = verify_archives(store)[0]
    assert check.status == ARCHIVE_MISSING, f"the check followed a path out of the record instead of the store's own directory: {check}"


@pytest.mark.parametrize("symlink", [False, True])
def test_recovery_preserves_an_existing_pending_path(tmp_path, symlink):
    store = _genesis_restart_store(tmp_path)
    before = _sha256_file(store)
    pending = store + ".reanchor-pending"
    if symlink:
        try:
            os.symlink(store, pending)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
    else:
        with open(pending, "wb") as handle:
            handle.write(b"previous operation evidence")
    error = None
    try:
        recover_chain(store, actor="operator", confirm=True)
    except (ChainRecoveryRefused, OSError) as exc:
        error = exc
    assert _sha256_file(store) == before, "pending-path alias must never truncate the evidence store"
    assert error is not None, "existing pending state must refuse recovery"
    assert os.path.lexists(pending), "preexisting pending state belongs to its original owner"
    assert not [name for name in os.listdir(os.path.dirname(store)) if ARCHIVE_INFIX in name]
    if not symlink:
        with open(pending, "rb") as handle:
            assert handle.read() == b"previous operation evidence"


def test_recovery_refuses_a_pending_symlink_created_after_precheck(tmp_path, monkeypatch):
    import mind_mem.evidence_recovery as recovery

    _emulate_windows_readonly_unlink(monkeypatch)
    store = _genesis_restart_store(tmp_path)
    before = _sha256_file(store)
    pending = store + ".reanchor-pending"
    # Verify host support before beginning the interleaving.
    probe = tmp_path / "probe-link"
    try:
        os.symlink(store, probe)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    os.unlink(probe)
    original = recovery._anchor_metadata

    def insert_alias(*args, **kwargs):
        os.symlink(store, pending)
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "_anchor_metadata", insert_alias)
    with pytest.raises((ChainRecoveryRefused, OSError)):
        recover_chain(store, actor="operator", confirm=True)
    assert _sha256_file(store) == before, "exclusive pending creation must close the check/write race"
    assert os.path.islink(pending), "do not delete the path that won the race"
    assert not [name for name in os.listdir(os.path.dirname(store)) if ARCHIVE_INFIX in name]


@pytest.mark.parametrize("failure", [PermissionError("denied"), UnicodeError("invalid text")])
def test_unreadable_live_recovery_claims_never_report_no_anchor(tmp_path, monkeypatch, failure):
    import builtins

    from mind_mem.verify_cli import VerifyReport, check_evidence_archives

    store = _genesis_restart_store(tmp_path)
    recover_chain(store, actor="operator", confirm=True)
    real_open = builtins.open

    def unreadable(path, *args, **kwargs):
        if os.fspath(path) == store:
            raise failure
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable)
    report = VerifyReport(workspace=str(tmp_path), ok=True)
    check_evidence_archives(str(tmp_path), report)
    assert report.checks["evidence_archives"] is False
    assert report.exit_code == EXIT_EVIDENCE
