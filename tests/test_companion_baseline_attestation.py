"""A legacy recovery anchor may be bound forward, never blessed.

An anchor minted before the companion-baseline keys existed carries no
baseline, so ``cross_ledger`` convicts it and the workspace cannot be declared
healthy. Rewriting that anchor is forbidden -- it sits in a tamper-evident
chain and editing it would forge history -- and waiving the check would trade a
real invariant for a green light.

The repair is forward-only: a later attestation names the anchor and supplies
the baseline. What makes it sound is that the attestation is never believed.
Every clause is re-derived from an artifact at check time -- the archive is
re-hashed off disk, and the companion prefix is re-read from the live chain --
so the record supplies a binding to check, not evidence in itself.

These tests hold that line. Fixtures are generic; no production values appear.
"""

import hashlib
import os

import pytest

from mind_mem import boundary_witness as bw
from mind_mem import cross_ledger
from mind_mem.cross_ledger import (
    resolve_supplemental_baseline,
    verify_companion_prefix,
    verify_recovery_boundary,
    well_formed_baseline,
)
from mind_mem.evidence_recovery import (
    ATTESTS_ANCHOR_HASH_KEY,
    ATTESTS_ANCHOR_ID_KEY,
    COMPANION_ENTRIES_KEY,
    COMPANION_PRESENT_KEY,
    COMPANION_TAIL_KEY,
    DENIES_PREDECESSOR_KEY,
    TRUST_RESTORED_KEY,
)
from mind_mem.hash_chain_v2 import GENESIS_HASH, HashChainV2

ANCHOR_ID = "anchor-0001"
ANCHOR_HASH = "a" * 64
ARCHIVE_NAME = "evidence_chain.jsonl.damaged-20260101T000000Z"

#: Every fixture row is written before this; the anchor seals at it. The
#: boundary check needs a seal time, and defaulting it to "" fails closed --
#: which is correct behaviour, so the fixtures must supply one.
ANCHOR_TS = "2026-01-02T00:00:00+00:00"


@pytest.fixture
def workspace(tmp_path):
    """A memory dir with an archive and a companion chain, both real."""
    mem = tmp_path / "memory"
    mem.mkdir()
    archive = mem / ARCHIVE_NAME
    archive.write_bytes(b'{"row": 1}\n{"row": 2}\n')

    chain = HashChainV2(str(mem / "hash_chain_v2.db"))
    for i in range(4):
        chain.append(block_id=f"B-{i}", action="create_block", content=f"row {i}", timestamp=f"2026-01-01T0{i}:00:00+00:00")
    return str(mem / "evidence_chain.jsonl")


def _archive_facts(workspace_path):
    path = os.path.join(os.path.dirname(workspace_path), ARCHIVE_NAME)
    data = open(path, "rb").read()
    return {
        "archived_chain": ARCHIVE_NAME,
        "archived_sha256": hashlib.sha256(data).hexdigest(),
        "archived_bytes": len(data),
    }


def _companion_facts(workspace_path):
    db = os.path.join(os.path.dirname(workspace_path), "hash_chain_v2.db")
    chain = HashChainV2.open_readonly(db)
    rows = chain.get_latest(1)
    return chain.length, (rows[-1].entry_hash if rows else GENESIS_HASH)


def _witness_content(workspace_path, count, tail, *, anchor_id=None, anchor_hash=None):
    return {
        "anchor_id": ANCHOR_ID if anchor_id is None else anchor_id,
        "anchor_hash": ANCHOR_HASH if anchor_hash is None else anchor_hash,
        "prefix_count": count,
        "prefix_tail": tail,
        "authority": "operator",
        "trust_assumption": "fixes the boundary; restores no historic trust",
    }


def _governed_digest(workspace_path):
    """The out-of-band pin: the digest of the TRUE boundary's witness."""
    entries, tail = _companion_facts(workspace_path)
    return bw.witness_digest(_witness_content(workspace_path, entries, tail))


def _attestation(workspace_path, *, line=9, anchor_id=ANCHOR_ID, anchor_hash=ANCHOR_HASH, **over):
    entries, tail = _companion_facts(workspace_path)
    att = {
        "line": line,
        ATTESTS_ANCHOR_ID_KEY: anchor_id,
        ATTESTS_ANCHOR_HASH_KEY: anchor_hash,
        COMPANION_PRESENT_KEY: True,
        COMPANION_ENTRIES_KEY: entries,
        COMPANION_TAIL_KEY: tail,
        # A real attestation denies both. The verifier ENFORCES both, so a
        # fixture that omitted them would exercise the refusal path, not the
        # happy path.
        DENIES_PREDECESSOR_KEY: True,
        TRUST_RESTORED_KEY: False,
        # The boundary is fixed by a governed witness, not by the clock. The
        # content is a claim; the digest it must reproduce comes from governance.
        bw.WITNESS_CONTENT_KEY: _witness_content(workspace_path, entries, tail),
    }
    att.update(_archive_facts(workspace_path))
    att.update(over)
    # Rebuild the witness AFTER overrides so it binds the count/tail the
    # attestation actually claims. A witness left pointing at the original
    # values would make every override fail on the witness mismatch instead of
    # on the clause the test is aiming at.
    if bw.WITNESS_CONTENT_KEY not in over:
        att[bw.WITNESS_CONTENT_KEY] = _witness_content(
            workspace_path,
            att[COMPANION_ENTRIES_KEY],
            att[COMPANION_TAIL_KEY],
            anchor_id=anchor_id,
            anchor_hash=anchor_hash,
        )
    return att


def _resolve(workspace_path, attestations, anchor_line=1, anchor_archive=None, anchor_ts=ANCHOR_TS, governed_digest=None):
    """*anchor_archive* is captured BEFORE any tampering.

    Recomputing it here would hash the tampered file and compare it against
    itself, so every tamper test would pass while checking nothing.
    """
    if anchor_archive is None:
        anchor_archive = _archive_facts(workspace_path)
    return resolve_supplemental_baseline(
        workspace_path,
        ANCHOR_ID,
        ANCHOR_HASH,
        anchor_line,
        anchor_archive,
        attestations,
        anchor_ts,
        _governed_digest(workspace_path) if governed_digest is None else governed_digest,
    )


class TestTheHappyPathActuallyResolves:
    """The positive control. Without it every refusal below proves nothing."""

    def test_a_sound_attestation_supplies_the_baseline(self, workspace):
        baseline, why = _resolve(workspace, [_attestation(workspace)])
        entries, tail = _companion_facts(workspace)
        assert baseline == (True, entries, tail), why
        assert why == ""

    def test_no_attestation_leaves_the_anchor_unbound_without_inventing_a_reason(self, workspace):
        baseline, why = _resolve(workspace, [])
        assert baseline is None
        assert why == "", "a missing attestation is not itself a fault; the anchor simply stays fail-closed"


class TestTamperIsRefused:
    def test_a_flipped_archive_byte_is_refused(self, workspace):
        att = _attestation(workspace)
        facts = _archive_facts(workspace)  # before the flip
        path = os.path.join(os.path.dirname(workspace), ARCHIVE_NAME)
        data = bytearray(open(path, "rb").read())
        data[0] ^= 0x01
        open(path, "wb").write(bytes(data))
        baseline, why = _resolve(workspace, [att], anchor_archive=facts)
        assert baseline is None
        assert "digest" in why

    def test_a_truncated_archive_is_refused(self, workspace):
        att = _attestation(workspace)
        facts = _archive_facts(workspace)  # before the truncation
        path = os.path.join(os.path.dirname(workspace), ARCHIVE_NAME)
        data = open(path, "rb").read()
        open(path, "wb").write(data[:-1])
        baseline, why = _resolve(workspace, [att], anchor_archive=facts)
        assert baseline is None
        assert why  # digest or length, either is a refusal

    def test_a_missing_archive_is_refused(self, workspace):
        att = _attestation(workspace)
        facts = _archive_facts(workspace)  # before the delete
        os.remove(os.path.join(os.path.dirname(workspace), ARCHIVE_NAME))
        baseline, why = _resolve(workspace, [att], anchor_archive=facts)
        assert baseline is None
        assert "missing" in why


class TestReplayAndWrongAnchorAreRefused:
    def test_a_correct_id_with_the_wrong_hash_is_refused(self, workspace):
        att = _attestation(workspace, anchor_hash="b" * 64)
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert "wrong evidence hash" in why

    def test_an_attestation_for_another_anchor_is_ignored(self, workspace):
        att = _attestation(workspace, anchor_id="anchor-9999")
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert why == ""

    def test_an_attestation_naming_a_different_archive_is_refused(self, workspace):
        att = _attestation(workspace, archived_sha256="c" * 64)
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert "different archive" in why


class TestOrderingIsForwardOnly:
    def test_an_attestation_before_its_anchor_is_ignored(self, workspace):
        att = _attestation(workspace, line=1)
        baseline, why = _resolve(workspace, [att], anchor_line=5)
        assert baseline is None, "an attestation cannot precede the anchor it speaks for"


class TestConflictingAttestationsAreRefused:
    """The conflict guard, and the honest account of when it can fire.

    Once the boundary check pins ONE correct count, every field in a match
    tuple is independently re-derived from an artifact: ``present`` by
    well-formedness, ``count`` by the boundary bracket, ``tail`` by the live
    prefix. Two attestations that both pass therefore CANNOT disagree -- the
    guard is unreachable through the public path.

    That is a reason to test it more carefully, not less. An earlier version of
    this test fed the second attestation a one-row-short prefix and asserted
    "competing"; the boundary now rejects that first, so the assertion was
    measuring the wrong refusal. Both facts are pinned below.
    """

    def test_a_one_row_short_prefix_is_refused_by_the_boundary_not_the_conflict(self, workspace):
        """One-too-small: individually valid, still refused. Root's control."""
        db = os.path.join(os.path.dirname(workspace), "hash_chain_v2.db")
        chain = HashChainV2.open_readonly(db)
        length = chain.length
        short_tail = chain.get_latest(2)[0].entry_hash

        good = _attestation(workspace, line=9)
        short = _attestation(
            workspace,
            line=10,
            **{COMPANION_ENTRIES_KEY: length - 1, COMPANION_TAIL_KEY: short_tail},
        )
        # positive control: the full-length one alone DOES resolve, so the
        # refusal below is about the short prefix and nothing else.
        assert _resolve(workspace, [good])[0] is not None

        baseline, why = _resolve(workspace, [short])
        assert baseline is None
        assert "boundary" in why or "seal" in why, why

        baseline, why = _resolve(workspace, [good, short])
        assert baseline is None, "a disagreeing pair must stay fail-closed"

    def test_the_conflict_guard_fires_when_reached(self, workspace, monkeypatch):
        """Cover the backstop directly, since valid input cannot reach it.

        The upstream checks are stubbed to admit both attestations -- exactly
        the situation if one of them were ever weakened. The guard must then be
        what refuses. Deleting the guard kills this test.
        """
        monkeypatch.setattr(cross_ledger, "verify_companion_prefix", lambda *a, **k: (True, ""))

        # The witness is now the boundary authority, so THAT is what has to be
        # stubbed to reach the conflict branch -- stubbing the retired clock
        # check would leave the test green while covering nothing.
        def _admit(workspace, anchor_id, anchor_hash, content, digest, **k):
            w = bw.parse_witness(content)
            return w, ""

        monkeypatch.setattr(cross_ledger, "verify_witness", _admit)

        a = _attestation(workspace, line=9, **{COMPANION_ENTRIES_KEY: 4})
        b = _attestation(workspace, line=10, **{COMPANION_ENTRIES_KEY: 3})
        # positive control: with the stubs in place each one alone resolves,
        # so "competing" below is the conflict branch and not a leftover refusal.
        assert _resolve(workspace, [a])[0] is not None
        assert _resolve(workspace, [b])[0] is not None

        baseline, why = _resolve(workspace, [a, b])
        assert baseline is None
        assert "competing" in why, why


class TestTheCompanionPrefixIsRederived:
    def test_the_real_prefix_verifies(self, workspace):
        entries, tail = _companion_facts(workspace)
        assert verify_companion_prefix(workspace, entries, tail) == (True, "")

    def test_a_wrong_tail_is_refused(self, workspace):
        entries, _tail = _companion_facts(workspace)
        ok, why = verify_companion_prefix(workspace, entries, "d" * 128)
        assert not ok and why

    def test_claiming_more_entries_than_exist_is_refused(self, workspace):
        entries, tail = _companion_facts(workspace)
        ok, why = verify_companion_prefix(workspace, entries + 50, tail)
        assert not ok
        assert "holds" in why

    def test_the_prefix_still_verifies_after_the_chain_grows(self, workspace):
        """The point of checking a PREFIX rather than the current tail."""
        entries, tail = _companion_facts(workspace)
        chain = HashChainV2(os.path.join(os.path.dirname(workspace), "hash_chain_v2.db"))
        chain.append(block_id="B-later", action="create_block", content="later", timestamp="2026-01-03T00:00:00+00:00")
        grown, new_tail = _companion_facts(workspace)
        assert grown == entries + 1 and new_tail != tail, "positive control: the chain really grew"
        assert verify_companion_prefix(workspace, entries, tail) == (True, ""), "an attestation must stay checkable after later appends"


class TestOneValidatorNotTwo:
    def test_the_supplemental_path_uses_the_same_well_formedness_rule(self, workspace):
        assert not well_formed_baseline(True, -1, "e" * 128)
        assert not well_formed_baseline("yes", 3, "e" * 128)
        assert not well_formed_baseline(True, 3, "short")
        assert not well_formed_baseline(False, 3, "e" * 128), "absent cannot claim entries"
        assert well_formed_baseline(False, 0, GENESIS_HASH)

    def test_a_malformed_baseline_in_an_attestation_is_refused(self, workspace):
        att = _attestation(workspace, **{COMPANION_TAIL_KEY: "nope"})
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert "malformed" in why


# ---------------------------------------------------------------------------
# A valid prefix is NOT the recovery boundary
#
# Root's design audit, and it found a real hole. EVERY prefix of a hash chain
# verifies, so an attestation offering a LARGER prefix -- the sealed rows plus
# admissions appended after recovery, with their correctly matching tail --
# passes a validity check while absorbing post-recovery admission obligations
# into the baseline. A SMALLER prefix understates it. Both are cryptographically
# valid; neither is the boundary.
#
# The boundary is bracketed against the anchor's timestamp. Trust assumption,
# stated because it is not eliminable: timestamps are hash-bound so they cannot
# be altered afterwards, but a writer supplies its own, so a row backdated at
# creation would satisfy the bracket. This catches an honest-but-wrong boundary
# and any later edit; it does not defeat a lying writer.
# ---------------------------------------------------------------------------

SEAL_TS = "2026-01-02T00:00:00+00:00"


@pytest.fixture
def sealed_workspace(tmp_path):
    """Four rows written BEFORE the seal, then two written after it."""
    from mind_mem.hash_chain_v2 import HashChainV2 as _C

    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / ARCHIVE_NAME).write_bytes(b'{"row": 1}\n{"row": 2}\n')

    chain = _C(str(mem / "hash_chain_v2.db"))
    for i in range(4):  # pre-seal
        chain.append(block_id=f"pre-{i}", action="create_block", content=f"pre {i}", timestamp=f"2026-01-01T0{i}:00:00+00:00")
    for i in range(2):  # post-seal admissions
        chain.append(block_id=f"post-{i}", action="create_block", content=f"post {i}", timestamp=f"2026-01-03T0{i}:00:00+00:00")
    return str(mem / "evidence_chain.jsonl")


def _tail_at(workspace_path, n):
    """entry_hash of the chain's n-th row -- a genuinely valid prefix tail."""
    from mind_mem.hash_chain_v2 import HashChainV2 as _C

    chain = _C.open_readonly(os.path.join(os.path.dirname(workspace_path), "hash_chain_v2.db"))
    return chain.get_latest(chain.length - n + 1)[0].entry_hash


class TestOnlyTheBoundaryPrefixIsAccepted:
    def test_the_exact_boundary_is_accepted(self, sealed_workspace):
        """Positive control. Without it the two refusals below prove nothing."""
        ok, why = verify_recovery_boundary(sealed_workspace, 4, SEAL_TS)
        assert ok, why

    def test_one_too_large_is_refused_though_the_prefix_is_valid(self, sealed_workspace):
        assert verify_companion_prefix(sealed_workspace, 5, _tail_at(sealed_workspace, 5)) == (True, ""), (
            "control: the larger prefix really is cryptographically valid"
        )
        ok, why = verify_recovery_boundary(sealed_workspace, 5, SEAL_TS)
        assert not ok
        assert "TOO LARGE" in why

    def test_one_too_small_is_refused_though_the_prefix_is_valid(self, sealed_workspace):
        assert verify_companion_prefix(sealed_workspace, 3, _tail_at(sealed_workspace, 3)) == (True, ""), (
            "control: the smaller prefix really is cryptographically valid"
        )
        ok, why = verify_recovery_boundary(sealed_workspace, 3, SEAL_TS)
        assert not ok
        assert "TOO SMALL" in why

    def test_the_whole_chain_is_refused_when_rows_postdate_the_seal(self, sealed_workspace):
        ok, why = verify_recovery_boundary(sealed_workspace, 6, SEAL_TS)
        assert not ok
        assert "TOO LARGE" in why

    def test_an_unparseable_anchor_timestamp_fails_closed(self, sealed_workspace):
        ok, why = verify_recovery_boundary(sealed_workspace, 4, "not-a-timestamp")
        assert not ok
        assert "no boundary can be established" in why

    def test_a_missing_anchor_timestamp_fails_closed(self, sealed_workspace):
        ok, why = verify_recovery_boundary(sealed_workspace, 4, "")
        assert not ok


class TestDuplicateAttestations:
    def test_identical_duplicates_are_collapsed_not_convicted(self, workspace):
        """Replaying the same claim adds nothing and removes nothing."""
        att = _attestation(workspace, line=9)
        twin = dict(att)
        twin["line"] = 10
        baseline, why = _resolve(workspace, [att, twin])
        assert baseline is not None, why


class TestTheDenialsAreEnforcedNotJustEmitted:
    """Emitting a denial the verifier never reads is decoration.

    Root's review: these were written into the record and never checked, so an
    attestation that omitted the denial -- or asserted the sealed history had
    been rehabilitated -- resolved exactly like one that denied it.
    """

    @pytest.mark.parametrize("bad", [None, "true", 1, False])
    def test_a_missing_or_non_true_continuity_denial_is_refused(self, workspace, bad):
        att = _attestation(workspace)
        if bad is None:
            att.pop(DENIES_PREDECESSOR_KEY)
        else:
            att[DENIES_PREDECESSOR_KEY] = bad
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert "continuity" in why, why

    @pytest.mark.parametrize("bad", [None, True, "false", 0])
    def test_a_missing_or_non_false_trust_denial_is_refused(self, workspace, bad):
        att = _attestation(workspace)
        if bad is None:
            att.pop(TRUST_RESTORED_KEY)
        else:
            att[TRUST_RESTORED_KEY] = bad
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        assert "trust" in why, why

    def test_the_positive_control_still_resolves(self, workspace):
        """Without this, both refusals above could be refusing for any reason."""
        assert _resolve(workspace, [_attestation(workspace)])[0] is not None


class TestArchiveClaimHardening:
    """Root's review: two ways a malformed claim slipped past the matcher."""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../" + ARCHIVE_NAME,
            "../../etc/passwd",
            "sub/" + ARCHIVE_NAME,
            "..",
            ".",
        ],
    )
    def test_a_traversal_bearing_archive_name_is_refused_not_normalised(self, workspace, bad_name):
        """basename() would REWRITE the claim into one that verifies.

        The old matcher passed the name through os.path.basename, so an
        attestation naming "../../etc/passwd" was silently reinterpreted as
        "passwd" -- a claim naming something it had no business naming became a
        different, well-behaved claim. Malformed must fail closed instead.
        """
        att = _attestation(workspace, **{"archived_chain": bad_name})
        baseline, why = _resolve(workspace, [att])
        assert baseline is None
        # It must be refused for NAMING the path, not merely for disagreeing
        # with the anchor -- otherwise basename() could return and go unnoticed.
        ok, matcher_why = cross_ledger._archive_matches(workspace, att)
        assert ok is False
        assert "basename" in matcher_why, matcher_why

    @pytest.mark.parametrize("bad_bytes", [None, "22", 22.0, True, False, -1])
    def test_a_malformed_byte_length_is_refused_not_skipped(self, workspace, bad_bytes):
        """The guard used to SKIP the comparison when the length was malformed.

        `isinstance(want_bytes, int) and ... and size != want_bytes` reads as a
        strict check and behaves as an optional one: any non-int made the whole
        clause false, so the length constraint simply did not apply.
        """
        att = _attestation(workspace, **{"archived_bytes": bad_bytes})
        ok, why = cross_ledger._archive_matches(workspace, att)
        assert ok is False
        assert "byte length" in why, why

    def test_the_real_length_still_passes(self, workspace):
        """Positive control: the refusals above are about malformation."""
        ok, why = cross_ledger._archive_matches(workspace, _attestation(workspace))
        assert ok is True, why

    def test_an_unreadable_archive_does_not_leak_its_path(self, workspace, monkeypatch):
        """A public error must not carry an absolute production path."""
        # Built BEFORE patching: _attestation reads the archive itself, so
        # constructing it under the patch raises before the code under test runs.
        att = _attestation(workspace)

        def boom(*a, **k):
            raise PermissionError(13, "denied", "/data/openclaw/workspace/memory/x")

        monkeypatch.setattr("builtins.open", boom)
        ok, why = cross_ledger._archive_matches(workspace, att)
        assert ok is False
        assert "/data/openclaw" not in why and "/home/" not in why, why
        assert "PermissionError" in why, why
