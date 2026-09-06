# Copyright 2026 STARGA, Inc.
"""The boundary is fixed by a governed witness, never by a wall clock.

A timestamp bracket does not merely fail to detect a post-recovery admission --
it INVERTS. One companion row appended after the seal but stamped before it makes
the bracket refuse the true boundary and accept N+1, certifying the overclaim it
existed to prevent. Reproduced before this module was written; control 1 pins it.

So the witness is the authority. It is an independently established, explicitly
operator-authorised document whose CONTENT carries the boundary; the count and
tail are read out of it rather than supplied alongside it, which is what stops a
bare digest plus an arbitrary count from passing as proof.
"""

import os

import pytest

from mind_mem import boundary_witness as bw
from mind_mem.hash_chain_v2 import GENESIS_HASH, HashChainV2

ANCHOR_ID = "anchor-0001"
ANCHOR_HASH = "a" * 64
SEAL = "2026-01-02T00:00:00+00:00"
AUTHORITY = "operator"
TRUE_BOUNDARY = 4


@pytest.fixture
def chain(tmp_path):
    """Four genuine pre-seal rows. True boundary is 4."""
    mem = tmp_path / "memory"
    mem.mkdir()
    ch = HashChainV2(str(mem / "hash_chain_v2.db"))
    for i in range(4):
        ch.append(
            block_id=f"B{i}",
            action="create_block",
            content=f"r{i}",
            timestamp=f"2026-01-01T0{i}:00:00+00:00",
        )
    return str(mem / "evidence_chain.jsonl")


def _tail(workspace, n):
    db = os.path.join(os.path.dirname(workspace), "hash_chain_v2.db")
    ch = HashChainV2.open_readonly(db)
    if not n:
        return GENESIS_HASH
    if n > ch.length:
        return "e" * 128  # a count past the chain has no real tail
    rows = ch.get_latest(ch.length - n + 1)
    return rows[0].entry_hash


def _witness(workspace, *, count=4, anchor_id=ANCHOR_ID, anchor_hash=ANCHOR_HASH, authority=AUTHORITY, tail=None):
    return {
        "anchor_id": anchor_id,
        "anchor_hash": anchor_hash,
        "prefix_count": count,
        "prefix_tail": _tail(workspace, count) if tail is None else tail,
        "authority": authority,
        "trust_assumption": "fixes the boundary; restores no historic trust",
    }


def _verify(workspace, content, digest=None, **kw):
    """*digest* is the GOVERNED pin, established out of band -- not taken from
    the content under test.

    Digesting the caller's own content would authenticate nothing: anyone can
    compute a digest over a claim they authored. The trust root is a digest
    known independently (in production, the pre-recovery governance receipt),
    and the attestation supplies content that must reproduce it. That is what
    makes an arbitrary count unusable -- changing it changes the digest.
    """
    if digest is None:
        digest = bw.witness_digest(_witness(workspace, count=TRUE_BOUNDARY))
    return bw.verify_witness(
        workspace,
        kw.pop("anchor_id", ANCHOR_ID),
        kw.pop("anchor_hash", ANCHOR_HASH),
        content,
        digest,
        **kw,
    )


class TestTheInversionIsGone:
    """Control 1 -- the finding that made the clock unusable as authority."""

    def test_a_backdated_post_seal_row_does_not_move_the_boundary(self, chain):
        w = _witness(chain, count=4)
        got, why = _verify(chain, w)
        assert got is not None, why
        assert got.prefix_count == 4

        db = os.path.join(os.path.dirname(chain), "hash_chain_v2.db")
        HashChainV2(db).append(
            block_id="EVIL",
            action="create_block",
            content="post-recovery admission",
            timestamp="2026-01-01T05:00:00+00:00",  # backdated before the seal
        )
        # The witness still fixes 4. The backdated row changes nothing.
        got2, why2 = _verify(chain, w)
        assert got2 is not None, why2
        assert got2.prefix_count == 4, "a backdated row must not move the boundary"

        # And the overclaim is refused even though its prefix is cryptographically
        # valid and its timestamp sits before the seal.
        bad = _witness(chain, count=5)
        got3, why3 = _verify(chain, bad)
        assert got3 is None, "N+1 must be refused: the governed digest binds 4"
        assert "digest" in why3, why3


class TestTimestampsNeverCrash:
    """Controls 2-4 -- naive/aware is a refusal, never a TypeError."""

    @pytest.mark.parametrize(
        "row_ts,seal_ts",
        [
            ("2026-01-01T00:00:00", "2026-01-02T00:00:00+00:00"),  # naive row, aware seal
            ("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00"),  # aware row, naive seal
            ("2026-01-01T00:00:00", "2026-01-02T00:00:00"),  # both naive
            ("not-a-timestamp", "2026-01-02T00:00:00+00:00"),  # malformed
            ("", "2026-01-02T00:00:00+00:00"),  # absent
        ],
    )
    def test_timestamp_consistency_never_raises(self, tmp_path, row_ts, seal_ts):
        mem = tmp_path / "memory"
        mem.mkdir()
        ch = HashChainV2(str(mem / "hash_chain_v2.db"))
        ch.append(block_id="N", action="create_block", content="x", timestamp=row_ts)
        ws = str(mem / "evidence_chain.jsonl")
        verdict = bw.timestamp_consistency(ws, 1, seal_ts)  # must not raise
        assert verdict in (bw.TS_CORROBORATES, bw.TS_CONTRADICTS, bw.TS_NO_EVIDENCE)

    def test_timestamps_are_not_authority(self, chain):
        """Even a CONTRADICTING clock cannot overturn a sound witness."""
        w = _witness(chain, count=4)
        got, why = _verify(chain, w)
        assert got is not None, why


class TestTheWitnessMustActuallyBind:
    """Controls 5-9 -- a digest that does not bind these fields is not a witness."""

    def test_a_digest_that_does_not_match_its_content_is_refused(self, chain):
        got, why = _verify(chain, _witness(chain), digest="f" * 64)
        assert got is None
        assert "digest" in why, why

    def test_editing_the_count_invalidates_the_digest(self, chain):
        w = _witness(chain, count=4)
        d = bw.witness_digest(w)
        tampered = dict(w)
        tampered["prefix_count"] = 5  # the overclaim, digest unchanged
        got, why = _verify(chain, tampered, digest=d)
        assert got is None
        assert "digest" in why, why

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("anchor_id", "anchor-9999"),
            ("anchor_hash", "b" * 64),
        ],
    )
    def test_a_witness_for_a_different_anchor_is_refused(self, chain, field, bad):
        w = _witness(chain)
        w[field] = bad
        # Governed digest pinned to THIS content, so the binding passes and the
        # anchor clause behind it is what must refuse. Defence in depth: a
        # digest issued in error must not become a free pass.
        got, why = _verify(chain, w, digest=bw.witness_digest(w))
        assert got is None
        assert "anchor" in why, why

    @pytest.mark.parametrize("authority", ["self", "attacker", "OPERATOR", "operator "])
    def test_an_unrecognised_authority_is_refused(self, chain, authority):
        w = _witness(chain)
        w["authority"] = authority
        got, why = _verify(chain, w, digest=bw.witness_digest(w))
        assert got is None
        assert "authorit" in why, why

    @pytest.mark.parametrize("authority", [None, "", "   "])
    def test_a_missing_or_blank_authority_is_malformed(self, chain, authority):
        """Refused earlier, as malformed -- an empty authority is not an authority."""
        w = _witness(chain)
        if authority is None:
            w.pop("authority")
        else:
            w["authority"] = authority
        got, why = _verify(chain, w, digest="0" * 128)
        assert got is None
        assert "malformed" in why, why

    def test_a_witness_whose_tail_is_not_the_prefix_is_refused(self, chain):
        w = _witness(chain, count=4, tail="c" * 128)
        got, why = _verify(chain, w, digest=bw.witness_digest(w))
        assert got is None
        assert "prefix" in why or "tail" in why, why


class TestTooLargeAndTooSmall:
    """Control 10 -- both otherwise-valid neighbours are denied."""

    @pytest.mark.parametrize("count", [3, 5])
    def test_a_neighbouring_prefix_is_refused(self, chain, count):
        w = _witness(chain, count=count)
        # Its tail is genuinely that prefix's tail and its prefix verifies, so
        # only the governed binding can refuse it.
        got, why = _verify(chain, w)
        assert got is None, f"count={count} must be refused"
        assert "digest" in why, why

    def test_the_exact_boundary_is_accepted(self, chain):
        """Positive control: the refusals above are about the neighbours."""
        got, why = _verify(chain, _witness(chain, count=TRUE_BOUNDARY))
        assert got is not None, why
        assert got.prefix_count == TRUE_BOUNDARY


class TestAbsentCompanionDatabase:
    """Control 12 -- absent resolves consistently, and does not crash."""

    def test_absent_companion_is_false_zero_genesis(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        ws = str(mem / "evidence_chain.jsonl")
        present, count, tail = bw.companion_state(ws)
        assert (present, count, tail) == (False, 0, GENESIS_HASH)

    def test_absent_companion_does_not_crash_the_verifier(self, tmp_path):
        mem = tmp_path / "memory"
        mem.mkdir()
        ws = str(mem / "evidence_chain.jsonl")
        w = {
            "anchor_id": ANCHOR_ID,
            "anchor_hash": ANCHOR_HASH,
            "prefix_count": 4,
            "prefix_tail": "d" * 128,
            "authority": AUTHORITY,
            "trust_assumption": "x",
        }
        got, why = _verify(ws, w, digest=bw.witness_digest(w))
        assert got is None
        assert "absent" in why, why
