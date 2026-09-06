# Copyright 2026 STARGA, Inc.
"""``mm chain`` — the operator door onto evidence-chain recovery.

The library refuses to append to a forked chain and gives no way back on
its own, deliberately: sealing a governance history is an operator
decision. This is where that decision is taken, so the verb has to be
harder to fire than it is to read. These tests pin that shape:

* ``survey`` writes nothing and exits 1 on damage, so it works as a gate;
* ``recover`` prints the census **before** anything happens, on every run;
* ``recover`` without ``--confirm`` leaves the store byte-identical;
* ``recover`` on a chain that verifies clean is refused outright, so the
  verb cannot be used to quietly retire an intact ledger.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from mind_mem.evidence_objects import _GENESIS_HASH, EvidenceAction, EvidenceChain
from mind_mem.mm_cli import main


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _seed(store: str, n: int) -> EvidenceChain:
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


def _restart_at_genesis(store: str) -> None:
    """Append the way a release with no fork refusal did."""
    ev = EvidenceChain()._forge(
        previous_hash=_GENESIS_HASH,
        action=EvidenceAction.APPLY,
        actor="stale-writer",
        target_block_id="B-restart",
        target_file="decisions/DECISIONS.md",
        payload_hash=hashlib.sha256(b"payload").hexdigest(),
        metadata={"evidence_schema": "v3.1"},
        confidence=1.0,
    )
    with open(store, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")


@pytest.fixture()
def damaged(tmp_path) -> str:
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    _seed(store, 3)
    _restart_at_genesis(store)
    return store


@pytest.fixture()
def clean(tmp_path) -> str:
    store = str(tmp_path / "memory" / "evidence_chain.jsonl")
    _seed(store, 3)
    return store


# ---------------------------------------------------------------------------
# survey
# ---------------------------------------------------------------------------


def test_survey_exits_one_on_damage_and_changes_nothing(damaged, capsys):
    before = _sha256(damaged)

    assert main(["chain", "survey", "--store", damaged]) == 1

    out = capsys.readouterr().out
    assert "DAMAGED" in out
    assert "genesis_restart" in out
    assert "line 4" in out
    assert _sha256(damaged) == before


def test_survey_exits_zero_on_a_clean_chain(clean, capsys):
    assert main(["chain", "survey", "--store", clean]) == 0
    assert "intact" in capsys.readouterr().out


def test_survey_json_carries_the_census(damaged, capsys):
    assert main(["chain", "survey", "--store", damaged, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["records"] == 4
    assert payload["break_count"] == 1
    assert payload["census"]["genesis_restart"] == 1
    assert payload["breaks"][0]["line"] == 4


def test_survey_resolves_the_store_from_a_workspace(tmp_path, damaged, capsys):
    assert main(["chain", "survey", str(tmp_path)]) == 1
    assert "DAMAGED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# recover
# ---------------------------------------------------------------------------


def test_recover_without_confirm_prints_the_census_and_writes_nothing(damaged, capsys):
    before = _sha256(damaged)
    directory = os.path.dirname(damaged)

    assert main(["chain", "recover", "--store", damaged]) == 0

    out = capsys.readouterr().out
    assert "DAMAGED" in out
    assert "nothing was written" in out
    assert "--confirm" in out
    assert _sha256(damaged) == before
    assert not [name for name in os.listdir(directory) if ".damaged-" in name]

    # Positive control for that emptiness: the confirmed run does produce a
    # file matching the same pattern, so the assertion above is about the
    # missing flag rather than about a filename nothing ever writes.
    assert main(["chain", "recover", "--store", damaged, "--confirm"]) == 0
    assert len([name for name in os.listdir(directory) if ".damaged-" in name]) == 1


def test_recover_refuses_a_clean_chain(clean, capsys):
    before = _sha256(clean)

    assert main(["chain", "recover", "--store", clean, "--confirm"]) == 3

    captured = capsys.readouterr()
    assert "verifies clean" in captured.err
    assert _sha256(clean) == before
    assert not [name for name in os.listdir(os.path.dirname(clean)) if ".damaged-" in name]


def test_recover_refuses_a_clean_chain_before_confirmation_is_even_offered(clean, capsys):
    """The dry run must not invite a confirmation the library will refuse.

    ``recover_chain`` refuses a clean chain whether or not the CLI checks
    first, so the ``--confirm`` run is safe either way. This pins the case
    the library cannot reach: without ``--confirm`` nothing calls into it,
    and a CLI that skipped its own check would print "re-run with
    --confirm" over an intact ledger and exit 0 — advice to run an
    operation that cannot succeed, on the one file where an operator
    should never be encouraged to try.
    """
    assert main(["chain", "recover", "--store", clean]) == 3

    captured = capsys.readouterr()
    assert "verifies clean" in captured.err
    assert "--confirm" not in captured.out


def test_recover_refuses_a_missing_store(tmp_path, capsys):
    missing = str(tmp_path / "memory" / "evidence_chain.jsonl")

    assert main(["chain", "recover", "--store", missing, "--confirm"]) == 3
    assert "no evidence store" in capsys.readouterr().err


def test_recover_seals_and_reanchors(damaged, capsys):
    original = _sha256(damaged)
    directory = os.path.dirname(damaged)

    assert main(["chain", "recover", "--store", damaged, "--confirm", "--actor", "nikolai", "--reason", "field"]) == 0

    out = capsys.readouterr().out
    # The census is printed BEFORE the action, on the run that acts.
    assert out.index("DAMAGED") < out.index("archived 4 record(s)")

    archives = [name for name in os.listdir(directory) if ".damaged-" in name]
    assert len(archives) == 1
    archive = os.path.join(directory, archives[0])
    assert _sha256(archive) == original

    chain = EvidenceChain(store_path=damaged)
    assert not chain.integrity_compromised
    anchor = chain.get_latest(1)[0]
    assert anchor.actor == "nikolai"
    assert anchor.metadata["reason"] == "field"
    assert anchor.previous_hash == _GENESIS_HASH
    assert anchor.payload_hash == _sha256(archive)
    ok, broken = chain.verify_chain()
    assert ok, broken


def test_recovered_workspace_accepts_a_governed_append(damaged):
    assert main(["chain", "recover", "--store", damaged, "--confirm"]) == 0

    chain = EvidenceChain(store_path=damaged)
    chain.create(
        action=EvidenceAction.APPLY,
        actor="post-recovery",
        target_block_id="B-new",
        target_file="decisions/DECISIONS.md",
        payload=b"after",
    )
    ok, broken = EvidenceChain(store_path=damaged).verify_chain()
    assert ok, broken
    assert len(EvidenceChain(store_path=damaged)) == 2


def test_recover_json_reports_the_archive_and_the_anchor(damaged, capsys):
    assert main(["chain", "recover", "--store", damaged, "--confirm", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["archived_records"] == 4
    assert payload["breaks"] == 1
    assert payload["census"]["genesis_restart"] == 1
    assert payload["anchor"]["previous_hash"] == _GENESIS_HASH
    assert payload["anchor"]["payload_hash"] == payload["archive_sha256"] == _sha256(payload["archive_path"])
