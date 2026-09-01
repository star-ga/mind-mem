"""``list_evidence`` — the audit family's only *reader* of the evidence chain.

Registered since the v3.2.0 audit-domain split, classified USER scope in the
ACL, and referenced by name in the training corpus and the eval harness — but
until now not exercised by a single test anywhere. That combination is the bad
one: the tool is the surface an operator reaches for when they want to see what
governance actually recorded, so a silent break here does not crash anything,
it just makes the audit trail *look* empty.

What these tests pin, in order of how much a caller would be hurt by a
regression:

* the success envelope (``_schema_version`` / ``count`` / ``evidence``) and the
  full eleven-field shape of a serialised ``EvidenceObject`` — a caller reads
  ``evidence_hash`` and ``previous_hash`` out of this, so dropping a field is a
  breaking change even though the JSON still parses;
* **chain order**. The tool returns a list, and a list from an append-only
  ledger is only meaningful if it is still in ledger order. Rather than assert
  "sorted by timestamp" we assert the thing that actually matters: record N+1's
  ``previous_hash`` is record N's ``evidence_hash``, rooted at genesis. A
  reordering or a de-duplication would pass a timestamp check and fail this one;
* the refusal paths — an unknown ``action`` gets a structured error envelope
  with *no* ``count``/``evidence`` keys, so a refusal can never be misread as
  "the chain is empty";
* three warts that are pinned as ACTUAL behaviour, not endorsed. Each is
  flagged in its own docstring so that fixing it turns this file red on purpose
  rather than letting the fix land unnoticed:
    1. ``block_id`` silently wins over ``action``, and the action is then never
       validated — a typo'd action returns records instead of the error;
    2. ``limit=0`` returns *everything* on the filtered path (``records[-0:]``
       is the whole list) while returning nothing on the unfiltered path;
    3. unlike ``verify_merkle`` and ``verify_chain`` in the same module,
       ``list_evidence`` never calls ``_check_workspace``, so an uninitialised
       workspace is answered with an empty chain — and the directory is created
       as a side effect — instead of "Run: mind-mem-init".
"""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.evidence_objects import _GENESIS_HASH, EvidenceAction
from mind_mem.governance_gate import evict_gate, get_gate
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.audit import list_evidence

# The four records every test in this file reasons about, in creation order.
# Two blocks, one of them touched by two different actions, so a block filter
# and an action filter cut the same corpus along different axes.
_SEEDED = (
    (EvidenceAction.PROPOSE, "DEC-20200101-001", "proposer"),
    (EvidenceAction.APPLY, "DEC-20200101-001", "applier"),
    (EvidenceAction.PROPOSE, "DEC-20200101-002", "second-proposer"),
    (EvidenceAction.VERIFY, "DEC-20200101-002", "verifier"),
)

_RECORD_FIELDS = {
    "evidence_id",
    "timestamp",
    "action",
    "actor",
    "target_block_id",
    "target_file",
    "payload_hash",
    "previous_hash",
    "evidence_hash",
    "metadata",
    "confidence",
}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Spending it here makes a LATER, unrelated
    test fail with "Rate limit exceeded" under random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _no_admin_scope(monkeypatch):
    """Run every test at the DEFAULT scope, with nothing granted.

    ``list_evidence`` is in ``USER_TOOLS``; reading the ledger is not a
    privileged act. Clearing the env var rather than setting it to "user" means
    a future re-classification to ADMIN_TOOLS breaks these tests loudly instead
    of being masked by a scope the tests handed themselves.
    """
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)


@pytest.fixture
def ws(tmp_path):
    """A workspace whose evidence chain holds the four ``_SEEDED`` records.

    The records are written through the same ``GovernanceGate`` singleton the
    tool reads, so this is the real chain — hashes linked, persisted to
    ``memory/evidence_chain.jsonl`` — not a stand-in.
    """
    w = tmp_path / "ws"
    # The Markdown corpus layout. list_evidence itself does not require it (see
    # TestUninitialisedWorkspace), but the sibling tools it is compared against
    # do, and a realistic workspace is what a regression should be measured in.
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)

    gate = get_gate(str(w))
    for i, (action, block_id, actor) in enumerate(_SEEDED):
        gate.evidence.create(
            action=action,
            actor=actor,
            target_block_id=block_id,
            target_file="decisions/DECISIONS.md",
            payload=f"payload-{i}".encode(),
            metadata={"seq": i},
            confidence=0.5,
        )
    yield w
    # The gate cache is keyed on the realpath of a directory pytest is about to
    # delete; evicting keeps one cached gate per test from accumulating.
    evict_gate(str(w))


def _call(workspace, **kwargs) -> dict:
    """Invoke the tool against *workspace* and parse its JSON envelope."""
    with use_workspace(str(workspace)):
        return json.loads(list_evidence(**kwargs))


class TestSuccessEnvelope:
    def test_envelope_has_exactly_the_documented_keys(self, ws) -> None:
        """Three keys, no more: a caller pattern-matching on them stays valid."""
        out = _call(ws)

        assert set(out) == {"_schema_version", "count", "evidence"}
        assert out["_schema_version"] == MCP_SCHEMA_VERSION

    def test_count_is_the_length_of_the_evidence_list(self, ws) -> None:
        """``count`` is a promise about the payload, not an independent total."""
        out = _call(ws)

        assert out["count"] == 4
        assert len(out["evidence"]) == 4

    def test_each_record_carries_all_eleven_evidence_fields(self, ws) -> None:
        """Dropping a field would still parse as JSON — pin the whole shape."""
        out = _call(ws)

        for record in out["evidence"]:
            assert set(record) == _RECORD_FIELDS

    def test_records_carry_the_values_they_were_created_with(self, ws) -> None:
        """Actions serialise as their string value, metadata survives intact."""
        out = _call(ws)

        assert [r["actor"] for r in out["evidence"]] == [a for _, _, a in _SEEDED]
        assert [r["action"] for r in out["evidence"]] == [a.value for a, _, _ in _SEEDED]
        assert [r["target_block_id"] for r in out["evidence"]] == [b for _, b, _ in _SEEDED]
        assert [r["metadata"]["seq"] for r in out["evidence"]] == [0, 1, 2, 3]
        assert {r["confidence"] for r in out["evidence"]} == {0.5}

    def test_records_come_back_in_chain_order_rooted_at_genesis(self, ws) -> None:
        """The strong ordering claim: the returned list IS the ledger prefix.

        Asserting hash linkage rather than timestamps because a reorder or a
        silently de-duplicated record would survive a timestamp check and be
        caught here.
        """
        evidence = _call(ws)["evidence"]

        assert evidence[0]["previous_hash"] == _GENESIS_HASH
        for earlier, later in zip(evidence, evidence[1:]):
            assert later["previous_hash"] == earlier["evidence_hash"]

    def test_empty_chain_reports_zero_rather_than_an_error(self, tmp_path) -> None:
        """A workspace that has recorded nothing yet is a success, not a fault."""
        w = tmp_path / "fresh"
        (w / "decisions").mkdir(parents=True)

        out = _call(w)

        assert out == {"_schema_version": MCP_SCHEMA_VERSION, "count": 0, "evidence": []}
        evict_gate(str(w))


class TestFilters:
    def test_block_id_filter_returns_only_that_blocks_records(self, ws) -> None:
        out = _call(ws, block_id="DEC-20200101-001")

        assert out["count"] == 2
        assert [r["actor"] for r in out["evidence"]] == ["proposer", "applier"]
        assert {r["target_block_id"] for r in out["evidence"]} == {"DEC-20200101-001"}

    def test_unmatched_block_id_is_an_empty_result_not_an_error(self, ws) -> None:
        """An empty result is an answer here; it must not look like a fault."""
        out = _call(ws, block_id="DEC-19700101-999")

        assert out["count"] == 0
        assert out["evidence"] == []
        assert "error" not in out

    def test_action_filter_selects_across_blocks(self, ws) -> None:
        out = _call(ws, action="PROPOSE")

        assert [r["actor"] for r in out["evidence"]] == ["proposer", "second-proposer"]

    def test_action_filter_is_case_insensitive(self, ws) -> None:
        """The docstring lists the actions upper-case; the parser upper-cases first."""
        lower = _call(ws, action="propose")
        mixed = _call(ws, action="PrOpOsE")
        upper = _call(ws, action="PROPOSE")

        assert lower == upper
        assert mixed == upper

    @pytest.mark.parametrize(
        "action",
        ["PROPOSE", "APPLY", "ROLLBACK", "CONTRADICT", "DRIFT", "RESOLVE", "VERIFY"],
    )
    def test_every_documented_action_is_accepted(self, ws, action) -> None:
        """All seven names in the docstring must parse — three are unused here."""
        out = _call(ws, action=action)

        assert "error" not in out
        assert set(out) == {"_schema_version", "count", "evidence"}


class TestUnknownAction:
    def test_unknown_action_returns_a_structured_error(self, ws) -> None:
        out = _call(ws, action="DELETE")

        assert set(out) == {"_schema_version", "error"}
        assert out["_schema_version"] == MCP_SCHEMA_VERSION
        assert "Unknown action: 'DELETE'" in out["error"]

    def test_the_error_names_every_valid_action(self, ws) -> None:
        """The message is the only discoverability the caller gets — keep it whole."""
        message = _call(ws, action="DELETE")["error"]

        for member in EvidenceAction:
            assert member.value in message

    def test_a_refusal_never_looks_like_an_empty_chain(self, ws) -> None:
        """No ``count``/``evidence`` keys on the error path.

        If the refusal carried ``count: 0`` a caller that only reads ``count``
        would conclude the ledger is empty, which is the worst possible answer
        from an audit tool.
        """
        out = _call(ws, action="nonsense")

        assert "count" not in out
        assert "evidence" not in out

    def test_empty_action_is_not_unknown_it_is_absent(self, ws) -> None:
        """``action=""`` is the documented default, so it must not be rejected."""
        assert _call(ws, action="") == _call(ws)


class TestLimit:
    def test_limit_keeps_the_most_recent_records(self, ws) -> None:
        """Truncation drops the OLDEST — a ledger reader wants the tail."""
        out = _call(ws, limit=2)

        assert out["count"] == 2
        assert [r["actor"] for r in out["evidence"]] == ["second-proposer", "verifier"]

    def test_limit_above_the_chain_length_returns_everything(self, ws) -> None:
        assert _call(ws, limit=1000)["count"] == 4

    def test_limit_truncates_a_block_filtered_result(self, ws) -> None:
        out = _call(ws, block_id="DEC-20200101-001", limit=1)

        assert [r["actor"] for r in out["evidence"]] == ["applier"]

    def test_limit_zero_returns_nothing_when_unfiltered(self, ws) -> None:
        out = _call(ws, limit=0)

        assert out["count"] == 0
        assert out["evidence"] == []

    def test_limit_is_unvalidated_on_the_filtered_path_WART(self, ws) -> None:
        """PINNED WART, not endorsed behaviour — actual output, deliberately.

        The truncation is ``records[-limit:] if len(records) > limit``. With a
        filter and ``limit=0`` that evaluates ``records[-0:]``, which is the
        WHOLE list, so ``limit=0`` returns everything — the exact opposite of
        the unfiltered path one test above. ``limit=-1`` slices ``records[1:]``
        and drops from the FRONT. Neither is what "Maximum number of records to
        return" promises.

        Fixing it (clamp ``limit`` at the boundary) is right and will turn this
        test red on purpose: update it to the clamped expectations then.
        """
        assert _call(ws, block_id="DEC-20200101-001", limit=0)["count"] == 2
        assert _call(ws, action="PROPOSE", limit=0)["count"] == 2

        dropped_from_the_front = _call(ws, block_id="DEC-20200101-001", limit=-1)
        assert [r["actor"] for r in dropped_from_the_front["evidence"]] == ["applier"]


class TestFilterPrecedence:
    def test_block_id_wins_over_action(self, ws) -> None:
        """Both filters given: the block wins and the action is ignored.

        Not a bug on its own — the branch order is explicit — but it is
        undocumented, and the returned APPLY record proves the action was never
        applied as a second predicate.
        """
        out = _call(ws, block_id="DEC-20200101-001", action="VERIFY")

        assert [r["action"] for r in out["evidence"]] == ["PROPOSE", "APPLY"]

    def test_an_invalid_action_is_never_validated_when_block_id_is_set_WART(self, ws) -> None:
        """PINNED WART: the same input is a refusal alone and a success paired.

        ``action="DELETE"`` returns "Unknown action"; ``block_id=... ,
        action="DELETE"`` returns records. Validation lives inside the ``elif``,
        so a typo'd action is silently discarded whenever a block is named —
        a caller filtering on both gets a broader result than they asked for
        and no signal that their action was garbage.
        """
        assert "error" in _call(ws, action="DELETE")

        paired = _call(ws, block_id="DEC-20200101-001", action="DELETE")
        assert "error" not in paired
        assert paired["count"] == 2


class TestUninitialisedWorkspace:
    def test_a_missing_workspace_is_answered_not_refused_WART(self, tmp_path) -> None:
        """PINNED WART: no ``_check_workspace`` call on this path.

        Its module siblings gate first and return "Run: mind-mem-init" for a
        workspace that does not exist. ``list_evidence`` does not, so pointing
        it at a typo'd path reports an empty ledger — indistinguishable from a
        real workspace that has recorded nothing. Contrast asserted against
        ``verify_merkle``, which is user-scoped like this tool and does gate.
        """
        from mind_mem.mcp.tools.audit import verify_merkle

        missing = tmp_path / "never-initialised"

        with use_workspace(str(missing)):
            gated = json.loads(verify_merkle("DEC-20200101-001", "a" * 64))
            ungated = json.loads(list_evidence())

        assert gated["error"] == "Workspace not found. Run: mind-mem-init <path>"
        assert ungated == {"_schema_version": MCP_SCHEMA_VERSION, "count": 0, "evidence": []}
        evict_gate(str(missing))

    def test_reading_the_ledger_creates_the_workspace_memory_dir_WART(self, tmp_path) -> None:
        """PINNED WART: a read has a filesystem side effect.

        ``get_gate`` builds a ``GovernanceGate``, whose ``__init__`` does
        ``os.makedirs(<ws>/memory)``. So calling this tool on a path that does
        not exist CREATES it. Worth knowing before anyone treats the tool as
        safe to point at arbitrary paths.
        """
        missing = tmp_path / "conjured"
        assert not missing.exists()

        _call(missing)

        assert os.path.isdir(missing / "memory")
        evict_gate(str(missing))


class TestBackendFailure:
    def test_a_workspace_path_that_is_a_file_returns_an_error_envelope(self, tmp_path) -> None:
        """The gate raises ``NotADirectoryError``; the tool must not propagate it.

        An MCP tool that raises kills the stdio session for every other tool,
        so the contract is that any backend fault becomes a structured
        response. Same no-``count``/no-``evidence`` rule as the unknown-action
        refusal.
        """
        not_a_dir = tmp_path / "workspace.txt"
        not_a_dir.write_text("this is a file, not a workspace", encoding="utf-8")

        out = _call(not_a_dir)

        assert set(out) == {"_schema_version", "error"}
        assert out["error"].startswith("Evidence listing failed:")
        assert "count" not in out


class TestReachability:
    def test_the_tool_is_registered_on_the_audit_family(self) -> None:
        """Unregistered is the defect class this file exists to guard against."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.audit import register

        register(_Mcp())
        assert "list_evidence" in registered

    def test_the_tool_is_classified_user_scope(self) -> None:
        """Registered but unclassified is unreachable, not merely unprivileged."""
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "list_evidence" in USER_TOOLS
        assert "list_evidence" not in ADMIN_TOOLS

    def test_the_acl_lets_it_through_with_no_scope_granted(self, ws) -> None:
        """The behavioural half of the classification: reading needs no grant.

        ``_no_admin_scope`` has cleared ``MIND_MEM_SCOPE``, so this is the
        default posture an agent runs under.
        """
        assert "MIND_MEM_SCOPE" not in os.environ

        out = _call(ws)

        assert out["count"] == 4
        assert "not in ACL policy" not in json.dumps(out)
