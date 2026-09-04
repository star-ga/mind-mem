# Copyright 2026 STARGA, Inc.
"""AUD-05 — one word, four ledgers, and the two nobody walked.

This product writes four hash-linked ledgers. Until 5.0.2 the verifier
walked two of them:

===================  ==========================================  ========
ledger               file                                        walked?
===================  ==========================================  ========
hash_chain           ``memory/hash_chain_v2.db``                  yes
evidence_chain       ``memory/evidence_chain.jsonl``              yes
audit_sidecar        ``.mind-mem-audit/chain.jsonl``              **no**
served_ledger        ``.mind-mem-ledger/served.jsonl``            **no**
===================  ==========================================  ========

``served_ledger`` even shipped its own :func:`verify_served_chain`, which
no verifier called. A workspace whose sidecar or served rows had been
rewritten verified clean and said so, and "tamper-evident" was true of
half the evidence.

The gates here:

* every ledger contributes a named row to the report, always
  (:data:`LEDGER_CHECKS`);
* corrupting each ledger fails **that** row and no other, with its own
  exit code, so the report localises damage rather than going red as a
  block;
* :data:`VALID_OPERATIONS` is re-derived from the source by AST, so the
  sidecar cannot advertise a verb no door writes -- the specific defect
  being that it advertised twelve and four doors wrote three;
* a workspace is born with the governance gate armed.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib

import pytest

import mind_mem
from mind_mem.audit_chain import RETIRED_OPERATIONS, VALID_OPERATIONS, AuditChain
from mind_mem.evidence_objects import EvidenceAction, EvidenceChain
from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.init_workspace import init
from mind_mem.mm_cli import config_set
from mind_mem.recall_digests import query_hash, served_set_digest
from mind_mem.served_ledger import append_served_run, ledger_path
from mind_mem.verify_cli import (
    EXIT_AUDIT,
    EXIT_CHAIN,
    EXIT_EVIDENCE,
    EXIT_OK,
    EXIT_SERVED,
    LEDGER_CHECKS,
    NON_LEDGER_CHECKS,
    verify_workspace,
)

SRC = pathlib.Path(mind_mem.__file__).parent

PIPELINE = "b" * 64


#: NOT a constant. ``index_anchor`` is ``sha256(preimage(INDEX_ANCHOR_TAG,
#: head))`` over the chain entry a run observed, and the 5.0.2
#: ``cross_ledger`` check joins on exactly that. A hand-picked 64-hex
#: value describes a workspace no run could produce -- a served row
#: anchored to an entry that never existed -- so the fixture resolved it
#: from the live chain instead. Strengthening the fixture, not relaxing
#: the check: with the literal back in place, `cross_ledger` correctly
#: convicts the row (measured: "1 served row(s) anchor to a chain entry
#: that is gone: seq [0]", exit 11).
def _live_anchor(ws: str) -> str:
    from mind_mem.recall_attestation import _resolve_index_anchor

    return _resolve_index_anchor(ws)


INSTANT = "2026-09-01"
SERVED_IDS = ("D-20260901-201", "D-20260901-202")


# ---------------------------------------------------------------------------
# A workspace carrying all four ledgers
# ---------------------------------------------------------------------------


def _write_all_four(ws: str) -> None:
    """Populate every ledger, so a per-ledger corruption has something to break."""
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)

    HashChainV2(os.path.join(ws, "memory", "hash_chain_v2.db")).append("D-1", "create", "hello")

    EvidenceChain(store_path=os.path.join(ws, "memory", "evidence_chain.jsonl")).create(
        action=EvidenceAction.APPLY,
        actor="tester",
        target_block_id="D-1",
        target_file="decisions/DECISIONS.md",
        payload="hello",
    )

    AuditChain(ws).append("update_field", "decisions/DECISIONS.md", agent="tester")

    row = append_served_run(
        ws,
        query_hash=query_hash("why did the rollout land"),
        served_digest=served_set_digest(SERVED_IDS),
        ids=SERVED_IDS,
        pipeline_hash=PIPELINE,
        index_anchor=_live_anchor(ws),
        scoring_instant=INSTANT,
    )
    assert row is not None, "positive control: the served ledger must be enabled or the rest is vacuous"


@pytest.fixture
def four_ledger_ws(tmp_path) -> str:
    ws = str(tmp_path / "ws")
    init(ws)
    # Enable the opt-in served ledger through ``mm config set``: init
    # armed the gate against the config it wrote, and this command writes
    # and re-attests in one step, so a legitimate setting change cannot be
    # read back as tampering.
    config_set(os.path.join(ws, "mind-mem.json"), "served_ledger", {"enabled": True})

    _write_all_four(ws)
    return ws


# ---------------------------------------------------------------------------
# 1. Every ledger is walked
# ---------------------------------------------------------------------------


def test_every_ledger_contributes_a_row_to_a_populated_workspace(four_ledger_ws: str) -> None:
    report = verify_workspace(four_ledger_ws)
    for name in LEDGER_CHECKS:
        assert name in report.checks, f"{name} is not walked by verify_workspace"
    assert report.ok, report.messages
    assert report.exit_code == EXIT_OK


def test_every_ledger_contributes_a_row_to_an_empty_workspace(tmp_path) -> None:
    """The row is unconditional: a missing ledger is *reported* absent, not skipped."""
    report = verify_workspace(str(tmp_path))
    for name in LEDGER_CHECKS:
        assert name in report.checks, f"{name} vanishes from the report when its ledger is absent"
    assert ".mind-mem-audit/chain.jsonl" in report.missing
    assert "mind-mem.json:served_ledger.enabled" in report.missing


def test_no_check_function_is_left_unwired() -> None:
    """Every ``check_*`` in the module is called by ``verify_workspace``.

    This is the general form of the defect. ``served_ledger`` shipped a
    complete ``verify_served_chain`` that no verifier called: the code
    was written, correct, and unreachable, and nothing failed. A checker
    that exists but is not invoked is indistinguishable from one that
    passes, so being *defined* must not be enough.
    """
    source = (SRC / "verify_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("check_")}
    assert defined, "positive control: no check_* functions found — the scan is wrong"

    orchestrator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_workspace")
    called = {node.func.id for node in ast.walk(orchestrator) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    unwired = sorted(defined - called)
    assert not unwired, f"{unwired} defined in verify_cli but never called by verify_workspace"


def test_every_ledger_check_is_named_in_LEDGER_CHECKS(four_ledger_ws: str) -> None:
    """The declared list and the produced rows agree in both directions.

    Subset in one direction only would let a ledger row be produced that
    the declared list never mentions, which is how a fifth ledger would
    arrive unannounced.
    """
    checks = set(verify_workspace(four_ledger_ws).checks)
    assert checks - set(NON_LEDGER_CHECKS) == set(LEDGER_CHECKS), sorted(checks - set(NON_LEDGER_CHECKS))


def test_json_output_names_all_four_ledgers(four_ledger_ws: str) -> None:
    payload = json.loads(json.dumps(verify_workspace(four_ledger_ws).as_dict()))
    assert set(LEDGER_CHECKS) <= set(payload["checks"])


def test_verifying_creates_nothing(tmp_path) -> None:
    """A verifier that mkdir's the artifact it is asking about is not read-only.

    ``AuditChain.__init__`` creates ``.mind-mem-audit/``. Constructing one
    to answer "is there a sidecar?" would leave the directory behind in
    every workspace ever inspected — and the next run would find it.
    """
    ws = str(tmp_path / "empty")
    os.makedirs(ws)
    before = sorted(os.listdir(ws))
    verify_workspace(ws)
    assert sorted(os.listdir(ws)) == before, "verify created something"
    assert not os.path.exists(os.path.join(ws, ".mind-mem-audit"))


# ---------------------------------------------------------------------------
# 2. Corrupting each ledger fails exactly its own row
# ---------------------------------------------------------------------------


def _corrupt_hash_chain(ws: str) -> None:
    import sqlite3

    conn = sqlite3.connect(os.path.join(ws, "memory", "hash_chain_v2.db"))
    try:
        conn.execute("UPDATE hash_chain SET content_hash = 'deadbeef'")
        conn.commit()
    finally:
        conn.close()


def _corrupt_evidence_chain(ws: str) -> None:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows[0]["actor"] = "someone-else"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _corrupt_audit_sidecar(ws: str) -> None:
    path = os.path.join(ws, ".mind-mem-audit", "chain.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows[0]["reason"] = "tampered"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _corrupt_served_ledger(ws: str) -> None:
    path = ledger_path(ws)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows[0]["ids"] = list(rows[0]["ids"]) + ["D-20260901-999"]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


CORRUPTIONS = [
    ("hash_chain", _corrupt_hash_chain, EXIT_CHAIN),
    ("evidence_chain", _corrupt_evidence_chain, EXIT_EVIDENCE),
    ("audit_sidecar", _corrupt_audit_sidecar, EXIT_AUDIT),
    ("served_ledger", _corrupt_served_ledger, EXIT_SERVED),
]


@pytest.mark.parametrize("ledger,corrupt,exit_code", CORRUPTIONS, ids=[c[0] for c in CORRUPTIONS])
def test_corrupting_one_ledger_fails_exactly_that_row(four_ledger_ws, ledger, corrupt, exit_code) -> None:
    clean = verify_workspace(four_ledger_ws)
    assert clean.ok, f"positive control: workspace must verify clean first — {clean.messages}"
    assert clean.checks[ledger] is True

    corrupt(four_ledger_ws)

    report = verify_workspace(four_ledger_ws)
    assert report.checks[ledger] is False, f"{ledger} was corrupted and still passed"
    assert report.exit_code == exit_code
    others = {n for n in LEDGER_CHECKS if n != ledger and report.checks.get(n) is False}
    assert not others, f"corrupting {ledger} also failed {sorted(others)} — the report does not localise"


# ---------------------------------------------------------------------------
# 2b. The MCP tool publishes the SAME verdict, over the same four ledgers
# ---------------------------------------------------------------------------
#
# R2-05. ``verify_chain`` kept a private walk of two ledgers and published
# ``valid`` over it, so a tampered served ledger left the tool saying
# ``valid: true`` while ``verify_workspace`` on the same directory said the
# workspace was broken. The tool now delegates; these are the per-ledger
# positive controls that keep it delegating, reusing the corruption table
# above so a fifth ledger cannot be covered here and missed there.


def _verify_chain_tool(ws: str) -> dict:
    """Call the MCP ``verify_chain`` tool against *ws* and parse its envelope."""
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import audit

    with use_workspace(ws):
        return json.loads(audit.verify_chain.__wrapped__())


def test_the_tool_names_every_row_the_verifier_produced(four_ledger_ws: str) -> None:
    report = verify_workspace(four_ledger_ws)
    envelope = _verify_chain_tool(four_ledger_ws)

    assert envelope["valid"] is True, envelope["messages"]
    assert envelope["workspace_valid"] is report.ok is True, envelope["messages"]
    assert envelope["checks"] == report.checks, "the tool republishes a different set of rows"
    assert envelope["ledgers"] == list(LEDGER_CHECKS)
    for name in LEDGER_CHECKS:
        assert name in envelope, f"{name} has no object of its own in the tool envelope"
        assert envelope[name]["valid"] is report.checks[name]


@pytest.mark.parametrize("ledger,corrupt,exit_code", CORRUPTIONS, ids=[c[0] for c in CORRUPTIONS])
def test_tampering_any_ledger_turns_the_tool_verdict_false(four_ledger_ws, ledger, corrupt, exit_code) -> None:
    """One control per ledger: the tool's ``valid`` must follow each of the four.

    Two of these four went green over a tampered ledger before 5.0.2 --
    ``audit_sidecar`` and ``served_ledger`` -- because the tool never looked
    at them. Parametrising over the same table the verifier uses is what
    stops the pair diverging again.
    """
    clean = _verify_chain_tool(four_ledger_ws)
    assert clean["valid"] is True, f"positive control: the tool must pass first -- {clean['messages']}"
    assert clean[ledger]["valid"] is True

    corrupt(four_ledger_ws)

    after = _verify_chain_tool(four_ledger_ws)
    assert after["valid"] is False, f"{ledger} was tampered and the tool still reported valid"
    assert after["workspace_valid"] is False
    assert after[ledger]["valid"] is False
    assert after["exit_code"] == exit_code


def test_a_non_ledger_failure_is_reported_under_its_own_name(four_ledger_ws: str) -> None:
    """Spec drift is a finding, and it is not a chain break.

    ``valid`` is this tool's declared subject -- the four ledgers. The
    verifier's whole answer travels as ``workspace_valid`` beside it, with
    the failing row named in ``checks`` and a non-zero ``exit_code``, so
    the distinction costs no information. Collapsing the two would make one
    edit to ``mind-mem.json`` -- the only configuration path this product
    documents -- read as ledger tampering.
    """
    clean = _verify_chain_tool(four_ledger_ws)
    assert clean["valid"] is True and clean["workspace_valid"] is True, clean["messages"]

    config_path = os.path.join(four_ledger_ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["auto_recall"] = not config.get("auto_recall", True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)

    after = _verify_chain_tool(four_ledger_ws)
    assert after["checks"]["spec_binding"] is False, "positive control: the edit must actually drift the binding"
    assert after["workspace_valid"] is False, "the whole verdict must carry the drift"
    assert after["exit_code"] != 0
    assert after["valid"] is True, "a config edit is not a ledger break and must not be reported as one"
    assert all(after[name]["valid"] for name in LEDGER_CHECKS)


def test_the_tool_keeps_the_per_ledger_fields_its_callers_read(four_ledger_ws: str) -> None:
    """``review_evidence._chain`` reads these by name; delegating must not drop them."""
    envelope = _verify_chain_tool(four_ledger_ws)
    assert envelope["hash_chain"]["length"] == 1
    assert envelope["hash_chain"]["broken_at"] == -1
    assert envelope["evidence_chain"]["broken_ids"] == []
    assert envelope["served_ledger"]["rows_checked"] == 1


def test_the_tool_creates_nothing(tmp_path) -> None:
    """It used to build the governance gate, which writes. The verifier does not."""
    ws = str(tmp_path / "empty")
    os.makedirs(ws)
    before = sorted(os.listdir(ws))
    envelope = _verify_chain_tool(ws)
    assert "error" not in envelope, envelope
    assert sorted(os.listdir(ws)) == before, "verify_chain created something in the workspace"


def test_a_disabled_served_ledger_is_absent_not_passed(four_ledger_ws: str) -> None:
    """Off and deleted must not read alike to a machine consumer."""
    config_path = os.path.join(four_ledger_ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["served_ledger"] = {"enabled": False}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)

    report = verify_workspace(four_ledger_ws, strict=True)
    assert report.checks["served_ledger"] is True
    assert "mind-mem.json:served_ledger.enabled" in report.missing


# ---------------------------------------------------------------------------
# 3. VALID_OPERATIONS is derived from the source, not asserted by hand
# ---------------------------------------------------------------------------


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings.

    Needed because ``compliance/audit.py`` appends
    ``REDACTION_OPERATION``, not a literal. A scanner that only reads
    ``ast.Constant`` first arguments misses it and under-reports the
    written set — which would make this gate pass while the contract was
    still wrong.
    """
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.target.id] = node.value.value
    return out


def _audit_receivers(tree: ast.Module) -> set[str]:
    """Attribute/name targets bound to ``AuditChain(...)`` in this module.

    Binding receivers by assignment rather than by name shape is what
    keeps ``governance_gate``'s ``self._chain = HashChainV2(...)`` out of
    the result: the two share an attribute name and share nothing else.
    """
    receivers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "AuditChain"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                receivers.add(target.attr)
            elif isinstance(target, ast.Name):
                receivers.add(target.id)
    return receivers


def _written_operations() -> dict[str, set[str]]:
    """``operation -> {module}`` for every audit-sidecar append in ``src/``."""
    written: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "AuditChain" not in source:
            continue
        tree = ast.parse(source)
        constants = _module_string_constants(tree)
        receivers = _audit_receivers(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"):
                continue
            recv = node.func.value
            direct = isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name) and recv.func.id == "AuditChain"
            bound = (isinstance(recv, ast.Attribute) and recv.attr in receivers) or (isinstance(recv, ast.Name) and recv.id in receivers)
            if not (direct or bound):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                op = first.value
            elif isinstance(first, ast.Name) and first.id in constants:
                op = constants[first.id]
            else:
                raise AssertionError(
                    f"{path.name}: audit append with an unresolvable operation — "
                    "the scanner would under-report and this gate would pass falsely"
                )
            written.setdefault(op, set()).add(path.name)
    return written


def test_the_ast_scanner_finds_a_literal_writer() -> None:
    """Negative control: the scanner sees the ordinary case."""
    assert "field_audit.py" in _written_operations()["update_field"]


def test_the_ast_scanner_resolves_an_indirect_writer() -> None:
    """Negative control for the blind spot that would silently shrink the answer.

    ``compliance/audit.py`` appends ``REDACTION_OPERATION``. A scanner
    blind to name indirection reports it as unwritten; the gate then
    passes while a real writer is missing from the derived set.
    """
    assert "audit.py" in _written_operations()["update_field"]


def test_valid_operations_equals_what_the_source_writes() -> None:
    written = set(_written_operations())
    assert written == set(VALID_OPERATIONS), (
        f"VALID_OPERATIONS advertises {sorted(set(VALID_OPERATIONS) - written)} that no door writes "
        f"and is missing {sorted(written - set(VALID_OPERATIONS))} that some door does. "
        "The sidecar's contract must equal its writers."
    )


def test_retired_and_valid_operations_do_not_overlap() -> None:
    assert not (VALID_OPERATIONS & RETIRED_OPERATIONS)


def test_a_retired_verb_is_refused_and_says_where_the_record_lives(tmp_path) -> None:
    chain = AuditChain(str(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        chain.append("rollback", "a.md")
    message = str(excinfo.value)
    assert "retired" in message
    assert "evidence_chain" in message


def test_a_pre_5_0_2_ledger_carrying_retired_verbs_still_verifies(tmp_path) -> None:
    """Correcting the contract must not make an operator's ledger unreadable.

    Shrinking an accepted-input set is only safe if reads were never
    gated on it. They were not — :meth:`AuditChain.verify` consults
    neither set — and this is the test that keeps it that way.
    """
    ws = str(tmp_path)
    chain = AuditChain(ws)
    chain.append("create_block", "a.md", agent="tester")
    chain.append("update_field", "a.md", agent="tester")

    # Rewrite an entry's verb to a retired one *and re-seal the chain*, the
    # way a pre-5.0.2 release would have written it.
    path = os.path.join(ws, ".mind-mem-audit", "chain.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    from mind_mem.audit_chain import AuditEntry

    rows[1]["operation"] = "set_status"
    rows[1]["entry_hash"] = AuditEntry.compute_entry_hash(
        rows[1]["seq"],
        rows[1]["timestamp"],
        rows[1]["operation"],
        rows[1]["target"],
        rows[1]["agent"],
        rows[1]["reason"],
        rows[1]["payload_hash"],
        rows[1]["prev_hash"],
    )
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    ok, errors = AuditChain(ws).verify()
    assert ok, errors
    assert [e.operation for e in AuditChain(ws).entries()] == ["create_block", "set_status"]

    report = verify_workspace(ws)
    assert report.checks["audit_sidecar"] is True


# ---------------------------------------------------------------------------
# 4. The sidecar row names the admission it happened under
# ---------------------------------------------------------------------------


def test_a_row_written_outside_an_admission_scope_omits_the_link(tmp_path) -> None:
    """Absent means "not written inside an admission" — never a fabricated id."""
    entry = AuditChain(str(tmp_path)).append("create_block", "a.md")
    assert entry.admission_entry_id is None
    assert "admission_entry_id" not in entry.to_dict()


def test_a_row_written_inside_an_admission_scope_names_it(tmp_path) -> None:
    """The link is taken from the ambient scope, so no caller can forget it."""
    from mind_mem.admission import AdmissionReceipt, _open_admission
    from mind_mem.enums import IngestTier

    receipt = AdmissionReceipt(
        entry_id="entry-abc123",
        content_hash="f" * 128,
        kind="block",
        covers=("D-1",),
        chain_verified=True,
        actor="tester",
        tier=IngestTier.RESTAMP,
    )
    ws = str(tmp_path)
    with _open_admission(receipt):
        entry = AuditChain(ws).append("update_field", "a.md", agent="tester")

    assert entry.admission_entry_id == "entry-abc123"
    assert entry.to_dict()["admission_entry_id"] == "entry-abc123"
    # Survives a round trip through the ledger, and the chain still verifies.
    reread = AuditChain(ws).entries()[-1]
    assert reread.admission_entry_id == "entry-abc123"
    ok, errors = AuditChain(ws).verify()
    assert ok, errors


# ---------------------------------------------------------------------------
# 5. The gate is armed at birth, and configuration still works
# ---------------------------------------------------------------------------
#
# These two tests used to characterise a gap: ``init`` wrote no
# ``.spec_binding.json``, so a fresh workspace detected no config edit,
# and arming it had been tried and reverted because the drift response
# ignored ``governance_mode`` — one edit to a documented key stopped
# every governed write, in the shipped default mode, and hand-editing
# ``mind-mem.json`` was the only configuration path the product offered.
#
# Both halves have landed, in the only order that works: the drift
# response honours ``governance_mode`` first, ``init`` arms second, and
# ``mm config set`` writes-and-re-attests so configuring a workspace is
# no longer indistinguishable from tampering with it. What follows is the
# same three facts, asserted as the invariant rather than as the gap —
# and each keeps its opposite in view, because "armed" is only meaningful
# next to a drift that is actually caught.


def test_a_fresh_workspace_is_armed_at_birth(tmp_path) -> None:
    """``init`` binds the config it wrote, so step 1 is live from the start."""
    ws = str(tmp_path / "fresh")
    init(ws)
    assert os.path.exists(os.path.join(ws, ".spec_binding.json"))

    report = verify_workspace(ws, strict=True)
    assert report.checks["spec_binding"] is True, "init armed the workspace but strict verify does not see it"
    assert ".spec_binding.json" not in report.missing


def test_an_unarmed_workspace_still_fails_strict_verify(tmp_path) -> None:
    """The control for the test above: strict verify can still say no.

    A workspace made before 5.0.2 — or one whose binding was removed —
    is the case ``spec_binding: False`` exists for. Deleting the binding
    is how that state is reached now, and if this passed anyway the
    assertion above would be measuring nothing.
    """
    ws = str(tmp_path / "fresh")
    init(ws)
    os.remove(os.path.join(ws, ".spec_binding.json"))

    report = verify_workspace(ws, strict=True)
    assert report.checks["spec_binding"] is False, "strict verify must not call an unarmed workspace clean"
    assert ".spec_binding.json" in report.missing


def test_the_documented_config_path_keeps_a_fresh_workspace_writable(tmp_path) -> None:
    """The blocker that stopped arming at birth, gone — measured both ways.

    Setting a documented key on a freshly-armed workspace must leave it
    writable. ``mm config set`` is what makes that true: it writes and
    re-attests in one step, so the edit is a configuration change and not
    drift. The hand edit below is the positive control — same workspace,
    same key, same value — and it must still be *caught*, or this test
    would pass on a gate that stopped checking.
    """
    from mind_mem.enums import IngestTier
    from mind_mem.governance_gate import evict_gate, get_gate

    ws = str(tmp_path / "fresh")
    init(ws)
    config_path = os.path.join(ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        assert json.load(handle)["governance_mode"] == "detect_only", "the shipped default is what this test is about"

    config_set(config_path, "auto_recall", False)
    with get_gate(ws).admit_block("WRITE", "D-1", "content", tier=IngestTier.RESTAMP, actor="op"):
        pass
    chain = HashChainV2(os.path.join(ws, "memory", "hash_chain_v2.db"))
    assert chain.get_block_chain("D-1"), "the write was admitted but nothing reached the ledger"

    # Positive control: the same change, made by hand, is still drift. Under
    # the shipped default that is recorded rather than refused, so the DRIFT
    # row is what "caught" means here; `enforce` is checked below.
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["auto_recall"] = True
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    evict_gate(ws)
    with get_gate(ws).admit_block("WRITE", "D-2", "content", tier=IngestTier.RESTAMP, actor="op"):
        pass
    drift = [r for r in _evidence_records(ws) if (r.get("metadata") or {}).get("action_verb") == "DRIFT"]
    assert drift, "a hand edit left no DRIFT row; the gate is no longer watching the config"


def test_under_enforce_the_supported_path_writes_and_a_hand_edit_refuses(tmp_path) -> None:
    """The strict half, on the mode that raises rather than records.

    ``detect_only`` cannot show this: it records drift and admits anyway,
    so a broken gate and a working one look identical there. Under
    ``enforce`` they do not — the supported path must still write, and a
    hand edit must still stop the write.
    """
    from mind_mem.enums import IngestTier
    from mind_mem.governance_gate import GovernanceBypassError, evict_gate, get_gate

    ws = str(tmp_path / "strict")
    init(ws)
    config_path = os.path.join(ws, "mind-mem.json")

    config_set(config_path, "governance_mode", "enforce")
    config_set(config_path, "auto_recall", False)
    evict_gate(ws)
    with get_gate(ws).admit_block("WRITE", "D-1", "content", tier=IngestTier.RESTAMP, actor="op"):
        pass
    chain = HashChainV2(os.path.join(ws, "memory", "hash_chain_v2.db"))
    assert chain.get_block_chain("D-1"), "enforce refused a change made the supported way"

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["auto_recall"] = True
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    evict_gate(ws)
    with pytest.raises(GovernanceBypassError, match="spec-hash drifted"):
        with get_gate(ws).admit_block("WRITE", "D-2", "content", tier=IngestTier.RESTAMP, actor="op"):
            pass


def test_config_set_refuses_to_re_attest_an_unreviewed_edit(tmp_path) -> None:
    """``mm config set`` is not a way around ``mm bind --rebind``.

    If it re-attested whatever it found, any unrelated key would launder
    a hand edit into an attested config for free — which would make the
    two tests above pass while the gate protected nothing.
    """
    from mind_mem.mm_cli import ConfigSetError

    ws = str(tmp_path / "laundry")
    init(ws)
    config_path = os.path.join(ws, "mind-mem.json")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["governance_mode"] = "off"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with pytest.raises(ConfigSetError, match="already drifted"):
        config_set(config_path, "auto_recall", False)


def _evidence_records(ws: str) -> list[dict]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
