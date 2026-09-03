# Copyright 2026 STARGA, Inc.
"""R2-05 — one aggregate verifier, and the gates that keep it the only one.

The MCP tool named ``verify_chain`` walked the hash chain and the evidence
chain, then published ``valid`` over the pair. The product writes four
hash-linked ledgers. So a workspace whose served-recall ledger had been
rewritten came back ``valid: true`` from the tool and ``ok: False`` from
``mind-mem-verify`` on the same directory, and the docs called the tool's
answer "end to end".

The fix is delegation, not a third walk: the tool calls
:func:`~mind_mem.verify_cli.verify_workspace` and republishes its rows. The
behavioural controls -- tamper each of the four in turn, watch the tool's
``valid`` go false -- live in ``tests/test_ledger_hierarchy.py`` beside the
corruption table they share. What lives here are the structural gates that
stop a second aggregator being written:

* the tool's body delegates and walks nothing (AST over that one function);
* no module in the package except ``verify_cli`` walks more than one ledger
  -- one call site is a component naming its own ledger
  (``accountability_dashboard`` renders the served panel, ``replay_check``
  joins one row against an attestation), two is an aggregation, and
  aggregation has exactly one home;
* the docs rows name the ledgers the tool actually verifies, so the
  "end to end" over-claim cannot come back through prose.

Every scan here carries a positive control: a synthetic source that must be
flagged, and a real file the scanner must find instances in. A scanner that
cannot fail is not a gate.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import mind_mem
from mind_mem.verify_cli import LEDGER_CHECKS

SRC = pathlib.Path(mind_mem.__file__).parent
DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

#: Calls that walk a hash-linked ledger. ``verify_chain`` is the method on
#: both :class:`~mind_mem.hash_chain_v2.HashChainV2` and
#: :class:`~mind_mem.evidence_objects.EvidenceChain`;
#: ``verify_served_chain`` is the served ledger's own walker, which no
#: verifier called until 5.0.2.
WALKER_CALLS = frozenset({"verify_chain", "verify_served_chain"})

#: The single aggregate verifier. This is not an exemption bolted on to make
#: the scan pass -- it is the invariant. ``test_the_sole_aggregator_actually_aggregates``
#: fails if this module stops walking more than one ledger, so the entry
#: cannot rot into a blanket excuse for a module that no longer verifies.
SOLE_AGGREGATOR = "verify_cli.py"


def walker_sites(source: str) -> list[str]:
    """Every ledger-walker call site in *source*, as ``name:line`` strings."""
    sites: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in WALKER_CALLS:
            sites.append(f"{func.attr}:{node.lineno}")
        elif isinstance(func, ast.Name) and func.id in WALKER_CALLS:
            sites.append(f"{func.id}:{node.lineno}")
    return sites


def _function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found — the scan is pointed at the wrong source")


def _called_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


# The tool as it stood before 5.0.2, verbatim in shape: two ledgers walked
# privately, one ``valid`` published over the pair. It is the positive
# control for both AST gates below — a scan that does not flag this is a
# scan that would not have caught the defect.
PRE_FIX_TOOL = """
def verify_chain() -> str:
    gate = get_gate(ws)
    chain = gate.chain
    hc_valid, broken_at = chain.verify_chain()
    evidence = gate.evidence
    ev_valid, broken_ids = evidence.verify_chain()
    return json.dumps({"valid": hc_valid and ev_valid})
"""


# ---------------------------------------------------------------------------
# 1. The scanner can see what is there (positive controls)
# ---------------------------------------------------------------------------


def test_the_scanner_finds_the_walkers_that_do_exist() -> None:
    """Against the one module that legitimately walks several ledgers."""
    sites = walker_sites((SRC / SOLE_AGGREGATOR).read_text(encoding="utf-8"))
    assert len(sites) >= 3, f"scanner found {sites} in {SOLE_AGGREGATOR} — it cannot see present instances"


def test_the_scanner_flags_the_pre_fix_tool() -> None:
    assert len(walker_sites(PRE_FIX_TOOL)) == 2, walker_sites(PRE_FIX_TOOL)


def test_the_scanner_is_quiet_on_a_module_that_walks_nothing() -> None:
    assert walker_sites("def f():\n    return verify_workspace(ws)\n") == []


# ---------------------------------------------------------------------------
# 2. The tool delegates and walks nothing
# ---------------------------------------------------------------------------


def test_the_tool_delegates_to_the_single_verifier() -> None:
    source = (SRC / "mcp" / "tools" / "audit.py").read_text(encoding="utf-8")
    called = _called_names(_function(source, "verify_chain"))
    assert "verify_workspace" in called, "verify_chain no longer delegates to the single verifier"


def test_the_tool_body_walks_no_ledger_itself() -> None:
    source = (SRC / "mcp" / "tools" / "audit.py").read_text(encoding="utf-8")
    func = _function(source, "verify_chain")
    sites = walker_sites(ast.unparse(func))
    assert sites == [], f"verify_chain grew a private ledger walk again: {sites}"
    assert "get_gate" not in _called_names(func), "verify_chain builds the governance gate again — it must create nothing"


def test_the_tool_gate_would_fail_on_the_pre_fix_body() -> None:
    """Positive control for the two assertions above, on the real prior code."""
    func = _function(PRE_FIX_TOOL, "verify_chain")
    assert walker_sites(ast.unparse(func)) != []
    assert "get_gate" in _called_names(func)
    assert "verify_workspace" not in _called_names(func)


# ---------------------------------------------------------------------------
# 3. No second aggregator anywhere in the package
# ---------------------------------------------------------------------------


def _package_modules() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_the_module_scan_is_not_vacuous() -> None:
    modules = _package_modules()
    assert len(modules) > 100, f"only {len(modules)} modules scanned — the glob is wrong"
    assert SRC / SOLE_AGGREGATOR in modules


def test_only_the_sole_aggregator_walks_more_than_one_ledger() -> None:
    """One call site is a component. Two is a verdict, and verdicts have one home."""
    offenders: dict[str, list[str]] = {}
    for path in _package_modules():
        if path.name == SOLE_AGGREGATOR:
            continue
        sites = walker_sites(path.read_text(encoding="utf-8"))
        if len(sites) > 1:
            offenders[str(path.relative_to(SRC))] = sites
    assert not offenders, f"a second aggregate verifier appeared: {offenders}"


def test_the_sole_aggregator_actually_aggregates() -> None:
    """The one allowed name has to be earning it, or the allowance is a hole."""
    sites = walker_sites((SRC / SOLE_AGGREGATOR).read_text(encoding="utf-8"))
    assert len(sites) > 1, f"{SOLE_AGGREGATOR} walks {sites} — it is no longer the aggregate verifier"


def test_the_aggregation_gate_flags_a_synthetic_second_aggregator() -> None:
    """Positive control: the rule the scan applies, applied to a violation."""
    assert len(walker_sites(PRE_FIX_TOOL)) > 1


# ---------------------------------------------------------------------------
# 3b. A ledger row the verifier did not produce is not a pass
# ---------------------------------------------------------------------------


def _envelope_over(report) -> dict:
    """Run the tool against a report handed to it, bypassing disk."""
    import json
    import tempfile
    from unittest import mock

    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import audit

    with mock.patch("mind_mem.verify_cli.verify_workspace", return_value=report):
        with use_workspace(tempfile.mkdtemp(prefix="verify-chain-")):
            return json.loads(audit.verify_chain.__wrapped__())


def test_a_missing_ledger_row_does_not_count_as_verified() -> None:
    """Fail closed. ``all()`` over an empty selection is ``True``.

    The rows are unconditional today, so this cannot happen by accident --
    which is exactly why the default matters: if it ever stops being true,
    the tool must go silent-red rather than quietly pass on nothing.
    """
    from mind_mem.verify_cli import VerifyReport

    report = VerifyReport(workspace="/nowhere", ok=False)
    report.record("hash_chain", True)

    envelope = _envelope_over(report)
    assert envelope["valid"] is False, "three ledgers went unreported and the tool called it valid"


def test_the_same_report_with_every_row_present_passes() -> None:
    """Positive control: the assertion above is about the missing rows, not the mock."""
    from mind_mem.verify_cli import VerifyReport

    report = VerifyReport(workspace="/nowhere", ok=True)
    for name in LEDGER_CHECKS:
        report.record(name, True)

    assert _envelope_over(report)["valid"] is True


# ---------------------------------------------------------------------------
# 4. The docs row names what the tool verifies
# ---------------------------------------------------------------------------

DOC_FILES = ("api-reference.md", "mcp-integration.md")


def _tool_row(doc: str, tool: str) -> str:
    for line in (DOCS / doc).read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| `{tool}`"):
            return line
    return ""


@pytest.mark.parametrize("doc", DOC_FILES)
def test_the_docs_row_exists(doc: str) -> None:
    """Positive control: the row-finder finds a row, so the next test is not vacuous."""
    assert _tool_row(doc, "verify_chain"), f"no verify_chain row in docs/{doc}"


@pytest.mark.parametrize("doc", DOC_FILES)
def test_the_docs_row_names_every_ledger_verified(doc: str) -> None:
    """The over-claim was prose: "end to end" over two of four ledgers.

    A description that names the ledgers cannot quietly outgrow them —
    adding a fifth to ``LEDGER_CHECKS`` fails this until the docs say so.
    """
    row = _tool_row(doc, "verify_chain")
    unnamed = [name for name in LEDGER_CHECKS if name not in row]
    assert not unnamed, f"docs/{doc} verify_chain row does not name {unnamed}: {row}"


def test_the_docs_gate_flags_a_row_that_names_too_few() -> None:
    """Positive control, on the exact wording that shipped."""
    row = "| `verify_chain` | Verify the SHA3-512 governance hash chain end to end | — |"
    assert [name for name in LEDGER_CHECKS if name not in row] == list(LEDGER_CHECKS)
