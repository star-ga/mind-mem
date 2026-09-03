# Copyright 2026 STARGA, Inc.
"""mind-mem-verify — standalone verifier (v2.0.0rc1).

Verifies the integrity of a mind-mem workspace *without opening the live
retrieval stack*.

**Every hash-linked ledger this product writes is walked here.** That is
the point of the module and the property to preserve when adding one:
until 5.0.2 it walked two of the four, so a workspace whose field-audit
sidecar or served-recall ledger had been rewritten verified clean and
said so. A ledger outside this verifier is a ledger nobody checks.

**And it is the only aggregate verifier.** The MCP ``verify_chain`` tool
kept a private walk of two ledgers and published ``valid`` over it, so a
tampered served ledger was reported valid by the tool while this module
reported the workspace broken -- a two-of-four verdict presented as
whole. That walk is gone: the tool calls :func:`verify_workspace` and
republishes its rows. A module may consume ONE ledger for its own
feature (``accountability_dashboard`` renders the served ledger's panel,
``replay_check`` joins one row against an attestation) and must name
which; combining ledgers into a single ``valid`` happens here and
nowhere else. ``tests/test_mcp_audit_verify_chain.py`` fails the build on
a second aggregator.

The four ledgers, each of which contributes exactly one named row to the
report:

1. ``hash_chain`` — ``memory/hash_chain_v2.db``, the SHA3-512
   append-only chain of record (what the gate admitted).
2. ``evidence_chain`` — ``memory/evidence_chain.jsonl``, structured
   governance evidence (why it was admitted).
3. ``audit_sidecar`` — ``.mind-mem-audit/chain.jsonl``, the
   field-granular sidecar (see :mod:`mind_mem.audit_chain`).
4. ``served_ledger`` — ``.mind-mem-ledger/served.jsonl``, what recall
   actually served. **On by default since 5.0.2**; recorded absent when
   the workspace has explicitly opted out with a literal
   ``served_ledger.enabled: false``, so "opted out" and "deleted" do not
   read alike.

Plus three non-ledger checks: ``spec_binding`` (governance config hash
binding), ``open_scopes`` (write scopes that opened and never landed —
:func:`~mind_mem.governance_gate.unclosed_write_scopes`, which shipped
with no caller in any verifier), and ``merkle_root`` / ``chain_head``
from an optional
``memory/<snapshot>/manifest.json``.

The CLI is pure Python / stdlib. No network, no MCP, no dependency on
the recall pipeline, and **no writes**: every check probes for its
artifact with :func:`os.path.isfile` before constructing any reader, so
verifying a workspace never creates the directory a reader would have
made. A successful verification returns exit code 0; any failure maps to
a specific non-zero code so wrapper scripts can discriminate.

Exit codes:
    0  — all checks passed
    1  — generic failure (paths missing, unreadable)
    2  — hash chain integrity violation
    3  — spec-hash binding drifted or corrupted
    4  — evidence chain integrity violation
    5  — Merkle root mismatch
    6  — chain-head / snapshot anchor mismatch
    7  — field-audit sidecar integrity violation
    8  — served-recall ledger integrity violation
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Optional

from .evidence_objects import EvidenceChain
from .hash_chain_v2 import HashChainV2
from .merkle_tree import MerkleTree
from .spec_binding import SpecBindingCorruptedError, SpecBindingManager

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


EXIT_OK: int = 0
EXIT_GENERIC: int = 1
EXIT_CHAIN: int = 2
EXIT_SPEC: int = 3
EXIT_EVIDENCE: int = 4
EXIT_MERKLE: int = 5
EXIT_SNAPSHOT: int = 6
EXIT_AUDIT: int = 7
EXIT_SERVED: int = 8
#: A write scope opened and no close record for it was ever written.
#: Its own code, not folded into ``EXIT_EVIDENCE``: a broken evidence
#: chain means the ledger cannot be trusted, an open scope means the
#: ledger is intact and says a write did not finish. A CI gate that
#: treats those identically loses the distinction that makes the second
#: one actionable.
EXIT_SCOPE: int = 9

#: Every ledger row :func:`verify_workspace` is required to produce. The
#: gate in ``tests/test_ledger_hierarchy.py`` asserts the report carries
#: exactly these, so adding a ledger without walking it fails the build
#: rather than shipping an unchecked chain.
LEDGER_CHECKS: tuple[str, ...] = (
    "hash_chain",
    "evidence_chain",
    "audit_sidecar",
    "served_ledger",
)

#: Every row :func:`verify_workspace` can produce that is NOT a ledger
#: walk. Declared here, beside :data:`LEDGER_CHECKS`, because the two are
#: one decision: the gate in ``tests/test_ledger_hierarchy.py`` asserts
#: that the report's rows minus these equals ``LEDGER_CHECKS`` exactly,
#: so a fifth ledger cannot arrive unannounced. That gate held the
#: non-ledger names as a LITERAL, which made adding any non-ledger check
#: -- ``open_scopes``, say -- fail as though a ledger had been smuggled
#: in. Same names, one owner: adding a check now means adding it to
#: exactly one of these two tuples, and choosing which one is the
#: declaration being made.
NON_LEDGER_CHECKS: tuple[str, ...] = (
    "workspace",
    "spec_binding",
    "open_scopes",
    "snapshot_anchor",
    "merkle_root",
    "chain_head",
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class VerifyReport:
    """Structured verification result. Serialised to JSON on demand."""

    workspace: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    exit_code: int = EXIT_OK
    #: Artifacts a check looked for and did not find. Always populated,
    #: independently of ``ok`` — a lenient run stays green, but a machine
    #: consumer can still see that nothing was actually verified.
    missing: list[str] = field(default_factory=list)
    #: Per-check structured facts (chain length, broken index, row
    #: counts). ``messages`` says the same in prose; a machine consumer
    #: reading a number out of a sentence is how a second parser of our
    #: own output gets written, so the numbers travel typed as well.
    details: dict[str, dict] = field(default_factory=dict)

    def record(self, name: str, ok: bool, detail: str = "", *, details: Optional[dict] = None) -> None:
        self.checks[name] = ok
        if details:
            self.details[name] = dict(details)
        if detail:
            self.messages.append(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
        else:
            self.messages.append(f"[{'ok' if ok else 'fail'}] {name}")
        if not ok:
            self.ok = False

    def record_absent(
        self,
        name: str,
        artifact: str,
        detail: str,
        *,
        strict: bool,
        details: Optional[dict] = None,
    ) -> None:
        """Record a check whose artifact is not on disk.

        Lenient (default): passes — an empty workspace is legitimately
        clean. Strict: fails with :data:`EXIT_GENERIC` so a CI gate that
        expects a written-to workspace cannot read "nothing to verify" as
        "verified". Either way the artifact lands in :attr:`missing`.
        """
        self.missing.append(artifact)
        if strict:
            self.record(name, False, f"{detail} — required by --strict", details=details)
            if self.exit_code == EXIT_OK:
                self.exit_code = EXIT_GENERIC
        else:
            self.record(name, True, detail, details=details)

    def as_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "ok": self.ok,
            "checks": self.checks,
            "messages": self.messages,
            "exit_code": self.exit_code,
            "missing": self.missing,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_hash_chain(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Walk the hash chain and confirm every entry's self-hash + linkage.

    Opens the ledger read-only so auditing never mutates the schema, even
    on databases predating the current ``HashChainV2`` layout.
    """
    db_path = os.path.join(workspace, "memory", "hash_chain_v2.db")
    if not os.path.isfile(db_path):
        # An empty workspace with no writes yet is still valid — unless
        # the caller asked for strict, in which case a deleted ledger
        # must not read the same as one that was never written.
        report.record_absent(
            "hash_chain",
            "memory/hash_chain_v2.db",
            "no ledger present (empty workspace)",
            strict=strict,
            details={"length": 0, "broken_at": -1},
        )
        return
    try:
        chain = HashChainV2.open_readonly(db_path)
        ok, broken_idx = chain.verify_chain()
    except (sqlite3.DatabaseError, OSError) as exc:
        report.record("hash_chain", False, f"cannot read ledger: {exc}")
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_CHAIN
        return
    if ok:
        report.record(
            "hash_chain",
            True,
            f"{chain.length} entries verified",
            details={"length": chain.length, "broken_at": broken_idx},
        )
    else:
        report.record(
            "hash_chain",
            False,
            f"first broken entry at index {broken_idx}",
            details={"length": chain.length, "broken_at": broken_idx},
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_CHAIN


def check_spec_binding(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Confirm the governance spec hash matches the stored binding."""
    config_path = os.path.join(workspace, "mind-mem.json")
    if not os.path.isfile(config_path):
        report.record_absent("spec_binding", "mind-mem.json", "no config present", strict=strict)
        return
    mgr = SpecBindingManager(config_path)
    binding_path = os.path.join(workspace, ".spec_binding.json")
    if not os.path.isfile(binding_path):
        report.record_absent(
            "spec_binding",
            ".spec_binding.json",
            "no binding — not yet attested (optional)",
            strict=strict,
        )
        return
    try:
        valid, reason = mgr.verify()
    except SpecBindingCorruptedError as exc:
        report.record("spec_binding", False, f"binding corrupted: {exc}")
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_SPEC
        return
    if valid:
        report.record("spec_binding", True, reason)
    else:
        report.record("spec_binding", False, reason)
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_SPEC


def check_evidence_chain(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Load the evidence JSONL and check every entry + linkage."""
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        report.record_absent(
            "evidence_chain",
            "memory/evidence_chain.jsonl",
            "no evidence ledger present",
            strict=strict,
            details={"entries": 0, "broken_ids": []},
        )
        return
    chain = EvidenceChain(store_path=path)
    ok, broken = chain.verify_chain()
    if ok:
        report.record(
            "evidence_chain",
            True,
            f"{len(chain)} entries verified",
            details={"entries": len(chain), "broken_ids": broken},
        )
    else:
        report.record(
            "evidence_chain",
            False,
            f"{len(broken)} broken entries: {broken[:3]}",
            details={"entries": len(chain), "broken_ids": broken},
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_EVIDENCE


def check_open_scopes(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Report write scopes that opened and never closed.

    :func:`~mind_mem.governance_gate.unclosed_write_scopes` shipped
    complete, tested, and reachable from nothing: no verifier called it,
    so "opened, not landed" was a condition the product could detect and
    never did. That is the same defect ``served_ledger``'s
    ``verify_served_chain`` had, one layer up — a checker that exists but
    is not invoked is indistinguishable from one that does not exist.

    Wired here rather than given a new surface, because this is where the
    reachable verifiers already are: ``mind-mem-verify``, the ``mm``
    accountability paths, and the ``verify_chain`` MCP tool all funnel
    through :func:`verify_workspace`.

    An open scope does not mean the chain is corrupt — it means a write
    began and the ledger never recorded it finishing, so the open row's
    claim cannot be trusted. Hence :data:`EXIT_SCOPE` rather than
    :data:`EXIT_EVIDENCE`: "the ledger is broken" and "the ledger is
    intact and says a write died" call for different responses.

    Reads the same JSONL as :func:`check_evidence_chain` and, like it,
    probes with :func:`os.path.isfile` first: :class:`EvidenceChain`
    creates nothing, but building one to answer "is there a ledger?"
    would still be asking the question by doing the thing.
    """
    path = os.path.join(workspace, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        report.record_absent(
            "open_scopes",
            "memory/evidence_chain.jsonl",
            "no evidence ledger present, so no scope can be open",
            strict=strict,
            details={"open_scopes": 0, "scope_ids": []},
        )
        return

    from .governance_gate import unclosed_write_scopes

    chain = EvidenceChain(store_path=path)
    open_rows = unclosed_write_scopes(chain)
    scope_ids = [row.evidence_id for row in open_rows]
    if not open_rows:
        report.record(
            "open_scopes",
            True,
            "no write scope left open",
            details={"open_scopes": 0, "scope_ids": []},
        )
        return
    report.record(
        "open_scopes",
        False,
        f"{len(open_rows)} write scope(s) opened and never landed: {scope_ids[:3]}",
        details={"open_scopes": len(open_rows), "scope_ids": scope_ids},
    )
    if report.exit_code == EXIT_OK:
        report.exit_code = EXIT_SCOPE


def check_audit_sidecar(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Verify the field-audit sidecar (:mod:`mind_mem.audit_chain`).

    The ledger is probed with :func:`os.path.isfile` before
    :class:`~mind_mem.audit_chain.AuditChain` is constructed, and that
    ordering is load-bearing rather than stylistic: the constructor
    ``makedirs`` its own directory, so building one to answer "is there a
    sidecar?" would create the artifact the answer is about, and a
    verifier would leave ``.mind-mem-audit/`` behind in every workspace
    it inspected.
    """
    path = os.path.join(workspace, ".mind-mem-audit", "chain.jsonl")
    if not os.path.isfile(path):
        report.record_absent(
            "audit_sidecar",
            ".mind-mem-audit/chain.jsonl",
            "no field-audit sidecar present",
            strict=strict,
            details={"entries": 0, "errors": []},
        )
        return

    from .audit_chain import AuditChain

    chain = AuditChain(workspace)
    ok, errors = chain.verify()
    if ok:
        report.record(
            "audit_sidecar",
            True,
            f"{chain.entry_count()} entries verified",
            details={"entries": chain.entry_count(), "errors": []},
        )
        return
    report.record(
        "audit_sidecar",
        False,
        f"{len(errors)} broken entries: {errors[:3]}",
        details={"entries": chain.entry_count(), "errors": [str(e) for e in errors]},
    )
    if report.exit_code == EXIT_OK:
        report.exit_code = EXIT_AUDIT


def check_served_ledger(workspace: str, report: VerifyReport, *, strict: bool = False) -> None:
    """Verify the served-recall ledger (:mod:`mind_mem.served_ledger`).

    The ledger is on unless the workspace opted out. When
    :func:`~mind_mem.served_ledger.ledger_enabled` says off — a literal
    ``served_ledger.enabled: false``, or no readable ``mind-mem.json`` at
    all — the row is recorded *absent* rather than passed silently, so an
    opted-out ledger and a deleted one do not present identically to the
    machine consumer reading ``missing``. Otherwise the chain is walked by
    its own :func:`~mind_mem.served_ledger.verify_served_chain`, which
    until 5.0.2 no verifier called.
    """
    from .served_ledger import ledger_enabled, ledger_path, verify_served_chain

    if not ledger_enabled(workspace):
        report.record_absent(
            "served_ledger",
            "mind-mem.json:served_ledger.enabled",
            "served-recall ledger opted out (default is on)",
            strict=False,
            details={"enabled": False, "rows_checked": 0, "bad_seq": -1},
        )
        return
    if not os.path.isfile(ledger_path(workspace)):
        report.record_absent(
            "served_ledger",
            ".mind-mem-ledger/served.jsonl",
            "ledger enabled but no rows written yet",
            strict=strict,
            details={"enabled": True, "rows_checked": 0, "bad_seq": -1},
        )
        return

    verdict = verify_served_chain(workspace)
    if verdict.ok:
        report.record(
            "served_ledger",
            True,
            f"{verdict.rows_checked} rows verified",
            details={"enabled": True, "rows_checked": verdict.rows_checked, "bad_seq": verdict.bad_seq},
        )
        return
    report.record(
        "served_ledger",
        False,
        f"row {verdict.bad_seq}: {verdict.reason}",
        details={
            "enabled": True,
            "rows_checked": verdict.rows_checked,
            "bad_seq": verdict.bad_seq,
            "reason": verdict.reason,
        },
    )
    if report.exit_code == EXIT_OK:
        report.exit_code = EXIT_SERVED


def check_snapshot(
    workspace: str,
    report: VerifyReport,
    snapshot: Optional[str] = None,
) -> None:
    """Verify that a snapshot's chain-head + Merkle-root anchors still hold.

    The ``snapshot`` arg is anchored against ``workspace`` so a caller
    cannot point the verifier at an external directory via ``../``
    segments. A traversal attempt is reported as an explicit failure
    instead of silently reading the other directory's manifest.
    """
    if snapshot is None:
        report.record("snapshot_anchor", True, "no snapshot requested")
        return
    ws_root = os.path.realpath(workspace)
    manifest_path = os.path.realpath(os.path.join(workspace, snapshot, "manifest.json"))
    if not (manifest_path == os.path.join(ws_root, os.path.relpath(manifest_path, ws_root)) and manifest_path.startswith(ws_root + os.sep)):
        report.record(
            "snapshot_anchor",
            False,
            f"snapshot path escapes workspace: {snapshot!r}",
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_SNAPSHOT
        return
    if not os.path.isfile(manifest_path):
        report.record(
            "snapshot_anchor",
            False,
            f"snapshot manifest missing: {manifest_path}",
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_SNAPSHOT
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        report.record("snapshot_anchor", False, f"cannot read manifest: {exc}")
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_SNAPSHOT
        return

    chain_head = manifest.get("chain_head")
    if chain_head:
        db_path = os.path.join(workspace, "memory", "hash_chain_v2.db")
        if not os.path.isfile(db_path):
            report.record(
                "snapshot_anchor",
                False,
                "manifest references chain_head but ledger is missing",
            )
            if report.exit_code == EXIT_OK:
                report.exit_code = EXIT_SNAPSHOT
            return
        try:
            latest = HashChainV2.open_readonly(db_path).get_latest(n=1)
        except (sqlite3.DatabaseError, OSError) as exc:
            report.record("snapshot_anchor", False, f"cannot read ledger: {exc}")
            if report.exit_code == EXIT_OK:
                report.exit_code = EXIT_SNAPSHOT
            return
        if not latest or latest[-1].entry_hash != chain_head:
            report.record(
                "snapshot_anchor",
                False,
                f"chain_head mismatch: manifest={chain_head[:16]}…",
            )
            if report.exit_code == EXIT_OK:
                report.exit_code = EXIT_SNAPSHOT
            return
        report.record("snapshot_anchor", True, "chain_head matches")

    merkle_root = manifest.get("merkle_root")
    merkle_leaves = manifest.get("merkle_leaves")
    # Signal the obvious corruption case: exactly one of the Merkle
    # anchors is present. A snapshot that anchored once must keep both
    # or neither — a lone root / lone leaf list means something dropped.
    if bool(merkle_root) != bool(merkle_leaves):
        report.record(
            "merkle_root",
            False,
            "manifest specifies one Merkle anchor but not the other",
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_MERKLE
        return

    if merkle_root and merkle_leaves:
        leaves = [(entry["block_id"], entry["content_hash"]) for entry in merkle_leaves]
        tree = MerkleTree()
        tree.build(leaves)
        if tree.root_hash == merkle_root:
            report.record(
                "merkle_root",
                True,
                f"root matches ({len(leaves)} leaves)",
            )
        else:
            report.record(
                "merkle_root",
                False,
                f"expected {merkle_root[:16]}… got {tree.root_hash[:16]}…",
            )
            if report.exit_code == EXIT_OK:
                report.exit_code = EXIT_MERKLE


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def verify_workspace(
    workspace: str,
    *,
    snapshot: Optional[str] = None,
    strict: bool = False,
) -> VerifyReport:
    """Run every verification check against *workspace* and return a report.

    Every ledger in :data:`LEDGER_CHECKS` contributes exactly one row,
    always, so ``--json`` names all four whatever the workspace holds.
    ``open_scopes`` is not a ledger and is not in that tuple — it is a
    question asked OF the evidence ledger — but it is unconditional for
    the same reason: a row that appears only when there is something to
    say cannot be distinguished from a check that did not run.

    ``strict`` turns every absent artifact into a failure. Without it a
    workspace whose ledgers were deleted verifies exactly like a fresh
    one: the messages disclose the state, but ``ok`` / ``exit_code`` —
    the layer a CI gate reads — cannot tell the two apart.

    The one deliberate exception is ``served_ledger`` when the workspace
    has opted out: a ledger that was told not to record has no rows to
    lose, so ``--strict`` must not fail on it. Its artifact still lands
    in ``missing``, naming the config key rather than a file, so the
    reason it is absent is legible instead of inferred.
    """
    workspace = os.path.realpath(workspace)
    report = VerifyReport(workspace=workspace, ok=True)

    if not os.path.isdir(workspace):
        report.record("workspace", False, f"not a directory: {workspace}")
        report.exit_code = EXIT_GENERIC
        return report

    check_hash_chain(workspace, report, strict=strict)
    check_spec_binding(workspace, report, strict=strict)
    check_evidence_chain(workspace, report, strict=strict)
    check_open_scopes(workspace, report, strict=strict)
    check_audit_sidecar(workspace, report, strict=strict)
    check_served_ledger(workspace, report, strict=strict)
    check_snapshot(workspace, report, snapshot=snapshot)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mind-mem-verify",
        description=(
            "Standalone integrity verifier for a mind-mem workspace. "
            "Reads ledgers and manifests only — no network, no DB writes, "
            "no dependency on the live retrieval pipeline."
        ),
    )
    parser.add_argument(
        "workspace",
        help="Path to a mind-mem workspace directory.",
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        default=None,
        help=("Optional relative path to a snapshot directory whose manifest will be verified against the live chain and Merkle leaves."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of human-readable output.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (exit 1) when an expected artifact is absent — the hash chain, "
            "the evidence chain or the spec binding. Use this in a CI gate: without "
            "it a workspace whose ledgers were deleted verifies green."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = verify_workspace(args.workspace, snapshot=args.snapshot, strict=args.strict)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for line in report.messages:
            print(line)
        if report.missing and not args.strict:
            print()
            print(f"note: {len(report.missing)} artifact(s) absent, not verified: {', '.join(report.missing)}")
            print("      run with --strict to treat an absent artifact as a failure")
        print()
        print("OK" if report.ok else f"FAIL (exit={report.exit_code})")

    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
