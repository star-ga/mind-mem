# Copyright 2026 STARGA, Inc.
"""mind-mem-verify — standalone verifier (v2.0.0rc1).

Verifies the integrity of a mind-mem workspace *without opening the live
retrieval stack*. Only four on-disk artifacts are read:

1. ``memory/hash_chain_v2.db`` — the SHA3-512 append-only ledger.
2. ``memory/evidence_chain.jsonl`` — structured governance evidence.
3. ``.spec_binding.json`` — governance config hash binding.
4. ``memory/<snapshot>/manifest.json`` — optional snapshot metadata,
   may include ``merkle_root`` and ``chain_head`` anchors for the
   corresponding chain state at snapshot time.

The CLI is pure Python / stdlib. No network, no MCP, no dependency on
the recall pipeline. A successful verification returns exit code 0;
any failure maps to a specific non-zero code so wrapper scripts can
discriminate.

Exit codes:
    0  — all checks passed
    1  — generic failure (paths missing, unreadable)
    2  — hash chain integrity violation
    3  — spec-hash binding drifted or corrupted
    4  — evidence chain integrity violation
    5  — Merkle root mismatch
    6  — chain-head / snapshot anchor mismatch
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

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = ok
        if detail:
            self.messages.append(f"[{'ok' if ok else 'fail'}] {name}: {detail}")
        else:
            self.messages.append(f"[{'ok' if ok else 'fail'}] {name}")
        if not ok:
            self.ok = False

    def record_absent(self, name: str, artifact: str, detail: str, *, strict: bool) -> None:
        """Record a check whose artifact is not on disk.

        Lenient (default): passes — an empty workspace is legitimately
        clean. Strict: fails with :data:`EXIT_GENERIC` so a CI gate that
        expects a written-to workspace cannot read "nothing to verify" as
        "verified". Either way the artifact lands in :attr:`missing`.
        """
        self.missing.append(artifact)
        if strict:
            self.record(name, False, f"{detail} — required by --strict")
            if self.exit_code == EXIT_OK:
                self.exit_code = EXIT_GENERIC
        else:
            self.record(name, True, detail)

    def as_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "ok": self.ok,
            "checks": self.checks,
            "messages": self.messages,
            "exit_code": self.exit_code,
            "missing": self.missing,
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
        report.record("hash_chain", True, f"{chain.length} entries verified")
    else:
        report.record(
            "hash_chain",
            False,
            f"first broken entry at index {broken_idx}",
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
        )
        return
    chain = EvidenceChain(store_path=path)
    ok, broken = chain.verify_chain()
    if ok:
        report.record(
            "evidence_chain",
            True,
            f"{len(chain)} entries verified",
        )
    else:
        report.record(
            "evidence_chain",
            False,
            f"{len(broken)} broken entries: {broken[:3]}",
        )
        if report.exit_code == EXIT_OK:
            report.exit_code = EXIT_EVIDENCE


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

    ``strict`` turns every absent artifact into a failure. Without it a
    workspace whose ledgers were deleted verifies exactly like a fresh
    one: the messages disclose the state, but ``ok`` / ``exit_code`` —
    the layer a CI gate reads — cannot tell the two apart.
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
