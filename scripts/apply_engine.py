#!/usr/bin/env python3
"""Mind Mem Apply Engine v1.0 — Atomic proposal application with rollback.

Reads proposals from intelligence/proposed/, validates, executes Ops,
runs post-checks, and rolls back on failure.

Usage:
    python3 maintenance/apply_engine.py <ProposalId> [workspace_path]
    python3 maintenance/apply_engine.py P-20260213-002
    python3 maintenance/apply_engine.py P-20260213-002 --dry-run
    python3 maintenance/apply_engine.py --rollback <ReceiptTS>

Exit codes: 0 = applied, 1 = failed (rolled back), 2 = validation error
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

# Import block parser from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backup_restore import WAL
from block_parser import get_by_id, parse_file
from filelock import FileLock
from namespaces import NamespaceManager

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

VALID_OPS = {
    "append_block", "insert_after_block", "update_field",
    "append_list_item", "replace_range", "set_status",
    "supersede_decision"
}

VALID_RISKS = {"low", "medium", "high"}
VALID_STATUSES = {"staged", "applied", "rejected", "deferred", "expired", "rolled_back"}
VALID_TYPES = {"decision", "task", "edit"}

PROPOSED_FILES = [
    "intelligence/proposed/DECISIONS_PROPOSED.md",
    "intelligence/proposed/TASKS_PROPOSED.md",
    "intelligence/proposed/EDITS_PROPOSED.md",
]

# Files to snapshot for rollback (intelligence/applied excluded to prevent recursive nesting)
SNAPSHOT_DIRS = [
    "decisions", "tasks", "entities", "summaries", "memory"
]
SNAPSHOT_FILES = ["AGENTS.md", "MEMORY.md", "IDENTITY.md", "mind-mem.json"]


def _list_workspace_files(ws):
    """List all files in workspace (relative paths) for orphan detection."""
    result = set()
    for d in SNAPSHOT_DIRS:
        dirpath = os.path.join(ws, d)
        if os.path.isdir(dirpath):
            for root, dirs, files in os.walk(dirpath):
                for f in files:
                    result.add(os.path.relpath(os.path.join(root, f), ws))
    for f in SNAPSHOT_FILES:
        if os.path.isfile(os.path.join(ws, f)):
            result.add(f)
    # Intelligence files (excluding applied/)
    intel_dir = os.path.join(ws, "intelligence")
    if os.path.isdir(intel_dir):
        for root, dirs, files in os.walk(intel_dir):
            if "applied" in root.split(os.sep):
                continue
            for f in files:
                result.add(os.path.relpath(os.path.join(root, f), ws))
    return result


def _cleanup_orphan_files(ws, pre_apply_files):
    """Delete files created during a failed transaction (orphan cleanup)."""
    current_files = _list_workspace_files(ws)
    orphans = current_files - pre_apply_files
    for orphan in orphans:
        path = os.path.join(ws, orphan)
        if os.path.isfile(path):
            os.remove(path)
            print(f"  Cleaned orphan: {orphan}")


# ═══════════════════════════════════════════════
# Path Safety
# ═══════════════════════════════════════════════

def _safe_resolve(ws, rel_path):
    """Resolve rel_path within ws, rejecting traversal and symlink escapes.

    Returns the resolved absolute path.
    Raises ValueError if the path escapes the workspace.
    """
    ws_real = os.path.realpath(ws)
    joined = os.path.join(ws, rel_path)
    resolved = os.path.realpath(joined)
    if not resolved.startswith(ws_real + os.sep) and resolved != ws_real:
        raise ValueError(f"Path escapes workspace: {rel_path}")
    return resolved


# ═══════════════════════════════════════════════
# Proposal Discovery & Validation
# ═══════════════════════════════════════════════

def find_proposal(ws, proposal_id):
    """Find a proposal block by ProposalId across all proposed files."""
    for pfile in PROPOSED_FILES:
        path = os.path.join(ws, pfile)
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for b in blocks:
            if b.get("ProposalId") == proposal_id or b.get("_id") == proposal_id:
                return b, path
    return None, None


def validate_proposal(proposal):
    """Validate a proposal block before execution. Returns list of errors."""
    errors = []

    # Required fields
    for field in ("ProposalId", "Type", "TargetBlock", "Risk", "Status", "Evidence", "Rollback", "Fingerprint"):
        if not proposal.get(field):
            errors.append(f"Missing required field: {field}")

    # Enum checks
    if proposal.get("Risk") not in VALID_RISKS:
        errors.append(f"Invalid Risk: {proposal.get('Risk')} (must be {VALID_RISKS})")
    if proposal.get("Type") not in VALID_TYPES:
        errors.append(f"Invalid Type: {proposal.get('Type')} (must be {VALID_TYPES})")
    if proposal.get("Status") != "staged":
        errors.append(f"Status must be 'staged' to apply (got '{proposal.get('Status')}')")

    # Evidence must be non-empty
    evidence = proposal.get("Evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [e for e in evidence if isinstance(e, str) and e.strip()]
    if not evidence:
        errors.append("Evidence is empty")

    # Ops validation
    ops = proposal.get("Ops", [])
    if not ops:
        errors.append("No Ops defined")
    for i, op in enumerate(ops):
        op_type = op.get("op")
        if op_type not in VALID_OPS:
            errors.append(f"Ops[{i}]: invalid op type '{op_type}'")
        if not op.get("file"):
            errors.append(f"Ops[{i}]: missing 'file'")
        if op_type in ("update_field", "append_list_item", "set_status", "replace_range",
                        "insert_after_block", "supersede_decision") and not op.get("target"):
            errors.append(f"Ops[{i}]: op '{op_type}' requires 'target'")

    # Reject paths with traversal components
    for i, op in enumerate(ops):
        f = op.get("file", "")
        if ".." in f.split(os.sep) or ".." in f.split("/") or os.path.isabs(f) or f.startswith("/"):
            errors.append(f"Ops[{i}]: path '{f}' contains traversal or is absolute")

    # FilesTouched must match Ops files
    files_touched = set(proposal.get("FilesTouched", []))
    ops_files = set(op.get("file", "") for op in ops)
    if files_touched and ops_files and not ops_files.issubset(files_touched):
        errors.append(f"Ops reference files not in FilesTouched: {ops_files - files_touched}")

    # Fingerprint integrity: recompute and verify against stored value
    stored_fp = proposal.get("Fingerprint", "")
    if stored_fp and ops:
        computed_fp = compute_fingerprint(proposal)
        if computed_fp != stored_fp:
            errors.append(f"Fingerprint mismatch: stored={stored_fp}, computed={computed_fp} (proposal may have been tampered)")

    return errors


# ═══════════════════════════════════════════════
# Precondition Checks
# ═══════════════════════════════════════════════

def check_preconditions(ws):
    """Run validate.sh and intel_scan.py. Returns (ok, report)."""
    report = []
    # Use scripts from our own installation directory, not from the workspace
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    # P2: validate.sh (from installation, not workspace)
    validate_sh = os.path.join(_script_dir, "validate.sh")
    if not os.path.isfile(validate_sh):
        validate_sh = os.path.join(ws, "maintenance/validate.sh")
    try:
        result = subprocess.run(
            ["bash", validate_sh, ws],
            capture_output=True, text=True, timeout=60
        )
        # Find the TOTAL line (contains "issues")
        total_line = ""
        for line in result.stdout.strip().split("\n"):
            if "issues" in line and "TOTAL" in line:
                total_line = line.strip()
                break
        if re.search(r"\b0 issues\b", total_line):
            report.append(f"validate: PASS ({total_line})")
        else:
            report.append(f"validate: FAIL ({total_line or 'no TOTAL line found'})")
            return False, report
    except Exception as e:
        report.append(f"validate: ERROR ({e})")
        return False, report

    # P3: intel_scan.py (from installation, not workspace)
    intel_scan = os.path.join(_script_dir, "intel_scan.py")
    if not os.path.isfile(intel_scan):
        intel_scan = os.path.join(ws, "maintenance/intel_scan.py")
    try:
        result = subprocess.run(
            ["python3", intel_scan, ws],
            capture_output=True, text=True, timeout=60
        )
        # Find the TOTAL line
        total_line = ""
        for line in result.stdout.strip().split("\n"):
            if "critical" in line and "TOTAL" in line:
                total_line = line.strip()
                break
        if "0 critical" in total_line:
            report.append(f"intel_scan: PASS ({total_line})")
        else:
            report.append(f"intel_scan: FAIL ({total_line or 'no TOTAL line found'})")
            return False, report
    except Exception as e:
        report.append(f"intel_scan: ERROR ({e})")
        return False, report

    return True, report


# ═══════════════════════════════════════════════
# Snapshot & Rollback
# ═══════════════════════════════════════════════

def _safe_copy(src, dst):
    """Copy a file for snapshot purposes. Always uses copy2 (not hardlinks).

    Hardlinks are unsuitable for mutable-file snapshots because Python's
    open("w") truncates the inode in-place, corrupting both the workspace
    file and its hardlinked snapshot copy. A true copy ensures the snapshot
    preserves the original content after the workspace file is modified.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def create_snapshot(ws, ts, files_touched=None):
    """Create a pre-apply snapshot for rollback.

    When files_touched is provided (list of relative paths from the proposal's
    FilesTouched field), only those files are snapshotted — O(touched) instead
    of O(workspace). Falls back to full snapshot when files_touched is empty
    or None.

    The intelligence/applied/ directory is always excluded to prevent recursive
    nesting (snapshots containing snapshots).
    """
    snap_dir = os.path.join(ws, "intelligence/applied", ts)
    os.makedirs(snap_dir, exist_ok=True)

    if files_touched:
        # Minimal snapshot: only snapshot files the proposal will modify
        for rel_path in files_touched:
            resolved = os.path.normpath(os.path.join(ws, rel_path))
            if not resolved.startswith(ws + os.sep) and resolved != ws:
                continue  # skip paths that escape workspace
            src = resolved
            if os.path.isfile(src):
                _safe_copy(src, os.path.join(snap_dir, rel_path))
        # Always snapshot config files (needed for rollback integrity)
        for f in SNAPSHOT_FILES:
            src = os.path.join(ws, f)
            if os.path.isfile(src):
                _safe_copy(src, os.path.join(snap_dir, f))
    else:
        # Full snapshot: copy all directories
        for d in SNAPSHOT_DIRS:
            src = os.path.join(ws, d)
            dst = os.path.join(snap_dir, d)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        # Copy intelligence files individually (NOT recursively) to avoid
        # snapshotting intelligence/applied/ into itself
        intel_src = os.path.join(ws, "intelligence")
        intel_dst = os.path.join(snap_dir, "intelligence")
        os.makedirs(intel_dst, exist_ok=True)
        if os.path.isdir(intel_src):
            for item in os.listdir(intel_src):
                src_path = os.path.join(intel_src, item)
                dst_path = os.path.join(intel_dst, item)
                if item == "applied":
                    continue  # Skip to prevent recursive nesting
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        for f in SNAPSHOT_FILES:
            src = os.path.join(ws, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snap_dir, f))

    return snap_dir


def restore_snapshot(ws, snap_dir):
    """Restore workspace from snapshot using atomic temp+rename pattern.

    For each directory: copy snapshot to temp dir, delete original, rename temp.
    This ensures that if a crash occurs, the temp copy survives as a recovery point.
    Restores intelligence files individually (skipping intelligence/applied/
    to prevent deleting the snapshot itself).
    """
    for d in SNAPSHOT_DIRS:
        src = os.path.join(snap_dir, d)
        dst = os.path.join(ws, d)
        if os.path.isdir(src):
            # Atomic pattern: copy to temp first, then swap
            tmp_dst = dst + ".rollback_tmp"
            if os.path.islink(tmp_dst):
                os.unlink(tmp_dst)
            elif os.path.isdir(tmp_dst):
                shutil.rmtree(tmp_dst)
            shutil.copytree(src, tmp_dst)
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.isdir(dst):
                shutil.rmtree(dst)
            os.rename(tmp_dst, dst)

    # Restore intelligence files (skip applied/ to avoid deleting active snapshots)
    intel_snap = os.path.join(snap_dir, "intelligence")
    intel_ws = os.path.join(ws, "intelligence")
    if os.path.isdir(intel_snap):
        for item in os.listdir(intel_snap):
            if item == "applied":
                continue
            src = os.path.join(intel_snap, item)
            dst = os.path.join(intel_ws, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                tmp_dst = dst + ".rollback_tmp"
                if os.path.islink(tmp_dst):
                    os.unlink(tmp_dst)
                elif os.path.isdir(tmp_dst):
                    shutil.rmtree(tmp_dst)
                shutil.copytree(src, tmp_dst)
                if os.path.islink(dst):
                    os.unlink(dst)
                elif os.path.isdir(dst):
                    shutil.rmtree(dst)
                os.rename(tmp_dst, dst)

    for f in SNAPSHOT_FILES:
        src = os.path.join(snap_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ws, f))


# ═══════════════════════════════════════════════
# Apply Receipt
# ═══════════════════════════════════════════════

def write_receipt(snap_dir, proposal, ts, pre_checks, status="in_progress"):
    """Write APPLY_RECEIPT.md."""
    receipt_path = os.path.join(snap_dir, "APPLY_RECEIPT.md")
    ops_desc = ", ".join(op.get("op", "?") for op in proposal.get("Ops", []))
    lines = [
        f"[AR-{ts}]",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Proposal: {proposal.get('ProposalId', '?')}",
        f"Action: {ops_desc}",
        f"Result: {status}",
        f"Snapshot: {ts}",
        f"Risk: {proposal.get('Risk', '?')}",
        f"TargetBlock: {proposal.get('TargetBlock', '?')}",
        "FilesTouched:",
    ]
    files_touched = proposal.get("FilesTouched", [])
    if not files_touched:
        files_touched = list({op.get("file", "") for op in proposal.get("Ops", []) if op.get("file")})
    for f in files_touched:
        lines.append(f"- {f}")
    lines.append("PreChecks:")
    for c in pre_checks:
        lines.append(f"- {c}")
    lines.append(f"RollbackPlan: {proposal.get('Rollback', ['?'])[0] if isinstance(proposal.get('Rollback'), list) else proposal.get('Rollback', '?')}")
    lines.append(f"Status: {status}")
    lines.append("")

    with open(receipt_path, "w") as fh:
        fh.write("\n".join(lines))
    return receipt_path


def update_receipt(receipt_path, post_checks, delta, status, diff_text=None):
    """Update receipt with post-check results and diff."""
    with open(receipt_path, "a") as fh:
        fh.write("PostChecks:\n")
        for c in post_checks:
            fh.write(f"- {c}\n")
        fh.write("Delta:\n")
        for k, v in delta.items():
            fh.write(f"- {k}: {v}\n")

        if diff_text:
            fh.write("DIFF:\n")
            # Indent diff lines for block inclusion
            for line in diff_text.split("\n"):
                fh.write(f"  {line}\n")

        fh.write(f"Result: {status}\n")


def _get_mode(ws="."):
    """Read current governance_mode from intel-state.json.

    Supports legacy 'self_correcting_mode' for backward compatibility.
    """
    try:
        with open(os.path.join(ws, "memory/intel-state.json")) as f:
            state = json.load(f)
        return state.get("governance_mode", state.get("self_correcting_mode", "unknown"))
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════
# Op Executors
# ═══════════════════════════════════════════════

def execute_op(ws, op):
    """Execute a single op. Returns (success, message)."""
    op_type = op.get("op")
    raw_file = op.get("file", "")

    # Security: reject path traversal and symlink escape
    try:
        filepath = _safe_resolve(ws, raw_file)
    except ValueError as e:
        return False, f"SECURITY: {e}"

    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}"

    try:
        if op_type == "append_block":
            return _op_append_block(filepath, op)
        elif op_type == "insert_after_block":
            return _op_insert_after_block(filepath, op)
        elif op_type == "update_field":
            return _op_update_field(filepath, op)
        elif op_type == "append_list_item":
            return _op_append_list_item(filepath, op)
        elif op_type == "set_status":
            return _op_set_status(filepath, op)
        elif op_type == "replace_range":
            return _op_replace_range(filepath, op)
        elif op_type == "supersede_decision":
            return _op_supersede_decision(filepath, op)
        else:
            return False, f"Unknown op: {op_type}"
    except (OSError, IOError, ValueError, KeyError, IndexError) as e:
        return False, f"Op {op_type} failed: {e}"


def _op_append_block(filepath, op):
    """Append a new block at end of file."""
    patch = op.get("patch", "")
    if not patch:
        return False, "append_block: empty patch"

    with open(filepath, "a") as f:
        f.write(f"\n{patch}\n")
    return True, "append_block: OK"


def _op_insert_after_block(filepath, op):
    """Insert block after target block ID."""
    target = op.get("target")
    patch = op.get("patch", "")
    if not target or not patch:
        return False, "insert_after_block: missing target or patch"

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find end of target block (next block header or EOF)
    target_pattern = re.compile(rf"^\[{re.escape(target)}\]")
    found = False
    insert_at = None

    for i, line in enumerate(lines):
        if target_pattern.match(line):
            found = True
            continue
        if found and re.match(r"^\[[A-Z]+-[^\]]+\]\s*$", line):
            insert_at = i
            break

    if not found:
        return False, f"insert_after_block: target {target} not found"

    if insert_at is None:
        insert_at = len(lines)

    # Insert patch
    patch_lines = [line + "\n" for line in patch.split("\n")]
    lines[insert_at:insert_at] = ["\n"] + patch_lines

    with open(filepath, "w") as f:
        f.writelines(lines)
    return True, f"insert_after_block: inserted after {target}"


def _op_update_field(filepath, op):
    """Update a field value within a specific block."""
    target = op.get("target")
    field = op.get("field")
    value = op.get("value")
    if not target or not field:
        return False, "update_field: missing target or field"

    with open(filepath, "r") as f:
        lines = f.readlines()

    target_pattern = re.compile(rf"^\[{re.escape(target)}\]")
    in_target = False
    updated = False

    for i, line in enumerate(lines):
        if target_pattern.match(line):
            in_target = True
            continue
        if in_target and re.match(r"^\[[A-Z]+-[^\]]+\]\s*$", line):
            break
        if in_target:
            field_match = re.match(rf"^{re.escape(field)}:\s+.*$", line)
            if field_match:
                lines[i] = f"{field}: {value}\n"
                updated = True
                break

    if not updated:
        return False, f"update_field: field '{field}' not found in block {target}"

    with open(filepath, "w") as f:
        f.writelines(lines)
    return True, f"update_field: {target}.{field} = {value}"


def _op_append_list_item(filepath, op):
    """Append an item to a list field within a block."""
    target = op.get("target")
    list_field = op.get("list")
    item = op.get("item", "")
    if not target or not list_field:
        return False, "append_list_item: missing target or list"

    with open(filepath, "r") as f:
        lines = f.readlines()

    target_pattern = re.compile(rf"^\[{re.escape(target)}\]")
    in_target = False
    in_list = False
    insert_at = None

    for i, line in enumerate(lines):
        if target_pattern.match(line):
            in_target = True
            continue
        if in_target and re.match(r"^\[[A-Z]+-[^\]]+\]\s*$", line):
            break
        if in_target:
            # Find the list field
            if re.match(rf"^{re.escape(list_field)}:", line):
                in_list = True
                insert_at = i + 1
                continue
            if in_list:
                # Keep tracking list items
                if line.startswith("- ") or line.startswith("  -"):
                    insert_at = i + 1
                elif line.strip() == "":
                    insert_at = i
                    break
                else:
                    # New field = end of list
                    insert_at = i
                    break

    if insert_at is None:
        return False, f"append_list_item: list '{list_field}' not found in {target}"

    # Clean item — remove surrounding quotes if present
    item_clean = item.strip().strip('"').strip("'")
    lines.insert(insert_at, f"- {item_clean}\n")

    with open(filepath, "w") as f:
        f.writelines(lines)
    return True, f"append_list_item: added to {target}.{list_field}"


def _op_set_status(filepath, op):
    """Update Status field + auto-append History entry."""
    target = op.get("target")
    status = op.get("status")
    history = op.get("history", "")
    if not target or not status:
        return False, "set_status: missing target or status"

    # First update the Status field
    ok, msg = _op_update_field(filepath, {"target": target, "field": "Status", "value": status})
    if not ok:
        return False, f"set_status: field update failed: {msg}"

    # Then append History entry if provided
    if history:
        ok2, msg2 = _op_append_list_item(filepath, {
            "target": target, "list": "History", "item": history
        })
        if not ok2:
            return False, f"set_status: history append failed: {msg2}"

    return True, f"set_status: {target} -> {status}"


def _op_replace_range(filepath, op):
    """Replace content between start/end markers within a block."""
    target = op.get("target")
    range_spec = op.get("range", {})
    patch = op.get("patch", "")
    start_marker = range_spec.get("start", "")
    end_marker = range_spec.get("end", "")

    if not target or not start_marker or not end_marker:
        return False, "replace_range: missing target, range.start, or range.end"

    with open(filepath, "r") as f:
        lines = f.readlines()

    target_pattern = re.compile(rf"^\[{re.escape(target)}\]")
    in_target = False
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        if target_pattern.match(line):
            in_target = True
            continue
        if in_target and re.match(r"^\[[A-Z]+-[^\]]+\]\s*$", line):
            break
        if in_target:
            if start_marker in line and start_line is None:
                start_line = i
            if end_marker in line and start_line is not None:
                end_line = i
                break

    if start_line is None or end_line is None:
        return False, f"replace_range: markers not found in {target}"

    # Replace lines between markers (exclusive of end marker line)
    patch_lines = [line + "\n" for line in patch.split("\n")]
    lines[start_line:end_line] = patch_lines

    with open(filepath, "w") as f:
        f.writelines(lines)
    return True, f"replace_range: replaced {end_line - start_line} lines in {target}"


def _op_supersede_decision(filepath, op):
    """Atomic supersede: append new block + mark old as superseded."""
    target = op.get("target")
    new_block = op.get("new_block") or op.get("patch", "")
    if not target or not new_block:
        return False, "supersede_decision: missing target or new_block/patch"

    # Check that target exists and is not invariant enforcement
    blocks = parse_file(filepath)
    old = get_by_id(blocks, target)
    if not old:
        return False, f"supersede_decision: target {target} not found"

    # Invariant check
    sigs = old.get("ConstraintSignatures", [])
    has_invariant = any(s.get("enforcement") == "invariant" for s in sigs)
    if has_invariant:
        return False, f"supersede_decision: {target} has invariant enforcement (manual edit required — invariants cannot be modified by automation)"

    # Step 1: Mark old decision as superseded
    ok, msg = _op_update_field(filepath, {"target": target, "field": "Status", "value": "superseded"})
    if not ok:
        return False, f"supersede_decision step 1 (mark superseded): {msg}"

    # Step 2: Append new block
    ok2, msg2 = _op_append_block(filepath, {"patch": new_block})
    if not ok2:
        return False, f"supersede_decision step 2 (append new): {msg2}"

    return True, f"supersede_decision: {target} -> superseded, new block appended"


# ═══════════════════════════════════════════════
# v2.1.2 Operational Hardening
# ═══════════════════════════════════════════════

def compute_fingerprint(proposal):
    """Deterministic fingerprint from proposal content. Prevents duplicates.

    Includes op payload fields (value, patch, status) to distinguish proposals
    targeting the same block with different mutations.
    """
    canon = json.dumps({
        "type": proposal.get("Type", ""),
        "target": proposal.get("TargetBlock", ""),
        "ops": [
            {
                "op": op.get("op"),
                "file": op.get("file"),
                "target": op.get("target"),
                "value": op.get("value", ""),
                "patch": op.get("patch", ""),
                "status": op.get("status", ""),
            }
            for op in proposal.get("Ops", [])
        ]
    }, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def check_fingerprint_dedup(ws, proposal):
    """Check if a different proposal with same fingerprint already exists (staged or deferred)."""
    fp = compute_fingerprint(proposal)
    my_id = proposal.get("ProposalId", proposal.get("_id", ""))
    for pfile in PROPOSED_FILES:
        path = os.path.join(ws, pfile)
        if not os.path.isfile(path):
            continue
        blocks = parse_file(path)
        for b in blocks:
            if b.get("Status") in ("staged", "deferred"):
                bid = b.get("ProposalId", b.get("_id", ""))
                if bid == my_id:
                    continue  # Skip self
                existing_fp = b.get("Fingerprint", "")
                if existing_fp == fp:
                    return True, bid
    return False, None


def check_backlog_limit(ws):
    """Count staged proposals. Returns (count, limit_exceeded)."""
    # Read limit from mind-mem.json (source of truth for config), not intel-state.json
    config_path = os.path.join(ws, "mind-mem.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        config = {}
    limit = config.get("proposal_budget", {}).get("backlog_limit", 30)
    count = 0
    for pfile in PROPOSED_FILES:
        path = os.path.join(ws, pfile)
        if not os.path.isfile(path):
            continue
        blocks = parse_file(path)
        count += sum(1 for b in blocks if b.get("Status") == "staged")
    return count, count >= limit


def check_no_touch_window(ws):
    """Check if enough time has passed since last apply. Returns (ok, reason)."""
    state = _load_intel_state(ws)
    last_ts = state.get("last_apply_ts")
    if not last_ts:
        return True, "No previous apply"
    try:
        last = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        # Normalize both to naive for safe comparison
        last_naive = last.replace(tzinfo=None)
        now_naive = datetime.now()
        delta = now_naive - last_naive
        if delta < timedelta(minutes=10):
            remaining = timedelta(minutes=10) - delta
            return False, f"No-touch window: {remaining.seconds // 60}m {remaining.seconds % 60}s remaining"
    except (ValueError, TypeError):
        pass
    return True, "Cooldown clear"


def check_deferred_cooldown(ws, proposal):
    """Check if a rejected/deferred proposal for same target is still in cooldown."""
    state = _load_intel_state(ws)
    cooldown_days = state.get("defer_cooldown_days", 7)
    target = proposal.get("TargetBlock", "")
    if not target:
        return True, "No target"

    cutoff = datetime.now() - timedelta(days=cooldown_days)

    for pfile in PROPOSED_FILES:
        path = os.path.join(ws, pfile)
        if not os.path.isfile(path):
            continue
        blocks = parse_file(path)
        for b in blocks:
            if b.get("Status") in ("rejected", "deferred") and b.get("TargetBlock") == target:
                created = b.get("Created", "")
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt > cutoff:
                        return False, f"Target {target} has {b.get('Status')} proposal {b.get('ProposalId')} within {cooldown_days}d cooldown"
                except (ValueError, TypeError):
                    pass
    return True, "No cooldown conflict"


def update_last_apply_ts(ws):
    """Record the timestamp of the last apply."""
    state = _load_intel_state(ws)
    state["last_apply_ts"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    _save_intel_state(ws, state)


def generate_diff_text(ws, snap_dir, files_touched):
    """Generate unified diff showing what changed during apply."""
    diff_lines = []
    for rel_path in files_touched:
        old_path = os.path.join(snap_dir, rel_path)
        new_path = os.path.join(ws, rel_path)

        old_lines = []
        new_lines = []
        if os.path.isfile(old_path):
            with open(old_path, "r", errors="replace") as f:
                old_lines = f.readlines()
        if os.path.isfile(new_path):
            with open(new_path, "r", errors="replace") as f:
                new_lines = f.readlines()

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
            lineterm=""
        )
        diff_text = "\n".join(diff)
        if diff_text:
            diff_lines.append(diff_text)

    if not diff_lines:
        return "(no differences detected)"
    return "\n\n".join(diff_lines)


def _load_intel_state(ws):
    """Load intel-state.json."""
    path = os.path.join(ws, "memory/intel-state.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_intel_state(ws, state):
    """Save intel-state.json atomically (write to temp, then rename)."""
    path = os.path.join(ws, "memory/intel-state.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


# ═══════════════════════════════════════════════
# Main Apply Pipeline
# ═══════════════════════════════════════════════

def _get_workspace_lock_path(ws):
    """Return the path for the workspace-wide apply lock."""
    return os.path.join(ws, ".mind-mem-apply")


def apply_proposal(ws, proposal_id, dry_run=False, agent_id=None):
    """Main apply pipeline. Returns (success, message).

    Args:
        ws: Workspace path.
        proposal_id: Proposal ID to apply.
        dry_run: Validate without executing.
        agent_id: Optional agent ID for namespace ACL enforcement.
    """
    # Resolve symlinks early so all downstream code uses canonical paths.
    # Fixes macOS /var → /private/var mismatch in relpath computations.
    ws = os.path.realpath(ws)
    print("═══ Mind Mem Apply Engine v1.0 ═══")
    print(f"Proposal: {proposal_id}")
    print(f"Workspace: {ws}")
    print(f"Dry run: {dry_run}")
    print()

    # 0. Mode gate: detect_only blocks all apply/dry-run
    mode = _get_mode(ws)
    print(f"Mode: {mode}")
    if mode == "detect_only":
        print("ERROR: Cannot apply proposals in detect_only mode.")
        print("Switch to 'propose' or 'enforce' mode first.")
        return False, "Blocked: detect_only mode does not allow apply"

    # 1. Find proposal
    proposal, source_file = find_proposal(ws, proposal_id)
    if not proposal:
        print(f"ERROR: Proposal {proposal_id} not found in proposed/ files.")
        return False, "Proposal not found"

    print(f"Found in: {source_file}")
    print(f"Type: {proposal.get('Type')}  Risk: {proposal.get('Risk')}  Status: {proposal.get('Status')}")

    # 2. Validate proposal
    errors = validate_proposal(proposal)
    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return False, f"Validation failed: {len(errors)} error(s)"

    print("Proposal validation: PASS")

    # 2b. Fingerprint
    fp = compute_fingerprint(proposal)
    print(f"Fingerprint: {fp}")

    # 2c. v2.1.2 hardening checks (hard fail)
    print("\n--- Hardening Checks ---")

    # Dedup check
    is_dup, dup_id = check_fingerprint_dedup(ws, proposal)
    if is_dup:
        print(f"  FAIL: Duplicate fingerprint — matches {dup_id}")
        return False, f"Duplicate proposal (matches {dup_id})"
    print("  Dedup: PASS")

    # Backlog limit
    backlog_count, over_limit = check_backlog_limit(ws)
    if over_limit:
        print(f"  FAIL: Backlog limit exceeded ({backlog_count} staged)")
        return False, f"Backlog limit exceeded ({backlog_count} staged)"
    print(f"  Backlog: PASS ({backlog_count} staged)")

    # Deferred cooldown
    ok_cool, cool_reason = check_deferred_cooldown(ws, proposal)
    if not ok_cool:
        print(f"  FAIL: {cool_reason}")
        return False, cool_reason
    print("  Cooldown: PASS")

    # No-touch window (warning on dry-run, hard fail on real apply)
    ok_touch, touch_reason = check_no_touch_window(ws)
    if not ok_touch and not dry_run:
        print(f"  FAIL: {touch_reason}")
        return False, touch_reason
    elif not ok_touch:
        print(f"  WARN: {touch_reason} (dry-run, continuing)")
    else:
        print("  No-touch window: PASS")

    if dry_run:
        # Dry run: show ops without mutation (skip preconditions to avoid side effects)
        print("\n--- DRY RUN: would execute these ops ---")
        for i, op in enumerate(proposal.get("Ops", [])):
            print(f"  [{i}] {op.get('op')} -> {op.get('file')}:{op.get('target', 'eof')}")
        print("\nDry run complete. No changes made.")
        return True, "Dry run OK"

    # Namespace ACL check: verify agent has write permission for all target files
    if agent_id:
        ns = NamespaceManager(ws, agent_id=agent_id)
        for op in proposal.get("Ops", []):
            op_file = op.get("file", "")
            if op_file and not ns.can_write(op_file):
                print(f"\nACL DENIED: agent '{agent_id}' cannot write to '{op_file}'")
                return False, f"ACL denied: agent '{agent_id}' cannot write to '{op_file}'"
        print(f"  ACL: PASS (agent '{agent_id}' has write access)")

    # Acquire workspace-wide lock to prevent concurrent applies
    lock_path = _get_workspace_lock_path(ws)
    lock = FileLock(lock_path, timeout=30.0)
    try:
        lock.acquire()
    except Exception as e:
        print(f"\nERROR: Could not acquire workspace lock: {e}")
        return False, f"Workspace lock timeout: {e}"

    try:
        return _apply_proposal_locked(ws, proposal, proposal_id, source_file, lock)
    finally:
        lock.release()


def _apply_proposal_locked(ws, proposal, proposal_id, source_file, lock):
    """Execute the apply pipeline while holding the workspace lock."""

    # 3. Replay WAL from any prior crash BEFORE snapshotting (ensures clean state)
    wal = WAL(ws)
    replayed = wal.replay()
    if replayed:
        print(f"\n  WAL: Replayed {replayed} pending entry(ies) from prior crash")

    # 4. Create snapshot AFTER WAL recovery (O1: snapshot before any mutation)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\n--- Creating Snapshot: {ts} ---")
    # Record pre-apply file listing for orphan detection
    pre_apply_files = _list_workspace_files(ws)
    # Minimal snapshot: only files the proposal will touch (O(touched) vs O(workspace))
    files_touched = proposal.get("FilesTouched", [])
    if not files_touched:
        files_touched = list({op.get("file", "") for op in proposal.get("Ops", []) if op.get("file")})
    snap_dir = create_snapshot(ws, ts, files_touched=files_touched or None)
    snap_mode = "minimal" if files_touched else "full"
    print(f"  Snapshot: {snap_dir} ({snap_mode}, {len(files_touched)} files)")

    # 5. Check preconditions (may run validate.sh + intel_scan.py which write reports)
    print("\n--- Precondition Checks ---")
    ok, pre_report = check_preconditions(ws)
    for r in pre_report:
        print(f"  {r}")
    if not ok:
        print("PRECONDITIONS FAILED — rolling back.")
        restore_snapshot(ws, snap_dir)
        _cleanup_orphan_files(ws, pre_apply_files)
        return False, "Precondition check failed"

    receipt_path = write_receipt(snap_dir, proposal, ts, pre_report)
    print(f"  Receipt: {receipt_path}")

    # 6. Execute ops with WAL protection

    print(f"\n--- Executing {len(proposal.get('Ops', []))} Ops (WAL-protected) ---")
    delta = {"created": [], "modified": []}
    wal_entries = []  # Track WAL entries for this apply
    for i, op in enumerate(proposal.get("Ops", [])):
        raw_file = op.get("file", "")
        try:
            filepath = _safe_resolve(ws, raw_file)
        except ValueError:
            filepath = raw_file

        # WAL: log intention before mutation
        wal_id = wal.begin(
            operation=op.get("op", "unknown"),
            target_path=filepath,
            content=json.dumps(op, default=str),
        )
        wal_entries.append(wal_id)

        ok, msg = execute_op(ws, op)
        print(f"  [{i}] {op.get('op')}: {msg}")
        if not ok:
            # WAL: rollback this failed op's WAL entry
            wal.rollback(wal_id)
            print(f"\nOP FAILED at step {i} — rolling back.")
            # Also rollback any previously committed WAL entries via snapshot
            restore_snapshot(ws, snap_dir)
            _cleanup_orphan_files(ws, pre_apply_files)
            update_receipt(receipt_path, ["ABORTED: op failure"], delta, "rolled_back")
            return False, f"Op {i} failed: {msg}"

        # WAL: commit successful op
        wal.commit(wal_id)

        # Track delta
        target = op.get("target", "")
        if op.get("op") in ("append_block", "insert_after_block", "supersede_decision"):
            delta["created"].append(target or "new")
        else:
            delta["modified"].append(target)

    # 6. Post-checks
    print("\n--- Post-checks ---")
    ok, post_report = check_preconditions(ws)
    for r in post_report:
        print(f"  {r}")

    if not ok:
        print("\nPOST-CHECKS FAILED — rolling back.")
        restore_snapshot(ws, snap_dir)
        _cleanup_orphan_files(ws, pre_apply_files)
        update_receipt(receipt_path, post_report, delta, "rolled_back")
        # Also mark proposal as rolled back
        _mark_proposal_status(source_file, proposal_id, "rolled_back")
        return False, "Post-checks failed, rolled back"

    # 7. Generate DIFF text
    files_touched = proposal.get("FilesTouched", [])
    if not files_touched:
        # Derive from Ops when FilesTouched is absent
        files_touched = list({op.get("file", "") for op in proposal.get("Ops", []) if op.get("file")})
    diff_text = generate_diff_text(ws, snap_dir, files_touched)

    # 8. Commit receipt + mark proposal as applied + update last_apply_ts
    update_receipt(receipt_path, post_report, delta, "applied", diff_text)
    _mark_proposal_status(source_file, proposal_id, "applied")
    update_last_apply_ts(ws)
    print(f"\n═══ APPLIED: {proposal_id} ═══")
    print(f"Receipt: {receipt_path}")
    return True, f"Applied successfully. Receipt: {receipt_path}"


def _mark_proposal_status(source_file, proposal_id, new_status):
    """Update the Status field in the proposal's source file."""
    try:
        with open(source_file, "r") as f:
            content = f.read()
        # Find the proposal block and update its Status
        # Simple approach: find line with "ProposalId: <id>" then find "Status:" nearby
        lines = content.split("\n")
        in_proposal = False
        for i, line in enumerate(lines):
            if f"ProposalId: {proposal_id}" in line:
                in_proposal = True
                continue
            if in_proposal and line.startswith("Status:"):
                lines[i] = f"Status: {new_status}"
                break
            if in_proposal and re.match(r"^\[[A-Z]+-[^\]]+\]", line):
                break
        with open(source_file, "w") as f:
            f.write("\n".join(lines))
    except (OSError, IOError, json.JSONDecodeError) as e:
        print(f"WARNING: Could not update proposal status in {source_file}: {e}", file=sys.stderr)


def rollback(ws, receipt_ts):
    """Rollback from a receipt timestamp."""
    ws = os.path.realpath(ws)
    # Sanitize receipt_ts: must match YYYYMMDD-HHMMSS format (no traversal)
    if not re.match(r"^\d{8}-\d{6}$", receipt_ts):
        print(f"ERROR: Invalid receipt timestamp format: {receipt_ts} (expected YYYYMMDD-HHMMSS)")
        return False

    try:
        snap_dir = _safe_resolve(ws, os.path.join("intelligence/applied", receipt_ts))
    except ValueError as e:
        print(f"ERROR: {e}")
        return False

    if not os.path.isdir(snap_dir):
        print(f"ERROR: Snapshot directory not found: {snap_dir}")
        return False

    print(f"Restoring from snapshot: {snap_dir}")
    restore_snapshot(ws, snap_dir)

    # Re-run checks
    print("\n--- Post-rollback checks ---")
    ok, report = check_preconditions(ws)
    for r in report:
        print(f"  {r}")

    # Update receipt
    receipt_path = os.path.join(snap_dir, "APPLY_RECEIPT.md")
    if os.path.isfile(receipt_path):
        with open(receipt_path, "a") as f:
            f.write(f"\nRolledBack: {datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write("Result: rolled_back\n")

    print(f"\n═══ ROLLED BACK from {receipt_ts} ═══")
    return True


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mind Mem Apply Engine v1.0")
    parser.add_argument("proposal_id", nargs="?", help="ProposalId to apply (e.g. P-20260213-002)")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show ops without executing")
    parser.add_argument("--rollback", metavar="TS", help="Rollback to receipt timestamp")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace)
    os.chdir(ws)

    if args.rollback:
        ok = rollback(ws, args.rollback)
        sys.exit(0 if ok else 1)

    if not args.proposal_id:
        parser.error("proposal_id is required (or use --rollback)")

    ok, msg = apply_proposal(ws, args.proposal_id, dry_run=args.dry_run)
    print(f"\n{msg}")
    sys.exit(0 if ok else 1)
