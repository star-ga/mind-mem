#!/usr/bin/env python3
"""Mind Mem Integrity Validator (Python, cross-platform).

Runs the same structural checks as validate.sh but works on any OS.
Usage: python3 scripts/validate_py.py [workspace_path]
"""

import os
import re
import sys
from datetime import datetime, timezone

# Allow importing block_parser from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file


class Validator:
    def __init__(self, workspace):
        self.ws = os.path.abspath(workspace)
        self.checks = 0
        self.passed = 0
        self.issues = 0
        self.warnings = 0
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def pass_(self, msg):
        self.checks += 1
        self.passed += 1
        self.log(f"  PASS: {msg}")

    def fail(self, msg):
        self.checks += 1
        self.issues += 1
        self.log(f"  FAIL: {msg}")

    def warn(self, msg):
        self.warnings += 1
        self.log(f"  WARN: {msg}")

    def section(self, title):
        self.log("")
        self.log(f"=== {title} ===")

    def file_exists(self, rel_path, label=None):
        path = os.path.join(self.ws, rel_path)
        label = label or rel_path
        if os.path.isfile(path):
            self.pass_(f"{label} exists")
            return True
        else:
            self.fail(f"{label} MISSING")
            return False

    def run(self):
        # Workspace detection
        if not os.path.isfile(os.path.join(self.ws, "mind-mem.json")):
            print(f"ERROR: No mind-mem.json found in '{self.ws}'.")
            print("")
            print("This does not appear to be an initialized mind-mem workspace.")
            print("To initialize: python3 scripts/init_workspace.py /path/to/workspace")
            return 1

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.log("Mind Mem Integrity Validation Report (Python)")
        self.log(f"Date: {now}")
        self.log(f"Workspace: {self.ws}")

        self._check_file_structure()
        self._check_decisions()
        self._check_tasks()
        self._check_entities()
        self._check_provenance()
        self._check_cross_refs()
        self._check_intelligence()

        self.log("")
        self.log("=" * 43)
        self.log(
            f"TOTAL: {self.checks} checks | {self.passed} passed"
            f" | {self.issues} issues | {self.warnings} warnings"
        )
        self.log("=" * 43)

        report = "\n".join(self.lines)
        print(report)

        # Write report
        report_path = os.path.join(self.ws, "maintenance", "validation-report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

        return 1 if self.issues > 0 else 0

    def _check_file_structure(self):
        self.section("0. FILE STRUCTURE")
        for f in [
            "decisions/DECISIONS.md", "tasks/TASKS.md",
            "entities/projects.md", "entities/people.md",
            "entities/tools.md", "entities/incidents.md",
        ]:
            self.file_exists(f)

        weekly_dir = os.path.join(self.ws, "summaries", "weekly")
        if os.path.isdir(weekly_dir):
            md_files = [f for f in os.listdir(weekly_dir) if f.endswith(".md")]
            if md_files:
                self.pass_(f"summaries/weekly/ has {len(md_files)} file(s)")
            else:
                self.warn("summaries/weekly/ has no .md files (expected for new workspaces)")
        else:
            self.warn("summaries/weekly/ directory missing")

        self.file_exists("memory/maint-state.json")

        mem_path = os.path.join(self.ws, "MEMORY.md")
        if os.path.isfile(mem_path):
            self.pass_("MEMORY.md exists")
            with open(mem_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "Memory Protocol v1.0" in content:
                self.pass_("MEMORY.md has Protocol v1.0 header")
            else:
                self.fail("MEMORY.md missing Protocol v1.0 header")
        else:
            self.fail("MEMORY.md MISSING")

    def _check_blocks(self, rel_path, id_pattern, label, required_fields,
                      status_values=None, extra_checks=None):
        path = os.path.join(self.ws, rel_path)
        if not os.path.isfile(path):
            return []

        try:
            blocks = parse_file(path)
        except Exception:
            self.fail(f"Failed to parse {rel_path}")
            return []

        # ID format
        id_re = re.compile(id_pattern)
        bad_ids = [b["_id"] for b in blocks if not id_re.match(b.get("_id", ""))]
        if not bad_ids:
            self.pass_(f"All {len(blocks)} {label} IDs match {id_pattern}")
        else:
            self.fail(f"Malformed {label} IDs: {bad_ids}")

        # Required fields
        for field in required_fields:
            missing = sum(1 for b in blocks if not b.get(field))
            if missing == 0:
                self.pass_(f"{label}: {field}: present in all {len(blocks)} blocks")
            else:
                self.fail(f"{label}: {field}: missing in {missing}/{len(blocks)} blocks")

        # Status values
        if status_values:
            bad = [b["_id"] for b in blocks
                   if b.get("Status") and b["Status"] not in status_values]
            if not bad:
                self.pass_(f"{label}: all Status values valid ({len(blocks)})")
            else:
                self.fail(f"{label}: invalid Status values in {bad}")

        if extra_checks:
            extra_checks(blocks)

        return blocks

    def _check_decisions(self):
        self.section("1. DECISIONS - IDs, fields, values")
        required = ["Date", "Status", "Scope", "Statement", "Rationale",
                     "Supersedes", "Tags", "Sources"]
        valid_status = {"active", "superseded", "revoked"}

        blocks = self._check_blocks(
            "decisions/DECISIONS.md",
            r"^D-\d{8}-\d{3}$", "Decisions", required,
            status_values=valid_status,
        )

        if blocks:
            # Scope validation
            scope_re = re.compile(r"^(global|project:\S+|channel:\S+)$")
            bad_scope = [b["_id"] for b in blocks
                         if b.get("Scope") and not scope_re.match(b["Scope"])]
            if not bad_scope:
                self.pass_(f"Decisions: all Scope values valid ({len(blocks)})")
            else:
                self.fail(f"Decisions: invalid Scope in {bad_scope}")

            # Supersedes validation
            sup_re = re.compile(r"^(none|D-\d{8}-\d{3})$")
            bad_sup = [b["_id"] for b in blocks
                       if b.get("Supersedes") and not sup_re.match(b["Supersedes"])]
            if not bad_sup:
                self.pass_(f"Decisions: all Supersedes values valid ({len(blocks)})")
            else:
                self.fail(f"Decisions: invalid Supersedes in {bad_sup}")

    def _check_tasks(self):
        self.section("2. TASKS - IDs, fields, values")
        required = ["Date", "Title", "Status", "Priority", "Project",
                     "Due", "Owner", "Context", "Next", "Dependencies",
                     "Sources", "History"]
        valid_status = {"todo", "doing", "blocked", "done", "canceled"}

        def extra(blocks):
            # Priority
            bad_pri = [b["_id"] for b in blocks
                       if b.get("Priority") and b["Priority"] not in
                       {"P0", "P1", "P2", "P3"}]
            if not bad_pri:
                self.pass_(f"Tasks: all Priority values valid ({len(blocks)})")
            else:
                self.fail(f"Tasks: invalid Priority in {bad_pri}")

            # Owner
            bad_own = [b["_id"] for b in blocks
                       if b.get("Owner") and b["Owner"] not in {"user", "bot"}]
            if not bad_own:
                self.pass_(f"Tasks: all Owner values valid ({len(blocks)})")
            else:
                self.fail(f"Tasks: invalid Owner in {bad_own}")

        self._check_blocks(
            "tasks/TASKS.md",
            r"^T-\d{8}-\d{3}$", "Tasks", required,
            status_values=valid_status,
            extra_checks=extra,
        )

    def _check_entities(self):
        self.section("3. ENTITIES - IDs & required fields")
        for rel, pat, label in [
            ("entities/projects.md", r"^PRJ-[a-z0-9-]+$", "Projects"),
            ("entities/people.md", r"^PER-[a-z0-9-]+$", "People"),
            ("entities/tools.md", r"^TOOL-[a-z0-9-]+$", "Tools"),
        ]:
            self._check_blocks(rel, pat, label, [])

        # Incidents with extra fields
        inc_required = ["Date", "Title", "Impact", "Summary",
                        "RootCause", "Fix", "Prevention", "Sources"]
        self._check_blocks(
            "entities/incidents.md",
            r"^INC-\d{8}-[a-z0-9-]+$", "Incidents", inc_required,
        )

    def _check_provenance(self):
        self.section("4. PROVENANCE - Sources not empty")
        for rel in ["decisions/DECISIONS.md", "tasks/TASKS.md",
                     "entities/incidents.md"]:
            path = os.path.join(self.ws, rel)
            if not os.path.isfile(path):
                continue
            try:
                blocks = parse_file(path)
            except Exception:
                continue
            fname = os.path.basename(rel)
            with_sources = sum(1 for b in blocks if b.get("Sources"))
            if with_sources >= len(blocks):
                self.pass_(f"{fname}: all {len(blocks)} blocks have Sources:")
            else:
                self.fail(f"{fname}: blocks without Sources: ({with_sources}/{len(blocks)})")

            empty = sum(1 for b in blocks
                        if b.get("Sources") is not None and b["Sources"] in ("", []))
            if empty == 0:
                self.pass_(f"{fname}: no empty Sources: blocks")
            else:
                self.fail(f"{fname}: {empty} empty Sources: blocks")

    def _check_cross_refs(self):
        self.section("5. CROSS-REFERENCE INTEGRITY")
        # Collect all defined IDs
        defined = set()
        id_patterns = {
            "decisions/DECISIONS.md": r"^D-\d{8}-\d{3}$",
            "tasks/TASKS.md": r"^T-\d{8}-\d{3}$",
            "entities/incidents.md": r"^INC-\d{8}-[a-z0-9-]+$",
            "entities/projects.md": r"^PRJ-[a-z0-9-]+$",
            "entities/people.md": r"^PER-[a-z0-9-]+$",
            "entities/tools.md": r"^TOOL-[a-z0-9-]+$",
        }
        for rel in id_patterns:
            path = os.path.join(self.ws, rel)
            if not os.path.isfile(path):
                continue
            try:
                blocks = parse_file(path)
                for b in blocks:
                    if b.get("_id"):
                        defined.add(b["_id"])
            except Exception:
                pass

        # Scan for references
        ref_re = re.compile(
            r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|INC-\d{8}-[a-z0-9-]+"
            r"|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+)\b"
        )
        referenced = set()
        scan_dirs = ["decisions", "tasks", "entities", "summaries"]
        for d in scan_dirs:
            dirpath = os.path.join(self.ws, d)
            if not os.path.isdir(dirpath):
                continue
            for root, _, files in os.walk(dirpath):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                # Skip template/comment lines
                                if line.lstrip().startswith(">"):
                                    continue
                                for match in ref_re.finditer(line):
                                    referenced.add(match.group(1))
                    except Exception:
                        pass

        dangling = referenced - defined
        if not dangling:
            self.pass_("All cross-references resolve to defined IDs")
        else:
            for d in sorted(dangling):
                self.fail(f"MISSING: {d} (referenced but not defined)")

    def _check_intelligence(self):
        self.section("6. INTELLIGENCE FILES")
        for f in ["SIGNALS.md", "CONTRADICTIONS.md", "DRIFT.md",
                   "IMPACT.md", "BRIEFINGS.md", "AUDIT.md"]:
            path = os.path.join(self.ws, "intelligence", f)
            if os.path.isfile(path):
                self.pass_(f"intelligence/{f} exists")
            else:
                self.warn(f"intelligence/{f} MISSING")

        if os.path.isfile(os.path.join(self.ws, "memory", "intel-state.json")):
            self.pass_("memory/intel-state.json exists")
        else:
            self.warn("memory/intel-state.json MISSING")

        if os.path.isdir(os.path.join(self.ws, "intelligence", "proposed")):
            self.pass_("intelligence/proposed/ directory exists")
        else:
            self.warn("intelligence/proposed/ directory MISSING")


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    validator = Validator(workspace)
    sys.exit(validator.run())


if __name__ == "__main__":
    main()
