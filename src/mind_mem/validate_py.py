#!/usr/bin/env python3
"""Mind Mem Integrity Validator — canonical engine.

Single source of truth for SPEC.md-defined invariant enforcement.
Cross-platform (runs on Linux, macOS, Windows without WSL), covered
by the full pytest suite, and the designated replacement for the
bash sibling ``validate.sh`` that ships alongside. ``validate.sh``
is on a deprecation path starting v3.1.x; v3.2.0 will convert it
into a thin shim that execs this module.

Usage:
    python3 -m mind_mem.validate_py [workspace_path]

Exit codes:
    0 — clean (no issues)
    1 — at least one invariant failed
"""

import os
import re
import sys
from datetime import datetime, timezone

# Allow importing block_parser from same directory
from .block_parser import parse_file
from .corpus_registry import VALIDATE_DIRS
from .enums import TaskStatus


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
            print("To initialize: mind-mem-init /path/to/workspace")
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
        self._check_signatures_v11()
        self._check_proposed_fingerprints()

        self.log("")
        self.log("=" * 43)
        self.log(f"TOTAL: {self.checks} checks | {self.passed} passed | {self.issues} issues | {self.warnings} warnings")
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
            "decisions/DECISIONS.md",
            "tasks/TASKS.md",
            "entities/projects.md",
            "entities/people.md",
            "entities/tools.md",
            "entities/incidents.md",
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
            with open(mem_path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if "Memory Protocol v1.0" in content:
                self.pass_("MEMORY.md has Protocol v1.0 header")
            else:
                self.fail("MEMORY.md missing Protocol v1.0 header")
        else:
            self.fail("MEMORY.md MISSING")

    def _check_blocks(self, rel_path, id_pattern, label, required_fields, status_values=None, extra_checks=None):
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
            bad = [b["_id"] for b in blocks if b.get("Status") and b["Status"] not in status_values]
            if not bad:
                self.pass_(f"{label}: all Status values valid ({len(blocks)})")
            else:
                self.fail(f"{label}: invalid Status values in {bad}")

        if extra_checks:
            extra_checks(blocks)

        return blocks

    def _check_decisions(self):
        self.section("1. DECISIONS - IDs, fields, values")
        required = ["Date", "Status", "Scope", "Statement", "Rationale", "Supersedes", "Tags", "Sources"]
        valid_status = {"active", "superseded", "revoked"}

        blocks = self._check_blocks(
            "decisions/DECISIONS.md",
            r"^D-\d{8}-\d{3}$",
            "Decisions",
            required,
            status_values=valid_status,
        )

        if blocks:
            # Scope validation
            scope_re = re.compile(r"^(global|project:\S+|channel:\S+)$")
            bad_scope = [b["_id"] for b in blocks if b.get("Scope") and not scope_re.match(b["Scope"])]
            if not bad_scope:
                self.pass_(f"Decisions: all Scope values valid ({len(blocks)})")
            else:
                self.fail(f"Decisions: invalid Scope in {bad_scope}")

            # Supersedes validation
            sup_re = re.compile(r"^(none|D-\d{8}-\d{3})$")
            bad_sup = [b["_id"] for b in blocks if b.get("Supersedes") and not sup_re.match(b["Supersedes"])]
            if not bad_sup:
                self.pass_(f"Decisions: all Supersedes values valid ({len(blocks)})")
            else:
                self.fail(f"Decisions: invalid Supersedes in {bad_sup}")

    def _check_tasks(self):
        self.section("2. TASKS - IDs, fields, values")
        required = [
            "Date",
            "Title",
            "Status",
            "Priority",
            "Project",
            "Due",
            "Owner",
            "Context",
            "Next",
            "Dependencies",
            "Sources",
            "History",
        ]
        valid_status = {s.value for s in TaskStatus}

        def extra(blocks):
            # Priority
            bad_pri = [b["_id"] for b in blocks if b.get("Priority") and b["Priority"] not in {"P0", "P1", "P2", "P3"}]
            if not bad_pri:
                self.pass_(f"Tasks: all Priority values valid ({len(blocks)})")
            else:
                self.fail(f"Tasks: invalid Priority in {bad_pri}")

            # Owner
            bad_own = [b["_id"] for b in blocks if b.get("Owner") and b["Owner"] not in {"user", "bot"}]
            if not bad_own:
                self.pass_(f"Tasks: all Owner values valid ({len(blocks)})")
            else:
                self.fail(f"Tasks: invalid Owner in {bad_own}")

        self._check_blocks(
            "tasks/TASKS.md",
            r"^T-\d{8}-\d{3}$",
            "Tasks",
            required,
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
        inc_required = ["Date", "Title", "Impact", "Summary", "RootCause", "Fix", "Prevention", "Sources"]
        self._check_blocks(
            "entities/incidents.md",
            r"^INC-\d{8}-[a-z0-9-]+$",
            "Incidents",
            inc_required,
        )

    def _check_provenance(self):
        self.section("4. PROVENANCE - Sources not empty")
        for rel in ["decisions/DECISIONS.md", "tasks/TASKS.md", "entities/incidents.md"]:
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

            empty = sum(1 for b in blocks if b.get("Sources") is not None and b["Sources"] in ("", []))
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
        for d in VALIDATE_DIRS:
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

    def _check_signatures_v11(self):
        """v1.1 ConstraintSignature checks — ports V2.1 / V2.2 / V2.6 / V2.7
        from the legacy bash engine.

        Covered signals:
          V2.1 — decisions tagged integrity|security|memory|retrieval
                 must carry at least one ConstraintSignature.
          V2.2 — every signature carries the nine required fields.
          V2.6 — signature.axis.key, signature.relation, signature.enforcement
                 are present and drawn from the legal enums.
          V2.7 — signature.lifecycle.created_by is present.
        """
        self.section("7. CONSTRAINT SIGNATURES (v1.1)")
        dec_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        if not os.path.isfile(dec_path):
            self.warn("decisions/DECISIONS.md missing — skipping signature checks")
            return
        try:
            blocks = parse_file(dec_path)
        except Exception as e:
            self.fail(f"Failed to parse decisions/DECISIONS.md: {e}")
            return

        required_tags = {"integrity", "security", "memory", "retrieval"}
        required_sig_fields = [
            "id",
            "domain",
            "subject",
            "predicate",
            "object",
            "modality",
            "priority",
            "scope",
            "evidence",
        ]
        valid_relations = {
            "standalone",
            "requires",
            "implies",
            "composes_with",
            "overrides",
            "equivalent",
        }
        valid_enforcement = {"invariant", "structural", "policy", "guideline"}

        v21_missing: list[str] = []
        v22_issues: list[str] = []
        v26_issues: list[str] = []
        v27_issues: list[str] = []
        for b in blocks:
            if b.get("Status") != "active":
                continue
            tags_raw = b.get("Tags", "") or ""
            tags = {t.strip() for t in tags_raw.split(",") if t.strip()}
            bid = b.get("_id", "<no-id>")
            sigs = b.get("ConstraintSignatures", []) or []

            if tags & required_tags and not sigs:
                v21_missing.append(bid)

            for sig in sigs if isinstance(sigs, list) else []:
                if not isinstance(sig, dict):
                    continue
                sid = sig.get("id", "?")
                missing = [f for f in required_sig_fields if f not in sig]
                if missing:
                    v22_issues.append(f"{bid}:{sid} missing {','.join(missing)}")

                ax = sig.get("axis")
                if not isinstance(ax, dict) or not ax.get("key"):
                    v26_issues.append(f"{bid}:{sid} missing axis.key")
                rel = sig.get("relation")
                if rel and rel not in valid_relations:
                    v26_issues.append(f"{bid}:{sid} bad relation {rel!r}")
                enf = sig.get("enforcement")
                if enf and enf not in valid_enforcement:
                    v26_issues.append(f"{bid}:{sid} bad enforcement {enf!r}")

                lc = sig.get("lifecycle")
                if not isinstance(lc, dict) or not lc.get("created_by"):
                    v27_issues.append(f"{bid}:{sid} missing lifecycle.created_by")

        if v21_missing:
            for bid in v21_missing:
                self.warn(f"V2.1: {bid} tagged integrity/security/memory/retrieval but no ConstraintSignatures")
        else:
            self.pass_("V2.1: All relevant active decisions have ConstraintSignatures")

        if v22_issues:
            for msg in v22_issues:
                self.fail(f"V2.2: {msg}")
        else:
            self.pass_("V2.2: All ConstraintSignatures have required fields")

        if v26_issues:
            for msg in v26_issues:
                self.warn(f"V2.6: {msg}")
        else:
            self.pass_("V2.6: All signatures have valid axis.key/relation/enforcement")

        if v27_issues:
            for msg in v27_issues:
                self.warn(f"V2.7: {msg}")
        else:
            self.pass_("V2.7: All signatures have lifecycle.created_by")

    def _check_proposed_fingerprints(self):
        """V2.9 — staged proposals must carry a Fingerprint field."""
        self.section("8. PROPOSED FINGERPRINTS")
        proposed_dir = os.path.join(self.ws, "intelligence", "proposed")
        if not os.path.isdir(proposed_dir):
            self.warn("intelligence/proposed/ missing — skipping V2.9")
            return

        any_files = False
        for fname in sorted(os.listdir(proposed_dir)):
            if not fname.endswith("_PROPOSED.md"):
                continue
            any_files = True
            path = os.path.join(proposed_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                self.warn(f"V2.9: {fname} read error: {e}")
                continue
            staged = len(re.findall(r"^Status: staged", content, re.MULTILINE))
            fingerprints = len(re.findall(r"^Fingerprint: ", content, re.MULTILINE))
            if staged > 0 and fingerprints < staged:
                self.warn(f"V2.9: {fname} has {staged} staged proposals but only {fingerprints} Fingerprint field(s)")
            elif staged > 0:
                self.pass_(f"V2.9: {fname} staged proposals have Fingerprints")

        if not any_files:
            self.warn("V2.9: no *_PROPOSED.md files present to validate")

    def _check_intelligence(self):
        self.section("6. INTELLIGENCE FILES")
        for f in ["SIGNALS.md", "CONTRADICTIONS.md", "DRIFT.md", "IMPACT.md", "BRIEFINGS.md", "AUDIT.md"]:
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
