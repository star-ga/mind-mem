#!/usr/bin/env python3
"""Mind Mem Intelligence Scanner v2.0 — Self-hosted, zero external dependencies.

Runs: contradiction detection, drift analysis, impact graph, state snapshots,
and weekly briefing generation.

Usage:
    python3 maintenance/intel_scan.py [workspace_path]
    python3 maintenance/intel_scan.py --snapshot-only
    python3 maintenance/intel_scan.py --briefing

Output: maintenance/intel-report.txt
Exit code: 0 = clean, 1 = critical issues found
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta

# Import block parser from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_parser import parse_file
from filelock import FileLock

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

VALID_DOMAINS = {
    "integrity", "memory", "retrieval", "security", "llm_strategy",
    "workflow", "project", "comms", "finance", "other"
}

VALID_MODALITIES = {"must", "must_not", "should", "should_not", "may"}

# Modality incompatibility matrix
MODALITY_CONFLICTS = {
    ("must", "must_not"): "critical",
    ("must_not", "must"): "critical",
    ("should", "must_not"): "medium",
    ("must_not", "should"): "medium",
    ("should", "should_not"): "medium",
    ("should_not", "should"): "medium",
    ("must", "should_not"): "medium",
    ("should_not", "must"): "medium",
    ("may", "must_not"): "low",
    ("must_not", "may"): "low",
}


class IntelReport:
    """Accumulates findings and generates report."""

    def __init__(self):
        self.lines = []
        self.contradictions = []
        self.drift_signals = []
        self.impact_records = []
        self.critical = 0
        self.warnings = 0
        self.info = 0

    def section(self, title):
        self.lines.append("")
        self.lines.append(f"=== {title} ===")

    def critical_msg(self, msg):
        self.critical += 1
        self.lines.append(f"  CRITICAL: {msg}")

    def warn(self, msg):
        self.warnings += 1
        self.lines.append(f"  WARN: {msg}")

    def info_msg(self, msg):
        self.info += 1
        self.lines.append(f"  INFO: {msg}")

    def ok(self, msg):
        self.lines.append(f"  OK: {msg}")

    def text(self):
        return "\n".join(self.lines)


def load_all(ws):
    """Load all parseable files."""
    data = {}
    files = {
        "decisions": f"{ws}/decisions/DECISIONS.md",
        "tasks": f"{ws}/tasks/TASKS.md",
        "projects": f"{ws}/entities/projects.md",
        "people": f"{ws}/entities/people.md",
        "tools": f"{ws}/entities/tools.md",
        "incidents": f"{ws}/entities/incidents.md",
        "contradictions": f"{ws}/intelligence/CONTRADICTIONS.md",
        "drift": f"{ws}/intelligence/DRIFT.md",
        "impact": f"{ws}/intelligence/IMPACT.md",
    }
    for key, path in files.items():
        if os.path.exists(path):
            try:
                data[key] = parse_file(path)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                data[key] = []
                print(f"WARNING: Failed to parse {path}: {e}", file=sys.stderr)
        else:
            data[key] = []
    return data


def load_intel_state(ws):
    """Load intel-state.json. Returns defaults on missing or corrupt file."""
    path = f"{ws}/memory/intel-state.json"
    defaults = {"governance_mode": "detect_only", "counters": {}}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print("[WARN] intel-state.json is not a dict, using defaults", file=sys.stderr)
                return defaults
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] intel-state.json corrupt: {e}, using defaults", file=sys.stderr)
            return defaults
    return defaults


def save_intel_state(ws, state):
    """Save intel-state.json atomically (write to temp, then rename).

    Uses file locking to prevent concurrent scanner writes.
    """
    path = f"{ws}/memory/intel-state.json"
    tmp_path = path + ".tmp"
    with FileLock(path):
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            f.write("\n")
        os.replace(tmp_path, path)


# ═══════════════════════════════════════════════
# 1. Contradiction Detection Engine
# ═══════════════════════════════════════════════

def detect_contradictions(decisions, report):
    """Detect contradictions between active decisions using ConstraintSignatures."""
    report.section("1. CONTRADICTION DETECTION")

    active = [d for d in decisions if d.get("Status") == "active"]
    if not active:
        report.ok("No active decisions to check.")
        return []

    # Extract all signatures with parent decision IDs
    sigs = []
    for d in active:
        for sig in d.get("ConstraintSignatures", []):
            sigs.append({"sig": sig, "decision": d["_id"]})

    report.info_msg(f"Scanning {len(sigs)} signatures across {len(active)} active decisions.")

    # Group signatures by axis key to avoid O(N²) full comparison.
    # Only signatures sharing an axis key can conflict, so we compare
    # within each bucket: O(Σ n_k²) instead of O(N²).
    from collections import defaultdict
    axis_groups = defaultdict(list)
    for s in sigs:
        axis_groups[get_axis_key(s["sig"])].append(s)

    contradictions = []
    checked = set()

    for group in axis_groups.values():
        for i, s1 in enumerate(group):
            for j in range(i + 1, len(group)):
                s2 = group[j]
                if s1["decision"] == s2["decision"]:
                    continue

                pair_key = tuple(sorted([
                    f"{s1['decision']}:{s1['sig']['id']}",
                    f"{s2['decision']}:{s2['sig']['id']}",
                ]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                conflict = check_signature_conflict(s1["sig"], s2["sig"])
                if conflict:
                    contradictions.append({
                        "sig1": s1,
                        "sig2": s2,
                        "severity": conflict["severity"],
                        "reason": conflict["reason"],
                    })

    if contradictions:
        for c in contradictions:
            sev = c["severity"]
            msg = (
                f"{c['sig1']['decision']}:{c['sig1']['sig']['id']} vs "
                f"{c['sig2']['decision']}:{c['sig2']['sig']['id']} — "
                f"{c['reason']}"
            )
            if sev == "critical":
                report.critical_msg(msg)
            else:
                report.warn(msg)
        report.info_msg(f"Found {len(contradictions)} contradiction(s).")
    else:
        report.ok(f"No contradictions found among {len(sigs)} signatures.")

    return contradictions


def get_axis_key(sig):
    """Extract axis.key from a signature. Falls back to domain/subject for v1.0 sigs."""
    axis = sig.get("axis", {})
    if isinstance(axis, dict) and axis.get("key"):
        return axis["key"]
    # Fallback for v1.0 signatures without axis field
    return f"{sig.get('domain', 'other')}.{sig.get('subject', 'unknown')}"


def get_relation(sig):
    """Get relation field, defaulting to standalone."""
    return sig.get("relation", "standalone")


def get_composable_ids(sig):
    """Get set of IDs this signature composes_with or requires."""
    ids = set()
    cw = sig.get("composes_with", [])
    if isinstance(cw, list):
        ids.update(cw)
    req = sig.get("requires", [])
    if isinstance(req, list):
        ids.update(req)
    return ids


def check_signature_conflict(sig1, sig2):
    """Check if two signatures conflict. Returns None or {severity, reason}.

    v1.1 logic:
    1. Must share axis.key (not just domain+subject)
    2. Skip pairs linked by composes_with or requires relation
    3. Check scope overlap
    4. Detect modality incompatibility or competing requirements
    """
    # ── v1.1: axis.key scoping ──
    ax1 = get_axis_key(sig1)
    ax2 = get_axis_key(sig2)
    if ax1 != ax2:
        return None

    # ── v1.1: relation-based suppression ──
    # If either signature declares the other as composes_with or requires, skip
    composable1 = get_composable_ids(sig1)
    composable2 = get_composable_ids(sig2)
    id1 = sig1.get("id", "")
    id2 = sig2.get("id", "")
    if id2 in composable1 or id1 in composable2:
        return None
    # Also skip if both declare relation=composes_with on same axis
    if get_relation(sig1) == "composes_with" and get_relation(sig2) == "composes_with":
        return None
    if get_relation(sig1) == "requires" and get_relation(sig2) == "requires":
        return None

    # ── Scope overlap check ──
    if not scopes_overlap(sig1.get("scope", {}), sig2.get("scope", {})):
        return None

    # ── Modality conflict detection ──
    m1 = sig1.get("modality", "may")
    m2 = sig2.get("modality", "may")

    conflict_level = MODALITY_CONFLICTS.get((m1, m2))
    if conflict_level:
        # Only flag modality conflicts when objects overlap or are unspecified
        obj1 = sig1.get("object", "").lower()
        obj2 = sig2.get("object", "").lower()
        if not obj1 or not obj2 or obj1 == obj2:
            return {
                "severity": conflict_level,
                "reason": f"modality conflict: {m1} vs {m2} on axis={ax1}",
            }

    # ── Competing requirements: same modality, same predicate, different objects ──
    shared_predicate = sig1.get("predicate", "").lower() == sig2.get("predicate", "").lower()
    shared_object = sig1.get("object", "").lower() == sig2.get("object", "").lower()

    if shared_predicate and not shared_object:
        # "must X" vs "must Y" = critical only if axis is exclusive (default: true)
        # "must_not X" vs "must_not Y" = compatible (just avoid both)
        # axis.exclusive: false means additive constraints are valid (e.g., "must hire Alice" + "must hire Bob")
        axis1 = sig1.get("axis", {})
        axis_exclusive = axis1.get("exclusive", True)
        if m1 == "must" and m2 == "must":
            if not axis_exclusive:
                return None  # Non-exclusive axis: additive constraints are valid
            return {
                "severity": "critical",
                "reason": (
                    f"competing hard requirements: both must "
                    f"{sig1.get('predicate', '?')} but different objects "
                    f"({sig1.get('object')} vs {sig2.get('object')}) on axis={ax1}"
                ),
            }
        if m1 == "should" and m2 == "should":
            return {
                "severity": "low",
                "reason": (
                    f"competing soft requirements: both should "
                    f"{sig1.get('predicate', '?')} but different objects "
                    f"({sig1.get('object')} vs {sig2.get('object')}) on axis={ax1}"
                ),
            }

    return None


def scopes_overlap(scope1, scope2):
    """Check if two scopes overlap (conservative: overlap unless explicitly disjoint)."""
    # If either has no projects restriction, they overlap
    p1 = scope1.get("projects", [])
    p2 = scope2.get("projects", [])
    if p1 and p2 and not set(p1) & set(p2):
        return False

    c1 = scope1.get("channels", [])
    c2 = scope2.get("channels", [])
    if c1 and c2 and not set(c1) & set(c2):
        return False

    # Time overlap check
    t1 = scope1.get("time", {})
    t2 = scope2.get("time", {})
    if t1.get("end") and t2.get("start"):
        if t1["end"] < t2["start"]:
            return False
    if t2.get("end") and t1.get("start"):
        if t2["end"] < t1["start"]:
            return False

    return True


# ═══════════════════════════════════════════════
# 2. Drift Analysis Engine
# ═══════════════════════════════════════════════

def detect_drift(data, report):
    """Detect strategic drift signals."""
    report.section("2. DRIFT ANALYSIS")

    decisions = [d for d in data["decisions"] if d.get("Status") == "active"]
    tasks = data["tasks"]
    incidents = data["incidents"]

    signals = []

    # 2a. Dead decisions: active decisions not referenced by any active task
    # v1.1: Only flag if enforcement != invariant AND priority >= 7 AND no impact edges
    active_tasks = [t for t in tasks if t.get("Status") in ("todo", "doing", "blocked")]
    task_refs = set()
    for t in active_tasks:
        for key in ("AlignsWith", "Dependencies", "Context", "Next"):
            val = t.get(key, "")
            if isinstance(val, str):
                task_refs.update(re.findall(r"D-\d{8}-\d{3}", val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        task_refs.update(re.findall(r"D-\d{8}-\d{3}", item))

    dead_decisions = []
    dead_skipped_exempt = 0  # invariant or structural
    dead_skipped_enforced = 0  # has EnforcedBy pointer
    dead_skipped_low_priority = 0
    for d in decisions:
        if d["_id"] in task_refs:
            continue

        sigs = d.get("ConstraintSignatures", [])

        # Exempt: enforcement=invariant or enforcement=structural
        EXEMPT_ENFORCEMENT = {"invariant", "structural"}
        has_exempt = any(s.get("enforcement") in EXEMPT_ENFORCEMENT for s in sigs)
        if has_exempt:
            dead_skipped_exempt += 1
            continue

        # Exempt: has EnforcedBy pointer (code/validator/config enforces it)
        has_enforced_by = any(s.get("enforced_by") for s in sigs)
        if has_enforced_by:
            dead_skipped_enforced += 1
            continue

        # Check max priority across signatures
        max_priority = 0
        for s in sigs:
            try:
                max_priority = max(max_priority, int(s.get("priority", 0)))
            except (ValueError, TypeError):
                pass
        # Also check decision-level Priority field (handles both numeric and P0-P3 format)
        dec_pri_str = d.get("Priority", "")
        if isinstance(dec_pri_str, str) and dec_pri_str.startswith("P") and len(dec_pri_str) == 2:
            # P0 = highest (map to 10), P1 = 9, P2 = 7, P3 = 5
            pri_map = {"P0": 10, "P1": 9, "P2": 7, "P3": 5}
            max_priority = max(max_priority, pri_map.get(dec_pri_str, 0))
        elif dec_pri_str:
            try:
                max_priority = max(max_priority, int(dec_pri_str))
            except (ValueError, TypeError):
                pass

        if max_priority < 7:
            dead_skipped_low_priority += 1
            continue

        dead_decisions.append(d["_id"])

    if dead_decisions:
        signals.append({
            "signal": "dead_decisions",
            "severity": "medium",
            "summary": f"{len(dead_decisions)} active decision(s) not referenced by any active task",
            "evidence": dead_decisions,
        })
        report.warn(f"Dead decisions (not referenced by active tasks): {', '.join(dead_decisions)}")
    else:
        report.ok("All active decisions referenced or exempt.")

    exempt_parts = []
    if dead_skipped_exempt:
        exempt_parts.append(f"{dead_skipped_exempt} invariant/structural")
    if dead_skipped_enforced:
        exempt_parts.append(f"{dead_skipped_enforced} enforced-by-code")
    if dead_skipped_low_priority:
        exempt_parts.append(f"{dead_skipped_low_priority} below priority threshold (< 7)")
    if exempt_parts:
        report.info_msg(f"Dead-decision exemptions: {', '.join(exempt_parts)}.")

    # 2b. Stalled tasks: blocked tasks without resolution path
    blocked = [t for t in tasks if t.get("Status") == "blocked"]
    if blocked:
        signals.append({
            "signal": "stalled_tasks",
            "severity": "medium",
            "summary": f"{len(blocked)} blocked task(s)",
            "evidence": [t["_id"] for t in blocked],
        })
        for t in blocked:
            report.warn(f"Blocked task: {t['_id']} — {t.get('Title', '?')}")
    else:
        report.ok("No blocked tasks.")

    # 2c. Repeated incidents of same class
    inc_types = {}
    for inc in incidents:
        rc = inc.get("RootCause", "unknown")
        inc_types.setdefault(rc, []).append(inc["_id"])
    repeated = {k: v for k, v in inc_types.items() if len(v) > 1}
    if repeated:
        for rc, ids in repeated.items():
            signals.append({
                "signal": "repeated_incidents",
                "severity": "high",
                "summary": f"Repeated incident pattern ({len(ids)}x): {rc[:60]}",
                "evidence": ids,
            })
            report.warn(f"Repeated incidents ({len(ids)}x): {', '.join(ids)} — {rc[:60]}")
    else:
        report.ok("No repeated incident patterns.")

    # 2d. Tasks without AlignsWith (v2 compliance)
    unaligned = []
    for t in active_tasks:
        aligns = t.get("AlignsWith", "")
        justification = t.get("Justification", "")
        if (not aligns or aligns == "none") and not justification:
            unaligned.append(t["_id"])

    if unaligned:
        signals.append({
            "signal": "tasks_not_aligned",
            "severity": "low",
            "summary": f"{len(unaligned)} active task(s) without AlignsWith or Justification",
            "evidence": unaligned,
        })
        report.warn(f"Unaligned tasks: {', '.join(unaligned)}")
    else:
        report.ok("All active tasks have AlignsWith or Justification.")

    # 2e. Coverage score: % of non-exempt decisions referenced by active tasks or code
    non_exempt = len(decisions) - dead_skipped_exempt - dead_skipped_low_priority
    covered = non_exempt - len(dead_decisions)
    coverage_pct = (covered / non_exempt * 100) if non_exempt > 0 else 100.0

    # 2f. Metrics summary
    report.info_msg(
        f"Metrics: active_decisions={len(decisions)}, "
        f"active_tasks={len(active_tasks)}, "
        f"blocked={len(blocked)}, "
        f"dead_decisions={len(dead_decisions)}, "
        f"incidents={len(incidents)}, "
        f"decision_coverage={coverage_pct:.0f}%"
    )

    return signals


# ═══════════════════════════════════════════════
# 3. Decision Impact Graph
# ═══════════════════════════════════════════════

def build_impact_graph(data, report):
    """Build decision impact graph by tracing references."""
    report.section("3. DECISION IMPACT GRAPH")

    decisions = [d for d in data["decisions"] if d.get("Status") == "active"]
    tasks = data["tasks"]
    projects = data["projects"]
    incidents = data["incidents"]

    impacts = []

    for d in decisions:
        did = d["_id"]
        affected_projects = set()
        affected_tasks = set()
        affected_incidents = set()

        # Check scope from signatures
        for sig in d.get("ConstraintSignatures", []):
            scope = sig.get("scope", {})
            for prj in scope.get("projects", []):
                if isinstance(prj, str):
                    affected_projects.add(prj)

        # Check which tasks reference this decision
        for t in tasks:
            for field in ("AlignsWith", "Dependencies", "Context", "Next"):
                val = t.get(field, "")
                if isinstance(val, str) and did in val:
                    affected_tasks.add(t["_id"])
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and did in item:
                            affected_tasks.add(t["_id"])

        # Check which incidents reference this decision
        for inc in incidents:
            for field in ("Prevention", "Fix", "Summary"):
                val = inc.get(field, "")
                if isinstance(val, str) and did in val:
                    affected_incidents.add(inc["_id"])

        # Check which projects this decision's tags/scope match
        for p in projects:
            desc = p.get("Description", "")
            keywords = p.get("Keywords", "")
            for tag in d.get("Tags", "").split(","):
                tag = tag.strip()
                if tag and (tag in desc.lower() or tag in keywords.lower()):
                    affected_projects.add(p["_id"])

        if affected_projects or affected_tasks or affected_incidents:
            impact = {
                "decision": did,
                "projects": sorted(affected_projects),
                "tasks": sorted(affected_tasks),
                "incidents": sorted(affected_incidents),
            }
            impacts.append(impact)
            report.info_msg(
                f"{did} -> "
                f"PRJ:{len(affected_projects)} "
                f"T:{len(affected_tasks)} "
                f"INC:{len(affected_incidents)}"
            )

    if not impacts:
        report.ok("No impact edges found.")
    else:
        report.ok(f"Built impact graph: {len(impacts)} decision(s) with edges.")

    return impacts


# ═══════════════════════════════════════════════
# 4. State Snapshot
# ═══════════════════════════════════════════════

def generate_snapshot(data, ws, report):
    """Generate state snapshot as JSON."""
    report.section("4. STATE SNAPSHOT")

    today = datetime.now().strftime("%Y-%m-%d")
    decisions = data["decisions"]
    tasks = data["tasks"]

    snapshot = {
        "date": today,
        "generated_at": datetime.now().isoformat() + "Z",
        "decisions": {
            "active": [d["_id"] for d in decisions if d.get("Status") == "active"],
            "superseded": [d["_id"] for d in decisions if d.get("Status") == "superseded"],
            "revoked": [d["_id"] for d in decisions if d.get("Status") == "revoked"],
        },
        "tasks": {
            "todo": [t["_id"] for t in tasks if t.get("Status") == "todo"],
            "doing": [t["_id"] for t in tasks if t.get("Status") == "doing"],
            "blocked": [t["_id"] for t in tasks if t.get("Status") == "blocked"],
            "done": [t["_id"] for t in tasks if t.get("Status") == "done"],
            "canceled": [t["_id"] for t in tasks if t.get("Status") == "canceled"],
        },
        "projects": {
            "active": [p["_id"] for p in data["projects"] if p.get("Status") == "active"],
        },
        "metrics": {
            "contradictions_open": len([c for c in data.get("contradictions", [])
                                        if c.get("Status") == "open"]),
            "drift_signals_open": len([d for d in data.get("drift", [])
                                       if d.get("Status") == "open"]),
            "total_decisions": len(decisions),
            "total_tasks": len(tasks),
            "total_incidents": len(data.get("incidents", [])),
        },
    }

    snap_dir = f"{ws}/intelligence/state/snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = f"{snap_dir}/S-{today}.json"
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    report.ok(f"Snapshot saved: {snap_path}")
    report.info_msg(
        f"Active: {len(snapshot['decisions']['active'])} decisions, "
        f"{len(snapshot['tasks']['todo']) + len(snapshot['tasks']['doing'])} active tasks"
    )

    return snapshot


# ═══════════════════════════════════════════════
# 5. Weekly Briefing
# ═══════════════════════════════════════════════

def generate_briefing(data, contradictions, drift_signals, impacts, ws, report):
    """Generate weekly strategic briefing."""
    report.section("5. WEEKLY BRIEFING")

    today = datetime.now()
    # ISO week
    year, week, _ = today.isocalendar()
    week_id = f"B-{year}-W{week:02d}"

    # Week range
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    week_range = f"{monday.strftime('%Y-%m-%d')} .. {sunday.strftime('%Y-%m-%d')}"

    decisions = data["decisions"]
    tasks = data["tasks"]
    active_decisions = [d for d in decisions if d.get("Status") == "active"]
    active_tasks = [t for t in tasks if t.get("Status") in ("todo", "doing")]

    # Decisions this week (compare full date range, not just month prefix)
    monday_str = monday.strftime("%Y-%m-%d")
    sunday_str = sunday.strftime("%Y-%m-%d")
    week_decisions = [d for d in decisions
                      if monday_str <= d.get("Date", "") <= sunday_str]

    # Done tasks
    done_tasks = [t for t in tasks if t.get("Status") == "done"]

    # Top 5 tasks by priority for next week focus
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    focus_tasks = sorted(
        active_tasks,
        key=lambda t: priority_order.get(t.get("Priority", "P3"), 9)
    )[:5]

    briefing_lines = [
        f"[{week_id}]",
        f"Range: {week_range}",
        "ExecutiveSummary:",
        f"- {len(active_decisions)} active decisions, {len(active_tasks)} active tasks.",
    ]

    if contradictions:
        briefing_lines.append(
            f"- {len(contradictions)} contradiction(s) detected — requires attention."
        )
    if drift_signals:
        high_drift = [s for s in drift_signals if s["severity"] in ("high", "critical")]
        if high_drift:
            briefing_lines.append(
                f"- {len(high_drift)} high-severity drift signal(s) detected."
            )

    briefing_lines.append("Wins:")
    recent_done = [t for t in done_tasks
                   if any(
                       monday_str <= h[:10] <= sunday_str
                       for h in t.get("History", [])
                       if isinstance(h, str) and len(h) >= 10
                   )]
    if recent_done:
        for t in recent_done[:5]:
            briefing_lines.append(f"- {t['_id']}: {t.get('Title', '?')}")
    else:
        briefing_lines.append("- (none this period)")

    briefing_lines.append("Risks:")
    if contradictions:
        for c in contradictions:
            briefing_lines.append(
                f"- Contradiction: {c['sig1']['decision']} vs {c['sig2']['decision']} ({c['severity']})"
            )
    if drift_signals:
        for s in drift_signals:
            if s["severity"] in ("high", "medium"):
                briefing_lines.append(f"- Drift: {s['signal']} — {s['summary']}")
    if not contradictions and not drift_signals:
        briefing_lines.append("- No active risks.")

    briefing_lines.append("DecisionsThisWeek:")
    if week_decisions:
        for d in week_decisions:
            briefing_lines.append(f"- {d['_id']}: {d.get('Statement', '?')[:80]}")
    else:
        briefing_lines.append("- (none)")

    briefing_lines.append("TaskFocusNextWeek:")
    for t in focus_tasks:
        briefing_lines.append(
            f"- {t['_id']}: {t.get('Title', '?')[:60]} [{t.get('Priority', '?')}]"
        )

    briefing_lines.append("DriftSignals:")
    if drift_signals:
        for s in drift_signals:
            briefing_lines.append(f"- {s['signal']}: {s['summary'][:80]}")
    else:
        briefing_lines.append("- none")

    briefing_lines.append("Contradictions:")
    if contradictions:
        for c in contradictions:
            briefing_lines.append(
                f"- {c['sig1']['sig']['id']} vs {c['sig2']['sig']['id']} ({c['severity']})"
            )
    else:
        briefing_lines.append("- none")

    briefing_lines.append("RecommendedActions:")
    if contradictions:
        briefing_lines.append("- Resolve open contradictions (manual review or auto-supersede).")
    if drift_signals:
        for s in drift_signals:
            if s["signal"] == "dead_decisions":
                briefing_lines.append("- Create tasks to support dead decisions or supersede them.")
            elif s["signal"] == "tasks_not_aligned":
                briefing_lines.append("- Add AlignsWith to unaligned tasks.")
            elif s["signal"] == "stalled_tasks":
                briefing_lines.append("- Unblock stalled tasks or cancel them.")
    if not contradictions and not drift_signals:
        briefing_lines.append("- Continue current trajectory. No corrective action needed.")

    briefing_lines.append("Sources:")
    briefing_lines.append("- decisions/DECISIONS.md")
    briefing_lines.append("- tasks/TASKS.md")
    briefing_lines.append(f"- intelligence/state/snapshots/S-{today.strftime('%Y-%m-%d')}.json")

    briefing_text = "\n".join(briefing_lines)

    # Append to BRIEFINGS.md
    briefing_path = f"{ws}/intelligence/BRIEFINGS.md"
    if not os.path.isfile(briefing_path):
        os.makedirs(os.path.dirname(briefing_path), exist_ok=True)
        with open(briefing_path, "w") as f:
            f.write("# Intelligence Briefings\n\n")
    with open(briefing_path, "r") as f:
        existing = f.read()

    # Check if this week's briefing already exists
    if week_id in existing:
        report.info_msg(f"Briefing {week_id} already exists, skipping.")
    else:
        with open(briefing_path, "a") as f:
            f.write(f"\n{briefing_text}\n")
        report.ok(f"Briefing {week_id} generated and appended.")

    return briefing_text


# ═══════════════════════════════════════════════
# 6. Write CONTRADICTIONS / DRIFT / IMPACT files
# ═══════════════════════════════════════════════

def write_contradictions(contradictions, ws, report):
    """Append new contradictions to CONTRADICTIONS.md."""
    if not contradictions:
        return

    path = f"{ws}/intelligence/CONTRADICTIONS.md"
    try:
        with open(path, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    today = datetime.now().strftime("%Y%m%d")
    # Find highest existing index for today to avoid ID collisions
    existing_ids = re.findall(rf"^\[C-{today}-(\d{{3}})\]", existing, re.MULTILINE)
    existing_max = max((int(x) for x in existing_ids), default=0)

    new_blocks = []
    for i, c in enumerate(contradictions):
        cid = f"C-{today}-{existing_max + i + 1:03d}"

        # Skip if signatures already recorded
        sig_pair = f"{c['sig1']['sig']['id']} vs {c['sig2']['sig']['id']}"
        if sig_pair in existing:
            continue

        block = f"""
[{cid}]
Date: {datetime.now().strftime('%Y-%m-%d')}
Severity: {c['severity']}
Type: decision_vs_decision
Statement: {c['reason']}
Objects:
- {c['sig1']['decision']}
- {c['sig2']['decision']}
Evidence:
- {c['sig1']['sig']['id']}: {c['sig1']['sig'].get('domain')}/{c['sig1']['sig'].get('modality')}
- {c['sig2']['sig']['id']}: {c['sig2']['sig'].get('domain')}/{c['sig2']['sig'].get('modality')}
ProposedFix: manual_review
Status: open
Resolution: none
Sources:
- decisions/DECISIONS.md"""
        new_blocks.append(block)

    if new_blocks:
        with FileLock(path):
            with open(path, "a") as f:
                for block in new_blocks:
                    f.write(block + "\n")
        report.info_msg(f"Wrote {len(new_blocks)} new contradiction(s) to CONTRADICTIONS.md")


def write_drift(drift_signals, ws, report):
    """Append new drift signals to DRIFT.md."""
    if not drift_signals:
        return

    path = f"{ws}/intelligence/DRIFT.md"
    try:
        with open(path, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    today = datetime.now().strftime("%Y%m%d")
    existing_ids = re.findall(rf"^\[DREF-{today}-(\d{{3}})\]", existing, re.MULTILINE)
    existing_max = max((int(x) for x in existing_ids), default=0)

    new_blocks = []
    for i, s in enumerate(drift_signals):
        dref_id = f"DREF-{today}-{existing_max + i + 1:03d}"

        # Skip if similar signal already recorded today
        if s["signal"] in existing and today[:8] in existing:
            continue

        evidence_lines = "\n".join(f"- {e}" for e in s.get("evidence", [])[:10])

        block = f"""
[{dref_id}]
Date: {datetime.now().strftime('%Y-%m-%d')}
Severity: {s['severity']}
Signal: {s['signal']}
Summary: {s['summary']}
Metrics:
- see snapshot
Evidence:
{evidence_lines}
ProposedAction: manual_review
Status: open
Sources:
- maintenance/intel-report.txt"""
        new_blocks.append(block)

    if new_blocks:
        with FileLock(path):
            with open(path, "a") as f:
                for block in new_blocks:
                    f.write(block + "\n")
        report.info_msg(f"Wrote {len(new_blocks)} new drift signal(s) to DRIFT.md")


def write_impact(impacts, ws, report):
    """Write impact graph to IMPACT.md."""
    if not impacts:
        return

    path = f"{ws}/intelligence/IMPACT.md"
    today = datetime.now().strftime("%Y%m%d")

    lines = [
        "# IMPACT — Mind Mem v2.0",
        "",
        "> Decision impact graph: decision -> affects -> projects/tasks/incidents/invariants.",
        f"> Last rebuilt: {datetime.now().strftime('%Y-%m-%d')}",
        "> ID format: I-YYYYMMDD-###",
        "",
        "---",
    ]

    for i, imp in enumerate(impacts):
        iid = f"I-{today}-{i + 1:03d}"
        lines.append("")
        lines.append(f"[{iid}]")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"Decision: {imp['decision']}")
        lines.append("Affects:")
        if imp["projects"]:
            lines.append(f"- Projects: {', '.join(imp['projects'])}")
        if imp["tasks"]:
            lines.append(f"- Tasks: {', '.join(imp['tasks'])}")
        if imp["incidents"]:
            lines.append(f"- Incidents: {', '.join(imp['incidents'])}")
        lines.append("Reason: auto-generated from references and scope analysis")
        lines.append("Confidence: 0.8")
        lines.append("Sources:")
        lines.append("- decisions/DECISIONS.md")
        lines.append("- tasks/TASKS.md")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    report.ok(f"Impact graph rebuilt: {len(impacts)} entries.")


# ═══════════════════════════════════════════════
# 7. Proposal Generation (propose/enforce modes)
# ═══════════════════════════════════════════════

def _load_config(ws):
    """Load mind-mem.json config."""
    path = os.path.join(ws, "mind-mem.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _count_staged_proposals(ws):
    """Count proposals with Status: staged across all proposed/ files."""
    count = 0
    proposed_dir = os.path.join(ws, "intelligence/proposed")
    if not os.path.isdir(proposed_dir):
        return 0
    for fn in os.listdir(proposed_dir):
        if fn.endswith(".md"):
            path = os.path.join(proposed_dir, fn)
            with open(path) as f:
                content = f.read()
            count += content.count("\nStatus: staged")
    return count


def generate_proposals(contradictions, drift_signals, ws, intel_state, report):
    """Generate fix proposals from scan findings (propose/enforce modes only).

    Respects proposal_budget.per_run and proposal_budget.per_day limits.
    """
    report.section("7. PROPOSAL GENERATION")

    config = _load_config(ws)
    budget = config.get("proposal_budget", {})
    per_run = budget.get("per_run", 3)
    per_day = budget.get("per_day", 6)
    backlog_limit = budget.get("backlog_limit", 30)

    # Check backlog limit
    staged = _count_staged_proposals(ws)
    if staged >= backlog_limit:
        report.warn(f"Backlog limit reached ({staged}/{backlog_limit}) — skipping proposals.")
        return 0

    # Check daily cap
    today = datetime.now().strftime("%Y-%m-%d")
    daily_count = intel_state.get("counters", {}).get("proposals_today", 0)
    daily_date = intel_state.get("counters", {}).get("proposals_date", "")
    if daily_date != today:
        daily_count = 0  # Reset for new day

    remaining_daily = per_day - daily_count
    remaining_run = per_run
    remaining = min(remaining_daily, remaining_run, backlog_limit - staged)

    if remaining <= 0:
        report.warn(f"Budget exhausted (daily: {daily_count}/{per_day}, run: 0/{per_run})")
        return 0

    proposals = []
    proposal_date = datetime.now().strftime("%Y%m%d")

    # Find highest existing proposal index for today to avoid ID collisions
    existing_max_idx = 0
    proposed_dir = os.path.join(ws, "intelligence/proposed")
    if os.path.isdir(proposed_dir):
        for fname in os.listdir(proposed_dir):
            fpath = os.path.join(proposed_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    for line in f:
                        if line.startswith(f"ProposalId: P-{proposal_date}-"):
                            try:
                                idx = int(line.strip().rsplit("-", 1)[1])
                                existing_max_idx = max(existing_max_idx, idx)
                            except (ValueError, IndexError):
                                pass

    # Generate proposals from contradictions (supersede the lower-priority one)
    for c in contradictions:
        if len(proposals) >= remaining:
            break
        sig1 = c["sig1"]["sig"]
        sig2 = c["sig2"]["sig"]
        p1 = int(sig1.get("priority", 5))
        p2 = int(sig2.get("priority", 5))
        # Propose superseding the lower-priority decision (skip invariants)
        e1 = sig1.get("enforcement", "")
        e2 = sig2.get("enforcement", "")
        if e1 == "invariant" and e2 == "invariant":
            continue  # Cannot supersede either invariant
        if p1 >= p2 and e2 != "invariant":
            target_dec = c["sig2"]["decision"]
        elif e1 != "invariant":
            target_dec = c["sig1"]["decision"]
        else:
            continue  # Both candidates are invariants

        pid = f"P-{proposal_date}-{existing_max_idx+len(proposals)+1:03d}"
        proposals.append({
            "id": pid,
            "type": "edit",
            "target": target_dec,
            "risk": "high",
            "evidence": c["reason"],
            "ops": [{"op": "set_status", "file": "decisions/DECISIONS.md",
                      "target": target_dec, "status": "revoked"}],
            "rollback": "restore_snapshot",
        })

    # Generate proposals from drift signals (dead decisions)
    for s in drift_signals:
        if len(proposals) >= remaining:
            break
        if s["signal"] == "dead_decisions":
            for did in s.get("evidence", [])[:2]:
                if len(proposals) >= remaining:
                    break
                pid = f"P-{proposal_date}-{existing_max_idx+len(proposals)+1:03d}"
                proposals.append({
                    "id": pid,
                    "type": "edit",
                    "target": did,
                    "risk": "medium",
                    "evidence": "Decision not referenced by any active task",
                    "ops": [{"op": "set_status", "file": "decisions/DECISIONS.md",
                              "target": did, "status": "revoked"}],
                    "rollback": "restore_snapshot",
                })

    # Route proposals to correct file based on Type
    TYPE_TO_FILE = {
        "decision": "intelligence/proposed/DECISIONS_PROPOSED.md",
        "task": "intelligence/proposed/TASKS_PROPOSED.md",
        "edit": "intelligence/proposed/EDITS_PROPOSED.md",
    }

    if proposals:
        # Group proposals by target file
        by_file = {}
        for p in proposals:
            if p["type"] not in TYPE_TO_FILE:
                report.warn(f"Skipping proposal {p['id']}: invalid type '{p['type']}'")
                continue
            target_file = TYPE_TO_FILE[p["type"]]
            by_file.setdefault(target_file, []).append(p)

        new_blocks_total = 0
        for target_file, file_proposals in by_file.items():
            proposed_path = os.path.join(ws, target_file)
            existing = ""
            if os.path.isfile(proposed_path):
                with open(proposed_path, "r") as f:
                    existing = f.read()

            new_blocks = []
            batch_fingerprints = set()
            for p in file_proposals:
                ops_lines = "\n".join(
                    f"- op: {op['op']}\n  file: {op['file']}\n  target: {op.get('target', '')}"
                    + (f"\n  status: {op['status']}" if 'status' in op else "")
                    for op in p.get("ops", [])
                )
                # Compute fingerprint (compatible with apply_engine.compute_fingerprint)
                fp_canon = json.dumps({
                    "type": p.get("type", ""),
                    "target": p.get("target", ""),
                    "ops": [
                        {"op": op.get("op"), "file": op.get("file"),
                         "target": op.get("target"), "value": op.get("value", ""),
                         "patch": op.get("patch", ""), "status": op.get("status", "")}
                        for op in p.get("ops", [])
                    ]
                }, sort_keys=True)
                fp = hashlib.sha256(fp_canon.encode()).hexdigest()[:16]
                block = (
                    f"\n[{p['id']}]\n"
                    f"ProposalId: {p['id']}\n"
                    f"Type: {p['type']}\n"
                    f"TargetBlock: {p['target']}\n"
                    f"Risk: {p['risk']}\n"
                    f"Evidence:\n- {p['evidence']}\n"
                    f"Rollback: {p.get('rollback', 'restore_snapshot')}\n"
                    f"Ops:\n{ops_lines}\n"
                    f"Fingerprint: {fp}\n"
                    f"Status: staged\n"
                    f"Sources:\n"
                    f"- maintenance/intel-report.txt\n"
                )
                # Skip if same fingerprint already exists in proposed file or current batch
                if f"Fingerprint: {fp}" in existing or fp in batch_fingerprints:
                    continue
                batch_fingerprints.add(fp)
                new_blocks.append(block)

            if new_blocks:
                with open(proposed_path, "a") as f:
                    for block in new_blocks:
                        f.write(block)
                new_blocks_total += len(new_blocks)

        # Update daily counter
        intel_state.setdefault("counters", {})
        intel_state["counters"]["proposals_today"] = daily_count + new_blocks_total
        intel_state["counters"]["proposals_date"] = today

        report.ok(f"Generated {new_blocks_total} proposal(s) (budget: {remaining_run}/run, {remaining_daily}/day)")
    else:
        report.ok("No proposals needed.")

    return len(proposals)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mind Mem Intelligence Scanner v2.0")
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--briefing", action="store_true")
    args = parser.parse_args()

    ws = args.workspace
    report = IntelReport()

    report.lines.append("Mind Mem Intelligence Scan Report v2.0")
    report.lines.append(f"Date: {datetime.now().isoformat()}Z")
    report.lines.append(f"Workspace: {ws}")

    # Load state
    intel_state = load_intel_state(ws)
    mode = intel_state.get("governance_mode", "detect_only")
    report.lines.append(f"Mode: {mode}")

    # Load all data
    data = load_all(ws)

    if args.snapshot_only:
        generate_snapshot(data, ws, report)
    else:
        # Full scan
        contradictions = detect_contradictions(data["decisions"], report)
        drift_signals = detect_drift(data, report)
        impacts = build_impact_graph(data, report)
        generate_snapshot(data, ws, report)

        # Write findings
        write_contradictions(contradictions, ws, report)
        write_drift(drift_signals, ws, report)
        write_impact(impacts, ws, report)

        # Mode-aware proposal generation
        if mode in ("propose", "enforce") and (contradictions or drift_signals):
            generate_proposals(
                contradictions, drift_signals, ws, intel_state, report
            )
        else:
            if mode == "detect_only" and (contradictions or drift_signals):
                report.info_msg(
                    "Mode is detect_only — skipping proposal generation. "
                    "Switch to 'propose' to generate fix proposals."
                )

        # Generate briefing
        generate_briefing(data, contradictions, drift_signals, impacts, ws, report)

        # Update intel-state
        now = datetime.now().isoformat() + "Z"
        intel_state["last_scan"] = now
        intel_state["last_snapshot"] = now
        intel_state["counters"]["contradictions_open"] = len(contradictions)
        intel_state["counters"]["drift_signals_open"] = len(drift_signals)
        intel_state["counters"]["impact_records"] = len(impacts)
        intel_state["counters"]["snapshots"] = intel_state["counters"].get("snapshots", 0) + 1
        intel_state["counters"]["briefings"] = intel_state["counters"].get("briefings", 0) + 1
        save_intel_state(ws, intel_state)

    # Summary
    report.lines.append("")
    report.lines.append("=" * 50)
    report.lines.append(
        f"TOTAL: {report.critical} critical | {report.warnings} warnings | {report.info} info"
    )
    report.lines.append("=" * 50)

    # Write report
    report_path = f"{ws}/maintenance/intel-report.txt"
    with open(report_path, "w") as f:
        f.write(report.text() + "\n")

    # Print to stdout
    print(report.text())

    # Exit code
    sys.exit(1 if report.critical > 0 else 0)


if __name__ == "__main__":
    main()
