# Mind Mem Formal Specification v1.0

This document defines the grammars, invariants, state machine, and atomicity guarantees that govern Mind Mem behavior. All implementations MUST conform to this specification.

---

## 1. Block Grammar (EBNF)

All structured data in Mind Mem is stored as **blocks** — markdown sections with a typed ID header and key-value body.

```ebnf
Block         ::= Header NewLine Body
Header        ::= "[" BlockID "]"
BlockID       ::= Prefix "-" DatePart "-" Counter
                 | Prefix "-" Counter

Prefix        ::= "D" | "T" | "PRJ" | "PER" | "TOOL" | "INC"
                 | "C" | "DREF" | "SIG" | "P" | "I" | "B" | "S"
DatePart      ::= Digit{8}                          (* YYYYMMDD *)
Counter       ::= Digit{3}                          (* 001-999 *)

Body          ::= { Field NewLine }
Field         ::= Key ":" Space Value
Key           ::= Letter { Letter | Digit | "_" }
Value         ::= { AnyChar }                       (* until NewLine *)
Continuation  ::= Space Space { AnyChar }           (* appends to previous Value with \n *)

(* Typed ID examples *)
(* D-20260213-001   = Decision                     *)
(* T-20260213-001   = Task                         *)
(* PRJ-001          = Project                      *)
(* PER-001          = Person                       *)
(* TOOL-001         = Tool                         *)
(* INC-001          = Incident                     *)
(* C-20260213-001   = Contradiction                *)
(* DREF-20260213-001 = Drift Reference             *)
(* SIG-20260213-001 = Signal (auto-captured)       *)
(* P-20260213-001   = Proposal                     *)
(* I-20260213-001   = Impact Record                *)
(* B-2026-W07       = Briefing                     *)
(* S-2026-02-14     = Snapshot                     *)
```

### Required Fields by Type

| Type               | Required Fields                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Decision (D-)      | Date, Status, Scope, Statement, Rationale, Supersedes, Tags, Sources                              |
| Task (T-)          | Date, Status, Title, Priority, Project, Due, Owner, Context, Next, Dependencies, Sources, History |
| Project (PRJ-)     | Name, Status                                                                                      |
| Person (PER-)      | Name                                                                                              |
| Tool (TOOL-)       | Name                                                                                              |
| Incident (INC-)    | Date, Status, Summary                                                                             |
| Contradiction (C-) | Date, DecisionA, DecisionB, Description                                                           |
| Drift (DREF-)      | Date, Type, Source                                                                                |
| Signal (SIG-)      | Date, Type, Source, Status, Excerpt                                                               |
| Proposal (P-)      | Date, Type, Status, Target                                                                        |

### Status Values

```ebnf
DecisionStatus ::= "active" | "superseded" | "revoked"
TaskStatus     ::= "todo" | "doing" | "done" | "blocked" | "canceled"
TaskOwner      ::= "user" | "bot"
SignalStatus   ::= "pending" | "accepted" | "rejected"
ProposalStatus ::= "staged" | "applied" | "rejected" | "deferred" | "expired" | "rolled_back"
```

---

## 2. ConstraintSignature Grammar

ConstraintSignatures encode the semantic intent of decisions as structured metadata, enabling automated contradiction detection.

```ebnf
(* Both "ConstraintSignature:" and "ConstraintSignatures:" are accepted *)
ConstraintSignatures ::= ("ConstraintSignature:" | "ConstraintSignatures:") NewLine
                         { SigItem }

SigItem       ::= "- id:" Space SigID NewLine { SigField NewLine }
SigField      ::= Indent SigKey ":" Space SigValue
Indent        ::= Space Space                       (* 2 spaces *)

(* Required fields *)
SigKey        ::= "id"                              (* unique signature ID, e.g. CS-db-engine *)
                | "domain"                          (* functional domain *)
                | "subject"                         (* what is constrained *)
                | "predicate"                       (* action verb *)
                | "object"                          (* target value *)
                | "modality"                        (* must | must_not | should | should_not | may *)
                | "priority"                        (* 1-10, 10 = highest *)
                | "scope"                           (* {projects: [], channels: [], time: {}} *)
                | "evidence"                        (* justification text *)

(* Optional fields *)
                | "axis"                            (* {key: "domain.subject"} — grouping key *)
                | "relation"                        (* standalone | requires | composes_with | ... *)
                | "enforcement"                     (* invariant | structural | policy | guideline *)
                | "composes_with"                   (* [SigID, ...] — related signatures *)
                | "lifecycle"                       (* {created_by, created_at, expires, review_by} *)
                | "enforced_by"                     (* code path or tool enforcing this *)

(* Enumerated values *)
Domain        ::= "integrity" | "memory" | "retrieval" | "security"
                | "llm_strategy" | "workflow" | "project" | "comms" | "finance" | "other"
Modality      ::= "must" | "must_not" | "should" | "should_not" | "may"
Relation      ::= "standalone" | "requires" | "implies" | "composes_with"
                | "overrides" | "equivalent"
Enforcement   ::= "invariant" | "structural" | "policy" | "guideline"
```

### Contradiction Detection Rule

Two signatures **contradict** if all of the following hold:

```
1. sig_a.axis.key == sig_b.axis.key
   (Fallback: if axis.key is absent, use "{domain}.{subject}")
2. Both parent decisions have Status == "active"
3. Scopes overlap (time ranges and project sets intersect)
4. Neither signature lists the other in its composes_with set
5. At least one of:
   a. Modality conflict: (must vs must_not) or (must_not vs must) → critical
      Also: (should vs should_not), (should vs must_not), etc → medium/low
   b. Competing requirements: same predicate, different objects,
      both modality == "must", and axis.exclusive != false → critical
      (Note: must_not + must_not with different objects is compatible)
      (Note: axis.exclusive defaults to true; set false for additive constraints)
   c. Preference tension: same predicate, different objects,
      both modality == "should" → warning
```

Signatures with `relation: composes_with` or `relation: requires` (both) are exempt.

---

## 3. Proposal Grammar

Proposals are staged mutations that require explicit approval before touching source of truth.

```ebnf
Proposal      ::= Header NewLine ProposalBody
ProposalBody  ::= ProposalId NewLine
                  ProposalType NewLine
                  ProposalTarget NewLine
                  ProposalRisk NewLine
                  ProposalReason NewLine
                  ProposalRollback NewLine
                  ProposalAction NewLine
                  ProposalFingerprint NewLine
                  ProposalStatus NewLine
                  ProposalSources NewLine

ProposalId     ::= "ProposalId:" Space ID
ProposalType   ::= "Type:" Space ("decision" | "task" | "edit")
ProposalTarget ::= "TargetBlock:" Space BlockID
ProposalRisk   ::= "Risk:" Space ("low" | "medium" | "high" | "critical")
ProposalReason ::= "Evidence:" Space EvidenceList
ProposalRollback ::= "Rollback:" Space ("restore_snapshot" | "manual")
ProposalAction ::= "Ops:" Space OpsBlock
ProposalFingerprint ::= "Fingerprint:" Space HexString16
ProposalStatus ::= "Status:" Space ("staged" | "applied" | "rejected"
                  | "deferred" | "expired" | "rolled_back")
ProposalSources ::= "Sources:" Space SourceList
```

### Proposal Invariants

1. **Budget**: No scan run may generate more than `proposal_budget.per_run` proposals
2. **Daily cap**: No day may accumulate more than `proposal_budget.per_day` proposals
3. **Backlog**: Total pending proposals must not exceed `proposal_budget.backlog_limit`
4. **No duplicate**: A proposal targeting the same BlockID with the same Action must not exist in pending state
5. **Defer cooldown**: A deferred proposal may not be re-proposed for `defer_cooldown_days` days

---

## 4. Mode State Machine

```
                    ┌─────────────┐
                    │ detect_only │ ← initial state
                    └──────┬──────┘
                           │ observation_week_clean == true
                           │ explicit user action
                           ▼
                    ┌─────────────┐
                    │   propose   │
                    └──────┬──────┘
                           │ propose_weeks_clean >= 2
                           │ explicit user action
                           ▼
                    ┌─────────────┐
                    │   enforce   │
                    └─────────────┘

Transitions:
  detect_only → propose:   requires flip_gate_week1_clean == true
  propose     → enforce:   requires explicit opt-in in mind-mem.json
  enforce     → propose:   any time (downgrade always safe)
  propose     → detect_only: any time (downgrade always safe)
  enforce     → detect_only: any time (downgrade always safe)

No upward transition happens automatically. All upgrades require explicit action.
```

### Mode Capabilities

| Capability               | detect_only | propose | enforce |
| ------------------------ | :---------: | :-----: | :-----: |
| Run integrity scan       |     Yes     |   Yes   |   Yes   |
| Detect contradictions    |     Yes     |   Yes   |   Yes   |
| Detect drift             |     Yes     |   Yes   |   Yes   |
| Generate proposals       |     No      |   Yes   |   Yes   |
| Apply proposals (manual) |     No      |   Yes   |   Yes   |
| Auto-apply low-risk      |     No      |   No    |   Yes   |
| Supersede decisions      |     No      |   No    |   Yes   |

---

## 5. Apply Engine Atomicity Guarantees

The apply engine provides ACID-like guarantees for memory mutations.

### Transaction Protocol

```
1. PRE-CHECK
   - Validate proposal format
   - Verify target block exists
   - Check mode allows operation
   - Check budget not exceeded

2. SNAPSHOT
   - Save current state of all affected files
   - Record snapshot ID in apply receipt

3. EXECUTE
   - Perform mutation(s) on target file(s)
   - One mutation per proposal (no batching)

4. POST-CHECK
   - Run validate.sh on affected files
   - Verify no new contradictions introduced
   - Verify no structural invariant broken

5. COMMIT or ROLLBACK
   - If post-check passes: mark proposal "applied", log receipt
   - If post-check fails: restore all files from snapshot, mark proposal "failed"
```

### Atomicity Rules

1. **All-or-nothing**: Either all mutations in a proposal succeed, or none do
2. **Snapshot before mutate**: No file is modified until a snapshot is taken
3. **Post-validate**: Every apply is followed by structural validation
4. **Rollback on failure**: If validation fails, all files revert to pre-apply state
5. **Receipt required**: Every apply produces an `APPLY_RECEIPT.md` in its snapshot directory (`intelligence/applied/<timestamp>/`)
6. **No cascade**: One proposal per apply. No proposal may trigger another proposal
7. **Snapshot scope**: Snapshots include workspace state directories (`decisions`, `tasks`, `entities`, `summaries`, `memory`), root-level files (`AGENTS.md`, `MEMORY.md`, `IDENTITY.md`), and `intelligence/` files copied individually. Excluded: `maintenance/` (transient reports), `intelligence/applied/` (prevents recursive nesting)

### Apply Receipt Format

```ebnf
Receipt       ::= Header NewLine ReceiptBody
ReceiptBody   ::= "Date:" Space ISOTimestamp NewLine
                  "Proposal:" Space BlockID NewLine
                  "Action:" Space ActionDesc NewLine
                  "Result:" Space ("applied" | "rolled_back" | "rejected") NewLine
                  "Snapshot:" Space SnapshotID NewLine
                  [ "Risk:" Space RiskLevel NewLine ]
                  [ "TargetBlock:" Space BlockID NewLine ]
                  [ "FilesTouched:" NewLine { "- " FilePath NewLine } ]
                  [ "PreChecks:" Space CheckResult NewLine ]
                  [ "RollbackPlan:" Space PlanDesc NewLine ]
                  [ "DIFF:" NewLine DiffBlock ]
```

---

## 6. Invariant Lock Rules

These invariants MUST hold at all times. Any operation that would violate them MUST be rejected.

### Structural Invariants

| #   | Invariant                                                                                                        | Enforcement                             |
| --- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| S1  | Every BlockID is unique within its file                                                                          | validate.sh (per-file ID format checks) |
| S2  | Every Decision must have Date, Status, Statement, Rationale, Supersedes, Tags, Sources                           | validate.sh                             |
| S3  | Every Task must have Title, Status, Priority, Project, Due, Owner, Context, Next, Dependencies, Sources, History | validate.sh                             |
| S4  | Every active Decision with integrity/security/memory/retrieval tags should have ConstraintSignatures             | validate.sh (warning)                   |
| S5  | Every ConstraintSignature must have id, domain, subject, predicate, object, modality, priority, scope, evidence  | validate.sh (warning)                   |
| S6  | Supersedes field must reference a valid D-ID or be "none"                                                        | validate.sh                             |
| S7  | Active tasks should have AlignsWith or Justification                                                             | validate.sh (warning)                   |
| S8  | Daily logs are append-only — existing content must not be modified                                               | protocol                                |
| S9  | Status values must be from the defined enum for each type                                                        | validate.sh                             |

### Semantic Invariants

| #   | Invariant                                                                                             | Enforcement   |
| --- | ----------------------------------------------------------------------------------------------------- | ------------- |
| M1  | No two active decisions may share the same axis.key with conflicting hard constraints                 | intel_scan.py |
| M2  | Decisions are never edited — they are superseded with a new decision                                  | protocol      |
| M3  | Every memory claim must have a source (no source = no claim)                                          | protocol      |
| M4  | Auto-capture writes to SIGNALS only, never to DECISIONS or TASKS                                      | capture.py    |
| M5  | Mode transitions upward require explicit user action                                                  | state machine |
| M6  | Proposals respect budget limits (per_run, per_day, backlog_limit)                                     | intel_scan.py |
| M7  | Dead decision detection exempts decisions with priority < 7 or enforcement in {invariant, structural} | intel_scan.py |

### Operational Invariants

| #   | Invariant                                                                        | Enforcement       |
| --- | -------------------------------------------------------------------------------- | ----------------- |
| O1  | Apply engine takes snapshot before any mutation                                  | apply_engine.py   |
| O2  | Apply engine rolls back on post-check failure                                    | apply_engine.py   |
| O3  | Every applied proposal produces a receipt in its snapshot directory              | apply_engine.py   |
| O4  | No cascade: proposals cannot trigger other proposals                             | apply_engine.py   |
| O5  | init_workspace.py never overwrites existing files                                | init_workspace.py |
| O6  | validate.sh is idempotent (writes report to `maintenance/validation-report.txt`) | validate.sh       |

---

## 7. File Authority Map

Which scripts are authorized to write to which files:

| File/Directory                 | intel_scan.py | apply_engine.py | capture.py | init_workspace.py | validate.sh |
| ------------------------------ | :-----------: | :-------------: | :--------: | :---------------: | :---------: |
| decisions/                     |     Read      |    **Write**    |    Read    |    **Create**     |    Read     |
| tasks/                         |     Read      |    **Write**    |    Read    |    **Create**     |    Read     |
| entities/                      |     Read      |    **Write**    |    Read    |    **Create**     |    Read     |
| memory/*.md                    |     Read      |      Read       |    Read    |       Read        |    Read     |
| memory/intel-state.json        |   **Write**   |    **Write**    |    Read    |    **Create**     |    Read     |
| intelligence/CONTRADICTIONS.md |   **Write**   |      Read       |    Read    |    **Create**     |    Read     |
| intelligence/DRIFT.md          |   **Write**   |      Read       |    Read    |    **Create**     |    Read     |
| intelligence/SIGNALS.md        |     Read      |    **Write**    | **Write**  |    **Create**     |    Read     |
| intelligence/IMPACT.md         |   **Write**   |      Read       |    Read    |    **Create**     |    Read     |
| intelligence/AUDIT.md          |     Read      |    **Write**    |    Read    |    **Create**     |    Read     |
| intelligence/SCAN_LOG.md       |   **Write**   |      Read       |    Read    |    **Create**     |    Read     |
| intelligence/proposed/         |   **Write**   |    **Write**    |    Read    |    **Create**     |    Read     |
| intelligence/state/snapshots/  |   **Write**   |    **Write**    |    Read    |    **Create**     |    Read     |
| mind-mem.json                  |     Read      |      Read       |    Read    |    **Create**     |    Read     |

**Key rule**: `capture.py` may only write to `intelligence/SIGNALS.md`. It has no write access to any other file.

---

## 8. Versioning

This specification follows semantic versioning:
- **Major**: Breaking changes to grammar or invariants
- **Minor**: New optional fields or capabilities
- **Patch**: Clarifications or corrections

Current version: **1.0.3**

---

*Copyright 2026 STARGA Inc. MIT License.*
