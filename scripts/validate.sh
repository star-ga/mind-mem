#!/usr/bin/env bash
# mind-mem Integrity Validator v1.1
# Based on: Integrity Validator Checklist v1.0 (Telegram:msg-3268)
# Run: bash maintenance/validate.sh [workspace_path]
# Creates: maintenance/validation-report.txt
# Exit code: 0 = clean, 1 = issues found

set -uo pipefail

WS="${1:-.}"

# Workspace detection: warn if running in repo root instead of initialized workspace
if [[ ! -f "$WS/mind-mem.json" ]]; then
  echo "ERROR: No mind-mem.json found in '$WS'."
  echo ""
  echo "This does not appear to be an initialized mind-mem workspace."
  echo "To initialize a workspace, run:"
  echo ""
  echo "  python3 scripts/init_workspace.py /path/to/your/workspace"
  echo ""
  echo "Then validate with:"
  echo ""
  echo "  bash maintenance/validate.sh /path/to/your/workspace"
  exit 1
fi

REPORT="$WS/maintenance/validation-report.txt"
ISSUES=0
CHECKS=0
PASSED=0
WARNINGS=0

log()     { echo "$1" >> "$REPORT"; }
pass()    { CHECKS=$((CHECKS+1)); PASSED=$((PASSED+1)); log "  PASS: $1"; }
fail()    { CHECKS=$((CHECKS+1)); ISSUES=$((ISSUES+1)); log "  FAIL: $1"; }
warn()    { WARNINGS=$((WARNINGS+1)); log "  WARN: $1"; }
section() { log ""; log "=== $1 ==="; }

# Ensure report directory exists (graceful on uninitialized workspaces)
REPORT_DIR=$(dirname "$REPORT")
if [[ ! -d "$REPORT_DIR" ]]; then
  mkdir -p "$REPORT_DIR"
fi

cat > "$REPORT" <<EOF
mind-mem Integrity Validation Report v1.1
Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Workspace: $WS
EOF

# ═══════════════════════════════════════════
section "0. FILE STRUCTURE"
# ═══════════════════════════════════════════

for f in decisions/DECISIONS.md tasks/TASKS.md entities/projects.md entities/people.md entities/tools.md entities/incidents.md; do
  if [[ -f "$WS/$f" ]]; then
    pass "$f exists"
  else
    fail "$f MISSING"
  fi
done

weekly_count=$(find "$WS/summaries/weekly" -name '*.md' 2>/dev/null | wc -l || true)
if [[ "$weekly_count" -gt 0 ]]; then
  pass "summaries/weekly/ has $weekly_count file(s)"
else
  warn "summaries/weekly/ has no .md files (expected for new workspaces)"
fi

if [[ -f "$WS/memory/maint-state.json" ]]; then
  pass "memory/maint-state.json exists"
else
  fail "memory/maint-state.json MISSING"
fi

if [[ -f "$WS/MEMORY.md" ]]; then
  pass "MEMORY.md exists"
  if grep -q 'Memory Protocol v1.0' "$WS/MEMORY.md"; then
    pass "MEMORY.md has Protocol v1.0 header"
  else
    fail "MEMORY.md missing Protocol v1.0 header"
  fi
else
  fail "MEMORY.md MISSING"
fi

# ═══════════════════════════════════════════
section "1. DECISIONS — IDs, fields, values"
# ═══════════════════════════════════════════

DEC_FILE="$WS/decisions/DECISIONS.md"
if [[ -f "$DEC_FILE" ]]; then
  dec_count=$(grep -cE '^\[D-[0-9]{8}-[0-9]{3}\]$' "$DEC_FILE" || true)
  bad_dec=$(grep -E '^\[D-' "$DEC_FILE" | grep -vE '^\[D-[0-9]{8}-[0-9]{3}\]$' || true)

  if [[ -z "$bad_dec" ]]; then
    pass "All $dec_count decision IDs match [D-YYYYMMDD-###]"
  else
    fail "Malformed decision IDs:"; echo "$bad_dec" | while read -r line; do log "    $line"; done
  fi

  # Required fields — per-block validation (not global count)
  for field in "Date:" "Status:" "Scope:" "Statement:" "Rationale:" "Supersedes:" "Tags:" "Sources:"; do
    # Use awk to count blocks missing this field
    missing=$(awk -v f="$field" '
      /^\[D-[0-9]{8}-[0-9]{3}\]$/ { if (block && !found) m++; block=1; found=0 }
      block && $0 ~ "^"f { found=1 }
      END { if (block && !found) m++; print m+0 }
    ' "$DEC_FILE")
    if [[ "$missing" -eq 0 ]]; then
      pass "Decisions: $field present in all $dec_count blocks"
    else
      fail "Decisions: $field missing in $missing/$dec_count blocks"
    fi
  done

  # Value validation: Status
  valid_status=$(grep -cE '^Status:[[:space:]]*(active|superseded|revoked)[[:space:]]*$' "$DEC_FILE" || true)
  bad_status=$(grep -E '^Status:' "$DEC_FILE" | grep -vE '^Status:[[:space:]]*(active|superseded|revoked)[[:space:]]*$' || true)
  if [[ -z "$bad_status" ]]; then
    pass "Decisions: all Status values valid ($valid_status)"
  else
    fail "Decisions: invalid Status values:"; echo "$bad_status" | while read -r line; do log "    $line"; done
  fi

  # Value validation: Scope
  valid_scope=$(grep -cE '^Scope:[[:space:]]*(global|project:[^ ]+|channel:[^ ]+)[[:space:]]*$' "$DEC_FILE" || true)
  bad_scope=$(grep -E '^Scope:' "$DEC_FILE" | grep -vE '^Scope:[[:space:]]*(global|project:[^ ]+|channel:[^ ]+)[[:space:]]*$' || true)
  if [[ -z "$bad_scope" ]]; then
    pass "Decisions: all Scope values valid ($valid_scope)"
  else
    fail "Decisions: invalid Scope values:"; echo "$bad_scope" | while read -r line; do log "    $line"; done
  fi

  # Supersedes: must be "none" or valid D-ID
  valid_sup=$(grep -cE '^Supersedes:[[:space:]]*(D-[0-9]{8}-[0-9]{3}|none)[[:space:]]*$' "$DEC_FILE" || true)
  bad_sup=$(grep -E '^Supersedes:' "$DEC_FILE" | grep -vE '^Supersedes:[[:space:]]*(D-[0-9]{8}-[0-9]{3}|none)[[:space:]]*$' || true)
  if [[ -z "$bad_sup" ]]; then
    pass "Decisions: all Supersedes values valid ($valid_sup)"
  else
    fail "Decisions: invalid Supersedes values:"; echo "$bad_sup" | while read -r line; do log "    $line"; done
  fi
fi

# ═══════════════════════════════════════════
section "2. TASKS — IDs, fields, values"
# ═══════════════════════════════════════════

TASK_FILE="$WS/tasks/TASKS.md"
if [[ -f "$TASK_FILE" ]]; then
  task_count=$(grep -cE '^\[T-[0-9]{8}-[0-9]{3}\]$' "$TASK_FILE" || true)
  bad_task=$(grep -E '^\[T-' "$TASK_FILE" | grep -vE '^\[T-[0-9]{8}-[0-9]{3}\]$' || true)

  if [[ -z "$bad_task" ]]; then
    pass "All $task_count task IDs match [T-YYYYMMDD-###]"
  else
    fail "Malformed task IDs:"; echo "$bad_task" | while read -r line; do log "    $line"; done
  fi

  # Required fields — per-block validation (not global count)
  for field in "Date:" "Title:" "Status:" "Priority:" "Project:" "Due:" "Owner:" "Context:" "Next:" "Dependencies:" "Sources:" "History:"; do
    missing=$(awk -v f="$field" '
      /^\[T-[0-9]{8}-[0-9]{3}\]$/ { if (block && !found) m++; block=1; found=0 }
      block && $0 ~ "^"f { found=1 }
      END { if (block && !found) m++; print m+0 }
    ' "$TASK_FILE")
    if [[ "$missing" -eq 0 ]]; then
      pass "Tasks: $field present in all $task_count blocks"
    else
      fail "Tasks: $field missing in $missing/$task_count blocks"
    fi
  done

  # Value validation: Status
  valid_ts=$(grep -cE '^Status:[[:space:]]*(todo|doing|blocked|done|canceled)[[:space:]]*$' "$TASK_FILE" || true)
  bad_ts=$(grep -E '^Status:' "$TASK_FILE" | grep -vE '^Status:[[:space:]]*(todo|doing|blocked|done|canceled)[[:space:]]*$' || true)
  if [[ -z "$bad_ts" ]]; then
    pass "Tasks: all Status values valid ($valid_ts)"
  else
    fail "Tasks: invalid Status values:"; echo "$bad_ts" | while read -r line; do log "    $line"; done
  fi

  # Value validation: Priority
  valid_pri=$(grep -cE '^Priority:[[:space:]]*P[0-3][[:space:]]*$' "$TASK_FILE" || true)
  bad_pri=$(grep -E '^Priority:' "$TASK_FILE" | grep -vE '^Priority:[[:space:]]*P[0-3][[:space:]]*$' || true)
  if [[ -z "$bad_pri" ]]; then
    pass "Tasks: all Priority values valid ($valid_pri)"
  else
    fail "Tasks: invalid Priority values:"; echo "$bad_pri" | while read -r line; do log "    $line"; done
  fi

  # Value validation: Owner
  valid_own=$(grep -cE '^Owner:[[:space:]]*(user|bot)[[:space:]]*$' "$TASK_FILE" || true)
  bad_own=$(grep -E '^Owner:' "$TASK_FILE" | grep -vE '^Owner:[[:space:]]*(user|bot)[[:space:]]*$' || true)
  if [[ -z "$bad_own" ]]; then
    pass "Tasks: all Owner values valid ($valid_own)"
  else
    fail "Tasks: invalid Owner values:"; echo "$bad_own" | while read -r line; do log "    $line"; done
  fi

  # History: must have at least one dated entry per block
  history_entries=$(grep -cE '^[[:space:]]*-[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2}:[[:space:]].+' "$TASK_FILE" || true)
  if [[ "$history_entries" -ge "$task_count" ]]; then
    pass "Tasks: History has dated entries ($history_entries entries across $task_count blocks)"
  else
    fail "Tasks: some blocks may lack dated History entries ($history_entries/$task_count)"
  fi
fi

# ═══════════════════════════════════════════
section "3. ENTITIES — IDs & required fields"
# ═══════════════════════════════════════════

# Projects: PRJ-slug
if [[ -f "$WS/entities/projects.md" ]]; then
  prj_count=$(grep -cE '^\[PRJ-[a-z0-9-]+\]$' "$WS/entities/projects.md" || true)
  bad_prj=$(grep -E '^\[PRJ-' "$WS/entities/projects.md" | grep -vE '^\[PRJ-[a-z0-9-]+\]$' || true)
  if [[ -z "$bad_prj" ]]; then
    pass "Projects: all $prj_count IDs match [PRJ-slug]"
  else
    fail "Projects: malformed IDs: $bad_prj"
  fi
fi

# People: PER-slug
if [[ -f "$WS/entities/people.md" ]]; then
  per_count=$(grep -cE '^\[PER-[a-z0-9-]+\]$' "$WS/entities/people.md" || true)
  bad_per=$(grep -E '^\[PER-' "$WS/entities/people.md" | grep -vE '^\[PER-[a-z0-9-]+\]$' || true)
  if [[ -z "$bad_per" ]]; then
    pass "People: all $per_count IDs match [PER-slug]"
  else
    fail "People: malformed IDs: $bad_per"
  fi
fi

# Tools: TOOL-slug
if [[ -f "$WS/entities/tools.md" ]]; then
  tool_count=$(grep -cE '^\[TOOL-[a-z0-9-]+\]$' "$WS/entities/tools.md" || true)
  bad_tool=$(grep -E '^\[TOOL-' "$WS/entities/tools.md" | grep -vE '^\[TOOL-[a-z0-9-]+\]$' || true)
  if [[ -z "$bad_tool" ]]; then
    pass "Tools: all $tool_count IDs match [TOOL-slug]"
  else
    fail "Tools: malformed IDs: $bad_tool"
  fi
fi

# Incidents: INC-YYYYMMDD-slug + required fields
if [[ -f "$WS/entities/incidents.md" ]]; then
  inc_count=$(grep -cE '^\[INC-[0-9]{8}-[a-z0-9-]+\]$' "$WS/entities/incidents.md" || true)
  bad_inc=$(grep -E '^\[INC-' "$WS/entities/incidents.md" | grep -vE '^\[INC-[0-9]{8}-[a-z0-9-]+\]$' || true)
  if [[ -z "$bad_inc" ]]; then
    pass "Incidents: all $inc_count IDs match [INC-YYYYMMDD-slug]"
  else
    fail "Incidents: malformed IDs: $bad_inc"
  fi

  for field in "Date:" "Title:" "Impact:" "Summary:" "RootCause:" "Fix:" "Prevention:" "Sources:"; do
    fc=$(grep -c "^$field" "$WS/entities/incidents.md" || true)
    if [[ "$fc" -ge "$inc_count" ]]; then
      pass "Incidents: $field present ($fc/$inc_count)"
    else
      fail "Incidents: $field missing in some blocks ($fc/$inc_count)"
    fi
  done
fi

# ═══════════════════════════════════════════
section "4. WEEKLY SUMMARIES — IDs & sections"
# ═══════════════════════════════════════════

for wfile in "$WS/summaries/weekly"/*.md; do
  [[ -f "$wfile" ]] || continue
  fname=$(basename "$wfile")

  wid=$(grep -cE '^\[W-[0-9]{4}-W[0-9]{2}\]$' "$wfile" || true)
  if [[ "$wid" -gt 0 ]]; then
    pass "$fname: valid [W-YYYY-W##] ID"
  else
    fail "$fname: missing or malformed [W-YYYY-W##] ID"
  fi

  for sect in "Range:" "Highlights:" "Progress:" "Sources:"; do
    if grep -q "^$sect" "$wfile"; then
      pass "$fname: has $sect"
    else
      fail "$fname: missing $sect"
    fi
  done
done

# ═══════════════════════════════════════════
section "5. PROVENANCE — Sources not empty"
# ═══════════════════════════════════════════

for mfile in "$WS/decisions/DECISIONS.md" "$WS/tasks/TASKS.md" "$WS/entities/incidents.md"; do
  [[ -f "$mfile" ]] || continue
  fname=$(basename "$mfile")

  blocks=$(grep -cE '^\[' "$mfile" || true)
  sources=$(grep -c '^Sources:' "$mfile" || true)

  if [[ "$sources" -ge "$blocks" ]]; then
    pass "$fname: all $blocks blocks have Sources: ($sources)"
  else
    fail "$fname: blocks without Sources: ($sources/$blocks)"
  fi

  # Check for empty Sources (Sources: followed by non-list line)
  empty_src=$(awk '
    /^Sources:[[:space:]]*$/ { src_line=NR; in_src=1; next }
    in_src && /^-[[:space:]]/ { in_src=0; next }
    in_src && /^[A-Z][A-Za-z]+:/ { print "line " src_line; in_src=0 }
    in_src && /^\[/ { print "line " src_line; in_src=0 }
  ' "$mfile" || true)

  if [[ -z "$empty_src" ]]; then
    pass "$fname: no empty Sources: blocks"
  else
    fail "$fname: empty Sources: at $empty_src"
  fi
done

# ═══════════════════════════════════════════
section "6. CROSS-REFERENCE INTEGRITY"
# ═══════════════════════════════════════════

# Build truth sets
TMP=$(mktemp -d "${TMPDIR:-/tmp}/memos_validate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

grep -hoE '^\[D-[0-9]{8}-[0-9]{3}\]$' "$WS/decisions/DECISIONS.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_d.txt" || true
grep -hoE '^\[T-[0-9]{8}-[0-9]{3}\]$' "$WS/tasks/TASKS.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_t.txt" || true
grep -hoE '^\[INC-[0-9]{8}-[a-z0-9-]+\]$' "$WS/entities/incidents.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_inc.txt" || true
grep -hoE '^\[PRJ-[a-z0-9-]+\]$' "$WS/entities/projects.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_prj.txt" || true
grep -hoE '^\[PER-[a-z0-9-]+\]$' "$WS/entities/people.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_per.txt" || true
grep -hoE '^\[TOOL-[a-z0-9-]+\]$' "$WS/entities/tools.md" 2>/dev/null | tr -d '[]' | sort -u > "$TMP/ids_tool.txt" || true

# S1: Check per-file ID uniqueness
for idcheck in \
  "decisions/DECISIONS.md:D-[0-9]{8}-[0-9]{3}" \
  "tasks/TASKS.md:T-[0-9]{8}-[0-9]{3}" \
  "entities/incidents.md:INC-[0-9]{8}-[a-z0-9-]+" \
  "entities/projects.md:PRJ-[a-z0-9-]+" \
  "entities/people.md:PER-[a-z0-9-]+" \
  "entities/tools.md:TOOL-[a-z0-9-]+"; do
  IFS=':' read -r idfile idpat <<< "$idcheck"
  [[ -f "$WS/$idfile" ]] || continue
  total_ids=$(grep -oE "^\[$idpat\]$" "$WS/$idfile" 2>/dev/null | wc -l || echo 0)
  unique_ids=$(grep -oE "^\[$idpat\]$" "$WS/$idfile" 2>/dev/null | sort -u | wc -l || echo 0)
  # Sanitize to integer
  total_ids=$(echo "$total_ids" | tr -d '[:space:]')
  unique_ids=$(echo "$unique_ids" | tr -d '[:space:]')
  if [[ "$total_ids" -gt "$unique_ids" ]]; then
    fail "S1: Duplicate BlockIDs in $idfile ($total_ids total, $unique_ids unique)"
  elif [[ "$total_ids" -gt 0 ]]; then
    pass "S1: All BlockIDs unique in $idfile ($total_ids)"
  fi
done

# Collect all references across corpus (exclude schema/comment lines starting with >)
grep -RnE '\b(D-[0-9]{8}-[0-9]{3}|T-[0-9]{8}-[0-9]{3}|INC-[0-9]{8}-[a-z0-9-]+|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+)\b' \
  "$WS/decisions" "$WS/tasks" "$WS/entities" "$WS/summaries" "$WS/maintenance/MAINTENANCE.md" \
  --include='*.md' 2>/dev/null | grep -vE '^\S+:.*>[[:space:]]+Schema:' | grep -hoE '\b(D-[0-9]{8}-[0-9]{3}|T-[0-9]{8}-[0-9]{3}|INC-[0-9]{8}-[a-z0-9-]+|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+)\b' \
  | sort -u > "$TMP/all_refs.txt" || true

dangling=0

for prefix_label in "D-:ids_d:DECISION" "T-:ids_t:TASK" "INC-:ids_inc:INCIDENT" "PRJ-:ids_prj:PROJECT" "PER-:ids_per:PERSON" "TOOL-:ids_tool:TOOL"; do
  IFS=':' read -r prefix idfile label <<< "$prefix_label"
  grep -E "^$prefix" "$TMP/all_refs.txt" 2>/dev/null | sort -u > "$TMP/refs_${prefix}.txt" || true
  missing=$(comm -23 "$TMP/refs_${prefix}.txt" "$TMP/$idfile.txt" 2>/dev/null || true)
  if [[ -n "$missing" ]]; then
    while read -r mid; do
      fail "MISSING $label: $mid (referenced but not defined)"
      dangling=$((dangling+1))
    done <<< "$missing"
  fi
done

if [[ "$dangling" -eq 0 ]]; then
  pass "All cross-references resolve to defined IDs"
fi

# ═══════════════════════════════════════════
section "7. SUPERSEDED CHAIN"
# ═══════════════════════════════════════════

if [[ -f "$DEC_FILE" ]]; then
  # Check Supersedes format
  sup_valid=$(grep -cE '^Supersedes:[[:space:]]*(D-[0-9]{8}-[0-9]{3}|none)[[:space:]]*$' "$DEC_FILE" || true)
  sup_bad=$(grep -E '^Supersedes:' "$DEC_FILE" | grep -vE '^Supersedes:[[:space:]]*(D-[0-9]{8}-[0-9]{3}|none)[[:space:]]*$' || true)
  if [[ -z "$sup_bad" ]]; then
    pass "Supersedes: all values are 'none' or valid D-ID ($sup_valid)"
  else
    fail "Supersedes: invalid format:"; echo "$sup_bad" | while read -r line; do log "    $line"; done
  fi

  # Check targets exist
  sup_targets=$(grep -E '^Supersedes:' "$DEC_FILE" | grep -oE 'D-[0-9]{8}-[0-9]{3}' || true)
  chain_ok=true
  for sid in $sup_targets; do
    if ! grep -qF "[$sid]" "$DEC_FILE"; then
      fail "Supersedes target $sid not found in DECISIONS.md"
      chain_ok=false
    fi
  done
  if $chain_ok; then
    pass "Superseded chain: all targets exist"
  fi
fi

# ═══════════════════════════════════════════
section "8. PROSE-ONLY REFERENCE WARNING"
# ═══════════════════════════════════════════

log "  Scanning for decision-like language without D-/T- IDs nearby..."
prose_hits=$(grep -RInE '\b(decided|agreed|commit(ment)?|constraint|deadline|budget|invariant)\b' \
  "$WS/memory" "$WS/summaries" "$WS/docs" "$WS/specs" "$WS/hackathon" \
  --include='*.md' 2>/dev/null | head -20 || true)

if [[ -n "$prose_hits" ]]; then
  prose_count=$(echo "$prose_hits" | wc -l)
  warn "$prose_count lines with decision-like language in logs/docs (review manually)"
  echo "$prose_hits" | head -5 | while read -r line; do log "    $line"; done
  if [[ "$prose_count" -gt 5 ]]; then
    log "    ... and $((prose_count - 5)) more"
  fi
else
  pass "No obvious prose-only decision references found"
fi

# ═══════════════════════════════════════════
section "9. v2.0 CHECKS (warnings only)"
# ═══════════════════════════════════════════

# V2.1: Active decisions with integrity/security/memory/retrieval tags must have ConstraintSignatures
if [[ -f "$DEC_FILE" ]]; then
  # Get IDs of active decisions with relevant tags
  needs_sig=$(MIND_MEM_WS="$WS" MIND_MEM_DECFILE="$DEC_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_DECFILE'])
required_tags = {'integrity','security','memory','retrieval'}
for b in blocks:
    if b.get('Status') != 'active': continue
    tags = set(t.strip() for t in b.get('Tags','').split(','))
    if tags & required_tags:
        sigs = b.get('ConstraintSignatures', [])
        if not sigs:
            print(b['_id'])
" 2>/dev/null || true)

  if [[ -z "$needs_sig" ]]; then
    pass "V2.1: All relevant active decisions have ConstraintSignatures"
  else
    echo "$needs_sig" | while read -r did; do
      warn "V2.1: $did tagged integrity/security/memory/retrieval but no ConstraintSignatures"
    done
  fi
fi

# V2.2: ConstraintSignatures have required fields
if [[ -f "$DEC_FILE" ]]; then
  sig_issues=$(MIND_MEM_WS="$WS" MIND_MEM_DECFILE="$DEC_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_DECFILE'])
required = ['id','domain','subject','predicate','object','modality','priority','scope','evidence']
for b in blocks:
    for sig in b.get('ConstraintSignatures', []):
        missing = [f for f in required if f not in sig or sig[f] in (None, '', [])]
        if missing:
            print(f\"{b['_id']}:{sig.get('id','?')}: missing {','.join(missing)}\")
" 2>/dev/null || true)

  if [[ -z "$sig_issues" ]]; then
    pass "V2.2: All ConstraintSignatures have required fields"
  else
    echo "$sig_issues" | while read -r issue; do
      warn "V2.2: $issue"
    done
  fi
fi

# V2.3: domain and modality in valid enums, priority in 1-10
if [[ -f "$DEC_FILE" ]]; then
  enum_issues=$(MIND_MEM_WS="$WS" MIND_MEM_DECFILE="$DEC_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_DECFILE'])
valid_domains = {'integrity','memory','retrieval','security','llm_strategy','workflow','project','comms','finance','other'}
valid_modalities = {'must','must_not','should','should_not','may'}
for b in blocks:
    for sig in b.get('ConstraintSignatures', []):
        sid = sig.get('id','?')
        d = sig.get('domain','')
        if d not in valid_domains:
            print(f\"{b['_id']}:{sid}: invalid domain '{d}'\")
        m = sig.get('modality','')
        if m not in valid_modalities:
            print(f\"{b['_id']}:{sid}: invalid modality '{m}'\")
        p = sig.get('priority', 0)
        if not isinstance(p, int) or p < 1 or p > 10:
            print(f\"{b['_id']}:{sid}: priority {p} out of range 1-10\")
" 2>/dev/null || true)

  if [[ -z "$enum_issues" ]]; then
    pass "V2.3: All domains, modalities, priorities valid"
  else
    echo "$enum_issues" | while read -r issue; do
      warn "V2.3: $issue"
    done
  fi
fi

# V2.4: AlignsWith field present on active tasks
if [[ -f "$TASK_FILE" ]]; then
  align_issues=$(MIND_MEM_WS="$WS" MIND_MEM_TASKFILE="$TASK_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_TASKFILE'])
for b in blocks:
    if b.get('Status') in ('todo','doing','blocked'):
        aligns = b.get('AlignsWith','')
        justification = b.get('Justification','')
        if (not aligns or aligns == 'none') and not justification:
            print(b['_id'])
" 2>/dev/null || true)

  if [[ -z "$align_issues" ]]; then
    pass "V2.4: All active tasks have AlignsWith or Justification"
  else
    echo "$align_issues" | while read -r tid; do
      warn "V2.4: $tid missing AlignsWith and Justification"
    done
  fi
fi

# V2.5: intelligence/ directory structure exists
for ifile in SIGNALS.md CONTRADICTIONS.md DRIFT.md IMPACT.md BRIEFINGS.md AUDIT.md; do
  if [[ -f "$WS/intelligence/$ifile" ]]; then
    pass "V2.5: intelligence/$ifile exists"
  else
    warn "V2.5: intelligence/$ifile MISSING"
  fi
done

if [[ -f "$WS/memory/intel-state.json" ]]; then
  pass "V2.5: memory/intel-state.json exists"
else
  warn "V2.5: memory/intel-state.json MISSING"
fi

# V2.6: v1.1 signature fields: axis.key, relation, enforcement present
if [[ -f "$DEC_FILE" ]]; then
  v11_issues=$(MIND_MEM_WS="$WS" MIND_MEM_DECFILE="$DEC_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_DECFILE'])
valid_relations = {'standalone','requires','implies','composes_with','overrides','equivalent'}
valid_enforcement = {'invariant','structural','policy','guideline'}
for b in blocks:
    for sig in b.get('ConstraintSignatures', []):
        sid = sig.get('id','?')
        ax = sig.get('axis', {})
        if not isinstance(ax, dict) or not ax.get('key'):
            print(f\"{b['_id']}:{sid}: missing axis.key\")
        rel = sig.get('relation', '')
        if rel and rel not in valid_relations:
            print(f\"{b['_id']}:{sid}: invalid relation '{rel}'\")
        enf = sig.get('enforcement', '')
        if enf and enf not in valid_enforcement:
            print(f\"{b['_id']}:{sid}: invalid enforcement '{enf}'\")
" 2>/dev/null || true)

  if [[ -z "$v11_issues" ]]; then
    pass "V2.6: All signatures have valid axis.key, relation, enforcement"
  else
    echo "$v11_issues" | while read -r issue; do
      warn "V2.6: $issue"
    done
  fi
fi

# V2.7: lifecycle.created_by present on all signatures
if [[ -f "$DEC_FILE" ]]; then
  lc_issues=$(MIND_MEM_WS="$WS" MIND_MEM_DECFILE="$DEC_FILE" python3 -c "
import os, sys; sys.path.insert(0, os.path.join(os.environ['MIND_MEM_WS'], 'scripts'))
from block_parser import parse_file
blocks = parse_file(os.environ['MIND_MEM_DECFILE'])
for b in blocks:
    for sig in b.get('ConstraintSignatures', []):
        sid = sig.get('id','?')
        lc = sig.get('lifecycle', {})
        if not isinstance(lc, dict) or not lc.get('created_by'):
            print(f\"{b['_id']}:{sid}: missing lifecycle.created_by\")
" 2>/dev/null || true)

  if [[ -z "$lc_issues" ]]; then
    pass "V2.7: All signatures have lifecycle.created_by"
  else
    echo "$lc_issues" | while read -r issue; do
      warn "V2.7: $issue"
    done
  fi
fi

# V2.8: intelligence/proposed/ directory exists
if [[ -d "$WS/intelligence/proposed" ]]; then
  pass "V2.8: intelligence/proposed/ directory exists"
  # V2.9: Staged proposals must have Fingerprint field
  for pfile in "$WS"/intelligence/proposed/*_PROPOSED.md; do
    [[ -f "$pfile" ]] || continue
    staged_count=$(grep -c "^Status: staged" "$pfile" 2>/dev/null || echo 0)
    fp_count=$(grep -c "^Fingerprint: " "$pfile" 2>/dev/null || echo 0)
    # Sanitize to integer
    staged_count=$(echo "$staged_count" | tr -d '[:space:]')
    fp_count=$(echo "$fp_count" | tr -d '[:space:]')
    if [[ "$staged_count" -gt 0 && "$fp_count" -lt "$staged_count" ]]; then
      warn "V2.9: $(basename "$pfile") has staged proposals missing Fingerprint field"
    elif [[ "$staged_count" -gt 0 ]]; then
      pass "V2.9: $(basename "$pfile") staged proposals have Fingerprint"
    fi
  done
else
  warn "V2.8: intelligence/proposed/ directory MISSING"
fi

# ═══════════════════════════════════════════
log ""
log "═══════════════════════════════════════"
log "TOTAL: $CHECKS checks | $PASSED passed | $ISSUES issues | $WARNINGS warnings"
log "═══════════════════════════════════════"

cat "$REPORT"

if [[ "$ISSUES" -gt 0 ]]; then
  exit 1
else
  exit 0
fi