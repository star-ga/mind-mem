#!/usr/bin/env bash
# Mem OS Smoke Test — end-to-end verification
# Run: bash scripts/smoke_test.sh
# Creates a temp workspace, runs init → validate → scan → recall → capture, then cleans up.
# Exit code: 0 = all passed, 1 = failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TMPWS=$(mktemp -d "${TMPDIR:-/tmp}/memos_smoke.XXXXXX")
trap 'rm -rf "$TMPWS"' EXIT

PASS=0
FAIL=0

check() {
  if eval "$2" > /dev/null 2>&1; then
    echo "  PASS: $1"
    PASS=$((PASS+1))
  else
    echo "  FAIL: $1"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Mem OS Smoke Test ==="
echo "Temp workspace: $TMPWS"
echo ""

# 1. Init
echo "--- init ---"
python3 "$REPO_ROOT/scripts/init_workspace.py" "$TMPWS" > /dev/null 2>&1
check "init creates mind-mem.json" "[ -f '$TMPWS/mind-mem.json' ]"
check "init creates DECISIONS.md" "[ -f '$TMPWS/decisions/DECISIONS.md' ]"
check "init creates validate.sh" "[ -f '$TMPWS/maintenance/validate.sh' ]"

# 2. Validate
echo "--- validate ---"
bash "$TMPWS/maintenance/validate.sh" "$TMPWS" > /dev/null 2>&1
check "validate exits 0 on fresh workspace" "bash '$TMPWS/maintenance/validate.sh' '$TMPWS'"
check "validate report exists" "[ -f '$TMPWS/maintenance/validation-report.txt' ]"
check "validate report shows 0 issues" "grep -q '0 issues' '$TMPWS/maintenance/validation-report.txt'"

# 3. Scan
echo "--- scan ---"
python3 "$TMPWS/maintenance/intel_scan.py" "$TMPWS" > /dev/null 2>&1
check "scan exits 0" "python3 '$TMPWS/maintenance/intel_scan.py' '$TMPWS'"
check "scan creates intel-state.json" "[ -f '$TMPWS/memory/intel-state.json' ]"

# 4. Recall
echo "--- recall ---"
RECALL_OUT=$(python3 "$TMPWS/maintenance/recall.py" --query "test" --workspace "$TMPWS" 2>&1 || true)
check "recall runs without error" "python3 '$TMPWS/maintenance/recall.py' --query test --workspace '$TMPWS'"

# 5. Capture
echo "--- capture ---"
CAPTURE_OUT=$(python3 "$TMPWS/maintenance/capture.py" "$TMPWS" 2>&1 || true)
check "capture runs without error" "python3 '$TMPWS/maintenance/capture.py' '$TMPWS'"

# 6. Unit tests
echo "--- pytest ---"
check "all unit tests pass" "python3 -m pytest '$REPO_ROOT/tests/' -q"

echo ""
echo "═══════════════════════════════════════"
echo "SMOKE TEST: $PASS passed | $FAIL failed"
echo "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
