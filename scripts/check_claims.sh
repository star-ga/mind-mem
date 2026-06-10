#!/usr/bin/env bash
#
# Cross-repo docs-claim regression gate (mind-mem side).
#
# Runs the canonical mind gate (../mind/scripts/check_claims.py) against THIS
# repo's public surfaces, using the shared capability manifest in the mind repo
# as the single source of truth. This stops mind-mem docs from silently drifting
# back to claims the wider MIND project has already fixed (IR version strings,
# runtime-boundary wording, tool counts).
#
# Mode is "phrases": forbidden-phrase + canonical-IR checks only. The shared
# manifest also carries mind-specific [counts] (e.g. Rust test-file globs) that
# don't apply to a Python repo, so the count gate is intentionally skipped here.
#
# Best-effort: if the mind gate isn't checked out alongside this repo, the script
# prints a skip notice and exits 0 (CI without the sibling tree is not failed).
#
# Usage:
#   scripts/check_claims.sh            # check README.md + docs/**/*.md
#   MIND_REPO=/path/to/mind scripts/check_claims.sh   # override sibling location
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIND_REPO="${MIND_REPO:-$HERE/../mind}"
GATE="$MIND_REPO/scripts/check_claims.py"
CAPS="$MIND_REPO/config/capabilities.toml"

if [[ ! -f "$GATE" || ! -f "$CAPS" ]]; then
  echo "check_claims.sh: mind gate not found at $GATE — skipping (clone star-ga/mind beside this repo to enable)."
  exit 0
fi

CHECK_CLAIMS_ROOT="$HERE" \
CHECK_CLAIMS_CAPS="$CAPS" \
CHECK_CLAIMS_SURFACES="README.md:docs/**/*.md" \
CHECK_CLAIMS_MODE="phrases" \
  python3 "$GATE"
