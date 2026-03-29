#!/usr/bin/env bash
# anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
# Install: ln -sf ../../scripts/anatomy-hook.sh .git/hooks/pre-commit
#   or:   cp scripts/anatomy-hook.sh .git/hooks/pre-commit
#
# If ANATOMY.md is stale after staged changes, regenerates and stages it.
# Author: STARGA Inc <noreply@star.ga>

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" \
  || ROOT_DIR="$(git rev-parse --show-toplevel)"

ANATOMY="$ROOT_DIR/ANATOMY.md"
SCRIPT="$ROOT_DIR/scripts/anatomy.sh"

# Only run if anatomy.sh exists in this repo
[ -x "$SCRIPT" ] || exit 0

# Check if any tracked source files are staged (not just ANATOMY.md itself)
staged_files=$(git diff --cached --name-only --diff-filter=ACMR | grep -v '^ANATOMY.md$' || true)
[ -z "$staged_files" ] && exit 0

# Regenerate
cd "$ROOT_DIR"
"$SCRIPT" . --output ANATOMY.md 2>/dev/null

# Stage the updated ANATOMY.md if it changed
if ! git diff --quiet -- ANATOMY.md 2>/dev/null; then
  git add ANATOMY.md
fi

exit 0
