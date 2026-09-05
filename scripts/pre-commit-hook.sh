#!/usr/bin/env bash

# STARGA author guard (chained first: a wrong-identity commit must never be created).
bash "/home/n/mind-sdlc/bin/sdlc" precommit "$(git rev-parse --show-toplevel)" || exit 1

# pre-commit-hook.sh — the repository's composite pre-commit hook.
#
# Install:
#   ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit
#
# This exists because there can only be ONE .git/hooks/pre-commit. Previously
# that slot was a symlink straight to scripts/anatomy-hook.sh; adding a second
# check by pointing the slot elsewhere would have silently disabled the first.
# New pre-commit checks belong here, in order.
#
# Ordering rationale: identity is checked FIRST and is fail-closed. There is no
# point regenerating documentation for a commit that must not be made.

set -euo pipefail

# Resolve to the REAL script directory. This hook is invoked through the
# .git/hooks/pre-commit symlink, so a bare dirname "${BASH_SOURCE[0]}" yields
# .git/hooks/ and every sibling lookup below silently misses — which, because a
# hook that fails to run is a hook that permits the commit, would leave the
# guard installed-but-inert. Prefer the git-tracked scripts/ dir.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# 1. Identity guard — refuses a commit that would enter history under a
#    non-STARGA identity. Public history cannot be un-leaked, so this is a hard
#    stop, not a warning.
"$SCRIPT_DIR/check_author_identity.sh" staged

# 2. Anatomy refresh — regenerates and stages ANATOMY.md if stale.
[ -x "$SCRIPT_DIR/anatomy-hook.sh" ] && "$SCRIPT_DIR/anatomy-hook.sh"

# 3. Docs-alignment refresh — regenerates and stages the derived claim counts
#    (test functions, MCP tools, CI matrix) if a staged change made them stale.
#    Same shape as step 2 and advisory for the same reason: these numbers are a
#    cache of what the tree already knows, so the commit that invalidates them
#    should be the commit that refreshes them. Measured: the check_docs_alignment
#    CI job went red twice in one hour from three seats, none of whom edited a doc.
[ -x "$SCRIPT_DIR/docs-alignment-hook.sh" ] && "$SCRIPT_DIR/docs-alignment-hook.sh"

exit 0
