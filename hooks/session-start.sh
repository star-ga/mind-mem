#!/usr/bin/env bash
# mind-mem SessionStart hook — prints health summary for context injection
set -euo pipefail

WS="${MIND_MEM_WORKSPACE:-.}"
STATE="$WS/memory/intel-state.json"
CONFIG="$WS/mind-mem.json"

# `auto_recall` is the documented switch for this hook: "show recall context
# automatically on session start". It shipped in DEFAULT_CONFIG and in the
# config tables of README.md and docs/configuration.md, and NOTHING read it —
# an operator who set it to false silenced nothing and was told otherwise by
# two documents. Read here rather than deleted: the setting is wanted, it was
# simply never wired, and "no reader" is a statement about wiring, not worth.
# Default true, so a workspace with no config behaves exactly as before.
# Pass paths via env vars to prevent injection from paths with special chars.
if [ -f "$CONFIG" ]; then
  RECALL=$(MIND_MEM_CONFIG="$CONFIG" python3 -c "
import json, os
try:
    d = json.load(open(os.environ['MIND_MEM_CONFIG']))
    print('true' if d.get('auto_recall', True) else 'false')
except Exception:
    print('true')
" 2>/dev/null || echo "true")
  if [ "$RECALL" = "false" ]; then
    exit 0
  fi
fi

if [ ! -f "$STATE" ]; then
  echo "SessionStart:compact mind-mem not initialized. Run: mind-mem-init"
  exit 0
fi

# Parse JSON with python3 (robust, no jq dependency)
read -r MODE LAST CONTRADICTIONS < <(MIND_MEM_STATE="$STATE" python3 -c "
import json, os, sys
try:
    d = json.load(open(os.environ['MIND_MEM_STATE']))
    print(d.get('self_correcting_mode', d.get('governance_mode', 'unknown')),
          d.get('last_scan', 'never'),
          d.get('counters', {}).get('contradictions_open', 0))
except Exception:
    print('unknown never 0')
")

echo "SessionStart:compact mind-mem health: mode=$MODE last_scan=$LAST contradictions=$CONTRADICTIONS"
