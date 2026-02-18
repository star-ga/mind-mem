#!/usr/bin/env bash
# mind-mem SessionStart hook — prints health summary for context injection
set -euo pipefail

WS="${MIND_MEM_WORKSPACE:-.}"
STATE="$WS/memory/intel-state.json"

if [ ! -f "$STATE" ]; then
  echo "SessionStart:compact mind-mem not initialized. Run: python3 scripts/init_workspace.py"
  exit 0
fi

# Parse JSON with python3 (robust, no jq dependency)
# Pass path via env var to prevent injection from paths with special chars
read -r MODE LAST CONTRADICTIONS < <(MIND_MEM_STATE="$STATE" python3 -c "
import json, os, sys
try:
    d = json.load(open(os.environ['MIND_MEM_STATE']))
    print(d.get('governance_mode', 'unknown'),
          d.get('last_scan', 'never'),
          d.get('counters', {}).get('contradictions_open', 0))
except Exception:
    print('unknown never 0')
")

echo "SessionStart:compact mind-mem health: mode=$MODE last_scan=$LAST contradictions=$CONTRADICTIONS"
