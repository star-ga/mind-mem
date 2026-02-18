#!/usr/bin/env bash
# mind-mem uninstaller — removes MCP server entries from all configured clients
# Usage: ./uninstall.sh [--all] [--claude-code] [--codex] [--gemini] ...
#        ./uninstall.sh --purge   # also removes workspace data
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[mind-mem]${NC} $*"; }
ok()    { echo -e "${GREEN}[mind-mem]${NC} $*"; }
warn()  { echo -e "${YELLOW}[mind-mem]${NC} $*"; }

remove_json_mcp() {
  local config="$1" key="$2"
  [ ! -f "$config" ] && return
  python3 - "$config" "$key" <<'PYEOF'
import json, sys, os
config_path, mcp_key = sys.argv[1:3]
if not os.path.isfile(config_path):
    sys.exit(0)
with open(config_path) as f:
    data = json.load(f)
servers = data.get(mcp_key, {})
if "mind-mem" in servers:
    del servers[mcp_key]["mind-mem"] if mcp_key in data else None
    data.get(mcp_key, {}).pop("mind-mem", None)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
PYEOF
}

remove_toml_mcp() {
  local config="$1"
  [ ! -f "$config" ] && return
  python3 - "$config" <<'PYEOF'
import re, sys
config_path = sys.argv[1]
with open(config_path) as f:
    content = f.read()
pattern = r'\n?\[mcp_servers\.mind-mem\]\n(?:(?!\[)[^\n]*\n)*(?:\[mcp_servers\.mind-mem\.env\]\n(?:(?!\[)[^\n]*\n)*)?'
content = re.sub(pattern, '\n', content)
with open(config_path, 'w') as f:
    f.write(content)
PYEOF
}

purge=false
targets=("claude-code" "claude-desktop" "codex" "gemini" "cursor" "windsurf" "zed" "openclaw")

for arg in "$@"; do
  case "$arg" in
    --purge) purge=true ;;
  esac
done

echo -e "\n${BLUE}  mind-mem${NC} uninstaller\n"

# Claude Code
info "Claude Code CLI..."
remove_json_mcp "$HOME/.claude/mcp.json" "mcpServers" && ok "removed" || true

# Claude Desktop
if [ "$(uname)" = "Darwin" ]; then
  info "Claude Desktop..."
  remove_json_mcp "$HOME/Library/Application Support/Claude/claude_desktop_config.json" "mcpServers" && ok "removed" || true
else
  info "Claude Desktop..."
  remove_json_mcp "$HOME/.config/Claude/claude_desktop_config.json" "mcpServers" && ok "removed" || true
fi

# Codex
info "Codex CLI..."
remove_toml_mcp "$HOME/.codex/config.toml" && ok "removed" || true

# Gemini
info "Gemini CLI..."
remove_json_mcp "$HOME/.gemini/settings.json" "mcpServers" && ok "removed" || true

# Cursor
info "Cursor..."
remove_json_mcp "$HOME/.cursor/mcp.json" "mcpServers" && ok "removed" || true

# Windsurf
info "Windsurf..."
remove_json_mcp "$HOME/.codeium/windsurf/mcp_config.json" "mcpServers" && ok "removed" || true

# Zed
info "Zed..."
if [ -f "$HOME/.config/zed/settings.json" ]; then
  python3 -c "
import json
with open('$HOME/.config/zed/settings.json') as f: d = json.load(f)
d.get('context_servers', {}).pop('mind-mem', None)
with open('$HOME/.config/zed/settings.json', 'w') as f: json.dump(d, f, indent=2); f.write('\n')
" && ok "removed" || true
fi

# OpenClaw
info "OpenClaw..."
rm -rf "$HOME/.openclaw/hooks/mind-mem" 2>/dev/null && ok "hooks removed" || true
if [ -f "$HOME/.openclaw/openclaw.json" ]; then
  python3 -c "
import json
with open('$HOME/.openclaw/openclaw.json') as f: d = json.load(f)
d.get('hooks', {}).get('internal', {}).get('entries', {}).pop('mind-mem', None)
with open('$HOME/.openclaw/openclaw.json', 'w') as f: json.dump(d, f, indent=2); f.write('\n')
" && ok "config cleaned" || true
fi

if $purge; then
  warn "Purging workspace data..."
  rm -rf "$HOME/.mind-mem"
  ok "Workspace data removed"
fi

echo ""
ok "Uninstall complete. Restart your AI coding clients to apply changes."
echo ""
