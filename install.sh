#!/usr/bin/env bash
# mind-mem installer — sets up MCP server + hooks for all supported clients
# Usage: ./install.sh [--all] [--claude-code] [--claude-desktop] [--codex] [--gemini]
#        [--cursor] [--windsurf] [--openclaw] [--workspace PATH]
set -euo pipefail

MIND_MEM_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKSPACE="$HOME/.mind-mem/workspace"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}[mind-mem]${NC} $*"; }
ok()    { echo -e "${GREEN}[mind-mem]${NC} $*"; }
warn()  { echo -e "${YELLOW}[mind-mem]${NC} $*"; }
err()   { echo -e "${RED}[mind-mem]${NC} $*" >&2; }

# ---------- helpers ----------

ensure_python3() {
  if ! command -v python3 &>/dev/null; then
    err "python3 is required but not found. Install Python 3.10+ first."
    exit 1
  fi
  local ver
  ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  local major minor
  major=$(echo "$ver" | cut -d. -f1)
  minor=$(echo "$ver" | cut -d. -f2)
  if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
    err "Python 3.10+ required (found $ver)"
    exit 1
  fi
  ok "Python $ver"
}

ensure_workspace() {
  local ws="$1"
  if [ ! -d "$ws" ]; then
    info "Creating workspace at $ws"
    python3 "$MIND_MEM_DIR/scripts/init_workspace.py" "$ws"
  fi
  ok "Workspace: $ws"
}

# JSON MCP config helper — merges mind-mem entry into a JSON file with mcpServers key
# Usage: install_json_mcp <config_path> <mcp_key> <workspace>
install_json_mcp() {
  local config="$1" key="$2" ws="$3"
  local dir
  dir=$(dirname "$config")
  mkdir -p "$dir"

  python3 - "$config" "$key" "$ws" "$MIND_MEM_DIR" <<'PYEOF'
import json, sys, os

config_path, mcp_key, workspace, mind_mem_dir = sys.argv[1:5]

# Read existing or create new
data = {}
if os.path.isfile(config_path):
    with open(config_path) as f:
        data = json.load(f)

# Ensure mcpServers key exists
if mcp_key not in data:
    data[mcp_key] = {}

# Add mind-mem entry
data[mcp_key]["mind-mem"] = {
    "command": "python3",
    "args": [os.path.join(mind_mem_dir, "mcp_server.py")],
    "env": {
        "MIND_MEM_WORKSPACE": workspace
    }
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
}

# TOML MCP config helper — appends mind-mem entry to a TOML file (Codex CLI)
# Usage: install_toml_mcp <config_path> <workspace>
install_toml_mcp() {
  local config="$1" ws="$2"

  # Check if already configured
  if grep -q '\[mcp_servers\.mind-mem\]' "$config" 2>/dev/null; then
    warn "Codex: mind-mem already configured in $config, updating..."
    # Remove existing mind-mem block and re-add
    python3 - "$config" "$ws" "$MIND_MEM_DIR" <<'PYEOF'
import re, sys, os

config_path, workspace, mind_mem_dir = sys.argv[1:4]

with open(config_path) as f:
    content = f.read()

# Remove existing mind-mem MCP block (section + its keys until next section or EOF)
pattern = r'\n?\[mcp_servers\.mind-mem\]\n(?:(?!\[)[^\n]*\n)*(?:\[mcp_servers\.mind-mem\.env\]\n(?:(?!\[)[^\n]*\n)*)?'
content = re.sub(pattern, '\n', content)

# Append new block
mcp_py = os.path.join(mind_mem_dir, "mcp_server.py")
block = f'''
[mcp_servers.mind-mem]
command = "python3"
args = ["{mcp_py}"]

[mcp_servers.mind-mem.env]
MIND_MEM_WORKSPACE = "{workspace}"
'''
content = content.rstrip() + '\n' + block.lstrip()

with open(config_path, 'w') as f:
    f.write(content)
PYEOF
  else
    # Append new block
    local mcp_py="$MIND_MEM_DIR/mcp_server.py"
    cat >> "$config" <<TOML

[mcp_servers.mind-mem]
command = "python3"
args = ["$mcp_py"]

[mcp_servers.mind-mem.env]
MIND_MEM_WORKSPACE = "$ws"
TOML
  fi
}

# ---------- client installers ----------

install_claude_code() {
  local ws="$1"
  local config="$HOME/.claude/mcp.json"
  info "Claude Code CLI: $config"
  install_json_mcp "$config" "mcpServers" "$ws"
  ok "Claude Code CLI configured"
}

install_claude_desktop() {
  local ws="$1"
  # Linux: ~/.config/Claude/claude_desktop_config.json
  # macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
  local config
  if [ "$(uname)" = "Darwin" ]; then
    config="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
  else
    config="$HOME/.config/Claude/claude_desktop_config.json"
  fi
  info "Claude Desktop: $config"
  install_json_mcp "$config" "mcpServers" "$ws"
  ok "Claude Desktop configured"
}

install_codex() {
  local ws="$1"
  local config="$HOME/.codex/config.toml"
  if [ ! -f "$config" ]; then
    warn "Codex CLI config not found at $config, creating..."
    mkdir -p "$HOME/.codex"
    touch "$config"
  fi
  info "Codex CLI: $config"
  install_toml_mcp "$config" "$ws"
  ok "Codex CLI configured"
}

install_gemini() {
  local ws="$1"
  local config="$HOME/.gemini/settings.json"
  if [ ! -f "$config" ]; then
    warn "Gemini CLI config not found at $config, creating..."
    mkdir -p "$HOME/.gemini"
    echo '{}' > "$config"
  fi
  info "Gemini CLI: $config"
  install_json_mcp "$config" "mcpServers" "$ws"
  ok "Gemini CLI configured"
}

install_cursor() {
  local ws="$1"
  local config="$HOME/.cursor/mcp.json"
  info "Cursor: $config"
  install_json_mcp "$config" "mcpServers" "$ws"
  ok "Cursor configured"
}

install_windsurf() {
  local ws="$1"
  local config="$HOME/.codeium/windsurf/mcp_config.json"
  info "Windsurf: $config"
  install_json_mcp "$config" "mcpServers" "$ws"
  ok "Windsurf configured"
}

install_zed() {
  local ws="$1"
  # Zed uses ~/.config/zed/settings.json with context_servers key
  local config="$HOME/.config/zed/settings.json"
  if [ ! -f "$config" ]; then
    warn "Zed config not found at $config, skipping..."
    return
  fi
  info "Zed: $config"
  python3 - "$config" "$ws" "$MIND_MEM_DIR" <<'PYEOF'
import json, sys, os

config_path, workspace, mind_mem_dir = sys.argv[1:4]

with open(config_path) as f:
    data = json.load(f)

if "context_servers" not in data:
    data["context_servers"] = {}

data["context_servers"]["mind-mem"] = {
    "command": {
        "path": "python3",
        "args": [os.path.join(mind_mem_dir, "mcp_server.py")],
        "env": {"MIND_MEM_WORKSPACE": workspace}
    }
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
  ok "Zed configured"
}

install_openclaw() {
  local ws="$1"
  local hooks_dir="$HOME/.openclaw/hooks/mind-mem"

  info "OpenClaw: copying hooks to $hooks_dir"
  mkdir -p "$hooks_dir"
  cp "$MIND_MEM_DIR/hooks/openclaw/mind-mem/handler.js" "$hooks_dir/handler.js"
  cp "$MIND_MEM_DIR/hooks/openclaw/mind-mem/HOOK.md" "$hooks_dir/HOOK.md"

  # Update openclaw.json if it exists
  local oc_config="$HOME/.openclaw/openclaw.json"
  if [ -f "$oc_config" ]; then
    python3 - "$oc_config" "$ws" "$MIND_MEM_DIR" <<'PYEOF'
import json, sys, os

config_path, workspace, mind_mem_dir = sys.argv[1:4]

with open(config_path) as f:
    data = json.load(f)

# Ensure hooks.internal.entries exists
hooks = data.setdefault("hooks", {})
internal = hooks.setdefault("internal", {"enabled": True})
entries = internal.setdefault("entries", {})

entries["mind-mem"] = {
    "enabled": True,
    "env": {
        "MIND_MEM_WORKSPACE": workspace,
        "MIND_MEM_HOME": mind_mem_dir
    }
}

# Remove old mem-os entry if present
entries.pop("mem-os", None)

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
    ok "OpenClaw config updated"
  else
    warn "OpenClaw config not found at $oc_config (create it or install OpenClaw first)"
  fi

  # Also configure ~/.claude/mcp.json — OpenClaw spawns Claude CLI which reads it
  info "OpenClaw: configuring MCP server (Claude CLI used by OpenClaw reads ~/.claude/mcp.json)"
  install_claude_code "$ws"

  ok "OpenClaw hooks + MCP configured"
}

install_claude_code_hooks() {
  local ws="$1"
  # Claude Code hooks.json for SessionStart/Stop
  local hooks_dir="$HOME/.claude/hooks"
  local hooks_file="$hooks_dir/hooks.json"

  if [ ! -d "$hooks_dir" ]; then
    mkdir -p "$hooks_dir"
  fi

  # Claude Code supports hooks via settings — just install the MCP server
  # The hooks.json in the repo is for reference; Claude Code hooks go in settings
  info "Claude Code hooks: MCP server handles all functionality via tools"
  ok "No separate hooks needed for Claude Code (MCP server provides all tools)"
}

# ---------- auto-detect ----------

detect_clients() {
  local found=()
  [ -d "$HOME/.claude" ] && found+=("claude-code")
  # Check Claude Desktop
  if [ "$(uname)" = "Darwin" ] && [ -d "$HOME/Library/Application Support/Claude" ]; then
    found+=("claude-desktop")
  elif [ -d "$HOME/.config/Claude" ]; then
    found+=("claude-desktop")
  fi
  [ -d "$HOME/.codex" ] && found+=("codex")
  [ -d "$HOME/.gemini" ] && found+=("gemini")
  [ -d "$HOME/.cursor" ] && found+=("cursor")
  [ -d "$HOME/.codeium/windsurf" ] && found+=("windsurf")
  [ -d "$HOME/.config/zed" ] && found+=("zed")
  [ -d "$HOME/.openclaw" ] && found+=("openclaw")
  echo "${found[@]}"
}

# ---------- main ----------

usage() {
  cat <<EOF
mind-mem installer

Usage: $0 [OPTIONS]

Options:
  --all              Install for all detected clients
  --claude-code      Install for Claude Code CLI
  --claude-desktop   Install for Claude Desktop
  --codex            Install for Codex CLI (OpenAI)
  --gemini           Install for Gemini CLI (Google)
  --cursor           Install for Cursor
  --windsurf         Install for Windsurf
  --zed              Install for Zed
  --openclaw         Install for OpenClaw
  --workspace PATH   Set workspace path (default: ~/.mind-mem/workspace)
  --help             Show this help

If no client flags are given, auto-detects installed clients and prompts.

Examples:
  $0 --all                              # Install for all detected clients
  $0 --claude-code --codex              # Install for Claude Code + Codex
  $0 --all --workspace /path/to/ws      # Custom workspace
EOF
  exit 0
}

main() {
  local workspace="$DEFAULT_WORKSPACE"
  local clients=()
  local auto=false

  while [ $# -gt 0 ]; do
    case "$1" in
      --all)           auto=true; shift ;;
      --claude-code)   clients+=("claude-code"); shift ;;
      --claude-desktop) clients+=("claude-desktop"); shift ;;
      --codex)         clients+=("codex"); shift ;;
      --gemini)        clients+=("gemini"); shift ;;
      --cursor)        clients+=("cursor"); shift ;;
      --windsurf)      clients+=("windsurf"); shift ;;
      --zed)           clients+=("zed"); shift ;;
      --openclaw)      clients+=("openclaw"); shift ;;
      --workspace)     workspace="$2"; shift 2 ;;
      --help|-h)       usage ;;
      *)               err "Unknown option: $1"; usage ;;
    esac
  done

  echo ""
  echo -e "${BLUE}  mind-mem${NC} — Drop-in memory for AI coding agents"
  echo -e "  ${BLUE}https://github.com/star-ga/mind-mem${NC}"
  echo ""

  # Check python3
  ensure_python3

  # Auto-detect if --all or no clients specified
  if $auto || [ ${#clients[@]} -eq 0 ]; then
    local detected
    detected=$(detect_clients)
    if [ -z "$detected" ]; then
      err "No supported AI coding clients detected."
      err "Install Claude Code, Codex, Gemini CLI, Cursor, Windsurf, Zed, or OpenClaw first."
      exit 1
    fi

    if $auto; then
      read -ra clients <<< "$detected"
    else
      echo "Detected clients:"
      local i=1
      local detected_arr
      read -ra detected_arr <<< "$detected"
      for c in "${detected_arr[@]}"; do
        echo "  $i) $c"
        ((i++))
      done
      echo ""
      read -rp "Install for all? [Y/n] " ans
      if [ "${ans,,}" != "n" ]; then
        clients=("${detected_arr[@]}")
      else
        read -rp "Enter numbers (comma-separated, e.g. 1,3): " nums
        IFS=',' read -ra selections <<< "$nums"
        for sel in "${selections[@]}"; do
          sel=$(echo "$sel" | tr -d ' ')
          if [ "$sel" -ge 1 ] && [ "$sel" -le "${#detected_arr[@]}" ] 2>/dev/null; then
            clients+=("${detected_arr[$((sel-1))]}")
          fi
        done
      fi
    fi
  fi

  if [ ${#clients[@]} -eq 0 ]; then
    err "No clients selected."
    exit 1
  fi

  info "Installing for: ${clients[*]}"
  echo ""

  # Initialize workspace
  ensure_workspace "$workspace"
  echo ""

  # Install for each client
  for client in "${clients[@]}"; do
    case "$client" in
      claude-code)     install_claude_code "$workspace" ;;
      claude-desktop)  install_claude_desktop "$workspace" ;;
      codex)           install_codex "$workspace" ;;
      gemini)          install_gemini "$workspace" ;;
      cursor)          install_cursor "$workspace" ;;
      windsurf)        install_windsurf "$workspace" ;;
      zed)             install_zed "$workspace" ;;
      openclaw)        install_openclaw "$workspace" ;;
      *)               warn "Unknown client: $client" ;;
    esac
  done

  echo ""
  ok "Installation complete!"
  echo ""
  echo "  Workspace: $workspace"
  echo "  MCP server: $MIND_MEM_DIR/mcp_server.py"
  echo ""
  echo "  Next steps:"
  echo "    1. Restart your AI coding client to pick up the new MCP server"
  echo "    2. Try: recall \"what do I know about...\""
  echo "    3. Run 'python3 $MIND_MEM_DIR/scripts/intel_scan.py $workspace' for a health check"
  echo ""
}

main "$@"
