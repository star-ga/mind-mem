#!/usr/bin/env bash
# mind-mem installer — installs the package + wires MCP config for AI clients
# Usage: ./install.sh [--all] [--claude-code] [--claude-desktop] [--codex] [--gemini]
#        [--cursor] [--windsurf] [--zed] [--openclaw] [--workspace PATH]
#        [--no-install] [--installer pipx|pip] [--package SPEC]
set -euo pipefail

MIND_MEM_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKSPACE="$HOME/.mind-mem/workspace"
DEFAULT_PACKAGE_SPEC=""   # populated by select_package_spec()

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}[mind-mem]${NC} $*"; }
ok()    { echo -e "${GREEN}[mind-mem]${NC} $*"; }
warn()  { echo -e "${YELLOW}[mind-mem]${NC} $*"; }
err()   { echo -e "${RED}[mind-mem]${NC} $*" >&2; }

# ---------- runtime + package install ----------

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

# Decide which package spec to pass to pipx/pip:
#   * If we are inside a checkout that contains pyproject.toml and a `src/`
#     tree, install the local checkout in editable mode — avoids pulling
#     a stale copy from PyPI when the user is hacking on the repo.
#   * Otherwise install the published wheel from PyPI.
select_package_spec() {
  if [ -n "${MIND_MEM_PACKAGE_SPEC:-}" ]; then
    DEFAULT_PACKAGE_SPEC="$MIND_MEM_PACKAGE_SPEC"
    return
  fi
  if [ -f "$MIND_MEM_DIR/pyproject.toml" ] && [ -d "$MIND_MEM_DIR/src" ]; then
    # Use the relative form so install_package can ``cd`` into
    # MIND_MEM_DIR first. Passing an absolute path with brackets
    # (e.g. ``/d/a/mind-mem/mind-mem[mcp]``) trips pipx's package
    # spec parser on Windows under git-bash, where MSYS path
    # translation produces forward-slash paths that pipx rejects
    # as "Unable to parse package spec".
    DEFAULT_PACKAGE_SPEC=".[mcp]"
  else
    DEFAULT_PACKAGE_SPEC="mind-mem[mcp]"
  fi
}

# Choose the install backend. Default order: pipx (preferred) -> pip --user.
# Caller can force one with --installer pipx|pip.
select_installer() {
  local forced="${1:-auto}"
  case "$forced" in
    pipx)
      if ! command -v pipx &>/dev/null; then
        err "--installer pipx requested but pipx not on PATH. Install pipx first (https://pipx.pypa.io)."
        exit 1
      fi
      echo "pipx"
      return
      ;;
    pip)
      echo "pip"
      return
      ;;
    auto)
      if command -v pipx &>/dev/null; then
        echo "pipx"
      else
        echo "pip"
      fi
      return
      ;;
    *)
      err "Unknown --installer value: $forced (use pipx or pip)"
      exit 1
      ;;
  esac
}

# Install the package + the [mcp] extra so `mind-mem-mcp` is on PATH.
# When ``spec`` starts with ``./`` or equals ``.[...]`` we cd into the
# checkout first so pipx/pip both resolve the local source tree without
# ever seeing an absolute MSYS path (Windows-fragile, see
# select_package_spec).
install_package() {
  local installer="$1" spec="$2"
  local is_local=false
  case "$spec" in
    .[*|.|./*) is_local=true ;;
  esac

  case "$installer" in
    pipx)
      info "Installing $spec via pipx (isolated venv)"
      # `pipx install --force` upgrades or replaces an existing install.
      # mind-mem ships several entry points (mind-mem-init, mind-mem-mcp, ...);
      # `pipx install` registers the lot.
      local pipx_ok=true
      if $is_local; then
        ( cd "$MIND_MEM_DIR" && pipx install --force "$spec" ) || pipx_ok=false
      else
        pipx install --force "$spec" || pipx_ok=false
      fi
      if ! $pipx_ok; then
        # Hosts where pipx refuses to run (e.g. PIPX_HOME contains a
        # space, or pipx and the system python disagree on the active
        # interpreter) shouldn't dead-end the install. Fall back to
        # pip --user with the same package spec.
        warn "pipx install failed; falling back to pip --user"
        install_package pip "$spec"
      fi
      ;;
    pip)
      info "Installing $spec via pip --user"
      # PEP 668 ships an `EXTERNALLY-MANAGED` marker on Debian / Ubuntu /
      # recent Fedora that blocks even `pip install --user`. Retry with
      # `--break-system-packages` once when we detect that. `--user`
      # already isolates the install to ``~/.local`` so the marker's
      # protection is effectively redundant for this path.
      local pip_cwd="$PWD"
      if $is_local; then
        pip_cwd="$MIND_MEM_DIR"
      fi
      # Use mktemp instead of a $PID-suffixed /tmp path — $PID is guessable
      # on a shared host and the predictable filename is a symlink-race
      # surface. Cleanup is explicit at each exit branch (a `RETURN` trap
      # would fire after the local went out of scope under `set -u` and
      # bashisms — Windows-CI burnt that on the previous attempt).
      local pip_err
      pip_err=$(mktemp -t mind-mem-pip-err.XXXXXX) || { echo "mktemp failed" >&2; exit 1; }
      if ! ( cd "$pip_cwd" && python3 -m pip install --user --upgrade "$spec" ) 2>"$pip_err"; then
        if grep -q "externally-managed\|EXTERNALLY-MANAGED\|PEP 668" "$pip_err"; then
          warn "pip --user blocked by PEP 668; retrying with --break-system-packages"
          rm -f "$pip_err"
          ( cd "$pip_cwd" && python3 -m pip install --user --upgrade --break-system-packages "$spec" )
        else
          cat "$pip_err" >&2
          rm -f "$pip_err"
          exit 1
        fi
      else
        rm -f "$pip_err"
      fi
      ;;
  esac
}

# Verify the package and its console script are reachable.
# Records the absolute path to `mind-mem-mcp` on stdout.
resolve_mcp_command() {
  local cmd
  cmd=$(command -v mind-mem-mcp 2>/dev/null || true)
  if [ -z "$cmd" ]; then
    # pipx installs to ~/.local/bin on most platforms, ~/Library/... on macOS.
    for cand in "$HOME/.local/bin/mind-mem-mcp" \
                "$HOME/Library/Python"/*/bin/mind-mem-mcp; do
      if [ -x "$cand" ]; then
        cmd="$cand"
        break
      fi
    done
  fi
  if [ -z "$cmd" ]; then
    err "mind-mem-mcp console script not found on PATH after install."
    err "If you used pipx, run \`pipx ensurepath\` and re-open your shell."
    exit 1
  fi
  echo "$cmd"
}

# Smoke-test: import the package + invoke the console script.
verify_install() {
  local cmd="$1"
  info "Smoke test: $cmd --help"
  if ! "$cmd" --help >/dev/null 2>&1; then
    err "mind-mem-mcp --help failed. The install is broken — aborting."
    exit 1
  fi
  ok "mind-mem-mcp resolves and runs"
}

# ---------- workspace + client config helpers ----------

ensure_workspace() {
  local ws="$1"
  if [ ! -d "$ws" ]; then
    info "Creating workspace at $ws"
    # Prefer the ``mind-mem-init`` console script so this step works
    # whether mind-mem was installed via pipx (isolated venv, not on
    # the system python's import path) or via pip --user. Fall back to
    # ``python3 -m mind_mem.init_workspace`` only if the console
    # script isn't on PATH (e.g. --no-install used with a checkout
    # that hasn't been pip-installed yet — devs running directly).
    if command -v mind-mem-init &>/dev/null; then
      mind-mem-init "$ws"
    else
      python3 -m mind_mem.init_workspace "$ws"
    fi
  fi
  ok "Workspace: $ws"
}

# JSON MCP config helper — merges mind-mem entry into a JSON file with mcpServers key
# Usage: install_json_mcp <config_path> <mcp_key> <workspace> <mcp_command>
install_json_mcp() {
  local config="$1" key="$2" ws="$3" cmd="$4"
  local dir
  dir=$(dirname "$config")
  mkdir -p "$dir"

  python3 - "$config" "$key" "$ws" "$cmd" <<'PYEOF'
import json, sys, os

config_path, mcp_key, workspace, command = sys.argv[1:5]

data = {}
if os.path.isfile(config_path):
    with open(config_path) as f:
        data = json.load(f)

if mcp_key not in data:
    data[mcp_key] = {}

data[mcp_key]["mind-mem"] = {
    "command": command,
    "args": [],
    "env": {
        "MIND_MEM_WORKSPACE": workspace,
    },
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
}

# TOML MCP config helper — appends mind-mem entry to a TOML file (Codex CLI)
# Usage: install_toml_mcp <config_path> <workspace> <mcp_command>
install_toml_mcp() {
  local config="$1" ws="$2" cmd="$3"

  if grep -q '\[mcp_servers\.mind-mem\]' "$config" 2>/dev/null; then
    warn "Codex: mind-mem already configured in $config, updating..."
    python3 - "$config" "$ws" "$cmd" <<'PYEOF'
import re, sys

config_path, workspace, command = sys.argv[1:4]

with open(config_path) as f:
    content = f.read()

pattern = r'\n?\[mcp_servers\.mind-mem\]\n(?:(?!\[)[^\n]*\n)*(?:\[mcp_servers\.mind-mem\.env\]\n(?:(?!\[)[^\n]*\n)*)?'
content = re.sub(pattern, '\n', content)

block = f'''
[mcp_servers.mind-mem]
command = "{command}"
args = []

[mcp_servers.mind-mem.env]
MIND_MEM_WORKSPACE = "{workspace}"
'''
content = content.rstrip() + '\n' + block.lstrip()

with open(config_path, 'w') as f:
    f.write(content)
PYEOF
  else
    cat >> "$config" <<TOML

[mcp_servers.mind-mem]
command = "$cmd"
args = []

[mcp_servers.mind-mem.env]
MIND_MEM_WORKSPACE = "$ws"
TOML
  fi
}

# ---------- client installers ----------

install_claude_code() {
  local ws="$1" cmd="$2"
  local config="$HOME/.claude/mcp.json"
  info "Claude Code CLI: $config"
  install_json_mcp "$config" "mcpServers" "$ws" "$cmd"
  ok "Claude Code CLI configured"
}

install_claude_desktop() {
  local ws="$1" cmd="$2"
  local config
  if [ "$(uname)" = "Darwin" ]; then
    config="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
  else
    config="$HOME/.config/Claude/claude_desktop_config.json"
  fi
  info "Claude Desktop: $config"
  install_json_mcp "$config" "mcpServers" "$ws" "$cmd"
  ok "Claude Desktop configured"
}

install_codex() {
  local ws="$1" cmd="$2"
  local config="$HOME/.codex/config.toml"
  if [ ! -f "$config" ]; then
    warn "Codex CLI config not found at $config, creating..."
    mkdir -p "$HOME/.codex"
    touch "$config"
  fi
  info "Codex CLI: $config"
  install_toml_mcp "$config" "$ws" "$cmd"
  ok "Codex CLI configured"
}

install_gemini() {
  local ws="$1" cmd="$2"
  local config="$HOME/.gemini/settings.json"
  if [ ! -f "$config" ]; then
    warn "Gemini CLI config not found at $config, creating..."
    mkdir -p "$HOME/.gemini"
    echo '{}' > "$config"
  fi
  info "Gemini CLI: $config"
  install_json_mcp "$config" "mcpServers" "$ws" "$cmd"
  ok "Gemini CLI configured"
}

install_cursor() {
  local ws="$1" cmd="$2"
  local config="$HOME/.cursor/mcp.json"
  info "Cursor: $config"
  install_json_mcp "$config" "mcpServers" "$ws" "$cmd"
  ok "Cursor configured"
}

install_windsurf() {
  local ws="$1" cmd="$2"
  local config="$HOME/.codeium/windsurf/mcp_config.json"
  info "Windsurf: $config"
  install_json_mcp "$config" "mcpServers" "$ws" "$cmd"
  ok "Windsurf configured"
}

install_zed() {
  local ws="$1" cmd="$2"
  local config="$HOME/.config/zed/settings.json"
  if [ ! -f "$config" ]; then
    warn "Zed config not found at $config, skipping..."
    return
  fi
  info "Zed: $config"
  python3 - "$config" "$ws" "$cmd" <<'PYEOF'
import json, sys

config_path, workspace, command = sys.argv[1:4]

with open(config_path) as f:
    data = json.load(f)

if "context_servers" not in data:
    data["context_servers"] = {}

data["context_servers"]["mind-mem"] = {
    "command": {
        "path": command,
        "args": [],
        "env": {"MIND_MEM_WORKSPACE": workspace},
    },
}

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
  ok "Zed configured"
}

install_openclaw() {
  local ws="$1" cmd="$2"
  local hooks_dir="$HOME/.openclaw/hooks/mind-mem"

  info "OpenClaw: copying hooks to $hooks_dir"
  mkdir -p "$hooks_dir"
  if [ -f "$MIND_MEM_DIR/hooks/openclaw/mind-mem/handler.js" ]; then
    cp "$MIND_MEM_DIR/hooks/openclaw/mind-mem/handler.js" "$hooks_dir/handler.js"
    cp "$MIND_MEM_DIR/hooks/openclaw/mind-mem/HOOK.md" "$hooks_dir/HOOK.md"
  else
    warn "OpenClaw hook sources not found in $MIND_MEM_DIR/hooks/openclaw/ (skipping handler copy)"
  fi

  local oc_config="$HOME/.openclaw/openclaw.json"
  if [ -f "$oc_config" ]; then
    python3 - "$oc_config" "$ws" "$MIND_MEM_DIR" <<'PYEOF'
import json, sys

config_path, workspace, mind_mem_dir = sys.argv[1:4]

with open(config_path) as f:
    data = json.load(f)

hooks = data.setdefault("hooks", {})
internal = hooks.setdefault("internal", {"enabled": True})
entries = internal.setdefault("entries", {})

entries["mind-mem"] = {
    "enabled": True,
    "env": {
        "MIND_MEM_WORKSPACE": workspace,
        "MIND_MEM_HOME": mind_mem_dir,
    },
}

entries.pop("mem-os", None)

with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PYEOF
    ok "OpenClaw config updated"
  else
    warn "OpenClaw config not found at $oc_config (create it or install OpenClaw first)"
  fi

  info "OpenClaw: configuring MCP server (Claude CLI used by OpenClaw reads ~/.claude/mcp.json)"
  install_claude_code "$ws" "$cmd"

  ok "OpenClaw hooks + MCP configured"
}

# ---------- auto-detect ----------

detect_clients() {
  local found=()
  [ -d "$HOME/.claude" ] && found+=("claude-code")
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
  --installer KIND   Force pipx or pip (default: auto)
  --package SPEC     Override the package spec (default: local checkout if
                     pyproject.toml is present, else "mind-mem[mcp]"). Set
                     MIND_MEM_PACKAGE_SPEC to override globally.
  --no-install       Skip the package install step (assumes mind-mem-mcp
                     is already on PATH).
  --help             Show this help

If no client flags are given, auto-detects installed clients and prompts.

Examples:
  $0 --all                              # Install for all detected clients
  $0 --claude-code --codex              # Install for Claude Code + Codex
  $0 --all --workspace /path/to/ws      # Custom workspace
  $0 --all --installer pip              # Force pip --user instead of pipx
  $0 --all --no-install                 # Wire clients only; skip pip/pipx install
EOF
  exit 0
}

main() {
  local workspace="$DEFAULT_WORKSPACE"
  local clients=()
  local auto=false
  local installer_choice="auto"
  local skip_install=false
  local package_override=""

  while [ $# -gt 0 ]; do
    case "$1" in
      --all)            auto=true; shift ;;
      --claude-code)    clients+=("claude-code"); shift ;;
      --claude-desktop) clients+=("claude-desktop"); shift ;;
      --codex)          clients+=("codex"); shift ;;
      --gemini)         clients+=("gemini"); shift ;;
      --cursor)         clients+=("cursor"); shift ;;
      --windsurf)       clients+=("windsurf"); shift ;;
      --zed)            clients+=("zed"); shift ;;
      --openclaw)       clients+=("openclaw"); shift ;;
      --workspace)      workspace="$2"; shift 2 ;;
      --installer)      installer_choice="$2"; shift 2 ;;
      --package)        package_override="$2"; shift 2 ;;
      --no-install)     skip_install=true; shift ;;
      --help|-h)        usage ;;
      *)                err "Unknown option: $1"; usage ;;
    esac
  done

  echo ""
  echo -e "${BLUE}  mind-mem${NC} — Drop-in memory for AI coding agents"
  echo -e "  ${BLUE}https://github.com/star-ga/mind-mem${NC}"
  echo ""

  ensure_python3

  # Resolve the package spec + the install backend, then install (unless
  # --no-install) and verify the console script is reachable.
  select_package_spec
  if [ -n "$package_override" ]; then
    DEFAULT_PACKAGE_SPEC="$package_override"
  fi

  if ! $skip_install; then
    local installer
    installer=$(select_installer "$installer_choice")
    install_package "$installer" "$DEFAULT_PACKAGE_SPEC"
  else
    info "Skipping package install (--no-install)"
  fi

  local mcp_command
  mcp_command=$(resolve_mcp_command)
  verify_install "$mcp_command"

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

  ensure_workspace "$workspace"
  echo ""

  for client in "${clients[@]}"; do
    case "$client" in
      claude-code)     install_claude_code "$workspace" "$mcp_command" ;;
      claude-desktop)  install_claude_desktop "$workspace" "$mcp_command" ;;
      codex)           install_codex "$workspace" "$mcp_command" ;;
      gemini)          install_gemini "$workspace" "$mcp_command" ;;
      cursor)          install_cursor "$workspace" "$mcp_command" ;;
      windsurf)        install_windsurf "$workspace" "$mcp_command" ;;
      zed)             install_zed "$workspace" "$mcp_command" ;;
      openclaw)        install_openclaw "$workspace" "$mcp_command" ;;
      *)               warn "Unknown client: $client" ;;
    esac
  done

  echo ""
  ok "Installation complete!"
  echo ""
  echo "  Workspace:  $workspace"
  echo "  MCP server: $mcp_command"
  echo ""
  echo "  Next steps:"
  echo "    1. Restart your AI coding client to pick up the new MCP server"
  echo "    2. Try: recall \"what do I know about...\""
  echo "    3. Run 'mind-mem-init --health $workspace' for a health check"
  echo ""
}

main "$@"
