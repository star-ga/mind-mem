#!/usr/bin/env bash
# mind-mem uninstaller — removes the mind-mem MCP server entry from client configs.
# Run with --help for usage. Nothing is changed until the arguments parse.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[mind-mem]${NC} $*"; }
ok()    { echo -e "${GREEN}[mind-mem]${NC} $*"; }
warn()  { echo -e "${YELLOW}[mind-mem]${NC} $*"; }
err()   { echo -e "${RED}[mind-mem]${NC} $*" >&2; }

ALL_TARGETS=(claude-code claude-desktop codex gemini cursor windsurf zed openclaw)

usage() {
  cat <<'EOF'
Usage: ./uninstall.sh [options] [client flags]

Removes the "mind-mem" MCP server entry from AI client configs. Other entries
and settings are kept; a changed file is backed up first
(<file>.bak-mind-mem-uninstall-YYYYmmdd-HHMMSS). A config that is not valid
JSON (for example JSONC with comments) is left untouched with a warning.

Client flags (default: all clients):
  --all             every client below
  --claude-code     ~/.claude/mcp.json (and ~/.claude.json via mm)
  --claude-desktop  Claude Desktop config
  --codex           ~/.codex/config.toml
  --gemini          ~/.gemini/settings.json
  --cursor          ~/.cursor/mcp.json
  --windsurf        ~/.codeium/windsurf/mcp_config.json
  --zed             ~/.config/zed/settings.json
  --openclaw        ~/.openclaw hooks and openclaw.json entry
  --client NAME     any other client in mind-mem's registry (repeatable);
                    runs `mm uninstall-all --agent NAME`. With --all (or no
                    client flags) every detected registry client is also
                    cleaned this way when `mm` is available.

Options:
  --dry-run         show what would be changed, change nothing
  --purge           also delete workspace data in ~/.mind-mem
  -h, --help        show this help and exit
EOF
}

purge=false
dry_run=false
selected=()
registry=()
all_registry=false
while [ $# -gt 0 ]; do
  arg="$1"
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --purge) purge=true ;;
    --dry-run) dry_run=true ;;
    --all) selected=("${ALL_TARGETS[@]}"); all_registry=true ;;
    --claude-code|--claude-desktop|--codex|--gemini|--cursor|--windsurf|--zed|--openclaw)
      selected+=("${arg#--}") ;;
    --client)
      if [ $# -lt 2 ]; then err "--client needs a NAME"; exit 2; fi
      registry+=("$2"); shift ;;
    --client=*) registry+=("${arg#--client=}") ;;
    *)
      err "unknown argument: $arg"
      usage >&2
      exit 2 ;;
  esac
  shift
done
if [ "${#selected[@]}" -eq 0 ] && [ "${#registry[@]}" -eq 0 ]; then
  selected=("${ALL_TARGETS[@]}")
  all_registry=true
fi

wants() {
  local t
  for t in "${selected[@]}"; do [ "$t" = "$1" ] && return 0; done
  return 1
}

# remove_entry <file> <format: json|toml> <key path...>
# Exit 0 = removed, 3 = nothing to remove, 4 = refused (unparseable). Never
# rewrites a file it cannot parse; backs up and replaces atomically otherwise.
remove_entry() {
  local config="$1"
  shift
  [ -f "$config" ] || return 3
  MM_DRY_RUN="$dry_run" python3 - "$config" "$@" <<'PYEOF'
import json, os, re, shutil, sys, tempfile, time

path, fmt, *keys = sys.argv[1:]
dry = os.environ.get("MM_DRY_RUN") == "true"
try:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
except (OSError, UnicodeDecodeError) as exc:
    print(f"cannot read {path}: {exc}; left untouched", file=sys.stderr)
    sys.exit(4)

if fmt == "toml":
    pattern = r"\n?\[mcp_servers\.mind-mem\]\n(?:(?!\[)[^\n]*\n)*(?:\[mcp_servers\.mind-mem\.env\]\n(?:(?!\[)[^\n]*\n)*)?"
    new = re.sub(pattern, "\n", text)
    if new == text:
        sys.exit(3)
else:
    try:
        data = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        print(f"{path} is not strict JSON (comments?); left untouched — remove the mind-mem entry by hand", file=sys.stderr)
        sys.exit(4)
    node = data
    for key in keys[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
    if not isinstance(node, dict) or keys[-1] not in node:
        sys.exit(3)
    del node[keys[-1]]
    new = json.dumps(data, indent=2) + "\n"

if dry:
    print(f"would remove the mind-mem entry from {path}")
    sys.exit(0)
target = os.path.realpath(path)
shutil.copy2(target, f"{target}.bak-mind-mem-uninstall-{time.strftime('%Y%m%d-%H%M%S')}")
fd, tmp = tempfile.mkstemp(prefix=".mind-mem-", dir=os.path.dirname(target) or ".")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(new)
shutil.copymode(target, tmp)
os.replace(tmp, target)
PYEOF
}

# report <label> — turns remove_entry's status into one line.
report() {
  local label="$1" rc="$2"
  case "$rc" in
    0) if $dry_run; then info "$label: would remove"; else ok "$label: removed"; fi ;;
    3) info "$label: no mind-mem entry" ;;
    4) warn "$label: left untouched (see message above)" ;;
    *) warn "$label: failed (exit $rc)" ;;
  esac
}

run() {
  local label="$1"
  shift
  local rc=0
  remove_entry "$@" || rc=$?
  report "$label" "$rc"
}

echo -e "\n${BLUE}  mind-mem${NC} uninstaller$($dry_run && echo ' (dry run)')\n"

wants claude-code && run "Claude Code" "$HOME/.claude/mcp.json" json mcpServers mind-mem
wants claude-code && run "Claude Code (user scope)" "$HOME/.claude.json" json mcpServers mind-mem
if wants claude-desktop; then
  if [ "$(uname)" = "Darwin" ]; then
    run "Claude Desktop" "$HOME/Library/Application Support/Claude/claude_desktop_config.json" json mcpServers mind-mem
  else
    run "Claude Desktop" "$HOME/.config/Claude/claude_desktop_config.json" json mcpServers mind-mem
  fi
fi
wants codex && run "Codex CLI" "$HOME/.codex/config.toml" toml
wants gemini && run "Gemini CLI" "$HOME/.gemini/settings.json" json mcpServers mind-mem
wants cursor && run "Cursor" "$HOME/.cursor/mcp.json" json mcpServers mind-mem
wants windsurf && run "Windsurf" "$HOME/.codeium/windsurf/mcp_config.json" json mcpServers mind-mem
wants zed && run "Zed" "$HOME/.config/zed/settings.json" json context_servers mind-mem
if wants openclaw; then
  if [ -d "$HOME/.openclaw/hooks/mind-mem" ]; then
    if $dry_run; then info "OpenClaw hooks: would remove"; else rm -rf "$HOME/.openclaw/hooks/mind-mem" && ok "OpenClaw hooks: removed"; fi
  fi
  run "OpenClaw config" "$HOME/.openclaw/openclaw.json" json hooks internal entries mind-mem
fi

# Registry clients: `mm uninstall-all` removes only the mind-mem entry from
# each client's MCP config (backing a changed file up first).
mm_cli() {
  if command -v mm &>/dev/null; then
    mm "$@"
  elif python3 -c 'import mind_mem.mm_cli' 2>/dev/null; then
    python3 -m mind_mem.mm_cli "$@"
  else
    return 127
  fi
}
if $all_registry || [ "${#registry[@]}" -gt 0 ]; then
  mm_args=(uninstall-all)
  $dry_run && mm_args+=(--dry-run)
  for name in ${registry[@]+"${registry[@]}"}; do mm_args+=(--agent "$name"); done
  rc=0
  out=$(MIND_MEM_WORKSPACE="${MIND_MEM_WORKSPACE:-$HOME}" mm_cli "${mm_args[@]}" 2>&1) || rc=$?
  if [ "$rc" -eq 127 ]; then
    warn "mm not found; registry clients beyond the ones above were not checked"
  else
    MM_OUT="$out" python3 - <<'PYEOF' || warn "mm uninstall-all: $out"
import json, os
for r in json.loads(os.environ["MM_OUT"]).get("results", []):
    for x in [r, *r.get("also", [])]:
        if x.get("removed"):
            verb = "removed" if x.get("written") else "would remove"
            print(f"[mind-mem] {x['agent']}: {verb} the mind-mem entry from {x.get('path', '')}")
        elif x.get("skipped"):
            print(f"[mind-mem] {x['agent']}: left untouched ({x.get('reason', '')})")
PYEOF
  fi
fi

if $purge; then
  if $dry_run; then
    info "would delete workspace data in $HOME/.mind-mem"
  else
    warn "Purging workspace data..."
    rm -rf "$HOME/.mind-mem"
    ok "Workspace data removed"
  fi
fi

echo ""
if $dry_run; then
  info "Dry run complete. Nothing was changed."
else
  ok "Uninstall complete. Restart your AI coding clients to apply changes."
fi
echo ""
