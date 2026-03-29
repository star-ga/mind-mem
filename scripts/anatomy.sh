#!/usr/bin/env bash
# anatomy — Generate ANATOMY.md for any repo
# Universal file index for coding agents (Claude Code, Codex, Gemini CLI, etc.)
# Usage: anatomy [dir] [--update] [--depth N] [--exclude pattern]
#
# Author: STARGA Inc <noreply@star.ga>
# License: MIT

set -euo pipefail

VERSION="1.0.0"
MAX_DEPTH=4
EXCLUDE_PATTERNS=()
TARGET_DIR="."
UPDATE_MODE=false
OUTPUT_FILE="ANATOMY.md"

# Token estimate: ~4 chars per token (GPT/Claude average)
estimate_tokens() {
  local file="$1"
  local chars
  chars=$(wc -c < "$file" 2>/dev/null || echo 0)
  echo $(( (chars + 3) / 4 ))
}

# Extract first meaningful line (skip shebangs, empty lines, imports)
extract_summary() {
  local file="$1"
  local ext="${file##*.}"
  
  case "$ext" in
    py)
      # Look for docstring or first comment
      grep -m1 -E '^\s*("""|#\s+\w)' "$file" 2>/dev/null | sed 's/^[[:space:]]*//' | sed 's/"""//' | head -c 120 || true
      ;;
    rs|ts|js|tsx|jsx|go|c|cpp|h|java|swift|kt)
      # First // or /// comment
      grep -m1 -E '^\s*//[/!]?\s+\w' "$file" 2>/dev/null | sed 's|^[[:space:]]*//[/!]*[[:space:]]*||' | head -c 120 || true
      ;;
    mind)
      # First // comment or fn declaration
      grep -m1 -E '^\s*(//|fn |module )' "$file" 2>/dev/null | sed 's|^[[:space:]]*//[[:space:]]*||' | head -c 120 || true
      ;;
    md)
      # First heading
      grep -m1 -E '^#' "$file" 2>/dev/null | sed 's/^#\+[[:space:]]*//' | head -c 120 || true
      ;;
    json)
      # Just say what it is
      local keys
      keys=$(python3 -c "import json,sys; d=json.load(open('$file')); print(', '.join(list(d.keys())[:5]))" 2>/dev/null || echo "")
      [ -n "$keys" ] && echo "Keys: $keys" || true
      ;;
    toml|yaml|yml)
      grep -m1 -E '^\[|^[a-z]' "$file" 2>/dev/null | head -c 120 || true
      ;;
    sh|bash|zsh)
      grep -m1 -E '^#\s+\w' "$file" 2>/dev/null | sed 's/^#[[:space:]]*//' | head -c 120 || true
      ;;
    *)
      head -1 "$file" 2>/dev/null | head -c 80 || true
      ;;
  esac
}

# Get file category for grouping
file_category() {
  local file="$1"
  local ext="${file##*.}"
  case "$ext" in
    py|rs|ts|js|tsx|jsx|go|c|cpp|h|java|swift|kt|mind) echo "source" ;;
    md|txt|rst) echo "docs" ;;
    json|toml|yaml|yml|ini|cfg|env) echo "config" ;;
    sh|bash|zsh) echo "scripts" ;;
    test.*|spec.*|*_test.*) echo "tests" ;;
    *) echo "other" ;;
  esac
}

# Size bucket for quick scanning
size_bucket() {
  local tokens="$1"
  if [ "$tokens" -lt 50 ]; then echo "tiny"
  elif [ "$tokens" -lt 200 ]; then echo "small"
  elif [ "$tokens" -lt 500 ]; then echo "medium"
  elif [ "$tokens" -lt 1500 ]; then echo "large"
  else echo "huge"
  fi
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --update) UPDATE_MODE=true; shift ;;
    --depth) MAX_DEPTH="$2"; shift 2 ;;
    --exclude) EXCLUDE_PATTERNS+=("$2"); shift 2 ;;
    --output|-o) OUTPUT_FILE="$2"; shift 2 ;;
    --version) echo "anatomy $VERSION"; exit 0 ;;
    --help|-h)
      echo "anatomy $VERSION — Generate ANATOMY.md for coding agents"
      echo ""
      echo "Usage: anatomy [dir] [options]"
      echo ""
      echo "Options:"
      echo "  --update        Only update changed files (preserve manual edits)"
      echo "  --depth N       Max directory depth (default: 4)"
      echo "  --exclude PAT   Exclude glob pattern (repeatable)"
      echo "  --output FILE   Output file (default: ANATOMY.md)"
      echo "  --version       Show version"
      echo "  --help          Show this help"
      exit 0
      ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) TARGET_DIR="$1"; shift ;;
  esac
done

cd "$TARGET_DIR"

# Collect files (respect .gitignore if in a git repo)
collect_files() {
  if git rev-parse --is-inside-work-tree &>/dev/null; then
    git ls-files --cached --others --exclude-standard 2>/dev/null
  else
    find . -maxdepth "$MAX_DEPTH" -type f \
      -not -path '*/.git/*' \
      -not -path '*/node_modules/*' \
      -not -path '*/.venv/*' \
      -not -path '*/__pycache__/*' \
      -not -path '*/target/*' \
      -not -path '*/.wolf/*' \
      -not -path '*/dist/*' \
      | sed 's|^\./||' | sort
  fi
}

# Filter out excluded patterns and binary files
filter_files() {
  while IFS= read -r file; do
    # Skip the output file itself
    [ "$file" = "$OUTPUT_FILE" ] && continue
    
    # Skip binary files
    case "$file" in
      *.png|*.jpg|*.jpeg|*.gif|*.ico|*.woff|*.woff2|*.ttf|*.eot|*.svg|*.mp3|*.mp4|*.zip|*.tar|*.gz|*.bz2|*.xz|*.pyc|*.pyo|*.so|*.dylib|*.dll|*.exe|*.o|*.a|*.wasm|*.db|*.sqlite|*.lock) continue ;;
    esac
    
    # Skip excluded patterns
    local skip=false
    for pat in "${EXCLUDE_PATTERNS[@]+"${EXCLUDE_PATTERNS[@]}"}"; do
      if [[ "$file" == $pat ]]; then
        skip=true
        break
      fi
    done
    $skip && continue
    
    # Skip files larger than 100KB (likely generated)
    local size
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    [ "$size" -gt 102400 ] && continue
    
    echo "$file"
  done
}

# Build the anatomy
# Sort: root files first (no /), then by directory path
files=$(collect_files | filter_files | awk '{
  n = split($0, parts, "/")
  if (n == 1) printf "0\t%s\n", $0
  else printf "1\t%s\n", $0
}' | sort -t$'\t' -k1,1 -k2,2 | cut -f2)
total_files=0
total_tokens=0

# Get project name
project_name=$(basename "$(pwd)")

# Count stats first
declare -A dir_tokens
declare -A dir_files

while IFS= read -r file; do
  [ -z "$file" ] && continue
  [ ! -f "$file" ] && continue
  
  tokens=$(estimate_tokens "$file")
  dir=$(dirname "$file")
  
  dir_tokens["$dir"]=$(( ${dir_tokens["$dir"]:-0} + tokens ))
  dir_files["$dir"]=$(( ${dir_files["$dir"]:-0} + 1 ))
  
  total_files=$((total_files + 1))
  total_tokens=$((total_tokens + tokens))
done <<< "$files"

# Generate output
{
  cat << 'EOF'
# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

EOF

  echo "**Project:** \`$project_name\`"
  echo "**Files:** $total_files | **Est. tokens:** ~$(printf "%'d" $total_tokens)"
  echo "**Generated:** $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo ""
  
  # Token budget guidance
  echo "## Token Budget Guide"
  echo ""
  echo "| Size | Tokens | Read strategy |"
  echo "|------|--------|---------------|"
  echo "| tiny | <50 | Always safe to read |"
  echo "| small | 50-200 | Read freely |"
  echo "| medium | 200-500 | Read if relevant |"
  echo "| large | 500-1500 | Use summary first, read specific sections |"
  echo "| huge | >1500 | Avoid full read — use grep or read specific lines |"
  echo ""
  
  # Directory overview
  echo "## Directory Overview"
  echo ""
  echo "| Directory | Files | Est. tokens |"
  echo "|-----------|-------|-------------|"
  for dir in $(echo "${!dir_tokens[@]}" | tr ' ' '\n' | sort); do
    printf "| \`%s/\` | %d | ~%s |\n" "$dir" "${dir_files[$dir]}" "$(printf "%'d" ${dir_tokens[$dir]})"
  done
  echo ""
  
  # File listing grouped by directory
  echo "## Files"
  echo ""
  
  current_dir=""
  while IFS= read -r file; do
    [ -z "$file" ] && continue
    [ ! -f "$file" ] && continue
    
    dir=$(dirname "$file")
    base=$(basename "$file")
    tokens=$(estimate_tokens "$file")
    bucket=$(size_bucket "$tokens")
    summary=$(extract_summary "$file")
    
    # New directory header
    if [ "$dir" != "$current_dir" ]; then
      current_dir="$dir"
      echo "### \`$dir/\`"
      echo ""
    fi
    
    # File entry
    if [ -n "$summary" ]; then
      echo "- \`$base\` (~${tokens} tok, $bucket) — $summary"
    else
      echo "- \`$base\` (~${tokens} tok, $bucket)"
    fi
    
  done <<< "$files"
  
  echo ""
  echo "---"
  echo "*Generated by \`anatomy $VERSION\`. Edit descriptions manually — re-run preserves structure.*"
  
} > "$OUTPUT_FILE"

echo "✓ $OUTPUT_FILE: $total_files files, ~$(printf "%'d" $total_tokens) estimated tokens"
