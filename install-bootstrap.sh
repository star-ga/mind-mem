#!/usr/bin/env bash
# mind-mem one-command bootstrap installer
#
# Usage (local):   bash install-bootstrap.sh
# Usage (remote):  curl -sSL https://install.mind-mem.sh | bash
#                  curl -sSL https://install.mind-mem.sh | bash -s -- --venv
#
# Flow:
#   1. Detect Python 3.10+
#   2. Optionally create a venv (--venv / MIND_MEM_VENV=1)
#   3. pip install mind-mem from PyPI (or TestPyPI with --test)
#   4. Bootstrap a workspace at ~/.mind-mem/workspace
#   5. Print next-steps for wiring up MCP clients
#
# The existing in-repo install.sh is for *client-hook* configuration
# after a source checkout. This bootstrap is for users who just want
# `pip install mind-mem` + a working workspace in one command.

set -euo pipefail

# ───────────────────────── colors / logging ─────────────────────────

if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; NC=""
fi

info()  { printf '%s[mind-mem]%s %s\n' "$BLUE" "$NC" "$*"; }
ok()    { printf '%s[mind-mem]%s %s\n' "$GREEN" "$NC" "$*"; }
warn()  { printf '%s[mind-mem]%s %s\n' "$YELLOW" "$NC" "$*"; }
err()   { printf '%s[mind-mem]%s %s\n' "$RED" "$NC" "$*" >&2; }

# ───────────────────────── argument parsing ─────────────────────────

USE_VENV="${MIND_MEM_VENV:-0}"
USE_TESTPYPI=0
WORKSPACE="${MIND_MEM_WORKSPACE:-$HOME/.mind-mem/workspace}"
PYTHON="${MIND_MEM_PYTHON:-python3}"
SKIP_CLIENT_HOOKS=0
PIP_EXTRAS=""

usage() {
    cat <<EOF
mind-mem one-command installer

Options:
  --venv                Create a dedicated venv at ~/.mind-mem/venv
  --test                Install from TestPyPI instead of PyPI
  --workspace PATH      Override workspace path (default: ~/.mind-mem/workspace)
  --python PYEXE        Use a specific Python interpreter
  --extras LIST         pip-install extras (e.g. "mcp,encryption,vector")
  --skip-client-hooks   Skip Claude Code / Desktop / Cursor / etc. hook setup
  -h, --help            Show this help and exit

Env-var equivalents:
  MIND_MEM_VENV=1       Same as --venv
  MIND_MEM_WORKSPACE    Same as --workspace
  MIND_MEM_PYTHON       Same as --python
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --venv)               USE_VENV=1 ;;
        --test|--testpypi)    USE_TESTPYPI=1 ;;
        --workspace)          WORKSPACE="$2"; shift ;;
        --python)             PYTHON="$2"; shift ;;
        --extras)             PIP_EXTRAS="$2"; shift ;;
        --skip-client-hooks)  SKIP_CLIENT_HOOKS=1 ;;
        -h|--help)            usage; exit 0 ;;
        *)                    err "unknown option: $1"; usage >&2; exit 2 ;;
    esac
    shift
done

# ───────────────────────── python detection ─────────────────────────

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    err "Python interpreter not found: $PYTHON"
    err "Install Python 3.10+ from python.org or your package manager, then re-run."
    exit 1
fi

PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYMAJ="${PYVER%.*}"
PYMIN="${PYVER#*.}"
if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    err "Python 3.10+ required (found $PYVER via $PYTHON)"
    exit 1
fi
ok "Python $PYVER via $PYTHON"

# ───────────────────────── venv (optional) ──────────────────────────

PIP_CMD="$PYTHON -m pip"
VENV_DIR=""

if [ "$USE_VENV" = "1" ]; then
    VENV_DIR="${MIND_MEM_VENV_DIR:-$HOME/.mind-mem/venv}"
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating venv at $VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    PIP_CMD="$PYTHON -m pip"
    ok "Activated venv at $VENV_DIR"
fi

# ───────────────────────── pip install ──────────────────────────────

PACKAGE="mind-mem"
if [ -n "$PIP_EXTRAS" ]; then
    PACKAGE="mind-mem[$PIP_EXTRAS]"
fi

info "Upgrading pip (quiet)"
$PIP_CMD install --quiet --upgrade pip setuptools wheel >/dev/null

PIP_INDEX_ARGS=""
if [ "$USE_TESTPYPI" = "1" ]; then
    PIP_INDEX_ARGS="--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/"
    info "Installing $PACKAGE from TestPyPI"
else
    info "Installing $PACKAGE from PyPI"
fi

EXTRA_FLAGS=""
if [ "$USE_VENV" = "0" ] && "$PYTHON" -c 'import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)' 2>/dev/null; then
    EXTRA_FLAGS="--user"
fi

# shellcheck disable=SC2086
$PIP_CMD install $PIP_INDEX_ARGS $EXTRA_FLAGS "$PACKAGE"

INSTALLED_VERSION="$($PYTHON -c 'import mind_mem, sys; print(getattr(mind_mem, "__version__", "unknown"))' 2>/dev/null || echo unknown)"
ok "Installed mind-mem $INSTALLED_VERSION"

# ───────────────────────── workspace bootstrap ──────────────────────

mkdir -p "$WORKSPACE"
if [ ! -f "$WORKSPACE/mind-mem.json" ]; then
    info "Bootstrapping workspace at $WORKSPACE"
    if command -v mind-mem-init >/dev/null 2>&1; then
        mind-mem-init "$WORKSPACE"
    else
        "$PYTHON" -m mind_mem.init_workspace "$WORKSPACE" || warn "Skipped workspace init — module missing or failed."
    fi
    ok "Workspace initialised"
else
    ok "Workspace already present at $WORKSPACE"
fi

# ───────────────────────── client hooks (optional) ──────────────────

if [ "$SKIP_CLIENT_HOOKS" = "0" ] && command -v mm >/dev/null 2>&1; then
    info "Wiring MCP clients (Claude Code, Claude Desktop, Cursor, ...)"
    if mm install-all --force >/dev/null 2>&1; then
        ok "MCP client hooks installed"
    else
        warn "Some client hooks could not be installed. Run 'mm install-all --force' manually."
    fi
fi

# ───────────────────────── final summary ────────────────────────────

cat <<EOF

${BOLD}mind-mem ${INSTALLED_VERSION} installed.${NC}

Workspace:  $WORKSPACE
Python:     $PYTHON ($PYVER)
${VENV_DIR:+Venv:       $VENV_DIR
}
Next steps:
  • Verify:   mm status
  • Search:   mm recall "example query"
  • Docs:     https://mind-mem.readthedocs.io/
  • PyPI:     https://pypi.org/project/mind-mem/

If ${BOLD}mm${NC} is not on your PATH, prepend one of:
  • venv:   ${VENV_DIR:-$HOME/.mind-mem/venv}/bin
  • --user: ~/.local/bin

EOF

ok "Bootstrap complete."
