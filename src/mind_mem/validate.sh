#!/usr/bin/env bash
# src/mind_mem/validate.sh — thin forwarder to the Python validator.
#
# As of v3.2.0 this file is a one-line shim that execs
# `python3 -m mind_mem.validate_py`. The canonical implementation
# lives in `src/mind_mem/validate_py.py` which ships the same
# invariant surface (V2.1 ConstraintSignatures, V2.2 required fields,
# V2.6 axis.key/relation/enforcement, V2.7 lifecycle.created_by,
# V2.9 staged-proposal fingerprints) plus the pre-existing file-
# structure / decisions / tasks / entities / provenance / cross-refs
# / intelligence sections.
#
# This forwarder is kept for backward compat with anyone who had
# `bash validate.sh` wired into automation. New scripts should call
# the Python module directly.
#
# Set MIND_MEM_VALIDATE_BASH=1 to bypass the forwarder and run the
# legacy bash engine — the pre-forwarder copy lives alongside as
# `validate.sh.pre-forwarder` until v4.0 for emergency parity audits.
#
# STARGA, Inc. — Apache-2.0.

set -euo pipefail

if [[ "${MIND_MEM_VALIDATE_BASH:-0}" == "1" ]]; then
    exec bash "$(dirname "${BASH_SOURCE[0]}")/validate.sh.pre-forwarder" "$@"
fi

cat >&2 <<'EOF'
[mind-mem][deprecation] validate.sh is now a forwarder.
    Canonical: python3 -m mind_mem.validate_py [workspace_path]
    Bypass:    MIND_MEM_VALIDATE_BASH=1 bash validate.sh
The bash shim is removed in v4.0.
EOF

# --- interpreter resolution -------------------------------------------------
# `exec python3 -m mind_mem.validate_py` was wrong whenever the `python3` first
# on PATH is not the interpreter mind_mem is installed into. Measured 2026-08-29:
# a fresh init() run under python3.12 wrote a workspace whose validate.sh then
# resolved `python3` -> 3.14.4, which has no mind_mem, so validate.sh exited 1
# with a bare ModuleNotFoundError on stderr. The caller printed only stdout, so
# the failure surfaced as an empty message ("validate.sh failed:" and nothing).
#
# Running the sibling maintenance/validate_py.py directly is NOT an option: it
# uses package-relative imports (`from .block_parser import ...`) and raises
# "attempted relative import with no known parent package" as a loose file.
#
# Order: explicit override, then the interpreter baked in at init time, then
# whatever is on PATH that can actually import mind_mem.
PY_BAKED="@MIND_MEM_PYTHON@"
# The sentinel is assembled from two halves ON PURPOSE. A global
# replace of the placeholder string would otherwise rewrite the comparison
# below as well, turning the "was it substituted?" test into [[ X != X ]] --
# always false, so the baked interpreter would be silently ignored and the
# shim would fall through to the PATH search. (Hit exactly this 2026-08-29.)
_MM_UNSUBSTITUTED='@MIND_MEM_'"PYTHON@"

_mm_can_import() { "$1" -c 'import mind_mem' >/dev/null 2>&1; }

PY=""
_mm_pick() {  # $1 = interpreter; succeeds if it can import mind_mem as-is
    command -v "$1" >/dev/null 2>&1 && _mm_can_import "$1"
}

if [[ -n "${MIND_MEM_PYTHON:-}" ]]; then
    PY="$MIND_MEM_PYTHON"          # explicit override always wins
elif [[ "$PY_BAKED" != "$_MM_UNSUBSTITUTED" ]] && [[ -x "$PY_BAKED" ]] \
        && _mm_can_import "$PY_BAKED"; then
    PY="$PY_BAKED"                 # workspace copy: interpreter baked at init
else
    for cand in python3 python; do
        if _mm_pick "$cand"; then PY="$cand"; break; fi
    done
    if [[ -z "$PY" ]]; then
        # In-repo copy (src/mind_mem/validate.sh) with mind_mem not installed into
        # any interpreter on PATH: the package is sitting right next to us, so put
        # its parent on PYTHONPATH rather than giving up. This is the path the repo's
        # own tests take -- they invoke the package copy directly, which never gets
        # an interpreter baked in (that only happens when init() writes a workspace).
        _MM_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        _MM_PKG_PARENT="$(dirname "$_MM_HERE")"
        if [[ -f "$_MM_PKG_PARENT/mind_mem/__init__.py" ]]; then
            for cand in python3 python; do
                command -v "$cand" >/dev/null 2>&1 || continue
                if PYTHONPATH="$_MM_PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}" \
                        "$cand" -c 'import mind_mem' >/dev/null 2>&1; then
                    export PYTHONPATH="$_MM_PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}"
                    PY="$cand"
                    break
                fi
            done
        fi
    fi
fi

if [[ -z "$PY" ]]; then
    echo "[mind-mem][error] no interpreter on PATH can 'import mind_mem'." >&2
    echo "    Set MIND_MEM_PYTHON=/path/to/python to point at the right one." >&2
    exit 127
fi

exec "$PY" -m mind_mem.validate_py "$@"
