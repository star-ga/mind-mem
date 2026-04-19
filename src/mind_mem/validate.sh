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

exec python3 -m mind_mem.validate_py "$@"
