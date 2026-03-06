#!/usr/bin/env python3
"""Source-checkout entrypoint for the packaged Mind-Mem MCP server."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_IMPL_PATH = os.path.join(_SRC_DIR, "mind_mem", "mcp_server.py")
with open(_IMPL_PATH, "r", encoding="utf-8") as f:
    _SOURCE = f.read()

exec(compile(_SOURCE, _IMPL_PATH, "exec"), globals(), globals())
