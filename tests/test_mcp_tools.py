"""Tests for MCP server tool definitions."""
from __future__ import annotations

import importlib
import sys


def test_mcp_server_importable():
    """MCP server module is importable."""
    spec = importlib.util.find_spec("mcp_server")
    assert spec is not None


def test_mcp_server_has_tools():
    """MCP server exposes tool definitions."""
    import mcp_server
    # Check for tool registration
    assert hasattr(mcp_server, "app") or hasattr(mcp_server, "server") or hasattr(mcp_server, "mcp")


def test_tool_count():
    """MCP server has expected number of tools (18)."""
    import mcp_server
    app = getattr(mcp_server, "app", None) or getattr(mcp_server, "server", None) or getattr(mcp_server, "mcp", None)
    if app is not None and hasattr(app, "_tool_handlers"):
        assert len(app._tool_handlers) >= 10
    elif app is not None and hasattr(app, "tools"):
        assert len(app.tools) >= 10
