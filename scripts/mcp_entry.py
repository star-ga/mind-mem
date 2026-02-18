"""Thin entry point for mind-mem-mcp console script.

Routes to the main mcp_server.py at repo root.
"""

import os
import sys


def main():
    """Launch the Mind-Mem MCP server."""
    # mcp_server.py lives at repo root (one level up from scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_py = os.path.join(repo_root, "mcp_server.py")

    if os.path.isfile(server_py):
        # Add repo root to path so mcp_server.py can be imported
        sys.path.insert(0, repo_root)
        from mcp_server import main as server_main
        server_main()
    else:
        # Fallback: run directly via exec (installed via pip)
        print("Error: mcp_server.py not found. Run from the mind-mem repository root,")
        print("or use: python3 /path/to/mind-mem/mcp_server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
