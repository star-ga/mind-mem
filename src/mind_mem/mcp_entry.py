"""Thin entry point for mind-mem-mcp console script."""

import os
import sys


def main():
    """Launch the Mind-Mem MCP server."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server_py = os.path.join(repo_root, "mcp_server.py")
    src_dir = os.path.join(repo_root, "src")

    if os.path.isfile(server_py):
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    try:
        from mind_mem.mcp_server import main as server_main
    except ImportError:
        print("Error: mind_mem.mcp_server module not found.")
        print("Run from the mind-mem repository root or install the package first.")
        sys.exit(1)

    server_main()


if __name__ == "__main__":
    main()
