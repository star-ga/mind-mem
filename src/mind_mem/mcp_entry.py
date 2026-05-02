"""Thin entry point for the ``mind-mem-mcp`` console script.

Resolves and launches :func:`mind_mem.mcp_server.main`. When invoked
from a source checkout (as a fallback for developers who run
``./install.sh`` against an editable copy without first running
``pip install -e .``), it injects the checkout's ``src/`` and repo
root onto ``sys.path``. In a packaged install the import succeeds
on the first try.
"""

from __future__ import annotations

import os
import sys


def _augment_path_for_source_checkout() -> None:
    """When running from a checkout that has ``mcp_server.py`` at the
    repo root and a ``src/`` tree, make those importable.

    No-op in a packaged install — ``mind_mem`` is already importable.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    server_py = os.path.join(repo_root, "mcp_server.py")
    src_dir = os.path.join(repo_root, "src")
    if not os.path.isfile(server_py):
        return
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    """Launch the Mind-Mem MCP server."""
    try:
        from mind_mem.mcp_server import main as server_main
    except ImportError:
        _augment_path_for_source_checkout()
        try:
            from mind_mem.mcp_server import main as server_main
        except ImportError as exc:
            print(
                "Error: cannot import mind_mem.mcp_server (" + str(exc) + ").",
                file=sys.stderr,
            )
            print(
                'Install the MCP extra:  pipx install "mind-mem[mcp]"',
                file=sys.stderr,
            )
            print(
                '                     or pip install "mind-mem[mcp]"',
                file=sys.stderr,
            )
            sys.exit(1)

    server_main()


if __name__ == "__main__":
    main()
