"""The MCP handshake identifies the product, not its framework dependency."""

import asyncio

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402

from mind_mem import __version__  # noqa: E402
from mind_mem.mcp.server import mcp  # noqa: E402


def test_initialize_reports_mind_mem_package_version() -> None:
    async def initialize() -> None:
        async with Client(mcp) as client:
            # FastMCP 3.x returns the legacy InitializeResult. FastMCP 4.x
            # may negotiate server/discover instead, where the same metadata
            # is exposed through the era-neutral client.server_info property.
            try:
                result = await client.initialize()
            except RuntimeError as exc:
                if "no InitializeResult" not in str(exc):
                    raise
                server_info = client.server_info
            else:
                server_info = result.serverInfo
            assert server_info is not None
            assert server_info.name == "mind-mem"
            assert server_info.version == __version__

    asyncio.run(initialize())
