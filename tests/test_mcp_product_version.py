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
            result = await client.initialize()
            assert result.serverInfo.name == "mind-mem"
            assert result.serverInfo.version == __version__

    asyncio.run(initialize())
