"""mind_mem.tool_output — context-offload for large command/tool output (§5).

Pipe a 10k–50k-line ``cargo test`` / ``pytest`` / build log in → get back a compact
``{handle, summary}`` → keep the spam out of the agent's context → recall the full
text by handle ONLY when a failure needs the detail. The summary is DETERMINISTIC
(no LLM) and fail-SAFE (a failure line is never silently dropped).
"""

from .store import StoreResult, ToolOutputStore
from .summarize import (
    DEFAULT_CONFIG,
    SUMMARIZER_VERSION,
    SummarizerConfig,
    ToolOutputSummary,
    make_handle,
    summarize,
)

__all__ = [
    "ToolOutputStore",
    "StoreResult",
    "summarize",
    "ToolOutputSummary",
    "SummarizerConfig",
    "DEFAULT_CONFIG",
    "SUMMARIZER_VERSION",
    "make_handle",
]
