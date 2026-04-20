"""Storage factory for mind-mem block stores (v3.2.0).

Provides a single entry point for constructing a BlockStore from
workspace config. Callers should migrate from direct
``MarkdownBlockStore(workspace)`` construction to ``get_block_store(workspace)``
at their own pace. The Postgres adapter ships in v3.2.0 PR-5.

Usage::

    from mind_mem.storage import get_block_store

    store = get_block_store("/path/to/workspace")
    store = get_block_store("/path/to/workspace", config={"block_store": {"backend": "encrypted"}})
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..block_store import BlockStore, MarkdownBlockStore

__all__ = ["get_block_store"]

_SUPPORTED_BACKENDS = ("markdown", "encrypted", "postgres")


def _load_workspace_config(workspace: str) -> dict[str, Any]:
    """Load mind-mem.json from *workspace*, return empty dict on failure."""
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
    return {}


def get_block_store(workspace: str, config: dict[str, Any] | None = None) -> BlockStore:
    """Construct and return a BlockStore for *workspace*.

    Reads ``config["block_store"]`` to determine the backend. When *config*
    is ``None`` the factory auto-loads ``<workspace>/mind-mem.json``.

    Args:
        workspace: Absolute or relative path to the mind-mem workspace root.
        config:    Full config dict (the parsed contents of ``mind-mem.json``),
                   or ``None`` to auto-load from disk.

    Returns:
        A :class:`~mind_mem.block_store.BlockStore` instance ready for use.

    Raises:
        ValueError: The ``block_store.backend`` value is not recognised, or
                    the ``encrypted`` backend is requested but
                    ``MIND_MEM_ENCRYPTION_PASSPHRASE`` is not set.
        NotImplementedError: The ``postgres`` backend is requested (lands in
                             v3.2.0 PR-5).
    """
    if config is None:
        config = _load_workspace_config(workspace)

    bs_cfg: dict[str, Any] = config.get("block_store", {})
    backend: str = bs_cfg.get("backend", "markdown")

    if backend == "markdown":
        return MarkdownBlockStore(workspace)

    if backend == "encrypted":
        passphrase = os.environ.get("MIND_MEM_ENCRYPTION_PASSPHRASE", "").strip()
        if not passphrase:
            raise ValueError(
                "block_store.backend='encrypted' requires the "
                "MIND_MEM_ENCRYPTION_PASSPHRASE environment variable to be set"
            )
        from ..block_store_encrypted import EncryptedBlockStore

        inner = MarkdownBlockStore(workspace)
        return EncryptedBlockStore(workspace, passphrase=passphrase, inner=inner)

    if backend == "postgres":
        raise NotImplementedError("Postgres backend lands in v3.2.0 PR-5")

    raise ValueError(
        f"Unknown block_store.backend={backend!r}. "
        f"Supported values: {', '.join(repr(b) for b in _SUPPORTED_BACKENDS)}"
    )
