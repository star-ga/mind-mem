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
from typing import Any, cast

from ..block_store import BlockStore, MarkdownBlockStore
from ..observability import get_logger

__all__ = ["get_block_store", "iter_active_blocks", "get_active_blocks"]

_SUPPORTED_BACKENDS = ("markdown", "encrypted", "postgres")

# Backends whose blocks of record live on the local Markdown corpus
# (decisions/DECISIONS.md, …). For these we enumerate via the
# ``CORPUS_FILES`` registry + ``parse_file``; for every other backend
# (e.g. ``postgres``) the blocks live in the store and must be read
# through ``get_block_store(ws).get_all(active_only=True)``.
_MARKDOWN_BACKENDS: frozenset[str] = frozenset({"markdown", "encrypted"})

_log = get_logger("storage")


def _load_workspace_config(workspace: str) -> dict[str, Any]:
    """Load mind-mem.json from *workspace*, return empty dict on failure."""
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                raw: dict[str, Any] = json.load(fh)
                return raw
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
    return {}  # type: ignore[return-value]


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
                    ``MIND_MEM_ENCRYPTION_PASSPHRASE`` is not set, or the
                    ``postgres`` backend is selected but ``dsn`` is absent.
        ImportError: The ``postgres`` backend is requested but
                     ``psycopg[binary]`` is not installed.
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
            raise ValueError("block_store.backend='encrypted' requires the MIND_MEM_ENCRYPTION_PASSPHRASE environment variable to be set")
        from ..block_store_encrypted import EncryptedBlockStore

        inner = MarkdownBlockStore(workspace)
        # cast: EncryptedBlockStore satisfies BlockStore structurally;
        # mypy can't prove it without Protocol membership (PR-4 will add it).
        return cast(BlockStore, EncryptedBlockStore(workspace, passphrase=passphrase, inner=inner))

    if backend == "postgres":
        dsn: str = bs_cfg.get("dsn", "")
        if not dsn:
            raise ValueError("block_store.backend='postgres' requires block_store.dsn to be set in mind-mem.json")
        try:
            from ..block_store_postgres import PostgresBlockStore
        except ImportError as exc:
            raise ImportError('The PostgreSQL backend requires psycopg. Install it with: pip install "mind-mem[postgres]"') from exc

        # v3.9: route through ReplicatedPostgresBlockStore when
        # block_store.replicas is a non-empty list. Reads round-robin
        # to replicas; writes always go to the primary.
        replicas = bs_cfg.get("replicas") or []
        if not isinstance(replicas, list):
            raise ValueError("block_store.replicas must be a list of DSN strings")
        replicas = [r for r in replicas if isinstance(r, str) and r.strip()]
        schema = bs_cfg.get("schema", "mind_mem")
        if replicas:
            from ..block_store_postgres_replica import ReplicatedPostgresBlockStore

            return cast(
                BlockStore,
                ReplicatedPostgresBlockStore(
                    primary_dsn=dsn,
                    replica_dsns=replicas,
                    schema=schema,
                    workspace=workspace,
                ),
            )
        return cast(BlockStore, PostgresBlockStore(dsn=dsn, schema=schema, workspace=workspace))

    raise ValueError(f"Unknown block_store.backend={backend!r}. Supported values: {', '.join(repr(b) for b in _SUPPORTED_BACKENDS)}")


def _backend_name(workspace: str, config: dict[str, Any] | None = None) -> str:
    """Return the configured ``block_store.backend`` for *workspace*.

    Defaults to ``"markdown"`` when no config or no ``block_store``
    section is present — matching :func:`get_block_store`. Never raises;
    a malformed config degrades to the markdown default so the SQLite /
    Markdown zero-config path is unaffected.
    """
    if config is None:
        config = _load_workspace_config(workspace)
    bs_cfg = config.get("block_store") if isinstance(config, dict) else None
    if not isinstance(bs_cfg, dict):
        return "markdown"
    backend = bs_cfg.get("backend", "markdown")
    return backend if isinstance(backend, str) else "markdown"


def _iter_markdown_active_blocks(workspace: str) -> list[dict[str, Any]]:
    """Enumerate active blocks from the local Markdown corpus.

    Single source of truth for the markdown-corpus enumeration used by
    the feature layer (scan / governance / export / reindex). Mirrors
    the ``_recall_core`` corpus walk: iterate :data:`CORPUS_FILES`,
    :func:`parse_file` each present file, keep only active blocks, tag
    each with ``_source_file`` / ``_source_label``, and exclude
    unreviewed pending signals (the same ``#429`` rule recall applies).
    """
    # Lazy imports keep this module import-safe (no recall/parse cost at
    # ``import mind_mem.storage`` time) and avoid an import cycle through
    # the recall constants.
    from .._recall_constants import CORPUS_FILES
    from ..block_parser import get_active, parse_file

    blocks: list[dict[str, Any]] = []
    for label, rel_path in CORPUS_FILES.items():
        path = os.path.join(workspace, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            parsed = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.debug("corpus_parse_failed", file=rel_path, error=str(exc))
            continue
        parsed = get_active(parsed)
        # #429: unreviewed signals are not part of the active corpus.
        if label == "signals":
            parsed = [b for b in parsed if str(b.get("Status", "")).lower() != "pending"]
        for b in parsed:
            b.setdefault("_source_file", rel_path)
            b.setdefault("_source_label", label)
            blocks.append(b)
    return blocks


def iter_active_blocks(workspace: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the configured backend's active blocks for *workspace*.

    This is the backend-aware enumeration primitive the feature layer
    (scan, governance, export, reindex, dream_cycle, drift) must route
    through instead of hand-rolling a ``parse_file`` over the Markdown
    corpus. It guarantees a Postgres user's blocks are visible to those
    features while leaving the Markdown / SQLite default path byte-for-
    byte unchanged.

    Behaviour:

    * **Markdown / encrypted backend** (the default) — enumerate the
      local Markdown corpus via :func:`parse_file` over
      :data:`CORPUS_FILES`, keep active blocks, and tag each with
      ``_source_file`` / ``_source_label`` (see
      :func:`_iter_markdown_active_blocks`).
    * **Any other backend** (e.g. ``postgres``) — delegate to
      ``get_block_store(workspace).get_all(active_only=True)`` so the
      blocks of record in the store are returned.

    Args:
        workspace: Path to the mind-mem workspace root.
        config:    Parsed ``mind-mem.json`` dict, or ``None`` to
                   auto-load from ``<workspace>/mind-mem.json``.

    Returns:
        A list of block dicts (each carrying at least ``_id``). The list
        is fresh on every call; callers may mutate it freely.

    Notes:
        Never raises for a missing / malformed config — it degrades to
        the markdown default. A non-markdown store that itself fails
        (e.g. Postgres unreachable) propagates that store's error, since
        silently returning ``[]`` would hide real blocks from
        governance.
    """
    if config is None:
        config = _load_workspace_config(workspace)
    backend = _backend_name(workspace, config)
    if backend in _MARKDOWN_BACKENDS:
        return _iter_markdown_active_blocks(workspace)
    store = get_block_store(workspace, config=config)
    return store.get_all(active_only=True)


def get_active_blocks(workspace: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Alias of :func:`iter_active_blocks` (companion accessor).

    Provided for call-site readability where ``get_active_blocks`` reads
    more naturally than ``iter_active_blocks`` (the two are identical).
    """
    return iter_active_blocks(workspace, config=config)
