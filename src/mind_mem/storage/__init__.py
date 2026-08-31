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
from collections.abc import Callable
from typing import Any, cast

from ..block_store import BlockStore, MarkdownBlockStore
from ..observability import get_logger

__all__ = ["get_block_store", "iter_active_blocks", "get_active_blocks", "iter_blocks"]

_SUPPORTED_BACKENDS = ("markdown", "encrypted", "postgres")

# Backends whose blocks of record live on the local Markdown corpus
# (decisions/DECISIONS.md, …). For these we enumerate via the
# ``CORPUS_FILES`` registry; for every other backend (e.g. ``postgres``)
# the blocks live in the store and must be read through
# ``get_block_store(ws).get_all(active_only=True)``.
#
# ``encrypted`` belongs here — it wraps the markdown backend and its
# blocks of record are those same corpus files — but the two do NOT
# share a *reader*: an encrypted corpus holds ciphertext, so the file
# is opened through :func:`_corpus_parse_fn`, never ``parse_file``
# directly. Membership in this set means "corpus-resident", not
# "plaintext".
_MARKDOWN_BACKENDS: frozenset[str] = frozenset({"markdown", "encrypted"})

_log = get_logger("storage")


def _load_workspace_config(workspace: str) -> dict[str, Any]:
    """Load ``mind-mem.json`` from *workspace*; empty dict on failure.

    The empty dict is a *degrade*, and the degrade is load-bearing: with
    no config, :func:`get_block_store` builds a ``MarkdownBlockStore``
    and the corpus walk reads ``decisions/DECISIONS.md``. So a workspace
    configured for Postgres whose config file is corrupt or unreadable
    silently starts reading and writing the local Markdown corpus
    instead of the database — a data-destination change, reported as
    success. The failure used to be swallowed by a bare ``pass``; it is
    now logged with the path and the reason, which is the difference
    between a diagnosable downgrade and an invisible one.

    Never raises: every caller (including the never-raising
    :func:`_backend_name`) depends on that.
    """
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw: Any = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        _log.warning(
            "workspace_config_unreadable",
            path=config_path,
            error=f"{type(exc).__name__}: {exc}",
            effect="falling back to the default markdown block store",
        )
        return {}
    if not isinstance(raw, dict):
        _log.warning(
            "workspace_config_not_an_object",
            path=config_path,
            found=type(raw).__name__,
            effect="falling back to the default markdown block store",
        )
        return {}
    return raw


def _block_store_section(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate ``config["block_store"]``; return ``(section, error)``.

    One predicate for both entry points. They used to enforce different
    ones on the same value: :func:`_backend_name` checked
    ``isinstance(bs_cfg, dict)`` and degraded, while
    :func:`get_block_store` called ``.get`` on whatever was there — so
    ``{"block_store": "postgres"}`` made the router answer "markdown"
    (quietly reading the wrong store) and the constructor raise
    ``AttributeError: 'str' object has no attribute 'get'``, which is not
    the ``ValueError`` its docstring promises. Two entry points, one
    config, two different and both-wrong answers.

    *error* is the empty string when the section is usable. The two
    callers dispose of it differently — the constructor raises, the
    never-raising router logs and degrades — but they now agree on
    *whether* the config is malformed.
    """
    section = config.get("block_store", {})
    if section is None:
        return {}, ""
    if not isinstance(section, dict):
        return {}, f"block_store must be an object, got {type(section).__name__}"
    backend = section.get("backend", "markdown")
    if not isinstance(backend, str):
        return section, f"block_store.backend must be a string, got {type(backend).__name__}"
    return section, ""


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

    bs_cfg, cfg_error = _block_store_section(config)
    if cfg_error:
        raise ValueError(f"Malformed block_store configuration: {cfg_error}")
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
    Markdown zero-config path is unaffected — but the degrade is
    **logged**, and the malformed-ness is decided by
    :func:`_block_store_section`, the same predicate
    :func:`get_block_store` rejects on. The two disagreeing about what
    counts as malformed is what let one silently read the local corpus
    while the other raised ``AttributeError``.
    """
    if config is None:
        config = _load_workspace_config(workspace)
    if not isinstance(config, dict):
        _log.warning(
            "block_store_config_malformed",
            reason=f"config must be an object, got {type(config).__name__}",
            effect="degrading to the markdown backend",
        )
        return "markdown"
    bs_cfg, cfg_error = _block_store_section(config)
    if cfg_error:
        # Same predicate get_block_store rejects on; this entry point must
        # not raise (recall, drift, dream and the MCP workspace probe all
        # call it on every request), so it degrades — but audibly, so the
        # degrade is not indistinguishable from "no config".
        _log.warning("block_store_config_malformed", reason=cfg_error, effect="degrading to the markdown backend")
        return "markdown"
    backend = bs_cfg.get("backend", "markdown")
    return backend if isinstance(backend, str) else "markdown"


def _corpus_has_ciphertext(workspace: str) -> bool:
    """True when any corpus file the walk reads starts with the encryption marker.

    Probes exactly the paths :func:`_iter_markdown_active_blocks`
    enumerates, so the answer is about the bytes that walk will actually
    read. Cheap: one open + a 6-byte read per registered corpus file,
    missing files skipped.
    """
    from .._recall_constants import CORPUS_FILES
    from ..encryption import _MAGIC

    for rel_path in CORPUS_FILES.values():
        try:
            with open(os.path.join(workspace, rel_path), "rb") as fh:
                if fh.read(len(_MAGIC)) == _MAGIC:
                    return True
        except OSError:
            continue
    return False


def _corpus_parse_fn(workspace: str, backend: str) -> Callable[[str], list[dict[str, Any]]]:
    """Return the reader the corpus walk must use for *backend*.

    ``markdown`` reads its corpus files straight off disk. ``encrypted``
    keeps the blocks of record in those same files, but
    ``encrypt_workspace`` / the ``encrypt_file`` tool rewrite them in
    place as ciphertext — and ``parse_file`` would decode that ciphertext
    with ``errors="replace"``, find no ``[ID]`` header, and return zero
    blocks *without raising*. Every consumer of
    :func:`iter_active_blocks` (reindex, scan, drift, dream, export,
    workspace health) would then run on an empty corpus and report
    success on a workspace that is full of blocks. So the encrypted
    backend reads through :class:`~mind_mem.block_store_encrypted.EncryptedBlockStore`,
    which decrypts a file carrying the magic header and passes a plain
    one through untouched — a partially-migrated workspace stays
    readable either way.

    The decrypting reader is built only once ciphertext is actually on
    disk: constructing it derives a PBKDF2 key (600k iterations) and
    mints ``.mind-mem-keys/salt``, neither of which a not-yet-migrated
    workspace should pay for on a read.

    Raises:
        ValueError: *backend* is ``encrypted``, the corpus is ciphertext,
            and ``MIND_MEM_ENCRYPTION_PASSPHRASE`` is unset — the blocks
            exist but cannot be read, and reporting an empty corpus would
            hide them from governance. Raised here, before the walk, so
            it cannot be swallowed by the walk's per-file
            ``except ValueError`` (which exists to skip one unreadable
            file, not to mask an unreadable workspace).
    """
    from ..block_parser import parse_file

    if backend != "encrypted":
        return parse_file
    if not _corpus_has_ciphertext(workspace):
        # Configured for encryption but not migrated yet: the corpus is
        # still plaintext, so the plain reader is the correct one.
        return parse_file

    passphrase = os.environ.get("MIND_MEM_ENCRYPTION_PASSPHRASE", "").strip()
    if not passphrase:
        raise ValueError(
            "block_store.backend='encrypted' with an encrypted corpus on disk requires "
            "the MIND_MEM_ENCRYPTION_PASSPHRASE environment variable to be set"
        )

    from ..block_store_encrypted import EncryptedBlockStore

    # Reuse the store's own per-file read primitive (magic-header sniff →
    # decrypt → parse) instead of duplicating it here; a second copy of
    # that dance is a second thing to keep in sync with the envelope.
    return EncryptedBlockStore(workspace, passphrase=passphrase)._parse_maybe_encrypted


def _iter_markdown_active_blocks(
    workspace: str,
    *,
    active_only: bool = True,
    parse: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate blocks from the local Markdown corpus.

    Single source of truth for the markdown-corpus enumeration used by
    the feature layer (scan / governance / export / reindex). Mirrors
    the ``_recall_core`` corpus walk: iterate :data:`CORPUS_FILES`,
    :func:`parse_file` each present file, keep only active blocks, tag
    each with ``_source_file`` / ``_source_label``, and exclude
    unreviewed pending signals (the same ``#429`` rule recall applies).

    ``active_only=False`` skips both status filters — an explicit
    "everything on disk" enumeration, for a caller that is opening a
    named mailbox rather than searching memory. It is opt-in; every
    existing caller keeps the filtered default.

    *parse* is the per-file reader; it defaults to
    :func:`~mind_mem.block_parser.parse_file`. The ``encrypted`` backend
    passes a decrypting reader (see :func:`_corpus_parse_fn`) so the same
    walk — same file order, same labels, same ``#429`` rule — works on a
    ciphertext corpus.
    """
    # Lazy imports keep this module import-safe (no recall/parse cost at
    # ``import mind_mem.storage`` time) and avoid an import cycle through
    # the recall constants.
    from .._recall_constants import CORPUS_FILES
    from ..block_parser import get_active, parse_file

    read = parse_file if parse is None else parse
    blocks: list[dict[str, Any]] = []
    for label, rel_path in CORPUS_FILES.items():
        path = os.path.join(workspace, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            parsed = read(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.debug("corpus_parse_failed", file=rel_path, error=str(exc))
            continue
        if active_only:
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
      local Markdown corpus over :data:`CORPUS_FILES`, keep active
      blocks, and tag each with ``_source_file`` / ``_source_label``
      (see :func:`_iter_markdown_active_blocks`). An ``encrypted``
      corpus is decrypted on the way in — the reader comes from
      :func:`_corpus_parse_fn`, not ``parse_file``.
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
    return iter_blocks(workspace, config=config, active_only=True)


def iter_blocks(workspace: str, config: dict[str, Any] | None = None, *, active_only: bool = True) -> list[dict[str, Any]]:
    """Backend-aware block enumeration, optionally including withheld blocks.

    :func:`iter_active_blocks` is this with ``active_only=True`` and is
    what the feature layer should use. ``active_only=False`` exists for
    the one shape that is *not* a search: opening a named mailbox, where
    the caller asked for specific blocks by construction rather than
    having them retrieved into its context. It is never the recall path.
    """
    if config is None:
        config = _load_workspace_config(workspace)
    backend = _backend_name(workspace, config)
    if backend in _MARKDOWN_BACKENDS:
        return _iter_markdown_active_blocks(workspace, active_only=active_only, parse=_corpus_parse_fn(workspace, backend))
    store = get_block_store(workspace, config=config)
    return store.get_all(active_only=active_only)


def get_active_blocks(workspace: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Alias of :func:`iter_active_blocks` (companion accessor).

    Provided for call-site readability where ``get_active_blocks`` reads
    more naturally than ``iter_active_blocks`` (the two are identical).
    """
    return iter_active_blocks(workspace, config=config)
