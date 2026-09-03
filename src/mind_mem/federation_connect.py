# Copyright 2026 STARGA, Inc.
"""``mind-mem-connect`` — join an existing federation without hand-editing config.

The substrate for shared context across several CLIs already exists: a
Postgres-backed block store and a Redis recall cache that every mind-mem process
pointed at the same pair can see. What was missing was the join. A second CLI on
a second machine had to be told, by hand, to open ``mind-mem.json``, add a
``block_store`` section with the right three keys, add ``cache.redis_url``, and
set ``recall.backend`` to the value that makes the FTS index — rather than the
now-empty local Markdown corpus — the thing recall reads. Getting any one of
those wrong produces a workspace that starts cleanly and finds nothing.

This module is that join, as one command.

**It edits configuration and nothing else.** It writes no block, appends no
audit entry, and creates no corpus. A workspace that was already initialised
keeps every setting this command does not name: the existing config is read,
merged onto, and written back, so ``governance_mode``, ACLs, limits and every
other section survive a connect.

**Secrets.** A Postgres DSN and a Redis URL usually carry a password, and a
password on a command line is visible in ``ps`` to every user on the box and is
kept in the shell history file. So the environment is the default source
(``MIND_MEM_DSN`` / ``MIND_MEM_REDIS_URL``), the ``--dsn`` / ``--redis-url``
flags exist for scripted use and say what they cost, the config file is written
``0600``, and every line this command prints has the credential redacted —
including its error messages, which is where a URL most often leaks.

**Fail closed on the URL scheme.** A DSN is a connection instruction; accepting
an arbitrary scheme means accepting ``file://`` and whatever a client library
does with it. Only ``postgresql``/``postgres`` and ``redis``/``rediss``/``unix``
are admitted, and an unrecognised scheme is refused with the scheme named and
the rest of the URL withheld.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "POSTGRES_SCHEMES",
    "REDIS_SCHEMES",
    "ConnectResult",
    "build_federation_config",
    "connect",
    "main",
    "redact_url",
]

#: Accepted Postgres DSN schemes. Anything else is refused rather than passed
#: to a driver to interpret.
POSTGRES_SCHEMES: Final[frozenset[str]] = frozenset({"postgresql", "postgres"})

#: Accepted Redis URL schemes — TCP, TLS, and a local unix socket.
REDIS_SCHEMES: Final[frozenset[str]] = frozenset({"redis", "rediss", "unix"})

#: The recall backend a Postgres-backed workspace must use. Mirrors
#: ``init_workspace._BACKEND_RECALL["postgres"]``: with the corpus in Postgres,
#: the local Markdown tree is empty and BM25-over-Markdown finds nothing, so
#: recall has to read the backend-mirroring SQLite FTS index instead.
POSTGRES_RECALL_BACKEND: Final = "sqlite"

#: Default Postgres schema, matching ``init_workspace``.
DEFAULT_SCHEMA: Final = "mind_mem"

_CONFIG_NAME: Final = "mind-mem.json"


class ConnectError(ValueError):
    """A connect request that must not be written.

    Carries an already-redacted message: a URL that failed validation is
    exactly the URL most likely to be pasted into a bug report.
    """


def redact_url(url: str) -> str:
    """Return *url* with any password replaced by ``***``.

    Used on every printed and logged form of a DSN. Falls back to the scheme
    alone when the URL cannot be parsed, because an unparseable string is
    precisely the one whose shape cannot be reasoned about.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    if not parts.hostname and not parts.path:
        return "<unparseable url>"
    if parts.password is None:
        return url
    user = parts.username or ""
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{user}:***@{host}{port}" if user else f"***@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _validated_scheme(url: str, allowed: frozenset[str], label: str) -> str:
    """Refuse *url* unless its scheme is in *allowed*. Returns the scheme."""
    try:
        scheme = urlsplit(url).scheme.lower()
    except ValueError as exc:
        raise ConnectError(f"{label} is not a parseable URL ({exc})") from exc
    if not scheme:
        raise ConnectError(f"{label} has no scheme; expected one of: {', '.join(sorted(allowed))}")
    if scheme not in allowed:
        raise ConnectError(f"{label} scheme {scheme!r} is not accepted; expected one of: {', '.join(sorted(allowed))}")
    return scheme


def build_federation_config(
    existing: dict[str, Any] | None,
    *,
    dsn: str | None = None,
    redis_url: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Return a **new** config: *existing* with the federation keys applied.

    Immutable by construction — *existing* is deep-copied through JSON before
    anything is set, so a caller that passes a live config dict does not have it
    changed underneath. Sections the caller did not name are carried through
    untouched; this is a merge, not a template.

    Args:
        existing: The workspace's current config, or ``None`` for a fresh one.
        dsn: Postgres connection string. When given, sets ``block_store`` and
            moves ``recall.backend`` to :data:`POSTGRES_RECALL_BACKEND`.
        redis_url: Redis URL for the shared recall cache. When given, sets
            ``cache.redis_url`` and leaves ``cache.enabled`` alone unless it is
            absent, in which case the cache is switched on — a connect that
            wired a URL into a disabled cache would be a silent no-op.
        schema: Postgres schema; defaults to :data:`DEFAULT_SCHEMA`, and an
            existing schema in the config wins over that default so a reconnect
            that names only a new DSN does not silently relocate the corpus.

    Raises:
        ConnectError: neither *dsn* nor *redis_url* was supplied, or a URL's
            scheme is not accepted.
    """
    if not dsn and not redis_url:
        raise ConnectError("nothing to connect: supply a Postgres DSN, a Redis URL, or both")
    if dsn:
        _validated_scheme(dsn, POSTGRES_SCHEMES, "the Postgres DSN")
    if redis_url:
        _validated_scheme(redis_url, REDIS_SCHEMES, "the Redis URL")

    config: dict[str, Any] = json.loads(json.dumps(existing)) if isinstance(existing, dict) else {}

    if dsn:
        previous_store = config.get("block_store")
        previous_schema = previous_store.get("schema") if isinstance(previous_store, dict) else None
        config["block_store"] = {
            "backend": "postgres",
            "dsn": dsn,
            "schema": schema or previous_schema or DEFAULT_SCHEMA,
        }
        recall = config.get("recall")
        recall = dict(recall) if isinstance(recall, dict) else {}
        recall["backend"] = POSTGRES_RECALL_BACKEND
        config["recall"] = recall

    if redis_url:
        cache = config.get("cache")
        cache = dict(cache) if isinstance(cache, dict) else {}
        cache["redis_url"] = redis_url
        cache.setdefault("enabled", True)
        config["cache"] = cache

    return config


class ConnectResult:
    """What a connect did, in a form safe to print.

    ``summary`` is redacted; the raw DSN never leaves the config file.
    """

    def __init__(self, *, config_path: str, config: dict[str, Any], written: bool) -> None:
        self.config_path = config_path
        self.config = config
        self.written = written

    @staticmethod
    def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
        section = config.get(name)
        return section if isinstance(section, dict) else {}

    @property
    def summary(self) -> dict[str, Any]:
        store = self._section(self.config, "block_store")
        cache = self._section(self.config, "cache")
        recall = self._section(self.config, "recall")
        return {
            "config_path": self.config_path,
            "written": self.written,
            "block_store_backend": store.get("backend", ""),
            "dsn": redact_url(str(store.get("dsn", ""))),
            "schema": store.get("schema", ""),
            "redis_url": redact_url(str(cache.get("redis_url", ""))),
            "cache_enabled": cache.get("enabled", None),
            "recall_backend": recall.get("backend", ""),
        }


def _read_config(path: str) -> dict[str, Any]:
    """The workspace's current config, or ``{}`` when there is none.

    A malformed config is an error rather than an empty dict: overwriting it
    would silently discard settings the operator cannot get back.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ConnectError(f"{path} exists but cannot be read as JSON ({exc}); refusing to overwrite it") from exc
    if not isinstance(data, dict):
        raise ConnectError(f"{path} does not hold a JSON object; refusing to overwrite it")
    return data


def _write_config(path: str, config: dict[str, Any]) -> None:
    """Write *config* atomically at ``0600``.

    Atomic because a half-written config is a workspace that will not start,
    and ``0600`` because this file now holds a database password. The mode is
    set on the temporary file *before* the rename, so the credential is never
    on disk world-readable, not even briefly.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".mind-mem-connect-", suffix=".json")
    try:
        # ``os.fchmod`` exists on Windows only from 3.13; ``requires-python``
        # admits 3.10, and every Windows CI row below 3.13 raised
        # AttributeError here -- `mind-mem-connect` could not write a config on
        # that platform at all. ``mkstemp`` already creates the file 0600 on
        # POSIX regardless of umask; the fchmod is the belt to that braces and
        # the chmod-by-path is its portable spelling. On Windows the mode bits
        # only carry read-only, so the owner-only property there is the
        # directory's ACL, which this function cannot express and does not
        # claim to.
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        else:  # pragma: no cover - Windows below 3.13
            os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def connect(
    workspace: str,
    *,
    dsn: str | None = None,
    redis_url: str | None = None,
    schema: str | None = None,
    dry_run: bool = False,
) -> ConnectResult:
    """Point *workspace* at a federation's Postgres + Redis, and report what changed.

    Idempotent: running it twice with the same inputs leaves the same file.
    ``dry_run`` computes and returns the config without touching disk, so an
    operator can see the merged result — redacted — before committing to it.
    """
    root = os.path.abspath(workspace)
    path = os.path.join(root, _CONFIG_NAME)
    merged = build_federation_config(_read_config(path), dsn=dsn, redis_url=redis_url, schema=schema)
    if not dry_run:
        _write_config(path, merged)
    return ConnectResult(config_path=path, config=merged, written=not dry_run)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mind-mem-connect",
        description=(
            "Join an existing mind-mem federation: write this workspace's Postgres DSN "
            "and Redis URL into mind-mem.json, with the recall backend set to match. "
            "Configuration only — no blocks are written and the audit chain is untouched."
        ),
        epilog=(
            "Credentials are read from MIND_MEM_DSN and MIND_MEM_REDIS_URL by default. "
            "The --dsn / --redis-url flags exist for scripted use, but a password on a "
            "command line is visible in `ps` and lands in the shell history file."
        ),
    )
    parser.add_argument("--workspace", default=".", help="Workspace root holding mind-mem.json (default: .)")
    parser.add_argument("--dsn", default=None, help="Postgres DSN. Prefer MIND_MEM_DSN.")
    parser.add_argument("--redis-url", default=None, help="Redis URL. Prefer MIND_MEM_REDIS_URL.")
    parser.add_argument("--schema", default=None, help=f"Postgres schema (default: existing, else {DEFAULT_SCHEMA})")
    parser.add_argument("--dry-run", action="store_true", help="Print the merged config's summary; write nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code; prints only redacted values."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("MIND_MEM_DSN") or None
    redis_url = args.redis_url or os.environ.get("MIND_MEM_REDIS_URL") or None

    try:
        result = connect(
            args.workspace,
            dsn=dsn,
            redis_url=redis_url,
            schema=args.schema,
            dry_run=args.dry_run,
        )
    except ConnectError as exc:
        print(f"mind-mem-connect: {exc}")
        return 2
    except OSError as exc:
        print(f"mind-mem-connect: could not write the config ({exc})")
        return 1

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    if not result.written:
        print("\nDry run — nothing was written.")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised through main()
    raise SystemExit(main())
