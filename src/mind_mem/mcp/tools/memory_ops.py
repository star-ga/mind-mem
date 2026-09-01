"""Memory operations MCP tools — index / lifecycle / health / export.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, memory_ops domain). Eight tools cover the workspace's
lifecycle + introspection surface:

* ``index_stats`` / ``reindex`` — FTS5 index state + rebuild.
* ``delete_memory_item`` — atomic admin-scope block removal with
  an append-only recovery log.
* ``export_memory`` — JSONL dump of every block with configurable
  metadata + size cap.
* ``get_block`` / ``memory_health`` / ``compact`` / ``stale_blocks`` —
  block lookup, health dashboard, compaction, and staleness-flag
  management built on the causal graph.

Also hosts the ``_BLOCK_PREFIX_MAP`` + ``_find_block_file``
resolver shared by ``delete_memory_item`` and ``get_block``.
"""

from __future__ import annotations

import json
import os
import re as _re_mod
import sqlite3
import tempfile
from typing import Any

from mind_mem.block_parser import BlockCorruptedError, get_active, parse_file
from mind_mem.corpus_registry import CORPUS_DIRS
from mind_mem.mind_ffi import get_mind_dir
from mind_mem.mind_ffi import is_available as mind_kernel_available
from mind_mem.mind_ffi import is_protected as mind_kernel_protected
from mind_mem.mind_ffi import list_kernels as ffi_list_kernels
from mind_mem.mind_filelock import FileLock
from mind_mem.sqlite_index import _db_path as fts_db_path
from mind_mem.storage import _MARKDOWN_BACKENDS, _backend_name, get_block_store

from ..infra.config import _load_config, _load_extra_categories
from ..infra.constants import MCP_SCHEMA_VERSION
from ..infra.observability import _is_db_locked, _sqlite_busy_error, mcp_tool_observe
from ..infra.workspace import _check_workspace, _workspace
from ._helpers import _retrieval_metrics_enabled, _signal_store_path, get_logger, metrics


def _kernel_backend_reporting_enabled() -> bool:
    """Read ``v4.mind_kernels``, fail-closed and QUIET.

    ``is_enabled_quiet``, not ``is_enabled``: the loud variant logs
    ``v4_config_unreadable`` on a malformed config, which would make an
    index_stats call on the flag-OFF path emit a line the unwired build
    never emitted.
    """
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet

        return is_enabled_quiet("mind_kernels")
    except Exception:
        return False


def _v4_kind_index_section(ws: str) -> dict[str, Any] | None:
    """``v4.hnsw_kind_index`` / ``v4.pq`` state for ``index_stats``, or None.

    ``index_stats`` answers "what indexes exist and are they current". Two v4
    side indexes now can exist, and an operator has no other way to learn
    whether ``mm kinds backfill`` ever ran or which backend answers a
    kind-filtered query.

    Both probes are QUIET (``is_enabled_quiet`` never logs) and both are read
    ONCE per tool call, before any database is opened. With both flags off
    this function opens nothing, writes nothing and returns ``None``, so the
    payload is exactly the payload of the build that never had the section.
    The OFF cost is the two small config reads that decide it -- this is an
    explicitly-invoked diagnostic, not a per-recall path.
    """
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet
    except Exception:  # pragma: no cover - defensive: v4 surface absent
        return None
    out: dict[str, Any] = {}
    if is_enabled_quiet("hnsw_kind_index"):
        try:
            from mind_mem.v4.hnsw_kind_index import backend_status

            # ``backend_status`` opens (and therefore CREATES) index.db to
            # probe for the sqlite-vec extension. A stats call must not be the
            # thing that brings a store into existence, so an unbuilt
            # workspace is reported as unbuilt instead of being built.
            if os.path.isfile(os.path.join(ws, "index.db")):
                out["hnsw_kind_index"] = backend_status(ws)
            else:
                out["hnsw_kind_index"] = {"backend": "brute_force", "status": "not_built"}
        except Exception as exc:  # noqa: BLE001 - a stats section never breaks the tool
            out["hnsw_kind_index"] = {"error": str(exc)}
    if is_enabled_quiet("pq"):
        try:
            from mind_mem.v4.pq import load_codebook

            # The codebook is stored under the embedding model's name, because
            # a model swap invalidates every code; ``VectorBackend``'s own
            # default is the fallback so the two names cannot drift.
            name = str((_load_config(ws).get("recall") or {}).get("model") or "all-MiniLM-L6-v2")
            cb = load_codebook(ws, name)
            out["pq"] = {
                "codebook": name,
                "trained": cb is not None,
                # Bytes per encoded vector == subvector count: one 8-bit
                # centroid index per subvector position.
                "bytes_per_vector": cb.cfg.subvectors if cb is not None else None,
            }
        except Exception as exc:  # noqa: BLE001
            out["pq"] = {"error": str(exc)}
    return out or None


def _v4_health_section(ws: str) -> dict[str, Any] | None:
    """``v4/health``'s probe sweep for ``memory_health``, or None when OFF.

    ``health_check`` itself is never flag-gated -- see its module docstring --
    but its CALL SITE here is, because ``memory_health``'s payload is pinned
    byte-for-byte with flags off and this section carries a ``latency_ms`` and
    a ``checked_at`` clock read. ONE quiet flag read decides it, before the
    module is imported at all, so with ``v4.health`` off nothing here reads a
    clock, opens a database or emits a line.

    ``health_check`` never raises by contract; the guard is for the import.
    """
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet

        if not is_enabled_quiet("health"):
            return None
        from mind_mem.v4.health import health_check
    except Exception as exc:  # noqa: BLE001 - v4 surface absent
        _log.debug("health_v4_probe_unavailable", error=str(exc))
        return None
    return health_check(ws)


_log = get_logger("mcp_server")


_BLOCK_PREFIX_MAP = {
    "D": ("decisions", "DECISIONS.md"),
    "T": ("tasks", "TASKS.md"),
    "C": ("intelligence", "CONTRADICTIONS.md"),
    "INC": ("entities", "incidents.md"),
    "PRJ": ("entities", "projects.md"),
    "PER": ("entities", "people.md"),
    "TOOL": ("entities", "tools.md"),
    # v3.9: inbox folder ingestion (text + PDF) writes here.
    "INBOX": ("memory", "INBOX.md"),
    # v4.0.19: agent-to-agent messaging (`mm send` / `mm inbox`). Keep in
    # lockstep with ``block_store._BLOCK_PREFIX_MAP``.
    "MSG": ("memory", "MESSAGES.md"),
    # Migration importers (roadmap Group G) (`mm import --from ...`). Keep in
    # lockstep with ``block_store._BLOCK_PREFIX_MAP``.
    "IMP": ("memory", "IMPORTED.md"),
    # 5.0.1: `mm ingest-serve` webhook door. Keep in lockstep with
    # ``block_store._BLOCK_PREFIX_MAP``.
    "INGEST": ("memory", "INGEST.md"),
}


def _find_block_file(ws: str, block_id: str) -> str | None:
    """Resolve a block ID to its source .md file path."""
    for prefix, (subdir, filename) in _BLOCK_PREFIX_MAP.items():
        if block_id.startswith(prefix + "-"):
            return os.path.join(ws, subdir, filename)
    return None


def _is_markdown_backend(ws: str) -> bool:
    """Return True when *ws*'s blocks of record live on the Markdown corpus.

    Audit bug #5: ``export_memory`` / ``get_block`` enumerated blocks via
    ``parse_file`` over :data:`CORPUS_DIRS`, so a Postgres-backed workspace
    (whose blocks live in the DB and whose corpus files are empty init
    templates) could not see its own blocks of record.

    Defaults to the Markdown corpus — the zero-config SQLite / Markdown
    default — when no config / no ``block_store`` section is present, and
    on any config-read failure. This keeps the default path byte-for-byte
    unchanged; only an explicitly non-Markdown backend (e.g. ``postgres``)
    routes through the block store.
    """
    try:
        return _backend_name(ws) in _MARKDOWN_BACKENDS
    except Exception:  # pragma: no cover - defensive: config read failure
        return True


def _store_block_is_active(block: dict) -> bool:
    """Activity of a store-resident block, on the Markdown branch's terms.

    ``memory_health`` reports one ``total_active`` field, so both branches
    must answer the same question or the number is not comparable across
    backends. The Status half therefore delegates to the very function the
    Markdown branch uses, :func:`block_parser.get_active` — spelling it out
    a second time is how the two drifted: the store branch read
    ``str(Status or "active").lower() == "active"``, which treated a
    MISSING ``Status`` as active, so a deactivated row whose metadata omits
    the field was counted.

    A store row additionally carries the column-derived ``_active`` flag
    (``block_store_postgres._row_to_block``), which ``Status`` never
    reflects — a row deactivated in the store but whose metadata still says
    ``active`` must not be counted. Both must hold; a store that surfaces
    no flag has no opinion.
    """
    if not bool(block.get("_active", True)):
        return False
    return bool(get_active([block]))


def _mrs_health_section(ws: str) -> dict[str, Any] | None:
    """The ``mrs`` block of :func:`memory_health`, or None when OFF.

    Flag: ``mrs.enabled`` in ``mind-mem.json``, **default OFF**. With the
    flag off this returns before :mod:`mind_mem.mrs` is even imported —
    no log line, no metric, no key in the payload — so ``memory_health``
    is byte-identical to what it returned before this wiring existed.
    ``tests/test_mrs_wiring.py`` pins that by differencing the output of
    a workspace with no ``mrs`` section against one whose section is
    present and disabled.

    What it scores (all readings injected, so the module itself stays a
    pure function of them — see :mod:`mind_mem.mrs`):

    * **latency** — the raw ``mcp_tool_duration_ms`` observations this
      process has recorded, which is the only series in the package that
      can answer a p99 question. An operator can override the metric
      name, or supply a literal ``latency_ms`` list.
    * **error rate** — the ``mcp_tool_failure`` / ``mcp_tool_success``
      counters. Omitted when nothing has been called yet.
    * **corpus health** — drift, contradictions and staleness, counted
      over the *admitted* corpus.

    A breach is routed to :mod:`mind_mem.alerting` unless ``alert`` is
    false. Alert-routing failure degrades to a debug line: a dashboard
    must not fail because a webhook did.
    """
    cfg = _load_config(ws)
    from mind_mem.mrs import is_mrs_enabled

    if not is_mrs_enabled(cfg):
        return None

    from mind_mem.alerting import get_alert_router
    from mind_mem.mrs import resolve_mrs_config, route_mrs_alerts, workspace_mrs_report

    resolved = resolve_mrs_config(cfg)
    latency = resolved["latency_ms"]
    if latency is None:
        latency = metrics.samples(str(resolved["latency_metric"]))
    failures = metrics.get("mcp_tool_failure")
    successes = metrics.get("mcp_tool_success")

    report = workspace_mrs_report(
        ws,
        latency_ms=latency,
        error_count=int(failures),
        request_count=int(failures) + int(successes),
        observation_days=float(resolved["observation_days"]),
        slo_spec=resolved["slo"],
        target=str(resolved["target"]),
    )
    if resolved["alert"]:
        try:
            route_mrs_alerts(
                report,
                router=get_alert_router(ws),
                alert_below=float(resolved["alert_below"]),
                severity=str(resolved["alert_severity"]),
            )
        except (OSError, ValueError) as exc:
            _log.debug("mrs_alert_routing_skipped", error=str(exc))
    return report.as_dict()


@mcp_tool_observe
def index_stats() -> str:
    """Block counts, index staleness, vector coverage, and MIND kernel status."""
    ws = _workspace()
    stats: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION}

    db = fts_db_path(ws)
    fts_exists = os.path.isfile(db) if db else False
    stats["fts_index_exists"] = fts_exists

    if fts_exists:
        try:
            from mind_mem.sqlite_index import index_status as fts_status

            fts_info = fts_status(ws)
            stats["total_blocks"] = fts_info.get("blocks", 0)
            stats["last_build"] = fts_info.get("last_build")
            stats["stale_files"] = fts_info.get("stale_files", 0)
            stats["db_size_bytes"] = fts_info.get("db_size_bytes", 0)
        except sqlite3.OperationalError as exc:
            if _is_db_locked(exc):
                return _sqlite_busy_error()
            raise
        except (OSError, ValueError, KeyError) as e:
            _log.debug("fts_status_failed", error=str(e))
            fts_exists = False

    if not fts_exists:
        if _is_markdown_backend(ws):
            for kind in CORPUS_DIRS:
                d = os.path.join(ws, kind)
                if os.path.isdir(d):
                    count = 0
                    for fn in os.listdir(d):
                        if fn.endswith(".md"):
                            try:
                                blocks = parse_file(os.path.join(d, fn))
                                count += len(blocks)
                            except (OSError, ValueError) as e:
                                _log.debug("index_stats_parse_failed", file=fn, error=str(e))
                    stats[f"{kind}_blocks"] = count
        else:
            # Audit bug #5: on a non-Markdown backend (e.g. Postgres) the
            # corpus files are empty init templates; report the store's
            # block count so an un-indexed PG workspace is not reported as
            # empty.
            try:
                store = get_block_store(ws)
                stats["total_blocks"] = len(store.get_all(active_only=False))
            except Exception as e:  # pragma: no cover - defensive: store probe
                _log.debug("index_stats_store_count_failed", error=str(e))

    v4_indexes = _v4_kind_index_section(ws)
    if v4_indexes is not None:
        stats["v4_indexes"] = v4_indexes

    mind_dir = get_mind_dir(ws)
    kernels = ffi_list_kernels(mind_dir)
    stats["mind_kernels"] = kernels
    stats["mind_kernel_compiled"] = mind_kernel_available()
    stats["mind_kernel_protected"] = mind_kernel_protected()

    # v4.mind_kernels (default OFF): report which library the ONE loader
    # actually bound. `mind_kernel_compiled` above answers "did a .so load
    # at some point in this process" from a cached singleton; this answers
    # "what would a caller asking for kernels right now get". With the flag
    # OFF the key is absent and the response is byte-identical to 5.0.0.
    if _kernel_backend_reporting_enabled():
        from mind_mem.mind_ffi import load_kernels as _load_kernels

        stats["mind_kernel_backend"] = _load_kernels().backend

    try:
        from mind_mem.prefix_cache import all_stats as _prefix_all_stats

        stats["prefix_caches"] = [s.as_dict() for s in _prefix_all_stats()]
    except (ImportError, AttributeError) as exc:
        _log.debug("prefix_cache_stats_unavailable", error=str(exc))
        stats["prefix_caches"] = []

    try:
        from mind_mem.speculative_prefetch import get_default_predictor

        stats["speculative_prefetch"] = get_default_predictor().stats().as_dict()
    except (ImportError, AttributeError) as exc:
        _log.debug("speculative_prefetch_stats_unavailable", error=str(exc))
        stats["speculative_prefetch"] = {}

    try:
        from mind_mem.interaction_signals import SignalStore

        sig_store = SignalStore(_signal_store_path(ws))
        stats["interaction_signals"] = sig_store.stats().as_dict()
    except (ImportError, AttributeError, OSError) as exc:
        _log.debug("interaction_signal_stats_unavailable", error=str(exc))
        stats["interaction_signals"] = {}

    # Online-training state, beside the signals it is derived from.
    # ``is_enabled_quiet``, and the key is ABSENT rather than empty when the
    # flag is off: an added `"online_training": {}` would already be an
    # observable difference in a response clients diff.
    try:
        from mind_mem.v4.feature_flags import is_enabled_quiet

        _online_training = is_enabled_quiet("online_training")
    except Exception:  # pragma: no cover - defensive: flag resolver
        _online_training = False
    if _online_training:
        try:
            from mind_mem.model_gate import promotion_stats
            from mind_mem.online_trainer import harvest_stats

            stats["online_training"] = {**harvest_stats(ws), "promotions": promotion_stats()}
        except (ImportError, AttributeError, OSError, ValueError) as exc:
            _log.debug("online_training_stats_unavailable", error=str(exc))
            stats["online_training"] = {}

    # v4.retrieval_metrics: measured retrieval quality. Absent with the flag
    # off, so the envelope is byte-identical to the one this tool has always
    # returned.
    if _retrieval_metrics_enabled(ws):
        try:
            stats["mrr"] = _mrr_drift(ws)
        except Exception as exc:
            _log.debug("mrr_drift_unavailable", error=str(exc))
            stats["mrr"] = {}
        try:
            from mind_mem.tracking import default_packing_meter

            stats["packing_quality"] = default_packing_meter().stats()
        except (ImportError, AttributeError) as exc:
            _log.debug("packing_quality_unavailable", error=str(exc))
            stats["packing_quality"] = {}

    metrics.inc("mcp_index_stats")
    _log.info("mcp_index_stats", stats=stats)
    return json.dumps(stats, indent=2)


#: Newest signals folded into the MRR series. The ledger is append-only and
#: unbounded; a stats call must not turn into a full-history scan on a
#: long-lived workspace. Newest-first, so the recent weeks the delta is
#: computed from are always the ones that survive the cap.
_MRR_SIGNAL_SCAN_CAP: int = 20_000


def _mrr_drift(ws: str) -> dict[str, Any]:
    """Weekly mean MRR + week-over-week delta for *ws*.

    Both halves of a real MRR already exist in the workspace and were never
    joined: ``observe_signal`` stores the ranked block ids a recall returned
    (plus when), and ``calibration_feedback`` stores which ids a caller then
    accepted. This scores the first against the second, buckets by ISO week
    using each signal's OWN timestamp, and reports the change.

    DETERMINISTIC: a pure function of (signal ledger, calibration labels).
    No clock is read — not here, and not in ``tracking`` on this path, which
    is why replaying a ledger reproduces the same weeks and the same delta
    rather than sliding as "this week" moves.

    The output is aggregate numbers only — means, counts, a delta. No block
    id, statement or tag leaves this function, so nothing here can surface
    corpus content that never passed admission.

    ``queries_scored: 0`` is the honest answer for a workspace with signals
    but no feedback labels, and it is reported rather than smoothed into a
    plausible-looking score. An unmeasured retrieval stack should say so.
    """
    from mind_mem.calibration import CalibrationManager, query_fingerprint
    from mind_mem.interaction_signals import SignalStore
    from mind_mem.tracking import mrr_events_from_signals, mrr_from_events

    signals = SignalStore(_signal_store_path(ws)).all_signals()[-_MRR_SIGNAL_SCAN_CAP:]
    fingerprints: set[str] = set()
    for sig in signals:
        for query in (sig.previous_query, sig.new_query):
            if query:
                fingerprints.add(query_fingerprint(query))
    labels = CalibrationManager(ws).accepted_ids_by_fingerprint(fingerprints)
    events = mrr_events_from_signals(signals, labels, fingerprint=query_fingerprint)
    out = mrr_from_events(events).as_dict()
    out["signals_scanned"] = len(signals)
    out["signals_unlabelled"] = len(signals) - len(events)
    return out


@mcp_tool_observe
def reindex(include_vectors: bool = False) -> str:
    """Trigger FTS index rebuild, optionally with vector indexing."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err
    results: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION, "fts": False, "vectors": False}

    try:
        from mind_mem.sqlite_index import build_index

        build_index(ws)
        results["fts"] = True
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as e:
        _log.warning("reindex_fts_failed", error=str(e))
        results["fts_error"] = "FTS index rebuild failed. Run: mind-mem-scan --reindex"

    if include_vectors:
        try:
            from mind_mem.recall_vector import rebuild_index

            rebuild_index(ws)
            results["vectors"] = True
        except ImportError:
            results["vectors_error"] = "sentence-transformers not installed"
        except (OSError, ValueError) as exc:
            _log.warning("reindex_vectors_failed", error=str(exc))
            results["vectors_error"] = "Vector index rebuild failed"

    try:
        from mind_mem.category_distiller import CategoryDistiller

        extra_cats = _load_extra_categories(ws)
        distiller = CategoryDistiller(extra_categories=extra_cats if extra_cats else None)
        written = distiller.distill(ws)
        results["categories"] = len(written)
    except ImportError:
        _log.debug("reindex_category_distiller_unavailable")
    except (OSError, ValueError) as exc:
        _log.warning("reindex_categories_failed", error=str(exc))
        results["categories_error"] = "Category distillation failed"

    metrics.inc("mcp_reindex")
    _log.info("mcp_reindex", results=results)
    return json.dumps(results, indent=2)


@mcp_tool_observe
def delete_memory_item(block_id: str) -> str:
    """Delete a block by ID from the workspace's blocks of record (admin-scope).

    Backend-aware: the Markdown / encrypted corpus is edited in place
    (line-splice + atomic replace, with a recovery journal); every other
    backend deletes through its own block store.
    """
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9-]+$", block_id):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid block ID format: {block_id}",
            }
        )

    # Audit bug #5: on a non-Markdown backend (e.g. Postgres) the block of
    # record lives in the store; the local corpus files are empty init
    # templates. The Markdown line-splicing below would then report
    # "Source file not found" / "Block not found in DECISIONS.md" while the
    # block stayed in the store — a delete that reports the ID as wrong and
    # removes nothing. Route through the store, which owns its own deletion
    # journal (``deleted_blocks``).
    if not _is_markdown_backend(ws):
        backend = _backend_name(ws)
        try:
            removed = get_block_store(ws).delete_block(block_id)
        except Exception as exc:
            _log.warning("mcp_delete_memory_item_store_failed", block_id=block_id, error=str(exc))
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"Delete failed on the {backend} block store: {exc}",
                    "block_id": block_id,
                }
            )
        if not removed:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"Block {block_id} not found in the block store.",
                    "block_id": block_id,
                }
            )
        metrics.inc("mcp_delete_memory_item")
        _log.info("mcp_delete_memory_item", block_id=block_id, backend=backend)
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "deleted",
                "block_id": block_id,
                "backend": backend,
            },
            indent=2,
        )

    filepath = _find_block_file(ws, block_id)
    if filepath is None:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Unrecognized block ID prefix: {block_id}",
                "hint": "Supported prefixes: " + ", ".join(sorted(_BLOCK_PREFIX_MAP.keys())),
            }
        )

    if not os.path.isfile(filepath):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Source file not found: {filepath}",
                "block_id": block_id,
            }
        )

    with FileLock(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        block_start: int | None = None
        block_end: int | None = None
        block_header = f"[{block_id}]"

        for i, line in enumerate(lines):
            if line.strip() == block_header:
                block_start = i
            elif block_start is not None and block_end is None:
                if line.startswith("[") and line.strip().endswith("]") and _re_mod.match(r"^\[[A-Z]+-", line.strip()):
                    block_end = i
                # ``startswith``, not ``strip() == ``: block_parser's own
                # separator rule, and the one block_store._locate_block_in_text
                # uses. An INDENTED ``---`` is a value escaped by
                # block_store._neutralise_value, not a boundary.
                elif line.startswith("---"):
                    preceding_blank = (i == 0) or (lines[i - 1].strip() == "")
                    if preceding_blank:
                        block_end = i + 1

        if block_start is None:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "error": f"Block {block_id} not found in {os.path.basename(filepath)}",
                    "block_id": block_id,
                }
            )

        if block_end is None:
            block_end = len(lines)

        deleted_content = "\n".join(lines[block_start:block_end])
        new_lines = lines[:block_start] + lines[block_end:]
        new_content = "\n".join(new_lines)

        from datetime import datetime, timezone

        deleted_log = os.path.join(ws, "memory", "deleted_blocks.jsonl")
        os.makedirs(os.path.dirname(deleted_log), exist_ok=True)
        # Audit S-11: concurrent delete_memory_item invocations were
        # interleaving append() bytes from two records, producing
        # invalid JSONL that broke recovery. Use the project's
        # FileLock (BSD flock / msvcrt) plus an O_APPEND open so the
        # kernel-level append is atomic per write() syscall AND
        # serialized across processes. The combination is needed
        # because POSIX only guarantees atomicity for writes <
        # PIPE_BUF on O_APPEND, and deleted content can be >4K.
        entry = {
            "block_id": block_id,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "content": deleted_content,
        }
        line = json.dumps(entry, default=str) + "\n"
        with FileLock(deleted_log + ".lock"):
            with open(deleted_log, "a", encoding="utf-8") as dl:
                dl.write(line)
                dl.flush()
                os.fsync(dl.fileno())

        dir_name = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".md.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(new_content)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    metrics.inc("mcp_delete_memory_item")
    _log.info("mcp_delete_memory_item", block_id=block_id, file=os.path.basename(filepath))

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "deleted",
            "block_id": block_id,
            "file": os.path.basename(filepath),
            "lines_removed": block_end - block_start,
        },
        indent=2,
    )


@mcp_tool_observe
def export_memory(format: str = "jsonl", include_metadata: bool = False, max_blocks: int = 10000) -> str:
    """Export all workspace blocks as JSONL."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    if format != "jsonl":
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Unsupported format: {format}. Use 'jsonl'.",
            }
        )

    all_blocks: list[dict] = []

    def _strip_internal(block: dict) -> dict:
        """Drop underscore-prefixed metadata unless ``include_metadata``."""
        if include_metadata:
            return block
        for key in list(block.keys()):
            if key.startswith("_") and key not in ("_id", "_source_file"):
                del block[key]
        return block

    if _is_markdown_backend(ws):
        # Default Markdown / encrypted path — byte-for-byte unchanged.
        for subdir in CORPUS_DIRS:
            dir_path = os.path.join(ws, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fn in sorted(os.listdir(dir_path)):
                if not fn.endswith(".md"):
                    continue
                filepath = os.path.join(dir_path, fn)
                try:
                    blocks = parse_file(filepath)
                except (OSError, ValueError) as exc:
                    _log.warning("export_parse_failed", file=fn, error=str(exc))
                    continue
                for block in blocks:
                    block["_source_file"] = f"{subdir}/{fn}"
                    all_blocks.append(_strip_internal(block))
    else:
        # Audit bug #5: a non-Markdown backend (e.g. Postgres) keeps its
        # blocks of record in the store, not in local Markdown files.
        # Export every block (active + inactive, matching the markdown
        # path which exports all parsed blocks) from the configured store.
        store = get_block_store(ws)
        for block in store.get_all(active_only=False):
            all_blocks.append(_strip_internal(block))

    truncated = False
    total = len(all_blocks)
    if len(all_blocks) > max_blocks:
        all_blocks = all_blocks[:max_blocks]
        truncated = True
        _log.warning("export_memory_truncated", total=total, max_blocks=max_blocks)

    jsonl_lines = [json.dumps(b, default=str) for b in all_blocks]
    jsonl_output = "\n".join(jsonl_lines)

    metrics.inc("mcp_export_memory")
    _log.info("mcp_export_memory", format=format, blocks=len(all_blocks))

    envelope: dict[str, Any] = {
        "_schema_version": MCP_SCHEMA_VERSION,
        "format": format,
        "block_count": len(all_blocks),
        "data": jsonl_output,
    }
    if truncated:
        envelope["warning"] = f"Output truncated to {max_blocks} blocks (total: {total}). Increase max_blocks to export more."

    return json.dumps(envelope, indent=2)


@mcp_tool_observe
def get_block(block_id: str) -> str:
    """Retrieve a single block by its ID with full content."""
    if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", block_id):
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Invalid block_id format: {block_id}",
            }
        )

    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    def _found(block: dict) -> str:
        metrics.inc("mcp_get_block")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "block_id": block_id,
                "found": True,
                "block": block,
            },
            indent=2,
            default=str,
        )

    # Audit bug #5: a non-Markdown backend (e.g. Postgres) keeps the block
    # of record in the store, not in local Markdown files, so the
    # corpus-file resolution below would never find it. Query the store
    # directly by primary key.
    if not _is_markdown_backend(ws):
        store = get_block_store(ws)
        block = store.get_by_id(block_id)
        if block is not None:
            return _found(block)
        metrics.inc("mcp_get_block_miss")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "block_id": block_id,
                "found": False,
                "error": f"Block {block_id} not found in the block store.",
                "hint": "Check the block ID and ensure the workspace is initialized.",
            },
            indent=2,
        )

    filepath = _find_block_file(ws, block_id)
    if filepath and os.path.isfile(filepath):
        try:
            blocks = parse_file(filepath)
            for block in blocks:
                if block.get("_id") == block_id:
                    rel_path = os.path.relpath(filepath, ws)
                    block["_source_file"] = rel_path.replace(os.sep, "/")
                    return _found(block)
        except (OSError, ValueError, BlockCorruptedError) as exc:
            _log.debug("get_block_parse_failed", file=filepath, error=str(exc))

    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(ws, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if not fn.endswith(".md"):
                continue
            fpath = os.path.join(dir_path, fn)
            if fpath == filepath:
                continue
            try:
                blocks = parse_file(fpath)
                for block in blocks:
                    if block.get("_id") == block_id:
                        block["_source_file"] = f"{subdir}/{fn}"
                        return _found(block)
            except (OSError, ValueError, BlockCorruptedError):
                continue

    metrics.inc("mcp_get_block_miss")
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "block_id": block_id,
            "found": False,
            "error": f"Block {block_id} not found in any corpus file.",
            "hint": "Check the block ID and ensure the workspace is initialized.",
        },
        indent=2,
    )


@mcp_tool_observe
def memory_health() -> str:
    """Deep health dashboard for the memory workspace."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    health: dict[str, Any] = {"_schema_version": MCP_SCHEMA_VERSION}
    recommendations: list[str] = []

    corpus_stats: dict[str, dict[str, int]] = {}
    total_blocks = 0
    total_active = 0
    if _is_markdown_backend(ws):
        for subdir in CORPUS_DIRS:
            dir_path = os.path.join(ws, subdir)
            if not os.path.isdir(dir_path):
                corpus_stats[subdir] = {"total": 0, "active": 0}
                continue
            sub_total = 0
            sub_active = 0
            for fn in os.listdir(dir_path):
                if not fn.endswith(".md") or fn.endswith("_ARCHIVE.md"):
                    continue
                try:
                    blocks = parse_file(os.path.join(dir_path, fn))
                    sub_total += len(blocks)
                    sub_active += len(get_active(blocks))
                except (OSError, ValueError):
                    pass
            corpus_stats[subdir] = {"total": sub_total, "active": sub_active}
            total_blocks += sub_total
            total_active += sub_active
    else:
        # Audit bug #5: derive corpus stats from the configured block
        # store (e.g. Postgres) instead of the empty Markdown templates.
        # Bucket each block under its source subdir (parsed from the
        # store's ``_source_file``) so the per-corpus breakdown is
        # preserved; blocks with an unknown source land in ``other``.
        corpus_stats = {subdir: {"total": 0, "active": 0} for subdir in CORPUS_DIRS}
        try:
            store_blocks = get_block_store(ws).get_all(active_only=False)
        except Exception as exc:  # pragma: no cover - defensive: store probe
            _log.debug("memory_health_store_count_failed", error=str(exc))
            store_blocks = []
        for block in store_blocks:
            src = str(block.get("_source_file", "") or "")
            # The Markdown branch skips ``*_ARCHIVE.md`` outright, so an
            # archived block counts in neither total nor active there.
            # Skip the store's equivalents too, or the two branches count
            # different populations under the same field name.
            if src.rsplit("/", 1)[-1].endswith("_ARCHIVE.md"):
                continue
            subdir = src.split("/", 1)[0] if "/" in src else ""
            bucket = subdir if subdir in corpus_stats else "other"
            stat = corpus_stats.setdefault(bucket, {"total": 0, "active": 0})
            stat["total"] += 1
            is_active = _store_block_is_active(block)
            if is_active:
                stat["active"] += 1
            total_blocks += 1
            if is_active:
                total_active += 1
    health["corpus"] = corpus_stats
    health["total_blocks"] = total_blocks
    health["total_active"] = total_active

    stale_count = 0
    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)
        stale = cg.get_stale_blocks()
        stale_count = len(stale)
        health["stale_blocks"] = stale_count
        if stale_count > 0:
            health["stale_block_ids"] = [s["block_id"] for s in stale[:10]]
            recommendations.append(
                f"{stale_count} stale block(s) need review. Use stale_blocks tool for details, then update or clear staleness."
            )
    except (ImportError, sqlite3.OperationalError, OSError, ValueError) as exc:
        health["stale_blocks"] = 0
        _log.debug("health_stale_check_skipped", error=str(exc))

    drift_path = os.path.join(ws, "intelligence", "DRIFT.md")
    drift_count = 0
    if os.path.isfile(drift_path):
        try:
            drift_count = len(parse_file(drift_path))
        except (OSError, ValueError):
            pass
    health["drift_items"] = drift_count
    if drift_count > 0:
        recommendations.append(f"{drift_count} drift item(s) detected. Review intelligence/DRIFT.md for belief shifts.")

    import struct as _struct_mod

    try:
        from mind_mem import recall_vector as _rv

        vec_path = _rv._index_path(ws)  # type: ignore[attr-defined]
        if os.path.isfile(vec_path):
            with open(vec_path, "rb") as f:
                header = f.read(8)
                if len(header) >= 4:
                    embedded_count = _struct_mod.unpack("<I", header[:4])[0]
                    health["embedded_blocks"] = embedded_count
                    if total_blocks > 0:
                        coverage = round(embedded_count / total_blocks * 100, 1)
                        health["embedding_coverage_pct"] = coverage
                        if coverage < 80:
                            recommendations.append(f"Embedding coverage is {coverage}%. Run reindex(include_vectors=True).")
                else:
                    health["embedded_blocks"] = 0
                    health["embedding_coverage_pct"] = 0.0
        else:
            health["embedded_blocks"] = 0
            health["embedding_coverage_pct"] = 0.0
            if total_blocks > 10:
                recommendations.append("No vector index found. Run reindex(include_vectors=True) for hybrid search.")
    except (ImportError, AttributeError, OSError, _struct_mod.error) as exc:
        # AttributeError: the vector-index path accessor is optional and
        # version-dependent; degrade the embedding-coverage probe to
        # "unknown" rather than crashing the whole health dashboard.
        health["embedded_blocks"] = "unknown"
        health["embedding_coverage_pct"] = "unknown"
        _log.debug("health_embedding_probe_skipped", error=str(exc))

    signals_path = os.path.join(ws, "intelligence", "SIGNALS.md")
    pending_signals = 0
    if os.path.isfile(signals_path):
        try:
            sigs = parse_file(signals_path)
            pending_signals = len([s for s in sigs if s.get("Status", "pending") == "pending"])
        except (OSError, ValueError):
            pass
    health["pending_signals"] = pending_signals
    if pending_signals > 5:
        recommendations.append(f"{pending_signals} pending signals. Review and apply or reject them.")

    contra_path = os.path.join(ws, "intelligence", "CONTRADICTIONS.md")
    contra_count = 0
    if os.path.isfile(contra_path):
        try:
            contra_count = len(parse_file(contra_path))
        except (OSError, ValueError):
            pass
    health["unresolved_contradictions"] = contra_count
    if contra_count > 0:
        recommendations.append(f"{contra_count} unresolved contradiction(s). Use list_contradictions for details.")

    db = fts_db_path(ws)
    if db and os.path.isfile(db):
        try:
            from mind_mem.sqlite_index import index_status as fts_status

            info = fts_status(ws)
            health["fts_index"] = {
                "exists": True,
                "blocks_indexed": info.get("blocks", 0),
                "stale_files": info.get("stale_files", 0),
                "last_build": info.get("last_build"),
                "db_size_bytes": info.get("db_size_bytes", 0),
            }
            stale_files = info.get("stale_files", 0)
            if stale_files > 0:
                recommendations.append(f"FTS index has {stale_files} stale file(s). Run reindex tool.")
        except (sqlite3.OperationalError, OSError, ValueError):
            health["fts_index"] = {"exists": True, "error": "Could not read index status"}
    else:
        health["fts_index"] = {"exists": False}
        recommendations.append("No FTS index. Run reindex tool for fast keyword search.")

    try:
        from mind_mem.compaction import archive_completed_blocks, compact_signals

        archivable = archive_completed_blocks(ws, days=90, dry_run=True)
        compactable_signals = compact_signals(ws, days=60, dry_run=True)
        health["compaction"] = {
            "archivable_blocks": len(archivable),
            "compactable_signals": len(compactable_signals),
        }
        total_compactable = len(archivable) + len(compactable_signals)
        if total_compactable > 0:
            recommendations.append(f"{total_compactable} item(s) ready for compaction. Run compact tool.")
    except (ImportError, OSError, ValueError) as exc:
        health["compaction"] = {"error": str(exc)}

    # Model Reliability Score (flag-gated, default OFF — see
    # _mrs_health_section). Deliberately last: it is the only section
    # that can fire an alert, so everything the dashboard reports has
    # already been gathered by the time it runs.
    try:
        mrs_section = _mrs_health_section(ws)
    except (ImportError, sqlite3.OperationalError, OSError, ValueError) as exc:
        mrs_section = None
        _log.debug("health_mrs_skipped", error=str(exc))
    if mrs_section is not None:
        health["mrs"] = mrs_section
        if mrs_section["violations"]:
            recommendations.append(f"MRS {mrs_section['score']}/100 — SLI breach: {', '.join(mrs_section['violations'])}.")

    # v4/health probe sweep (flag-gated, default OFF — see _v4_health_section).
    # After mrs so the alert-firing section still runs against a complete
    # dashboard, and because a degraded v4 surface is a recommendation, not a
    # reason to withhold the numbers already gathered.
    v4_health = _v4_health_section(ws)
    if v4_health is not None:
        health["v4"] = v4_health
        if v4_health.get("status") != "ok":
            broken = sorted(name for name, st in v4_health.get("modules", {}).items() if st != "ok" and st != "disabled")
            if broken:
                recommendations.append(f"v4 health {v4_health.get('status')}: {', '.join(broken)}.")

    health["recommendations"] = recommendations
    health["score"] = "healthy" if not recommendations else "needs_attention"

    metrics.inc("mcp_memory_health")
    _log.info("mcp_memory_health", total_blocks=total_blocks, recommendations=len(recommendations))
    return json.dumps(health, indent=2, default=str)


@mcp_tool_observe
def compact(dry_run: bool = True, archive_days: int = 90, signal_days: int = 60, snapshot_days: int = 30) -> str:
    """Run workspace compaction — archive old blocks, clean snapshots, remove resolved signals."""
    ws = _workspace()
    ws_err = _check_workspace(ws)
    if ws_err:
        return ws_err

    from mind_mem.compaction import (
        archive_completed_blocks,
        cleanup_daily_logs,
        cleanup_snapshots,
        compact_signals,
    )

    actions: dict[str, list[str]] = {}

    try:
        block_actions = archive_completed_blocks(ws, days=archive_days, dry_run=dry_run)
        actions["archived_blocks"] = block_actions
    except (OSError, ValueError) as exc:
        actions["archived_blocks_error"] = [str(exc)]
        _log.warning("compact_archive_failed", error=str(exc))

    try:
        snap_actions = cleanup_snapshots(ws, days=snapshot_days, dry_run=dry_run)
        actions["cleaned_snapshots"] = snap_actions
    except (OSError, ValueError) as exc:
        actions["cleaned_snapshots_error"] = [str(exc)]
        _log.warning("compact_snapshots_failed", error=str(exc))

    try:
        signal_actions = compact_signals(ws, days=signal_days, dry_run=dry_run)
        actions["compacted_signals"] = signal_actions
    except (OSError, ValueError) as exc:
        actions["compacted_signals_error"] = [str(exc)]
        _log.warning("compact_signals_failed", error=str(exc))

    try:
        log_actions = cleanup_daily_logs(ws, days=180, dry_run=dry_run)
        actions["archived_logs"] = log_actions
    except (OSError, ValueError) as exc:
        actions["archived_logs_error"] = [str(exc)]
        _log.warning("compact_logs_failed", error=str(exc))

    total_actions = sum(len(v) for v in actions.values() if isinstance(v, list))

    metrics.inc("mcp_compact")
    _log.info("mcp_compact", dry_run=dry_run, total_actions=total_actions)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "dry_run" if dry_run else "executed",
            "dry_run": dry_run,
            "total_actions": total_actions,
            "actions": actions,
            "next_step": (
                "Call again with dry_run=False to execute."
                if dry_run and total_actions > 0
                else "Workspace is clean — nothing to compact."
                if total_actions == 0
                else None
            ),
        },
        indent=2,
    )


@mcp_tool_observe
def stale_blocks(limit: int = 20, clear_block_id: str = "") -> str:
    """List blocks flagged as stale, or clear a staleness flag."""
    ws = _workspace()

    try:
        from mind_mem.causal_graph import CausalGraph

        cg = CausalGraph(ws)

        if clear_block_id:
            if not _re_mod.match(r"^[A-Z]+-[a-zA-Z0-9_.-]+$", clear_block_id):
                return json.dumps(
                    {
                        "_schema_version": MCP_SCHEMA_VERSION,
                        "error": f"Invalid block_id format: {clear_block_id}",
                    }
                )
            cleared = cg.clear_staleness(clear_block_id)
            metrics.inc("mcp_stale_cleared")
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "action": "cleared",
                    "block_id": clear_block_id,
                    "was_stale": cleared,
                },
                indent=2,
            )

        stale = cg.get_stale_blocks()
        stale = stale[: max(1, min(limit, 100))]

        metrics.inc("mcp_stale_blocks")
        _log.info("mcp_stale_blocks", count=len(stale))

        if not stale:
            return json.dumps(
                {
                    "_schema_version": MCP_SCHEMA_VERSION,
                    "status": "clean",
                    "stale_count": 0,
                    "message": "No stale blocks. All blocks are up to date.",
                },
                indent=2,
            )

        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "stale_found",
                "stale_count": len(stale),
                "blocks": stale,
                "hint": "Review each stale block and update or call stale_blocks(clear_block_id='...') to clear.",
            },
            indent=2,
            default=str,
        )

    except ImportError:
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "causal_graph module not available",
            },
            indent=2,
        )
    except sqlite3.OperationalError as exc:
        if _is_db_locked(exc):
            return _sqlite_busy_error()
        raise
    except (OSError, ValueError) as exc:
        _log.warning("stale_blocks_failed", error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": f"Stale block lookup failed: {exc}",
            },
            indent=2,
        )


def register(mcp) -> None:
    """Wire the memory_ops tools onto *mcp*."""
    mcp.tool(index_stats)
    mcp.tool(reindex)
    mcp.tool(delete_memory_item)
    mcp.tool(export_memory)
    mcp.tool(get_block)
    mcp.tool(memory_health)
    mcp.tool(compact)
    mcp.tool(stale_blocks)
