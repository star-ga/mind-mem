"""Memory operations MCP tools — index / lifecycle / health / export.

Extracted from ``mcp_server.py`` per docs/v3.2.0-mcp-decomposition-plan.md
(PR-3 slice, memory_ops domain). Eight tools cover the workspace's
lifecycle + introspection surface:

* ``index_stats`` / ``reindex`` — FTS5 index state + rebuild.
* ``delete_memory_item`` — governed admin-scope block removal: one
  ``admit_delete`` scope, one authorisation record, one removal record,
  plus the append-only local recovery log.
* ``export_memory`` — JSONL dump of every block with configurable
  metadata + size cap.
* ``get_block`` / ``memory_health`` / ``compact`` / ``stale_blocks`` —
  block lookup, health dashboard, compaction, and staleness-flag
  management built on the causal graph.

Also hosts the ``_BLOCK_PREFIX_MAP`` + ``_find_block_file``
resolver shared by ``delete_memory_item`` and ``get_block``.
"""

from __future__ import annotations

import getpass
import json
import os
import re as _re_mod
import sqlite3
import tempfile
from typing import Any

from mind_mem.admission import AdmissionReceipt, admit_read, admit_read_one
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
    # 5.0.2 (GAP-1): drift findings became writable through the store. Keep
    # in lockstep with ``block_store._BLOCK_PREFIX_MAP``.
    "DREF": ("intelligence", "DRIFT.md"),
    # 5.0.2 (GAP-2): captured signals became writable through the store. Keep
    # in lockstep with ``block_store._BLOCK_PREFIX_MAP``.
    "SIG": ("intelligence", "SIGNALS.md"),
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


#: Rationale recorded for a ``delete_memory_item`` call that carried none.
#:
#: An MCP tool call arrives from an agent, not from a form: it carries no
#: reason of its own unless the caller passes one, and a plausible
#: sentence invented here would put a justification in the audit chain
#: that nobody ever gave. Naming the door is the honest floor — the same
#: decision, for the same reason, as
#: :data:`mind_mem.http_transport.DEFAULT_DELETE_RATIONALE`. A caller that
#: does have a reason passes ``rationale`` and it reaches the record
#: verbatim.
DEFAULT_DELETE_RATIONALE = "mcp-delete-memory-item"

#: Prefix of the actor recorded for anything this MCP server destroys.
#: The suffix is the door's name, never its ACL scope: "admin" is a
#: permission this call held, not somebody who held it.
MCP_STDIO_ACTOR_PREFIX = "mcp-stdio:"

#: Operator-set name for the client on the other end of the stdio pipe
#: (``"claude-code"``, ``"the-nightly-compactor"``). A stdio transport has
#: no credential to derive an identity from — the pipe is the trust
#: boundary — so the operator who wired the client is the one who can
#: name it. Unset falls back to the OS account the server runs as, which
#: is a real, checkable identity rather than a placeholder.
MCP_CLIENT_ENV = "MIND_MEM_MCP_CLIENT"

#: Used when the process has no resolvable OS account (no ``LOGNAME`` /
#: ``USER`` / ``USERNAME`` and no passwd entry — a distroless container
#: running under a bare numeric uid). Still prefixed, so it still names
#: the door even when it cannot name the operator.
MCP_UNRESOLVED_USER = "unresolved-user"


def _mcp_door_actor() -> str:
    """Identity recorded for a delete arriving over MCP stdio.

    Measured on 5.0.1: this tool passed ``actor=""``, the gate resolved
    that through the REST contextvar, and every MCP delete landed in the
    chain as ``actor="anonymous"`` — on the most-used delete door in the
    product. There is no credential on a stdio pipe to hash the way the
    HTTP door hashes its bearer token, so the identity is the door's
    configured name, and the account it runs as when nobody configured
    one. Both are concrete; neither can be empty, because the prefix
    cannot be.
    """
    configured = os.environ.get(MCP_CLIENT_ENV, "").strip()
    if not configured:
        try:
            configured = getpass.getuser().strip()
        except Exception:  # pragma: no cover - no passwd entry and no env
            configured = ""
    # ``isprintable`` already excludes CR/LF/NUL, which is what a log or a
    # JSONL row needs kept out of a free-text name.
    name = "".join(ch for ch in configured if ch.isprintable())[:64].strip()
    return f"{MCP_STDIO_ACTOR_PREFIX}{name or MCP_UNRESOLVED_USER}"


def _delete_from_store(ws: str, block_id: str, backend: str, receipt: AdmissionReceipt) -> str:
    """Remove *block_id* through the configured block store.

    Runs inside the caller's open ``admit_delete`` scope. The store owns
    both halves of the delete contract —
    :func:`~mind_mem.admission.require_delete_admission` as its first
    statement and
    :meth:`~mind_mem.admission.AdmissionReceipt.record_removal` on
    success — so this leg neither checks admission nor records the
    removal itself; doing either here would be a second implementation of
    the seam.

    Audit bug #5: on a non-Markdown backend (e.g. Postgres) the block of
    record lives in the store; the local corpus files are empty init
    templates. Markdown line-splicing would then report "Source file not
    found" while the block stayed in the store — a delete that reports
    the ID as wrong and removes nothing.
    """
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
    _log.info("mcp_delete_memory_item", block_id=block_id, backend=backend, admission=receipt.entry_id)
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "deleted",
            "block_id": block_id,
            "backend": backend,
            "admission": receipt.entry_id,
        },
        indent=2,
    )


def _delete_from_corpus(ws: str, block_id: str, filepath: str, receipt: AdmissionReceipt) -> str:
    """Splice *block_id* out of the Markdown corpus of record.

    Runs inside the caller's open ``admit_delete`` scope. Unlike the store
    leg this one edits the file itself, so it owes the *second* half of
    the delete contract directly: once the atomic replace lands, the
    block is gone and
    :meth:`~mind_mem.admission.AdmissionReceipt.record_removal` reports
    what died, at the same point in the sequence
    ``MarkdownBlockStore.delete_block`` reports it — after the replace,
    outside the file lock. Without that call the scope would close with an
    empty ledger and the chain would carry an authorisation for a death it
    never recorded.
    """
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

    # The block is gone from disk. Report it under the scope that
    # authorised it: the gate turns this ledger into the one chain record
    # naming what died when the scope closes. The local journal above is
    # recovery; this is the audit fact.
    receipt.record_removal(block_id, deleted_content)

    metrics.inc("mcp_delete_memory_item")
    _log.info("mcp_delete_memory_item", block_id=block_id, file=os.path.basename(filepath), admission=receipt.entry_id)

    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "deleted",
            "block_id": block_id,
            "file": os.path.basename(filepath),
            "lines_removed": block_end - block_start,
            "admission": receipt.entry_id,
        },
        indent=2,
    )


@mcp_tool_observe
def delete_memory_item(block_id: str, rationale: str = "") -> str:
    """Delete a block by ID from the workspace's blocks of record (admin-scope).

    **One governed death, one chain record** — the third and last door
    that reached a delete without one. ``DELETE /memories/{id}`` and
    ``POST /clear`` were closed in 5.0.2; this tool was not, and the two
    legs failed differently. Measured by live probe on a workspace built
    by ``mind-mem-init`` (backend ``markdown``, the zero-config default):
    the Markdown leg never called ``delete_block`` at all, so the
    store-side gate could not see it — it returned ``{"status":
    "deleted"}``, the block left the corpus, and the evidence chain held
    **zero** rows. The store leg, by contrast, did call the now-gated
    ``delete_block``; its ``UngatedDeleteError`` came back to the caller
    as ``{"error": "Delete failed on the <backend> block store: ungated
    delete of ..."}`` and the block survived — fail-closed, which is the
    right direction, but not a working delete surface.

    Both legs now run inside one
    :meth:`~mind_mem.governance_gate.GovernanceGate.admit_delete` scope —
    a single call site, the same shape ``_handle_delete_memory`` uses —
    so the authorisation record is written before anything is touched and
    the removal record when something actually goes. The store leg
    inherits ``record_removal`` from the store; the Markdown leg reports
    its own removal, because it is the code that does the removing.

    A refused scope is reported as a refusal, never as a deletion: the
    block is still there, and telling a caller their content is gone when
    it is not is the one answer a memory product must never give.

    Backend-aware: the Markdown / encrypted corpus is edited in place
    (line-splice + atomic replace, with a recovery journal); every other
    backend deletes through its own block store.

    Every record this tool mints is attributed to
    :func:`_mcp_door_actor` — ``mcp-stdio:<name>``, from
    :data:`MCP_CLIENT_ENV` or the account the server runs as. It used to
    pass ``actor=""``, which the gate resolves through a REST contextvar
    an MCP process never sets, so the chain said ``anonymous``.

    Args:
        block_id: The block to remove.
        rationale: Why it is being removed. Optional, and empty records
            :data:`DEFAULT_DELETE_RATIONALE` — the door's own name rather
            than a reason nobody gave. See that constant.
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

    markdown = _is_markdown_backend(ws)
    backend = _backend_name(ws)

    # Routing, not existence: both refusals below are decided by the
    # block id's prefix and by whether the corpus file exists, neither of
    # which says whether the block is there. They stay outside the scope
    # so an id this door cannot route mints no authorisation record.
    filepath: str | None = None
    if markdown:
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

    # Imported here, as ``http_transport._handle_delete_memory`` does:
    # the governance layer is not a dependency of importing the MCP tool
    # module, only of running a delete.
    from mind_mem.governance_gate import GovernanceBypassError, get_gate

    try:
        with get_gate(ws).admit_delete(
            block_id,
            rationale=rationale.strip() or DEFAULT_DELETE_RATIONALE,
            # A concrete door identity, never "". Empty used to be passed
            # here on the reasoning that a stdio call has no authenticated
            # identity — true, and it does not follow that the record
            # should name nobody: the gate resolved "" through the REST
            # contextvar this process never sets, so every MCP delete was
            # attributed to ``anonymous``. The ACL scope ("admin") is
            # still not the answer — that is a permission, not a name —
            # so :func:`_mcp_door_actor` names the door and the account
            # it runs as.
            actor=_mcp_door_actor(),
            target_file=os.path.relpath(filepath, ws) if filepath else "",
            metadata={"door": "mcp.delete_memory_item", "backend": backend},
        ) as receipt:
            if markdown:
                # ``filepath`` is not None on this branch — the routing
                # checks above return before reaching here.
                return _delete_from_corpus(ws, block_id, str(filepath), receipt)
            return _delete_from_store(ws, block_id, backend, receipt)
    except GovernanceBypassError as exc:
        # The block is still there. Reporting a refused authorisation as
        # a completed deletion is the fail-open direction, and the gate
        # exists to make that impossible.
        _log.error("mcp_delete_memory_item_refused", block_id=block_id, error=str(exc))
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "error": "Delete refused by governance.",
                "detail": str(exc),
                "block_id": block_id,
            }
        )


@mcp_tool_observe
def export_memory(format: str = "jsonl", include_metadata: bool = False, max_blocks: int = 10000) -> str:
    """Export every ADMITTED workspace block as JSONL.

    "Every block" used to mean every block: a quarantined inbox drop or an
    unreviewed agent message came out of here verbatim, so an export was a
    way around the admission gate that ``recall`` enforces. Export now runs
    the same egress decision as recall (``admission.admit_read``) and the
    envelope carries ``withheld_count`` so a short export is visibly short
    rather than silently incomplete.

    There is deliberately **no bypass parameter**. A full-fidelity copy of
    the corpus, withheld content included, is ``snapshot()`` — a backup
    surface with its own governance — not a tool any caller can point at
    the corpus. An admin reviewing quarantined content uses the governance
    tools, which read the store directly.

    Args:
        format: Only ``"jsonl"`` is supported.
        include_metadata: Keep underscore-prefixed internal fields.
        max_blocks: Size cap applied AFTER admission, so the two refusals
            (withheld vs truncated) stay distinguishable in the envelope.

    Returns:
        JSON envelope with ``block_count``, ``withheld_count`` and the
        JSONL payload.
    """
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

    # EGRESS GATE. Export used to hand out every parsed block verbatim,
    # quarantined ones included -- the same defect class as ``get_block``
    # below, one surface out. It runs BEFORE truncation so ``withheld_count``
    # counts what admission refused rather than what the size cap dropped;
    # the two are different refusals and an operator needs to tell them apart.
    #
    # No bypass parameter, deliberately: a full-fidelity copy of the corpus is
    # ``snapshot()``, which is a backup surface with its own governance, not a
    # tool that serves unadmitted content to whoever calls it.
    decision = admit_read(all_blocks, workspace=ws, surface="export_memory")
    all_blocks = decision.admitted
    withheld = decision.withheld
    if withheld:
        _log.info("export_memory_withheld", withheld=withheld)

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
        "withheld_count": withheld,
        "data": jsonl_output,
    }
    if truncated:
        envelope["warning"] = f"Output truncated to {max_blocks} blocks (total: {total}). Increase max_blocks to export more."

    return json.dumps(envelope, indent=2)


@mcp_tool_observe
def get_block(block_id: str) -> str:
    """Retrieve a single ADMITTED block by its ID, with full content.

    A block that exists but has not passed admission answers
    ``{"found": false, "withheld": true}`` -- no content, no status, no
    source file. That is a different answer from "not found", and both
    are refusals, but only one of them is true: telling the caller "not
    found" for a block it named would be a lie an operator cannot debug.

    Scope does not widen this. Quarantine is a property of the content,
    not of the caller's role, so an admin sees the same refusal here and
    reviews withheld content through the governance tools instead.
    """
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

    block, where = _resolve_block_for_read(ws, block_id)

    # EGRESS GATE. Resolution above says whether the bytes EXIST; this says
    # whether this caller may see them. ``get_block`` is a USER-scope tool and
    # it used to answer the first question only, so a quarantined inbox drop
    # came back verbatim to anyone who knew (or guessed) its id -- while
    # ``recall`` withheld the same block. One resolved block, one decision,
    # taken here rather than at each of the three resolution sites, because
    # three decisions is how the third one gets forgotten.
    decision = admit_read_one(block, workspace=ws, surface="get_block")
    admitted = decision.sole
    if admitted is not None:
        metrics.inc("mcp_get_block")
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "block_id": block_id,
                "found": True,
                "block": admitted,
            },
            indent=2,
            default=str,
        )

    if decision.withheld:
        # "withheld", not "not found". The caller supplied the id, so saying
        # the block exists tells it nothing it did not already have, and the
        # honest answer is the one an operator can act on. No content, no
        # status detail, no source file.
        metrics.inc("mcp_get_block_withheld")
        _log.info("mcp_get_block_withheld", block_id=block_id)
        return json.dumps(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "block_id": block_id,
                "found": False,
                "withheld": True,
                "error": f"Block {block_id} has not passed admission and is not servable.",
                "hint": "Content from an untrusted door stays withheld until a governed release admits it.",
            },
            indent=2,
        )

    metrics.inc("mcp_get_block_miss")
    missing_in = "the block store" if where == "store" else "any corpus file"
    return json.dumps(
        {
            "_schema_version": MCP_SCHEMA_VERSION,
            "block_id": block_id,
            "found": False,
            "error": f"Block {block_id} not found in {missing_in}.",
            "hint": "Check the block ID and ensure the workspace is initialized.",
        },
        indent=2,
    )


def _resolve_block_for_read(ws: str, block_id: str) -> tuple[dict | None, str]:
    """Find *block_id*'s block dict, and name where we looked for it.

    Resolution only -- no admission decision, no metrics, no envelope. The
    caller applies the egress gate to whatever comes back, which is what
    keeps the gate on ONE path instead of once per resolution branch.

    Returns:
        ``(block, where)`` with *where* in ``{"store", "corpus"}``. The
        not-found message differs between the two backends and a caller
        that could not tell them apart would report the wrong one.
    """
    # Audit bug #5: a non-Markdown backend (e.g. Postgres) keeps the block
    # of record in the store, not in local Markdown files, so the
    # corpus-file resolution below would never find it. Query the store
    # directly by primary key.
    if not _is_markdown_backend(ws):
        return get_block_store(ws).get_by_id(block_id), "store"

    filepath = _find_block_file(ws, block_id)
    if filepath and os.path.isfile(filepath):
        try:
            for block in parse_file(filepath):
                if block.get("_id") == block_id:
                    rel_path = os.path.relpath(filepath, ws)
                    block["_source_file"] = rel_path.replace(os.sep, "/")
                    return block, "corpus"
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
                for block in parse_file(fpath):
                    if block.get("_id") == block_id:
                        block["_source_file"] = f"{subdir}/{fn}"
                        return block, "corpus"
            except (OSError, ValueError, BlockCorruptedError):
                continue

    return None, "corpus"


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
