"""v4.0 feature-flag registry.

Every v4 surface is gated. Flags live in ``mind-mem.json`` under the
``v4`` key; absence of the key (or the specific sub-flag) means the
surface is OFF. v3.x deployments that never touch the config see no
change in behaviour.

Example ``mind-mem.json`` snippet to enable v4 block kinds + long-context
recall:

    {
        "version": "4.0.0-alpha.1",
        "v4": {
            "block_kinds": { "enabled": true },
            "long_context_recall": { "enabled": true, "max_tokens": 32000 }
        }
    }

The flags listed in :data:`ALL_V4_FLAGS` are the authoritative set; new
surfaces must register here when they're added.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

from ..observability import get_logger

_log = get_logger("v4_feature_flags")

#: Last ``(path, error)`` warned about, so a broken config is reported
#: loudly but once — every flag lookup re-reads the file, and a caller
#: that checks a flag per operation must not flood the log with the same
#: parse error. Cleared on a successful read, so a re-break is reported
#: again.
_last_config_warning: tuple[str, str] | None = None

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Authoritative list of v4 feature flags. Each maps to a sub-key under
#: ``mind-mem.json: v4: {...}``. Order roughly matches ROADMAP §A→§E.
ALL_V4_FLAGS: Final[tuple[str, ...]] = (
    # Group A — cognition / model layer
    "cognitive_kernel",
    "surprise_retrieval",
    # 5.1.0: per-provider / per-domain reliability EMA fed by report_outcome
    # and persisted to intelligence/llm_profiles.json; surfaced in
    # calibration_stats. Sidecar only — nothing on the scored path reads it.
    "llm_noise_profile",
    # Group B — knowledge graph
    "block_kinds",
    "long_context_recall",
    "fusion",
    "streaming_recall",
    "chat",
    "prompt_schema",
    # Group C — governance / UX
    "idle_ingest",
    "lint",
    "contradiction_states",
    "self_heal",
    "viewer",
    "contradiction_stream",
    "world_staleness",  # external-anchor liveness check surfaced through scan()
    "maintenance_layout",  # first-run maintenance/ tracked|append-only split at apply time
    "ingest_serve",  # `mm ingest-serve` webhook door + its governed drain consumer
    "bootstrap_corpus",  # one-shot post-init backfill CLI (mind-mem-bootstrap); every write quarantined
    # Group D — platform scale (selected v4-introduced items)
    "rust_hot_path",
    "embedding_fallback",
    "pq",  # product-quantization codec (audit 4/4 consensus 2026-05-10)
    "hnsw_kind_index",  # HNSW kind-filtered ANN (audit 3/4)
    "federation",  # cross-agent version vectors + conflict log (round 2 audit 4/4)
    "embedding_pipeline",  # auto-derive embeddings (round 2 audit 4/4)
    "kind_summaries",  # GraphRAG-style per-kind summaries (round 2 audit 3/4)
    "self_editing",  # MemGPT-style propose_edit / approve_edit (round 2 audit 2/4)
    "granularity_align",  # named merge operation surfaced in plan_consolidation (proposal-only)
    "multi_modal",  # sidecar-described image/audio inbox drops + modality-aware pack cost
    "observability",  # counters / gauges / histograms (round 3 audit 4/4)
    "logging_context",  # correlation-ID + kv context on every log line (round 4 audit)
    "backpressure",  # ingestion overload signal (round 4 audit, DeepSeek 9.75→10)
    "block_metadata",  # ChromaDB-style tags + Weaviate-style validators (round 4 audit)
    "circuit_breaker",  # external dependency CB (round 5 audit, Mistral + GLM 9.9→10)
    # Group E — compliance-sensitive opt-in
    "redaction",
    "time_bounded_recall",
    "vocabulary",
    "provenance",
    "evidence",
    "tenant_kms",
    "tenant_chains",
    "compliance_export",
    "contraindicates_edges",
    "typed_edges",  # first-class typed relation layer + proposal-gated writes (roadmap §a)
    "entity_observations",  # per-entity accreted-facts field on the entity registry (roadmap §b)
    "core_export",  # .mmcore static export (OKF / JSON-LD / markdown) + governed OKF re-import
    # 5.1.0: MIND kernels. Turns the hash-chain import door from a
    # per-entry check into a SEQUENCE check, so a v3->v1 entry-hash
    # downgrade inside an imported segment is refused at the door
    # instead of being written and only discovered by a later
    # verify_chain(). Default OFF; see hash_chain_v2.import_jsonl.
    "mind_kernels",
    # 5.1.0: trajectory memory. Capture half writes a TRAJ- sidecar per
    # report_outcome; recall half is the similar_trajectories MCP tool. The
    # sidecar is never served by recall() and never bypasses the HITL gate.
    "trajectory",
    # 5.1.0: pack_recall_budget sizes its budget to the TARGET MODEL's real
    # context window instead of to whatever number the caller typed.
    "context_budget",
    # 5.1.0: measured retrieval quality in index_stats — weekly MRR drift
    # from the signal ledger, and the packed-vs-referenced token ratio.
    "retrieval_metrics",
    # 5.1.0: online training. Harvest half is a dream_cycle pass draining the
    # interaction-signal ledger into admission-filtered training tuples;
    # registry half is model_gate's persisted weight-promotion ledger.
    "online_training",
    # 5.1.0: v4/health's probe sweep folded into the `memory_health` MCP tool.
    # The flag gates the CALL SITE, not the check: `health.health_check` is
    # deliberately never flag-gated (an operator debugging a failure needs it
    # regardless) and each of its probes reports "disabled" for its own OFF
    # feature. What the flag buys is that `memory_health`'s payload -- which a
    # test pins byte-for-byte -- is unchanged until an operator asks for the
    # section, and that no clock is read on the OFF path.
    "health",
)


class FeatureDisabledError(RuntimeError):
    """Raised when a v4 surface is invoked while its flag is OFF.

    Caller should either flip the flag in ``mind-mem.json`` or stay on
    the v3.x equivalent.
    """


# ---------------------------------------------------------------------------
# Config loading (read-only; we never write here)
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Resolve the active ``mind-mem.json``.

    Search order:
        1. ``$MIND_MEM_CONFIG`` env var (explicit override)
        2. ``$MIND_MEM_WORKSPACE/mind-mem.json`` (workspace-local)
        3. ``./mind-mem.json`` (cwd)
        4. ``~/.mind-mem/mind-mem.json`` (user-level)

    Returns the first path that exists; if nothing is found, returns
    the cwd default — which won't exist, so ``is_enabled`` falls back to
    "OFF" cleanly.
    """
    explicit = os.environ.get("MIND_MEM_CONFIG")
    if explicit:
        return Path(explicit)

    ws = os.environ.get("MIND_MEM_WORKSPACE")
    if ws:
        candidate = Path(ws) / "mind-mem.json"
        if candidate.is_file():
            return candidate

    cwd = Path.cwd() / "mind-mem.json"
    if cwd.is_file():
        return cwd

    user = Path.home() / ".mind-mem" / "mind-mem.json"
    return user


def _read_v4_block(*, quiet: bool = False) -> tuple[dict, str]:
    """Return ``(v4 block, error)`` for the active config.

    ``error`` is ``""`` when the config was read successfully (including
    the ordinary case of no config file at all) and a short description
    when it exists but could not be read or parsed. Callers need the
    difference: an unparseable config turns *every* v4 surface off at
    once, which is indistinguishable from every flag being unset unless
    the parse failure is carried out of here.

    ``quiet=True`` is for a PROBE — a caller asking "is this surface on?"
    whose answer, when it is *no*, must leave no trace. The loud path logs
    ``v4_config_unreadable`` on a malformed config, so a probe running on a
    default-OFF code path would emit a line the unwired build never emitted;
    that is a flag-off behaviour difference, and the whole restoration lands
    under "flag-off is byte-identical". A quiet read therefore neither logs
    nor touches ``_last_config_warning`` — it is a pure function of the file,
    indistinguishable from not having been called at all, so it can never
    swallow (or provoke) a warning the loud path owes its caller.
    """
    global _last_config_warning
    p = _config_path()
    if not p.is_file():
        if not quiet:
            _last_config_warning = None
        return {}, ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        error = f"{p}: {exc}"
        if not quiet and _last_config_warning != (str(p), str(exc)):
            _last_config_warning = (str(p), str(exc))
            _log.warning("v4_config_unreadable", path=str(p), error=str(exc))
        return {}, error
    if not quiet:
        _last_config_warning = None
    block = data.get("v4") if isinstance(data, dict) else None
    return (block if isinstance(block, dict) else {}), ""


def _load_v4_block(*, quiet: bool = False) -> dict:
    """Return the ``v4`` block from active config, or ``{}`` if absent."""
    return _read_v4_block(quiet=quiet)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


#: Parsed ``v4`` blocks, keyed by (path, mtime_ns, size).
#:
#: ``is_enabled_quiet`` is a PROBE: an off-by-default surface calls it to
#: decide whether to run at all, so it lands on hot paths — ``publish()``
#: called it once per event. Uncached, every one of those calls re-read AND
#: re-parsed ``mind-mem.json`` (measured: 1000 config reads per 1000 flag-OFF
#: publishes, 2.5x slower than a build with no config file at all). That is
#: the flag-off inertness rule broken: a default-OFF deployment must not pay
#: a cost the unwired build never paid.
#:
#: Keying on (mtime_ns, size) rather than time means an edited config still
#: takes effect on the next call — the cache changes the cost, never the
#: answer. A same-nanosecond same-size edit is not distinguishable, which is
#: why this is only used by the silent probe; ``is_enabled`` still reads live.
_QUIET_CACHE: dict[tuple[str, int, int], dict] = {}
_QUIET_CACHE_MAX = 32


def _quiet_block(path: Path, mtime_ns: int, size: int) -> dict:
    """Return the parsed ``v4`` block for *path*, cached on its stat."""
    key = (str(path), mtime_ns, size)
    hit = _QUIET_CACHE.get(key)
    if hit is not None:
        return hit
    block = json.loads(path.read_text(encoding="utf-8")).get("v4") or {}
    if not isinstance(block, dict):
        block = {}
    if len(_QUIET_CACHE) >= _QUIET_CACHE_MAX:
        _QUIET_CACHE.clear()
    _QUIET_CACHE[key] = block
    return block


def is_enabled_quiet(flag: str) -> bool:
    """:func:`is_enabled`, but guaranteed to emit nothing.

    Same resolver (``_config_path``, so ``MIND_MEM_CONFIG`` and the workspace
    search order still apply) and the same fail-closed
    ``{"enabled": true}`` interpretation — only the logging is skipped.

    Use this, never :func:`is_enabled`, for a PROBE that decides whether a
    feature is on. ``is_enabled`` goes through ``_read_v4_block``, which
    ``_log.warning("v4_config_unreadable", ...)`` on a malformed config; a
    probe on an OFF path that logs makes the flag-off build observably
    different from the build that never had the feature. (Caught in slice 1
    against a pristine tree: ``v4.logging_context``'s probe emitted a stderr
    line with the flag off.) A feature that is ON is free to call
    ``is_enabled`` afterwards and get the warning.
    """
    if flag not in ALL_V4_FLAGS:
        return False
    try:
        path = _config_path()
        st = path.stat()
        block = _quiet_block(path, st.st_mtime_ns, st.st_size)
        sub = block.get(flag) if isinstance(block, dict) else None
        return isinstance(sub, dict) and sub.get("enabled") is True
    except Exception:
        # Missing, unreadable, or malformed config; a non-dict v4 block; a
        # resolver failure. Every one of them means "off", silently.
        return False


def is_enabled(flag: str) -> bool:
    """Return True iff the v4 sub-flag is set to ``{"enabled": true}``.

    Unknown flag names always return False — fail-closed so a typo in a
    config file can't accidentally turn things on.
    """
    if flag not in ALL_V4_FLAGS:
        return False
    cfg = _load_v4_block()
    sub = cfg.get(flag)
    if not isinstance(sub, dict):
        return False
    return sub.get("enabled") is True


def require_enabled(flag: str) -> None:
    """Raise :class:`FeatureDisabledError` if the flag is OFF.

    Surfaces should call this at the public-API entry point so callers
    get a clear, structured error instead of silent fallback.

    When the config exists but could not be parsed, the error says so
    instead of telling the operator to add a flag that is already sitting
    in the file — the flag is not the problem, the trailing comma is, and
    a message that names the wrong cause sends the diagnosis the wrong way.
    """
    if is_enabled(flag):
        return
    _block, error = _read_v4_block()
    if error:
        raise FeatureDisabledError(
            f"mind-mem v4 surface '{flag}' is off because the active config could not be read ({error}). "
            "Every v4 surface is off until it parses; fix the file rather than the flag."
        )
    raise FeatureDisabledError(
        f'mind-mem v4 surface \'{flag}\' is disabled. Enable via mind-mem.json: "v4": {{ "{flag}": {{ "enabled": true }} }}'
    )


def flag_config(flag: str, *, quiet: bool = False) -> dict:
    """Return the full sub-config dict for a flag (e.g.
    ``{"enabled": true, "max_tokens": 32000}``), or ``{}`` if unset.

    Surfaces use this to read their own tunables alongside the enable
    bit. Always returns a dict; never raises for missing flags.

    ``quiet=True`` suppresses the ``v4_config_unreadable`` warning and the
    warning-dedup bookkeeping — see :func:`_read_v4_block`. Use it when the
    call is the probe that decides whether an OFF-by-default surface runs at
    all, so answering "off" stays unobservable.

    The enable bit is NOT interpreted here: the result may be any JSON value
    the config holds under *flag*. Callers deciding whether a surface is on
    must apply the canonical ``isinstance(sub, dict) and sub.get("enabled")
    is True`` test themselves (or call :func:`is_enabled`), so a bare
    ``true`` still cannot switch a surface on.
    """
    if flag not in ALL_V4_FLAGS:
        return {}
    return _load_v4_block(quiet=quiet).get(flag, {}) or {}


def is_enabled_for_workspace(workspace: str, flag: str) -> bool:
    """Quiet, workspace-first probe: is *flag* on for *workspace*?

    The workspace's own ``mind-mem.json`` wins over the ambient config,
    because a caller holding one explicit workspace directory means that
    one — not whatever ``MIND_MEM_CONFIG`` or the cwd happens to resolve
    to. When the workspace config says nothing about *flag*, the ambient
    resolver answers (:func:`is_enabled_quiet`).

    Reads only, logs nothing, raises nothing. That is a hard requirement,
    not politeness: with the flag off this build must be indistinguishable
    from the one that never had the feature, and a probe that logged
    ``v4_config_unreadable`` on a malformed config would break exactly
    that.

    This is the hoist ``multi_modal.flag_enabled``'s deferred note asked
    for — one shared implementation of a shape that had been copied into
    three modules. New callers use this one; the copies migrate as slices
    touch them for other reasons.
    """
    if flag not in ALL_V4_FLAGS:
        return False
    if workspace:
        try:
            data = json.loads((Path(workspace) / "mind-mem.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            block = data.get("v4")
            if isinstance(block, dict) and flag in block:
                sub = block.get(flag)
                return isinstance(sub, dict) and sub.get("enabled") is True
    return is_enabled_quiet(flag)


def flag_config_for_workspace(workspace: str, flag: str) -> dict:
    """The sub-config dict for *flag*, workspace-first. Quiet, like the probe.

    The enable bit is not interpreted here — see :func:`flag_config`.
    """
    if flag not in ALL_V4_FLAGS:
        return {}
    if workspace:
        try:
            data = json.loads((Path(workspace) / "mind-mem.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            block = data.get("v4")
            if isinstance(block, dict) and flag in block:
                sub = block.get(flag)
                return sub if isinstance(sub, dict) else {}
    return flag_config(flag, quiet=True)


def config_error() -> str:
    """Describe why the active config could not be read — ``""`` when it can.

    Health probes and diagnostics use this to distinguish "every v4 flag
    is off" from "the config that holds every v4 flag does not parse".
    """
    return _read_v4_block()[1]


__all__ = [
    "ALL_V4_FLAGS",
    "FeatureDisabledError",
    "config_error",
    "is_enabled",
    "is_enabled_quiet",
    "require_enabled",
    "flag_config",
    "flag_config_for_workspace",
    "is_enabled_for_workspace",
]
