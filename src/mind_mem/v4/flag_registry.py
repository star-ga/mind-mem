"""Three-state registry for every declared v4 feature flag.

A declared flag with no consumer is a promise the code does not keep: an
operator sets ``{"enabled": true}``, nothing reads it, and the product
answers with silence. Silence is indistinguishable from success, which is
why 20 of the 52 flags in :data:`~mind_mem.v4.feature_flags.ALL_V4_FLAGS`
could sit unconsumed without anyone noticing.

This module forces every declared flag into exactly one of three states,
carried here as data and checked against the source tree by
:func:`resolve_consumers`:

``WIRED``
    At least one call site in ``src/`` gates a feature on the flag.
    Default-OFF opt-in: absent config means the feature does not run.

``KILL_SWITCH``
    The feature already ships ungated, so the flag is read **default-ON**
    (:func:`~mind_mem.v4.feature_flags.is_kill_switch_active`) and setting
    ``{"enabled": false}`` is what changes behaviour. Adding one is
    additive by construction: with the key absent the answer is the same
    as before the flag existed.

``UNIMPLEMENTED``
    No consumer and no shipped feature behind it. Enabling one raises
    :class:`UnimplementedCapabilityError` naming the flag, because a
    capability that is declared but absent must refuse rather than
    silently succeed.

There is no fourth state. A flag that loses its last consumer stops
matching its declared state and ``tests/test_flag_registry.py`` fails,
naming the flag — the classification cannot go stale in silence.

**Deletion discipline.** An ``UNIMPLEMENTED`` entry is a question about
wiring, never a verdict about worth: every one keeps its declaration, its
place in ``ALL_V4_FLAGS``, and a ``note`` recording what a consumer would
have to do. Removing a flag to make this gate pass is the one repair that
is never correct.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Iterable, Mapping


class FlagState(str, Enum):
    """The three states a declared flag may be in.

    A ``str`` enum so the value that reaches a report, a log line, or a
    JSON payload is the plain lowercase tag, never ``"FlagState.WIRED"``.
    A reader that does not know a tag must compare the raw string and
    treat anything unrecognised as :data:`UNIMPLEMENTED` — the
    fail-closed reading. By the registry ruling there is no fourth state
    to add; if that ever changes, an older reader still parses the record
    because it dispatches on the string, not on this class.
    """

    WIRED = "wired"
    KILL_SWITCH = "kill_switch"
    UNIMPLEMENTED = "unimplemented"


class UnimplementedCapabilityError(RuntimeError):
    """Raised when config enables a flag with nothing behind it.

    Deliberately **not** a subclass of
    :class:`~mind_mem.v4.feature_flags.FeatureDisabledError`: a caller
    that writes ``except FeatureDisabledError`` is saying "fall back to
    the v3 path", which is the right handling for a surface that is off
    and exactly the wrong handling for a surface that does not exist.
    Catching one must not swallow the other, so this error propagates
    past every such handler and reaches the operator who set the flag.
    """


@dataclass(frozen=True)
class FlagRecord:
    """One flag's declared state plus the reasoning behind it."""

    name: str
    state: FlagState
    note: str


def _wired(name: str, note: str) -> FlagRecord:
    return FlagRecord(name=name, state=FlagState.WIRED, note=note)


def _unimplemented(name: str, note: str) -> FlagRecord:
    return FlagRecord(name=name, state=FlagState.UNIMPLEMENTED, note=note)


#: Every declared flag, keyed by name. ``set(FLAG_STATES) ==
#: set(ALL_V4_FLAGS)`` is asserted by the test suite, so a new flag that
#: arrives without a state fails the build instead of defaulting to one.
FLAG_STATES: Final[Mapping[str, FlagRecord]] = {
    record.name: record
    for record in (
        # -- Group A: cognition / model layer ------------------------------
        _wired("cognitive_kernel", "v4/cognitive_kernel.py refuses at four entry points; v4/health.py probes it."),
        _wired("surprise_retrieval", "v4/surprise_retrieval.py gates its entry point and reads its own tunables."),
        _wired("llm_noise_profile", "llm_noise_profile.py probes before writing the sidecar profile."),
        # -- Group B: knowledge graph --------------------------------------
        _wired("block_kinds", "v4/block_kinds.py refuses at nine entry points; v4/health.py probes it."),
        _unimplemented(
            "long_context_recall",
            "Declared with a max_tokens tunable that nothing reads. Wiring question: the recall packer "
            "would have to consult it instead of its own budget constant — overlaps context_budget, "
            "which IS wired. Resolve the overlap before wiring; do not delete either.",
        ),
        _unimplemented(
            "fusion",
            "RRF fusion ships ungated in hybrid_recall.py. Wiring question: it is a KILL_SWITCH candidate "
            "(default-ON), not an opt-in — a consumer must read is_kill_switch_active, and the twin must "
            "prove ON is byte-identical to today.",
        ),
        _unimplemented(
            "streaming_recall",
            "No streaming recall surface exists. Wiring question: change_stream.py streams block events, "
            "not recall results; a consumer would have to be built first.",
        ),
        _unimplemented(
            "chat",
            "chat_memory.py / chat_cli.py ship ungated. Wiring question: KILL_SWITCH candidate — an operator "
            "who wants the chat surface off currently has no way to turn it off.",
        ),
        _unimplemented(
            "prompt_schema",
            "No prompt-schema surface exists. Wiring question: the coding_schemas.py vocabulary is the "
            "nearest shipped thing; whether it is what this flag was declared for is unrecorded.",
        ),
        # -- Group C: governance / UX --------------------------------------
        _unimplemented(
            "idle_ingest",
            "No idle-ingest scheduler exists. Wiring question: daemon.py runs the maintenance passes that "
            "such a scheduler would hook; the pass itself was never built.",
        ),
        _wired("lint", "lint.py reads the v4 block directly in _ambient_flag_enabled and flag_enabled."),
        _unimplemented(
            "contradiction_states",
            "contradiction_detector.py ships its states ungated. Wiring question: KILL_SWITCH candidate.",
        ),
        _unimplemented(
            "self_heal",
            "No self-heal pass exists. Wiring question: auto_resolver.py is the closest shipped surface and "
            "is governed by proposals, which may be the honest answer instead of a flag.",
        ),
        _unimplemented(
            "viewer",
            "No viewer surface ships in this package. Wiring question: the flag may belong to a client, not "
            "to mind-mem; if so it should move rather than be deleted.",
        ),
        _unimplemented(
            "contradiction_stream",
            "change_stream.py has no contradiction channel. Wiring question: event_fanout.py is the emission seam a consumer would use.",
        ),
        _wired("world_staleness", "world_staleness_config.py resolves the section workspace-first, then ambient."),
        _wired("maintenance_layout", "maintenance_migrate.py gates migrate_if_enabled on it."),
        _wired("ingest_serve", "ingestion_pipeline.py reads the v4 block directly in both flag helpers."),
        _wired("bootstrap_corpus", "bootstrap_corpus.flag_enabled probes it before the one-shot backfill."),
        # -- Group D: platform scale ---------------------------------------
        _unimplemented(
            "rust_hot_path",
            "No Rust extension ships. Wiring question: the compiled accelerator here is the Cython "
            "_mic_map_accel; a Rust path would need to exist before a flag can select it.",
        ),
        _unimplemented(
            "embedding_fallback",
            "embedding_pipeline.py falls back unconditionally today. Wiring question: KILL_SWITCH candidate — "
            "an operator who wants a hard failure instead of a degraded embedding cannot ask for one.",
        ),
        _wired("pq", "v4/pq.py refuses at ten entry points; recall_vector.py and mcp/tools/memory_ops.py probe it."),
        _wired("hnsw_kind_index", "v4/hnsw_kind_index.py refuses at six entry points; recall and memory_ops probe it."),
        _wired("federation", "v4/federation.py refuses at six entry points; v4/health.py probes it."),
        _wired("embedding_pipeline", "v4/embedding_pipeline.py refuses at three entry points; kind_backfill probes it."),
        _wired("kind_summaries", "v4/kind_summaries.py refuses at five entry points; benchmark and backfill probe it."),
        _wired("self_editing", "v4/self_editing.py and v4/block_versioning.py refuse at ten entry points."),
        _wired("granularity_align", "mcp/tools/consolidation.py reads the section workspace-first, then ambient."),
        _wired("multi_modal", "multi_modal.flag_enabled probes it workspace-first before the ingest door."),
        _wired("observability", "v4/observability.py refuses and reads its tunables; v4/health.py probes it."),
        _wired("logging_context", "observability.py reads the v4 block directly to install the context filter."),
        _wired("backpressure", "v4/backpressure.py refuses, reads tunables, and probes on the publish path."),
        _wired("block_metadata", "v4/block_metadata.py refuses at eight entry points."),
        _wired("circuit_breaker", "v4/circuit_breaker.py refuses at two entry points and reads its tunables."),
        # -- Group E: compliance-sensitive opt-in --------------------------
        _unimplemented(
            "redaction",
            "No redaction pass exists. Wiring question: redactable tombstones are the RA.3 unit with five "
            "named preconditions; until one lands, enabling this must refuse rather than imply redaction.",
        ),
        _unimplemented(
            "time_bounded_recall",
            "Time-bounded recall ships ungated in _recall_temporal.py. Wiring question: KILL_SWITCH candidate — "
            "the knob currently misleads an operator who sets it to false expecting the bound to lift.",
        ),
        _wired("vocabulary", "v4/vocabulary.py refuses at three entry points."),
        _unimplemented(
            "provenance",
            "Provenance fields ship ungated in block_provenance.py and capture.py. Wiring question: "
            "KILL_SWITCH candidate; a default-ON consumer must prove ON is byte-identical to today.",
        ),
        _unimplemented(
            "evidence",
            "Evidence objects and bundles ship ungated. Wiring question: KILL_SWITCH candidate, and the "
            "widest-blast-radius one — the evidence chain is load-bearing, so an OFF path needs its own design.",
        ),
        _wired("tenant_kms", "encryption.py probes it before resolving a per-tenant key."),
        _unimplemented(
            "tenant_chains",
            "tenant_audit.py has no per-tenant chain writer. Wiring question: it needs the chain seam, not a flag, first.",
        ),
        _unimplemented(
            "compliance_export",
            "The `mm export --policy` verb does not exist. Wiring question: the X-1 prov-o export unit is what "
            "would give this flag something to gate.",
        ),
        _unimplemented(
            "contraindicates_edges",
            "knowledge_graph.py has no contraindication relation. Wiring question: it is a typed-edge kind, so "
            "it depends on typed_edges landing first.",
        ),
        _unimplemented(
            "typed_edges",
            "The typed relation layer is declared in the roadmap and not built. Wiring question: proposal-gated "
            "edge writes exist; the type vocabulary does not.",
        ),
        _wired("entity_observations", "knowledge_graph.py refuses the accreted-facts field when it is off."),
        _wired("core_export", "mcp/tools/core.py probes it before the .mmcore export surface."),
        _wired("mind_kernels", "hash_chain_v2.py probes it at the import door; memory_ops.py probes it too."),
        _wired("trajectory", "outcome_attribution.py and mcp/tools/trajectory.py probe TRAJECTORY_FLAG."),
        _wired("context_budget", "mcp/tools/_helpers.py probes it through the shared workspace-first wrapper."),
        _wired("retrieval_metrics", "mcp/tools/_helpers.py probes it through the shared workspace-first wrapper."),
        _wired("online_training", "dream_cycle.py, online_trainer.py and memory_ops.py all probe it."),
        _wired("health", "mcp/tools/memory_ops.py probes it before folding the probe sweep into memory_health."),
    )
}


def _names_in(state: FlagState) -> frozenset[str]:
    return frozenset(name for name, record in FLAG_STATES.items() if record.state is state)


#: Flags with at least one opt-in consumer.
WIRED: Final[frozenset[str]] = _names_in(FlagState.WIRED)

#: Flags read default-ON, where ``{"enabled": false}`` is the operative setting.
KILL_SWITCH: Final[frozenset[str]] = _names_in(FlagState.KILL_SWITCH)

#: Flags with nothing behind them. Enabling one refuses.
UNIMPLEMENTED: Final[frozenset[str]] = _names_in(FlagState.UNIMPLEMENTED)


def state_of(flag: str) -> FlagState | None:
    """The declared state of *flag*, or ``None`` when it is not declared."""
    record = FLAG_STATES.get(flag)
    return record.state if record is not None else None


def require_implemented(flag: str) -> None:
    """Raise :class:`UnimplementedCapabilityError` if *flag* has nothing behind it.

    An undeclared name is not this function's business — the caller's own
    fail-closed check owns that — so it returns quietly. Only a name the
    registry knows to be ``UNIMPLEMENTED`` refuses.
    """
    record = FLAG_STATES.get(flag)
    if record is None or record.state is not FlagState.UNIMPLEMENTED:
        return
    raise UnimplementedCapabilityError(
        f"mind-mem v4 flag '{flag}' is declared but not implemented: nothing in mind-mem reads it, "
        f"so enabling it would change nothing. {record.note} "
        f'Remove "{flag}" from the "v4" block, or wire a consumer and move it out of '
        "mind_mem.v4.flag_registry.UNIMPLEMENTED."
    )


# ---------------------------------------------------------------------------
# Consumer resolution — AST over the source tree, never grep
# ---------------------------------------------------------------------------

#: Flag-resolver entry points, mapped to the positional index of the flag
#: argument. A call to one of these with a resolvable flag name IS a
#: consumer; that is the definition the registry is checked against.
PROBE_ARG_INDEX: Final[Mapping[str, int]] = {
    "is_enabled": 0,
    "is_enabled_quiet": 0,
    "require_enabled": 0,
    "flag_config": 0,
    "is_kill_switch_active": 0,
    "is_enabled_for_workspace": 1,
    "flag_config_for_workspace": 1,
}

#: Modules that declare or resolve flags rather than consume them. Naming a
#: flag here is bookkeeping, not a consumer, and counting it would let the
#: registry satisfy itself.
_NON_CONSUMER_MODULES: Final[frozenset[str]] = frozenset(
    {
        "mind_mem/v4/feature_flags.py",
        "mind_mem/v4/flag_registry.py",
    }
)

_MAX_WRAPPER_DEPTH: Final[int] = 4


def default_source_root() -> Path:
    """The directory holding the ``mind_mem`` package."""
    return Path(__file__).resolve().parents[2]


def _callee_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _module_constants(tree: ast.AST) -> dict[str, str]:
    """Every ``NAME = "literal"`` binding in *tree*, module level or not.

    Flag names are spelled as module constants far more often than as
    inline literals (``TRAJECTORY_FLAG``, ``INGEST_SERVE_FLAG``,
    ``FLAG``), so a resolver that only saw literals would report a live
    consumer as absent — and the registry would then refuse a shipping
    feature.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and isinstance(node.target, ast.Name)
        ):
            found[node.target.id] = node.value.value
    return found


def _dotted(rel: Path) -> str:
    parts = rel.with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import(module_dotted: str, node: ast.ImportFrom) -> str | None:
    """The dotted module an ``ImportFrom`` refers to, absolute or relative."""
    if node.level == 0:
        return node.module
    package = module_dotted.split(".")[:-1]
    if node.level > 1:
        package = package[: -(node.level - 1)] if node.level - 1 <= len(package) else []
    base = ".".join(package)
    if not node.module:
        return base or None
    return f"{base}.{node.module}" if base else node.module


def _imported_constants(module_dotted: str, tree: ast.AST, table: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """String constants this module pulled in with ``from X import NAME``.

    ``outcome_attribution.py`` does exactly this with
    ``trajectory.TRAJECTORY_FLAG``; without following the import the flag
    reads as unconsumed while two live call sites gate on it.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        source = _resolve_import(module_dotted, node)
        if source is None:
            continue
        exported = table.get(source)
        if not exported:
            continue
        for alias in node.names:
            value = exported.get(alias.name)
            if value is not None:
                found[alias.asname or alias.name] = value
    return found


def _reads_v4_block(tree: ast.AST) -> bool:
    """True when the module reaches into the raw ``"v4"`` config block.

    Five modules resolve their own flag rather than calling the shared
    resolver (``lint``, ``ingestion_pipeline``, ``observability``,
    ``multi_modal``, ``mcp/tools/consolidation``). Inside such a module a
    flag name used as a mapping key is a real consumer, so the scan has
    to admit that shape — a resolver that only knew the shared API would
    misreport three shipping features as unimplemented.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node.func) == "get" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and arg.value == "v4":
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "v4":
            return True
    return False


def _derived_probes(tree: ast.AST) -> dict[str, int]:
    """Local wrappers that forward a parameter into a known probe.

    ``mcp/tools/_helpers._flag_enabled(ws, flag)`` is one: the real flag
    names appear at its call sites, one frame out from the shared API.
    """
    probes = dict(PROBE_ARG_INDEX)
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for _ in range(_MAX_WRAPPER_DEPTH):
        grew = False
        for func in functions:
            positional = [a.arg for a in func.args.args]
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                name = _callee_name(node.func)
                index = probes.get(name) if name is not None else None
                if index is None or len(node.args) <= index:
                    continue
                arg = node.args[index]
                if not isinstance(arg, ast.Name) or arg.id not in positional:
                    continue
                forwarded = positional.index(arg.id)
                if probes.get(func.name) != forwarded:
                    probes[func.name] = forwarded
                    grew = True
        if not grew:
            break
    return probes


def resolve_consumers(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Map every declared flag to the ``file:line:how`` of each consumer.

    Four shapes count, because all four ship today:

    1. a call to a resolver in :data:`PROBE_ARG_INDEX` with a literal or a
       module-constant flag name;
    2. the same, through a local wrapper that forwards the argument;
    3. the same, with the constant imported from another module;
    4. a mapping read of the flag name inside a module that parses the raw
       ``"v4"`` config block itself.

    Everything is resolved from the AST. A grep for ``is_enabled("x")``
    sees only the first shape and misses five live consumers.
    """
    source_root = (root or default_source_root()).resolve()
    files = sorted(p for p in source_root.rglob("*.py") if p.is_file())

    parsed: dict[str, tuple[Path, ast.Module]] = {}
    constants: dict[str, Mapping[str, str]] = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        dotted = _dotted(path.relative_to(source_root))
        parsed[dotted] = (path, tree)
        constants[dotted] = _module_constants(tree)

    found: dict[str, list[str]] = {name: [] for name in FLAG_STATES}
    for dotted, (path, tree) in parsed.items():
        rel = path.relative_to(source_root).as_posix()
        if rel in _NON_CONSUMER_MODULES:
            continue
        names = dict(constants[dotted])
        names.update(_imported_constants(dotted, tree, constants))
        probes = _derived_probes(tree)
        raw_reader = _reads_v4_block(tree)

        def resolve(node: ast.expr | None, _names: Mapping[str, str] = names) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name):
                return _names.get(node.id)
            if isinstance(node, ast.Attribute):
                return _names.get(node.attr)
            return None

        def record(flag: str | None, line: int, how: str) -> None:
            if flag in found:
                found[str(flag)].append(f"{rel}:{line}:{how}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _callee_name(node.func)
                index = probes.get(name) if name is not None else None
                if index is not None and len(node.args) > index:
                    record(resolve(node.args[index]), node.lineno, f"{name}()")
                if raw_reader and name == "get" and node.args:
                    record(resolve(node.args[0]), node.lineno, "v4-block .get()")
            elif raw_reader and isinstance(node, ast.Subscript):
                record(resolve(node.slice), node.lineno, "v4-block []")
            elif raw_reader and isinstance(node, ast.Compare) and any(isinstance(op, ast.In) for op in node.ops):
                record(resolve(node.left), node.lineno, "v4-block in")

    return {name: tuple(sorted(set(sites))) for name, sites in found.items()}


def expected_state(consumer_count: int, declared: FlagState) -> FlagState:
    """The state the source tree justifies, given a declared state.

    ``WIRED`` and ``KILL_SWITCH`` are both "has a consumer" and differ in
    how the consumer reads the flag, which the consumer count cannot see —
    so a declared ``KILL_SWITCH`` with consumers stays ``KILL_SWITCH``.
    Zero consumers admits exactly one honest answer.
    """
    if consumer_count == 0:
        return FlagState.UNIMPLEMENTED
    return FlagState.KILL_SWITCH if declared is FlagState.KILL_SWITCH else FlagState.WIRED


def classification_drift(
    consumers: Mapping[str, Iterable[str]] | None = None,
    states: Mapping[str, FlagRecord] | None = None,
    *,
    root: Path | None = None,
) -> tuple[tuple[str, FlagState, FlagState, tuple[str, ...]], ...]:
    """Flags whose declared state disagrees with the source tree.

    Returns ``(flag, declared, actual, consumers)`` per disagreement, so a
    failure message can name the flag and show the evidence rather than
    asserting a bare count. Both mappings are injectable, which is what
    lets the mutation twin drive a disagreement without editing the file.
    """
    table = states if states is not None else FLAG_STATES
    sites = consumers if consumers is not None else resolve_consumers(root)
    drift: list[tuple[str, FlagState, FlagState, tuple[str, ...]]] = []
    for name, record in table.items():
        found = tuple(sites.get(name, ()))
        actual = expected_state(len(found), record.state)
        if actual is not record.state:
            drift.append((name, record.state, actual, found))
    return tuple(sorted(drift))


def kill_switch_call_sites(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Where ``is_kill_switch_active`` is called, keyed by the flag it names.

    A default-ON read of a flag the registry does not declare
    ``KILL_SWITCH`` inverts that flag's meaning, and nothing at runtime
    can tell. Checking it here makes the mistake a build failure at zero
    runtime cost.
    """
    source_root = (root or default_source_root()).resolve()
    sites: dict[str, list[str]] = {}
    for path in sorted(p for p in source_root.rglob("*.py") if p.is_file()):
        rel = path.relative_to(source_root).as_posix()
        if rel in _NON_CONSUMER_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        names = _module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee_name(node.func) != "is_kill_switch_active":
                continue
            if not node.args:
                continue
            arg = node.args[0]
            flag = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
            if flag is None and isinstance(arg, ast.Name):
                flag = names.get(arg.id)
            sites.setdefault(flag or "<unresolved>", []).append(f"{rel}:{node.lineno}")
    return {flag: tuple(sorted(found)) for flag, found in sites.items()}


def audit(root: Path | None = None) -> dict[str, object]:
    """A machine-readable snapshot of the registry against the source tree."""
    consumers = resolve_consumers(root)
    return {
        "counts": {
            "declared": len(FLAG_STATES),
            "wired": len(WIRED),
            "kill_switch": len(KILL_SWITCH),
            "unimplemented": len(UNIMPLEMENTED),
        },
        "consumers": consumers,
        "drift": classification_drift(consumers),
        "kill_switch_call_sites": kill_switch_call_sites(root),
    }


__all__ = [
    "FLAG_STATES",
    "KILL_SWITCH",
    "PROBE_ARG_INDEX",
    "UNIMPLEMENTED",
    "WIRED",
    "FlagRecord",
    "FlagState",
    "UnimplementedCapabilityError",
    "audit",
    "classification_drift",
    "default_source_root",
    "expected_state",
    "kill_switch_call_sites",
    "require_implemented",
    "resolve_consumers",
    "state_of",
]
