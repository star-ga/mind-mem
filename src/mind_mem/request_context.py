"""One immutable policy context per request, consumed by the engine itself.

WHY THIS MODULE EXISTS. Threading a captured config into ``attest_and_record`` was not enough,
and an independent review said exactly why: the engine reloads configuration on its own. Eight
call sites in :mod:`._recall_core` and one in :mod:`.sqlite_index` call ``_get_config(workspace)``
during a single recall, and ``prefetch`` fans out N+1 direct core recalls that each load live
config in their own thread. So a row could carry a hash captured by the caller while the ranking
it describes was produced under a different configuration. Moving the capture earlier fixed WHEN
the hash was read; it did not make the hash a fact about the ranking.

The binding therefore has to sit where the engine reads, not where the recorder writes.

WHAT "PROVEN" MEANS HERE, AND WHY THERE IS A RECEIPT. A context that is bound but never read
proves nothing — it is indistinguishable from no context at all, which is the vacuous-pass shape
this repo keeps paying for. So the context carries a :class:`_Receipt` that counts the reads the
engine actually served from it. A caller may only claim a recorded row when the receipt shows the
engine consumed the context at least once. The count is observational: it records what happened,
it never changes what any read returns.

THREAD-POOL WORKERS ARE EXPLICIT, NOT INHERITED. ``ContextVar`` values do not cross a
``ThreadPoolExecutor`` boundary on their own, which is the honest default here: a worker that was
never handed the context must not silently read live config and have it counted as bound. Workers
are bound by submitting work through :func:`bind_current`, which captures the context in the SUBMITTING
thread so the parent's receipt sees the worker's reads too, and a worker nobody bound stays visibly
unbound instead of quietly answering from disk.

WORKSPACE IS PART OF THE KEY. A context bound for workspace A must never answer a read for
workspace B. Without that check a multi-workspace server, or a test that binds once and recalls
twice, would attribute one workspace's policy to another's ranking.

WHAT IS DELIBERATELY *NOT* BOUND, so the exclusion is not mistaken for an oversight.
``served_ledger.ledger_enabled`` re-reads ``mind-mem.json`` on every served recall and keeps doing
so. Measured after this work: the ranked, axis and public doors each show ZERO ranking-path config
reads while bound; that one recorder-side read is the only remaining live read, and it stays live for
two reasons. It cannot affect WHICH POLICY PRODUCED THE RANKING, which is the entire claim this
module exists to support — it decides only whether a row is appended. And its read-failure path
returns ``False`` deliberately, fail-closed: routing it through a context whose config may be an
empty fallback after a failed capture would turn that into fail-OPEN, and it would reintroduce the
staleness window its own docstring says was rejected on purpose so an operator's ``enabled: false``
is honoured promptly. Binding it would trade a real safety property for a coherence that nothing
reads.

IMPORT RAIL. This module must not import :mod:`.served_ledger`: ``_recall_core`` consumes the
context and the scoring path is pinned to zero ledger import edges by two tests
(``test_recall_attestation_v2.py::test_t12_the_scoring_path_ledger_surface_is_pinned`` and
``test_recall_admissibility.py::test_the_scoring_path_has_no_import_edge_to_the_importer``).
Nothing here needs the ledger — the ledger reads the context, never the reverse.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional, TypeVar

__all__ = [
    "RequestContext",
    "bind_request_context",
    "active_request_context",
    "context_config_for",
    "context_was_consumed_for",
    "bind_current",
]

T = TypeVar("T")


@dataclass
class _Receipt:
    """How many reads the engine actually served from the context it was given.

    Deliberately mutable and deliberately NOT part of the context's value: two contexts with the
    same policy are the same policy whatever their read counts. It exists so "the engine consumed
    this" is a measurement rather than an assumption.
    """

    reads: int = 0

    def record(self) -> None:
        self.reads += 1


@dataclass(frozen=True)
class RequestContext:
    """The policy coordinates of ONE request, fixed before retrieval begins.

    ``config`` is a PLAIN ``dict``, deliberately, and that was a correction rather than a default.

    The first version wrapped it in a :class:`~types.MappingProxyType` so the engine could not rebind
    top-level sections. That guard silently broke storage selection: ``storage.resolve_backend`` (and
    every other consumer written against this repo's plain-dict convention) tests
    ``isinstance(config, dict)``, a ``mappingproxy`` is NOT a ``dict``, and the predicate degrades to
    the markdown backend on failure — audibly in the log, but a workspace on postgres or an encrypted
    store would silently serve from the wrong corpus. Observed live as
    ``block_store_config_malformed: config must be an object, got mappingproxy; degrading to markdown
    backend``.

    So the proxy is gone. Immutability of the context now rests on the fact that nothing here hands
    the mapping to a mutating caller — ``mcp.infra.config._load_config`` already returns ``dict(...)``
    for callers that merge defaults into it. That is a weaker guarantee than the proxy gave, and it is
    the right trade: a hypothetical rebind is a bug we would see, while a silent backend downgrade is
    a bug that returns plausible answers from the wrong store.
    """

    workspace: str
    config: Mapping[str, Any]
    config_hash: Optional[str] = None
    index_anchor: Optional[str] = None
    scoring_instant: Optional[str] = None
    pipeline_hash: Optional[str] = None
    backend_admission: Optional[Mapping[str, Any]] = None
    _receipt: _Receipt = field(default_factory=_Receipt, compare=False, repr=False)

    def __post_init__(self) -> None:
        # dict(), never MappingProxyType — see the class docstring. A proxy fails the
        # isinstance(config, dict) predicate that storage backend selection and other consumers use.
        # deepcopy on the way IN as well: the door hands us the mapping it also passes elsewhere, so a
        # shallow store would let a consumer's nested edit reach the snapshot through the original.
        object.__setattr__(self, "config", copy.deepcopy(dict(self.config)))
        object.__setattr__(self, "workspace", _normalise(self.workspace))

    @property
    def reads(self) -> int:
        """Reads the engine served from this context. Zero means it proved nothing."""
        return self._receipt.reads


_ACTIVE: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar("mind_mem_request_context", default=None)


def _normalise(workspace: str | os.PathLike[str]) -> str:
    """Compare workspaces by resolved path, so ``.`` and an absolute path agree."""
    try:
        return os.path.realpath(os.path.abspath(str(workspace)))
    except OSError:
        return str(workspace)


@contextmanager
def bind_request_context(context: RequestContext) -> Iterator[RequestContext]:
    """Bind *context* for the duration of the block, restoring any previous one after."""
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def active_request_context() -> Optional[RequestContext]:
    """The bound context, or ``None``. Does not count as a read — no policy was served."""
    return _ACTIVE.get()


def context_config_for(workspace: str | os.PathLike[str]) -> Optional[Mapping[str, Any]]:
    """The bound config for *workspace*, or ``None`` when nothing is bound for it.

    ``None`` means "no context governs this read", never "the config is empty" — the caller then
    reads live config as it always did, and the absent receipt entry is what keeps the resulting
    row honest about not being proven.
    """
    context = _ACTIVE.get()
    if context is None or context.workspace != _normalise(workspace):
        return None
    context._receipt.record()
    # A DEEP COPY, and a plain dict. Two requirements that pull in opposite directions meet here.
    # Consumers test ``isinstance(config, dict)`` — ``storage._backend_name`` degrades to the markdown
    # backend when that fails — so this may not be a ``MappingProxyType``; an earlier version was, and
    # it silently downgraded storage on non-markdown workspaces. But handing out the context's own
    # mapping would let any reader rebind the policy every later reader sees. A shallow copy satisfies
    # both: real ``dict`` for the predicates, and top-level rebinding cannot reach the context.
    # SHALLOW WAS NOT ENOUGH, and that was demonstrated rather than argued. A shallow copy still
    # shares nested sections, and ``HybridBackend.__init__`` MUTATES its nested recall config when
    # validation rejects a numeric value: a control passed {'recall': {'rrf_k': 'bad', ...}} through
    # ``from_config`` and ``rrf_k`` then disappeared from the captured context itself
    # (before=["rrf_k","vector_weight"], after=["vector_weight"]). A context a consumer can edit is not
    # a snapshot, and a later generation or config observation would describe a policy no caller
    # established. ``deepcopy`` costs a traversal of a small mapping per read, which is the right price
    # for the word "immutable" being true rather than aspirational.
    return copy.deepcopy(context.config)


def context_was_consumed_for(workspace: str | os.PathLike[str]) -> bool:
    """True only if a context is bound for *workspace* AND the engine read from it.

    This is the predicate a recorded row depends on. A bound-but-unread context returns False,
    because a context nothing consumed is not evidence about the ranking.
    """
    context = _ACTIVE.get()
    if context is None or context.workspace != _normalise(workspace):
        return False
    return context.reads > 0


def bind_current(func: Callable[..., T]) -> Callable[..., T]:
    """Wrap *func* so it runs under the context bound RIGHT NOW, wherever it later runs.

    THE CAPTURE MUST HAPPEN HERE, IN THE SUBMITTING THREAD. This is the whole reason the function
    exists and the reason it is not spelled ``bound_call(func, *args)``: a helper that reads the
    active context when it is *invoked* reads it inside the worker, where the context is fresh and
    empty, so it binds nothing and reports success. The context is therefore read while building
    the wrapper — in the parent, at submit time — and closed over.

    Use it as ``executor.submit(bind_current(fn), arg)`` or ``ex.map(bind_current(fn), items)``.
    Submitting ``fn`` bare is not a smaller version of this: the worker reads live config, and
    because it recorded no read against the context the resulting row stays honestly unproven.

    Returns *func* unchanged when nothing is bound, so an unbound path pays no wrapper.
    """
    context = _ACTIVE.get()
    if context is None:
        return func

    @functools.wraps(func)
    def _bound(*args: Any, **kwargs: Any) -> T:
        with bind_request_context(context):
            return func(*args, **kwargs)

    return _bound
