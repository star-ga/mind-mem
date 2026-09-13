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

IMPORT RAIL. This module must not import :mod:`.served_ledger`: ``_recall_core`` consumes the
context and the scoring path is pinned to zero ledger import edges by two tests
(``test_recall_attestation_v2.py::test_t12_the_scoring_path_ledger_surface_is_pinned`` and
``test_recall_admissibility.py::test_the_scoring_path_has_no_import_edge_to_the_importer``).
Nothing here needs the ledger — the ledger reads the context, never the reverse.
"""

from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
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

    ``config`` is wrapped in a :class:`~types.MappingProxyType` so the engine cannot rebind top
    level sections through the value it was handed. That is a guard against accident, not against
    a determined caller — nested dicts stay writable, and deep-freezing the whole tree would cost
    more than it buys on a hot path.
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
        if not isinstance(self.config, MappingProxyType):
            object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(self, "workspace", _normalise(self.workspace))

    @property
    def reads(self) -> int:
        """Reads the engine served from this context. Zero means it proved nothing."""
        return self._receipt.reads


_ACTIVE: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "mind_mem_request_context", default=None
)


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
    return context.config


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
