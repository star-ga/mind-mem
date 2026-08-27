"""Pluggable answer generators for the conversational chat layer.

A *generator* is any callable ``ChatRequest -> str``. It receives both
the rendered prompt (for text-generation backends) and the structured
evidence (for deterministic, offline backends), so the same seam serves
production and tests without a second abstraction.

Two generators ship in the box:

* :func:`extractive_generator` — fully deterministic, no network, no
  accelerator. It emits one citation-carrying sentence per evidence
  item, in recall rank order. This is the default for the CLI and the
  MCP tool so ``chat_with_memory`` works on a bare install.
* :func:`make_service_generator` — posts the rendered prompt to a local
  generation service (the same endpoint family
  :mod:`mind_mem.ollama_host` resolves for embedding / extraction) and
  returns its completion. Opt-in; never touched by the test suite.

Injecting a stub is the whole point of the seam::

    def stub(request):
        return f"Deploys happen on Fridays [[{request.evidence[0].block_id}]]."

    chat_with_memory(ws, "when do we deploy?", generator=stub)
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .observability import get_logger
from .ollama_host import ollama_base_url

_log = get_logger("chat_generators")

__all__ = [
    "DEFAULT_GENERATION_TAG",
    "ChatRequest",
    "EvidenceItem",
    "Generator",
    "GeneratorError",
    "extractive_generator",
    "make_service_generator",
    "resolve_generator",
]


#: Default tag for the local generation service. Operators override it
#: per call or through the workspace config; it is deployment config,
#: never user input.
DEFAULT_GENERATION_TAG = "mind-mem:4b"

_WHITESPACE = re.compile(r"\s+")

#: Sentence terminator *inside* an excerpt — folded to a semicolon so a
#: multi-sentence excerpt still produces exactly one cited claim.
_INTERNAL_TERMINATOR = re.compile(r"[.!?]+(\s+)")


class GeneratorError(RuntimeError):
    """Raised when a generator backend is unreachable or misconfigured."""


@dataclass(frozen=True)
class EvidenceItem:
    """One recalled block, normalised for generation + citation."""

    block_id: str
    excerpt: str
    score: float = 0.0
    source: str = ""
    date: str = ""

    def cite(self) -> str:
        """The citation token for this item."""
        return f"[[{self.block_id}]]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "excerpt": self.excerpt,
            "score": self.score,
            "source": self.source,
            "date": self.date,
        }


@dataclass(frozen=True)
class ChatRequest:
    """Everything a generator needs to answer one question."""

    question: str
    prompt: str
    evidence: tuple[EvidenceItem, ...]
    category: str = "single-hop"

    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.block_id for item in self.evidence)


#: ``ChatRequest -> answer text``.
Generator = Callable[[ChatRequest], str]


# ---------------------------------------------------------------------------
# Deterministic in-box generator
# ---------------------------------------------------------------------------


def extractive_generator(
    request: ChatRequest,
    *,
    max_sentences: int = 3,
    max_chars: int = 240,
) -> str:
    """Quote the top evidence back, one cited sentence per block.

    Deterministic for a fixed ``request`` — same input, same bytes. It
    invents nothing: every sentence is a trimmed excerpt of a real block
    followed by that block's citation. With no evidence it returns the
    literal no-record marker.

    Internal sentence terminators inside an excerpt are folded to
    semicolons so one evidence item always yields exactly one claim
    sentence — otherwise the leading half of a two-sentence excerpt
    would land in the answer as an uncited claim.
    """
    from .chat_citations import NO_RECORD

    if not request.evidence:
        return NO_RECORD

    sentences: list[str] = []
    for item in request.evidence[: max(1, max_sentences)]:
        excerpt = _WHITESPACE.sub(" ", item.excerpt or "").strip()
        if not excerpt:
            continue
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 1].rstrip() + "…"
        excerpt = _INTERNAL_TERMINATOR.sub(r";\1", excerpt).rstrip(".!?;")
        if not excerpt:
            continue
        sentences.append(f"{excerpt} {item.cite()}.")

    return " ".join(sentences) if sentences else NO_RECORD


# ---------------------------------------------------------------------------
# Local generation service adapter (opt-in, network)
# ---------------------------------------------------------------------------


def make_service_generator(
    *,
    config_section: Mapping[str, Any] | None = None,
    tag: str = DEFAULT_GENERATION_TAG,
    timeout: float = 120.0,
    temperature: float = 0.0,
    seed: int = 0,
) -> Generator:
    """Build a generator backed by the local generation service.

    The endpoint is resolved through :func:`mind_mem.ollama_host.ollama_base_url`,
    so a fleet node pointing at a central host needs no extra config
    here. ``temperature=0`` + a fixed ``seed`` are the defaults because a
    grounded answer should be reproducible.

    Raises :class:`GeneratorError` at call time when the endpoint is
    unreachable — never at build time, so constructing the generator is
    always side-effect free.
    """
    base = ollama_base_url(config_section)
    url = f"{base}/api/generate"

    def _generate(request: ChatRequest) -> str:
        payload = json.dumps(
            {
                "model": tag,
                "prompt": request.prompt,
                "stream": False,
                "options": {"temperature": float(temperature), "seed": int(seed)},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 — base URL from operator-controlled config/env only (ollama_base_url enforces http/https), never user input
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            _log.warning("chat_generation_service_failed", url=url, error=str(exc))
            raise GeneratorError(f"generation service at {url} is unreachable: {exc}") from exc
        text = body.get("response") if isinstance(body, dict) else None
        return text if isinstance(text, str) else ""

    # deferred: token streaming (roadmap docs/roadmap-v4.md §chat) is not
    # wired — the adapter requests a single non-streamed completion.
    # upgrade path: set "stream": True and yield decoded lines through a
    # StreamingGenerator variant of this factory.
    return _generate


# ---------------------------------------------------------------------------
# Name -> generator resolution (CLI / MCP surface)
# ---------------------------------------------------------------------------


_BUILTIN: dict[str, Callable[[], Generator]] = {
    "extractive": lambda: extractive_generator,
    "service": lambda: make_service_generator(),
}


def resolve_generator(name: str) -> Generator:
    """Resolve a generator by its CLI/MCP name.

    ``"extractive"`` (default) is offline and deterministic;
    ``"service"`` calls the local generation service. Unknown names
    raise :class:`ValueError` — an unrecognised backend must fail at the
    boundary, not silently fall back to a different answer source.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("generator name must be a non-empty string")
    key = name.strip().lower()
    factory = _BUILTIN.get(key)
    if factory is None:
        raise ValueError(f"unknown generator {name!r}; choose one of: {', '.join(sorted(_BUILTIN))}")
    return factory()
