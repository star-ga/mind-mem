#!/usr/bin/env python3
"""Real `Compressor` implementations for mind_mem.recompaction. Zero new deps.

`recompaction.py` defines the fixed-point re-compression loop and injects a
``Compressor = Callable[[str, list[dict]], str]``. This module supplies the
concrete compressors that satisfy that contract:

* :class:`EchoCompressor` — returns its input verbatim. The trivially-
  converging control: any harness measuring recompaction quality must score
  this compressor at perfect convergence and perfect fact retention, or the
  harness itself (not the model) is broken.
* :class:`OllamaCompressor` — calls a local ollama model over HTTP.

Purity is load-bearing here, not a nicety. ``recompact_cluster`` stops when a
pass returns bytes identical to its input; a compressor that is not pure
w.r.t. its inputs (uses the wall clock, real randomness, or non-zero sampling
temperature) can never legitimately reach that fixed point — the same
discipline the module docstring in ``recompaction.py`` names for
evidence-chain preimages. :class:`OllamaCompressor` therefore pins
``temperature=0`` and a fixed integer ``seed`` in the ollama request options
on every call.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT_S = 60.0

_PROMPT_TEMPLATE = """You normalize a cluster of related memory blocks into a single CANONICAL \
FACT LIST. This is a deterministic normal form, not free writing: the same \
facts must always produce the exact same bytes, so that re-normalizing an \
already-canonical list returns it byte-for-byte unchanged (a fixed point).

Apply these rules EXACTLY, in order:
1. Read the current fact list and the sibling blocks together.
2. Emit one atomic fact per line, each line beginning with "- " (hyphen, space).
3. PRESERVE EVERY FACT verbatim where possible — copy every number, identifier, \
date, proper noun, and decision exactly as written; never round, generalize, \
rephrase, or invent. A tighter list drops only EXACT duplicates and pure \
connective prose ("Additionally,", "As noted above"), never a fact.
4. If two lines state the same fact, keep only the first occurrence.
5. Order lines by the position their fact first appears across the current list \
then the sibling blocks (stable order — do not re-sort by any other key).
6. Do not add commentary, headings, blank lines, trailing punctuation beyond \
what a fact contains, markdown fences, or a preamble.

Because these rules are deterministic and idempotent, a list that is already \
canonical (no duplicates, no connective prose, already one-fact-per-line in \
first-appearance order) MUST be returned byte-for-byte identical.

Current fact list:
{current}

Sibling blocks:
{siblings}

Canonical fact list:"""


class CompressorError(RuntimeError):
    """A compressor call failed. Never swallowed into a silent input echo.

    Returning the input on error would look, byte-for-byte, exactly like a
    real fixed point to `recompact_cluster` — that would fake convergence
    rather than report the outage. Every failure path here raises instead.
    """


def _echo(current_text: str, blocks: list[dict[str, Any]]) -> str:
    return current_text


def EchoCompressor() -> Callable[[str, list[dict[str, Any]]], str]:  # noqa: N802 - factory named like the type it returns
    """Return a `Compressor` that returns its input verbatim.

    The trivially-converging control for benchmark baselining: it converges
    on the first pass (``recompact_cluster`` sees ``rewritten == current``
    immediately) and loses zero facts, so a benchmark run against it must
    score ``convergence_rate == 1.0`` and ``fact_retention == 1.0``.
    """
    return _echo


def _render_siblings(blocks: list[dict[str, Any]]) -> str:
    """Render sibling blocks as scannable text for the prompt.

    Mirrors the `_block_body` convention in `recompaction.py`: prefer an
    explicit ``body`` field, else join public (non ``_``) field values.
    """
    lines: list[str] = []
    for block in blocks:
        bid = str(block.get("_id", "?"))
        if "body" in block and isinstance(block["body"], str):
            text = block["body"]
        else:
            parts = [str(v) for k, v in block.items() if not k.startswith("_") and isinstance(v, (str, int, float))]
            text = " ".join(parts)
        lines.append(f"[{bid}] {text}")
    return "\n".join(lines)


# Preamble sentences a chat-tuned model tends to prepend before the actual
# summary. Stripped deterministically — a paraphrasing preamble that varies
# call to call would prevent the fixed point from ever being reached.
_PREAMBLE_RE = re.compile(
    r"^\s*(here'?s?\s+(is\s+)?the\s+.*?:|sure[,.]?\s*here'?s?\s+.*?:|summary:|tighter summary:)\s*",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", re.DOTALL)

# A leading reasoning block emitted by Qwen3-family models (`mind-mem:4b` is a
# Qwen3.5-4B fine-tune): `<think>...</think>` before the actual summary. It is
# not part of the summary and is the single biggest reason a compressing model
# fails to reach a byte fixed point — the block's whitespace/content varies pass
# to pass, so the same summary never compares equal to its predecessor. Strip it
# only at the very start (a `<think>` that appears mid-body is real content, not
# a reasoning wrapper) and non-greedily (`.*?`) so only the first block goes.
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _clean_response(raw: str) -> str:
    """Deterministically strip a reasoning block, markdown fences, and LLM preamble.

    Idempotent by construction (each sub-step only removes a fixed leading block,
    prefix, or fence wrapper) — cleaning an already-clean string is a no-op,
    which is required for the fixed point: `recompact_cluster` compares cleaned
    output to the previous cleaned output, not raw model text. The `<think>`
    strip runs first because the reasoning wrapper, when present, precedes any
    fence or preamble the summary itself might carry.
    """
    text = raw.strip()
    text = _THINK_RE.sub("", text).strip()
    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    return text


class OllamaCompressor:
    """Compressor backed by a local ollama model over HTTP.

    Calls ``POST {host}/api/generate`` with ``stream=false`` and
    ``options.temperature=0`` / ``options.seed={seed}`` pinned so repeated
    calls with identical inputs are byte-identical (see module docstring).
    Uses only ``urllib.request`` + ``json`` from the stdlib — no new
    dependency.
    """

    def __init__(
        self,
        model: str,
        host: str = _DEFAULT_HOST,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if temperature != 0.0:
            # Non-zero sampling temperature breaks purity w.r.t. inputs —
            # the whole fixed-point argument depends on temperature=0.
            raise ValueError("OllamaCompressor requires temperature=0.0 for a well-defined fixed point")
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._seed = seed
        self._timeout = timeout

    def __call__(self, current_text: str, blocks: list[dict[str, Any]]) -> str:
        prompt = _PROMPT_TEMPLATE.format(current=current_text, siblings=_render_siblings(blocks))
        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature, "seed": self._seed},
        }
        raw_response = self._post(body)
        return _clean_response(raw_response)

    def _post(self, body: dict[str, Any]) -> str:
        url = f"{self._host}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise CompressorError(f"ollama returned HTTP {status} for model {self._model!r}")
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise CompressorError(f"ollama request failed: {exc}") from exc
        except TimeoutError as exc:
            raise CompressorError(f"ollama request timed out after {self._timeout}s: {exc}") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CompressorError(f"ollama returned malformed json: {exc}") from exc

        if not isinstance(payload, dict) or "response" not in payload:
            raise CompressorError(f"ollama response missing 'response' field: {payload!r}")
        response_text = payload["response"]
        if not isinstance(response_text, str):
            raise CompressorError(f"ollama 'response' field is not a string: {type(response_text)!r}")
        return response_text


__all__ = ["CompressorError", "EchoCompressor", "OllamaCompressor"]
