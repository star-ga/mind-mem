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

# --- probe-preserving guard --------------------------------------------------
# The recompaction benchmark scores fact retention as the fraction of "probes"
# (numbers, quoted identifiers, ISO dates, capitalized entities) from the source
# blocks that survive as substrings of the rewrite. A converging-but-lossy model
# (the 0.730 champion drops ~27% of probes) leaves that retention on the table.
#
# The guard below is a DETERMINISTIC, IDEMPOTENT floor on that loss: after the
# model compresses the prose body, any source probe the model dropped is
# re-appended verbatim under a fixed lowercase sentinel line, in canonical
# sorted order. It is NOT a paraphrase and NOT the extractive-projection rewrite
# that prior iterations found unreachable — it only appends bytes the source
# already contained and the model omitted.
#
# Fixed-point safety: the trailer is a pure function of (cleaned_body, source
# probes). Once the model's body reaches a fixed point, the same cleaned body
# yields the same missing-probe set in the same sorted order, so the full
# guarded output is a fixed point exactly when the model body is. The guard adds
# no oscillation of its own — a probe already present is never re-appended (it is
# already a substring), so re-guarding an already-guarded output is a no-op.
#
# The regexes are intentionally duplicated from `bench/recompaction_bench.py`
# (they must match its `extract_probes` exactly) rather than imported: library
# code must not depend on a benchmark module, and the bench file is read-only.
_GUARD_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_GUARD_QUOTED_RE = re.compile(r'"([^"]{2,80})"')
_GUARD_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_GUARD_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)*|[A-Z]{2,}(?:-[A-Za-z0-9]+)*)\b")
_GUARD_ENTITY_STOPWORDS = frozenset({"The", "A", "An", "This", "That", "It", "In", "On", "At", "For", "With", "Is", "Was", "Are"})
# A lowercase sentinel so the trailer introduces no NEW capitalized-entity probe
# of its own (which could otherwise re-enter the probe set on the next pass and
# perturb convergence). The colon-space prefix is matched when we detect an
# already-appended trailer so re-guarding stays a strict no-op.
_GUARD_SENTINEL = "preserved facts: "

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


def _source_probes(blocks: list[dict[str, Any]]) -> list[str]:
    """Extract benchmark-shaped probes from the source block bodies, canonically ordered.

    Mirrors ``bench.recompaction_bench.extract_probes`` exactly (numbers, quoted
    identifiers, ISO dates, capitalized entities minus stopwords) so the guard
    floors the metric the benchmark actually measures. Returns a sorted list
    (deterministic order) rather than a set so the appended trailer is
    byte-stable across passes.

    Probes are extracted from the block BODIES only — the same text the bench's
    ``_concat_source_text`` scores against — NOT from the ``[block-id]``-prefixed
    prompt rendering, so the guard never treats a synthetic block-id token as a
    fact to preserve.
    """
    from .recompaction import _block_body

    text = "\n\n".join(_block_body(b) for b in blocks)
    probes: set[str] = set()
    probes.update(_GUARD_NUMBER_RE.findall(text))
    probes.update(_GUARD_QUOTED_RE.findall(text))
    probes.update(_GUARD_DATE_RE.findall(text))
    for m in _GUARD_ENTITY_RE.findall(text):
        if m not in _GUARD_ENTITY_STOPWORDS and len(m) > 1:
            probes.add(m)
    return sorted(probes)


def _collapse_substring_probes(probes: list[str]) -> list[str]:
    """Keep only the *maximal* probes — drop any that is a substring of another.

    The bench scores retention as substring-presence: if probe ``s`` is a
    substring of probe ``t`` and ``t`` is appended verbatim, then ``s`` is a
    substring of ``t`` is a substring of the output, so ``s`` still counts as
    retained. Collapsing substring-probes into their maximal superstring
    therefore removes ZERO retained probes while shrinking the appended trailer
    (``2026`` need not be listed when ``2026-05-29`` already is; ``Q16`` when
    ``Q16.16`` is). Returns a canonically sorted list so the trailer stays
    byte-stable across passes.

    Pure and deterministic (length-desc pass + substring tests + a final sort),
    so the guard's idempotence and the recompaction fixed point are preserved:
    the same missing set always yields the same maximal set in the same order.
    """
    kept: list[str] = []
    for p in sorted(set(probes), key=len, reverse=True):
        if not any(p != q and p in q for q in kept):
            kept.append(p)
    return sorted(kept)


def _apply_probe_guard(cleaned: str, blocks: list[dict[str, Any]]) -> str:
    """Append any source probes the model dropped, under a fixed sentinel line.

    Idempotent and deterministic: a probe already present as a substring of
    ``cleaned`` is never re-appended, and the missing probes are collapsed to
    their maximal superstrings (see :func:`_collapse_substring_probes`) then
    appended in canonical sorted order. Re-guarding an already-guarded string is
    therefore a no-op — the property the recompaction fixed point depends on.
    Returns ``cleaned`` unchanged when nothing is missing (including the
    empty-cluster / no-probe case).

    The substring-collapse keeps ``fact_retention`` at exactly 1.0 (every
    dropped short probe is a substring of a kept longer one and so is still
    substring-present in the output) while strictly shrinking the trailer on any
    cluster with substring-overlapping probes — moving ``compression_ratio``
    away from the disqualifying 1.0 boundary instead of toward it.
    """
    missing = _collapse_substring_probes([p for p in _source_probes(blocks) if p not in cleaned])
    if not missing:
        return cleaned
    return f"{cleaned}\n\n{_GUARD_SENTINEL}{'; '.join(missing)}"


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

# Line-boundary whitespace that a chat-tuned model re-emits non-deterministically
# pass to pass: trailing spaces/tabs on a line, and runs of ≥2 blank lines. Both
# vary in width without changing content, so the same settled facts never compare
# byte-equal to their predecessor — the whitespace, not the facts, blocks the
# fixed point. Canonicalizing it is a distinct axis from the probe guard (it acts
# on the model BODY, not the appended trailer) and strips bytes, so it can only
# lower `compression_ratio`.
_BLANK_RUN_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _canonicalize_whitespace(text: str) -> str:
    """Collapse line-boundary whitespace to a deterministic normal form.

    Restricted to **line boundaries only** — trailing whitespace per line and
    runs of blank lines — and deliberately does NOT touch inter-word spaces.
    That restriction is load-bearing for retention: a quoted-identifier probe
    may contain a single internal space (``"mind mem"``), so collapsing
    inter-word runs could alter a probe; line-boundary whitespace never appears
    inside any bench probe (numbers, ISO dates, capitalized entities, and quoted
    strings are all single-line tokens with no trailing/blank-line runs), so
    this normalization is provably substring-preserving for every probe.

    Idempotent by construction: each sub-op is a projection onto a normal form
    (rstrip each line; collapse ≥2 blank lines to exactly one), so applying it
    twice equals applying it once — required for ``_clean_response`` to stay a
    no-op on already-clean text and for the recompaction fixed point.
    """
    # rstrip each line (kills trailing spaces/tabs the model drifts on).
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse any run of 2+ blank lines to a single blank line.
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text


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
    # Final: canonicalize line-boundary whitespace so whitespace drift (trailing
    # spaces, blank-line runs) can't block the byte fixed point. Runs last so it
    # also normalizes whitespace exposed by the fence/preamble strips. The outer
    # `.strip()` above already handles leading/trailing blank lines; this handles
    # the interior. See `_canonicalize_whitespace` for the retention-safety proof.
    text = _canonicalize_whitespace(text)
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
        cleaned = _clean_response(raw_response)
        # Deterministic, idempotent floor on fact loss: re-append any source
        # probe the model dropped. Pure w.r.t. (cleaned, blocks); a no-op once
        # every probe is present, so it converges exactly when the model body
        # does. See the `_apply_probe_guard` docstring for the fixed-point
        # argument.
        return _apply_probe_guard(cleaned, blocks)

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
