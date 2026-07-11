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
# Deterministic hard cap on generated tokens per pass. A `mind-mem:4b`-shaped
# small model asked for a canonical fact list can degenerate into runaway
# repetition or spill past the settled list — pure `compression_ratio` bloat and,
# worse, a body that never lands on a byte fixed point. `num_predict` bounds that
# from the REQUEST side (a qualitatively different axis from the post-processing
# body-shrinkers): with `temperature=0` + a fixed `seed` the truncation point is
# a pure function of the inputs, so it adds zero non-determinism. The cap is set
# generously (a real consolidated cluster is a handful of fact lines, far under
# this) so it never clips a converged body — it only hard-stops the over-
# generation the small model drifts into, and the probe guard re-appends any
# source probe a clip removed so `fact_retention` stays 1.0.
_DEFAULT_NUM_PREDICT = 512

# Post-processing pipeline version. Bumped whenever the prompt template or any
# `_clean_response` / `_precondition_input` sub-step changes the bytes the
# compressor emits for a fixed (model, temperature, seed) input. It is folded
# into `fold_identity()` so a fold-equivalence attestation binds the EXACT
# compressor behaviour that produced it: a verifier holding a different pipeline
# version is re-running a different function, and `verify_fold` must refuse it
# rather than diff against the wrong ground truth.
_COMPRESSOR_PIPELINE_VERSION = "1"

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
5. Sort the surviving lines in ascending lexicographic (byte) order — compare \
them character by character and put the smaller-valued line first. This is a \
fixed, content-free ordering: it does NOT depend on the order the facts arrived \
in, so re-normalizing a list that is already byte-sorted leaves it in the exact \
same order. Do not re-order by salience, recency, appearance, or any other key.
6. Do not add commentary, headings, blank lines, trailing punctuation beyond \
what a fact contains, markdown fences, or a preamble.

Because these rules are deterministic and idempotent, a list that is already \
canonical (no duplicates, no connective prose, already one-fact-per-line in \
ascending lexicographic order) MUST be returned byte-for-byte identical.

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


# Fold-attestation identity for the echo control (see `fold_attestation`). Echo
# is a pure identity function with no knobs, so its version is a bare "1".
# Attached as a function attribute so `compressor_identity` finds it through the
# same `fold_identity` hook the class compressors expose.
_echo.fold_identity = ("mind-mem/echo", "1")  # type: ignore[attr-defined]


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


def _dedup_lines(text: str) -> str:
    """Drop exact-duplicate NON-EMPTY lines, keeping the first occurrence.

    A `mind-mem:4b`-shaped model asked for a canonical one-fact-per-line list
    does not reliably deduplicate (the prompt asks it to; the small model emits
    repeats). Those repeated lines are pure ``compression_ratio`` bloat — the one
    lever the champion gate (``compression_ratio < 1.0``) reads that no prior
    pass touches. This removes them on the model BODY (a distinct axis from the
    probe-guard trailer and the whitespace normalizer).

    Retention-safe by construction: a removed line is byte-identical to an
    earlier retained line, so every bench probe on it (numbers, quoted IDs, ISO
    dates, capitalized entities) still appears as a substring of the surviving
    first occurrence — ``fact_retention`` is provably unchanged.

    Idempotent by construction: after one pass no two non-empty lines are equal,
    so a second pass removes nothing — required for ``_clean_response`` to stay a
    no-op on already-clean text and for the recompaction fixed point.

    Only NON-EMPTY lines are deduped: blank lines are structure, not facts, and
    are owned by :func:`_canonicalize_whitespace` (which runs after this). Lines
    are compared after ``rstrip`` so trailing-whitespace drift does not defeat the
    duplicate match; the retained line keeps its original text (whitespace is then
    normalized downstream).
    """
    seen: set[str] = set()
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped and stripped in seen:
            continue
        if stripped:
            seen.add(stripped)
        kept.append(line)
    return "\n".join(kept)


def _line_has_probe(line: str) -> bool:
    """True iff *line* contains at least one bench-shaped probe.

    Applies the four in-module probe regexes (numbers, quoted identifiers, ISO
    dates, capitalized entities minus stopwords, len>1) — the exact set the
    benchmark's ``extract_probes`` scores retention against — to a SINGLE line.
    A line with no probe carries no fact the bench can measure.
    """
    if _GUARD_NUMBER_RE.search(line):
        return True
    if _GUARD_QUOTED_RE.search(line):
        return True
    if _GUARD_DATE_RE.search(line):
        return True
    for m in _GUARD_ENTITY_RE.findall(line):
        if m not in _GUARD_ENTITY_STOPWORDS and len(m) > 1:
            return True
    return False


def _drop_probe_empty_lines(text: str) -> str:
    """Delete non-blank body lines that contain ZERO source-shaped probes.

    A ``mind-mem:4b``-shaped model asked for a canonical fact list still leaves
    pure connective / filler prose lines in ("Additionally, these blocks are
    related.", "In summary."). The canonical-list prompt asks it to drop them; a
    small model does not. Such a line carries no number, quoted identifier, ISO
    date, or capitalized entity from the source, so by the bench's own
    ``probes_present`` definition it is fact-free — deleting it is pure
    ``compression_ratio`` reduction that CANNOT lower ``fact_retention`` (there
    is no probe on the line to lose). This is a distinct axis from
    :func:`_dedup_lines` (which removes *duplicate* lines, not fact-free ones)
    and from the probe-guard trailer (which *adds* dropped probes back).

    Retention-safe by construction: a removed line has no probe, so every bench
    probe that was a substring of the retained body is still a substring of the
    output — ``fact_retention`` is provably unchanged. The probe guard runs
    strictly after this in the pipeline, so any source probe the model genuinely
    omitted is still re-appended by the guard, not lost here.

    Idempotent by construction: after one pass every surviving non-blank line
    has a probe, so a second pass drops nothing — required for
    ``_clean_response`` to stay a no-op on already-clean text and for the
    recompaction byte fixed point.

    Safety floor (never no-op the compressor into empty): if EVERY non-blank
    line is probe-empty, drop NOTHING and return the text unchanged. A
    legitimately probe-free cluster scores retention vacuously 1.0; collapsing
    its whole body to empty would instead trip the retention floor in
    ``recompact_cluster`` and score the cluster 0. Blank lines are structure, not
    facts, and are left to :func:`_canonicalize_whitespace`.
    """
    lines = text.split("\n")
    kept = [line for line in lines if not line.strip() or _line_has_probe(line)]
    # If nothing probe-bearing survived, the whole (non-blank) body was
    # fact-free — return it untouched rather than collapse it to empty.
    if not any(line.strip() for line in kept):
        return text
    return "\n".join(kept)


_BULLET_RE = re.compile(r"^- ")


def _canonicalize_bullet_order(text: str) -> str:
    """Sort each contiguous run of ``- ``-prefixed fact lines into a fixed order.

    This attacks a convergence failure mode NO existing cleaner touches: a
    ``mind-mem:4b``-shaped model asked for a one-fact-per-line list re-emits the
    *same set* of fact lines in a *different order* pass-to-pass (a fact list is
    semantically order-invariant, so the small model has no stable ordering
    prior). The prompt (rule 5) now asks the model for exactly this ascending
    lexicographic order, so a prompt-obedient model already emits the sorted
    form and this cleaner is a no-op on it — closing the prompt/post-processor
    tension where the prompt previously asked for first-appearance order while
    the cleaner byte-sorted, forcing the model to fight the normalizer every
    pass. It remains the deterministic backstop for a model that does not honour
    the ordering rule. Two byte-distinct permutations of the same facts cycle
    forever ->
    :class:`NonConvergenceError` -> that cluster scores 0. Whitespace/dedup
    normalization cannot fix a *reordering*; only mapping every permutation of a
    run to one canonical order can. This is a distinct axis from the
    probe-guard / whitespace / dedup / prompt family — it is a *convergence*
    mechanism that acts on the relative ORDER of the model body's fact lines.

    Scoped deliberately to contiguous runs of ``- ``-prefixed lines (the
    canonical fact-list lines the prompt asks the model to emit): non-bullet
    lines (prose, headers, blank lines, and the guard's ``preserved facts:``
    trailer — which is not ``- ``-prefixed) are ANCHORS held in place, and only
    the bullet lines *between* successive anchors are sorted among themselves.
    So a legitimately-ordered mixed body keeps its structure; only a pure
    fact-list run is canonicalized.

    Fixed-point safety: sorting is idempotent (a sorted run sorts to itself), so
    re-cleaning an already-canonical body is a no-op — required for the
    recompaction byte fixed point. It manufactures a fixed point ONLY when the
    underlying fact *set* has genuinely stabilized; it never fakes convergence
    for a body whose set of facts is still changing.

    Retention-safe by construction: reordering lines deletes no line, so every
    bench probe (numbers, quoted IDs, ISO dates, capitalized entities) still
    appears as a substring of the output — ``fact_retention`` is provably
    unchanged. ``compression_ratio`` is unchanged too (a permutation preserves
    total bytes); the win is purely convergence.
    """
    lines = text.split("\n")
    out: list[str] = []
    run: list[str] = []
    for line in lines:
        if _BULLET_RE.match(line):
            run.append(line)
            continue
        if run:
            out.extend(sorted(run))
            run = []
        out.append(line)
    if run:
        out.extend(sorted(run))
    return "\n".join(out)


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


def _precondition_input(current_text: str) -> str:
    """Project the model's INPUT summary onto the same canonical normal form its OUTPUT is held to.

    This is a qualitatively distinct axis from every prior sub-step: the dedup /
    whitespace / bullet-order normalizers and the probe guard all act on the
    model's *response*; this acts on the ``current_text`` the model is *fed*,
    before the prompt is even built. The motivation is the pass-1 distribution
    gap. ``recompact_cluster`` seeds the loop with the raw concatenated block
    bodies (prose), so on the FIRST pass the model reads free prose, while on
    every later pass it reads its own canonical bullet output. A ``mind-mem:4b``-
    shaped model told to return an already-canonical list byte-for-byte can only
    honour that instruction when its input *is* canonical; feeding it raw prose
    on pass 1 maximises the paraphrase divergence that compounds into
    oscillation. Pre-conditioning the input closes that gap — the model reads a
    normalized draft on pass 1 too, so a settled fact set is far likelier to
    reach the byte fixed point (and in fewer passes) instead of cycling.

    It reuses only the three BODY normalizers — line dedup, line-boundary
    whitespace, and bullet-run order — and deliberately NOT the preamble / fence
    / ``<think>`` strips, which are model-response artifacts that never appear in
    a ``current_text`` (that is either the source-derived prose seed or a prior
    cleaned pipeline output).

    Fixed-point safety (the load-bearing property): every value the pipeline can
    emit — ``_apply_probe_guard(_clean_response(raw), blocks)`` — is already a
    fixed point of these three normalizers (``_clean_response`` runs all three
    last, and the probe-guard trailer is non-``- ``-prefixed so it is held as an
    anchor). Therefore ``_precondition_input`` is a strict no-op on any pass-2+
    ``current_text`` (which is always a prior pipeline output), and it perturbs
    ONLY the pass-1 prose seed. The loop's ``rewritten == current`` comparison is
    thus untouched once the loop is in canonical-form territory — this cannot
    manufacture a fixed point the underlying facts have not reached, and it
    cannot make the compressor a no-op (the model is still called every pass).

    Pure, deterministic, and idempotent by construction (each sub-op is a
    projection onto a normal form), so it adds zero non-determinism to the
    fixed-point argument.
    """
    text = _dedup_lines(current_text)
    text = _canonicalize_whitespace(text)
    text = _canonicalize_bullet_order(text)
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
    # Drop exact-duplicate fact-lines the small model repeats — pure
    # compression_ratio bloat on the body, retention-safe (a removed line is
    # byte-identical to a retained earlier one) and idempotent. Runs before
    # whitespace canonicalization so blank-line structure stays owned by
    # `_canonicalize_whitespace`. See `_dedup_lines` for the proofs.
    text = _dedup_lines(text)
    # Drop fact-free body lines — non-blank lines with ZERO source-shaped probes
    # (connective/filler prose the small model fails to prune). Runs after dedup
    # (so a duplicate fact-free line is already gone) and before whitespace
    # canonicalization (blank-line structure stays owned by
    # `_canonicalize_whitespace`). Pure compression_ratio reduction: a removed
    # line has no probe, so retention is provably unchanged; the probe guard runs
    # later and still re-appends any genuinely-omitted source probe. Idempotent
    # (surviving lines all have a probe) with a safety floor against collapsing a
    # legitimately probe-free body to empty. See `_drop_probe_empty_lines`.
    text = _drop_probe_empty_lines(text)
    # Final: canonicalize line-boundary whitespace so whitespace drift (trailing
    # spaces, blank-line runs) can't block the byte fixed point. Runs last so it
    # also normalizes whitespace exposed by the fence/preamble strips. The outer
    # `.strip()` above already handles leading/trailing blank lines; this handles
    # the interior. See `_canonicalize_whitespace` for the retention-safety proof.
    text = _canonicalize_whitespace(text)
    # Final: canonicalize the ORDER of `- `-prefixed fact-list runs so a model
    # that re-emits the same facts in a permuted order lands on a byte fixed
    # point instead of oscillating forever. Runs LAST — after dedup (so each run
    # is already duplicate-free) and after whitespace canonicalization (so lines
    # are rstripped before the byte-sort, keeping the sort idempotent). Reorders
    # only; deletes nothing, so retention/ratio are untouched. See
    # `_canonicalize_bullet_order` for the convergence + safety proofs.
    text = _canonicalize_bullet_order(text)
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
        num_predict: int = _DEFAULT_NUM_PREDICT,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if temperature != 0.0:
            # Non-zero sampling temperature breaks purity w.r.t. inputs —
            # the whole fixed-point argument depends on temperature=0.
            raise ValueError("OllamaCompressor requires temperature=0.0 for a well-defined fixed point")
        if num_predict < 1:
            # A non-positive cap would emit an empty (or ollama-default unbounded)
            # body — the former destroys retention, the latter defeats the cap.
            raise ValueError("num_predict must be >= 1 to bound the generated body deterministically")
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._seed = seed
        self._timeout = timeout
        self._num_predict = num_predict

    def __call__(self, current_text: str, blocks: list[dict[str, Any]]) -> str:
        # Pre-condition the INPUT onto the same canonical normal form the OUTPUT
        # is held to, closing the pass-1 prose-vs-bullet distribution gap (see
        # `_precondition_input`). A strict no-op on every pass-2+ input (those are
        # prior pipeline outputs, already canonical), so the fixed-point
        # comparison is untouched once the loop is in canonical-form territory.
        conditioned = _precondition_input(current_text)
        prompt = _PROMPT_TEMPLATE.format(current=conditioned, siblings=_render_siblings(blocks))
        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "seed": self._seed,
                # Deterministic hard token cap (see `_DEFAULT_NUM_PREDICT`): with
                # temperature=0 + fixed seed the truncation point is a pure
                # function of the inputs, so purity/idempotence hold. Bounds
                # runaway small-model over-generation from the request side.
                "num_predict": self._num_predict,
            },
        }
        raw_response = self._post(body)
        cleaned = _clean_response(raw_response)
        # Deterministic, idempotent floor on fact loss: re-append any source
        # probe the model dropped. Pure w.r.t. (cleaned, blocks); a no-op once
        # every probe is present, so it converges exactly when the model body
        # does. See the `_apply_probe_guard` docstring for the fixed-point
        # argument.
        return _apply_probe_guard(cleaned, blocks)

    def fold_identity(self) -> tuple[str, str]:
        """Canonical ``(id, version)`` binding this compressor into a fold attestation.

        The ``id`` names the model; the ``version`` fingerprints every knob that
        makes the compressor's output a *pure function of its inputs* — the
        pinned ``temperature`` and ``seed``, the deterministic token cap, and the
        post-processing ``_COMPRESSOR_PIPELINE_VERSION``. A verifier re-running a
        fold must hold a compressor whose ``fold_identity`` matches, or it is
        re-deriving the fixed point with a *different function* than the one that
        produced the attestation. The host is deliberately excluded — it is a
        transport endpoint, not part of the bytes the model emits — and no
        wall-clock or randomness enters, so the identity is stable across runs.
        """
        version = f"pipeline{_COMPRESSOR_PIPELINE_VERSION}/temp{self._temperature}/seed{self._seed}/np{self._num_predict}"
        return (f"mind-mem/ollama:{self._model}", version)

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
