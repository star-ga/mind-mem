"""Deterministic tool-output summarizer (mind-mem §5 — context offload).

A single ``cargo test`` / ``pytest`` / build run dumps 10k–50k lines into an
agent's context window. This module compresses that raw text into a compact,
DETERMINISTIC summary (same input + same config version → byte-identical output —
NO LLM, pure pattern extraction).

LOAD-BEARING INVARIANT — the summary is a *bounded view*, the full text is what's
stored. Three bounds keep the summary small REGARDLESS of input shape:
  * per-line cap (a 10 MB minified line can't blow the summary — it's truncated
    with an explicit ``…[+N chars]`` marker);
  * failure-display cap (a log where every line matches ``error`` shows the first
    ``max_failures_shown`` with an explicit ``… +N more failures`` marker — the
    TRUE total is always reported, nothing is claimed hidden);
  * head/tail windows for context.

FAIL-SAFE — nothing is ever silently lost: the FULL text is always stored and
recallable by handle, and every display truncation is EXPLICIT and COUNTED (dropped
middle-line count, capped-failure count, per-line char count). A failure is never
claimed absent.

DETERMINISM — no clock, no RNG, no set-iteration order in the output. Lines are
emitted in file order; the config is VERSIONED (:data:`SUMMARIZER_VERSION`) so a
summary is reproducible given its version tag.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

# Bump when the summary FORMAT or the pattern/threshold config changes — the
# version is stamped into the summary so a byte-comparison is version-scoped.
SUMMARIZER_VERSION = 1

# Failure / error markers across cargo, rustc, pytest, generic tooling. A line
# matching ANY of these is a "failure line" — always surfaced (up to the display
# cap), never claimed absent.
_DEFAULT_FAILURE_PATTERNS = (
    r"\bFAILED\b",
    r"\bFAIL\b",
    r"error\[",  # rustc: error[E0499]
    r"\berror:\s",  # cargo/rustc/clang: "error: ..."
    r"\bError:\s",  # generic "Error: ..."
    r"panicked",  # rust panic
    r"assertion",  # assertion failures
    r"AssertionError",  # pytest
    r"\bException\b",
    r"Traceback \(most recent call last\)",
    r"=== .*failed",  # pytest footer: "=== 3 failed, 10 passed ==="
    r"test result: FAILED",  # cargo test footer
    r"\b\d+ failed\b",
    r"\berror\b.*\baborting\b",
    r"segmentation fault|SIGSEGV|SIGABRT",
)

_TALLY_PATTERN = (
    r"(test result:.*)"
    r"|(=+ .*(passed|failed|error).* =+)"  # pytest footer
    r"|(\d+ passed.*\d+ failed)"
    r"|(running \d+ tests?)"
)


@dataclass(frozen=True)
class SummarizerConfig:
    """Versioned, inspectable knobs — the summary is a pure function of (text, this).
    No autonomous reweighting; callers override explicitly."""

    head: int = 40
    tail: int = 60
    max_line_chars: int = 500  # per-line cap — bounds a giant single line
    max_failures_shown: int = 200  # display cap — full count still reported
    max_tallies: int = 40  # cap tally lines too (a log of all-tally lines must stay bounded)
    failure_patterns: tuple[str, ...] = _DEFAULT_FAILURE_PATTERNS
    version: int = SUMMARIZER_VERSION

    def config_hash(self) -> str:
        """Stable digest of the config — two summaries with the same hash used the
        same rules (part of the reproducibility contract)."""
        blob = "|".join(
            [
                str(self.version),
                str(self.head),
                str(self.tail),
                str(self.max_line_chars),
                str(self.max_failures_shown),
                str(self.max_tallies),
                *self.failure_patterns,
            ]
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


DEFAULT_CONFIG = SummarizerConfig()


@dataclass(frozen=True)
class ToolOutputSummary:
    """The compact, BOUNDED result kept in context (full text lives out-of-context)."""

    summary: str
    line_count: int
    byte_count: int
    failure_lines: int  # TRUE count of failure-matching lines (never capped)
    failures_shown: int  # how many were displayed (<= failure_lines)
    dropped_lines: int  # middle lines elided from the view (audited)
    config_hash: str = ""  # which rules produced this (reproducibility)
    truncated_lines: int = 0  # how many emitted lines were per-line-capped


def _cap_line(line: str, limit: int) -> tuple[str, bool]:
    """Truncate a single line to ``limit`` chars with an explicit marker."""
    if len(line) <= limit:
        return line, False
    return line[:limit] + f"…[+{len(line) - limit} chars]", True


def summarize(
    text: str,
    source: str = "",
    exit_code: int | None = None,
    *,
    config: SummarizerConfig | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> ToolOutputSummary:
    """Compress ``text`` to a deterministic, BOUNDED summary. Pure — no clock/RNG/LLM.

    ``head``/``tail`` are convenience overrides on the default config (kept for the
    original call signature); pass ``config`` for full control.
    """
    cfg = config or DEFAULT_CONFIG
    if head is not None or tail is not None:
        cfg = replace(cfg, head=head if head is not None else cfg.head, tail=tail if tail is not None else cfg.tail)
    failure_re = re.compile("|".join(cfg.failure_patterns), re.IGNORECASE)
    tally_re = re.compile(_TALLY_PATTERN, re.IGNORECASE)

    lines = text.splitlines()
    n = len(lines)
    byte_count = len(text.encode("utf-8"))

    # Classify once (single pass): failure indices (true, uncapped) + tally indices.
    failure_idx: list[int] = []
    tally_idx: list[int] = []
    for i, line in enumerate(lines):
        if failure_re.search(line):
            failure_idx.append(i)
        elif tally_re.search(line):
            tally_idx.append(i)

    # Indices kept in the VIEW: head + tail + up-to-cap failures + up-to-cap tallies.
    # BOTH failures and tallies are display-capped so the summary stays bounded
    # regardless of input shape (a log of 50k all-tally lines must not explode the
    # summary). The pass/fail FOOTER tally lands in the tail window, so it is always
    # kept even when the tally cap trims the (rare) middle tally lines.
    keep: set[int] = set()
    keep.update(range(min(cfg.head, n)))
    keep.update(range(max(0, n - cfg.tail), n))
    keep.update(tally_idx[: cfg.max_tallies])
    shown_failures = failure_idx[: cfg.max_failures_shown]
    keep.update(shown_failures)

    kept = sorted(keep)
    dropped = n - len(kept)

    out: list[str] = []
    out.append(
        f"# tool-output summary v{cfg.version}/{cfg.config_hash()} — "
        f"source: {source or '(unknown)'}" + (f" · exit={exit_code}" if exit_code is not None else "")
    )
    out.append(
        f"# {n} lines / {byte_count} bytes · {len(failure_idx)} failure line(s) "
        f"({len(shown_failures)} shown) · {dropped} middle line(s) elided"
    )
    out.append("#" + "-" * 60)

    truncated = 0
    prev = -1
    for i in kept:
        if i > prev + 1:
            out.append(f"    … [{i - prev - 1} line(s) elided] …")
        capped, was_cut = _cap_line(lines[i], cfg.max_line_chars)
        truncated += was_cut
        out.append(f"{i + 1:>7}: {capped}")
        prev = i

    if failure_idx:
        out.append("#" + "-" * 60)
        out.append(
            f"# FAILURES ({len(failure_idx)} total"
            + (f", showing first {len(shown_failures)}" if len(shown_failures) < len(failure_idx) else "")
            + "):"
        )
        for i in shown_failures:
            capped, _ = _cap_line(lines[i].strip(), cfg.max_line_chars)
            out.append(f"  L{i + 1}: {capped}")
        if len(shown_failures) < len(failure_idx):
            out.append(f"  … +{len(failure_idx) - len(shown_failures)} more failure line(s) — recall the handle for the full log")

    return ToolOutputSummary(
        summary="\n".join(out) + "\n",
        line_count=n,
        byte_count=byte_count,
        failure_lines=len(failure_idx),
        failures_shown=len(shown_failures),
        dropped_lines=dropped,
        config_hash=cfg.config_hash(),
        truncated_lines=truncated,
    )


def make_handle(text: str, source: str) -> str:
    """Content-addressed short handle for a stored output — deterministic
    (``to-`` + first 16 hex of sha256(source ‖ text)). Same output → same handle,
    so a re-run of an identical command reuses the same handle (idempotent store)."""
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return "to-" + h.hexdigest()[:16]
