# Copyright 2026 STARGA, Inc.
"""The pre-write redaction pass: detect, rewrite, and say what changed.

Four modes, because "redact" is three different operator intents wearing
one word:

``off``
    The chain does not run. Not "runs and finds nothing" — *does not
    run*: no detector is instantiated, no pass is made over the text,
    nothing is hashed. A build with the flag off must be
    indistinguishable from the build that never had the feature, and a
    silent-but-still-scanning OFF path is a cost the unwired build never
    paid.
``flag``
    Report findings; change nothing. The mode for measuring a corpus
    before deciding what to do about it.
``redact``
    Replace each finding with ``[REDACTED:<detector>]``.
``reject``
    Refuse the write. For the workspaces where a secret reaching disk at
    all is the incident, and a rewritten copy is not a remedy.

The result carries SHA-256 of the input and of the output. Those two
digests are the whole provenance of the operation: they let an auditor
confirm *which* document was redacted and that a given output is the one
that was produced, while carrying nothing that could reconstruct what
was removed. Per-value hashes are deliberately absent — see
:mod:`mind_mem.compliance.detectors`.

Deterministic by construction: the detector chain is name-ordered, the
findings are canonically sorted and de-overlapped, and the replacement
token is a function of the detector alone. The same text redacts to the
same bytes on every run and every machine, which is what lets the export
bundle claim byte-identity.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from ..v4.feature_flags import flag_config_for_workspace, is_enabled_for_workspace
from .detectors import Detector, Finding, resolve_detectors, scan_text

__all__ = [
    "MODE_FLAG",
    "MODE_OFF",
    "MODE_REDACT",
    "MODE_REJECT",
    "REDACTION_FLAG",
    "RedactionConfigError",
    "RedactionRefused",
    "RedactionResult",
    "redact",
    "redaction_chain_for_workspace",
    "resolve_mode",
]

#: The declared feature flag this module is the consumer of. Wiring a
#: consumer is what moves the flag out of ``UNIMPLEMENTED`` in
#: :mod:`mind_mem.v4.flag_registry`; ``tests/test_flag_registry.py``
#: fails if the two ever disagree again.
REDACTION_FLAG = "redaction"

MODE_OFF = "off"
MODE_FLAG = "flag"
MODE_REDACT = "redact"
MODE_REJECT = "reject"

#: Configurable modes. ``off`` is not here: off is the absence of the
#: flag, not a value an operator sets, so there is one way to be off.
_SETTABLE_MODES = (MODE_FLAG, MODE_REDACT, MODE_REJECT)


class RedactionConfigError(ValueError):
    """The workspace asks for a redaction mode or detector that does not exist."""


class RedactionRefused(RuntimeError):
    """``reject`` mode: the text carries something that must not be written.

    Carries the :class:`RedactionResult` so a caller can report *what
    kind* of thing was found and where, without ever holding the value.
    """

    def __init__(self, result: "RedactionResult") -> None:
        kinds = ", ".join(sorted({f.detector for f in result.findings}))
        super().__init__(f"redaction policy 'reject': {len(result.findings)} finding(s) [{kinds}]; write refused")
        self.result = result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RedactionResult:
    """What the pass did, in a shape safe to write to a ledger."""

    mode: str
    text: str
    original_sha256: str
    redacted_sha256: str
    findings: tuple[Finding, ...] = ()
    detectors: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.original_sha256 != self.redacted_sha256

    @property
    def counts(self) -> dict[str, int]:
        """``detector -> hits``, sorted, for a one-line summary."""
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.detector] = out.get(finding.detector, 0) + 1
        return {name: out[name] for name in sorted(out)}

    def summary(self) -> str:
        """The human-readable line an audit entry carries as its reason."""
        if not self.findings:
            return f"redaction[{self.mode}]: no findings"
        kinds = ", ".join(f"{name} x{count}" for name, count in self.counts.items())
        digests = f"{self.original_sha256[:12]}->{self.redacted_sha256[:12]}"
        return f"redaction[{self.mode}]: {len(self.findings)} finding(s) {kinds}; sha256 {digests}"

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "detectors": list(self.detectors),
            "finding_count": len(self.findings),
            "counts": self.counts,
            "changed": self.changed,
            "original_sha256": self.original_sha256,
            "redacted_sha256": self.redacted_sha256,
            "findings": [f.to_dict() for f in self.findings],
        }


def resolve_mode(workspace: str) -> str:
    """The workspace's redaction mode, or :data:`MODE_OFF`.

    One probe, resolved by the caller **once per command** rather than
    once per block: the probe reads a config file, and a per-item read on
    an off path is a cost the unwired build never paid.
    """
    if not is_enabled_for_workspace(workspace, REDACTION_FLAG):
        return MODE_OFF
    raw = flag_config_for_workspace(workspace, REDACTION_FLAG).get("mode", MODE_REDACT)
    if raw not in _SETTABLE_MODES:
        raise RedactionConfigError(f"v4.redaction.mode is {raw!r}; expected one of {list(_SETTABLE_MODES)}")
    return str(raw)


def redaction_chain_for_workspace(workspace: str) -> tuple[Detector, ...]:
    """The detectors this workspace runs: its declared subset, or all of them.

    A name the registry does not know is a refusal, never a quietly
    shorter chain — the failure mode this whole module exists to prevent
    is a detector that is not running while everyone believes it is.
    """
    raw = flag_config_for_workspace(workspace, REDACTION_FLAG).get("detectors")
    if raw is None:
        return resolve_detectors(None)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RedactionConfigError("v4.redaction.detectors must be a list of detector names")
    try:
        return resolve_detectors([str(x) for x in raw])
    except KeyError as exc:
        raise RedactionConfigError(str(exc)) from None


def redact(text: str, *, mode: str, detectors: Sequence[Detector] | None = None) -> RedactionResult:
    """Run the chain over *text* under *mode*.

    ``off`` returns immediately with the text unchanged and no digests
    computed, so the disabled path costs one comparison.
    """
    if mode == MODE_OFF:
        return RedactionResult(mode=MODE_OFF, text=text, original_sha256="", redacted_sha256="", findings=(), detectors=())
    if mode not in _SETTABLE_MODES:
        raise RedactionConfigError(f"unknown redaction mode {mode!r}; expected one of {[MODE_OFF, *_SETTABLE_MODES]}")

    chain = resolve_detectors(None) if detectors is None else tuple(detectors)
    findings = tuple(scan_text(text, chain))
    original = _sha256(text)
    names = tuple(d.name for d in chain)

    unchanged = RedactionResult(
        mode=mode, text=text, original_sha256=original, redacted_sha256=original, findings=findings, detectors=names
    )
    if mode == MODE_REJECT:
        # A clean document is not refused: reject is a policy about
        # findings, not a policy about running.
        if findings:
            raise RedactionRefused(unchanged)
        return unchanged
    if mode == MODE_FLAG or not findings:
        return unchanged

    pieces: list[str] = []
    cursor = 0
    for finding in findings:
        pieces.append(text[cursor : finding.start])
        pieces.append(f"[REDACTED:{finding.detector}]")
        cursor = finding.end
    pieces.append(text[cursor:])
    redacted = "".join(pieces)
    return RedactionResult(
        mode=mode,
        text=redacted,
        original_sha256=original,
        redacted_sha256=_sha256(redacted),
        findings=findings,
        detectors=names,
    )
