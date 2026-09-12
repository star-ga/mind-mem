"""dry-run / enforce, so a correct validation rule can actually ship.

ROADMAP states the problem this solves better than a summary would: "every item
above describes a validator that *refuses* a non-conforming write. There is no
described way to turn it on. On a populated graph the first enforcing build rejects
some fraction of existing callers, so the rule either ships broken or never ships;
the predictable outcome is that the correct rule is written and left switched off."

The required shape, and the reason for each half:

* ``dry-run`` -- evaluate, record the decision and what it WOULD have refused,
  forward the write. A rule can therefore run against live ingest before anyone
  depends on it.
* ``enforce`` -- refuse.
* Both modes write the SAME decision row, distinguished by mode, "so a dry-run
  trail is directly comparable to what enforcement would produce; that
  comparability is the entire value." A dry-run row that omitted fields an
  enforcing row carries would make the comparison guesswork, which is why a test
  pins the shapes equal and the difference to exactly ``{mode, proceed}``.
* The mode is IN the row: "a validator whose mode is inferred rather than stated
  cannot be audited after the fact."

The names are ``dry-run`` and ``enforce`` deliberately -- the same two naestro R87
uses -- so a rule means the same thing in both stores. A third spelling here would
make cross-store comparison a translation exercise.

A BROKEN RULE FAILS CLOSED IN ENFORCE. If the rule raises, the property could not
be evaluated, and "could not evaluate" must not return what "evaluated clean"
returns -- the forgery-by-absence failure ``capabilities.py`` exists to prevent, in
its local form. In dry-run the write still forwards, because that is what dry-run
means, but the row records the failure so a promotion decision sees it.

Pure: no clock, no I/O. The caller owns persistence; this decides and describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: Evaluate and record, but forward the write.
DRY_RUN = "dry-run"

#: Refuse a non-conforming write.
ENFORCE = "enforce"

#: The closed set. Two modes, two spellings, no third.
MODES: frozenset[str] = frozenset({DRY_RUN, ENFORCE})

#: A rule takes the payload and returns a refusal reason, or "" to pass.
#: Returning a REASON rather than a bool is deliberate: a validator that only says
#: "no" produces a decision trail nobody can act on, and the whole point of
#: dry-run is reading the trail before promoting.
Rule = Callable[[Mapping[str, Any]], str]


class UnknownMode(ValueError):
    """A mode outside :data:`MODES`.

    Refused rather than defaulted, in both directions. Defaulting to ``dry-run``
    would silently stop enforcing a rule someone believed was on; defaulting to
    ``enforce`` would break live callers over a typo. Neither is a safe guess, so
    there is no guess.
    """


@dataclass(frozen=True)
class Decision:
    """One validation decision, in a shape both modes share."""

    mode: str
    proceed: bool
    would_refuse: bool
    reason: str

    def to_row(self) -> dict[str, Any]:
        """The decision as a flat row a caller can append to its own trail.

        Identical keys in both modes -- that identity is what makes a dry-run
        trail comparable to enforcement rather than merely suggestive of it.
        """
        return {
            "mode": self.mode,
            "proceed": self.proceed,
            "would_refuse": self.would_refuse,
            "reason": self.reason,
        }


def normalise_mode(raw: object) -> str:
    """The canonical mode name, or raise :class:`UnknownMode`.

    Accepts case and underscore spellings, because ``dry_run`` and ``dry-run`` are
    one mode and treating them as two would let a config typo silently disable a
    rule -- the failure this module exists to make impossible.
    """
    text = str(raw or "").strip().lower().replace("_", "-")
    if text not in MODES:
        raise UnknownMode(
            f"{raw!r} is not a validator mode. Legal modes: {sorted(MODES)}. "
            f"Refusing rather than defaulting: defaulting to {DRY_RUN!r} would "
            f"silently stop enforcing a rule someone believed was on, and "
            f"defaulting to {ENFORCE!r} would break live callers over a typo."
        )
    return text


def decide(rule: Rule, payload: Mapping[str, Any], *, mode: object) -> Decision:
    """Run *rule* over *payload* under *mode* and describe the outcome.

    In ``dry-run`` the write always proceeds, whatever the rule said, and the
    refusal it withheld is recorded. In ``enforce`` a refusal stops the write.

    A rule that RAISES is treated as a refusal: the property could not be
    evaluated, and an unevaluable property must not report what a clean one
    reports. Under ``dry-run`` the write still proceeds, since forwarding is what
    dry-run is for, but the row shows the failure.
    """
    canonical = normalise_mode(mode)
    try:
        reason = str(rule(payload) or "")
    except Exception as exc:                      # noqa: BLE001
        reason = f"rule error (treated as a refusal, not a pass): {exc}"
    would_refuse = bool(reason)
    proceed = (canonical == DRY_RUN) or not would_refuse
    return Decision(
        mode=canonical, proceed=proceed, would_refuse=would_refuse, reason=reason
    )


__all__ = [
    "DRY_RUN",
    "ENFORCE",
    "MODES",
    "Decision",
    "Rule",
    "UnknownMode",
    "decide",
    "normalise_mode",
]
