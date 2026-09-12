"""dry-run / enforce validator modes — so a correct rule can actually ship.

ROADMAP (Group K, the validator-mode requirement): "every item above describes a
validator that *refuses* a non-conforming write. There is no described way to turn
it on. On a populated graph the first enforcing build rejects some fraction of
existing callers, so the rule either ships broken or never ships; the predictable
outcome is that the correct rule is written and left switched off."

The required shape, verbatim: "two modes from the start: **dry-run** (evaluate,
record the decision and what it *would* have refused, forward the write) and
**enforce** (refuse) -- the same two names naestro R87 uses, deliberately, so a rule
means the same thing in both stores. Both modes write the same decision row,
distinguished by mode, so a dry-run trail is directly comparable to what
enforcement would produce; that comparability is the entire value."

And: "The mode in force is itself part of the record -- a validator whose mode is
inferred rather than stated cannot be audited after the fact."

So the tests below pin three things that are easy to get subtly wrong:
  1. dry-run FORWARDS the write while recording the refusal it withheld;
  2. both modes produce the SAME decision shape, differing only in `mode` and
     whether the write proceeded -- without that, the trail is not comparable and
     the mechanism has no value;
  3. the mode is IN the row, never inferred from context.
"""

from __future__ import annotations

import pytest

from mind_mem.validator_mode import (
    DRY_RUN,
    ENFORCE,
    UnknownMode,
    decide,
    normalise_mode,
)


def _rule_refuses(_payload):
    return "statement is empty"


def _rule_passes(_payload):
    return ""


# --------------------------------------------------------------------------
# The two names, and only those two
# --------------------------------------------------------------------------

def test_the_two_modes_are_the_names_naestro_uses():
    assert (DRY_RUN, ENFORCE) == ("dry-run", "enforce")


def test_an_unknown_mode_is_refused_not_defaulted():
    """A validator defaulting to dry-run would silently stop enforcing; one
    defaulting to enforce would break callers on a typo. Refuse instead."""
    with pytest.raises(UnknownMode):
        normalise_mode("enforcing")


def test_case_and_underscore_spellings_normalise():
    for raw in ("DRY-RUN", "dry_run", " dry-run "):
        assert normalise_mode(raw) == DRY_RUN


# --------------------------------------------------------------------------
# dry-run forwards but records
# --------------------------------------------------------------------------

def test_dry_run_forwards_a_write_the_rule_would_refuse():
    d = decide(_rule_refuses, {"statement": ""}, mode=DRY_RUN)
    assert d.proceed is True, "dry-run must forward the write"
    assert d.would_refuse is True, "and must record that it WOULD have refused"
    assert d.reason == "statement is empty"


def test_enforce_refuses_the_same_write():
    d = decide(_rule_refuses, {"statement": ""}, mode=ENFORCE)
    assert d.proceed is False
    assert d.would_refuse is True
    assert d.reason == "statement is empty"


def test_a_passing_write_proceeds_in_both_modes():
    """POSITIVE CONTROL. A validator that refused everything would pass the two
    tests above while making the store unusable."""
    for mode in (DRY_RUN, ENFORCE):
        d = decide(_rule_passes, {"statement": "fine"}, mode=mode)
        assert d.proceed is True and d.would_refuse is False, mode


# --------------------------------------------------------------------------
# Comparability — "that comparability is the entire value"
# --------------------------------------------------------------------------

def test_both_modes_emit_the_same_row_shape():
    a = decide(_rule_refuses, {"statement": ""}, mode=DRY_RUN).to_row()
    b = decide(_rule_refuses, {"statement": ""}, mode=ENFORCE).to_row()
    assert set(a) == set(b), (sorted(a), sorted(b))


def test_the_rows_differ_only_in_mode_and_whether_it_proceeded():
    a = decide(_rule_refuses, {"statement": ""}, mode=DRY_RUN).to_row()
    b = decide(_rule_refuses, {"statement": ""}, mode=ENFORCE).to_row()
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"mode", "proceed"}, differing


def test_the_mode_is_in_the_row_not_inferred():
    """'A validator whose mode is inferred rather than stated cannot be audited.'"""
    row = decide(_rule_refuses, {"statement": ""}, mode=DRY_RUN).to_row()
    assert row["mode"] == DRY_RUN


def test_a_rule_that_raises_fails_CLOSED_in_enforce_and_is_recorded_in_dry_run():
    """A broken rule must not become a silent pass.

    In enforce, an exception means the property could not be evaluated, so the
    write is refused. In dry-run the write still forwards -- that is what dry-run
    IS -- but the row records the failure so a promotion decision can see it.
    """
    def _boom(_payload):
        raise RuntimeError("rule blew up")

    enforced = decide(_boom, {}, mode=ENFORCE)
    assert enforced.proceed is False
    assert "blew up" in enforced.reason or "error" in enforced.reason.lower()

    dry = decide(_boom, {}, mode=DRY_RUN)
    assert dry.proceed is True
    assert dry.would_refuse is True
