"""M5 — fail-closed capability flags, so an unenforced property cannot pass.

ROADMAP M5's close condition is explicit that a document is not enough: "Each
audit finding should close by installing a flag that FAILS CLOSED while
unimplemented, so an unenforced property is a function returning 0 that gates the
path -- not a doc saying the property is aspirational. A caller intending to rely
on it must check and refuse."

The pattern it names is internal: 512-mind ships
`drift.semantic_mutation_scan_supported() -> u8 { 0 }`, with the reasoning in the
source -- "An undefined/empty mutation list must NEVER make `equivalent` true --
that was the forgery-by-absence path this fix closes."

FORGERY BY ABSENCE is the failure this module exists to prevent, and it is the same
failure this whole session has been closing: a check that cannot run returning the
same value as a check that passed. Here the unsupported answer is a distinct third
state that no supported path can produce, so "we never verified this" can never be
read as "this verified clean".
"""

from __future__ import annotations

import pytest

from mind_mem.capabilities import (
    Capability,
    CapabilityUnsupported,
    require,
    supported,
    verdict_for,
)


def test_every_capability_reports_a_bool():
    for cap in Capability:
        assert isinstance(supported(cap), bool), cap


def test_an_unsupported_capability_is_not_silently_true():
    """The whole point: unimplemented must never read as enforced."""
    unsupported = [c for c in Capability if not supported(c)]
    assert unsupported, (
        "every capability reports supported -- if that is genuinely true, this test "
        "should be replaced with per-capability enforcement tests rather than "
        "deleted, or the flag mechanism stops being able to fail"
    )


def test_require_refuses_an_unsupported_capability():
    unsupported = next((c for c in Capability if not supported(c)), None)
    if unsupported is None:
        pytest.skip("nothing unsupported to assert against")
    with pytest.raises(CapabilityUnsupported):
        require(unsupported)


def test_require_passes_a_supported_capability():
    """POSITIVE CONTROL. A require() that always raised would pass the test above
    while making every gated path permanently dead."""
    ok = next((c for c in Capability if supported(c)), None)
    assert ok is not None, "no capability is supported; the flag table is inert"
    require(ok)  # must not raise


def test_the_refusal_names_the_capability_and_what_it_means():
    unsupported = next((c for c in Capability if not supported(c)), None)
    if unsupported is None:
        pytest.skip("nothing unsupported to assert against")
    with pytest.raises(CapabilityUnsupported) as e:
        require(unsupported)
    msg = str(e.value)
    assert unsupported.value in msg, msg
    assert "not" in msg.lower()


# --------------------------------------------------------------------------
# The third state: UNSUPPORTED is not PASS and not FAIL
# --------------------------------------------------------------------------

def test_an_unsupported_verdict_is_distinguishable_from_a_pass():
    """Forgery by absence, closed structurally.

    If verdict_for returned the same value for "unsupported" and "verified clean",
    a caller could not tell them apart -- which is exactly how an unenforced
    property survives unnoticed.
    """
    unsupported = next((c for c in Capability if not supported(c)), None)
    if unsupported is None:
        pytest.skip("nothing unsupported to assert against")
    v = verdict_for(unsupported, observed_ok=True)
    assert v.supported is False
    assert v.ok is False, (
        "an unsupported capability returned ok=True -- observed_ok was trusted for "
        "a property nothing enforces, which is forgery by absence"
    )


def test_a_supported_capability_reports_the_observation():
    ok_cap = next((c for c in Capability if supported(c)), None)
    assert ok_cap is not None
    assert verdict_for(ok_cap, observed_ok=True).ok is True
    assert verdict_for(ok_cap, observed_ok=False).ok is False


def test_the_module_is_pure():
    import ast

    import mind_mem.capabilities as m

    tree = ast.parse(open(m.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    impure = imported & {"os", "time", "random", "datetime", "subprocess"}
    assert not impure, f"capabilities must stay pure; imports {sorted(impure)}"
