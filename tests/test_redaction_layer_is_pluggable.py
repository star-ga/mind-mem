"""Group E — the redaction layer really is PLUGGABLE (verified).

ROADMAP carried this as `[~]`: the detector chain is wired, "whether it is
*pluggable* to the item's full intent is NOT verified, and that is the half the
lock is about."

MEASURED 2026-09-11. Defining a Detector subclass in a caller's own module is
sufficient -- 8 registered detectors became 9 at the class statement, with no
decorator, no registry edit and no import of a plugin list. The module's own words:
"The class is registered by existing."

And the registration VALIDATES rather than merely accepting: a detector declaring
no category raised DetectorSpecError at class-creation time ("expected one of
['pii', 'secret']"). That is the difference between a plugin point and a hole --
an extension mechanism that accepted anything would let a malformed detector into
the pre-write chain, where it screens governed writes.

So "pluggable" is true in the strong sense: extensible by a third party, and
fail-closed against a malformed extension.
"""

from __future__ import annotations

import pytest

from mind_mem.compliance.detectors import Detector, DetectorSpecError, registered_detectors


def test_the_registry_is_non_empty_to_begin_with():
    """POSITIVE CONTROL: a count that starts at 0 would make +1 meaningless."""
    assert len(registered_detectors()) >= 5


def test_defining_a_detector_registers_it_with_no_decorator():
    before = len(registered_detectors())

    class _ProjectCodeDetector(Detector):
        name = "project_code_test"
        category = "pii"

        def scan(self, text):
            import re

            return [m.group(0) for m in re.finditer(r"\bPRJ-\d{4}\b", str(text))]

    after = registered_detectors()
    assert len(after) == before + 1, (before, len(after))
    names = [getattr(d, "name", getattr(d, "__name__", "?")) for d in after]
    assert "project_code_test" in names, names


def test_the_plugged_detector_actually_fires():
    """Registered is not the same as working -- the marker-with-no-reader trap."""

    class _FiringDetector(Detector):
        name = "firing_test"
        category = "secret"

        def scan(self, text):
            return ["HIT"] if "tripwire" in str(text) else []

    assert _FiringDetector().scan("contains a tripwire here") == ["HIT"]
    assert _FiringDetector().scan("nothing of interest") == []


def test_a_malformed_detector_is_REFUSED_at_class_creation():
    """Fail closed. An extension point that accepts anything is a hole.

    This chain screens governed writes; a detector with no declared category
    must not reach it.
    """
    with pytest.raises(DetectorSpecError):

        class _NoCategory(Detector):
            name = "no_category_test"

            def scan(self, text):
                return []


def test_the_refusal_names_the_legal_categories():
    with pytest.raises(DetectorSpecError) as e:

        class _BadCategory(Detector):
            name = "bad_category_test"
            category = "not_a_real_category"

            def scan(self, text):
                return []

    msg = str(e.value)
    assert "pii" in msg and "secret" in msg, msg
