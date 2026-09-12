"""A deliberately skipping test, used only by the evidence gate's self-test.

It exists so `check_evidence_executes.py --self-test` has something that is
guaranteed to skip. If the gate does not flag this, the gate is asleep and
every "evidence gate passed" line it has ever printed means nothing.

It is NOT in the manifest, so the ordinary gate run never sees it.

IT LIVES OUTSIDE `tests/` ON PURPOSE. Two of this repo's own gates want opposite
things from this file: `check_evidence_executes.py` needs a test that is guaranteed
to skip, and `test_no_vacuous_skips.py` forbids any unconditional skip in the suite
(an unconditional skip runs on no matrix row, so its assertions never execute
anywhere). Exempting the canary by name would have blunted the second gate for the
convenience of the first.

`pyproject.toml` sets `testpaths = ["tests"]`, so a default `pytest` run does not
collect this directory and `test_no_vacuous_skips.py` -- which scans its own tree --
does not see it. `check_evidence_executes.py --self-test` still reaches it by passing
the explicit node id, which overrides testpaths. Both gates keep full strength, by
construction rather than by exemption.
"""

import pytest


@pytest.mark.skip(reason="canary: the evidence gate's self-test requires a skip to detect")
def test_canary_evidence_test_that_skips():
    raise AssertionError("this body must never run")
