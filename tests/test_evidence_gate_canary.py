"""A deliberately skipping test, used only by the evidence gate's self-test.

It exists so `check_evidence_executes.py --self-test` has something that is
guaranteed to skip. If the gate does not flag this, the gate is asleep and
every "evidence gate passed" line it has ever printed means nothing.

It is NOT in the manifest, so the ordinary gate run never sees it.
"""

import pytest


@pytest.mark.skip(reason="canary: the evidence gate's self-test requires a skip to detect")
def test_canary_evidence_test_that_skips():
    raise AssertionError("this body must never run")
