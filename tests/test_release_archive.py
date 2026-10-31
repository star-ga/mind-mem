"""Non-dataset examples for exact prompt/answer reuse."""

import hashlib

import pytest

from benchmarks.release_archive import select_measured_outputs


def prompt(text):
    return {"id": "synthetic", "prompt": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_only_byte_identical_prompts_reuse_answers():
    base, target = prompt("first"), prompt("second")
    measured = {"id": base["id"], "prompt_sha256": base["prompt_sha256"], "generated": "A"}
    with pytest.raises(ValueError, match="Unmeasured"):
        select_measured_outputs([base], [target], [measured], [])
    new = {**measured, "prompt_sha256": target["prompt_sha256"], "generated": "B"}
    assert select_measured_outputs([base], [target], [measured], [new]) == [new]
    assert select_measured_outputs([base], [base], [measured], []) == [measured]


def test_conflicting_measurements_fail_closed():
    base = prompt("first")
    measured = {"id": base["id"], "prompt_sha256": base["prompt_sha256"], "generated": "A"}
    with pytest.raises(ValueError, match="Conflicting"):
        select_measured_outputs([base], [base], [measured], [measured, {**measured, "generated": "B"}])
