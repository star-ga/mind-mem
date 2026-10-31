"""Join measured answers to rendered prompts by exact identity and content hash.

This is the offline canonical selection mechanism. No inference, evaluation
labels, thresholds, or dataset-specific answer rules participate in selection.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib

from benchmarks.release_scoring import unique


def select_measured_outputs(base_prompts: list[dict], target_prompts: list[dict], baseline: list[dict], overlays: list[dict]) -> list[dict]:
    """Inputs use id/prompt_sha256; measurements also carry generated/votes/verdict."""
    base, target, measurements = map(unique, (base_prompts, target_prompts, baseline))
    if set(base) != set(target) or set(base) != set(measurements):
        raise ValueError("Baseline and target populations must agree exactly")
    changed = {}
    for row in overlays:
        pair = row["id"], row["prompt_sha256"]
        if pair in changed and changed[pair] != row:
            raise ValueError("Conflicting measurements for the same prompt")
        changed[pair] = row
    selected = []
    for iid, prompt in target.items():
        actual = hashlib.sha256(prompt["prompt"].encode()).hexdigest()
        if actual != prompt["prompt_sha256"]:
            raise ValueError("Rendered prompt hash mismatch")
        if actual == base[iid]["prompt_sha256"]:
            measured = measurements[iid]
        else:
            pair = iid, actual
            if pair not in changed:
                raise ValueError("Unmeasured rendered prompt; no score can be inferred")
            measured = changed[pair]
        if measured["prompt_sha256"] != actual:
            raise ValueError("Measurement prompt binding mismatch")
        selected.append(dict(measured))
    return selected
