"""Hash-of-code pipeline invalidation (v3.9.0 candidate).

When the extractor function, chunker, or prompt template changes,
existing blocks become stale even when the source bytes are
unchanged. This module computes a deterministic ``transform_hash``
for the *currently configured pipeline* so callers can detect blocks
that were extracted by an older pipeline version and need
re-extraction.

Inspired by the cocoindex (Apache-2.0) `target = F(source)` model
where the recompute key includes the transform code as well as the
source. mind-mem keeps its own indexer; we only borrow the concept.

The hash inputs (in order, NUL-separated to prevent boundary
ambiguity)::

    1. The mind-mem package version (``__version__``).
    2. The extractor backend name (``extraction.backend`` from
       mind-mem.json — e.g. ``"ollama"``, ``"openai-compatible"``).
    3. The extractor model id (``extraction.model``).
    4. A SHA-256 of the on-disk extractor source file (whose path is
       resolved via the backend name).
    5. A SHA-256 of the prompt template file when one is configured.

The output is a hex SHA-256 string that callers store on each block as
``transform_hash``. ``pipeline_dirty_blocks(workspace)`` walks the
workspace and returns block ids whose stored hash does not match the
current hash. ``v3.9`` ships the inspection primitive; v3.10 wires
re-extraction into the dream cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, overload

from . import __version__ as _pkg_version

__all__ = [
    "current_pipeline_hash",
    "PipelineHashInputs",
    "compute_pipeline_hash",
    "pipeline_dirty_blocks",
]

_log = logging.getLogger("mind_mem.pipeline_hash")

# Files whose contents contribute to the pipeline hash.
# Resolved relative to the mind_mem package root.
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# extraction.backend → on-disk source file that defines the extractor's logic.
# Keep the entries shallow; we only want the *current* pipeline's code,
# not every backend's code.
_BACKEND_SOURCE_FILES: dict[str, str] = {
    "ollama": os.path.join(_PACKAGE_ROOT, "llm_extractor.py"),
    "openai-compatible": os.path.join(_PACKAGE_ROOT, "llm_extractor.py"),
    "llama-cpp": os.path.join(_PACKAGE_ROOT, "llm_extractor.py"),
    "transformers": os.path.join(_PACKAGE_ROOT, "llm_extractor.py"),
    # Default / unknown backends fall through to a stub hash so an unknown
    # backend doesn't silently produce the same hash as the default.
}


# Field record we emit alongside the hex digest, useful for debugging.
@dataclass(frozen=True)
class PipelineHashInputs:
    package_version: str
    backend: str
    model: str
    extractor_source_sha256: str
    prompt_template_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "package_version": self.package_version,
            "backend": self.backend,
            "model": self.model,
            "extractor_source_sha256": self.extractor_source_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
        }


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 of *path*, or an empty string if unreadable."""
    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError as exc:
        _log.warning("pipeline_hash_read_failed", extra={"path": path, "error": str(exc)})
        return ""
    return h.hexdigest()


def compute_pipeline_hash(inputs: PipelineHashInputs) -> str:
    """Deterministic hex SHA-256 over a NUL-separated preimage."""
    preimage = "\x00".join(
        [
            inputs.package_version,
            inputs.backend,
            inputs.model,
            inputs.extractor_source_sha256,
            inputs.prompt_template_sha256,
        ]
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_workspace_config(workspace: str) -> dict[str, Any]:
    config_path = os.path.join(os.path.abspath(workspace), "mind-mem.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


@overload
def current_pipeline_hash(workspace: str) -> str: ...
@overload
def current_pipeline_hash(workspace: str, *, return_inputs: Literal[False]) -> str: ...
@overload
def current_pipeline_hash(workspace: str, *, return_inputs: Literal[True]) -> tuple[str, PipelineHashInputs]: ...


def current_pipeline_hash(workspace: str, *, return_inputs: bool = False) -> str | tuple[str, PipelineHashInputs]:
    """Compute the pipeline hash for *workspace*.

    Args:
        workspace: Workspace root.
        return_inputs: If True, also return the inputs dataclass for
            inspection.

    Returns:
        Hex SHA-256 (or tuple of (hash, inputs) when requested).
    """
    config = _load_workspace_config(workspace)
    extraction = config.get("extraction") if isinstance(config, dict) else None
    if not isinstance(extraction, dict):
        extraction = {}
    backend = str(extraction.get("backend", "ollama"))
    model = str(extraction.get("model", "default"))
    template_path = extraction.get("prompt_template")

    src_file = _BACKEND_SOURCE_FILES.get(backend)
    extractor_sha = _sha256_file(src_file) if src_file else ""
    if not extractor_sha and src_file:
        # Falls through with a stable sentinel so callers can still
        # detect "extractor source unreadable" deterministically.
        extractor_sha = "0" * 64
    elif not src_file:
        # Unknown backend → distinct sentinel so two unknown backends
        # don't collide with the default.
        extractor_sha = hashlib.sha256(("unknown:" + backend).encode("utf-8")).hexdigest()

    template_sha = _sha256_file(str(template_path)) if isinstance(template_path, str) else ""

    inputs = PipelineHashInputs(
        package_version=str(_pkg_version),
        backend=backend,
        model=model,
        extractor_source_sha256=extractor_sha,
        prompt_template_sha256=template_sha,
    )
    digest = compute_pipeline_hash(inputs)
    if return_inputs:
        return (digest, inputs)
    return digest


def pipeline_dirty_blocks(workspace: str) -> list[str]:
    """Return ids of blocks whose ``transform_hash`` ≠ current pipeline hash.

    Blocks without a ``transform_hash`` field (i.e. extracted before
    v3.9) are reported as dirty — operators can re-extract them at
    their convenience. The function is read-only; it never mutates a
    block.
    """
    current = current_pipeline_hash(workspace)
    if not isinstance(current, str):
        # current_pipeline_hash returns ``str`` on the no-args call; this
        # check makes the runtime contract explicit so the type narrows
        # for both mypy and bandit.
        return []

    from .storage import get_block_store

    try:
        store = get_block_store(workspace)
        blocks = store.get_all()
    except Exception as exc:
        _log.warning("pipeline_dirty_lookup_failed", extra={"error": str(exc)})
        return []

    dirty: list[str] = []
    for block in blocks:
        # Markdown block parser only retains fields whose name starts with
        # an uppercase letter, so the canonical field is TransformHash.
        # We also tolerate the lowercase form for forward-compat with
        # backends (Postgres, sqlite-vec) whose schemas don't impose the
        # capitalization rule.
        existing = block.get("TransformHash") or block.get("transform_hash")
        if not isinstance(existing, str) or existing != current:
            block_id = block.get("_id") or block.get("id")
            if block_id:
                dirty.append(str(block_id))
    return dirty
