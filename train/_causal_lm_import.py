"""Import the shared causal loader from a checkout or a deployed RunPod file."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    """Find the canonical loader without assuming the package is installed."""

    module_name = "mind_mem.causal_lm_loader"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # A source checkout and the RunPod deployment both carry the same
        # canonical file.  Do not mask unrelated import failures from an
        # installed package.
        if exc.name not in {"mind_mem", module_name}:
            raise

    candidates = (
        Path(__file__).resolve().parents[1] / "src" / "mind_mem" / "causal_lm_loader.py",
        Path(__file__).resolve().with_name("mind_mem_causal_lm_loader.py"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_mind_mem_causal_lm_loader", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError(
        "mind_mem.causal_lm_loader is unavailable; install mind-mem or deploy "
        "mind_mem_causal_lm_loader.py beside the training script"
    )


def load_causal_lm(source: str, **kwargs: Any) -> Any:
    """Delegate to the canonical loader found by :func:`_load_module`."""

    return _load_module().load_causal_lm(source, **kwargs)

