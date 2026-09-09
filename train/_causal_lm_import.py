"""Import the shared causal loader from a checkout or a deployed RunPod file."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    """Find the canonical loader without assuming the package is installed.

    A checkout or the staged RunPod bundle is authoritative for this helper.
    Importing an installed package first could silently select an older
    revision while the adjacent source had already been corrected.
    """

    module_name = "mind_mem.causal_lm_loader"
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

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Do not turn an installed loader's missing transitive dependency into
        # a misleading "loader unavailable" message or silently try another
        # revision. Only absence of the package/module itself is fallback-safe.
        if exc.name not in {"mind_mem", module_name}:
            raise
        raise ModuleNotFoundError(
            "mind_mem.causal_lm_loader is unavailable; install mind-mem or deploy "
            "mind_mem_causal_lm_loader.py beside the training script"
        ) from exc


def load_causal_lm(source: str, **kwargs: Any) -> Any:
    """Delegate to the canonical loader found by :func:`_load_module`."""

    return _load_module().load_causal_lm(source, **kwargs)
