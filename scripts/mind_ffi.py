"""mind-mem FFI bridge — loads compiled MIND .so and exposes scoring functions.

The MIND kernel is OPTIONAL. mind-mem works without it (pure Python fallback).
With it, scoring is native-speed compiled code with compile-time tensor shape checks.

The compiled .so exposes a C99-compatible ABI via mind_runtime.h.
Each function accepts flat float pointers and dimension parameters.

Also provides utility functions for listing .mind source files (used by MCP tools).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


# --- Library loading ---

_LIB_SEARCH_PATHS = [
    Path(__file__).parent.parent / "lib" / "libmindmem.so",
    Path(__file__).parent.parent / "lib" / "libmindmem.dylib",
]


class MindMemKernel:
    """Wrapper around compiled MIND scoring kernels.

    Usage:
        try:
            kernel = MindMemKernel()
            scores = kernel.rrf_fuse_py(bm25_ranks, vec_ranks)
        except (OSError, ImportError):
            pass  # Fallback to pure Python
    """

    def __init__(self, lib_path: Optional[str] = None):
        """Load the compiled MIND shared library.

        Raises:
            OSError: If library cannot be loaded.
        """
        self._lib = None

        if lib_path:
            self._lib = ctypes.CDLL(str(lib_path))
        else:
            env_path = os.environ.get("MIND_MEM_LIB", "")
            if env_path:
                # Restrict to allowed directories (prevent arbitrary .so loading)
                resolved = Path(env_path).resolve()
                allowed = [Path(__file__).parent.parent / "lib"]
                if any(str(resolved).startswith(str(d.resolve())) for d in allowed) and resolved.exists():
                    self._lib = ctypes.CDLL(str(resolved))

            if self._lib is None:
                for p in _LIB_SEARCH_PATHS:
                    if p.exists():
                        self._lib = ctypes.CDLL(str(p))
                        break

        if self._lib is None:
            raise OSError(
                "MIND kernel library not found. "
                "Compile with: mindc mind/*.mind --emit=shared -o lib/libmindmem.so"
            )

        # Declare argtypes for all kernel functions (prevents silent memory corruption)
        _f = ctypes.c_float
        _i = ctypes.c_int
        _fp = ctypes.POINTER(ctypes.c_float)
        _ip = ctypes.POINTER(ctypes.c_int)
        try:
            self._lib.rrf_fuse.argtypes = [_fp, _fp, _i, _f, _f, _f, _fp]
            self._lib.rrf_fuse.restype = None
            self._lib.bm25f_batch.argtypes = [_fp, _f, _f, _fp, _f, _f, _f, _f, _i, _fp]
            self._lib.bm25f_batch.restype = None
            self._lib.negation_penalty.argtypes = [_fp, _fp, _f, _i, _fp]
            self._lib.negation_penalty.restype = None
            self._lib.date_proximity.argtypes = [_fp, _f, _i, _fp]
            self._lib.date_proximity.restype = None
            self._lib.category_boost.argtypes = [_fp, _fp, _f, _i, _fp]
            self._lib.category_boost.restype = None
            self._lib.importance_batch.argtypes = [_ip, _fp, _f, _f, _i, _fp]
            self._lib.importance_batch.restype = None
            self._lib.confidence_score.argtypes = [_f, _f, _f, _f, _f, _f, _f, _f, _f, _f]
            self._lib.confidence_score.restype = _f
            self._lib.top_k_mask.argtypes = [_fp, _i, _i, _fp]
            self._lib.top_k_mask.restype = None
            self._lib.weighted_rank.argtypes = [_fp, _fp, _i, _fp]
            self._lib.weighted_rank.restype = None
            self._lib.category_affinity.argtypes = [_fp, _fp, _fp, _f, _f, _f, _i, _i, _fp]
            self._lib.category_affinity.restype = None
            self._lib.query_category_relevance.argtypes = [_fp, _fp, _i, _i, _fp]
            self._lib.query_category_relevance.restype = None
            self._lib.category_assign.argtypes = [_fp, _f, _i, _i, _fp]
            self._lib.category_assign.restype = None
        except AttributeError:
            pass  # Some builds may not export all functions

        # Check if the library includes runtime protection
        self._protected = False
        try:
            self._lib.mindmem_protected.argtypes = []
            self._lib.mindmem_protected.restype = ctypes.c_int
            self._protected = bool(self._lib.mindmem_protected())
        except AttributeError:
            pass  # Unprotected build (dev/CI fallback)

    def is_protected(self) -> bool:
        """Return True if the loaded library includes runtime protection."""
        return self._protected

    def rrf_fuse_py(self, bm25_ranks: list[float], vector_ranks: list[float],
                    k: float = 60.0, bm25_w: float = 1.0,
                    vector_w: float = 1.0) -> list[float]:
        """RRF fusion via compiled MIND kernel."""
        N = len(bm25_ranks)
        arr_t = ctypes.c_float * N
        out = arr_t()
        self._lib.rrf_fuse(
            arr_t(*bm25_ranks), arr_t(*vector_ranks), ctypes.c_int(N),
            ctypes.c_float(k), ctypes.c_float(bm25_w), ctypes.c_float(vector_w),
            out,
        )
        return list(out)

    def bm25f_batch_py(self, tfs: list[float], dfs: list[float], N_docs: float,
                       dls: list[float], avgdl: float,
                       k1: float = 1.2, b: float = 0.75,
                       field_weight: float = 1.0) -> list[float]:
        """BM25F batch scoring via compiled MIND kernel."""
        n = len(tfs)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.bm25f_batch(
            arr_t(*tfs), ctypes.c_float(dfs[0]), ctypes.c_float(N_docs),
            arr_t(*dls), ctypes.c_float(avgdl),
            ctypes.c_float(k1), ctypes.c_float(b), ctypes.c_float(field_weight),
            ctypes.c_int(n), out,
        )
        return list(out)

    def negation_penalty_py(self, scores: list[float], has_negation: list[bool],
                            penalty: float = 0.3) -> list[float]:
        """Negation penalty via compiled MIND kernel."""
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        flags = arr_t(*(1.0 if b else 0.0 for b in has_negation))
        self._lib.negation_penalty(
            arr_t(*scores), flags, ctypes.c_float(penalty), ctypes.c_int(n), out,
        )
        return list(out)

    def date_proximity_py(self, days_diff: list[float],
                          sigma: float = 30.0) -> list[float]:
        """Gaussian date proximity via compiled MIND kernel."""
        n = len(days_diff)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.date_proximity(
            arr_t(*days_diff), ctypes.c_float(sigma), ctypes.c_int(n), out,
        )
        return list(out)

    def category_boost_py(self, scores: list[float], matches: list[bool],
                          boost: float = 1.15) -> list[float]:
        """Category boost via compiled MIND kernel."""
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        flags = arr_t(*(1.0 if b else 0.0 for b in matches))
        self._lib.category_boost(
            arr_t(*scores), flags, ctypes.c_float(boost), ctypes.c_int(n), out,
        )
        return list(out)

    def importance_batch_py(self, access_counts: list[int], days_since: list[float],
                            base: float = 1.0, decay: float = 0.01) -> list[float]:
        """Importance batch scoring via compiled MIND kernel."""
        n = len(access_counts)
        float_arr = ctypes.c_float * n
        int_arr = ctypes.c_int * n
        out = float_arr()
        self._lib.importance_batch(
            int_arr(*access_counts), float_arr(*days_since),
            ctypes.c_float(base), ctypes.c_float(decay), ctypes.c_int(n), out,
        )
        return list(out)

    def confidence_score_py(self, entity_overlap: float, bm25_norm: float,
                            speaker_cov: float, evidence_density: float,
                            negation_asym: float,
                            weights: tuple = (0.30, 0.25, 0.15, 0.20, 0.10)) -> float:
        """Confidence score via compiled MIND kernel."""
        self._lib.confidence_score.restype = ctypes.c_float
        return self._lib.confidence_score(
            ctypes.c_float(entity_overlap), ctypes.c_float(bm25_norm),
            ctypes.c_float(speaker_cov), ctypes.c_float(evidence_density),
            ctypes.c_float(negation_asym),
            ctypes.c_float(weights[0]), ctypes.c_float(weights[1]),
            ctypes.c_float(weights[2]), ctypes.c_float(weights[3]),
            ctypes.c_float(weights[4]),
        )

    def top_k_mask_py(self, scores: list[float], k: int) -> list[bool]:
        """Top-K mask via compiled MIND kernel."""
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.top_k_mask(
            arr_t(*scores), ctypes.c_int(n), ctypes.c_int(k), out,
        )
        return [v > 0.5 for v in out]

    def weighted_rank_py(self, scores: list[float],
                         weights: list[float]) -> list[float]:
        """Weighted rank via compiled MIND kernel."""
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.weighted_rank(
            arr_t(*scores), arr_t(*weights), ctypes.c_int(n), out,
        )
        return list(out)

    def category_affinity_py(self, kw_overlap: list[float], tag_match: list[float],
                              ent_match: list[float], n_blocks: int, n_cats: int,
                              kw_w: float = 0.5, tag_w: float = 0.3,
                              ent_w: float = 0.2) -> list[float]:
        """Category affinity scoring via compiled MIND kernel.

        All inputs are flat row-major [N*C]. Returns flat [N*C] affinity scores.
        """
        total = n_blocks * n_cats
        arr_t = ctypes.c_float * total
        out = arr_t()
        self._lib.category_affinity(
            arr_t(*kw_overlap), arr_t(*tag_match), arr_t(*ent_match),
            ctypes.c_float(kw_w), ctypes.c_float(tag_w), ctypes.c_float(ent_w),
            ctypes.c_int(n_blocks), ctypes.c_int(n_cats), out,
        )
        return list(out)

    def query_category_relevance_py(self, query_kw: list[float],
                                     cat_kw: list[float], n_cats: int,
                                     n_keywords: int) -> list[float]:
        """Query-category relevance via compiled MIND kernel.

        query_kw: flat [K] keyword weights.
        cat_kw: flat row-major [C*K] category keyword profiles.
        Returns [C] relevance scores.
        """
        qk_t = ctypes.c_float * n_keywords
        ck_t = ctypes.c_float * (n_cats * n_keywords)
        out_t = ctypes.c_float * n_cats
        out = out_t()
        self._lib.query_category_relevance(
            qk_t(*query_kw), ck_t(*cat_kw),
            ctypes.c_int(n_cats), ctypes.c_int(n_keywords), out,
        )
        return list(out)

    def category_assign_py(self, affinity: list[float], threshold: float,
                            n_blocks: int, n_cats: int) -> list[float]:
        """Soft category assignment via compiled MIND kernel.

        Returns [N*C] sigmoid-thresholded assignment weights.
        """
        total = n_blocks * n_cats
        arr_t = ctypes.c_float * total
        out = arr_t()
        self._lib.category_assign(
            arr_t(*affinity), ctypes.c_float(threshold),
            ctypes.c_int(n_blocks), ctypes.c_int(n_cats), out,
        )
        return list(out)


# --- Module-level singleton ---

_kernel: Optional[MindMemKernel] = None
_USE_MIND: bool = False


def get_kernel() -> Optional[MindMemKernel]:
    """Get or create singleton kernel. Returns None if unavailable."""
    global _kernel, _USE_MIND
    if _kernel is not None:
        return _kernel
    try:
        _kernel = MindMemKernel()
        _USE_MIND = True
        return _kernel
    except (OSError, ImportError):
        _USE_MIND = False
        return None


def is_available() -> bool:
    """Check if compiled MIND kernel is available."""
    get_kernel()
    return _USE_MIND


def is_protected() -> bool:
    """Check if the MIND kernel has FORTRESS runtime protection."""
    k = get_kernel()
    if k is None:
        return False
    return k.is_protected()


# --- Utility functions for .mind source file listing ---
# Used by MCP tools to expose kernel metadata

def list_kernels(directory: str) -> list[str]:
    """List available .mind kernel source names in a directory.

    Args:
        directory: Path to the mind/ directory.

    Returns:
        Sorted list of kernel names (without .mind extension).
    """
    if not os.path.isdir(directory):
        return []
    try:
        return sorted(
            fname[:-5]
            for fname in os.listdir(directory)
            if fname.endswith(".mind") and not fname.startswith(".")
        )
    except OSError:
        return []


def get_mind_dir(workspace: str = "") -> str:
    """Resolve the mind/ directory.

    Checks workspace/mind/ first, then falls back to the package-level mind/.
    """
    if workspace:
        ws_mind = os.path.join(workspace, "mind")
        if os.path.isdir(ws_mind):
            return ws_mind

    pkg_mind = str(Path(__file__).parent.parent / "mind")
    if os.path.isdir(pkg_mind):
        return pkg_mind

    return os.path.join(workspace, "mind") if workspace else pkg_mind


def load_kernel(path: str) -> dict:
    """Load metadata from a .mind source file (extracts function signatures).

    Parses MIND source to extract function names and comments.
    Returns dict with kernel info for the MCP index_stats tool.
    """
    if not os.path.isfile(path):
        return {}

    functions = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("fn ") and "(" in stripped:
                    # Extract function name
                    name = stripped.split("(")[0].replace("fn ", "").strip()
                    functions.append(name)
    except (OSError, UnicodeDecodeError):
        return {}

    return {"functions": functions, "path": path}


def load_all_kernels(directory: str) -> dict[str, dict]:
    """Load metadata for all .mind kernels in a directory."""
    result = {}
    for name in list_kernels(directory):
        path = os.path.join(directory, f"{name}.mind")
        result[name] = load_kernel(path)
    return result


def get_kernel_param(config: dict, section: str, key: str, default=None):
    """Get a parameter from kernel config. Compatibility shim."""
    return config.get(section, {}).get(key, default)


# --- INI-style .mind config parsing ---
# .mind files use a simple [section] / key = value format for tuning params.

import re as _re  # noqa: E402


def _parse_value(raw: str):
    """Auto-detect value type from raw string."""
    stripped = raw.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if _re.match(r"^-?\d+$", stripped):
        return int(stripped)
    if _re.match(r"^-?\d+\.\d+$", stripped):
        return float(stripped)
    if "," in stripped:
        return [_parse_value(s.strip()) for s in stripped.split(",") if s.strip()]
    return stripped


def load_kernel_config(path: str) -> dict:
    """Load a .mind file as INI-style config. Returns {section: {key: value}}.

    This parses the declarative [section] / key = value format used by
    tuning kernels (recall.mind, rm3.mind, etc.).
    """
    if not os.path.isfile(path):
        return {}

    result = {}
    current_section = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                section_match = _re.match(r"^\[([a-zA-Z_][a-zA-Z0-9_]*)\]\s*$", stripped)
                if section_match:
                    current_section = section_match.group(1)
                    if current_section not in result:
                        result[current_section] = {}
                    continue

                kv_match = _re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)$", stripped)
                if kv_match and current_section is not None:
                    key = kv_match.group(1)
                    raw_value = kv_match.group(2).strip()
                    result[current_section][key] = _parse_value(raw_value)

    except (OSError, UnicodeDecodeError):
        return {}

    return result


def load_all_kernel_configs(directory: str) -> dict[str, dict]:
    """Load all .mind kernel configs from a directory as INI-style dicts."""
    result = {}
    for name in list_kernels(directory):
        path = os.path.join(directory, f"{name}.mind")
        result[name] = load_kernel_config(path)
    return result
