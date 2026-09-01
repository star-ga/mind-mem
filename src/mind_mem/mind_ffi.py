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
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import mind_kernels
from .observability import get_logger

_log = get_logger("ffi")

# --- Library loading ---

# Search paths for the compiled MIND kernel library, in probe order:
#   (a) ``<package>/lib/`` — co-located with the Python sources. This is
#       the path the "packaged install" case names, and it needs
#       ``Path(__file__).parent``: ``.parent`` is already the package
#       directory, so ``.parent.parent`` is the directory *containing*
#       the package (``site-packages/lib``), not ``<package>/lib``.
#   (b) ``<parent-of-package>/lib/`` — kept because it is what the
#       previous probe order actually looked at; in a src-layout
#       checkout that is ``repo/src/lib``.
#   (c) ``<repo>/lib/`` — editable dev install, where the layout is
#       ``repo/src/mind_mem/`` and ``repo/lib/`` holds the artifact.
# No wheel ships a .so today (see pyproject package-data), so in
# practice the kernel is checkout-only and its absence is the
# documented pure-Python path, not a failure.
_LIB_SEARCH_PATHS = [
    Path(__file__).parent / "lib" / "libmindmem.so",
    Path(__file__).parent / "lib" / "libmindmem.dylib",
    Path(__file__).parent.parent / "lib" / "libmindmem.so",
    Path(__file__).parent.parent / "lib" / "libmindmem.dylib",
    Path(__file__).parent.parent.parent / "lib" / "libmindmem.so",
    Path(__file__).parent.parent.parent / "lib" / "libmindmem.dylib",
]


def _get_python_version() -> str:
    """Get the Python package __version__ string."""
    try:
        # When running as installed package or scripts/ is on sys.path
        from __init__ import __version__

        return str(__version__)
    except ImportError:
        pass
    # Fallback: read from __init__.py next to this file
    init_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
    try:
        with open(init_path) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "0.0.0"


def _check_version_compat(so_version: str) -> bool:
    """Compare .so version against Python __version__. Warn on major.minor mismatch.

    Returns True if compatible, False otherwise.
    """
    py_version = _get_python_version()

    try:
        so_parts = so_version.split(".")
        py_parts = py_version.split(".")
        so_major_minor = (int(so_parts[0]), int(so_parts[1]))
        py_major_minor = (int(py_parts[0]), int(py_parts[1]))
    except (IndexError, ValueError):
        _log.warning("ffi_version_parse_error", so_version=so_version, py_version=py_version)
        return False

    if so_major_minor != py_major_minor:
        _log.warning("ffi_version_mismatch", so_version=so_version, py_version=py_version)
        return False

    return True


def allowed_lib_dirs() -> list[Path]:
    """Directories an operator-supplied ``.so`` path may live under.

    The same three ``lib/`` locations :data:`_LIB_SEARCH_PATHS` probes,
    minus the file names. Recomputed per call rather than frozen at import
    so a test that relocates the package still sees the right roots.
    """
    return [
        Path(__file__).parent / "lib",
        Path(__file__).parent.parent / "lib",
        Path(__file__).parent.parent.parent / "lib",
    ]


def resolve_allowlisted_lib(raw: str) -> tuple[Path | None, str]:
    """Resolve an operator-supplied library path against the allowlist.

    This is the package's ONE answer to "may I load this shared object?".
    It was previously inlined in :meth:`MindMemKernel.__init__` while
    ``mind_kernels.load_kernels`` ran a second, allowlist-free loader that
    handed ``$MIND_MEM_KERNELS_SO`` straight to ``ctypes.CDLL`` — any path
    in the environment could pull arbitrary native code into the process.
    That loader is gone; both env vars now come through here.

    Args:
        raw: The path as the operator wrote it.

    Returns:
        ``(resolved_path, "")`` when the path is inside an allowed
        directory and exists, else ``(None, reason)``. The caller decides
        whether a rejection is worth logging — a probe that runs on an
        OFF path must be able to ask quietly.
    """
    resolved = Path(raw).resolve()
    dirs = allowed_lib_dirs()
    in_allowed = any(resolved == d.resolve() or str(resolved).startswith(str(d.resolve()) + os.sep) for d in dirs)
    if in_allowed and resolved.exists():
        return resolved, ""
    return None, ("outside allowed directories" if not in_allowed else "file does not exist")


class MindMemKernel:
    """Wrapper around compiled MIND scoring kernels.

    Usage:
        try:
            kernel = MindMemKernel()
            scores = kernel.rrf_fuse_py(bm25_ranks, vec_ranks)
        except (OSError, ImportError):
            pass  # Fallback to pure Python
    """

    def __init__(self, lib_path: str | None = None):
        """Load the compiled MIND shared library.

        Raises:
            OSError: If library cannot be loaded.
        """
        self._lib = None
        # Use RTLD_LAZY (0x1) to defer resolution of protection-layer symbols
        _LAZY = 0x1

        if lib_path:
            self._lib = ctypes.CDLL(str(lib_path), mode=_LAZY)
        else:
            env_path = os.environ.get("MIND_MEM_LIB", "")
            if env_path:
                # Restrict to allowed directories (prevent arbitrary .so loading)
                resolved, reason = resolve_allowlisted_lib(env_path)
                if resolved is not None:
                    self._lib = ctypes.CDLL(str(resolved), mode=_LAZY)
                else:
                    # Say why the explicit override was dropped. Falling
                    # through in silence left the operator staring at
                    # "library not found" with no hint that the path they
                    # set was seen and rejected — or, worse, quietly
                    # running a different library than the one they named.
                    _log.warning(
                        "ffi_env_lib_rejected",
                        path=str(Path(env_path).resolve()),
                        reason=reason,
                        allowed=[str(d) for d in allowed_lib_dirs()],
                    )

            if self._lib is None:
                for p in _LIB_SEARCH_PATHS:
                    if p.exists():
                        self._lib = ctypes.CDLL(str(p), mode=_LAZY)
                        break

        if self._lib is None:
            raise OSError("MIND kernel library not found. Compile with: mindc mind/*.mind --emit=shared -o lib/libmindmem.so")

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

        # Version check: compare .so version against Python __version__.
        # The exported symbol is ``mindmem_version``; ``mindmem_get_version``
        # is probed second only in case an older build used that name. The
        # gate used to look for the second name alone, which no build
        # exports, so it never ran at all.
        self._so_version: str | None = None
        self._version_compatible: bool | None = None
        for symbol in ("mindmem_version", "mindmem_get_version"):
            try:
                fn = getattr(self._lib, symbol)
            except AttributeError:
                continue  # Build doesn't export this version symbol
            fn.argtypes = []
            fn.restype = ctypes.c_char_p
            try:
                raw = fn()
            except (OSError, ValueError):
                _log.warning("ffi_version_symbol_unreadable", symbol=symbol)
                break
            if raw:
                self._so_version = raw.decode("utf-8", errors="replace")
                self._version_compatible = _check_version_compat(self._so_version)
            break

    def is_protected(self) -> bool:
        """Return True if the loaded library includes runtime protection."""
        return self._protected

    def so_version(self) -> str | None:
        """Return the version string reported by the .so, or None."""
        return self._so_version

    def version_compatible(self) -> bool | None:
        """Return the verdict of the major.minor version check.

        ``True``/``False`` once a version symbol was read, ``None`` when
        the library exports none (nothing to compare, so nothing is
        claimed). Loading is deliberately not refused on ``False`` — the
        mismatch is reported, and it is the caller's call whether a
        drifted kernel is acceptable or whether to fall back to the pure
        Python path.
        """
        return self._version_compatible

    def rrf_fuse_py(
        self,
        bm25_ranks: list[float],
        vector_ranks: list[float],
        k: float = 60.0,
        bm25_w: float = 1.0,
        vector_w: float = 1.0,
    ) -> list[float]:
        """RRF fusion via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        N = len(bm25_ranks)
        arr_t = ctypes.c_float * N
        out = arr_t()
        self._lib.rrf_fuse(
            arr_t(*bm25_ranks),
            arr_t(*vector_ranks),
            ctypes.c_int(N),
            ctypes.c_float(k),
            ctypes.c_float(bm25_w),
            ctypes.c_float(vector_w),
            out,
        )
        return list(out)

    def bm25f_batch_py(
        self,
        tfs: list[float],
        dfs: list[float],
        N_docs: float,
        dls: list[float],
        avgdl: float,
        k1: float = 1.2,
        b: float = 0.75,
        field_weight: float = 1.0,
    ) -> list[float]:
        """BM25F batch scoring via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(tfs)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.bm25f_batch(
            arr_t(*tfs),
            ctypes.c_float(dfs[0]),
            ctypes.c_float(N_docs),
            arr_t(*dls),
            ctypes.c_float(avgdl),
            ctypes.c_float(k1),
            ctypes.c_float(b),
            ctypes.c_float(field_weight),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def negation_penalty_py(self, scores: list[float], has_negation: list[bool], penalty: float = 0.3) -> list[float]:
        """Negation penalty via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        flags = arr_t(*(1.0 if b else 0.0 for b in has_negation))
        self._lib.negation_penalty(
            arr_t(*scores),
            flags,
            ctypes.c_float(penalty),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def date_proximity_py(self, days_diff: list[float], sigma: float = 30.0) -> list[float]:
        """Gaussian date proximity via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(days_diff)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.date_proximity(
            arr_t(*days_diff),
            ctypes.c_float(sigma),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def category_boost_py(self, scores: list[float], matches: list[bool], boost: float = 1.15) -> list[float]:
        """Category boost via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        flags = arr_t(*(1.0 if b else 0.0 for b in matches))
        self._lib.category_boost(
            arr_t(*scores),
            flags,
            ctypes.c_float(boost),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def importance_batch_py(self, access_counts: list[int], days_since: list[float], base: float = 1.0, decay: float = 0.01) -> list[float]:
        """Importance batch scoring via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(access_counts)
        float_arr = ctypes.c_float * n
        int_arr = ctypes.c_int * n
        out = float_arr()
        self._lib.importance_batch(
            int_arr(*access_counts),
            float_arr(*days_since),
            ctypes.c_float(base),
            ctypes.c_float(decay),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def confidence_score_py(
        self,
        entity_overlap: float,
        bm25_norm: float,
        speaker_cov: float,
        evidence_density: float,
        negation_asym: float,
        weights: tuple = (0.30, 0.25, 0.15, 0.20, 0.10),
    ) -> float:
        """Confidence score via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        self._lib.confidence_score.restype = ctypes.c_float
        return float(
            self._lib.confidence_score(
                ctypes.c_float(entity_overlap),
                ctypes.c_float(bm25_norm),
                ctypes.c_float(speaker_cov),
                ctypes.c_float(evidence_density),
                ctypes.c_float(negation_asym),
                ctypes.c_float(weights[0]),
                ctypes.c_float(weights[1]),
                ctypes.c_float(weights[2]),
                ctypes.c_float(weights[3]),
                ctypes.c_float(weights[4]),
            )
        )

    def top_k_mask_py(self, scores: list[float], k: int) -> list[bool]:
        """Top-K mask via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.top_k_mask(
            arr_t(*scores),
            ctypes.c_int(n),
            ctypes.c_int(k),
            out,
        )
        return [v > 0.5 for v in out]

    def weighted_rank_py(self, scores: list[float], weights: list[float]) -> list[float]:
        """Weighted rank via compiled MIND kernel."""
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        n = len(scores)
        arr_t = ctypes.c_float * n
        out = arr_t()
        self._lib.weighted_rank(
            arr_t(*scores),
            arr_t(*weights),
            ctypes.c_int(n),
            out,
        )
        return list(out)

    def category_affinity_py(
        self,
        kw_overlap: list[float],
        tag_match: list[float],
        ent_match: list[float],
        n_blocks: int,
        n_cats: int,
        kw_w: float = 0.5,
        tag_w: float = 0.3,
        ent_w: float = 0.2,
    ) -> list[float]:
        """Category affinity scoring via compiled MIND kernel.

        All inputs are flat row-major [N*C]. Returns flat [N*C] affinity scores.
        """
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        total = n_blocks * n_cats
        arr_t = ctypes.c_float * total
        out = arr_t()
        self._lib.category_affinity(
            arr_t(*kw_overlap),
            arr_t(*tag_match),
            arr_t(*ent_match),
            ctypes.c_float(kw_w),
            ctypes.c_float(tag_w),
            ctypes.c_float(ent_w),
            ctypes.c_int(n_blocks),
            ctypes.c_int(n_cats),
            out,
        )
        return list(out)

    def query_category_relevance_py(self, query_kw: list[float], cat_kw: list[float], n_cats: int, n_keywords: int) -> list[float]:
        """Query-category relevance via compiled MIND kernel.

        query_kw: flat [K] keyword weights.
        cat_kw: flat row-major [C*K] category keyword profiles.
        Returns [C] relevance scores.
        """
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        qk_t = ctypes.c_float * n_keywords
        ck_t = ctypes.c_float * (n_cats * n_keywords)
        out_t = ctypes.c_float * n_cats
        out = out_t()
        self._lib.query_category_relevance(
            qk_t(*query_kw),
            ck_t(*cat_kw),
            ctypes.c_int(n_cats),
            ctypes.c_int(n_keywords),
            out,
        )
        return list(out)

    def category_assign_py(self, affinity: list[float], threshold: float, n_blocks: int, n_cats: int) -> list[float]:
        """Soft category assignment via compiled MIND kernel.

        Returns [N*C] sigmoid-thresholded assignment weights.
        """
        if self._lib is None:
            raise RuntimeError("MindMemKernel: MIND shared library is not loaded")
        total = n_blocks * n_cats
        arr_t = ctypes.c_float * total
        out = arr_t()
        self._lib.category_assign(
            arr_t(*affinity),
            ctypes.c_float(threshold),
            ctypes.c_int(n_blocks),
            ctypes.c_int(n_cats),
            out,
        )
        return list(out)


# --- Module-level singleton ---

_kernel: MindMemKernel | None = None
_USE_MIND: bool = False


def _try_native(lib_path: str | None = None) -> MindMemKernel | None:
    """Probe for the compiled kernel. Returns None if absent — and stays QUIET.

    The one native probe. Silence is the contract, not an oversight: this
    runs on paths where the kernel's absence is the documented normal case
    (no wheel ships a ``.so``), and a probe that logs makes a build with the
    feature wired observably different from one without it. Callers that
    want the "falling back to pure Python" notice emit it themselves —
    :func:`get_kernel` does.
    """
    try:
        return MindMemKernel(lib_path)
    except (OSError, ImportError):
        return None


def get_kernel() -> MindMemKernel | None:
    """Get or create singleton kernel. Returns None if unavailable."""
    global _kernel, _USE_MIND
    if _kernel is not None:
        return _kernel
    native = _try_native()
    if native is not None:
        _kernel = native
        _USE_MIND = True
        return _kernel
    _USE_MIND = False
    _log.info("MIND kernel .so not found — using pure Python fallback. Compile with: mindc mind/*.mind --emit=shared -o lib/libmindmem.so")
    return None


# --- The one kernel loader ---

#: Env vars an operator may point at a compiled kernel, in probe order.
#: ``MIND_MEM_KERNELS_SO`` is the name the retired ``mind_kernels.load_kernels``
#: read; it is still honoured, but it now goes through
#: :func:`resolve_allowlisted_lib` like everything else instead of being
#: handed to ``ctypes.CDLL`` unchecked.
_LIB_ENV_VARS = ("MIND_MEM_LIB", "MIND_MEM_KERNELS_SO")


@dataclass(frozen=True)
class Kernels:
    """A resolved kernel binding: four hot-path kernels plus the native handle.

    ``backend`` names which *library* answered the probe — ``"native"`` when
    a compiled ``libmindmem.so`` loaded, ``"python"`` otherwise. It does NOT
    mean the four callables below changed: they are always the pure-Python
    implementations in :mod:`mind_mem.mind_kernels`.

    deferred: the compiled ABI is *batched* (``bm25f_batch``, ``rrf_fuse``
    over flat float arrays) while these four are per-document/per-pair, so
    there is no shim to route them through the ``.so`` yet. Upgrade path:
    add scalar wrappers over ``native.bm25f_batch_py`` /
    ``native.rrf_fuse_py`` and bind them here when ``backend == "native"``,
    gated on a byte-identity test against the Python results. Until then a
    native library accelerates only the callers that reach for ``.native``
    directly (``category_distiller``).
    """

    bm25f_score: Callable[..., float]
    sha3_512_chain_verify: Callable[..., bool]
    cosine: Callable[..., float]
    dot: Callable[..., float]
    rrf_fusion: Callable[..., list]
    native: MindMemKernel | None
    backend: str


def load_kernels(path: str | None = None) -> Kernels:
    """Resolve the kernel binding — the package's single kernel loader.

    Probe order: an explicit *path*, then ``$MIND_MEM_LIB``, then
    ``$MIND_MEM_KERNELS_SO``, then :data:`_LIB_SEARCH_PATHS`. Every
    operator-supplied path is checked against
    :func:`resolve_allowlisted_lib` first, including the explicit argument —
    so no caller of this function can load a shared object from outside the
    package's ``lib/`` directories.

    A rejected env path is reported (``ffi_env_lib_rejected``), because an
    operator who set a variable is owed the reason it was ignored. A path
    that is simply absent is not: the kernel being missing is the normal,
    documented state of every install.

    Args:
        path: Explicit library path. Allowlist-checked like the env vars.

    Returns:
        A :class:`Kernels` binding. Never raises and never returns None —
        MIND kernels are optional, so "no library" is an answer, not a
        failure.
    """
    candidate: Path | None = None
    for raw, source in [(path, "argument")] + [(os.environ.get(v, ""), v) for v in _LIB_ENV_VARS]:
        if not raw:
            continue
        resolved, reason = resolve_allowlisted_lib(raw)
        if resolved is not None:
            candidate = resolved
            break
        _log.warning(
            "ffi_env_lib_rejected",
            path=str(Path(raw).resolve()),
            reason=reason,
            source=source,
            allowed=[str(d) for d in allowed_lib_dirs()],
        )

    if candidate is None:
        # Walk the search paths here rather than letting MindMemKernel do it:
        # its no-argument constructor re-reads MIND_MEM_LIB and would log a
        # second rejection for the path we just reported. These paths ARE the
        # allowlist, so handing one over explicitly skips no check.
        candidate = next((probe for probe in _LIB_SEARCH_PATHS if probe.exists()), None)

    native = _try_native(str(candidate)) if candidate is not None else None
    return Kernels(
        bm25f_score=mind_kernels.bm25f_score,
        sha3_512_chain_verify=mind_kernels.sha3_512_chain_verify,
        cosine=mind_kernels.cosine,
        dot=mind_kernels.dot,
        rrf_fusion=mind_kernels.rrf_fusion,
        native=native,
        backend="native" if native is not None else "python",
    )


def is_available() -> bool:
    """Check if compiled MIND kernel is available."""
    get_kernel()
    return _USE_MIND


def is_protected() -> bool:
    """Check if the MIND kernel has runtime protection."""
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
        return sorted(fname[:-5] for fname in os.listdir(directory) if fname.endswith(".mind") and not fname.startswith("."))
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

    result: dict[str, dict[str, object]] = {}
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
