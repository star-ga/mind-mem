"""Required, opt-in ABI gate for the checked-in C kernel reference.

The normal Python-only test suite skips this module explicitly because the C
compiler is an optional developer dependency. The dedicated CI job sets
``MIND_MEM_RUN_NATIVE_C_ABI=1``; in that mode compilation or loading failures
are test failures, never skips.

The production fixture always compiles the repository's current
``lib/kernels.c`` into a temporary directory and calls it through the real
``MindMemKernel`` ctypes bridge. The version-provider fixture is deliberately
separate: it is a tiny ABI probe used only to exercise version reporting,
because the current production C reference intentionally exports no version
symbol.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.bench_kernels import py_top_k_mask
from mind_mem import __version__
from mind_mem.mind_ffi import MindMemKernel

pytestmark = pytest.mark.skipif(
    os.environ.get("MIND_MEM_RUN_NATIVE_C_ABI") != "1",
    reason="native C ABI gate is opt-in; the ordinary Python-only install has no required compiler",
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_C_SOURCE = REPOSITORY_ROOT / "lib" / "kernels.c"


def _compiler() -> str:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.fail("native C ABI gate requires a cc compiler")
    return compiler


def _compile_shared(source: Path, output: Path) -> Path:
    """Compile *source* and fail with the compiler's diagnostics on error."""
    result = subprocess.run(
        [_compiler(), "-std=c99", "-O2", "-shared", "-fPIC", "-o", str(output), str(source), "-lm"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert output.is_file() and output.stat().st_size > 0
    return output


@pytest.fixture(scope="session")
def production_kernel(tmp_path_factory: pytest.TempPathFactory, record_testsuite_property) -> MindMemKernel:
    """Build and load the actual checked-in C implementation."""
    assert PRODUCTION_C_SOURCE.is_file(), f"missing production C source: {PRODUCTION_C_SOURCE}"
    output = tmp_path_factory.mktemp("native-c-abi") / "libmindmem.so"
    source_bytes = PRODUCTION_C_SOURCE.read_bytes()
    _compile_shared(PRODUCTION_C_SOURCE, output)
    assert PRODUCTION_C_SOURCE.read_bytes() == source_bytes, "C source changed during compilation"
    record_testsuite_property("native_c_source_sha256", hashlib.sha256(source_bytes).hexdigest())
    record_testsuite_property("native_c_artifact_path", str(output))
    record_testsuite_property("native_c_artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())
    record_testsuite_property("native_c_compiler", _compiler())
    kernel = MindMemKernel(str(output))
    assert kernel._lib._name == str(output), "consumer loaded a different library"
    return kernel


def test_production_c_library_loads_through_actual_ctypes_bridge(production_kernel: MindMemKernel):
    """A successful fixture construction alone must not be the only control."""
    assert production_kernel.rrf_fuse_py([1.0], [1.0]) == pytest.approx([2.0 / 61.0], abs=1e-9, rel=2e-7)


@pytest.mark.parametrize(
    ("bm25", "vector", "k", "bm25_weight", "vector_weight"),
    [
        # A non-symmetric rank permutation gives distinct expected scores.
        ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0], 60.0, 1.0, 1.0),
        ([0.0, 10.0, 100.0], [3.0, 3.0, 3.0], 10.0, 0.5, 2.0),
    ],
)
def test_rrf_matches_named_python_reference_vectors(
    production_kernel: MindMemKernel,
    bm25: list[float],
    vector: list[float],
    k: float,
    bm25_weight: float,
    vector_weight: float,
):
    expected = [bm25_weight / (k + bm25_rank) + vector_weight / (k + vector_rank) for bm25_rank, vector_rank in zip(bm25, vector)]
    actual = production_kernel.rrf_fuse_py(bm25, vector, k, bm25_weight, vector_weight)
    assert actual == pytest.approx(expected, abs=1e-9, rel=2e-7)


def test_rrf_empty_vectors_are_a_valid_empty_result(production_kernel: MindMemKernel):
    assert production_kernel.rrf_fuse_py([], []) == []


@pytest.mark.parametrize(
    ("scores", "k", "expected"),
    [
        ([0.9, 0.7, 0.8, 0.1], 2, [True, False, True, False]),
        ([0.5, 0.5, 0.4], 1, [True, False, False]),
        ([0.5, 0.5, 0.4], 2, [True, True, False]),
        ([2.0, 2.0, 3.0], 2, [True, False, True]),
        ([0.9, 0.7], 0, [False, False]),
        ([0.9, 0.7], 2, [True, True]),
        ([], 0, []),
        ([-2e30, -3e30], 1, [True, False]),
        ([-1e30, -1e30, -1e30], 2, [True, True, False]),
        ([1.0, -2e30, -3e30, -3.4028235e38], 3, [True, True, True, False]),
        # ── Non-finite contract (lib/kernels.c top_k_mask), UNIFORM across the
        # k<n partial-selection loop AND the k>=n fast path: a score is selected
        # iff it is > -INFINITY (finite or +inf). NaN and -inf are NEVER
        # selected; +inf ranks highest. Exercised at k<n, k=n, and k>n because
        # the fast path (k>=n) previously masked non-finite slots blindly.
        # -- k < n (partial-selection loop) --
        ([float("nan"), 1.0], 1, [False, True]),
        ([float("inf"), 2.0, 1.0], 1, [True, False, False]),
        # -- k == n (fast path; the cases that failed before the fix) --
        ([float("nan"), float("-inf")], 2, [False, False]),
        ([float("nan"), 1.0], 2, [False, True]),
        ([float("inf"), float("-inf")], 2, [True, False]),
        ([0.5, 0.7], 2, [True, True]),  # all-finite k==n still selects all
        # -- k > n (fast path) --
        ([float("nan"), 2.0], 3, [False, True]),
        ([float("inf"), float("-inf"), float("nan")], 5, [True, False, False]),
        # -- missing edge controls: negative k, signed-zero tie, empty input
        # with positive k, and under-filled all-ineligible partial selection --
        ([1.0, -2.0], -1, [False, False]),
        ([-0.0, 0.0, -1.0], 1, [True, False, False]),
        ([], 4, []),
        ([float("nan"), float("-inf")], 1, [False, False]),
    ],
)
def test_top_k_mask_boundaries_and_input_order_ties(production_kernel: MindMemKernel, scores: list[float], k: int, expected: list[bool]):
    # The C ABI receives float32 values.  Normalize once so the independent
    # Python reference and actual consumer are checked on identical inputs.
    normalized_scores = [ctypes.c_float(score).value for score in scores]
    assert py_top_k_mask(normalized_scores, k) == expected
    assert production_kernel.top_k_mask_py(normalized_scores, k) == expected


@pytest.mark.parametrize("k", [2**31, 2**32, -(2**31) - 1, -(2**32), 10**100, -(10**100)])
def test_top_k_mask_rejects_k_outside_native_c_int_range(production_kernel: MindMemKernel, k: int):
    with pytest.raises(OverflowError, match="native C int"):
        production_kernel.top_k_mask_py([1.0, 0.0], k)


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (-(1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)), [False, False]),
        ((1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)) - 1, [True, True]),
    ],
)
def test_top_k_mask_accepts_native_c_int_endpoints(production_kernel: MindMemKernel, k: int, expected: list[bool]):
    assert production_kernel.top_k_mask_py([1.0, 0.0], k) == expected


@pytest.mark.parametrize("k", [1.5, "2", None])
def test_top_k_mask_preserves_non_index_type_refusal(production_kernel: MindMemKernel, k: object):
    with pytest.raises(TypeError, match="interpreted as an integer"):
        production_kernel.top_k_mask_py([1.0, 0.0], k)  # type: ignore[arg-type]


def _compile_version_provider(tmp_path: Path, version: str) -> Path:
    """Build a symbol-only test provider; this is not the production C ABI."""
    source = tmp_path / "version_provider.c"
    source.write_text(
        f'const char *mindmem_version(void) {{ return "{version}"; }}\n',
        encoding="utf-8",
    )
    return _compile_shared(source, tmp_path / "libversion-provider.so")


def test_production_c_library_has_honest_optional_no_version_contract(production_kernel: MindMemKernel):
    assert production_kernel.so_version() is None
    assert production_kernel.version_compatible() is None


def test_version_reporting_provider_accepts_matching_major_minor(tmp_path: Path):
    major, minor = __version__.split(".")[:2]
    version = f"{major}.{minor}.9999"
    kernel = MindMemKernel(str(_compile_version_provider(tmp_path, version)))
    assert kernel.so_version() == version
    assert kernel.version_compatible() is True


def test_version_reporting_provider_reports_mismatched_major_minor(tmp_path: Path):
    major, minor = __version__.split(".")[:2]
    version = f"{int(major) + 1}.{minor}.0"
    kernel = MindMemKernel(str(_compile_version_provider(tmp_path, version)))
    assert kernel.so_version() == version
    assert kernel.version_compatible() is False
