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

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

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
        ([0.9, 0.7], 0, [False, False]),
        ([0.9, 0.7], 2, [True, True]),
        ([], 0, []),
    ],
)
def test_top_k_mask_boundaries_and_input_order_ties(production_kernel: MindMemKernel, scores: list[float], k: int, expected: list[bool]):
    assert production_kernel.top_k_mask_py(scores, k) == expected


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
