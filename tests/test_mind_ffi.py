"""Tests for MIND FFI module."""

from __future__ import annotations

import os
import tempfile

from mind_mem.mind_ffi import get_kernel_param, get_mind_dir, load_kernel_config


def test_get_mind_dir():
    mind_dir = get_mind_dir("/tmp/test")
    assert isinstance(mind_dir, str)


def test_load_kernel_config_missing():
    result = load_kernel_config("/nonexistent/path/kernel.mind")
    assert isinstance(result, dict)


def test_load_kernel_config_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = load_kernel_config(path)
        assert isinstance(result, dict)
    finally:
        os.unlink(path)


def test_get_kernel_param_default():
    result = get_kernel_param({}, "bm25", "k1", 1.2)
    assert result == 1.2


def test_get_kernel_param_from_config():
    config = {"bm25": {"k1": 2.0}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 2.0


def test_get_kernel_param_missing_section():
    config = {"other": {"key": "val"}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 1.2


# ---------------------------------------------------------------------------
# Library loading (regression)
# ---------------------------------------------------------------------------


class _RecordingLog:
    """Stand-in for the module's structured logger."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def _record(self, event, **kwargs):
        self.events.append((event, kwargs))

    warning = _record
    info = _record
    error = _record
    debug = _record


def _kernel_lib_path():
    """The .so the search path would find, or None."""
    from mind_mem.mind_ffi import _LIB_SEARCH_PATHS

    for p in _LIB_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def test_env_lib_outside_allowlist_is_reported(monkeypatch, tmp_path):
    """A MIND_MEM_LIB pointing outside the allowed directories was dropped in
    silence: the operator saw 'library not found' (or a different library
    loaded) with no sign that the path they set had been seen and rejected."""
    import mind_mem.mind_ffi as ffi

    rogue = tmp_path / "rogue.so"
    rogue.write_bytes(b"\x7fELF not a real library")
    monkeypatch.setenv("MIND_MEM_LIB", str(rogue))

    recorder = _RecordingLog()
    monkeypatch.setattr(ffi, "_log", recorder)
    try:
        ffi.MindMemKernel()
    except OSError:
        pass  # No kernel built here — the rejection still has to be reported.

    rejected = [kw for event, kw in recorder.events if event == "ffi_env_lib_rejected"]
    assert rejected, f"no rejection reported; events={recorder.events}"
    assert rejected[0]["path"] == str(rogue.resolve())
    assert "outside allowed directories" in rejected[0]["reason"]


def test_env_lib_missing_file_is_reported(monkeypatch, tmp_path):
    import mind_mem.mind_ffi as ffi

    lib_dir = _kernel_lib_path()
    if lib_dir is None:
        import pytest

        pytest.skip("no compiled MIND kernel in the search path")
    ghost = lib_dir.parent / "libmindmem-does-not-exist.so"
    monkeypatch.setenv("MIND_MEM_LIB", str(ghost))

    recorder = _RecordingLog()
    monkeypatch.setattr(ffi, "_log", recorder)
    try:
        ffi.MindMemKernel()
    except OSError:
        pass

    rejected = [kw for event, kw in recorder.events if event == "ffi_env_lib_rejected"]
    assert rejected, f"no rejection reported; events={recorder.events}"
    assert "does not exist" in rejected[0]["reason"]


def test_version_gate_reads_the_exported_symbol_and_keeps_the_verdict(monkeypatch):
    """The gate probed 'mindmem_get_version', which no build exports, and then
    threw away the bool it computed. Both halves were dead."""
    import pytest

    from mind_mem.mind_ffi import MindMemKernel

    lib = _kernel_lib_path()
    if lib is None:
        pytest.skip("no compiled MIND kernel in the search path")
    monkeypatch.delenv("MIND_MEM_LIB", raising=False)

    kernel = MindMemKernel()
    version = kernel.so_version()
    assert version, "version symbol was not read from the library"
    assert version[0].isdigit()
    # The verdict is now kept rather than discarded.
    assert kernel.version_compatible() is not None
    assert isinstance(kernel.version_compatible(), bool)
