"""Conditional setup hook for the optional Cython accelerator.

mind-mem is a pure-Python package by default; everything declared in
``pyproject.toml`` builds without this file. We add ``setup.py`` only
to opt-in to the Cython-compiled accelerator at
``src/mind_mem/_mic_map_accel.pyx`` when Cython is installed at build
time.

When Cython is NOT installed:

* The extension is silently skipped — no error, no warning.
* The wheel is the same pure-Python wheel as before.
* ``mind_mem.mic_map`` falls back to its pure-Python codec
  transparently.

When Cython IS installed (e.g. ``pip install mind-mem[accelerated]``):

* The .pyx is compiled to an extension module at install time.
* ``mind_mem.mic_map`` imports the accelerator and uses it for
  ULEB128 / read_exact hot paths.
* Per-platform behaviour is identical — the accelerator is a perf
  optimisation, never a behaviour change.

This keeps the default install path zero-toolchain (PyPI wheels are
pure Python, no C compiler needed) while letting operators who care
about throughput opt into the C path.
"""

from setuptools import setup

try:
    from Cython.Build import cythonize  # type: ignore[import-untyped]
except ImportError:
    # Cython not present — install pure Python without the extension.
    setup()
else:
    setup(
        ext_modules=cythonize(
            ["src/mind_mem/_mic_map_accel.pyx"],
            language_level=3,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
        ),
    )
