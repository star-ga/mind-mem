# Copyright 2026 STARGA, Inc.
"""Platform helpers for the test suite.

Tests kept reaching for APIs that exist only on the runner that happened
to be in front of the author:

* ``os.geteuid`` is Unix-only. On Windows it does not exist at all, so
  ``os.geteuid() == 0`` raised AttributeError and reddened every Windows
  job -- including at import time, inside a ``skipif`` decorator, where it
  takes the whole module down rather than one test.
* ``datetime.UTC`` arrived in 3.11, but ``requires-python`` is ``>=3.10``
  and the matrix runs 3.10.

Neither is a product limitation; both are tests asserting the host.
"""

from __future__ import annotations

import os

__all__ = ["is_root"]


def is_root() -> bool:
    """True when the process can ignore filesystem permission bits.

    Windows has no ``geteuid`` and no equivalent notion here, so it
    answers False: permission-based tests are skipped there by their own
    filesystem checks, not by a uid that does not exist.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:  # Windows
        return False
    return geteuid() == 0
