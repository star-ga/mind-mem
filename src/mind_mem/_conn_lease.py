"""In-flight protection for cached ConnectionManagers.

Closing a sqlite3 handle while a statement runs on it is a use-after-free in
the C layer (it segfaults the interpreter), and the raw read connections a
ConnectionManager hands out carry no lock ``close()`` could wait on. A cache
that closes managers on eviction therefore must not close one that is mid-call.

``close_or_retire`` closes an idle manager immediately and only marks a leased
one retired; the last ``release`` closes it. The check and the close happen
under one lock, so a ``lease`` cannot slip in between them.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from .connection_manager import ConnectionManager
from .observability import get_logger

_log = get_logger("conn_lease")


class ConnLeases:
    """Lease counts and retirement set for one cache."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[ConnectionManager, int] = {}
        self._retired: set[ConnectionManager] = set()

    def lease(self, mgr: ConnectionManager) -> None:
        with self._lock:
            self._leases[mgr] = self._leases.get(mgr, 0) + 1

    def release(self, mgr: ConnectionManager) -> None:
        with self._lock:
            left = self._leases.get(mgr, 1) - 1
            if left > 0:
                self._leases[mgr] = left
                return
            self._leases.pop(mgr, None)
            if mgr in self._retired:
                self._retired.discard(mgr)
                self._close_quietly(mgr)

    def close_or_retire(self, mgr: ConnectionManager) -> None:
        """Close *mgr* now if idle, else when its last lease is released."""
        with self._lock:
            if mgr in self._leases:
                self._retired.add(mgr)
            else:
                self._close_quietly(mgr)

    @contextlib.contextmanager
    def leased(self, mgr: ConnectionManager) -> Iterator[ConnectionManager]:
        self.lease(mgr)
        try:
            yield mgr
        finally:
            self.release(mgr)

    @staticmethod
    def _close_quietly(mgr: ConnectionManager) -> None:
        try:
            mgr.close()
        except Exception:  # nosec B110 - eviction/teardown must not be blocked by a dying manager
            _log.warning("conn_lease_close_failed", db_path=mgr.db_path)
