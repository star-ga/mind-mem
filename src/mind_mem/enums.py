"""Centralised enum definitions for mind-mem.

Single source of truth for status strings and other closed-set values
duplicated across the codebase. Per the 2026-04-18 audit, the task
status literals (`todo`, `doing`, `blocked`, `done`, `canceled`) are
hardcoded in eight files — `sqlite_index.py`, `validate_py.py`,
`_recall_core.py`, `intel_scan.py`, `recall_vector.py`,
`_recall_constants.py`, `capture.py`, and the `validate.sh` bash
mirror. This module is the landing pad for those constants; the
call-site migration ships as a coordinated patch in v3.2.0.

All enums here inherit from ``str`` so they remain serialisation-
compatible with the on-disk Markdown schema. An existing string
``"todo"`` literal can be replaced with ``TaskStatus.TODO`` without
changing any file-format output or JSON payload.
"""

from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    """State machine for a Task block (``[T-YYYYMMDD-###]``).

    Lifecycle:

        TODO -> DOING -> DONE
           \\      ^       ^
            \\     |       |
             -> BLOCKED ---+
              \\
               -> CANCELED  (terminal)

    Transitions are enforced at the apply-engine level; the enum
    itself captures the universe of legal states.
    """

    TODO = "todo"
    DOING = "doing"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELED = "canceled"

    @classmethod
    def open(cls) -> frozenset["TaskStatus"]:
        """Statuses that still count as open loops."""
        return frozenset({cls.TODO, cls.DOING, cls.BLOCKED})

    @classmethod
    def closed(cls) -> frozenset["TaskStatus"]:
        """Terminal statuses."""
        return frozenset({cls.DONE, cls.CANCELED})

    def is_open(self) -> bool:
        return self in self.open()

    def is_closed(self) -> bool:
        return self in self.closed()


__all__ = ["TaskStatus"]
