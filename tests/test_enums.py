"""Tests for centralised enums (mind_mem.enums)."""

from __future__ import annotations

from mind_mem.enums import TaskStatus


class TestTaskStatus:
    def test_values_match_wire_format(self):
        """On-disk values must be exact lowercase strings."""
        assert TaskStatus.TODO == "todo"
        assert TaskStatus.DOING == "doing"
        assert TaskStatus.BLOCKED == "blocked"
        assert TaskStatus.DONE == "done"
        assert TaskStatus.CANCELED == "canceled"

    def test_str_enum_serialises_as_string(self):
        """Behaves as a plain string for any str-accepting API.

        Avoids asserting on f-string output because ``str.__format__``
        dispatch for str-mixin enums diverged in Python 3.11 (PEP 663):
        3.10 yields the value ("todo"), 3.11+ yields the enum repr.
        The stable contract is membership in ``str`` plus ``.value``
        being the canonical wire-format literal.
        """
        import json

        assert isinstance(TaskStatus.TODO, str)
        assert TaskStatus.TODO == "todo"
        assert TaskStatus.TODO.value == "todo"
        # JSON dump round-trip — what every persistence path actually does.
        assert json.dumps(TaskStatus.TODO) == '"todo"'

    def test_open_closed_partition(self):
        """open() + closed() must cover the whole universe and be disjoint."""
        all_states = set(TaskStatus)
        open_states = set(TaskStatus.open())
        closed_states = set(TaskStatus.closed())
        assert open_states.isdisjoint(closed_states)
        assert open_states | closed_states == all_states

    def test_is_open_is_closed(self):
        assert TaskStatus.TODO.is_open()
        assert TaskStatus.DOING.is_open()
        assert TaskStatus.BLOCKED.is_open()
        assert not TaskStatus.DONE.is_open()
        assert not TaskStatus.CANCELED.is_open()
        assert TaskStatus.DONE.is_closed()
        assert TaskStatus.CANCELED.is_closed()

    def test_lookup_by_value(self):
        """Reverse lookup from a stored literal."""
        assert TaskStatus("todo") is TaskStatus.TODO
        assert TaskStatus("done") is TaskStatus.DONE
