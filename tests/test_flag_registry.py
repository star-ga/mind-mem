# Copyright 2026 STARGA, Inc.
"""The three-state flag registry: WIRED / KILL_SWITCH / UNIMPLEMENTED.

A declared flag with no consumer is a promise the code does not keep. An
operator writes ``{"redaction": {"enabled": true}}``, nothing reads it, and
the product answers with silence — which is indistinguishable from success.
Twenty of the fifty-two declared v4 flags were in exactly that state.

These tests hold three things:

1. **The classification cannot go stale.** Consumers are resolved from the
   AST of ``src/``, not from a grep, and a flag whose declared state
   disagrees with what the tree actually contains fails the build by name.
   A flag that loses its last consumer flips to ``UNIMPLEMENTED`` and says so.
2. **An absent capability refuses.** Enabling an ``UNIMPLEMENTED`` flag
   raises :class:`UnimplementedCapabilityError` naming the flag, and the
   error is not catchable as ``FeatureDisabledError`` — a caller falling
   back to the v3 path must not swallow "this does not exist".
3. **The search actually happened.** Every negative assertion here is
   paired with a positive control: the resolver is shown finding each of
   the five consumer shapes that ship, and the classification check is
   shown going red under a mutated registry. An empty finding list from a
   scan that could not see anything is a vacuous pass, not a clean one.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from mind_mem.v4 import feature_flags
from mind_mem.v4.feature_flags import (
    FeatureDisabledError,
    enabled_unimplemented_flags,
    is_enabled,
    is_enabled_quiet,
    is_kill_switch_active,
    require_enabled,
    require_valid_flag_config,
)
from mind_mem.v4.flag_registry import (
    FLAG_STATES,
    KILL_SWITCH,
    UNIMPLEMENTED,
    WIRED,
    FlagRecord,
    FlagState,
    UnimplementedCapabilityError,
    classification_drift,
    kill_switch_call_sites,
    resolve_consumers,
)

#: A flag the registry declares UNIMPLEMENTED, used as the sample subject for
#: every refusal test. Guarded below: if it ever gains a consumer the guard
#: fails loudly instead of these tests quietly stopping to test anything.
_ABSENT = "redaction"

#: A flag the registry declares WIRED — the positive control for every
#: refusal test, so "it raised" is never confused with "it always raises".
_PRESENT = "federation"


@functools.lru_cache(maxsize=1)
def _real_consumers() -> dict[str, tuple[str, ...]]:
    """The consumer map for the shipped tree, parsed once for the module.

    ``resolve_consumers`` re-parses every module under ``src/`` (~4s). Ten
    assertions want the same answer, and re-deriving it ten times buys
    nothing but wall-clock on a shared box. The production function stays
    uncached: an ``audit()`` caller must see the tree as it is now.
    """
    return dict(resolve_consumers())


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the warning-dedup slots and the stat-keyed probe cache.

    Two tests writing the same tmp path in the same nanosecond with the same
    byte count would otherwise share a cache entry, and the second would read
    the first one's answer.
    """
    monkeypatch.setattr(feature_flags, "_last_config_warning", None)
    monkeypatch.setattr(feature_flags, "_last_unimplemented_warning", None)
    feature_flags._QUIET_CACHE.clear()


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the flag reader at a config file this test writes."""
    path = tmp_path / "mind-mem.json"
    monkeypatch.setenv("MIND_MEM_CONFIG", str(path))

    def _write(payload: object) -> Path:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        path.write_text(text, encoding="utf-8")
        feature_flags._QUIET_CACHE.clear()
        return path

    return _write


class _Recorder:
    """Captures structured log events instead of emitting them."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def _record(self, event: str, **kwargs: object) -> None:
        self.events.append((event, dict(kwargs)))

    debug = info = warning = error = _record


# ---------------------------------------------------------------------------
# Guards — the samples must stay real, or every test below tests nothing
# ---------------------------------------------------------------------------


def test_sample_flags_are_registered_and_in_the_states_they_claim() -> None:
    assert _ABSENT in feature_flags.ALL_V4_FLAGS
    assert _PRESENT in feature_flags.ALL_V4_FLAGS
    assert _ABSENT in UNIMPLEMENTED
    assert _PRESENT in WIRED


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    def test_every_declared_flag_has_exactly_one_state(self) -> None:
        declared = set(feature_flags.ALL_V4_FLAGS)
        registered = set(FLAG_STATES)
        assert registered == declared, (
            f"flags declared but unclassified: {sorted(declared - registered)}; classified but undeclared: {sorted(registered - declared)}"
        )

    def test_the_three_states_partition_the_registry(self) -> None:
        assert WIRED & KILL_SWITCH == frozenset()
        assert WIRED & UNIMPLEMENTED == frozenset()
        assert KILL_SWITCH & UNIMPLEMENTED == frozenset()
        assert WIRED | KILL_SWITCH | UNIMPLEMENTED == set(FLAG_STATES)

    def test_there_is_no_fourth_state(self) -> None:
        assert {s.value for s in FlagState} == {"wired", "kill_switch", "unimplemented"}
        assert all(record.state in set(FlagState) for record in FLAG_STATES.values())

    def test_every_unimplemented_flag_keeps_its_declaration_and_a_wiring_note(self) -> None:
        """Deletion discipline: no consumer is a question about wiring, not a verdict.

        The cheapest way to make this whole gate green is to delete the
        twenty flags nothing reads. That is the one repair that is never
        correct, so each one must still be declared and must carry a note
        saying what wiring it would take.
        """
        for name in sorted(UNIMPLEMENTED):
            assert name in feature_flags.ALL_V4_FLAGS, f"{name} was deleted rather than wired"
            note = FLAG_STATES[name].note
            assert len(note) > 40, f"{name} has no wiring note"
            assert "Wiring question" in note or "does not exist" in note, f"{name}: note records no upgrade path"


# ---------------------------------------------------------------------------
# The resolver — positive controls first: prove it can see each shape
# ---------------------------------------------------------------------------


_SHAPES = {
    "literal": 'from mind_mem.v4.feature_flags import is_enabled\n\ndef go():\n    return is_enabled("pq")\n',
    "module_constant": (
        'from mind_mem.v4.feature_flags import is_enabled_quiet\n\nFLAG = "pq"\n\ndef go():\n    return is_enabled_quiet(FLAG)\n'
    ),
    "imported_constant": (
        "from mind_mem.v4.feature_flags import is_enabled_quiet\n"
        "from .holder import THE_FLAG\n\n"
        "def go():\n    return is_enabled_quiet(THE_FLAG)\n"
    ),
    "local_wrapper": (
        "from mind_mem.v4.feature_flags import is_enabled_for_workspace\n\n"
        "def _probe(ws, flag):\n    return is_enabled_for_workspace(ws, flag)\n\n"
        'def go(ws):\n    return _probe(ws, "pq")\n'
    ),
    "raw_v4_block": (
        "import json\n\n"
        "def go(path):\n"
        "    data = json.loads(open(path).read())\n"
        '    block = data.get("v4") or {}\n'
        '    sub = block.get("pq")\n'
        '    return isinstance(sub, dict) and sub.get("enabled") is True\n'
    ),
}


class TestResolverSeesEveryShipedShape:
    @pytest.mark.parametrize("shape", sorted(_SHAPES))
    def test_each_consumer_shape_is_resolved(self, shape: str, tmp_path: Path) -> None:
        """Positive control for the scan itself.

        All five shapes ship in ``src/`` today. A resolver that only knew
        the literal form would report five live consumers as absent, and
        the registry would then refuse three shipping features. So each
        shape is proven visible in isolation before the real tree is judged.
        """
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "holder.py").write_text('THE_FLAG = "pq"\n', encoding="utf-8")
        (pkg / "consumer.py").write_text(_SHAPES[shape], encoding="utf-8")

        found = resolve_consumers(tmp_path)

        assert found["pq"], f"resolver blind to the {shape} shape"
        assert any(site.startswith("pkg/consumer.py:") for site in found["pq"])

    def test_an_empty_tree_yields_no_consumers(self, tmp_path: Path) -> None:
        """The negative half of the control: the scan can also say "nothing".

        Without this, "the resolver found consumers" could just mean it
        returns a non-empty answer regardless of input.
        """
        found = resolve_consumers(tmp_path)
        assert all(sites == () for sites in found.values())
        assert set(found) == set(FLAG_STATES)

    def test_a_declaration_is_not_a_consumer(self, tmp_path: Path) -> None:
        """Naming a flag in the registry modules must not satisfy the registry."""
        pkg = tmp_path / "mind_mem" / "v4"
        pkg.mkdir(parents=True)
        (pkg / "feature_flags.py").write_text('ALL = ("pq",)\n', encoding="utf-8")
        (pkg / "flag_registry.py").write_text('NAMES = ("pq",)\n', encoding="utf-8")
        assert resolve_consumers(tmp_path)["pq"] == ()

    @pytest.mark.parametrize(
        "flag",
        [
            "trajectory",  # constant imported from another module
            "lint",  # raw v4-block read, literal key
            "logging_context",  # raw v4-block read, module-constant key
            "ingest_serve",  # raw v4-block read, module-constant key
            "context_budget",  # literal through a local forwarding wrapper
            "retrieval_metrics",  # literal through a local forwarding wrapper
        ],
    )
    def test_grep_invisible_consumers_are_found_in_the_real_tree(self, flag: str) -> None:
        """Six live consumers that a ``grep 'is_enabled("x")'`` cannot see.

        Each one was measured missing under a literal-only scan. If the
        resolver regresses to grep behaviour these six go dark first, and
        the registry would refuse six shipping features.
        """
        assert _real_consumers()[flag], f"{flag} consumer went invisible to the resolver"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestDeclaredStatesMatchTheTree:
    def test_no_classification_drift(self) -> None:
        drift = classification_drift(_real_consumers())
        assert drift == (), "\n".join(
            f"{flag}: declared {declared.value}, source tree says {actual.value} "
            f"({len(sites)} consumer(s): {', '.join(sites[:3]) or 'none'})"
            for flag, declared, actual, sites in drift
        )

    def test_every_wired_flag_has_a_consumer(self) -> None:
        consumers = _real_consumers()
        orphans = sorted(name for name in WIRED | KILL_SWITCH if not consumers[name])
        assert orphans == [], f"declared wired but nothing reads them: {orphans}"

    def test_no_src_call_site_probes_an_unimplemented_flag(self) -> None:
        consumers = _real_consumers()
        wired_but_unimplemented = {name: consumers[name] for name in sorted(UNIMPLEMENTED) if consumers[name]}
        assert wired_but_unimplemented == {}, (
            f"these flags gained a consumer and must move out of UNIMPLEMENTED — enabling them currently raises: {wired_but_unimplemented}"
        )

    def test_kill_switch_reads_only_target_declared_kill_switches(self) -> None:
        """A default-ON read of a non-kill-switch flag inverts its meaning.

        Nothing at runtime can notice that mistake, so it is caught here at
        zero runtime cost. Vacuous while no call site exists, which is why
        ``TestMutationTwin`` drives the same checker over a tree that does
        have one.
        """
        misused = {flag: sites for flag, sites in kill_switch_call_sites().items() if flag not in KILL_SWITCH}
        assert misused == {}, f"is_kill_switch_active called on non-kill-switch flags: {misused}"


# ---------------------------------------------------------------------------
# Fail loudly — the load-bearing behaviour
# ---------------------------------------------------------------------------


class TestEnablingAnAbsentCapabilityRefuses:
    def test_is_enabled_raises_and_names_the_flag(self, config) -> None:
        config({"v4": {_ABSENT: {"enabled": True}}})
        with pytest.raises(UnimplementedCapabilityError) as exc:
            is_enabled(_ABSENT)
        assert _ABSENT in str(exc.value)
        assert "declared but not implemented" in str(exc.value)

    def test_the_same_config_shape_still_enables_a_wired_flag(self, config) -> None:
        """Positive control: the refusal is about the flag, not the shape."""
        config({"v4": {_PRESENT: {"enabled": True}}})
        assert is_enabled(_PRESENT) is True
        require_enabled(_PRESENT)  # must not raise

    def test_an_unenabled_absent_flag_is_simply_off(self, config) -> None:
        """Silence is only wrong when the operator asked for the capability."""
        config({"v4": {}})
        assert is_enabled(_ABSENT) is False

    def test_a_bare_true_still_cannot_enable_anything(self, config) -> None:
        config({"v4": {_ABSENT: True}})
        assert is_enabled(_ABSENT) is False

    def test_require_enabled_refuses_regardless_of_config(self, config) -> None:
        """``require_enabled`` must not tell an operator to enable a lie.

        The old message said "Enable via mind-mem.json" — advice that, for
        a flag with nothing behind it, sends the diagnosis somewhere there
        is nothing to find.
        """
        config({"v4": {}})
        with pytest.raises(UnimplementedCapabilityError):
            require_enabled(_ABSENT)

    def test_the_refusal_escapes_feature_disabled_handlers(self, config) -> None:
        """``except FeatureDisabledError`` means "fall back to v3".

        That is right for a surface that is off and wrong for one that does
        not exist, so the two errors must not be catchable as one.
        """
        assert not issubclass(UnimplementedCapabilityError, FeatureDisabledError)
        assert not issubclass(FeatureDisabledError, UnimplementedCapabilityError)
        config({"v4": {_ABSENT: {"enabled": True}}})
        with pytest.raises(UnimplementedCapabilityError):
            try:
                is_enabled(_ABSENT)
            except FeatureDisabledError:  # pragma: no cover - must not catch
                pytest.fail("the absent-capability refusal was swallowed as a disabled feature")

    def test_the_quiet_probe_stays_silent_and_never_raises(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe contract outranks the refusal.

        ``is_enabled_quiet`` is what an OFF-by-default surface calls to
        decide whether to run; if it raised or logged, a flag-off build
        would stop being indistinguishable from the build that never had
        the feature. It never faces this case in practice — a flag with no
        consumer has no probe, which the AST gate above enforces.
        """
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config({"v4": {_ABSENT: {"enabled": True}}})
        assert is_enabled_quiet(_ABSENT) is True
        assert recorder.events == []


class TestTheConfigReaderAnnouncesWhatNothingElseCan:
    def test_enabled_unimplemented_flags_are_reported(self, config) -> None:
        config({"v4": {_ABSENT: {"enabled": True}, _PRESENT: {"enabled": True}}})
        assert enabled_unimplemented_flags() == (_ABSENT,)

    def test_a_healthy_config_reports_nothing(self, config) -> None:
        """Positive control for the reporter: it can also answer "none"."""
        config({"v4": {_PRESENT: {"enabled": True}}})
        assert enabled_unimplemented_flags() == ()

    def test_the_warning_names_the_flags_once(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config({"v4": {_ABSENT: {"enabled": True}}})

        for _ in range(5):
            feature_flags.config_error()

        assert [event for event, _ in recorder.events] == ["v4_unimplemented_flags_enabled"]
        assert recorder.events[0][1]["flags"] == [_ABSENT]

    def test_a_clean_config_logs_nothing(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config({"v4": {_PRESENT: {"enabled": True}}})
        feature_flags.config_error()
        assert recorder.events == []

    def test_require_valid_flag_config_lists_every_offender(self, config) -> None:
        absent = sorted(UNIMPLEMENTED)[:3]
        config({"v4": {name: {"enabled": True} for name in absent}})
        with pytest.raises(UnimplementedCapabilityError) as exc:
            require_valid_flag_config()
        for name in absent:
            assert name in str(exc.value)

    def test_require_valid_flag_config_passes_a_clean_config(self, config) -> None:
        config({"v4": {_PRESENT: {"enabled": True}}})
        require_valid_flag_config()


# ---------------------------------------------------------------------------
# The kill-switch resolver — the inverted fail direction
# ---------------------------------------------------------------------------


class TestKillSwitchResolver:
    def test_absent_config_leaves_the_feature_on(self, config) -> None:
        config({"v4": {}})
        assert is_kill_switch_active(_PRESENT) is True
        # Control: the opt-in resolver reads the same config as OFF. The two
        # answers differ on identical input, which is the whole point.
        assert is_enabled(_PRESENT) is False

    def test_no_config_file_at_all_leaves_the_feature_on(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "absent.json"))
        assert is_kill_switch_active(_PRESENT) is True

    def test_explicit_false_turns_it_off(self, config) -> None:
        config({"v4": {_PRESENT: {"enabled": False}}})
        assert is_kill_switch_active(_PRESENT) is False

    def test_explicit_true_leaves_it_on(self, config) -> None:
        config({"v4": {_PRESENT: {"enabled": True}}})
        assert is_kill_switch_active(_PRESENT) is True

    def test_a_malformed_config_never_disables_a_shipping_feature(self, config) -> None:
        """Fails ON, the mirror of ``is_enabled`` failing closed.

        A trailing comma must not silently switch off a feature that is
        running in production, just as it must not switch one on.
        """
        config('{"v4": {"federation": {"enabled": false},}}')  # trailing comma
        assert is_kill_switch_active(_PRESENT) is True
        assert is_enabled(_PRESENT) is False

    def test_an_unknown_flag_name_never_disables_anything(self, config) -> None:
        config({"v4": {"not-a-flag": {"enabled": False}}})
        assert is_kill_switch_active("not-a-flag") is True

    def test_a_non_dict_section_is_not_an_off_switch(self, config) -> None:
        config({"v4": {_PRESENT: False}})
        assert is_kill_switch_active(_PRESENT) is True

    def test_it_emits_nothing(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder()
        monkeypatch.setattr(feature_flags, "_log", recorder)
        config('{"v4": {"federation": {"enabled": false},}}')
        assert is_kill_switch_active(_PRESENT) is True
        assert recorder.events == []


# ---------------------------------------------------------------------------
# Mutation twins — every gate above is shown going red
# ---------------------------------------------------------------------------


def _mutated(states, flag: str, state: FlagState):
    table = dict(states)
    table[flag] = FlagRecord(name=flag, state=state, note=table[flag].note)
    return table


class TestMutationTwin:
    """Break the thing each gate guards; watch the gate go red.

    A gate never observed failing is not a gate. Each test here drives the
    same checker the real assertions use, over a deliberately wrong input,
    and fails if the checker still reports clean.
    """

    def test_declaring_an_unimplemented_flag_wired_is_caught(self) -> None:
        consumers = _real_consumers()
        drift = classification_drift(consumers, _mutated(FLAG_STATES, _ABSENT, FlagState.WIRED))
        assert [row[0] for row in drift] == [_ABSENT]
        assert drift[0][1] is FlagState.WIRED and drift[0][2] is FlagState.UNIMPLEMENTED

    def test_declaring_a_wired_flag_unimplemented_is_caught(self) -> None:
        consumers = _real_consumers()
        drift = classification_drift(consumers, _mutated(FLAG_STATES, _PRESENT, FlagState.UNIMPLEMENTED))
        assert [row[0] for row in drift] == [_PRESENT]
        assert drift[0][2] is FlagState.WIRED

    def test_a_flag_losing_its_last_consumer_is_caught(self) -> None:
        """The scenario the registry exists for: wiring is deleted, state is not."""
        consumers = dict(_real_consumers())
        assert consumers[_PRESENT], "positive control: the flag must have consumers to lose"
        consumers[_PRESENT] = ()
        drift = classification_drift(consumers, FLAG_STATES)
        assert [row[0] for row in drift] == [_PRESENT]
        assert drift[0][2] is FlagState.UNIMPLEMENTED

    def test_declaring_a_kill_switch_without_wiring_it_is_caught(self) -> None:
        consumers = _real_consumers()
        drift = classification_drift(consumers, _mutated(FLAG_STATES, _ABSENT, FlagState.KILL_SWITCH))
        assert [row[0] for row in drift] == [_ABSENT]
        assert drift[0][2] is FlagState.UNIMPLEMENTED

    def test_a_kill_switch_read_of_a_non_kill_switch_flag_is_caught(self, tmp_path: Path) -> None:
        """Drives the checker that is vacuous on today's tree over one that isn't."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "bad.py").write_text(
            f'from mind_mem.v4.feature_flags import is_kill_switch_active\n\ndef go():\n    return is_kill_switch_active("{_ABSENT}")\n',
            encoding="utf-8",
        )

        sites = kill_switch_call_sites(tmp_path)

        assert _ABSENT in sites, "the kill-switch checker cannot see a call site"
        misused = {flag: found for flag, found in sites.items() if flag not in KILL_SWITCH}
        assert misused, "the kill-switch checker reported clean over a tree that misuses it"

    def test_a_blind_resolver_makes_the_whole_registry_drift(self, tmp_path: Path) -> None:
        """If the scan stops seeing the tree, the gate must go red, not green.

        A verifier that died is not a verifier that passed: an all-empty
        consumer map has to read as total drift, never as a clean bill.
        """
        drift = classification_drift(resolve_consumers(tmp_path), FLAG_STATES)
        assert len(drift) == len(WIRED | KILL_SWITCH)
        assert all(row[2] is FlagState.UNIMPLEMENTED for row in drift)
