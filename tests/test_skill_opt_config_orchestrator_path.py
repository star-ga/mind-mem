"""The orchestrator location must be settable from OUTSIDE the installed package.

``skill_opt`` reaches the multi-LLM fleet through exactly one path:
``fleet_bridge._load_orchestrator`` puts ``config.ORCHESTRATOR_PATH`` on
``sys.path`` and imports ``providers`` / ``config`` from it. That path used to
be a hardcoded development-machine plugin directory, so after ``pip install
mind-mem`` on any other machine the feature was unreachable and the only
remediation the code offered was editing installed package source.

These tests drive :func:`resolve_orchestrator_path` with an explicit mapping
rather than reloading the module. ``importlib.reload`` re-executes a module in
its EXISTING ``__dict__``, so every already-imported function keeps a globals
dict that now holds a freshly built ``SkillOptConfig`` class — instances then
compare unequal to the class other test modules bound at their own import time.
Measured, not theorised: reloading here failed
``test_skill_opt.py::TestConfig::test_load_missing_file``.
"""

from __future__ import annotations

import os

from mind_mem.skill_opt.config import (
    DEFAULT_ORCHESTRATOR_PATH,
    ORCHESTRATOR_PATH,
    ORCHESTRATOR_PATH_ENV,
    resolve_orchestrator_path,
)


def test_env_override_relocates_the_orchestrator() -> None:
    resolved = resolve_orchestrator_path({ORCHESTRATOR_PATH_ENV: "/opt/fleet/multi-llm-orchestrator"})
    assert resolved == "/opt/fleet/multi-llm-orchestrator"


def test_env_override_expands_a_home_relative_path() -> None:
    """``~`` in an operator-supplied value must expand like the default does."""
    resolved = resolve_orchestrator_path({ORCHESTRATOR_PATH_ENV: "~/my-orchestrator"})
    assert resolved == os.path.expanduser("~/my-orchestrator")
    assert "~" not in resolved


def test_empty_env_value_falls_back_to_the_default() -> None:
    """An exported-but-empty variable must not resolve the import root to the cwd."""
    resolved = resolve_orchestrator_path({ORCHESTRATOR_PATH_ENV: ""})
    assert resolved == os.path.expanduser(DEFAULT_ORCHESTRATOR_PATH)
    assert resolved != os.path.abspath("")


def test_unset_env_keeps_the_documented_default() -> None:
    assert resolve_orchestrator_path({}) == os.path.expanduser(DEFAULT_ORCHESTRATOR_PATH)


def test_the_module_constant_is_what_the_resolver_returns() -> None:
    """Pins the WIRING, not just the resolver.

    A resolver nothing calls is the same dead weight as the constant this
    replaced, so assert the published constant is its output for the ambient
    environment — otherwise these tests would pass over an unused function.
    """
    assert ORCHESTRATOR_PATH == resolve_orchestrator_path(dict(os.environ))


def test_dead_env_path_constant_is_gone() -> None:
    """``ENV_PATH`` documented a key source no code ever read.

    API keys come from the injected orchestrator's own ``get_api_keys()``
    (``fleet_bridge._load_orchestrator``), never from this module. A constant
    naming a file the package never opens is a false statement about where
    credentials live, so it must not come back.
    """
    import mind_mem.skill_opt.config as config_mod

    assert not hasattr(config_mod, "ENV_PATH")
