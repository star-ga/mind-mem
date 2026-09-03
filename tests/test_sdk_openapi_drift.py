# Copyright 2026 STARGA, Inc.
"""The committed OpenAPI artifact must agree with the routes the server serves.

Roadmap RM-2323 asked for declarative specs so the two in-tree clients stop
being their own source of truth. Committing a spec is the easy half and the
dangerous one: a file that can quietly disagree with the server is worse than
no file, because a client generated from it fails at runtime against a server
that looks healthy. So the artifact ships with this gate, and the gate is the
deliverable.

Three independent legs, because each covers a different way the pair can rot:

1. **Structure** — regenerate from ``create_app`` and compare byte for byte
   after pinning ``info.version``. Any change to a path, verb, parameter,
   request body, response, security scheme or component schema shows up as a
   diff.
2. **Route census** — enumerate the app's own route objects and compare that
   set with the artifact's paths. This does not go through
   ``FastAPI.openapi()`` at all, so a route the generator drops (an
   ``include_in_schema=False`` added in passing, say) cannot hide behind a
   generator that agrees with itself.
3. **Version** — the artifact is a published document; one advertising the
   wrong release is a defect in its own right. Deliberately kept OUT of leg 1
   so a version bump does not turn every release commit red for a reason that
   has nothing to do with route drift.

Every negative assertion here is paired with a positive control that proves
the comparison can see a difference at all — a drift check that cannot go red
is a green light with no bulb behind it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed; skipping OpenAPI artifact tests")

from mind_mem import __version__ as PACKAGE_VERSION  # noqa: E402
from mind_mem.spec import export_openapi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = REPO_ROOT / "src" / "mind_mem" / "spec" / "export_openapi.py"
ARTIFACT = REPO_ROOT / "sdk" / "spec" / "openapi.json"


@pytest.fixture(scope="module")
def exporter() -> Any:
    """The exporter, imported by NAME.

    It used to be loaded with ``importlib.util.spec_from_file_location`` on the
    reasoning that ``sdk/`` is not a Python package — true, and irrelevant once
    the generator moved to ``src/mind_mem/spec/`` (see that package's
    docstring). The by-path load outlived its reason and cost something real:
    ``scripts/check_reachable_modules.py`` walks imports, a path load is not
    one, and CI invokes the module as ``python3 -m mind_mem.spec.export_openapi``
    which is not a path either — so the gate reported a shipped, working module
    as NEW unreachable debt. A plain import is the honest edge, and the
    ``mind-mem-openapi`` console script is the one the gate can see.
    """
    return export_openapi


@pytest.fixture(scope="module")
def live_spec(exporter: Any) -> dict[str, Any]:
    return exporter.build_live_spec()


@pytest.fixture(scope="module")
def committed_spec(exporter: Any) -> dict[str, Any]:
    return exporter.load_committed_spec()


class TestArtifactExists:
    def test_artifact_is_committed_and_parses(self, committed_spec: dict[str, Any]) -> None:
        assert committed_spec["openapi"].startswith("3."), committed_spec.get("openapi")
        assert committed_spec["info"]["title"] == "mind-mem REST API"

    def test_artifact_describes_a_real_surface(self, committed_spec: dict[str, Any]) -> None:
        # Positive control for every "X is present in the spec" assertion made
        # elsewhere in the suite: if the artifact were an empty skeleton, those
        # would all be vacuous.
        paths = committed_spec["paths"]
        assert len(paths) >= 10, f"only {len(paths)} paths — did the export run against a stripped app?"
        assert "/v1/recall" in paths
        assert "post" in paths["/v1/recall"]


class TestStructuralDrift:
    def test_committed_artifact_matches_live_routes(self, exporter: Any, committed_spec: dict[str, Any], live_spec: dict[str, Any]) -> None:
        diff = exporter.structural_diff(committed_spec, live_spec)
        assert diff == "", (
            "sdk/spec/openapi.json has drifted from the live routes.\n"
            "Regenerate with: python3 -m mind_mem.spec.export_openapi --write\n\n" + diff
        )

    def test_positive_control_a_renamed_path_is_detected(
        self, exporter: Any, committed_spec: dict[str, Any], live_spec: dict[str, Any]
    ) -> None:
        # Proof the comparator can fail: take the real artifact, rename one
        # path the way a server-side refactor would, and require a diff.
        mutated = copy.deepcopy(committed_spec)
        mutated["paths"]["/v1/recall-renamed"] = mutated["paths"].pop("/v1/recall")
        assert exporter.structural_diff(mutated, live_spec) != ""

    def test_positive_control_a_changed_verb_is_detected(
        self, exporter: Any, committed_spec: dict[str, Any], live_spec: dict[str, Any]
    ) -> None:
        # The exact defect the two in-tree clients carried: the right path
        # under the wrong verb.
        mutated = copy.deepcopy(committed_spec)
        mutated["paths"]["/v1/recall"]["get"] = mutated["paths"]["/v1/recall"].pop("post")
        assert exporter.structural_diff(mutated, live_spec) != ""

    def test_positive_control_a_dropped_request_body_is_detected(
        self, exporter: Any, committed_spec: dict[str, Any], live_spec: dict[str, Any]
    ) -> None:
        mutated = copy.deepcopy(committed_spec)
        mutated["paths"]["/v1/recall"]["post"].pop("requestBody")
        assert exporter.structural_diff(mutated, live_spec) != ""

    def test_export_is_deterministic(self, exporter: Any, tmp_path: Path) -> None:
        first = exporter.write_spec(tmp_path / "a.json")
        second = exporter.write_spec(tmp_path / "b.json")
        assert first == second
        assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()

    def test_write_reproduces_the_committed_bytes(self, exporter: Any, tmp_path: Path) -> None:
        regenerated = tmp_path / "openapi.json"
        exporter.write_spec(regenerated)
        assert regenerated.read_bytes() == ARTIFACT.read_bytes(), (
            "regenerating the artifact produces different bytes; run `mind-mem-openapi --write` and commit the result"
        )


class TestRouteCensus:
    """A second view of the surface that does not go through ``.openapi()``."""

    def test_every_schema_route_is_a_real_app_route(self, exporter: Any, committed_spec: dict[str, Any]) -> None:
        app_routes = self._app_routes(exporter)
        assert app_routes, "positive control: the app registered no routes at all"

        for path, operations in committed_spec["paths"].items():
            for method in operations:
                assert (method.upper(), path) in app_routes, f"{method.upper()} {path} is in the artifact but not on the app"

    def test_every_app_route_is_in_the_schema(self, exporter: Any, committed_spec: dict[str, Any]) -> None:
        documented = {(method.upper(), path) for path, operations in committed_spec["paths"].items() for method in operations}
        assert documented, "positive control: the artifact documented no operations"

        for method, path in self._app_routes(exporter):
            assert (method, path) in documented, f"{method} {path} is served but absent from sdk/spec/openapi.json"

    @staticmethod
    def _app_routes(exporter: Any) -> set[tuple[str, str]]:
        import os
        import tempfile

        from mind_mem.api.rest import create_app

        saved = os.environ.get("MIND_MEM_WORKSPACE")
        with tempfile.TemporaryDirectory(prefix="mind-mem-census-") as workspace:
            try:
                app = create_app(workspace)
            finally:
                if saved is None:
                    os.environ.pop("MIND_MEM_WORKSPACE", None)
                else:
                    os.environ["MIND_MEM_WORKSPACE"] = saved

        found: set[tuple[str, str]] = set()
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not isinstance(path, str) or not path.startswith("/v1/") or not methods:
                continue
            if not getattr(route, "include_in_schema", True):
                continue
            for method in methods:
                if method in {"HEAD", "OPTIONS"}:
                    continue
                found.add((method, path))
        return found


class TestArtifactVersion:
    def test_artifact_advertises_the_current_package_version(self, committed_spec: dict[str, Any]) -> None:
        assert committed_spec["info"]["version"] == PACKAGE_VERSION, (
            f"sdk/spec/openapi.json advertises {committed_spec['info']['version']!r} but the package is "
            f"{PACKAGE_VERSION!r}. Regenerate with: python3 -m mind_mem.spec.export_openapi --write"
        )

    def test_live_app_advertises_the_current_package_version(self, live_spec: dict[str, Any]) -> None:
        assert live_spec["info"]["version"] == PACKAGE_VERSION


class TestExporterCli:
    def test_check_passes_on_the_committed_tree(self, exporter: Any, capsys: Any) -> None:
        assert exporter._main(["--check"]) == 0

    def test_check_fails_on_a_corrupted_artifact(self, exporter: Any, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        # Positive control for the CLI's exit code, not just the comparator.
        corrupted = tmp_path / "openapi.json"
        spec = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        spec["paths"].pop("/v1/scan")
        corrupted.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        monkeypatch.setattr(exporter, "SPEC_PATH", corrupted)
        assert exporter._main(["--check"]) == 1


class TestExporterIsReachable:
    """The generator must be reachable by a route the gate can actually see.

    ``scripts/check_reachable_modules.py`` walks imports under ``src/``, entry
    points in ``[project.scripts]``, three consumer trees, and workflow steps
    that name a FILE PATH. The exporter is reached by none of those on its own:
    nothing in the package imports it, and CI runs it as
    ``python3 -m mind_mem.spec.export_openapi``, which is a module invocation,
    not a path. It therefore read as NEW unreachable debt while shipping and
    working — a false positive from the gate that exists to stop working code
    being deleted, which is the failure mode that costs the most.

    The console script is the fix, and this is the gate on the fix.
    """

    @staticmethod
    def _gate() -> Any:
        from scripts import check_reachable_modules as gate

        return gate

    def test_the_console_script_is_declared(self) -> None:
        scripts = self._console_scripts()
        assert scripts.get("mind-mem-openapi", "").startswith("mind_mem.spec.export_openapi:"), (
            f"mind-mem-openapi is gone or retargeted: {scripts.get('mind-mem-openapi')!r}"
        )

    def test_the_console_script_target_resolves(self) -> None:
        """A declared entry point that cannot be called is a dead door.

        The gate reads ``[project.scripts]`` as text and would count a
        misspelled target as a caller just the same, so the name is resolved
        here rather than trusted.
        """
        module_name, _, attr = self._console_scripts()["mind-mem-openapi"].partition(":")
        module = __import__(module_name, fromlist=[attr])
        assert callable(getattr(module, attr)), f"{module_name}:{attr} is not callable"

    def test_the_gate_sees_the_exporter_as_reached(self) -> None:
        unreachable, _stats = self._gate().scan()
        assert "spec.export_openapi" not in unreachable, (
            "the exporter is unreachable again — the console script is the only route the gate can see"
        )

    def test_positive_control_without_the_entry_point_it_is_unreachable(self, monkeypatch: Any) -> None:
        """Proof the assertion above is load-bearing rather than vacuous.

        Drop the entry-point leg and the module must come back as unreachable.
        If it does not, something else is holding it up and the console script
        could be deleted with the gate staying green — which is exactly the
        state this test exists to forbid.
        """
        gate = self._gate()
        monkeypatch.setattr(gate, "_entry_point_modules", set)
        unreachable, _stats = gate.scan()
        assert "spec.export_openapi" in unreachable, (
            "with no entry points the exporter is STILL reachable — this test proves nothing as written"
        )

    @staticmethod
    def _console_scripts() -> dict[str, str]:
        try:
            import tomllib as toml
        except ModuleNotFoundError:  # pragma: no cover - python 3.10 only
            import tomli as toml  # type: ignore[no-redef]

        data = toml.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return dict(data["project"]["scripts"])
