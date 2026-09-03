# Copyright 2026 STARGA, Inc.
"""The npm package must be publishable-by-script only, and must cohere.

Roadmap RM-2321 ("JavaScript / TypeScript SDK") is a publish-step item whose
publish step is blocked on a decision that is not an engineer's to make: the
manifest on disk claims ``@mind-mem/sdk`` and the roadmap names
``@star-ga/mind-mem-client``, and an npm name, once taken, is not reclaimable.
Everything up to that decision ships here — the packaging, the version
derivation, and the gate.

Two properties, both enforced below:

**The tree cannot publish itself.** ``sdk/js/package.json`` carries
``"private": true``, which npm refuses to publish. That turns "do not publish
until the name is settled" from a sentence in a plan into something the
tooling enforces. ``sdk/release/pack_js.py`` stages a copy with the flag
dropped, so there is exactly one door and it is a script that can be reviewed.

**The version is derived, not typed.** The manifest reads 0.1.0 while the
package is 5.0.1 — the classic hand-maintained-second-version drift. The
staged manifest takes its version from ``pyproject.toml`` at pack time, so
whatever is in the source manifest cannot reach a registry.

Coherence is checked against ``tsconfig.json`` rather than against a build
directory on purpose: the interesting failure is a manifest whose entry points
name files the compiler never emits, which publishes cleanly and breaks every
consumer's ``import``. Checking the compiler configuration instead of the
output means this gate runs on a machine with no Node installed — i.e. on
every row of the Python CI matrix, rather than skipping there and testing
nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from mind_mem import __version__ as PACKAGE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_MODULE = REPO_ROOT / "sdk" / "release" / "pack_js.py"
PACKAGE_JSON = REPO_ROOT / "sdk" / "js" / "package.json"
TSCONFIG = REPO_ROOT / "sdk" / "js" / "tsconfig.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack() -> Any:
    return _load(PACK_MODULE, "mind_mem_sdk_pack_js")


@pytest.fixture(scope="module")
def source_manifest(pack: Any) -> dict[str, Any]:
    return pack.load_source_manifest()


@pytest.fixture(scope="module")
def staged(pack: Any) -> dict[str, Any]:
    return pack.staged_manifest(PACKAGE_VERSION)


class TestSourceTreeCannotPublish:
    def test_source_manifest_is_private(self, source_manifest: dict[str, Any]) -> None:
        assert source_manifest.get("private") is True, (
            "sdk/js/package.json lost `private: true`; a stray `npm publish` in that directory "
            "would claim a package name that is still an open decision"
        )

    def test_staged_manifest_drops_private(self, staged: dict[str, Any]) -> None:
        assert "private" not in staged

    def test_staging_leaves_the_source_manifest_untouched(self, pack: Any, source_manifest: dict[str, Any]) -> None:
        pack.staged_manifest("9.9.9")
        assert pack.load_source_manifest() == source_manifest


class TestVersionIsDerived:
    def test_staged_version_comes_from_the_package(self, staged: dict[str, Any]) -> None:
        assert staged["version"] == PACKAGE_VERSION

    def test_the_source_version_is_not_the_authority(self, pack: Any, source_manifest: dict[str, Any]) -> None:
        # Positive control for the claim above: prove the source manifest
        # really does hold a different number, so "staged == package version"
        # is a substantive result and not a tautology.
        assert source_manifest["version"] != PACKAGE_VERSION, (
            "the source manifest now matches the package version by hand, which makes the derivation "
            "untestable — keep the stamped value the only authority"
        )
        assert pack.staged_manifest("7.7.7")["version"] == "7.7.7"


class TestManifestCoheres:
    def test_no_problems_on_the_committed_tree(self, pack: Any, staged: dict[str, Any]) -> None:
        assert pack.manifest_problems(staged) == []

    def test_cli_check_exits_zero(self, pack: Any) -> None:
        assert pack._main(["--check"]) == 0

    def test_entry_points_are_under_the_compiler_out_dir(self, pack: Any, staged: dict[str, Any]) -> None:
        out_dir = pack._load_jsonc(TSCONFIG)["compilerOptions"]["outDir"].strip("./")
        for key in ("main", "types"):
            assert staged[key].lstrip("./").startswith(out_dir + "/")

    @pytest.mark.parametrize(
        ("mutation", "fragment"),
        [
            ({"main": "./src/index.js"}, "outside the compiler outDir"),
            ({"files": ["src"]}, "does not include the build output"),
            ({"files": []}, "no non-empty 'files' allowlist"),
            ({"exports": {}}, "exports has no '.' entry"),
            ({"private": True}, "cannot be published"),
            ({"version": ""}, "no version"),
            ({"main": "./dist/nonexistent.js"}, "has no source at"),
        ],
    )
    def test_positive_control_each_incoherence_is_reported(
        self, pack: Any, staged: dict[str, Any], mutation: dict[str, Any], fragment: str
    ) -> None:
        # A coherence check that never fires is a comment. Break one property
        # at a time and require the matching complaint.
        broken = dict(staged)
        broken.update(mutation)
        problems = pack.manifest_problems(broken)
        assert any(fragment in problem for problem in problems), f"{mutation} produced {problems}"

    def test_positive_control_declaration_off_is_reported(self, pack: Any, staged: dict[str, Any]) -> None:
        tsconfig = pack._load_jsonc(TSCONFIG)
        tsconfig["compilerOptions"]["declaration"] = False
        problems = pack.manifest_problems(staged, tsconfig)
        assert any("declaration off" in problem for problem in problems), problems


class TestStaging:
    def test_stage_writes_a_publishable_tree(self, pack: Any, tmp_path: Path) -> None:
        dest = pack.stage(tmp_path / "pkg", PACKAGE_VERSION)
        manifest = json.loads((dest / "package.json").read_text(encoding="utf-8"))
        assert manifest["version"] == PACKAGE_VERSION
        assert "private" not in manifest
        assert (dest / "README.md").is_file()
        assert (dest / "LICENSE").is_file(), "the tarball must carry the licence it claims in the manifest"

    def test_stage_refuses_an_incoherent_manifest(self, pack: Any, tmp_path: Path, monkeypatch: Any) -> None:
        broken = tmp_path / "package.json"
        source = pack.load_source_manifest()
        source["main"] = "./src/index.js"
        broken.write_text(json.dumps(source), encoding="utf-8")
        monkeypatch.setattr(pack, "PACKAGE_JSON", broken)
        with pytest.raises(ValueError, match="not coherent"):
            pack.stage(tmp_path / "pkg", PACKAGE_VERSION)

    def test_stage_can_require_a_build(self, pack: Any, tmp_path: Path, monkeypatch: Any) -> None:
        # A source tree that is coherent but not built: the manifest checks
        # pass and `dist/` is absent, which is exactly the state a release
        # script must refuse rather than quietly publish an empty tarball.
        unbuilt = tmp_path / "js"
        (unbuilt / "src").mkdir(parents=True)
        (unbuilt / "src" / "index.ts").write_text("export {};\n", encoding="utf-8")
        monkeypatch.setattr(pack, "JS_DIR", unbuilt)

        assert pack.manifest_problems(pack.staged_manifest(PACKAGE_VERSION), js_dir=unbuilt) == []
        with pytest.raises(FileNotFoundError, match="npm run build"):
            pack.stage(tmp_path / "pkg", PACKAGE_VERSION, require_build=True)


class TestNpmScriptsAreRunnable:
    def test_test_script_compiles_before_running(self, source_manifest: dict[str, Any]) -> None:
        # `npm test` used to be `node --test test/*.test.js` against a
        # directory holding only .ts sources: it matched nothing, node
        # reported "tests 0", and exited 0. The suite existed and had never
        # run. The script now compiles first and asserts the compiled entry
        # point exists, so a glob that matches nothing is a failure.
        script = source_manifest["scripts"]["test"]
        assert "tsconfig.test.json" in script
        assert "test -f build/test/client.test.js" in script

    def test_test_tsconfig_includes_the_test_directory(self, pack: Any) -> None:
        config = pack._load_jsonc(REPO_ROOT / "sdk" / "js" / "tsconfig.test.json")
        assert "test" in config["include"]
        # The base config excludes test/ so dist/ carries no test code, and
        # `extends` inherits `exclude` — leaving it inherited emitted src only
        # and produced a green run of zero tests. The key must be PRESENT and
        # not name test/: `config.get("exclude", [])` would read a dropped
        # override as an empty list and pass, which is the same silent
        # inheritance this asserts against.
        assert "exclude" in config, "tsconfig.test.json must override the inherited exclude, not rely on it"
        assert "test" not in config["exclude"]

    def test_publish_build_still_excludes_tests(self, pack: Any) -> None:
        base = pack._load_jsonc(TSCONFIG)
        assert "test" in base["exclude"], "dist/ must not carry the test suite"
