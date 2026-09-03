# SDK release path

Everything here stops short of publishing. Both publish steps need a decision
that is an operator's to make, and both are irreversible, so what ships is the
packaging, the version derivation, and the gates — never the push.

## What is derived, and from what

`pyproject.toml`'s `[project].version` is the only version in the repository.
`version.py` derives everything else from it:

```bash
python3 sdk/release/version.py            # print the derived identifiers
python3 sdk/release/version.py --check    # exit 1 if the tree disagrees
```

```
package_version: 5.0.1
go_module_path:  github.com/star-ga/mind-mem/sdk/go/v5
go_tag:          sdk/go/v5.0.1
npm_version:     5.0.1
```

Gated by `tests/test_sdk_release_versioning.py` and
`tests/test_sdk_js_packaging.py`.

## Go module — the defect this closed

`sdk/go/go.mod` declared `module github.com/star-ga/mind-mem/sdk/go`, with no
major-version suffix. A subdirectory module publishes under a tag prefixed by
its directory, so the 5.x line publishes as `sdk/go/v5.0.2` — and Go refuses a
v2-or-higher version for a module path with no matching `/vN` suffix. The tag
would have resolved to nothing, `go get` would have failed for every consumer
with a module-path mismatch, and **a tag pushed to a public repository cannot
be withdrawn from the module proxy**. The item was filed as "only the publish
step is open"; the publish step was the broken part.

The suffix is now derived rather than typed, so a future major cannot leave it
behind: `version.py --check` goes red when `go.mod` and the package version
part company.

### Publishing (operator, after the release tag)

```bash
git tag sdk/go/v5.0.2        # exact string from `version.py`
git push origin sdk/go/v5.0.2
GOPROXY=proxy.golang.org go list -m github.com/star-ga/mind-mem/sdk/go/v5@v5.0.2
```

**Decision needed:** whether the Go client shares the package's version line at
all. It does today, which is why the `/v5` suffix is required. Giving it an
independent v1 line would drop the suffix and decouple the two — a legitimate
choice, but it must be made before the first tag, because the module path is
part of every consumer's import statement.

## npm package

```bash
python3 sdk/release/pack_js.py --check           # validate the staged manifest
cd sdk/js && npm install && npm run build        # produce dist/
python3 sdk/release/pack_js.py --stage /tmp/pkg  # publishable copy
cd /tmp/pkg && npm pack                          # tarball, for inspection
```

`sdk/js/package.json` carries `"private": true`, so `npm publish` in the source
tree refuses. The staged copy drops the flag and takes its version from
`pyproject.toml`, so the `0.1.0` sitting in the manifest cannot reach a
registry.

**Decision needed:** the package name. The manifest says `@mind-mem/sdk`; the
roadmap says `@star-ga/mind-mem-client`. An npm name is not reclaimable, and
the two names imply different scopes to own. Once settled, change `name` in
`sdk/js/package.json` and publish the staged directory:

```bash
cd /tmp/pkg && npm publish --access public
```

## The CI job

Not committed — `.github/workflows/` was outside this change's scope. The job
is three steps and needs no secrets, because it does not publish:

```yaml
# .github/workflows/ci.yml — add to the existing job matrix
- name: SDK gates
  run: |
    python3 -m mind_mem.spec.export_openapi --check
    python3 sdk/release/version.py --check
    python3 sdk/release/pack_js.py --check
- name: Go client
  run: cd sdk/go && go vet ./... && go test ./...
- name: JS client
  run: cd sdk/js && npm ci && npm test
```

The first step is already covered by `tests/test_sdk_*.py`, so the Python
matrix enforces it today with no workflow change at all. The Go and JS steps
are the genuinely new coverage: neither client's suite runs in CI right now,
which is how both of them drifted off the served routes without anything going
red.
