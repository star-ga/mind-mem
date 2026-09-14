# SDK release path

Both SDKs share the MIND-Mem version from `pyproject.toml`. The npm package is
`@star-ga/mind-mem-client`; the Go module is
`github.com/star-ga/mind-mem/sdk/go/v5`. Registry publication remains pending.

## Version and contract checks

```bash
python3 sdk/release/version.py --check
python3 sdk/release/pack_js.py --check
PYTHONPATH=src python3 -m pytest tests/test_sdk_response_contract.py tests/test_sdk_route_conformance.py tests/test_sdk_release_versioning.py tests/test_sdk_js_packaging.py
```

The response gate calls the authenticated REST API on a synthetic workspace.
Both client suites consume the resulting contract fixtures. Changes to a route
or served response therefore need corresponding client support.

## Build and inspect the npm artifact

Run these commands from the repository root after the client tests pass:

```bash
npm --prefix sdk/js ci --ignore-scripts
npm --prefix sdk/js test
npm --prefix sdk/js run build
python3 sdk/release/pack_js.py --stage /tmp/mind-mem-client-package
npm pack /tmp/mind-mem-client-package
```

Use a fresh staging directory for each build. `--stage` requires the compiled
JavaScript and declarations. It stamps the version, removes the private flag,
and omits source-only lifecycle scripts and development dependencies. In
particular, publication must not rebuild the staged artifact: it contains the
reviewed `dist/`, license and README, without the compiler sources.

The source manifest stays private and its placeholder version cannot reach the
registry through this path. Install the resulting tarball in a fresh consumer
project and check its import before uploading that same tarball:

```bash
npm publish /path/to/star-ga-mind-mem-client-VERSION.tgz --access public
```

This upload needs npm authentication with publish rights to the `@star-ga`
scope. The repository does not contain registry credentials.

## Publish the Go module

After the verified release commit is on the public repository, take the exact
subdirectory tag from `python3 sdk/release/version.py`. For the 5.0.3 candidate
it is `sdk/go/v5.0.3`. Create that new tag at the release commit, push it, and
verify it through the module proxy:

```bash
GOPROXY=proxy.golang.org go list -m github.com/star-ga/mind-mem/sdk/go/v5@v5.0.3
```

Never move an existing published tag. The `/v5` module path and the
`sdk/go/v5.x.y` tag prefix are both required by Go's module versioning rules.

## CI

The Python matrix checks OpenAPI, client routes, actual response shapes,
version derivation and packaging. The `sdk-go` and `sdk-js` jobs additionally
compile and test the actual clients. The JavaScript job builds, packs and
imports the staged artifact. These jobs do not publish packages.
