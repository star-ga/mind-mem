// The /v5 suffix is required, not cosmetic. This is a subdirectory module in
// a repository whose releases are v5.x, so its tags take the form
// `sdk/go/v5.0.2`; Go rejects a v2+ tag for a module path with no matching
// major-version suffix, which is what a `sdk/go/v5.0.2` tag against
// `github.com/star-ga/mind-mem/sdk/go` would have been. The suffix does NOT
// require a `sdk/go/v5/` directory — the major-version subdirectory layout is
// an alternative to, not a requirement of, the suffix.
//
// sdk/release/version.py derives this suffix from the package version in
// pyproject.toml, and tests/test_sdk_release_versioning.py fails when the two
// disagree, so a future 6.0.0 cannot ship a module path still claiming v5.
module github.com/star-ga/mind-mem/sdk/go/v5

go 1.21
