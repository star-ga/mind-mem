# Supply-Chain Security

mind-mem ships with a layered supply-chain security posture covering SAST,
dependency auditing, secret scanning, SBOM generation, and Sigstore signing
for every release artifact.

---

## What We Sign

Every `v*` tag triggers `.github/workflows/release.yml`, which signs:

| Artifact | Tool | Standard |
|---|---|---|
| `dist/*.whl` | Sigstore (`sigstore/gh-action-sigstore-python`) | SLSA provenance (keyless OIDC) |
| `dist/*.tar.gz` | Sigstore (`sigstore/gh-action-sigstore-python`) | SLSA provenance (keyless OIDC) |
| `sbom.cdx.json` | Attached to GitHub Release | CycloneDX 1.4+ |

PyPI uploads additionally generate PEP 740 attestations automatically via
`pypa/gh-action-pypi-publish` when the `id-token: write` permission is
present on a trusted publisher environment.

---

## How to Verify a Release

### Verify wheel or tarball with cosign (Sigstore)

```bash
# Install cosign
brew install cosign          # macOS
# or: pip install sigstore   # cross-platform Python client

# Download the wheel + its .sigstore bundle from the GitHub Release
cosign verify-blob \
  --bundle mind_mem-3.2.0-py3-none-any.whl.sigstore \
  mind_mem-3.2.0-py3-none-any.whl

# Expected output:
# Verified OK
```

### Verify via sigstore Python client

```bash
pip install sigstore
python -m sigstore verify github \
  --cert-identity "https://github.com/star-ga/mind-mem/.github/workflows/release.yml@refs/tags/v3.2.0" \
  dist/mind_mem-3.2.0-py3-none-any.whl
```

### Verify PyPI attestation (PEP 740)

```bash
pip install pypi-attestations
python -m pypi_attestations verify mind-mem==3.2.0
```

---

## Where SBOMs Live

- **GitHub Releases** — `sbom.cdx.json` is attached to every `v*` release
  as a CycloneDX JSON SBOM.  Download from:
  `https://github.com/star-ga/mind-mem/releases/latest`

- **CI artifacts** — the `sbom-cdx` artifact is retained for 90 days on
  every push to `main` and every PR (from the `sbom` job in
  `.github/workflows/security.yml`).

- **PyPI attestations** — PEP 740 attestations are attached to each PyPI
  upload and are visible at:
  `https://pypi.org/project/mind-mem/#attestations`

---

## SAST Results

### CodeQL

Runs on every push to `main`, every PR, and weekly (Monday 06:00 UTC).

- Config: `.github/workflows/security.yml` → `codeql` job
- Results: **GitHub Security tab** → Code scanning → CodeQL
- Query suite: `security-extended` (broader than the default suite)

### Bandit

Runs on every push to `main`, every PR, and weekly.

- Config: `.github/workflows/security.yml` → `bandit` job
- Results: **GitHub Security tab** → Code scanning → bandit (SARIF upload)
- Also enforced locally via `.pre-commit-config.yaml` (medium+ severity)

---

## Dependency Audit

`pip-audit` runs against the installed dependency graph on every push,
PR, and weekly schedule.

- Config: `.github/workflows/security.yml` → `pip-audit` job
- Artifact: `pip-audit-results` (JSON, retained 90 days)
- Advisory databases: PyPA, OSV

---

## Secret Scanning

Gitleaks scans the full commit history on every push and PR.

- Config: `.github/workflows/security.yml` → `secrets-scan` job
- Allowlist for test fixtures: `.gitleaks.toml` (dummy tokens, UUIDs,
  in-memory SQLite URIs)
- Local enforcement: `detect-secrets` pre-commit hook with baseline at
  `.secrets.baseline`

---

## Container Scanning

Trivy scans the Docker image for HIGH/CRITICAL CVEs whenever a `Dockerfile`
changes or on the weekly schedule.

- Config: `.github/workflows/security.yml` → `docker-scan` job
- Results: **GitHub Security tab** → Code scanning → trivy (SARIF upload)
- Severity gate: `HIGH,CRITICAL` with `--ignore-unfixed`

---

## Local Pre-Commit Enforcement

Install hooks once per clone:

```bash
pip install pre-commit
pre-commit install

# Generate detect-secrets baseline (first time only)
detect-secrets scan > .secrets.baseline
```

Hooks enforced on every `git commit`:

| Hook | What |
|---|---|
| `ruff` | Lint + auto-fix |
| `ruff-format` | Format |
| `bandit` | SAST (medium+ severity, `src/` only) |
| `detect-secrets` | Secret pattern detection |
| Standard hooks | Trailing whitespace, YAML/JSON/TOML validity, merge conflicts |
