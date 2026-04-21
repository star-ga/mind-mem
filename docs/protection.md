# mind-mem Library Protection

Defence-in-depth layers for the shipped `mind-mem` PyPI wheel. Protection
here means *tamper-detection + provenance*, not unbreakable DRM — Python
semantics make that impossible without native compilation — but it raises
the cost of silent modifications to the memory + governance layer that
downstream agents trust.

## Layers

| # | Layer                              | Where                          | Failure mode    |
|---|------------------------------------|--------------------------------|-----------------|
| 1 | OIDC trusted publishing            | `.github/workflows/release.yml`| PyPI rejects non-workflow uploads |
| 2 | Sigstore signatures (PEP 740)      | `.github/workflows/release.yml`| Verifiable at install with `pip --verify-attestations` |
| 3 | CycloneDX SBOM                     | `.github/workflows/release.yml`| Dependency audit trail in GitHub Release |
| 4 | Integrity manifest (SHA-256)       | `src/mind_mem/_integrity_manifest.json` | Import-time detection |
| 5 | `MIND_MEM_INTEGRITY=strict` mode   | `src/mind_mem/protection.py`   | Hard-fail on tamper |
| 6 | Author + license constants         | `mind_mem.__author__` / `.__license__` | `assert` at import by consumer |
| 7 | World-writable directory guard     | `_world_writable()` check      | Refuses to load if `chmod 777` |
| 8 | Frozen `AUTH_HEADER` / `AUDIT_TAG` | `protection.py`                | Stable constants — drift is a red flag |
| 9 | Secret scanning (gitleaks)         | `.github/workflows/security.yml` | Pre-commit + CI gate |
| 10| Trivy vulnerability scan           | `.github/workflows/security.yml` | Weekly + on PR |
| 11| Bandit static analysis             | `.github/workflows/lint.yml`   | Pre-commit gate |
| 12| Typechecking (mypy strict)         | `.github/workflows/ci.yml`     | Type-drift gate |
| 13| Hash-chained audit log             | `src/mind_mem/audit_chain.py`  | TAG_v1 genesis + HMAC |
| 14| Per-tenant audit isolation         | `src/mind_mem/tenant_audit.py` | HMAC-separated chains |
| 15| Envelope-encrypted DEKs            | `src/mind_mem/tenant_kms.py`   | AES-256-GCM wrap over master |

## Runtime verification

```python
import mind_mem

report = mind_mem.verify_integrity()
print(report.ok, report.mode, report.checked, report.mismatched)
```

Returns an immutable `IntegrityReport` with fields:

- `ok: bool` — overall pass/fail
- `mode: str` — `"fail-open"` (default) or `"strict"`
- `manifest_present: bool` — was `_integrity_manifest.json` found?
- `checked: int` — files hashed
- `mismatched: tuple[str, ...]` — files that failed the hash check
- `missing: tuple[str, ...]` — manifest entries with no file on disk
- `extra: tuple[str, ...]` — tracked modules absent from the manifest
- `warnings: tuple[str, ...]` — environmental anomalies (e.g.
  world-writable package dir)

The check runs automatically at `import mind_mem` and logs structured
warnings on mismatch. It is a **no-op for editable installs** — no
manifest is baked in during `pip install -e .`, so development work is
never blocked.

### Strict mode

Set the environment variable `MIND_MEM_INTEGRITY=strict` (values: `1`,
`true`, `yes`, `strict`) and `verify_integrity()` will raise
`RuntimeError` on any tamper. Use this in production agent deployments
where a silent modification is more dangerous than a crash.

```bash
MIND_MEM_INTEGRITY=strict python -m mind_mem.cli serve
```

## Build pipeline

1. Tag push (`v*`) triggers `.github/workflows/release.yml`.
2. `scripts/build_integrity_manifest.py` runs before `python -m build`,
   hashing every file in `mind_mem.protection._CRITICAL_MODULES` into
   `_integrity_manifest.json`.
3. `python -m build` produces wheel + sdist containing the manifest.
4. Sigstore signs both artifacts.
5. CycloneDX SBOM generated from the release environment.
6. OIDC trusted publishing uploads to PyPI.
7. GitHub Release assembles wheel + sdist + SBOM + signatures.

## Threat model

In scope:
- Tampered copies of `mind-mem` in a user's `site-packages` dir
- Package-in-the-middle attacks (uploaded non-workflow wheel to PyPI)
- World-writable install locations
- Silent modifications to governance thresholds (`AUTH_HEADER`,
  `AUDIT_TAG`)

Out of scope (Python's model makes these trivial):
- Dynamic monkey-patching by attacker code running in the same
  interpreter
- Debugger attachment
- `__import__` / `sys.modules` manipulation
- Decompilation / re-packaging of the wheel (there is no obfuscation)

For threats out of scope, run `mind-mem` in a process isolated from
untrusted code — the same model every Python library recommends.

## Consumer verification

Downstream agents pinning `mind-mem` can assert provenance at startup:

```python
import mind_mem

assert mind_mem.__author__ == "STARGA Inc <noreply@star.ga>"
assert mind_mem.__license__ == "Apache-2.0"
assert mind_mem.AUTH_HEADER == "X-MindMem-Token"
assert mind_mem.AUDIT_TAG == "TAG_v1"

report = mind_mem.verify_integrity()
if report.manifest_present and not report.ok:
    raise SystemExit(f"mind-mem integrity check failed: {report.mismatched}")
```

`pip install mind-mem --verify-attestations` verifies the Sigstore
bundle against the GitHub OIDC identity at install time (requires
`pip>=24.3`).

## Disabling for dev

Editable installs (`pip install -e .`) never ship a manifest, so
`verify_integrity()` returns `ok=True` in fail-open mode with
`manifest_present=False`. If a dev wheel does contain a manifest and
you want the loud warnings silenced:

```bash
# explicit fail-open (default)
unset MIND_MEM_INTEGRITY

# or remove the manifest from your install
rm $(python -c "import mind_mem, os; print(os.path.join(os.path.dirname(mind_mem.__file__), '_integrity_manifest.json'))")
```
