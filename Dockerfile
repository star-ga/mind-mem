FROM python:3.12-slim

# Patch OS packages so the security image scan (Trivy HIGH/CRITICAL,
# ignore-unfixed) sees no fixable base-image CVEs — e.g. libcap2
# CVE-2026-4878 (privesc TOCTOU), fixed in the distro security channel.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY scripts/ scripts/
COPY mcp_server.py ./
# The package uses a src/ layout (pyproject: where=["src"], package-dir
# {""="src"}); it must be present before the editable install or setuptools
# fails with `error in 'egg_base' option: 'src' does not exist`.
COPY src/ src/

# Upgrade the base-image pip/setuptools first: python:3.12-slim ships an
# older pip (25.0.1) carrying fixable CVEs (e.g. CVE-2025-8869,
# CVE-2026-3219/6357/1703) that the Trivy image scan flags and the release
# alerts-gate blocks on. mind-mem ships zero core deps so end users are not
# exposed via the package, but keeping the build image clean remediates the
# supply-chain finding at the source rather than suppressing it.
#
# Upgrade pip + setuptools to patched floors (the base image ships older ones
# with fixable CVEs). mind-mem ships ZERO core deps so end users are never
# exposed via the wheel; this only keeps the build/scan image clean.
#
# NOTE: pip additionally VENDORS its own setuptools 70.3.0 + msgpack 1.1.2
# (pip/_vendor), which the Trivy image scan reports under the generic "Python"
# target. Those copies are pinned by pip for its installer internals — they are
# un-upgradable independently of pip (even latest pip vendors them) and are
# never imported on the serving path (entrypoint is `python3 mcp_server.py`;
# pip is not invoked at runtime). The three advisory IDs are documented and
# narrowly scoped in .trivyignore at the repo root — see that file for the
# rationale; they are NOT the top-level packages, which are patched above.
RUN pip install --no-cache-dir --upgrade "pip>=25.2" "setuptools>=83.0.0" \
 && pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["python3", "mcp_server.py"]
