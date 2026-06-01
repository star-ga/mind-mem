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

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["python3", "mcp_server.py"]
