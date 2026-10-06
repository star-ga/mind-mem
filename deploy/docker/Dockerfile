# Stage 1: build — install all deps and produce a pruned site-packages
FROM python:3.12-slim AS builder

WORKDIR /build

# System build deps needed for native extensions (sqlite-vec, onnxruntime, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files pip needs first (layer-cache friendly)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY mcp_server.py ./
# The CLI manual skill ships as package data (pyproject data-files); the
# build fails without it.
COPY skills/ skills/

# Patch the build toolchain + constrain msgpack to the fixed floors, mirroring
# the root Dockerfile: setuptools CVE-2025-47273 (fixed 78.1.1) / CVE-2026-59890
# (fixed 83.0.0), msgpack GHSA-6v7p-g79w-8964 (fixed 1.2.1). Keeps the runtime
# image free of fixable HIGH/CRITICAL supply-chain CVEs.
RUN pip install --no-cache-dir --upgrade "pip>=25.2" "setuptools>=83.0.0"

# Install into an explicit prefix so we can COPY just that tree
RUN pip install --no-cache-dir --prefix=/install ".[mcp,embeddings]" "msgpack>=1.2.1"


# Stage 2: runtime — slim final image
FROM python:3.12-slim AS runtime

# Minimal runtime system deps. `apt-get upgrade` pulls the distro security
# channel over the base image's snapshot, mirroring the root Dockerfile: the
# scheduled Trivy scan (HIGH/CRITICAL, ignore-unfixed) flagged libpcre2-8-0
# CVE-2026-103111 (fixed in 10.46-1~deb13u3) the first time it scanned this
# image.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# The runtime stage starts from a fresh python:3.12-slim, so the pip upgrade in
# the builder never reaches it: it carries the base image's pip (25.0.1) and
# pip's vendored copies of setuptools/urllib3/msgpack, which the image scan
# reports and which cannot be upgraded independently of pip. Nothing at runtime
# installs packages (the entrypoint is `python3 -m mind_mem.mcp_server`), so pip
# is removed, exactly as the root Dockerfile does after its install.
RUN python3 -m pip uninstall -y pip

# Non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 mindmem

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy the application source
WORKDIR /app
COPY --chown=mindmem:mindmem src/ src/
COPY --chown=mindmem:mindmem mcp_server.py ./
COPY --chown=mindmem:mindmem pyproject.toml README.md LICENSE ./

# Workspace mount point — users bind-mount their workspace here
RUN mkdir -p /workspace && chown mindmem:mindmem /workspace
VOLUME /workspace

ENV MIND_MEM_WORKSPACE=/workspace
# Transport: stdio (default, for Claude Code) or http (for remote clients)
ENV MIND_MEM_TRANSPORT=stdio

USER mindmem

EXPOSE 8765 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import mind_mem; print('ok')" || exit 1

# Entrypoint respects MIND_MEM_TRANSPORT env var
ENTRYPOINT ["sh", "-c", \
    "if [ \"$MIND_MEM_TRANSPORT\" = \"http\" ]; then \
        exec python3 -m mind_mem.mcp_server --transport http --port 8765; \
     else \
        exec python3 -m mind_mem.mcp_server --transport stdio; \
     fi"]
