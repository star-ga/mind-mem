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

# Install into an explicit prefix so we can COPY just that tree
RUN pip install --no-cache-dir --prefix=/install ".[mcp,embeddings]"


# Stage 2: runtime — slim final image
FROM python:3.12-slim AS runtime

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
