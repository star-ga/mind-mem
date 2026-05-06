# Docker Deployment

Self-hosted MIND-Mem with Postgres+pgvector and Ollama in one command.

## Requirements

- Docker 24+ with Compose v2 (`docker compose version`)
- 8 GB RAM minimum (16 GB recommended if running Ollama models)

## Quick start

```bash
cd deploy
make up
```

This starts three services on `mindmem-net`:

| Service | Container | Port |
|---------|-----------|------|
| MIND-Mem MCP | `mind-mem` | 8765 |
| Postgres 16 + pgvector | `mind-mem-postgres` | (internal) |
| Ollama | `mind-mem-ollama` | 11434 |

Check status: `make status`  
Tail logs: `make logs`  
Stop everything: `make down`

## Workspace

The default compose stack mounts a named Docker volume (`workspace`) at
`/workspace` inside the container. To mount a directory from your host
instead, override the volume in a `docker-compose.override.yml`:

```yaml
services:
  mind-mem:
    volumes:
      - /path/to/your/workspace:/workspace
```

## Environment variables

Set these in your shell or in a `.env` file in the `deploy/` directory:

| Variable | Default | Description |
|----------|---------|-------------|
| `MIND_MEM_WORKSPACE` | `/workspace` | Workspace path inside the container |
| `MIND_MEM_TRANSPORT` | `http` | `http` or `stdio` |
| `MIND_MEM_TOKEN` | _(none — required for HTTP)_ | Bearer token for MCP HTTP auth. **As of v3.7.0, HTTP transport refuses to start when this is unset** unless `MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1` is also set (loopback opt-in). The bundled `docker-compose.yml` already enforces this via `${MIND_MEM_TOKEN:?must be set}`. |
| `MIND_MEM_ADMIN_TOKEN` | _(none — required for HTTP)_ | Admin-scope bearer token. Same fail-closed contract as `MIND_MEM_TOKEN`. |
| `MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST` | _(unset)_ | Opt-in escape hatch for loopback-only dev / test. Never set in containers exposed beyond `127.0.0.1`. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama endpoint |

## Pulling a model into Ollama

```bash
docker compose -f deploy/docker-compose.yml exec ollama ollama pull mind-mem:4b
```

## GPU passthrough (Ollama)

Uncomment the `deploy.resources` block in `docker-compose.yml` and ensure
`nvidia-container-toolkit` is installed on the host.

## Rebuilding after source changes

```bash
cd deploy
make build
make up
```
