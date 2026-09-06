# mind-mem federation & multi-machine setup

> Field notes from standing up a multi-machine shared-memory fleet (2026-06). These are the
> things that cost time the first time — do them up front and federation is quick.

## TL;DR — how to share ONE memory across many machines

**Federate via the Postgres backend directly. Do NOT rely on `mm http-serve` for a
Postgres-backed corpus.**

1. On the **hub** (the box that owns the corpus), put the corpus in Postgres:
   ```jsonc
   // mind-mem.json
   "block_store": { "backend": "postgres",
     "dsn": "postgresql://mindmem:<pw>@127.0.0.1:5432/mindmem", "schema": "mind_mem" }
   ```
2. Open Postgres to the LAN (hub):
   - `postgresql.conf`: `listen_addresses = '*'`
   - `pg_hba.conf`: `host  mindmem  mindmem  <YOUR_LAN_CIDR>  scram-sha-256`
   - `systemctl restart postgresql@<ver>-main` → it now binds `0.0.0.0:5432`
3. On every **node**, point mind-mem at the hub DSN (same config, host = the hub IP):
   ```
   postgresql://mindmem:<pw>@<HUB_IP>:5432/mindmem
   ```
   Set `MIND_MEM_CONFIG=<path to that mind-mem.json>` for the `mm` CLI and the MCP server.
4. Verify from a node: `python -c "import psycopg; print(psycopg.connect(host='<HUB_IP>',port=5432,dbname='mindmem',user='mindmem',password='<pw>').cursor().execute('select count(*) from mind_mem.blocks') or 'ok'")`
   — you should see the hub's block count. That's shared read+write memory.

## One-command join: `mind-mem-connect`

Steps 1 and 3 above are the same four config keys typed by hand on every new node,
and getting any one of them wrong produces a node that starts cleanly and finds
nothing — most often because `block_store` was set but `recall.backend` was left on
`bm25`, so recall reads the node's now-empty Markdown tree instead of the FTS index
that mirrors the hub.

```bash
export MIND_MEM_DSN='postgresql://mindmem:<pw>@<HUB_IP>:5432/mindmem'
export MIND_MEM_REDIS_URL='redis://<HUB_IP>:6379/0'
mind-mem-connect --workspace ~/my-workspace          # add --dry-run to look first
```

It reads the workspace's existing `mind-mem.json`, merges the federation keys onto it
and writes it back, so `governance_mode`, ACL settings, limits and everything else the
node already had survive. It is idempotent, and it writes **only** configuration — no
block, no audit entry, no corpus.

Notes worth knowing before you run it:

- **Credentials come from the environment by default.** `--dsn` / `--redis-url` exist
  for scripted use, but a password on a command line is visible in `ps` to every user
  on the box and lands in the shell history file.
- **The config file is written `0600`** once it holds a DSN, and every line the command
  prints — including its error messages — has the password redacted.
- **The DSN and Redis schemes are an allowlist.** `postgresql`/`postgres` and
  `redis`/`rediss`/`unix`; anything else is refused rather than handed to a driver to
  interpret.
- **A malformed existing `mind-mem.json` is refused, not overwritten** — settings you
  cannot get back are not worth a clean-looking run.

## Gotchas we hit (each one is a fix candidate)

1. **`mm http-serve` does not serve the Postgres corpus.** Even with `MIND_MEM_WORKSPACE`
   and `MIND_MEM_CONFIG` set to a Postgres-backed config, `http-serve` reports
   `memory_count: 1` / `workspace: <name>` — it uses a **workspace file store**, not the
   configured `block_store` backend. → **Fix candidate:** `serve_http` should build the
   store from the same config path the CLI uses (`block_store.backend`), or the docs must
   say plainly "HTTP transport = file store; use direct Postgres for a DB-backed corpus."
2. **Token env var is inconsistent.** `mm token rotate` emits `export MIND_MEM_TOKENS=…`
   (plural), but `serve_http`'s startup guard checks **`MIND_MEM_TOKEN`** (singular) and
   refuses to bind a non-loopback host without it. Set **both** until unified.
   → **Fix candidate:** accept either; document the one canonical name.
3. **Auth header is non-obvious:** the HTTP transport expects **`X-MindMem-Token`**
   (no hyphens between Mind/Mem), not `Authorization: Bearer` or `X-Mind-Mem-Token`.
   Worth a one-line note in the serve help text.
4. **Fleet version drift → `unknown_recall_config_keys` warnings.** A node on an older
   pip (`mind-mem==4.1.1`) reading a config written by a newer build warns on keys it
   doesn't know (`bm25_weight`, `model`, `ollama_embed_model`, `onnx_backend`, `provider`,
   `rrf_k`, `vector_enabled`, `postgres`). → **Keep mind-mem versions aligned across the
   fleet**, and the config loader should ignore-with-info unknown keys (it does) but the
   docs should list the version→config-schema mapping.
5. **No-GPU recall on shared nodes:** to guarantee the embedding model never loads on a
   node's GPU, run the MCP with `CUDA_VISIBLE_DEVICES=-1` (use `-1`, not empty string —
   some launchers reject an empty value). BM25 recall still works.

## Why direct-Postgres beats the HTTP transport here
- One source of truth, real read+write from every node, no transport/store mismatch.
- Postgres already does auth (scram), concurrency, and durability.
- The HTTP transport is fine for a single-workspace file store or a read cache; it is not
  (today) a Postgres federation gateway.

## Transport security on the HTTP federation leg

`FederationClient` is what one host uses to reach another's
`/federation/*` endpoints. Three things are true about its TLS — one is
unconditional, and two are switches an operator turns on:

**A TLS 1.3 floor, enforced by construction.** An `https://` peer URL builds
its `ssl.SSLContext` with `minimum_version = TLSv1_3` *before* the socket
exists, so a peer that can only speak TLS 1.2 fails the handshake — there is
no connection, and no request is sent to be inspected afterwards. There is no
environment variable that lowers it: a floor an operator can switch off is not
a floor. On an interpreter whose OpenSSL has no TLS 1.3 the client refuses to
be constructed at all rather than quietly negotiating downward.

**Mutual TLS, to bind the peer's identity.**

```python
from mind_mem.v4.federation_client import FederationClient

client = FederationClient(
    "https://peer.internal:8765",
    token=os.environ["MIND_MEM_TOKEN"],
    cafile="/etc/mind-mem/federation-ca.pem",   # your CA, replaces the system store
    client_cert="/etc/mind-mem/this-host.crt",  # client half of mTLS
    client_key="/etc/mind-mem/this-host.key",
)
```

The inbound half is the same shape, on either listener:

```bash
mm serve      --tls-certfile peer.crt --tls-keyfile peer.key --tls-client-ca ca.pem
mm http-serve --tls-certfile peer.crt --tls-keyfile peer.key --tls-client-ca ca.pem
```

Both floor the listener at TLS 1.3 before the socket is wrapped, and
`--tls-client-ca` makes client certificates mandatory. There is no
`--tls-key-password` flag on purpose — an argv value is readable by every
process on the machine; set `MIND_MEM_TLS_KEYFILE_PASSWORD` instead. The
library entry points are `mind_mem.api.rest.run(...)` and
`mind_mem.http_transport.serve_http(...)` with the same four arguments.

**Certificate pinning, opt-in.**

```python
client = FederationClient(
    "https://peer.internal:8765",
    cafile="/etc/mind-mem/federation-ca.pem",
    # SHA-256 of the peer's SubjectPublicKeyInfo — hex or base64, and a
    # list during a rotation so the incoming key is pinned before it lands.
    pinned_pubkey_sha256=["<current key>", "<next key>"],
)
```

A pinned peer whose key is not in the set is refused **after the handshake and
before the request is written**, so it never receives the request it would have
been sent. This is the defence mutual TLS does not give you: a certificate that
a trusted CA should never have issued — a mis-issuance, or a TLS-intercepting
proxy whose root the machine trusts — passes ordinary verification and fails
the pin.

It is off by default, and the reason is operational rather than technical: a
pinned key store turns every routine certificate renewal into a coordinated
outage unless the next key was pinned first, and an operator surprised by that
disables the pin, which is worse than never having had one. So pin the *key*
(SPKI), not the certificate — a renewal that keeps the key keeps working — and
add the incoming key **before** rotating. Mutual TLS with a private CA remains
the recommended default for binding a peer's identity; pinning is the extra
layer for a hostile network. The full reasoning is recorded in code as
`mind_mem.v4.tls_floor.CERT_PINNING_DECISION`. Fingerprints are read out of a
certificate with:

```bash
openssl x509 -in peer.crt -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256
```

## Audit headers across a federation hop

`X-MindMem-Request-Id`, `X-MindMem-Actor` and `X-MindMem-Purpose` are
propagated, not just echoed. A request served by the REST layer binds them for
the whole request, and any federation call made while serving it carries the
same request id onward — so one correlation token spans the hop. The actor sent
onward is the identity the server *authenticated*, never the one the caller
claimed; the claimed value is used only where nothing authenticated anybody.

_See also: `docs/docker-deployment.md`, RFC 0009 (federation-first package layer)._
