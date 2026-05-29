# Copilot CLI audit — mind-mem (2026-05-18)

---

  │ "GHAS #[0-9]+|B[0-9]{3}|#[0-9]{2,4}|T[1-4] #[0-9]+|queued
  │ separately|deferred v[0-9]" in *.{md,py,toml,yaml,yml,txt}
  └ 9 lines found

● Read runpod_deploy.py
  │ train/runpod_deploy.py
  └ L1:140 (140 lines read)

● Read runpod_deploy.py
  │ train/runpod_deploy.py
  └ L296:372 (77 lines read)

● Read backport_sweep.py
  │ train/backport_sweep.py
  └ L1:120 (120 lines read)

● Read roadmap.md
  │ docs/roadmap.md
  └ L30:60 (31 lines read)

● Read CHANGELOG.md
  └ L1:120 (119 lines read)

● Read ROADMAP.md
  └ L1096:1110 (15 lines read)

● Search (grep)
  │ "/tmp/[A-Za-z0-9._/-]+|TemporaryDirectory|NamedTemporaryFile|mkdtemp\(|mkste
  │ mp\(|tempfile\." in *.{py,sh,md}
  └ 9 lines found

● Search (grep)
  │ "GHAS #[0-9]+|B101|T[1-4] #[0-9]+|deferred v3\.3\.1|queued separately|issue
  │ #[0-9]+" in *.md
  └ 23 lines found

● Search (grep)
  │ "CI run [0-9]+|[0-9]+/[0-9]+ jobs success|CI green on all platforms|tests
  │ passing, CI green|All 9 matrix jobs green|job count" in *.md
  └ 10 lines found

● Search (grep)
  │ "/tmp/hf_write_token|/tmp/mm-backport-[^\s`'"]+|/tmp/mm-bench|/tmp/ws|/tmp/m
  │ m-train" in *.{py,sh,md}
  └ 58 lines found

● Read roadmap.md
  │ docs/roadmap.md
  └ L400:430 (31 lines read)

● Read storage-migration.md
  │ docs/storage-migration.md
  └ L268:278 (10 lines read)

● Read copilot-tool-output-1779103682115-xdi7vr.txt
  │ /tmp/copilot-tool-output-1779103682115-xdi7vr.txt
  └ L1:120 (120 lines read)

● Read copilot-tool-output-1779103682125-02kybt.txt
  │ /tmp/copilot-tool-output-1779103682125-02kybt.txt
  └ L1:160 (160 lines read)

● Read ROADMAP.md
  └ L1:90 (89 lines read)

● Read resume_pod_train.sh
  │ train/resume_pod_train.sh
  └ L1:60 (60 lines read)

● Read install.sh
  └ L130:150 (21 lines read)

● Search (grep)
  │ "<private/protected STARGA repo + commercial-runtime name patterns>" in
  │ *.{md,py,toml,yml,yaml,json,txt,sh}
  └ No matches found

● Search (grep)
  │ "AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{35}|ghp_[0-9A-Za-z]{36}|github_pat_[0-9
  │ A-Za-z_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-
  │ ]{10,}|BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY" in
  │ *.{md,py,toml,yml,yaml,json,txt,sh}
  └ 1 line found

● Search (grep)
  │ "HF_TOKEN_FILE|cat \"\$HF_TOKEN_FILE\"|<\$HF_TOKEN_FILE|\$\(cat
  │ \"\$HF_TOKEN_FILE\"\)|hf_write_token" (train/resume_pod_train.sh)
  └ 2 lines found

● Read resume_pod_train.sh
  │ train/resume_pod_train.sh
  └ L60:120 (25 lines read)

● Read roadmap.md
  │ docs/roadmap.md
  └ L34:40 (7 lines read)

● Read CHANGELOG.md
  └ L658:668 (11 lines read)

● Read CHANGELOG.md
  └ L4372:4382 (11 lines read)

| Category | Severity | File:line | Finding |
|---|---|---|---|
| A — private repos / customers / pricing / acquisition | **NONE** | — | **NONE found.** No matches for private/protected STARGA repo name patterns or commercial-runtime references; no internal customer names, per-customer pricing, or acquisition-strategy leakage found. |
| A — protected-binary internals | **HIGH** | `CHANGELOG.md:4477-4481` | Public changelog references protected-runtime hardening details (leak-verifier scope + before/after string counts) — recommend redacting the specifics from the public changelog. |
| A — protection signaling | **LOW** | `README.md:1027`, `src/mind_mem/mind_ffi.py:476` | Public docs/API explicitly advertise runtime-protection presence via `is_protected()`. |
| B — internal security/work-queue IDs | **MEDIUM** | `docs/roadmap.md:37`, `CHANGELOG.md:665` | Public files expose internal security-triage IDs: `GHAS #179/#180` and `B101`. |
| B — roadmap queue labels / deferred planning state | **MEDIUM** | `ROADMAP.md:1099-1108`, `docs/storage-migration.md:275` | Public roadmap uses internal queue notation (`T1 #1`…`T4 #10`, `roadmap task #15`) and shows deferred queue state (`deferred v3.3.1`). |
| C — exact CI run/job pin | **LOW** | `CHANGELOG.md:62-64` | Permanent text embeds exact CI run metadata: `CI run 25901731047` and `26/26 jobs success`. |
| C — stale CI snapshots in permanent docs | **LOW** | `ROADMAP.md:18`, `33`, `43`, `52`, `64`, `78`; `CHANGELOG.md:4375` | Repeated permanent “tests passing, CI green on all platforms” snapshots and “All 9 matrix jobs green” claims are the stale-status anti-pattern. |
| D — hardcoded plaintext secrets / API keys | **NONE** | — | **NONE found.** No plaintext credentials or hardcoded API keys/tokens matched the repo scan. |
| D — predictable `/tmp` token path + argv exposure | **HIGH** | `train/resume_pod_train.sh:11`, `73-76` | HuggingFace token is read from predictable `/tmp/hf_write_token` and then injected into `ssh ... "nohup env HF_TOKEN=..."`, exposing the secret via local process argv as well as a globally guessable temp path. |
| D — predictable `/tmp` temp file | **MEDIUM** | `install.sh:138-148` | Installer writes pip stderr to `/tmp/mind-mem-pip-err.$$`; PID-based temp names are guessable/symlinkable by a local hostile user. |


Changes   +0 -0
Requests  1 Premium (2m 42s)
Tokens    ↑ 462.2k • ↓ 8.7k • 283.8k (cached) • 5.3k (reasoning)
