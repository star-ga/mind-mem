# MIND-Mem Threat Model — `online_trainer.py` (T-009) — 2026-08-31

**Scope:** `src/mind_mem/online_trainer.py` and everything upstream that could reach it —
`interaction_signals.py`, the `observe_signal` / `signal_stats` / `index_stats` MCP tools, and
the root-level `generate_mind7b_training.py` corpus generator.
**Threat model:** single-operator localhost, untrusted *content* (blocks, proposals and query
text may be attacker-influenced), trusted operator.
**Methodology:** STRIDE + line-level source trace of every write and every caller. Read-only —
no code was changed by this review.
**Verdict:** *Yes, with the premise corrected.* The T-009 risk as written in
`threat-model-2026-04-28.md:60-63` ("agent feedback feeds local Ollama fine-tune so poisoned
proposals could shape the local model") **does not exist in this tree**: the trainer has no
production caller, and its only input carries query text and opaque block IDs — never block or
proposal content. What is real is a set of *latent* gaps that become live the moment the module
is wired, plus two *live today* resource issues in the signal ledger that feeds it.

---

## Executive Summary

`online_trainer.py` is a 321-line, stdlib-only, **unreachable** module. Nothing in `src/`,
`mcp/`, the CLI, or the package `__init__` imports it; the only importer in the repository is
`tests/test_v28_completion.py:21`. It performs no I/O of any kind — no network, no filesystem,
no subprocess — so it cannot exfiltrate, cannot be SSRF'd, and writes nothing to disk. The
actual gradient step is a caller-supplied callable (`TrainStepFn`, `online_trainer.py:241`) and
**no caller supplies one anywhere in the tree**.

The consequences for the roadmap's stated fear are worth stating plainly: for a poisoned
proposal to shape a local model, proposal *content* would have to reach training data. The
trainer's only defined input is an `interaction_signals` record, which carries two query strings
and a list of block **IDs** (`interaction_signals.py:191-207`) — no block body, no rationale, no
statement. That path is absent, not merely gated.

The nearest thing to the roadmap's fear that genuinely exists is **T-009.9**: the root-level
`generate_mind7b_training.py` already contains a `load_workspace_blocks()` function
(line 559) that reads `decisions/DECISIONS.md` and `intelligence/SIGNALS.md` — the files
governance writes proposal statements and rationales into — and a `make_real_entity_example()`
(line 579) that turns such a block into a training record. `main()` calls neither
(lines 597-640); the emitted corpus is 100% synthetic templates under `random.seed(42)`
(line 19). Two function calls stand between today's tree and untrusted corpus content becoming
training data with no gate at all. That is the concrete trigger the roadmap item was reaching
for, and it is what the pre-ingest checklist at the end of this document guards.

Findings are ranked: **live today** first, then **latent** (blocked only by the absence of
wiring), then explicit **no-finding** sections.

---

## Data-flow trace (question 1)

The complete path, end to end, with the break marked:

```
MCP client (any USER-scope caller — acl.py:114-115)
  └─ observe_signal(session_id, previous_query, new_query, previous_results)   signal.py:19-99
       ├─ workspace gate                                                       signal.py:57-60
       ├─ bounds: queries ≤8192 chars, ids capped at 64 entries                signal.py:56, 61
       └─ SignalStore(<ws>/memory/interaction_signals.jsonl)                    _helpers.py:56
            ├─ classify() → RE_QUERY | REFINEMENT | CORRECTION | None    interaction_signals.py:154
            └─ append one JSON line, flush + fsync                       interaction_signals.py:346-349

  ── ✂ NO CALLER — the chain stops here in this tree ✂ ──

  build_training_tuples(signals) → list[TrainingTuple]                    online_trainer.py:60-96
  TrainingLoop.submit(tuples) → try_flush() → train_step(batch)          online_trainer.py:273-299
  train_step: caller-supplied; no implementation exists in the repository  online_trainer.py:241
  WeightRegistry.set_candidate / promote / revert (in-memory only)       online_trainer.py:125-205
```

Verification of the break: `grep -rn "online_trainer"` over the repository returns only
`ANATOMY.md:680`, `CHANGELOG.md:5510,5573`, `ROADMAP.md:1525`, the two 2026-04-28 security
reports, `benchmarks/tasks/real_repo_tasks.json`, `docs/security-baselines/bandit-v3.2.0-baseline.json:1565`,
and `tests/test_v28_completion.py`. There is no `import` of the module outside that test, no
entry in `src/mind_mem/__init__.py`, no MCP tool, and no `mm` subcommand (`mm_cli.py` registers
no training command).

Ollama is used in this codebase for *extraction and chat inference* (`llm_extractor.py:5-23`,
`smart_chunker.py:12`) and for serving the local 4B tag pulled by `mm install-model`
(`mm_cli.py:612-760`). No code path connects interaction signals, or the trainer, to any of it.

---

## Live today

### T-009.5 — Unbounded signal ledger, re-read in full on every capture (MEDIUM)

- **Files:** `interaction_signals.py:282-309` (constructor reads the whole file),
  `:346-349` (append; no cap, no rotation), `:381-399` (`all_signals` full read),
  `:401-402` (`stats()` → `all_signals()`); `signal.py:63, 98`; `memory_ops.py:193-196`.
- **Issue:** The MCP tools construct a fresh `SignalStore` per call, and the constructor calls
  `_load_ids()`, which parses every line of the ledger to rebuild the dedup set. The ledger
  itself has no size cap, no rotation, and no compaction. `signal_stats` and `index_stats`
  each re-read the whole file on top of that.
- **Measured** (this host, synthetic records ~590 B each):

  | records | file size | `SignalStore()` construction |
  |---------|-----------|------------------------------|
  | 1 000   | 0.59 MB   | 3.8 ms                       |
  | 20 000  | 11.74 MB  | 71.9 ms                      |
  | 100 000 | 58.70 MB  | 371.7 ms                     |

  The per-call cost is linear in ledger size, so *N* captures cost O(N²) total I/O. At 100k
  records every subsequent `observe_signal` pays ~0.37 s before it can append.
- **Impact:** Degradation, not compromise, under single-operator localhost — but the ledger is
  writable by any USER-scope MCP client (`acl.py:114-115`), so a chatty or hostile agent can
  drive both disk growth and per-call latency without any elevation. It also drags
  `index_stats`, which operators poll.
- **Fix:** Cap and rotate the ledger, and keep the dedup set in memory across calls instead of
  rebuilding it per construction. In-repo precedent for the cap exists on both sides:
  `_REVERT_LOG_CAP = 10_000` with a `deque(maxlen=…)` (`online_trainer.py:32, 133`) and the
  `export_memory` `max_blocks` bound (roadmap N-11).

### T-009.6 — Per-record size is effectively unbounded (MEDIUM)

- **File:** `signal.py:52-61`.
- **Issue:** The boundary bounds *some* dimensions and not others. `session_id` must be a
  non-empty string but has no length limit (`:52-53`); each query is capped at 8192 chars
  (`:56`); `previous_results` is capped at **64 entries** but neither the total string length
  nor the length of any single id is checked (`:61`) — `bid.strip()` accepts an arbitrarily
  long token. A single call can therefore carry 64 ids of any size (a 6.4 MB argument is
  trivially constructible) and every byte is persisted verbatim into the JSONL record
  (`interaction_signals.py:346-348`) and re-parsed on every later call (T-009.5).
- **Impact:** Disk amplification against the workspace volume from an unprivileged MCP client;
  compounds T-009.5's quadratic re-read.
- **Fix:** Bound `session_id` and each block id to the same order as the existing id format, and
  bound the total serialized record; reject rather than truncate, matching the tool's existing
  reject-on-oversize style at `:56`.

### T-009.7 — Signal text bypasses the invisible-codepoint sanitiser (LOW)

- **Files:** `signal.py:19-99` and `interaction_signals.py:346-348` (no sanitiser call) versus
  `codepoint_sanitize.py:1-33` and its four existing call sites — `inbox.py:47,144`,
  `entity_ingest.py:25`, `importers/engine.py:568`, `ingestion_pipeline.py:29`.
- **Issue:** Every other ingest boundary strips Unicode tag characters (U+E0000-U+E007F),
  zero-width characters and bidi controls before content enters the corpus, config-gated
  default ON. The signal path applies nothing: query text is persisted byte-for-byte, and
  `Signal.from_dict` (`interaction_signals.py:223-236`) re-hydrates it with `str()` casts and
  no re-validation on read.
- **Impact:** Today: an operator reviewing the ledger, or any tool rendering it, sees text that
  hides instructions in invisible codepoints. On wiring: `build_training_tuples` copies
  `new_query` straight into `TrainingTuple.query` (`online_trainer.py:74-95`), so the hidden
  payload becomes training text.
- **Fix:** Call the existing `sanitize_text_for_ingest` at the `observe_signal` boundary. Do
  not add a second sanitiser — the mechanism already exists and is already config-gated.

### T-009.8 — Ledger permissions depend on who created `memory/` (LOW)

- **Files:** `interaction_signals.py:283` (`os.makedirs(..., exist_ok=True)`, default mode),
  `:346` (`open(..., "a")`, default mode) versus `init_workspace.py:73, 367`
  (`os.makedirs(path, mode=0o700)`).
- **Measured** under `umask 0002` on this host: when `SignalStore` creates the directory itself,
  the result is `memory/` at `0775` and `interaction_signals.jsonl` at `0664`.
- **Impact:** In a workspace created by `mind-mem-init`, `memory/` is `0700` and the ledger is
  protected by the directory regardless of its own `0664` mode — no exposure. In a workspace
  where the signal store created the directory first (a path not initialised by
  `mind-mem-init`), the ledger of every query the operator typed is group- and world-readable
  on a shared host. Low, because the documented install path is unaffected.
- **Fix:** Create the file `0600` and the directory `0700`, matching the in-repo precedent for
  a sensitive append-only log — the `decrypt_file` audit trail chmods `0600` on each append
  (`mcp/tools/encryption.py:88-93`), and `mm_cli.py:1666-1700` documents why `os.open` with an
  explicit mode beats `write_bytes` + best-effort `chmod`.

---

## Latent — becomes live the moment the trainer is wired

These are not exploitable today (there is no caller). They are recorded because "wire up a
`train_step`" is a small change that would silently inherit all of them.

### T-009.1 — No provenance or authenticity check on weight refs (MEDIUM, latent)

- **Files:** `online_trainer.py:105-122` (`WeightRef`), `:135-141` (`set_active`/`set_candidate`),
  `:152-178` (`promote`).
- **Issue:** `WeightRef.path` is an opaque string. Nothing in the registry stats it, hashes it,
  checks a signature, or scans the checkpoint. `set_candidate` accepts any ref; `promote`
  copies the path forward unexamined (`:171`).
- **Impact:** On wiring, whatever writes a candidate ref decides what the "active" model is,
  with no artifact-level check. Anything that can influence that call — a config file, a
  registry populated from a downloaded manifest, a future MCP tool — swaps model weights.
- **Fix:** Do not build a new mechanism; the repo already ships one.
  `verify_model_tool` (`mcp/tools/model.py:272`) verifies a signed checkpoint,
  `sign_model_tool` (`:150`) signs one, and `audit_model_tool` (`:88`) wraps
  `model_audit.audit_model` (`model_audit.py:438`) whose checks include `check_pickle_safety`
  (`:283`), `check_remote_code_hooks` (`:109`) and `check_safetensors_header` (`:386`);
  `model_provenance.check_provenance` (`model_provenance.py:166`) checks publisher lineage.
  Require an audit PASS plus a signature verification before `set_candidate` accepts a ref.
- **Adjacent, worth noting for the same reason:** `mm install-model` fetches the GGUF over HTTPS
  from a hardcoded repository and validates only `Content-Length` (`mm_cli.py:640-645, 713-740`).
  URL scheme/host pinning, path-traversal rejection, system-path refusal, symlink refusal and
  atomic rename are all present and well done (`:626-680, 731-737`) — but there is no hash or
  signature check on the artifact, while the signing tools above exist in the same package.
  Out of trainer scope; in scope for whatever eventually points a `WeightRef` at a file.

### T-009.2 — The promotion gate is vacuous with no baseline, and trusts a self-reported metric (MEDIUM, latent)

- **File:** `online_trainer.py:159-178`, and `promote_candidate` at `:211-233`.
- **Issue (a):** the regression check is `if prev is not None and new_mrr < prev.base_mrr + min_improvement`
  (`:164`). With **no active weights registered**, `prev is None`, the branch is skipped, and any
  candidate promotes at any MRR — `0.0`, or negative. First promotion is unguarded by
  construction.
- **Issue (b):** `new_mrr` is supplied by the caller and stored verbatim as the new baseline
  (`:172`). Nothing recomputes it, and nothing links it to the A/B harness that exists for
  exactly this purpose — `evaluate_ab` (`interaction_signals.py:469-505`), which replays real
  signals against baseline and candidate retrieval functions and returns both MRRs.
- **Issue (c):** `promote_candidate`'s `baseline_mrr` parameter (`:216`) is decorative. It is
  used only to compute the reported `improvement` (`:220, 231`); the gate that actually ran used
  `prev.base_mrr` from the registry (`:164`). The two can disagree, so a decision dict can read
  as a governed comparison against a baseline the gate never used.
- **Failure scenario:** a wired caller passes `candidate_mrr=0.99, baseline_mrr=0.0` from an
  eval it computed itself. `promote_candidate` returns `{"promoted": true, "improvement": 0.99}`
  and the swap is recorded as governed, while the registry either had no baseline at all (a) or
  compared against a different number (c).
- **Fix:** reject promotion when no baseline is registered unless an explicit
  `allow_initial_promotion` flag is passed; derive `new_mrr` from `evaluate_ab` over the signal
  ledger rather than accepting it from the caller; and either drop `baseline_mrr` or have
  `promote_candidate` assert it equals the registry's `prev.base_mrr`.

### T-009.3 — Promotions are unaudited; only reverts are (LOW-MEDIUM, latent)

- **File:** `online_trainer.py:133` (`_revert_events` deque), `:180-194` (`revert` appends an
  event), `:159-178` (`promote` appends nothing), `:196-205` (`stats` reports only
  `revert_events` count).
- **Issue:** The module docstring advertises "version-stamped weight refs … + audit log" and a
  "governance-gated swap" (`:11-13`), but the only event recorded is the revert. A promotion —
  the security-relevant transition — leaves no record beyond the mutated in-memory pointer, and
  the record it does leave is a bounded in-process deque, not the tamper-evident chain this
  repository ships (`audit_chain.py`, `hash_chain_v2.py`).
- **Impact:** On wiring, a model swap is not reconstructible after the fact. This is the same
  class of gap the 2026-04-28 report raised as T-007 for the audit log, one layer up.
- **Fix:** record promote events on the same bounded structure as reverts at minimum; write them
  to the audit chain when the trainer is wired to anything that persists.

### T-009.4 — Registry state is in-process only (LOW, latent; also the answer to "what lands on disk")

- **File:** `online_trainer.py:128-133` — three plain dicts and a deque behind an `RLock`.
- **Issue:** Nothing is persisted. A restart loses active, candidate, rollback and the revert
  log; `revert()` then returns `False` (`:182-184`) because the rollback slot is empty.
- **Impact:** Not a disclosure risk — it is why the trainer's on-disk footprint is *nothing*
  (see question 5). It is a correctness trap for wiring: the auto-revert hook the roadmap
  describes does not survive a process restart.

### T-009.9 — The corpus→training-data path already exists, dead-coded (MEDIUM, latent — this is the roadmap's trigger)

- **File:** `generate_mind7b_training.py:559-577` (`load_workspace_blocks`), `:579-590`
  (`make_real_entity_example`), `:597-640` (`main`), `:19` (`random.seed(42)`).
- **Issue:** `load_workspace_blocks()` reads `<ws>/decisions/DECISIONS.md` and
  `<ws>/intelligence/SIGNALS.md` and regex-extracts block bodies; `make_real_entity_example()`
  formats such a body into a chat-format training record. `main()` calls **neither** — the
  2000 emitted examples are synthetic templates from a fixed seed. So today the generator
  ingests no corpus content, and the file is not shipped (`pyproject.toml:141-146` packages only
  `src/`), making it a developer tool rather than a product surface.
- **Impact:** `SIGNALS.md` is where governance writes proposal statements and rationales —
  attacker-influenceable content under this threat model. Calling those two functions turns that
  content into training data with no filter, no provenance tag, and no review. That is precisely
  the shape T-009 was written to prevent, and it is one line away.
- **Note on the existing sanitiser:** proposal rationale *is* sanitised before it reaches
  `SIGNALS.md` (`mcp/tools/governance.py:279-282` → `apply_engine.py:1525-1551`), which closed
  T-003. But that sanitiser escapes **Markdown framing** — block delimiters, headers, and
  governance-looking key lines. It does not, and is not meant to, neutralise instruction-shaped
  natural language, which is the only thing that matters once the text is training data. Do not
  treat the T-003 fix as coverage for this path.
- **Fix:** leave both functions uncalled, and gate them behind the pre-ingest checklist below if
  they are ever enabled.

---

## No finding in this category — and why

**Poisoned block or proposal content reaching the trainer.** No path. The only defined trainer
input is an `interaction_signals` record, whose fields are two query strings, a session id, a
similarity float and a tuple of block **IDs** (`interaction_signals.py:191-207`). Block bodies,
proposal statements and rationales are never read by `interaction_signals.py` or
`online_trainer.py`. `build_training_tuples` consumes `new_query` and `previous_results`
(`online_trainer.py:70-95`) — text the operator typed and opaque identifiers. Corpus content
enters training data only via T-009.9, which is dead-coded in a non-packaged script.

**SSRF in the trainer** (carried open from `api-security-review-2026-04-28.md:148`, which asked
whether the trainer fetches URLs derived from proposal content). It does not. The module's
entire import set is `threading`, `time`, `collections.deque`, `dataclasses` and `typing`
(`online_trainer.py:27-34`) — no `urllib`, no `requests`, no `subprocess`, no `open`.
Independently corroborated by the pinned bandit baseline, which records 0 findings at every
severity and 0 `nosec` markers for the file
(`docs/security-baselines/bandit-v3.2.0-baseline.json:1565-1577`).

**Unbounded memory growth in `TrainingLoop`.** Bounded by construction: the buffer is
`deque(maxlen=buffer_cap)` (`:267`) defaulting to 100 000 (`:34`), with an explicit
`overflow_dropped` counter (`:278-280`) and a constructor that rejects `batch_size < 1` and
`buffer_cap < batch_size` (`:255-259`). One behavioural nuance worth recording rather than
filing: overflow evicts the **oldest** tuple, so a flood of late samples silently displaces
earlier ones — a training-distribution shift, not an exhaustion bug.

**Poison-pill DoS in `try_flush`.** No wedge is possible: the batch is `popleft`-ed while
holding the lock *before* `train_step` is called (`:287-292`), so a callable that always raises
still drains the buffer and the loop terminates. The failure is counted (`_errors`, `:298`) and
the batch is dropped silently — worth a caller-facing note, not a security finding.

**GPU or CPU exhaustion attributable to the trainer.** The module spawns no thread, process or
timer. `train_step` runs synchronously on the submitting thread (`:292`). Any compute cost
belongs to the caller-supplied callable, which does not exist in this tree.

**Injection through the training-tuple builder.** `build_training_tuples` (`:60-96`) rejects
non-`Mapping` entries (`:70`), coerces every field with `str()`, skips records with an empty
`new_query` (`:75-76`), and assigns weights from three hardcoded constants (`0.75 / 1.0 / 1.25`).
It performs no lookup, no eval, no formatting into a template. It also does not bound query
length or id count — but the only in-tree producer of its input is the MCP boundary, which does
(`signal.py:56, 61`).

**Docstring drift, recorded as documentation debt.** The module docstring advertises a
`SignalHarvest` class (`online_trainer.py:9`) that does not exist; the shipped equivalent is the
`build_training_tuples` function (`:60`). No security consequence.

---

## Answers to the six review questions

1. **What data reaches the trainer, and from where.** Query text and block IDs, from
   `observe_signal` (USER scope — `acl.py:114-115`), persisted to
   `<ws>/memory/interaction_signals.jsonl` (`_helpers.py:56`). Nothing consumes that ledger for
   training: the chain is broken between the store and `build_training_tuples`. Full trace above.
2. **Can a poisoned proposal or block influence the fine-tune, and what is the gate?** Not
   today, and the reason is stronger than a gate: there is no fine-tune, and corpus content is
   not in the trainer's input schema. The gates that *would* apply on wiring are the promotion
   check (`online_trainer.py:164`, vacuous with no baseline — T-009.2) and nothing else. The one
   real corpus→training path is T-009.9, dead-coded and ungated.
3. **Authenticity or provenance check on training samples?** None, in either direction.
   Signal records carry no origin tag distinguishing operator-typed from agent-supplied queries
   (`Signal`, `interaction_signals.py:191-207`), even though the block side has exactly that
   vocabulary — `content_source` with a normalising writer and a fail-closed reader
   (`block_provenance.py:85, 115, 148`). Weight refs carry no signature or hash check
   (T-009.1), despite `sign_model_tool` / `verify_model_tool` / `audit_model_tool` shipping in
   the same package.
4. **Resource exhaustion.** Live: unbounded ledger with a full re-read per call (T-009.5,
   measured) and unbounded per-record size (T-009.6). Not present: trainer memory is capped
   (`deque(maxlen=…)`), and the trainer consumes no GPU or disk of its own.
5. **What lands on disk, with what permissions.** From the trainer: **nothing** — the registry
   is in-process (T-009.4). From the path feeding it: one append-only JSONL ledger, created
   `0664` inside a `memory/` directory that is `0700` under `mind-mem-init` and `0775` when the
   store creates it (T-009.8, measured). From the generator: `mind7b_train.jsonl` written to the
   current working directory with default permissions (`generate_mind7b_training.py:628`).
6. **What changes if external training-data ingest is added.** Everything above flips from
   latent to live at once. The pre-ingest checklist follows.

---

## Pre-ingest checklist — required before any external training-data ingest lands

The roadmap named external ingest as the trigger for this review. These are the conditions that
must hold *before* such a change, each mapping to a finding above:

1. **Provenance tag on every training sample** — origin recorded at capture and carried
   through `build_training_tuples`; refuse to train on external-origin text without an explicit
   opt-in. Reuse the block side's `content_source` vocabulary
   (`block_provenance.py:85, 115, 148`) rather than inventing a second one. (T-009.3,
   question 3.)
2. **Sanitise at the capture boundary** — `sanitize_text_for_ingest` on `observe_signal`, using
   the existing sanitiser, before any text can become training text. (T-009.7.)
3. **Bound the ledger and the record** — size cap plus rotation on the JSONL, length caps on
   `session_id` and on each block id, and a persistent dedup index instead of a full re-read per
   construction. (T-009.5, T-009.6.)
4. **Derive the promotion metric, never accept it** — `new_mrr` from `evaluate_ab`
   (`interaction_signals.py:469`) over the recorded signals; refuse a first promotion with no
   baseline unless explicitly flagged. (T-009.2.)
5. **Verify the artifact before it can become `active`** — `audit_model` PASS plus signature
   verification in `set_candidate`, reusing `mcp/tools/model.py`. (T-009.1.)
6. **Audit the promotion, not just the revert** — a promote event on the same structure as
   `_revert_events`, and on the audit chain once anything persists. (T-009.3.)
7. **Keep `load_workspace_blocks` uncalled** — or, if corpus-derived training data is genuinely
   wanted, put it behind items 1, 2 and an explicit operator approval, and do not rely on the
   Markdown sanitiser as content review. (T-009.9.)

---

## Honest gaps in this review

- **Dynamic analysis was limited to what could be run without wiring.** The permission and
  ledger-scaling numbers are measured on this host (`umask 0002`); the trainer's behaviour under
  a real `train_step` could not be exercised because no implementation exists.
- **`mm install-model` is noted, not audited.** It is adjacent (it is how a weight file arrives)
  but out of this module's scope; a focused pass on the model-delivery path — hash pinning,
  signature verification, Ollama `Modelfile` construction — is worth queuing separately.
- **Not covered here**, still open from `threat-model-2026-04-28.md:80-86`: `dream_cycle`,
  `kalman_belief`, `speculative_prefetch`, `skill_opt/`.

---

## Remediation summary

| ID | Finding | Severity | State | Target |
|----|---------|----------|-------|--------|
| T-009.5 | Unbounded ledger + full re-read per call | MEDIUM | live | v3.2.x |
| T-009.6 | Unbounded per-record size | MEDIUM | live | v3.2.x |
| T-009.7 | Signal text skips codepoint sanitiser | LOW | live | v3.2.x |
| T-009.8 | Ledger `0664` / dir `0775` when self-created | LOW | live | v3.2.x |
| T-009.1 | No weight-ref provenance or signature check | MEDIUM | latent | before wiring |
| T-009.2 | Vacuous first promotion; self-reported MRR | MEDIUM | latent | before wiring |
| T-009.3 | Promotions unaudited (reverts only) | LOW-MED | latent | before wiring |
| T-009.4 | Registry state not persisted | LOW | latent | before wiring |
| T-009.9 | Dead-coded corpus→training-data path | MEDIUM | latent | before external ingest |

---

**Report metadata**
- Generated: 2026-08-31
- Closes: **T-009** (`ROADMAP.md`), opened in `security/threat-model-2026-04-28.md:60-63`
- Answers the carried question in `security/api-security-review-2026-04-28.md:148` (no SSRF)
- Methodology: STRIDE + line-level source trace + measured permission/scaling checks; read-only
- Reviewer: STARGA Inc.
- Next review: when a `train_step` implementation or any external training-data ingest lands
