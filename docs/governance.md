# MIND-Mem — governance implementation and boundaries

MIND-Mem provides governed memory and retrieval for the MIND ecosystem. This
document maps the five governance layers to current local implementations.
The [roadmap](../ROADMAP.md) owns completion status; the
[retrieval receipt contract](specs/retrieval-receipt-contract.md) defines the
remaining CVS, MIND Witness and canonical 512 lineage work.

## Layer summary

| Layer | What it enforces | Primary source | Verified by |
|---|---|---|---|
| **L1 architectural** | Declared architecture rules and fixture/source checks | `.arch-mind/rules.mind`, `.arch-mind/rescan.py` | `tests/test_arch_mind_rules_gate.py`, `tests/test_arch_mind_fixture_provenance.py` |
| **L2 training-time** | Checkpoint audit, publisher allowlist, training/evaluation provenance | `src/mind_mem/model_audit.py`, `src/mind_mem/model_provenance.py` | `tests/test_model_provenance.py`; training-readiness reports remain separate from model evaluation |
| **L3 request-time** | Authentication, tool scope, namespace admission and rate limits | `src/mind_mem/mcp/infra/`, `src/mind_mem/api/auth.py`, `src/mind_mem/namespaces.py` | Authentication, request-snapshot and namespace controls |
| **L4 memory operations** | Admitted reads, governed writes, contradiction/drift analysis and evidence | `src/mind_mem/admissibility.py`, `src/mind_mem/governance_gate.py`, `src/mind_mem/served_ledger.py` | Admission, proposal/apply, lifecycle and served-receipt controls |
| **L5 continuous** | Required CI and release preflight | `.github/workflows/ci.yml`, `.github/workflows/release.yml` | OS/Python matrix, documentation/identity gates and exact-commit release checks |

## L1 — Architectural

`.arch-mind/rules.mind` declares nine `[arch_rule]` constraints. The pytest
gate checks committed fixtures, pinned thresholds and source import edges.
Fixture regeneration uses a Git archive of the selected commit so nested
worktrees do not become part of the measured product. Historical scans are
retained separately; a prior passing fixture is not proof of a changed tree.

## L2 — Training-time

MIND-Mem ships a local fine-tuned model (`star-ga/mind-mem-4b`, Qwen3.5-4B base). L2 governance covers:

- **Checkpoint audit.** `model_provenance.py` implements a declared-upstream
  publisher allowlist used by the model audit. An allowed publisher name does
  not independently prove the weights' origin or training quality.
- **Training evidence.** The setup and training recipe document the current
  model and planned refresh. The published checkpoint was trained against an
  earlier tool surface. Runtime additions and development-data coverage do not
  establish competence on those additions or an uncontaminated evaluation.
- **Release separation.** A package release, model-weight release and training
  run are separate events. No training replay or bit-identical weight
  reproduction is inferred from a package CI result.

## L3 — Inference-time

MIND-Mem exposes MCP and optional REST/gRPC transports. The MCP serving boundary
provides:

1. **Authentication and scope.** HTTP bearer authentication supplies a verified
   subject and tool scope. One request snapshot binds namespace identity;
   missing or inconsistent authentication context cannot become a broad read.
   Local stdio scope is a separate deployment setting.
2. **Rate limits.** Client identifiers can key rate limits but do not grant
   namespace identity. Namespace access follows the workspace's explicit ACL.
3. **Tool dispatch.** 107 MCP tools have typed schemas and capability checks.
4. **Evidence scope.** Observability logs describe calls. Governed-write and
   serving receipts have their own explicit contracts; a logged invocation is
   not proof of successful execution or semantic correctness.

## L4 — Retrieval-time

1. **Read admission.** Serving paths apply source lifecycle, release and
   namespace rules before returning content. Index metadata is not a substitute
   for canonical source admission. Complete surface review remains tracked in M5.
2. **Analysis.** Contradiction and drift detectors are explicit mechanisms;
   their existence does not mean every request executes a scan or that every
   semantic contradiction is detectable. Chat and graph answers expose
   `semantic_verification: "not_established"`; callers requiring entailment
   verification receive an explicit abstention before generation.
3. **Governed changes.** `propose_update` stages a signal. Proposal approval and
   application are separate governed operations. Required-provenance profiles
   validate attribution without inventing missing legacy fields.
4. **Serving evidence.** Recall attestations and the append-only served ledger
   bind the declared local retrieval result and request coordinates. A recorder
   failure is reported as unproven. Local consistency does not establish an
   independent witness, portable occurrence identity or semantic entailment.
5. **Encryption.** The default block-file format uses a PBKDF2-derived
   HMAC-SHA256 keystream with encrypt-then-MAC. Opt-in `v4.tenant_kms` uses
   AES-256-GCM when its key and optional crypto dependency are configured.
   Existing files require explicit re-encryption to migrate. The local
   FTS5/sqlite-vec recall index remains plaintext; block-file encryption does
   not encrypt every copy of retrieved content.

### Proposal field screening

`propose_update` requires the caller's rationale for both `decision` and `task`
proposals. At least eight non-whitespace characters are required; omitted,
blank and whitespace-padded short reasons are refused before content screening
or proposal writes. The server does not invent a rationale. Clients that
previously omitted the reason on task proposals must now supply it.

The supplied reason is preserved as `Rationale` in the staged signal. When
redaction is enabled, screening covers the statement, rationale,
tags, confidence and supplied provenance fields before any redaction audit
metadata is recorded. A `reject` policy refuses matching content before the
proposal is written. A `redact` policy may rewrite content fields, including
purpose, but refuses a proposal if rewriting would change an identity or
provenance class. `off` and `flag` retain their non-rewriting behavior.

The signal still requires the ordinary governed approval process; retaining
its rationale does not approve it.

## L5 — Continuous

- **CI** on main pushes and pull requests — full pytest matrix (12,411 test functions across the suite; counted from source, so the number is the tree's and not one machine's).
- **Release preflight** requires matching versions, mainline ancestry, passing
  CI for the exact commit, an unused package version, tests, current public
  documentation and readable code-scanning results before publishing.
- **PyPI publication** uses OIDC trusted publishing. Existing released tags and
  withdrawn versions remain immutable.
- **Benchmarks** need their own frozen workloads and source/hardware evidence.
  The release workflow does not establish a new model-quality result or a
  LoCoMo benchmark for every release. Planned long-running and cross-deployment
  studies remain separate roadmap items.

## Cross-repo discipline

The adopted boundary is: agents agree terms, 512-MIND evaluates admissibility,
MIND-Mem serves within the admitted scope, and independent CVS records evidence
through an explicit adapter. Settlement is separate. MIND Witness is the
MIND-specific CVS adapter, not proof that MIND independently witnessed itself.

The canonical 512 commitment, 512-MIND implementation commitment and MIND
language-spec identity identify different objects. Current local spec binding
must not be described as cryptographic canonical lineage until the producer
commitment and runtime consumer tests required by RE.5 exist. The receipt
contract and roadmap track those dependencies explicitly.


### Ledger preservation during backup and restore

Backups and snapshots exclude the audit, evidence, served-run and hash-chain
ledgers, including registered sidecars. Restore checks both the member name
and its existing filesystem destination: alternate separators, interior dot
segments, symlinks and hardlinks cannot turn a ledger into ordinary corpus data.
Ordinary files and non-ledger hardlinks remain restorable.

Legacy directory staging preserves every registered ledger filename, including
sidecars that share an inode. A failed ledger file check aborts staging instead
of silently omitting that ledger. These checks protect the destinations observed
at validation time; they do not establish immunity to a separate process racing
to replace a path between validation and the write.
