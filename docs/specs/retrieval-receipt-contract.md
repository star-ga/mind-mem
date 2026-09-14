# Retrieval receipt contract

Status: **Draft specification, 2026-09-14. Implementation not started.**
Owner: STARGA Inc. Canonical milestones: [Group RE](../../ROADMAP.md#group-re--portable-retrieval-evidence).

This specification defines a proposed export and verification boundary over
MIND-Mem's existing serving evidence. It does not introduce a payment requirement
for ordinary memory use. The field names and interfaces below are proposed;
they are not currently available CLI commands, MCP tools, or a published wire API.

## 1. Purpose and priority

An operator should be able to answer: what did this retrieval return, which
recorded context produced it, and what can another verifier establish from the
available evidence? This is useful for debugging, provenance audits, incident
review, and accountable usage without charging anyone for a retrieval.

The first implementation should reuse the served ledger and existing digest
functions. A second retrieval ledger, a payment service in the core package, or
a new ranking signal would add responsibilities without addressing that question.

Commercial demand is unestablished. Pricing, billable-unit aggregation and
settlement remain optional extensions behind the demand gate in section 9.
They are not prerequisites for the next release, completion of the Pure-MIND
migration, or model training. Receipt work follows the existing release and
retrieval-correctness priorities; it does not close those milestones by itself.

## 2. Existing implementation and limits

The baseline examined is source revision
`58a86e23517fcd415152aceff6593d5b747a00e7`. This is a source inventory, not
evidence of publication or deployment of this proposed interface.

| Existing component | Reuse | Limit to preserve |
| --- | --- | --- |
| [`recall_digests.py`](../../src/mind_mem/recall_digests.py) | Ordered served-result digest and content-derived `run_id`. | `run_id` identifies an answer, not an individual occurrence. |
| [`served_ledger.py`](../../src/mind_mem/served_ledger.py) | `attach_served_run`, V1/V2 decoding, row hashing, chain and head verification. | Local rows and head can both be replaced by an actor controlling the directory. |
| [`mcp/tools/recall.py`](../../src/mind_mem/mcp/tools/recall.py) | Existing corpus and local-anticipation attachment paths. | Recording is synchronous after retrieval computation and before the handler returns; this is not an asynchronous external witness. |
| [`accountability_views.py`](../../src/mind_mem/accountability_views.py) | Derived serving and outcome views. | Counts and reported outcomes do not independently prove usefulness or payment eligibility. |
| [`outcome_attribution.py`](../../src/mind_mem/outcome_attribution.py) | Governed, validated run-bound outcome association. | An answer-bound outcome does not distinguish each repeated occurrence of that answer. |

The existing views combine durable ledger observations and windowed retrieval
logs while retaining source labels. Do not sum those views as independent
billable events. Run-scoped views keyed by `run_id` can collapse repeated answers;
missing historical intent or outcome remains unknown, not zero benefit.

V2 rows carry `serve_kind` and `context_digest`; V1 rows do not. Preserve the
existing V1 sentinel `unrecorded` for the missing serving kind. It means the kind
was not recorded, not that the row is absent. An `anticipation` receipt describes
a local bundle serve, not a fresh corpus `RECALL_ATTEST`. RA.1 remains partial;
the adapter must inventory actual serving coverage rather than assume it.

The existing attachment seam reports `served_proof=recorded` or `unproven`.
Some response paths lack an attestation entirely. A proposed exporter must
classify missing evidence explicitly; it must not manufacture an attestation
or silently read today's configuration to fill historical gaps.

## 3. Responsibilities and invariants

1. **Retrieval and admission:** the existing retrieval/governance path decides
   what may be returned. An external witness or receipt verifier cannot grant
   access, authorize a write, or overrule a refusal.
2. **Measurement:** record the observed event and its evidence scope. Returning
   blocks does not prove that the caller consumed, benefited from, or paid for them.
3. **Export:** project existing evidence into a versioned, bounded verification
   package. Export must not mutate the corpus, rank results, or re-anchor a chain.
4. **Verification:** report separate results for content binding, local chain
   consistency, issuer trust, and independently retained history. A hash alone
   does not establish all four.
5. **Optional economics:** a downstream consumer may apply voluntarily agreed
   terms to eligible measurements. Neither the receipt nor its witness forces a
   transfer. A service provider may separately require authorization under its
   access policy; that authority does not belong to the witness.
6. **Governance separation:** serving frequency, reported success, price and
   payment MUST NOT silently alter ranking, confirmations, retention or trust.
   Existing proposal and approval requirements remain in force.

These are logical roles, not a requirement for separate services or databases.
The core feature MUST remain useful with no network, payment account or external
witness configured, and MUST add no mandatory third-party dependency.

### 3.1 Upstream agreement and execution boundary

For the optional contracted-retrieval profile, the parties or their delegated
agents negotiate and accept terms upstream. The request arrives with a bounded,
authenticated agreement binding: parties/identity, permitted use, price or rule,
service scope, delegated limits, policy revision and revocation state.

The execution sequence is:

**Parties/agents agree -> 512 enforces the agreed constraints -> MIND-Mem serves
an admitted request -> CVS records the event -> downstream aggregation/settlement.**

512 MUST NOT negotiate, invent, widen or silently replace the terms. It checks
the supplied agreement and requested operation at the execution boundary. A
missing, invalid, out-of-scope, expired or revoked authorization stops that
contracted retrieval before protected data is served. A witness may record the
refusal under its permitted disclosure profile; it cannot turn refusal into
permission. Normal local memory use does not acquire a paid-service requirement.

### 3.2 Required constitutional lineage

Cryptographic lineage is a required architecture correction for a profile
claiming canonical 512 conformance. It is independent of commercial demand.
Keep three identities distinct:

| Identity | Meaning |
| --- | --- |
| `canonical_512_hash` | Immutable commitment identifying the original 512 constitution claimed. |
| `implementation_spec_hash_v2` | Versioned commitment binding that root to the 512-MIND decomposition, registry and implementation. |
| `mind_language_spec_ref` | Separately typed MIND language-spec identity; not an alias for the implementation hash. |

The required dependency is **constitution -> implementation -> event**. The
implementation preimage must include the canonical root, not merely display
two unrelated hashes beside each other. Runtime verdict and exported event
evidence must cryptographically bind both, plus the applicable MIND spec and
executed artifact identities. Changing the implementation changes its hash;
the canonical root stays fixed unless a different constitution is explicitly
selected under a new profile.

The 512-MIND producer owns the versioned, domain-separated preimage and manifests.
The receipt adapter consumes and verifies that binding; it must not invent a
second implementation-hash algorithm. Use unambiguous typed/length-delimited
encoding, resolve the full original commitment and its exact preimage, and test
each changed/missing field. A hash authenticates declared lineage, not semantic
conformance; invariant mapping and execution tests remain separate requirements.

This is a new required contract, not a claim that the current MIND-Mem rows or
every 512-MIND runtime proof already carry it. Existing `spec_hash` and historical
evidence retain their original preimages. Legacy records remain readable with
lineage status unknown; they cannot pass a new lineage-required profile through
an invented field or a silent fallback. Keep lineage in the appropriate evidence
plane: this does not authorize action identifiers or reward signals in an
I13-governed non-causal structural witness.

## 4. Identity and candidate data contract

### 4.1 Three distinct identities

Keep the existing `run_id` bytes and meaning unchanged. Two invocations producing
the same committed answer may legitimately share that value.

The proposed occurrence key is the tuple
`(issuer_id, ledger_id, seq)`. Bind that key to the existing derived `row_hash`.
Two records with the same occurrence key and different row hashes are a conflict,
not two billable occurrences. A redelivery of the same verified occurrence is
idempotent. A new retrieval with the same answer and a new sequence is distinct.

`issuer_id` is an authenticated, scoped issuer identity, not an arbitrary claimed
name. `ledger_id` identifies a provisioned append history, never a filesystem
path, hostname, query hash or current chain head. Its lifecycle MUST be specified
and tested before portable occurrence identity is enabled:

- A restart or exclusive move preserves the identity and history.
- A read-only replica can export the same occurrences; it cannot mint new ones.
- Concurrent independent writers and writable clones cannot share the identity.
- Recovery or a fork creates a new identity with an explicit predecessor
  reference; it never silently resets an existing sequence under the old identity.
- Historical rows are exportable, but provisioning an identity later does not
  prove who wrote them or that the history was complete before provisioning.

Request retry identity is separate. The existing row does not prove that two
retrievals were retries of one authorized request. A future contracted-service
profile needs an authenticated request/idempotency binding at admission time;
the exporter MUST NOT infer one afterward from matching `run_id` values.

### 4.2 Proposed export record

The following logical fields define the draft `MM_RETRIEVAL_RECEIPT_v1` profile.
Schema, encoding and golden vectors must be frozen together before implementation
is called interoperable. A record is a projection; the original row remains
authoritative for its existing commitments.

| Field | Type and meaning | Requirement |
| --- | --- | --- |
| `schema` | Exact versioned profile identifier. | Required; unknown versions are unsupported, never guessed. |
| `occurrence` | Object containing `issuer_id`, `ledger_id`, `seq`, `row_hash`. | Required for a portable occurrence claim; otherwise report missing identity. |
| `row_version` | Integer `1` or `2`. | Required; preserve the exact historical schema and hashing domain. |
| `answer` | Existing `run_id`, `query_hash`, `served_digest`, `pipeline_hash`. | Copy and verify from the committed row; no new served-result encoding. |
| `context` | Existing `index_anchor`, `scoring_instant`, and V2 `context_digest`. | Preserve V1 absence; do not infer an ACL decision from a context digest. |
| `serve_kind` | `attested`, `anticipation`, or V1 `unrecorded`. | Must agree with the row version and actual recorded kind. |
| `evidence` | References to supplied row, checkpoint and verification material. | Each reference binds bytes and scope; a pointer alone is not proof. |
| `disclosure_profile` | Versioned allowlist of fields and intended audience. | Required before any export outside the workspace trust boundary. |
| `measurements` | Versioned quantities with explicit units and input commitments. | Optional; absence means unmeasured, never zero. |
| `agreement_binding` | Verified agreement revision, parties, authorized use and request binding. | Optional extension; absence means no economic eligibility claim. |
| `governance_lineage` | Producer-profile reference, canonical root, implementation commitment, MIND spec/artifact references and verification material. | Required whenever canonical 512 lineage is claimed; absence is unknown in legacy/local records. |

Existing digest widths and encodings are retained. New quantities use bounded
integers or canonical decimal strings with explicit scale; binary floating-point
values MUST NOT enter a financial calculation or its canonical preimage.
The wire profile must reject duplicate keys, unknown required fields, mixed
versions, non-finite numbers, coercions and oversized packages. It must specify
UTF-8, escaping, ordering, maximum lengths, domain separation and exact hash
preimages in its machine-readable schema and committed vectors. Those wire
choices remain a blocking deliverable of RE.1, not implicit runtime defaults.

An export hash may identify one serialization or disclosure projection. It MUST
NOT replace the occurrence key for deduplication: two authorized projections of
the same event can have different bytes without representing different events.

The initial local-only profile may return `portable_identity=unavailable`; it
can still inspect a row under local trust. It MUST NOT synthesize issuer/ledger
identity or make external tenant/admission claims. Existing outcome joins remain
answer-level unless an authenticated occurrence join is added and tested.

### 4.3 Proof packages and verification result

For a first local audit profile, export a bounded ledger snapshot or sufficient
verified prefix together with its head and selected row. The current chain is
linear; do not claim compact Merkle membership proofs from its row hash alone.
Existing read APIs do not provide an atomic row/head snapshot. The exporter must
acquire the append lock, capture ledger bytes, head, file identity and length,
then release the lock and verify only the captured snapshot. Fail on capture
error or identity change; test an append racing capture. An alternative atomic
snapshot mechanism needs equivalent controls. Do slow export I/O after capture.

Do not infer absence from an empty tuple returned by a convenience reader:
current readers can collapse an `OSError` into no rows. Use an error-preserving
preflight/read seam and distinguish missing, empty, unreadable, malformed and
partially read inputs. RE.1 must freeze numeric byte/row/reference/identifier
limits, lock deadlines, memory/time budgets and pre/post-decompression bounds.
Exhaustion produces an explicit incomplete/unavailable result, never a passing
partial package. Use bounded streaming or a capped consistent snapshot.

The candidate logical interfaces are:

| Operation | Inputs | Result |
| --- | --- | --- |
| `export_receipt` | Workspace, row selector, disclosure profile, identity binding. | Receipt and evidence package, or explicit unavailable/refused result. |
| `verify_receipt` | Package, supported schema, caller-supplied trust policy. | Structured verification report with individual checks and scope. |
| `aggregate_receipts` | Verified receipts plus accepted agreement and calculation profile. | Optional billable-unit manifest; not part of the core implementation. |

These are interface responsibilities, not installed commands. Select Python/CLI
entry points by extending existing verification/export conventions; do not add
an MCP tool merely to increase the surface count.

The report must distinguish `unsupported`, `malformed`, `unavailable`,
`integrity_failed`, `untrusted_issuer`, `locally_consistent`, and
`externally_anchored` conditions, with individual check results. Local consistency
does not imply issuer trust; issuer authentication does not imply external
history retention. Missing checks remain unknown rather than passing by default.
An externally anchored claim requires a validated commitment retained by an
independent party and a verified link from the selected row to that commitment.
It still does not prove unobserved events or truthful physical measurement.

This specification adds no signature scheme. An external profile must reuse the
versioned ecosystem evidence policy, bind issuer keys to an independently trusted
identity, and refuse missing required algorithms or silent downgrade. A public
key delivered only inside the package is not an independently trusted identity.
Unsigned local inspection remains explicitly local inspection; it is not a claim
that releases or exported receipts are signed today.

## 5. Availability, scope and privacy

Ordinary recall retains its current availability behavior when evidence recording
fails. Exporting or validating a receipt is a separate operation and MUST refuse
an unsupported evidence claim. Never convert `served_proof=unproven`, a missing
attestation or an unavailable ledger into a verified receipt.

If a separately agreed service requires durable accounting before releasing a
result, specify that as an opt-in service-admission profile with its own failure
contract. It cannot silently change ordinary recall or give CVS veto authority.

Admission evidence must bind the authenticated actor/tenant, policy revision,
authorized purpose and decision point if a profile claims them. Existing row
fields alone do not establish these facts. A future revocation profile must
define propagation bounds, cache staleness, partitions and in-flight requests;
no instantaneous or presently implemented epoch-revocation claim is made here.

Default export is local and explicit. External disclosure requires a reviewed
allowlist. Raw queries, source text, credentials and private filesystem paths
must not be placed in a public receipt or ledger. IDs, timestamps and hashes may
still permit correlation or guessing. Redacting a committed row cannot preserve
full independent row verification unless the selected proof profile actually
supports that projection; otherwise refuse the stronger claim.

## 6. Optional billable units and settlement

A measurement receipt can be included downstream in a billable unit. One receipt
does not require one immediate charge or one network transaction. Charging depends
on accepted terms and delegated authority; a price declaration by the provider
is not acceptance by the buyer.

An optional agreement profile must bind parties, revision, allowed service/use,
units, tariff, currency or denomination, budget, retry policy, revocation rules
and evidence requirements. Any agent spending authority must be explicitly
delegated and bounded. Similarity, returned token volume and reported outcomes
are measurements or tariff inputs, not a proof of causal contribution.

A billable-unit manifest must commit to the exact eligible occurrence set,
agreement revision, calculation version and totals. It must define duplicate
handling, empty sets, missing evidence, corrections, reversals and partial
eligibility. Conflicting copies cannot be resolved by silently keeping one.
Corrections append lineage rather than rewriting earlier receipts.

Metering specifies precision, units, rounding stage/mode, permitted range,
overflow, zero-budget behavior and input provenance. No relevance-to-price
formula is selected by this specification. Independent implementations must
produce identical totals from identical accepted inputs before financial use.
This does not make different embedding providers' outputs bit-identical.

Settlement is an independent adapter over an authorized billable unit. It must
distinguish submitted, pending, failed and confirmed outcomes. A receipt hash,
public anchor or submitted transaction is not payment confirmation. No network,
custody model, monetary price or funded deployment is selected here.

## 7. Required acceptance evidence

These are future acceptance cases, not a claim that the new adapter passes them.
Reuse [served-ledger tests](../../tests/test_served_ledger.py),
[V1/V2 compatibility tests](../../tests/test_served_ledger_v2_compat.py), and
[attestation isolation tests](../../tests/test_recall_attestation_v2.py).
The adapter also needs end-to-end tests exercising its actual entry point.

| Gate | Positive control | Negative control and required outcome |
| --- | --- | --- |
| RE-A1 Existing semantics | Export authentic V1 and V2 rows, including anticipation. | Alter serving kind or invent missing context: refuse the claim. |
| RE-A2 Occurrences | Re-export one event idempotently; preserve distinct sequences sharing `run_id`. | Same occurrence key with another row hash: conflict, no silent deduplication. |
| RE-A3 Lifecycle | Restart and exclusive move preserve occurrences; replicas only read. | Writable clone, sequence reuse or hidden recovery reset: refuse identity continuity. |
| RE-A4 Integrity and trust | Verify row, chain, checkpoint and a separately trusted issuer when supplied. | Edited/reordered/truncated rows, replaced seal, self-declared key or invalid checkpoint: report the failed or missing proof layer. |
| RE-A5 Missing evidence | Ordinary recall returns its documented result while a failing recorder reports unavailable evidence. | Missing attestation, false status or reconstructed historical policy: cannot become a verified receipt. |
| RE-A6 Disclosure | Authorized local audit receives only its permitted package. | Wrong tenant/purpose, unauthorized fields or incompatible redaction: refuse export or reduce the claim explicitly. |
| RE-A7 Governance isolation | Receipt export and verification leave corpus, ranking and tier state unchanged. | Mutation wiring receipt counts or payment into automatic promotion must fail the isolation gate. |
| RE-A8 Bounds and concurrency | Bounded snapshot and concurrent exports produce consistent packages. | Oversized input, duplicate keys, mixed version, lost lock or resource exhaustion: explicit failure, no silent truncation. |
| RE-A9 Optional economics | Agreed tariff and deduplicated set reproduce a manifest and totals. | No consent, exceeded authority, conflicting receipt, inexact arithmetic or replayed settlement: no eligible charge. |
| RE-A10 Constitutional lineage | Recompute the producer's implementation commitment from the canonical root and manifests, then verify its binding into the actual event. | Changed root/decomposition/registry/implementation, swapped language or artifact identity, unknown legacy lineage or unbound side metadata: refuse a lineage-required claim. |

RE-A2 also covers an event appearing in both the retrieval-log and durable-ledger
views: report source-labelled observations unless a proven occurrence join
exists. Those diagnostic views are not an occurrence-complete billing source.
RE-A5 includes missing, empty, unreadable, malformed and interrupted histories.

RE-A9 is required only for a commercial extension. It cannot be substituted for
the core evidence gates. Passing a verifier on synthetic files is useful parser
coverage but does not prove a real retrieval was executed or validly admitted.
RE-A10 is required for canonical-lineage claims regardless of whether money is
involved. Ordinary local inspection must keep its weaker scope explicit.

## 8. Benchmarks and determinism claims

Compare the same corpus, queries, scoring instant, providers and hardware with
ordinary recall, existing recording, and the proposed export/verification path.
Report latency p50/p95/p99, throughput, CPU, peak memory, bytes per event, export
size, write amplification and failure rates. Include cold/warm state, V1/V2,
anticipation, repeated answers, concurrent clients and long histories.

Set workload-specific regression budgets before the implementation is scored;
record raw results, source revision, dependency versions and hardware identity.
External superiority claims require relevant external baselines under matched
conditions. No SOTA score follows from this design document.

Cross-machine tests compare canonical commitments for identical admitted inputs
and calculation profiles. Different occurrences, sequence numbers, policies or
issuer identities may correctly produce different receipt bytes. The core MIND
output-identity objective does not imply identical native binaries, arbitrary
third-party model scores or differently scoped evidence records.

## 9. Delivery sequence and demand gate

| Stage | Scope | Exit condition |
| --- | --- | --- |
| RE.1 | Freeze the smallest receipt contract and lifecycle rules. | Machine-readable schema, canonical vectors, coverage inventory and trust/disclosure decisions pass independent review. |
| RE.2 | Implement the local export adapter and verifier over existing evidence. | RE-A1 through RE-A8 pass at the actual entry point; no runtime dependency or ranking change. |
| RE.3 | Validate an optional independent evidence-exchange profile. | Authentic external checkpoint/identity controls and privacy review pass; local-only scope remains usable. |
| RE.4 | Run a practical operator pilot and publish scoped results. | At least one actual audit/debugging task demonstrates utility; performance budgets and source/artifact evidence are recorded. |
| RE.5 | Consume the required canonical 512 lineage profile. | Producer contract and manifest resolution are implemented; RE-A10 verifies authentic runtime verdict/event binding and rejects mutations. |
| Commercial discovery | Identify a willing provider and consumer with an agreed metering problem. | Both parties accept the service and evidence model; compare the cost of receipt handling/batching with a simpler usage log or invoice. |
| Optional aggregation | Build a settlement-neutral billable-unit adapter. | Demand gate passes, agreement profile is frozen, and RE-A9 plus independent arithmetic vectors pass. |
| Optional settlement | Integrate a selected authorized arrangement. | Funding, authority, failure/reconciliation and confirmation behavior are specified and tested. |

Stop or defer the commercial branch if there is no willing counterparty, if
existing accounting solves the need adequately, or if the evidence cannot
support the agreed service claim. Retain the useful local audit capability.
Model-training funding and milestones remain separate.

RE.2 depends on RE.1. The local pilot in RE.4 can follow RE.2 without waiting
for optional external exchange in RE.3. Any commercial adapter also requires
the proof profile its counterparties actually accept; local consistency must
not silently stand in for independent history.
RE.5 depends on the 512-MIND producer's canonical-lineage implementation; it
cannot be closed by adding receipt metadata or by this documentation change.

## 10. Documentation and attribution

The [canonical roadmap](../../ROADMAP.md) owns completion state. This file owns
the proposed interface and acceptance contract. The
[historical accountability proposal](../ROADMAP-RETRIEVAL-ACCOUNTABILITY.md)
retains its audit banner; its superseded promotion or tombstone instructions are
not revived by this extension. Update status and evidence links together when
implementation lands, and advertise only the profile actually verified.

This is STARGA's independent integration design, informed by the 512/CVS
invariants (Jon M. Watson) and the retrieval-accountability review dated
2026-09-14. Attribution does not imply external endorsement, validation or
co-authorship. The complete source review and its evidence remain in the
maintainer's review record; this specification does not reproduce the paper.

Copyright 2026 STARGA, Inc.
