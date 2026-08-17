# Design: M5 — enforcement in code, not in the prompt

Status: Draft (2026-08-17) · Owner: mind-mem · Roadmap: Group M / M5

## The principle

A prompt is a request. Code is what holds.

The external prior art states it in one line worth keeping verbatim as an
argument: a summariser is *told* not to include names, but a `scrub`
function is what actually prevents the name from being written. The
instruction has a failure rate; the function does not.

This project already argues exactly that, and it is the reason
`propose_update` → `approve_apply` exists rather than trusting a model to
remember what it wrote. M5 is not adopting a new principle — it is
**auditing whether we live by the one we already published.**

## Why this is worth a sweep rather than a spot check

The governed write path is the surface everyone looks at, so it is the
surface most likely to be correct. The risk lives in the paths nobody
frames as governance: a summariser, a distiller, a compactor, a category
classifier, an export. Each one passes content through a model, and each is
a place where "the model was asked not to" can quietly stand in for a
guarantee.

A first pass over the source supports the concern rather than dismissing
it. `scrub`/`redact` vocabulary appears in essentially one place
(`src/mind_mem/mm_cli.py`) — not in the write path, not in the distillers,
not in the compaction or export paths. That is not proof of a leak; it is
proof that **there is no enforcement layer to point at**, which is the
condition under which prompt-shaped properties survive unnoticed.

## Scope: what counts as a prompt-enforced property

A property is prompt-enforced when *all* of the following hold:

1. A correctness, privacy, scoping, or governance guarantee is being made
   (explicitly in a docstring or doc, or implicitly relied upon downstream).
2. The only thing standing between the guarantee and its violation is text
   in a model prompt.
3. No code path validates, filters, or rejects the model's output against
   the guarantee afterwards.

Point 3 matters: a prompt *plus* a validating gate is fine, and is in fact
the target state. The finding is the unguarded case.

## Method

Enumerate every call site where a model produces content that is then
persisted, exported, or shown across a scope boundary. For each, record:

| Field | Meaning |
|---|---|
| Site | file:line of the model call |
| Property | what the prompt asks the model to guarantee |
| Blast radius | what breaks if the model ignores it once |
| Enforcement | `code` / `code-after-prompt` / **`prompt-only`** |
| Replacement | the specific function that would make it hold |

Candidate surfaces to sweep, drawn from the module layout rather than
guessed: the distillers and summarisers, compaction, the auto-resolver and
conflict-resolver paths, the export path, the abstention and calibration
classifiers, agent messaging and the agent bridge, and any capture path
that ingests third-party content.

Two questions sharpen every row:

- **Cross-scope reads.** A block written from one context and read while
  answering in another is where identity and tenancy leak. Anywhere the
  read scope is wider than the write scope, prompt-shaped exclusion is not
  sufficient by construction.
- **"The model won't" as an assumption.** Grep the docstrings and comments
  for reassurances rather than mechanisms — a sentence saying content "is
  not included" with no function performing the exclusion is exactly a
  finding.

## Deliverable

**A list, not a refactor.** Each row names a prompt-enforced property and
the code-enforced replacement that would hold it. Prioritised by blast
radius: a property whose violation crosses a tenant or identity boundary
outranks one whose violation is merely untidy.

Implementing the replacements is deliberately *not* in this item. Bundling
the audit with its fixes guarantees the audit stops early — the first
interesting finding becomes a refactor and the sweep never finishes. The
audit's value is completeness.

### How a row closes: a fail-closed capability flag (added 2026-08-17)

A list alone leaves every finding in the state it was found in — the
property is still unenforced, now with documentation saying so. Documented
non-enforcement decays into assumed enforcement: the next reader sees a
governance-shaped function name and infers the guarantee.

The ecosystem already runs the pattern that fixes this. `512-mind` pairs
each unimplemented guarantee with a **capability flag that reads false**,
plus an undefined `extern` so a missing backend fails at link time rather
than resolving to a permissive default:

```
drift.mind:30           fn semantic_mutation_scan_supported() -> u8 { return 0 }
key_management.mind:324 fn signature_verification_supported() -> u8 { return 0 }
```

The load-bearing half is that a caller *reads the flag and refuses*.
`detect_drift` (`drift.mind:50`) returns `equivalent: false` whenever the
scan is unsupported, with the reasoning committed alongside it: *"An
undefined/empty mutation list must NEVER make `equivalent` true — that was
the forgery-by-absence path this fix closes."* Absence of evidence is not
evidence of compliance, and the code is what enforces that rather than a
comment asking the reader to remember it.

So each audit row closes by installing:

1. A predicate naming the property, returning false while unimplemented.
2. A consumer that reads it and takes the conservative branch — refuse,
   redact, or decline to assert — never the permissive one.
3. Where a backend is expected, an undefined symbol so a missing
   implementation fails loudly instead of defaulting open.

This preserves the audit/remediation split: the sweep produces the ranked
list, and flag installation is the queue that list feeds. It changes only
what "resolved" means for a row — a predicate that fails closed, not a
paragraph explaining that the property is aspirational.

**Applies to negative findings too.** If the sweep finds a property is
already code-enforced, no flag is needed; the row closes as verified. Flags
are for the gap between a named guarantee and a real mechanism.

## The honest possible outcome

The sweep may find that every material property is already code-enforced
and the only prompt-shaped ones are cosmetic (tone, formatting, length).
That is a good result and closes the item. It is written down here in
advance so that a clean finding is reported as a clean finding rather than
padded into a list of non-issues to justify the effort.

## Non-goals

- Not a security review. This is an enforcement-layer audit; genuine attack
  surface routes to the security review path.
- Not a prompt-quality exercise. Improving the wording of an instruction is
  the opposite of the point — the finding is that wording is the mechanism.
- No governance weight. The audit output is a measurement artifact: it does
  not enter a block, a hash chain, or the approval gate.

## Done when

- Every model-output-persisted call site is enumerated with its enforcement
  classification.
- Every `prompt-only` row names a specific replacement function **and the
  capability predicate that gates it while unimplemented**.
- The list is ranked by blast radius.
- The result — including "nothing material found" — is recorded with the
  commit.

## Provenance rail

Prior-art shape observed in a public tutorial; the principle is one this
project already held and published. No code adopted, nothing named in any
public artifact. The closing mechanism — fail-closed capability flags with
undefined-extern backends — is **internal precedent** from `512-mind`
(`drift.mind`, `key_management.mind`), not external. Citation in
`mind-internal`.
