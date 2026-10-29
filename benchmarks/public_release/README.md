# Reproducible historical benchmark kit

**Verification pending. These are saved historical results, not validated release scores or a 100% claim.**

No dataset files, dataset questions, or rendered prompts are distributed here.
The only source evidence quotations are those attached to the 96 independently
selected key changes. Dataset downloads remain subject to their upstream licenses.

From the repository root, prepare the pinned datasets once, then replay offline:

```bash
python3 benchmarks/repro_release.py locomo --prepare
python3 benchmarks/repro_release.py locomo --rescore
python3 benchmarks/repro_release.py longmemeval_s --rescore
python3 benchmarks/repro_release.py niah --rescore
```

Preparation downloads LoCoMo and cleaned LongMemEval-S using immutable URLs and
SHA-256 from `protocol.json`. The cache stays untracked. Both preparation and
replay reconstruct all 2486 historical prompts from downloaded source blocks,
saved retrieval IDs and frozen rendering arithmetic, and verify their published
SHA-256 hashes. This reconstructs saved retrieval selections, not a new retrieval
measurement. The corrected dataset is built in memory by applying exactly 96
answer changes; literal quotations and unchanged content are checked against
the downloaded original. Replay requires the prepared cache and no network,
answer model, judge API, or GPU.

| Categories 1–4, saved accuracy | Full | Dev | Held-out |
|---|---:|---:|---:|
| Official key | 1002/1540 | 531/802 | 471/738 |
| Corrected key, verification pending | 1003/1540 | 536/802 | 467/738 |

Complete full/dev/held-out accuracy, exact match and token F1 are in
`expected_scores.json`; replay compares them exactly. All 1986 questions remain
in the LoCoMo population, with category 5 reported separately. LongMemEval-S
replays 216/500 (dev 105/257, held-out 111/243). NIAH replays 250/250 retrieval
cases, all previously exposed, with no untouched held-out cases.

The saved category-5 answers were produced with the dataset category label in
the prompt. Scores containing these answers are not fair label-free release
measurements. Earlier development inspected both halves before the frozen
split existed. The split limits future tuning; it does not erase exposure.
No untouched benchmark has been run. The reference-answer configuration has
not been measured. Historical answer and judge revisions and exact library
versions were not recorded; fresh inference is not claimed to reproduce them.
The fresh runner refuses unresolved model pins and does not pass category
labels across its answerer boundary.

The corrected score replays historical answer-bound adjudications; it does not
independently rejudge them. Independent selection of corrections is not their
verification. Corrected-key verification remains **verification pending**;
new corrected-score claims require grading with a pinned judge. The held-out
decrease is disclosed. Excluded candidates and unchanged audit rows are not
included. Historical mixed-selection and older-baseline replay modes are not
part of this kit.

## Artifacts

- `corrections.jsonl`: exactly 96 changes, official/corrected answers, accepted
  alternatives, short evidence quotes with dialogue IDs, and question hashes.
- `correction_evidence.jsonl`: the same 96 evidence records and their hashes.
- `answers.jsonl`: our answers, boolean votes and verdicts, prompt hashes and
  split metadata keyed by question ID; no questions or official reference text.
- `adjudications.jsonl`: only adjudications associated with those 96 changes,
  bound to answer and revised-key hashes; no review narratives.
- `prompt_hashes.jsonl` and `prompt_sources.jsonl`: prompt hashes and source IDs,
  lengths and hashes needed for exact reconstruction; no source text.
- `HELDOUT_SPLIT_20261029.json`: unchanged frozen split.
- `judge_prompts.json`: historical judge templates.
- `protocol.json`, `manifest.json`, `expected_scores.json`: pinned inputs,
  artifact checksums, limitations and aggregate expectations.

## Licenses and attribution

LoCoMo is by the authors of *Evaluating Very Long-Term Conversational Memory of
LLM Agents*, distributed by snap-research. The evidence quotations and original
answer excerpts in the 96 change records are attributed to the
[pinned source](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)
and covered by **CC BY-NC 4.0**, not the repository code license; see
[the included license](locomo.LICENSE.txt). Corrected answers and accepted sets
are our changes and are not endorsed by the original authors. The dataset itself
is not redistributed.

LongMemEval is by Xiaowu Wang and collaborators. The
[pinned cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)
is downloaded separately; its MIT license is included as `longmemeval.LICENSE`.
NIAH is the repository's deterministic synthetic retrieval benchmark under the
repository license. None of these historical results establishes an untouched
benchmark result.
