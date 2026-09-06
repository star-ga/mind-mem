"""Stratified LongMemEval probe.

The earlier probe took the first N questions of the file. LongMemEval-S is
ORDERED BY QUESTION TYPE, so that sampled one category (single-session-user)
and never touched temporal-reasoning or multi-session -- together 53% of the
benchmark, and exactly where the date fixes and the diversity cap should show.
This samples evenly across every type and reports recall_any@5 AND
recall_all@5, since a multi-session question needs more than one gold session
and any@5 cannot see that at all.
"""

import argparse
import collections
import json
import os
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/n/mind-mem")
ap.add_argument("--per-type", type=int, default=12)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.root, "src"))
sys.path.insert(0, a.root)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from benchmarks._ch_minilm_spawn import build_ws  # noqa: E402
from mind_mem.sqlite_index import build_index, query_index  # noqa: E402

DATA = "/home/n/mind-mem/benchmarks/.cache/longmemeval_s.json"
allq = json.load(open(DATA, encoding="utf-8"))

by_type = collections.defaultdict(list)
for q in allq:
    if q.get("answer_session_ids"):
        by_type[q.get("question_type", "?")].append(q)

sample = []
for t in sorted(by_type):
    sample.extend(by_type[t][: a.per_type])
print("sampled %d questions across %d types" % (len(sample), len(by_type)), flush=True)

rows = []
for i, q in enumerate(sample, 1):
    gold = set(str(s) for s in q["answer_session_ids"])
    tmp = tempfile.mkdtemp()
    try:
        ws = build_ws(q, tmp, "all")
        build_index(ws, incremental=False)
        res = query_index(ws, q["question"], limit=5)
        got = []
        for r in res[:5]:
            rid = str(r.get("_id", ""))
            got.append(rid.split("__")[0].replace("SESSION-", "").split("::F")[0])
        gots = set(got)
        rows.append(
            {
                "qid": q.get("question_id"),
                "qtype": q.get("question_type"),
                "n_gold": len(gold),
                "any": bool(gold & gots),
                "all": gold.issubset(gots),
                "distinct": len(gots),
                "top5": got,
            }
        )
    except Exception as e:  # noqa: BLE001 -- a failed question is data
        rows.append(
            {
                "qid": q.get("question_id"),
                "qtype": q.get("question_type"),
                "n_gold": len(gold),
                "any": False,
                "all": False,
                "distinct": 0,
                "error": ("%s: %s" % (type(e).__name__, e))[:200],
            }
        )
    if i % 12 == 0:
        print("  [%d/%d]" % (i, len(sample)), flush=True)

with open(a.out, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, sort_keys=True) + "\n")

n = len(rows)
print(
    "OVERALL any@5 %d/%d = %.1f%%   all@5 %d/%d = %.1f%%   distinct/query %.2f"
    % (
        sum(r["any"] for r in rows),
        n,
        100 * sum(r["any"] for r in rows) / n,
        sum(r["all"] for r in rows),
        n,
        100 * sum(r["all"] for r in rows) / n,
        sum(r["distinct"] for r in rows) / n,
    )
)
for t in sorted(set(r["qtype"] for r in rows)):
    sub = [r for r in rows if r["qtype"] == t]
    print("  %-28s any %2d/%-2d  all %2d/%-2d" % (t, sum(r["any"] for r in sub), len(sub), sum(r["all"] for r in sub), len(sub)))
