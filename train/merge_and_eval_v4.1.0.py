"""Post-Kaggle: pull LoRA adapter, merge with v4.0.0-base, eval 131 probes.

Run locally after the Kaggle kernel finishes. Steps:
  1. Download adapter from star-ga/mind-mem-4b-lora-v4.1.0.
  2. Load base from local full-ft.retry2e-... (or HF v4.0.0-base) as FP16.
  3. Merge adapter into base → standard safetensors.
  4. Run eval_holdout.py (22 probes) + eval_harness.py (109 probes).
  5. If 131/131: stage for HF push + GGUF conversion.
  6. If <131: report exactly which probes regressed/missed; do NOT ship.

Lock-in: NEVER overwrite full-ft.retry2e-* (the 127/131 known-good).
Always write to a fresh dir per attempt.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LOCAL_BASE = Path("/data/checkpoints/mm-workspace/full-ft.retry2e-109of109+18of22")
DEFAULT_ADAPTER = "star-ga/mind-mem-4b-lora-v4.1.0"
SHIP_DIR_DEFAULT = Path("/data/checkpoints/mm-workspace/full-ft.v4.1.0-candidate")
TRAIN_DIR = Path(__file__).resolve().parent


def _assert_clean_ship_dir(ship_dir: Path) -> None:
    """Refuse to write into a path that already contains weights."""
    if ship_dir.exists() and any(ship_dir.iterdir()):
        sys.exit(f"REFUSE: {ship_dir} not empty — pick a fresh path. "
                 f"Will not overwrite weights.")


def _merge(adapter: str, ship_dir: Path) -> None:
    print(f"loading base from {LOCAL_BASE}")
    tok = AutoTokenizer.from_pretrained(LOCAL_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_BASE,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"attaching adapter {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    print("merging adapter into base weights")
    model = model.merge_and_unload()
    ship_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(ship_dir, safe_serialization=True)
    tok.save_pretrained(ship_dir)
    # Carry over config files that save_pretrained skips
    for fname in ("chat_template.jinja", "generation_config.json"):
        src = LOCAL_BASE / fname
        if src.is_file():
            shutil.copy2(src, ship_dir / fname)
    sz = (ship_dir / "model.safetensors").stat().st_size
    print(f"merged safetensors: {sz:,} bytes")


def _eval(ship_dir: Path) -> dict:
    """Run both eval harnesses, return combined report.

    Env wiring:
      MM_FULLFT_DIR     → model weights to evaluate
      MM_TRAIN_ROOT     → where eval_harness writes eval_report.json
      MM_HOLDOUT_REPORT → where eval_holdout writes its report
    """
    main_report_path = ship_dir / "eval_report.json"
    hold_report_path = ship_dir / "eval_holdout_report.json"
    env = os.environ.copy()
    env["MM_FULLFT_DIR"] = str(ship_dir)
    env["MM_TRAIN_ROOT"] = str(ship_dir)
    env["MM_HOLDOUT_REPORT"] = str(hold_report_path)

    results = {}
    for name, script in [
        ("main_109", TRAIN_DIR / "eval_harness.py"),
        ("holdout_22", TRAIN_DIR / "eval_holdout.py"),
    ]:
        print(f"\n=== running {name}: {script.name} ===")
        if not script.is_file():
            print(f"  WARN: {script} not found, skipping")
            continue
        r = subprocess.run(
            ["python3", str(script)],
            env=env,
            capture_output=True,
            text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            print(f"  stderr: {r.stderr[:500]}")
        results[name] = {"rc": r.returncode, "stdout": r.stdout}

    # Parse the JSON reports the harnesses write
    main_report = main_report_path
    hold_report = hold_report_path
    summary: dict = {}
    if main_report.is_file():
        d = json.loads(main_report.read_text())
        # eval_harness writes per-group dicts at top level with hits/total
        group_keys = [k for k, v in d.items()
                      if isinstance(v, dict) and "hits" in v and "total" in v]
        summary["main_total"] = sum(d[k]["total"] for k in group_keys)
        summary["main_hits"] = sum(d[k]["hits"] for k in group_keys)
    if hold_report.is_file():
        d = json.loads(hold_report.read_text())
        summary["holdout_total"] = d["total_probes"]
        summary["holdout_hits"] = d["total_hits"]
        summary["holdout_misses"] = (
            d["v4_holdout"]["misses"] + d["v312_holdout"]["misses"]
        )
    total_hits = summary.get("main_hits", 0) + summary.get("holdout_hits", 0)
    total_probes = summary.get("main_total", 0) + summary.get("holdout_total", 0)
    summary["overall"] = f"{total_hits}/{total_probes}"
    summary["perfect"] = (total_hits == total_probes == 131)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--ship-dir", type=Path, default=SHIP_DIR_DEFAULT)
    ap.add_argument("--skip-merge", action="store_true",
                    help="Skip merge (reuse existing ship-dir)")
    args = ap.parse_args()

    if not args.skip_merge:
        _assert_clean_ship_dir(args.ship_dir)
        _merge(args.adapter, args.ship_dir)

    summary = _eval(args.ship_dir)
    print("\n" + "=" * 60)
    print(f"OVERALL: {summary['overall']}")
    print(f"perfect 131/131: {summary['perfect']}")
    if summary.get("holdout_misses"):
        print(f"holdout misses ({len(summary['holdout_misses'])}):")
        for m in summary["holdout_misses"]:
            print(f"  - missing {m['missing']!r}: {m['prompt'][:80]}")
    print("=" * 60)
    if summary["perfect"]:
        print(f"\n✓ READY TO SHIP. Next: push {args.ship_dir} to HF main + GGUF + Ollama.")
    else:
        print("\n✗ NOT SHIPPING. Iterate corpus, retrain.")
        sys.exit(1)


if __name__ == "__main__":
    main()
