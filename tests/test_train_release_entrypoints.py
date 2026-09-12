"""Default-path integration for the staged training release entrypoints.

The test executes copied entrypoint modules in a subprocess so a passing test
cannot accidentally reuse the live checkout's module globals or an imported
model.  The checkpoint and corpus are synthetic; inference is the only mocked
boundary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_DRIVER = r"""
import json
import os
from pathlib import Path

train_root = Path(os.environ["MM_TRAIN_ROOT"]).resolve()
for name in ("MM_FULLFT_DIR", "MM_WEIGHTS_DIR", "MM_HOLDOUT_REPORT", "MM_CORPUS"):
    assert name not in os.environ, name

# Import only after the release root has been supplied.  The caller placed
# this driver inside a copied checkout and gives it that checkout on PATH.
import train.eval_harness as H
import eval_harness as Hflat
import train.eval_holdout as HO
import train.upload_to_hf as U
import train.build_model_card as B

checkout = Path(H.__file__).resolve().parents[1]
assert checkout != Path(os.environ["MM_LIVE_REPO"]).resolve()
assert str(checkout) in str(H.__file__)
assert H._BASE_DIR == train_root
assert HO.REPORT == train_root.parent / "full-ft" / "eval_holdout_report.json"
assert HO.CORPUS == train_root / "corpus.jsonl"
assert U.OUT_DIR == train_root
assert U.WEIGHTS_DIR == train_root / "full-ft"
assert B.OUT == train_root / "README.md"
assert B.EVAL_REPORT == train_root / "eval_report.json"


def fake_load(selection=None):
    return object(), object(), selection


def fake_bench(_tokenizer, _model, group, probes):
    items = []
    for index, probe in enumerate(probes):
        required = probe[1]
        response = required if isinstance(required, str) else " ".join(required)
        items.append({
            "index": index,
            "prompt": probe[0],
            "response": response,
            "passed": True,
        })
    return {
        "accuracy": 1.0,
        "hits": len(items),
        "total": len(items),
        "misses": [],
        "items": items,
    }


# The holdout module imports the legacy flat eval_harness name, while the
# package entrypoint uses train.eval_harness.  Patch both inference seams.
H._load_model = fake_load
HO._load_model = fake_load
H._bench_probes = fake_bench
Hflat._bench_probes = fake_bench


def invoke(label, entrypoint):
    try:
        entrypoint()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise AssertionError(f"{label} exited {exc.code!r}") from exc
    print("stage=" + label)


assert not HO.REPORT.parent.exists()
invoke("main-eval", H.main)
assert (train_root / "eval_report.json").is_file()
invoke("holdout-eval", HO.main)
assert HO.REPORT.parent.is_dir()
assert HO.REPORT.is_file()

reports = U._require_eval_receipts()
assert reports["main"]["receipt"]["complete"] is True
assert reports["holdout"]["receipt"]["complete"] is True
assert set(("overall_accuracy", "total_hits", "total_probes")) <= set(
    json.loads(HO.REPORT.read_text(encoding="utf-8"))
)
print("stage=publication-gate")

invoke("model-card", B.main)
card_path = train_root / "README.md"
card = card_path.read_text(encoding="utf-8")
assert card == B.render_release_card(reports)
for aggregate in ("overall_accuracy", "total_hits", "total_probes"):
    assert aggregate not in card
assert card_path.is_file()
print("stage=assertions")
print(json.dumps({
    "checkout": str(checkout),
    "train_root": str(train_root),
    "main_report": str(train_root / "eval_report.json"),
    "holdout_report": str(HO.REPORT),
    "model_card": str(card_path),
    "main_complete": reports["main"]["receipt"]["complete"],
    "holdout_complete": reports["holdout"]["receipt"]["complete"],
}, sort_keys=True))
"""


def test_default_release_entrypoints_run_in_order_without_live_imports(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("train", "src", "scripts"):
        shutil.copytree(REPO / name, checkout / name, symlinks=True)

    driver = checkout / "release_entrypoint_driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "release-test"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "release-test@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "train", "src", "scripts"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "release entrypoint fixture"], cwd=checkout, check=True)

    train_root = checkout / "train-output"
    fullft = train_root / "full-ft"
    fullft.mkdir(parents=True)
    (fullft / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (fullft / "model.safetensors").write_bytes(b"synthetic full-ft weights")
    (fullft / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (fullft / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (train_root / "corpus.jsonl").write_text("{}\n", encoding="utf-8")

    child_env = os.environ.copy()
    for name in ("MM_FULLFT_DIR", "MM_WEIGHTS_DIR", "MM_HOLDOUT_REPORT", "MM_CORPUS"):
        child_env.pop(name, None)
    child_env["MM_TRAIN_ROOT"] = str(train_root)
    child_env["MM_LIVE_REPO"] = str(REPO)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONPATH"] = os.pathsep.join((str(checkout), str(checkout / "train")))
    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=checkout,
        env=child_env,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for stage in ("main-eval", "holdout-eval", "publication-gate", "model-card", "assertions"):
        assert "stage=" + stage in result.stdout, output
    assert "Traceback" not in output, output
