"""mind-mem-4b v2 training script — Runpod H200 full-fine-tune.

Consumes the JSONL corpora produced by
:mod:`generate_dispatcher_examples` and :mod:`generate_retrieval_examples`
(plus any replay corpus the operator mixes in) and fine-tunes the
existing ``star-ga/mind-mem-4b`` checkpoint on the v3.2.x dispatcher
+ v3.3.0 retrieval surfaces.

Runs on a single H200 141GB with full fine-tune (QLoRA optional for
A100-class boxes). Zero accelerate orchestration — a single-process
trainer is enough for 4B on H200.

Usage::

    python3 benchmarks/train_mind_mem_4b.py \\
        --base-model star-ga/mind-mem-4b \\
        --data /workspace/train-corpus/mixed.jsonl \\
        --output-dir /workspace/mind-mem-4b-v2 \\
        --config benchmarks/train_config.yaml

The config lives alongside this file as a sibling YAML so the
recipe doc can reference an exact path. The script is guarded behind
an ``if torch is None`` check so the test suite can still import it
on CPU-only dev boxes — the actual fine-tune needs GPU + transformers
+ peft + trl installed via ``pip install mind-mem[train]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Lazy heavy-deps import — keeps this file importable on CPU-only boxes.
# ---------------------------------------------------------------------------


def _require_training_stack() -> None:
    missing: list[str] = []
    for mod in ("torch", "transformers", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(f"Training stack not installed: missing {', '.join(missing)}. Install with: pip install 'mind-mem[train]'")


# ---------------------------------------------------------------------------
# Prompt formatting — maps each task to the chat template mind-mem-4b expects.
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are mind-mem-4b, the local LLM that powers mind-mem's "
    "retrieval and governance surfaces. Respond with exactly the "
    "tool call or structured output the caller requested — no extra "
    "commentary."
)


def _format_dispatcher_example(example: dict[str, Any]) -> dict[str, str]:
    """(prompt, expected_call) → chat template with system/user/assistant."""
    call = example["expected_call"]
    assistant = json.dumps(call, default=str, indent=None)
    return {
        "prompt": example["prompt"],
        "target": assistant,
    }


def _format_retrieval_example(example: dict[str, Any]) -> dict[str, str]:
    """(task, prompt, expected_output) → chat template."""
    out = example["expected_output"]
    assistant = json.dumps(out, default=str, indent=None)
    # Per-task prefix so the model learns which shape is expected.
    prefix = f"[task={example['task']}] "
    return {
        "prompt": prefix + example["prompt"],
        "target": assistant,
    }


def _format_example(example: dict[str, Any]) -> dict[str, str] | None:
    """Dispatch between the two generator shapes; returns None when
    the example doesn't match either (operator-mixed-in custom data
    passes through via the ``prompt`` + ``target`` shape)."""
    if "expected_call" in example:
        return _format_dispatcher_example(example)
    if "task" in example and "expected_output" in example:
        return _format_retrieval_example(example)
    if "prompt" in example and "target" in example:
        return {"prompt": str(example["prompt"]), "target": str(example["target"])}
    return None


def iter_examples(data_path: str) -> Iterator[dict[str, str]]:
    """Yield ``{prompt, target}`` pairs from a JSONL corpus."""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            formatted = _format_example(raw)
            if formatted is not None:
                yield formatted


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    base_model: str = "star-ga/mind-mem-4b"
    output_dir: str = "/workspace/mind-mem-4b-v2"
    dtype: str = "bfloat16"
    optim: str = "adamw_torch_fused"
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 4096
    packing: bool = True
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 25
    report_to: str = "none"
    seed: int = 42
    quantize: str | None = None  # "4bit" enables QLoRA for A100-class boxes
    gradient_checkpointing: bool = False
    save_total_limit: int = 2  # cap saved checkpoints to avoid disk overflow
    extra_config: dict[str, Any] = field(default_factory=dict)


def load_train_config(path: str | None) -> TrainConfig:
    """Load YAML config; fallback to TrainConfig defaults when path is None."""
    cfg = TrainConfig()
    if not path:
        return cfg
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit("PyYAML required to parse --config. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected top-level mapping, got {type(data).__name__}")
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            cfg.extra_config[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Training entry point (lazy imports guarded)
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    _require_training_stack()
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    cfg = load_train_config(args.config)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[train] base_model={cfg.base_model}")
    print(f"[train] output_dir={cfg.output_dir}")
    print(f"[train] data={args.data}")

    examples = list(iter_examples(args.data))
    if not examples:
        print(f"[train] ERROR: no training examples in {args.data}", file=sys.stderr)
        return 1
    print(f"[train] Loaded {len(examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(batch: dict[str, list]) -> dict[str, list]:
        texts: list[str] = []
        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Use chat template when the tokenizer has one; otherwise
            # fall back to a minimal Llama-style format.
            if getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
            else:
                texts.append(f"{_SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant: {target}")
        return {"text": texts}

    ds = Dataset.from_list(examples).map(_format, batched=True, remove_columns=["prompt", "target"])

    def _tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    ds = ds.map(_tokenize, batched=True, remove_columns=["text"])

    torch_dtype = getattr(torch, cfg.dtype, torch.bfloat16)
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if cfg.quantize == "4bit":
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        except ImportError:
            print("[train] WARN: bitsandbytes not installed — ignoring quantize=4bit")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        seed=cfg.seed,
        optim=cfg.optim,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # DataCollatorForLanguageModeling(mlm=False) copies input_ids → labels
    # and masks pad tokens to -100 so the model returns a causal-LM loss.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # transformers >=4.46 renamed tokenizer= to processing_class=.
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": ds,
        "data_collator": data_collator,
    }
    import inspect

    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    # Auto-resume from latest checkpoint if output_dir already has one.
    # Lets us survive SSH hangups and transient pod hiccups without
    # losing previous training progress.
    resume_from: str | bool = False
    out = Path(cfg.output_dir)
    if out.is_dir():
        checkpoints = sorted(
            (p for p in out.iterdir() if p.name.startswith("checkpoint-")),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"[train] resuming from {resume_from}")

    print("[train] starting fit …")
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train] saved v2 checkpoint to {cfg.output_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL corpus (dispatcher + retrieval + replay)")
    parser.add_argument("--config", help="YAML training config override")
    parser.add_argument("--base-model", help="HF repo or local path of base model")
    parser.add_argument("--output-dir", help="Where to save v2 checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Parse + tokenize only; skip .train()")
    args = parser.parse_args()

    if args.dry_run:
        # Run everything but .train() so operators can smoke the pipeline
        # without burning GPU hours.
        _require_training_stack()
        examples = list(iter_examples(args.data))
        print(f"[dry-run] Loaded {len(examples)} formatted examples.")
        if examples:
            sample = examples[0]
            print(f"[dry-run] First example prompt: {sample['prompt'][:140]!r}")
            print(f"[dry-run] First example target: {sample['target'][:140]!r}")
        return 0

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
