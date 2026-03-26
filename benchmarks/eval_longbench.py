"""
LongBench-E evaluation for TurboQuant (Section 5, ICLR 2026).

Evaluates quantized KV cache quality on long-context tasks across six
categories: single-doc QA, multi-doc QA, summarization, few-shot learning,
synthetic tasks, and code completion.

Usage:
    python -m benchmarks.eval_longbench --model meta-llama/Llama-3.1-8B-Instruct --max-samples 20
"""

import argparse
import gc
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.cache import TurboQuantCache

TASK_CATEGORIES = {
    "Single-Doc QA": ["qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa"],
    "Summarization": ["gov_report", "multi_news"],
    "Few-shot": ["trec", "triviaqa"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

ALL_TASKS = [task for tasks in TASK_CATEGORIES.values() for task in tasks]

METRIC_BY_TASK = {
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "gov_report": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "accuracy",
    "triviaqa": "f1",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc": "prefix_match",
    "repobench-p": "prefix_match",
}

GENERATION_TOKENS = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 16,
    "triviaqa": 32,
    "passage_count": 16,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(normalize_text(prediction) == normalize_text(reference))
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_prefix_match(prediction: str, reference: str) -> float:
    pred_lines = prediction.strip().splitlines()
    ref_lines = reference.strip().splitlines()
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    match_count = 0
    for p, r in zip(pred_lines, ref_lines):
        if p.strip() == r.strip():
            match_count += 1
        else:
            break
    return match_count / len(ref_lines)


METRIC_FN = {
    "f1": compute_f1,
    "rouge_l": compute_rouge_l,
    "accuracy": compute_accuracy,
    "prefix_match": compute_prefix_match,
}


def score_prediction(task: str, prediction: str, references: list[str]) -> float:
    metric_name = METRIC_BY_TASK[task]
    fn = METRIC_FN[metric_name]
    return max(fn(prediction, ref) for ref in references)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_longbench_task(task: str, max_samples: int) -> list[dict]:
    ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    examples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            answer = row.get("answer", "")
            answers = [answer] if answer else [""]
        examples.append({
            "input": row["input"],
            "context": row["context"],
            "answers": answers,
            "length": row.get("length", 0),
        })
    return examples


def build_prompt(task: str, example: dict) -> str:
    context = example["context"]
    question = example["input"]

    if task in ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"):
        return (
            f"Read the following text and answer the question.\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    if task in ("gov_report", "multi_news"):
        return f"Summarize the following text.\n\nText: {context}\n\nSummary:"
    if task in ("trec",):
        return (
            f"{context}\n\n"
            f"Classify the following question: {question}\n\n"
            f"Answer:"
        )
    if task in ("triviaqa",):
        return (
            f"Answer the question based on the given context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    if task in ("passage_count", "passage_retrieval_en"):
        return f"{context}\n\n{question}\n\nAnswer:"
    if task in ("lcc", "repobench-p"):
        return f"{context}\n\n{question}"
    return f"{context}\n\n{question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def truncate_to_max_length(
    input_ids: torch.Tensor,
    tokenizer,
    max_ctx: int,
    gen_budget: int,
) -> torch.Tensor:
    allowed = max_ctx - gen_budget
    if input_ids.shape[1] > allowed:
        input_ids = input_ids[:, :allowed]
    return input_ids


@torch.no_grad()
def generate_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_ctx: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = truncate_to_max_length(
        inputs["input_ids"].to(model.device), tokenizer, max_ctx, max_new_tokens,
    )
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_quantized(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_ctx: int,
    bit_width: int,
) -> str:
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    num_outlier = 0
    outlier_bits = 0
    if bit_width == 3:
        num_outlier = min(8, head_dim // 4)
        outlier_bits = 4

    cache = TurboQuantCache(
        head_dim=head_dim,
        bit_width=bit_width,
        num_layers=num_layers,
        num_outlier_channels=num_outlier,
        outlier_bits=outlier_bits,
        device=model.device,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = truncate_to_max_length(
        inputs["input_ids"].to(model.device), tokenizer, max_ctx, max_new_tokens,
    )
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        past_key_values=cache,
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    model,
    tokenizer,
    task: str,
    examples: list[dict],
    bit_widths: list,
    max_ctx: int,
) -> dict:
    max_gen = GENERATION_TOKENS.get(task, 64)
    results = {bw: [] for bw in bit_widths}

    for idx, ex in enumerate(examples):
        prompt = build_prompt(task, ex)
        for bw in bit_widths:
            if bw == "baseline":
                pred = generate_baseline(model, tokenizer, prompt, max_gen, max_ctx)
            else:
                pred = generate_quantized(model, tokenizer, prompt, max_gen, max_ctx, bw)

            score = score_prediction(task, pred, ex["answers"])
            results[bw].append(score)
            gc.collect()

        sys.stdout.write(f"\r  {task}: {idx + 1}/{len(examples)}")
        sys.stdout.flush()

    print()
    return {
        bw: sum(scores) / len(scores) if scores else 0.0
        for bw, scores in results.items()
    }


def format_table(all_results: dict, bit_widths: list) -> str:
    lines = []
    bw_labels = [str(bw) + "-bit" if bw != "baseline" else "baseline" for bw in bit_widths]
    header = f"{'Task':<30} | " + " | ".join(f"{lbl:>10}" for lbl in bw_labels)
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    cat_avgs = {bw: [] for bw in bit_widths}

    for category, tasks in TASK_CATEGORIES.items():
        lines.append(f"  [{category}]")
        for task in tasks:
            if task not in all_results:
                continue
            metric = METRIC_BY_TASK[task]
            row_label = f"    {task} ({metric})"
            vals = []
            for bw in bit_widths:
                v = all_results[task].get(bw, 0.0) * 100
                vals.append(f"{v:>10.1f}")
                cat_avgs[bw].append(v)
            lines.append(f"{row_label:<30} | " + " | ".join(vals))
        lines.append("")

    lines.append(sep)
    avg_vals = []
    for bw in bit_widths:
        scores = cat_avgs[bw]
        avg = sum(scores) / len(scores) if scores else 0.0
        avg_vals.append(f"{avg:>10.1f}")
    lines.append(f"{'Overall Average':<30} | " + " | ".join(avg_vals))
    lines.append(sep)

    return "\n".join(lines)


def parse_bit_widths(s: str) -> list:
    result = []
    for part in s.split(","):
        part = part.strip()
        if part == "baseline":
            result.append("baseline")
        else:
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description="LongBench-E evaluation for TurboQuant")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--bit-widths", default="3,4,baseline")
    parser.add_argument("--output", default="results/longbench_results.json")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset to run")
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda, cpu)")
    args = parser.parse_args()

    bit_widths = parse_bit_widths(args.bit_widths)
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS

    print("=" * 70)
    print("  LongBench-E Evaluation — TurboQuant (Section 5, ICLR 2026)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Bit widths:  {bit_widths}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Tasks:       {len(tasks)}")
    print()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_ctx = getattr(model.config, "max_position_embeddings", 4096)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  max_position_embeddings={max_ctx}, head_dim={head_dim}")
    print(f"  layers={model.config.num_hidden_layers}, "
          f"kv_heads={getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)}")
    print()

    all_results = {}
    start_time = time.time()

    for task in tasks:
        if task not in ALL_TASKS:
            print(f"  Skipping unknown task: {task}")
            continue

        print(f"Loading {task}...")
        examples = load_longbench_task(task, args.max_samples)
        print(f"  {len(examples)} examples loaded")

        task_results = evaluate_task(model, tokenizer, task, examples, bit_widths, max_ctx)
        all_results[task] = task_results

        for bw in bit_widths:
            label = f"{bw}-bit" if bw != "baseline" else "baseline"
            print(f"    {label}: {task_results[bw] * 100:.1f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"  Results (elapsed: {elapsed / 60:.1f} min)")
    print(f"{'=' * 70}\n")
    print(format_table(all_results, bit_widths))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for task, bw_scores in all_results.items():
        serializable[task] = {str(bw): score for bw, score in bw_scores.items()}

    output_data = {
        "model": args.model,
        "bit_widths": [str(bw) for bw in bit_widths],
        "max_samples": args.max_samples,
        "elapsed_seconds": elapsed,
        "results": serializable,
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
