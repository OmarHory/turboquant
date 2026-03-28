"""
Needle-In-A-Haystack evaluation for TurboQuant KV cache quantization.

Reproduces the retrieval test from Section 5 of Zandieh et al., "TurboQuant:
Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026).

A known fact (the "needle") is inserted at a controlled depth within a long
filler context (the "haystack").  The model is then asked to recall the fact.
Retrieval accuracy is reported per (context_length, bit_width, depth).

Usage:
    python -m benchmarks.eval_needle --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.core import TurboQuantMSE, TurboQuantConfig
from turboquant.cache import TurboQuantCache

NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich "
    "and sit in Dolores Park on a sunny day."
)

RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"

RETRIEVAL_KEYWORDS = ["sandwich", "dolores park"]

FILLER_PARAGRAPHS = [
    (
        "The economic implications of technological advancement have been a subject of "
        "intense debate among scholars and policymakers for decades. As automation continues "
        "to reshape industries, workers find themselves needing to adapt to an ever-changing "
        "landscape of skill requirements. The transition from manufacturing economies to "
        "knowledge-based economies has created both tremendous opportunities and significant "
        "challenges for communities around the world. Educational institutions struggle to "
        "keep pace with the rapid evolution of workplace demands, leading to a persistent "
        "gap between the skills employers seek and those that graduates possess."
    ),
    (
        "Urban planning in the twenty-first century faces unprecedented challenges as cities "
        "grow at remarkable rates. Infrastructure built decades ago strains under the weight "
        "of populations that have doubled or tripled since those systems were designed. "
        "Transportation networks, water treatment facilities, and electrical grids all "
        "require massive investment to meet current and future demands. The tension between "
        "preserving historical character and accommodating modern needs creates complex "
        "trade-offs that city planners must navigate with limited budgets and competing "
        "political pressures from diverse stakeholder groups."
    ),
    (
        "The history of scientific discovery is filled with examples of serendipitous "
        "breakthroughs that emerged from unexpected directions. Researchers pursuing one "
        "line of inquiry have often stumbled upon findings that revolutionized entirely "
        "different fields. Penicillin, X-rays, and the microwave oven all resulted from "
        "accidental observations by scientists and engineers who had the curiosity to "
        "investigate anomalies rather than dismiss them. This pattern suggests that "
        "maintaining broad scientific funding across disciplines is essential for continued "
        "innovation, even when immediate practical applications are not apparent."
    ),
    (
        "Environmental conservation efforts have evolved significantly over the past "
        "century, moving from simple preservation of wilderness areas to comprehensive "
        "ecosystem management strategies. Modern conservationists recognize that protecting "
        "individual species requires understanding the complex web of interactions that "
        "sustain entire habitats. Climate change has added urgency to these efforts, as "
        "shifting temperature and precipitation patterns threaten to disrupt ecosystems "
        "faster than species can adapt. International cooperation has become essential, "
        "as migratory species and atmospheric processes respect no national boundaries."
    ),
    (
        "The philosophy of education has undergone substantial transformation since the "
        "Enlightenment era. Early educational theorists emphasized rote memorization and "
        "strict discipline as the foundations of learning. Progressive educators later "
        "championed experiential learning and critical thinking as more effective approaches "
        "to developing capable citizens. Today, debates continue about the proper balance "
        "between standardized curricula and individualized instruction, between technological "
        "tools and traditional methods, and between preparing students for economic "
        "productivity and fostering their personal development and civic engagement."
    ),
    (
        "Agricultural practices around the world reflect centuries of accumulated knowledge "
        "about local soil conditions, climate patterns, and crop varieties. Traditional "
        "farming methods developed through trial and error often prove remarkably well-suited "
        "to their specific environments. Modern agricultural science has introduced "
        "techniques that dramatically increase yields but sometimes at the cost of soil "
        "health and biodiversity. The challenge facing contemporary agriculture is to "
        "combine the productivity gains of modern methods with the sustainability insights "
        "of traditional practices to feed a growing global population."
    ),
    (
        "Maritime exploration played a pivotal role in connecting distant civilizations and "
        "reshaping the political geography of the world. The development of navigation "
        "technologies, from the magnetic compass to the chronometer, enabled sailors to "
        "venture farther from shore with increasing confidence. Trade routes established "
        "during the age of exploration created economic connections that persist to this day. "
        "The cultural exchanges that accompanied maritime trade introduced new foods, "
        "languages, and ideas to societies on every continent, fundamentally altering the "
        "course of human history in ways both beneficial and devastating."
    ),
    (
        "The study of linguistics reveals fascinating patterns in how languages evolve "
        "and influence one another over time. Language families branch and diversify as "
        "populations migrate and communities become isolated from one another. Contact "
        "between speakers of different languages produces pidgins, creoles, and borrowed "
        "vocabulary that enrich the expressive capacity of all languages involved. The "
        "rapid pace of globalization in recent decades has accelerated both language "
        "contact and language loss, raising important questions about the preservation "
        "of linguistic diversity and the cultural knowledge embedded within each tongue."
    ),
]


@dataclass
class BitWidthConfig:
    label: str
    base_bits: int
    num_outlier_channels: int
    outlier_bits: int

    @property
    def effective_bits(self) -> float:
        if self.num_outlier_channels == 0:
            return float(self.base_bits)
        return (96 * self.base_bits + 32 * self.outlier_bits) / 128


def get_bit_width_presets(head_dim: int) -> dict[str, BitWidthConfig]:
    outlier_ch = 32 if head_dim == 128 else (16 if head_dim == 64 else head_dim // 4)
    return {
        "baseline": BitWidthConfig("baseline (no quantization)", 0, 0, 0),
        "2.5": BitWidthConfig(f"2.5-bit (2b + {outlier_ch}ch outlier@3b)", 2, outlier_ch, 3),
        "3":   BitWidthConfig("3-bit (no outliers)", 3, 0, 0),
        "3.5": BitWidthConfig(f"3.5-bit (3b + {outlier_ch}ch outlier@4b)", 3, outlier_ch, 4),
        "4":   BitWidthConfig("4-bit (no outliers)", 4, 0, 0),
    }

BIT_WIDTH_PRESETS: dict[str, BitWidthConfig] = get_bit_width_presets(128)


@dataclass
class NeedleResult:
    context_length: int
    bit_width: str
    depth_pct: int
    retrieved: bool
    generated_text: str
    elapsed_s: float


def build_haystack(tokenizer, target_token_count: int) -> str:
    """Repeat filler paragraphs until we exceed the target token count."""
    paragraphs = []
    token_count = 0
    idx = 0
    while token_count < target_token_count:
        p = FILLER_PARAGRAPHS[idx % len(FILLER_PARAGRAPHS)]
        paragraphs.append(p)
        token_count += len(tokenizer.encode(p, add_special_tokens=False))
        idx += 1
    return "\n\n".join(paragraphs)


def insert_needle(haystack: str, needle: str, depth_pct: int) -> str:
    """Insert needle at the given depth (0 = very beginning, 100 = very end)."""
    paragraphs = haystack.split("\n\n")
    if len(paragraphs) <= 1:
        return needle + "\n\n" + haystack

    insert_idx = max(0, int(len(paragraphs) * depth_pct / 100))
    insert_idx = min(insert_idx, len(paragraphs))
    paragraphs.insert(insert_idx, needle)
    return "\n\n".join(paragraphs)


def check_retrieval(text: str) -> bool:
    text_lower = text.lower()
    return all(kw in text_lower for kw in RETRIEVAL_KEYWORDS)


def make_cache(model_config, bw_cfg: BitWidthConfig, device: torch.device):
    if bw_cfg.base_bits == 0:
        return None
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_layers = model_config.num_hidden_layers
    return TurboQuantCache(
        head_dim=head_dim,
        bit_width=bw_cfg.base_bits,
        num_layers=num_layers,
        num_outlier_channels=bw_cfg.num_outlier_channels,
        outlier_bits=bw_cfg.outlier_bits,
        device=device,
    )


def run_single(
    model,
    tokenizer,
    context: str,
    question: str,
    cache,
    device: torch.device,
    max_new_tokens: int = 64,
) -> tuple[str, float]:
    prompt = f"{context}\n\nBased on the text above, answer the following question:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache
        output = model.generate(**gen_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, elapsed


def run_evaluation(
    model_name: str,
    bit_width_keys: list[str],
    context_lengths: list[int],
    depths: list[int],
    output_path: str,
    hf_token: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    print(f"Model ready: {num_layers} layers, head_dim={head_dim}")

    presets = get_bit_width_presets(head_dim)
    bw_configs = {k: presets[k] for k in bit_width_keys if k in presets}
    if not bw_configs:
        print(f"No valid bit-width configs found in {bit_width_keys}")
        sys.exit(1)

    results: list[NeedleResult] = []
    total_runs = len(context_lengths) * len(bw_configs) * len(depths)
    run_idx = 0

    for ctx_len in context_lengths:
        haystack = build_haystack(tokenizer, ctx_len)
        haystack_tokens = len(tokenizer.encode(haystack, add_special_tokens=False))
        print(f"\nContext length target: {ctx_len} tokens (actual haystack: {haystack_tokens})")

        for depth in depths:
            context = insert_needle(haystack, NEEDLE, depth)

            for bw_key, bw_cfg in bw_configs.items():
                run_idx += 1
                cache = make_cache(model.config, bw_cfg, device)

                print(
                    f"  [{run_idx}/{total_runs}] ctx={ctx_len}, "
                    f"depth={depth}%, bits={bw_key} ... ",
                    end="",
                    flush=True,
                )

                try:
                    gen_text, elapsed = run_single(
                        model, tokenizer, context, RETRIEVAL_QUESTION, cache, device
                    )
                    retrieved = check_retrieval(gen_text)
                except torch.cuda.OutOfMemoryError:
                    gen_text = "<OOM>"
                    elapsed = 0.0
                    retrieved = False
                    torch.cuda.empty_cache()

                result = NeedleResult(
                    context_length=ctx_len,
                    bit_width=bw_key,
                    depth_pct=depth,
                    retrieved=retrieved,
                    generated_text=gen_text.strip()[:200],
                    elapsed_s=round(elapsed, 2),
                )
                results.append(result)
                status = "PASS" if retrieved else "FAIL"
                print(f"{status} ({elapsed:.1f}s)")

                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    accuracy_map: dict[str, dict[str, list[bool]]] = {}
    for r in results:
        key = f"ctx={r.context_length}"
        accuracy_map.setdefault(key, {})
        accuracy_map[key].setdefault(r.bit_width, []).append(r.retrieved)

    output = {
        "model": model_name,
        "needle": NEEDLE,
        "question": RETRIEVAL_QUESTION,
        "context_lengths": context_lengths,
        "bit_widths": bit_width_keys,
        "depths": depths,
        "results": [
            {
                "context_length": r.context_length,
                "bit_width": r.bit_width,
                "depth_pct": r.depth_pct,
                "retrieved": r.retrieved,
                "generated_text": r.generated_text,
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
        "summary": {
            ctx_key: {
                bw: round(sum(vals) / len(vals) * 100, 1)
                for bw, vals in bw_data.items()
            }
            for ctx_key, bw_data in accuracy_map.items()
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_file}")

    print_table(results, context_lengths, bit_width_keys, depths)


def print_table(
    results: list[NeedleResult],
    context_lengths: list[int],
    bit_widths: list[str],
    depths: list[int],
):
    print(f"\n{'=' * 80}")
    print("  Needle-In-A-Haystack Retrieval Accuracy")
    print(f"{'=' * 80}\n")

    lookup: dict[tuple[int, str, int], bool] = {
        (r.context_length, r.bit_width, r.depth_pct): r.retrieved for r in results
    }

    depth_header = "".join(f"{d:>8}%" for d in depths)
    print(f"  {'Context':>8}  {'Bits':>6}  {depth_header}  {'Avg':>6}")
    print(f"  {'-' * 8}  {'-' * 6}  {'-' * (9 * len(depths))}  {'-' * 6}")

    for ctx_len in context_lengths:
        for bw in bit_widths:
            row_vals = [lookup.get((ctx_len, bw, d)) for d in depths]
            row_strs = []
            for v in row_vals:
                if v is None:
                    row_strs.append(f"{'--':>9}")
                else:
                    row_strs.append(f"{'PASS' if v else 'FAIL':>9}")

            valid = [v for v in row_vals if v is not None]
            avg_pct = f"{sum(valid) / len(valid) * 100:.0f}%" if valid else "--"

            print(f"  {ctx_len:>8}  {bw:>6}  {''.join(row_strs)}  {avg_pct:>6}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Needle-In-A-Haystack eval for TurboQuant KV cache (Section 5, ICLR 2026)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--bit-widths",
        default="baseline,2.5,3,3.5,4",
        help="Comma-separated effective bit-widths (default: baseline,2.5,3,3.5,4)",
    )
    parser.add_argument(
        "--context-lengths",
        default="4096,8192,16384",
        help="Comma-separated context lengths in tokens (default: 4096,8192,16384)",
    )
    parser.add_argument(
        "--depths",
        default="0,25,50,75,100",
        help="Comma-separated needle insertion depths in %% (default: 0,25,50,75,100)",
    )
    parser.add_argument(
        "--output",
        default="results/needle_eval.json",
        help="Output JSON path (default: results/needle_eval.json)",
    )
    args = parser.parse_args()

    bit_widths = [bw.strip() for bw in args.bit_widths.split(",")]
    context_lengths = [int(c.strip()) for c in args.context_lengths.split(",")]
    depths = [int(d.strip()) for d in args.depths.split(",")]
    hf_token = os.environ.get("HF_TOKEN")

    all_valid = list(get_bit_width_presets(128).keys())
    invalid_bws = [bw for bw in bit_widths if bw not in all_valid]
    if invalid_bws:
        print(f"Unknown bit-widths: {invalid_bws}. Available: {all_valid}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"  TurboQuant Needle-In-A-Haystack Evaluation")
    print(f"  Model:           {args.model}")
    print(f"  Bit-widths:      {bit_widths}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Depths:          {depths}")
    print(f"{'=' * 60}\n")

    run_evaluation(
        model_name=args.model,
        bit_width_keys=bit_widths,
        context_lengths=context_lengths,
        depths=depths,
        output_path=args.output,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
