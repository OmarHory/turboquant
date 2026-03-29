"""Generate benchmark charts from results JSON files."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
})

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

COLORS = {
    'baseline': '#6B7280',
    'tq4': '#3B82F6',
    'tq3.5': '#10B981',
    'tq3': '#F59E0B',
    'tq2.5': '#EF4444',
}
LABELS = {
    'baseline': 'Baseline FP16',
    'tq4': '4-bit',
    'tq3.5': '3.5-bit (outlier)',
    'tq3': '3-bit',
    'tq2.5': '2.5-bit (outlier)',
}


def load_results(path):
    return json.loads(Path(path).read_text())


def avg(results, config_key, field):
    items = [r for r in results.get(config_key, []) if "error" not in r]
    return sum(r[field] for r in items) / len(items) if items else 0


def _config_keys(results):
    return [k for k in ['baseline', 'tq4', 'tq3.5', 'tq3', 'tq2.5'] if k in results]


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)


def chart_kv_memory(results, model_name, filename):
    configs = _config_keys(results)
    mems = [avg(results, k, 'kv_memory') / (1024 * 1024) for k in configs]
    colors = [COLORS[k] for k in configs]
    labels = [LABELS[k] for k in configs]

    baseline_mem = mems[0]
    ratios = [f'{baseline_mem / m:.1f}x' if m > 0 else '' for m in mems]
    ratios[0] = '1.0x'

    gpu = results.get("gpu", "GPU")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, mems, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

    for bar, mem, ratio in zip(bars, mems, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mems) * 0.02,
                f'{mem:.1f} MB\n({ratio})', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('KV Cache Memory (MB)')
    ax.set_title(f'KV Cache Memory — {model_name} ({gpu})')
    ax.set_ylim(0, max(mems) * 1.3)
    _clean_axes(ax)

    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {filename}")


def chart_throughput(results, model_name, filename):
    configs = _config_keys(results)
    tps = [avg(results, k, 'tps') for k in configs]
    colors = [COLORS[k] for k in configs]
    labels = [LABELS[k] for k in configs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, tps, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

    for bar, t in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tps) * 0.02,
                f'{t:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Tokens / sec')
    ax.set_title(f'Generation Throughput — {model_name} ({results.get("gpu", "GPU")})')
    ax.set_ylim(0, max(tps) * 1.25)
    _clean_axes(ax)

    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {filename}")


def chart_compression_vs_speed(results, model_name, filename):
    configs = _config_keys(results)
    baseline_mem = avg(results, 'baseline', 'kv_memory')

    xs, ys, cs, ls = [], [], [], []
    for k in configs:
        mem = avg(results, k, 'kv_memory')
        ratio = baseline_mem / mem if mem > 0 else 1.0
        tps = avg(results, k, 'tps')
        xs.append(ratio)
        ys.append(tps)
        cs.append(COLORS[k])
        ls.append(LABELS[k])

    fig, ax = plt.subplots(figsize=(10, 7))
    for x, y, c, label in zip(xs, ys, cs, ls):
        ax.scatter(x, y, c=c, s=200, zorder=5, edgecolors='white', linewidth=2)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 8),
                    fontsize=11, fontweight='bold', color=c)

    ax.set_xlabel('Compression Ratio (vs FP16)')
    ax.set_ylabel('Generation (tokens/sec)')
    ax.set_title(f'Compression vs Speed — {model_name} ({results.get("gpu", "GPU")})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(xs) * 1.15)

    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {filename}")


def chart_attention_speedup(results, model_name, filename):
    attn = results.get("attention_speedup", {})
    if not attn:
        return

    seq_lens = sorted(attn.keys(), key=int)
    baseline_ms = [attn[s]['baseline_ms'] for s in seq_lens]
    dequant_ms = [attn[s]['dequant_then_matmul_ms'] for s in seq_lens]
    qattn_ms = [attn[s]['quantized_attn_ms'] for s in seq_lens]
    seq_labels = [f'{int(s)//1024}K' if int(s) >= 1024 else s for s in seq_lens]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(seq_labels, dequant_ms, 's-', color=COLORS['tq4'], linewidth=2.5,
             markersize=8, label='Dequant + Matmul')
    ax1.plot(seq_labels, qattn_ms, 'D-', color=COLORS['tq2.5'], linewidth=2.5,
             markersize=8, label='Quantized Attention')
    ax1.fill_between(range(len(seq_labels)), dequant_ms, qattn_ms,
                     alpha=0.12, color=COLORS['tq2.5'])
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Attention Latency (quantized paths)')
    ax1.legend(fontsize=10)
    _clean_axes(ax1)

    speedup = [attn[s]['speedup_vs_dequant'] for s in seq_lens]
    bar_colors = [COLORS['tq2.5'] if s >= 1.0 else COLORS['baseline'] for s in speedup]
    bars = ax2.bar(seq_labels, speedup, color=bar_colors, width=0.5,
                   edgecolor='white', linewidth=1.5)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for bar, s in zip(bars, speedup):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{s:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup vs Dequant+Matmul')
    ax2.set_title('Quantized Attention Speedup')
    _clean_axes(ax2)

    gpu = results.get("gpu", "GPU")
    fig.suptitle(f'{model_name} — {gpu}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {filename}")


def chart_overview(results, model_name, filename):
    configs = _config_keys(results)
    mems_mb = [avg(results, k, 'kv_memory') / (1024 * 1024) for k in configs]
    tps_vals = [avg(results, k, 'tps') for k in configs]
    colors = [COLORS[k] for k in configs]
    labels = [LABELS[k] for k in configs]
    gpu = results.get("gpu", "GPU")

    baseline_mem = mems_mb[0]
    ratios = [baseline_mem / m if m > 0 else 1.0 for m in mems_mb]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    bars1 = ax1.bar(labels, tps_vals, color=colors, width=0.6,
                    edgecolor='white', linewidth=1.5)
    for bar, t in zip(bars1, tps_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tps_vals) * 0.02,
                 f'{t:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Tokens / sec')
    ax1.set_title('Throughput')
    ax1.set_ylim(0, max(tps_vals) * 1.2)
    _clean_axes(ax1)
    ax1.tick_params(axis='x', rotation=25)

    bars2 = ax2.bar(labels, mems_mb, color=colors, width=0.6,
                    edgecolor='white', linewidth=1.5)
    for bar, mem, ratio in zip(bars2, mems_mb, ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mems_mb) * 0.02,
                 f'{mem:.1f} MB\n({ratio:.1f}x)', ha='center', va='bottom',
                 fontweight='bold', fontsize=9)
    ax2.set_ylabel('KV Cache Memory (MB)')
    ax2.set_title('KV Cache Memory')
    ax2.set_ylim(0, max(mems_mb) * 1.35)
    _clean_axes(ax2)
    ax2.tick_params(axis='x', rotation=25)

    for x, y, c, label in zip(ratios, tps_vals, colors, labels):
        ax3.scatter(x, y, c=c, s=180, zorder=5, edgecolors='white', linewidth=2)
        ax3.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6),
                     fontsize=9, fontweight='bold', color=c)
    ax3.set_xlabel('Compression Ratio (vs FP16)')
    ax3.set_ylabel('Tokens / sec')
    ax3.set_title('Compression vs Speed')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(alpha=0.3)

    fig.suptitle(f'TurboQuant — {model_name} ({gpu})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {filename}")


def chart_generation_comparison(results, model_name, filename):
    """Side-by-side generation quality comparison as a bar-style figure."""
    configs = _config_keys(results)
    labels = [LABELS[k] for k in configs]
    prompts = results.get("baseline", [])
    if not prompts:
        return

    rows = []
    for i, bl in enumerate(prompts):
        if "error" in bl:
            continue
        prompt_name = bl.get("prompt", f"Prompt {i+1}")
        texts = {}
        for k in configs:
            items = results.get(k, [])
            if i < len(items) and "error" not in items[i]:
                raw = items[i]["text"].replace('\n', ' ').strip()
                texts[k] = raw[:80]
            else:
                texts[k] = "—"
        rows.append((prompt_name, texts))

    if not rows:
        return

    n_configs = len(configs)
    row_h = 0.45
    fig_h = 1.5 + len(rows) * row_h * (n_configs + 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.95
    ax.text(0.5, y, f'Generation Quality — {model_name}',
            fontsize=14, fontweight='bold', ha='center', va='top',
            transform=ax.transAxes)
    y -= 0.06

    colors = {
        'baseline': '#1E40AF', 'tq4': '#059669', 'tq3.5': '#D97706',
        'tq3': '#DC2626', 'tq2.5': '#7C3AED',
    }
    step = 1.0 / (len(rows) * (n_configs + 1.5) + 1)

    for prompt_name, texts in rows:
        ax.text(0.02, y, prompt_name, fontsize=10, fontweight='bold',
                va='top', transform=ax.transAxes)
        y -= step * 0.5
        for k, label in zip(configs, labels):
            color = colors.get(k, '#374151')
            snippet = texts.get(k, "—")
            ax.text(0.05, y, f"{label}:", fontsize=7.5, fontweight='bold',
                    color=color, va='top', transform=ax.transAxes)
            ax.text(0.18, y, snippet, fontsize=7.5, color='#374151',
                    va='top', transform=ax.transAxes, clip_on=False)
            y -= step
        y -= step * 0.8

    plt.savefig(ASSETS / filename, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  {filename}")


# Stable asset names for README / social (avoid a40_mistral_7b_overview.png in docs)
CHART_PREFIX_BY_FILE = {
    "a40_mistral_7b.json": "mistral",
    "a40_smollm2_17b.json": "smollm",
}


def chart_needle_heatmap(data: dict, filename: str):
    summary = data.get("summary") or {}
    if not summary:
        return

    def ctx_key_sort(k: str) -> int:
        try:
            return int(k.split("=", 1)[1])
        except (IndexError, ValueError):
            return 0

    ctx_keys = sorted(summary.keys(), key=ctx_key_sort)
    bit_widths = data.get("bit_widths") or []
    if not ctx_keys or not bit_widths:
        return

    mat = np.array([[summary[ck].get(bw, 0.0) for bw in bit_widths] for ck in ctx_keys])
    ctx_labels = [k.split("=", 1)[-1] for k in ctx_keys]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(ctx_keys) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(bit_widths)), labels=bit_widths, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ctx_labels)), labels=[f"{c} tok ctx" for c in ctx_labels])
    ax.set_xlabel("KV configuration")
    ax.set_ylabel("Context length")
    model = data.get("model", "model")
    ax.set_title(f"Needle retrieval accuracy (% correct) — {model}", fontsize=14, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, label="Accuracy %")
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


def chart_longbench_heatmap(data: dict, filename: str):
    raw = data.get("results") or {}
    bit_widths = data.get("bit_widths") or []
    if not raw or not bit_widths:
        return

    tasks = sorted(raw.keys())
    mat = np.array([[raw[t].get(bw, 0.0) for bw in bit_widths] for t in tasks])

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(bit_widths)), max(6, 0.22 * len(tasks) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(bit_widths)), labels=bit_widths, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(tasks)), labels=tasks, fontsize=9)
    ax.set_xlabel("KV configuration")
    ax.set_ylabel("LongBench-E task")
    model = data.get("model", "model")
    ax.set_title(f"LongBench-E scores — {model}", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Score (task metric)")
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


def chart_longbench_mean_bars(data: dict, filename: str):
    raw = data.get("results") or {}
    bit_widths = data.get("bit_widths") or []
    if not raw or not bit_widths:
        return

    means = []
    for bw in bit_widths:
        vals = [raw[t][bw] for t in raw if bw in raw[t]]
        means.append(sum(vals) / len(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(bit_widths)))
    bars = ax.bar(bit_widths, means, color=cmap, width=0.55, edgecolor="white", linewidth=1.2)
    ymax = max(means) if means else 1.0
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{m * 100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_ylim(0, min(1.05, ymax * 1.25 + 0.05) if ymax > 0 else 1)
    ax.set_ylabel("Mean score (across tasks)")
    ax.set_xlabel("KV configuration")
    model = data.get("model", "model")
    ax.set_title(f"LongBench-E — mean across tasks — {model}", fontsize=14, fontweight="bold")
    _clean_axes(ax)
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"

    for json_file in sorted(results_dir.glob("a40_*.json")):
        data = load_results(json_file)
        model_short = data.get("model", "").split("/")[-1]
        prefix = CHART_PREFIX_BY_FILE.get(json_file.name, json_file.stem.split("a40_")[-1])

        print(f"Generating charts for {model_short}...")
        chart_kv_memory(data, model_short, f"{prefix}_kv_memory.png")
        chart_throughput(data, model_short, f"{prefix}_throughput.png")
        chart_compression_vs_speed(data, model_short, f"{prefix}_compression_vs_speed.png")
        chart_attention_speedup(data, model_short, f"{prefix}_attention_speedup.png")
        chart_overview(data, model_short, f"{prefix}_overview.png")
        chart_generation_comparison(data, model_short, f"{prefix}_generation.png")
        print()

    for a100_file in sorted(results_dir.glob("a100_*.json")):
        data = load_results(a100_file)
        model_short = data.get("model", "").split("/")[-1]
        prefix = a100_file.stem

        print(f"Generating A100 charts for {model_short}...")
        chart_overview(data, model_short, f"{prefix}_overview.png")
        chart_kv_memory(data, model_short, f"{prefix}_kv_memory.png")
        chart_attention_speedup(data, model_short, f"{prefix}_attention_speedup.png")
        print()

    for needle_name in ["needle_mistral_7b.json", "needle_eval.json"]:
        needle_path = results_dir / needle_name
        if needle_path.exists():
            print(f"Needle eval charts ({needle_name})...")
            chart_needle_heatmap(load_results(needle_path), "needle_heatmap.png")
            print()
            break
    else:
        print("(skip) No needle results found\n")

    for lb_name in ["longbench_mistral_7b.json", "longbench_results.json"]:
        lb_path = results_dir / lb_name
        if lb_path.exists():
            print(f"LongBench-E charts ({lb_name})...")
            lb = load_results(lb_path)
            chart_longbench_mean_bars(lb, "longbench_mean.png")
            chart_longbench_heatmap(lb, "longbench_heatmap.png")
            print()
            break
    else:
        print("(skip) No LongBench results found\n")

    print(f"All charts saved to {ASSETS}/")
