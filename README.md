# TurboQuant

A faithful implementation of **TurboQuant** from the ICLR 2026 paper:

> Zandieh, Daliri, Hadian, Mirrokni. *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
> [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | [OpenReview](https://openreview.net/forum?id=tO3ASKZlok)

TurboQuant compresses LLM key-value (KV) caches to **3-4 bits per value** with **zero accuracy loss** and **no training or calibration** required. It is provably within 2.7x of the information-theoretic lower bound.

## What's Implemented

| Component | Paper Reference | File |
|---|---|---|
| `TurboQuantMSE` | Algorithm 1 — MSE-optimal quantizer via random rotation + Lloyd-Max | `turboquant.py` |
| `QJL` | Definition 1 — 1-bit Quantized Johnson-Lindenstrauss transform | `turboquant.py` |
| `TurboQuantProd` | Algorithm 2 — Unbiased inner-product quantizer (MSE + QJL on residual) | `turboquant.py` |
| `TurboQuantCache` | KV cache integration with outlier-aware channel separation (Section 4.3) | `patched_model.py` |
| `TurboQuantCUDA` | Fused Triton CUDA kernels for rotation + quantization | `turboquant_cuda.py` |

## Quick Start

### Local (CPU / Apple Silicon MPS)

```bash
git clone https://github.com/YOUR_USERNAME/turboquant.git
cd turboquant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python benchmark.py
```

Runs SmolLM2-1.7B-Instruct with baseline vs TurboQuant 4-bit vs 3-bit and prints a comparison table with side-by-side outputs.

### GPU via RunPod

Run the benchmark on a real GPU. The script creates a pod, runs the benchmark over SSH, prints results, and auto-terminates the pod — no lingering charges.

```bash
cp .env.example .env
# Edit .env — add your RunPod API key and (optional) HuggingFace token
```

```bash
# RTX 4090 (default, cheapest)
python benchmark_gpu.py

# Or pick a specific GPU
python benchmark_gpu.py --gpu a100
python benchmark_gpu.py --gpu a40
python benchmark_gpu.py --gpu h100
```

Available GPU options: `4090`, `a100`, `a100-sxm`, `h100`, `l4`, `a40`

## Results (NVIDIA A40)

### Quality (SmolLM2-1.7B-Instruct)

TurboQuant at 4-bit produces **identical or near-identical outputs** to full-precision:

| Prompt | Baseline FP16 | TurboQuant 4-bit | TurboQuant 3-bit |
|---|---|---|---|
| "Capital of France?" | "The capital of France is Paris." | **Identical** | **Identical** |
| Math reasoning | Distance = Speed x Time (correct) | Same approach (correct) | Same approach (correct) |
| Factorial code | Correct recursive implementation | **Identical code** | **Identical code** |

### KV Cache Memory

| Config | Avg KV Memory | Compression vs FP16 | Tokens/sec |
|---|---|---|---|
| Baseline FP16 | 16.6 MB | 1.0x | 36.0 |
| TurboQuant 4-bit | 4.6 MB | **3.6x** | 34.8 |
| TurboQuant 3-bit | 3.6 MB | **4.6x** | 38.2 |

### Memory Scaling (theoretical, SmolLM2-1.7B)

| Seq Length | Baseline FP32 | TQ 4-bit | TQ 3-bit |
|---|---|---|---|
| 128 tokens | 48 MB | 6.75 MB (7.1x) | 5.25 MB (9.1x) |
| 1024 tokens | 384 MB | 54 MB (7.1x) | 42 MB (9.1x) |
| 4096 tokens | 1.5 GB | 216 MB (7.1x) | 168 MB (9.1x) |

## How It Works

TurboQuant's algorithm is elegant:

1. **Random Rotation**: Multiply the input vector by a random orthogonal matrix. This makes every coordinate follow a known Beta distribution (converging to Gaussian in high dimensions), regardless of the input data.

2. **Scalar Quantization**: Since coordinates are now near-independent with a known distribution, apply an optimal Lloyd-Max scalar quantizer to each coordinate independently. The codebook is precomputed once.

3. **QJL Residual** (for inner-product variant): Apply a 1-bit sign quantization via the Johnson-Lindenstrauss transform to the residual error, making the inner product estimate unbiased.

```
Input x ──→ [Random Rotate] ──→ [Scalar Quantize per coord] ──→ Compressed
               Pi·x                   Lloyd-Max codebook           b bits/dim
```

The key insight: random rotation transforms any worst-case input into a well-behaved distribution, enabling simple per-coordinate quantization to achieve near-optimal distortion rates.

## File Structure

```
turboquant/
├── turboquant.py        # Core algorithms: TurboQuantMSE, QJL, TurboQuantProd
├── turboquant_cuda.py   # Fused Triton CUDA kernels (optional, for GPU)
├── patched_model.py     # KV cache integration for HuggingFace models
├── benchmark.py         # Local CPU/MPS benchmark
├── benchmark_gpu.py     # RunPod GPU benchmark (auto pod lifecycle)
├── requirements.txt     # Dependencies
├── .env.example         # Template for API keys
└── LICENSE
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT
