# TurboQuant

A faithful, from-scratch implementation of **TurboQuant** — the KV cache compression algorithm from Google Research that achieves **3-4 bit quantization with zero accuracy loss** and no training required.

> Zandieh, Daliri, Hadian, Mirrokni. *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
> ICLR 2026 — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## Key Results (NVIDIA A40, SmolLM2-1.7B)

### Quality — Identical Outputs at 4-bit

| Prompt | Baseline FP16 | TurboQuant 4-bit | TurboQuant 3-bit |
|---|---|---|---|
| "Capital of France?" | "The capital of France is Paris." | **Identical** | **Identical** |
| Math reasoning | Correct step-by-step | **Identical** | Same approach, correct |
| Recursive factorial | Correct Python code | **Identical code** | **Identical code** |

### Memory — 3.6-4.6x KV Cache Compression

| Config | Avg KV Memory | Compression | Tokens/sec |
|---|---|---|---|
| Baseline FP16 | 16.6 MB | 1.0x | 36.0 |
| TurboQuant 4-bit | 4.6 MB | **3.6x** | 34.8 |
| TurboQuant 3-bit | 3.6 MB | **4.6x** | 38.2 |

### Attention Speedup — Quantized Attention vs Dequantize-then-Matmul

| Seq Length | Baseline Q@K^T | Dequant+Matmul | Quantized Attn | Speedup vs Dequant |
|---:|---:|---:|---:|---:|
| 1,024 | 0.022ms | 0.187ms | 0.138ms | **1.35x** |
| 4,096 | 0.032ms | 0.718ms | 0.471ms | **1.53x** |
| 16,384 | 0.113ms | 2.803ms | 1.778ms | **1.58x** |

## What's Implemented

| Component | Paper Reference | File |
|---|---|---|
| `TurboQuantMSE` | Algorithm 1 — MSE-optimal quantizer via random rotation + Lloyd-Max | `turboquant/core.py` |
| `QJL` | Definition 1 — 1-bit Quantized Johnson-Lindenstrauss transform | `turboquant/core.py` |
| `TurboQuantProd` | Algorithm 2 — Unbiased inner-product quantizer (MSE + QJL) | `turboquant/core.py` |
| `TurboQuantCache` | KV cache integration with outlier-aware channel separation (Sec 4.3) | `turboquant/cache.py` |
| `QuantizedAttention` | Compute Q@K^T directly on compressed indices (no dequantize) | `turboquant/attention.py` |
| `TurboQuantCUDA` | Fused Triton CUDA kernels for rotation + quantization | `turboquant/cuda_kernels.py` |

## Quick Start

### Local (CPU / Apple Silicon MPS)

```bash
git clone https://github.com/YOUR_USERNAME/turboquant.git
cd turboquant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m benchmarks.local
```

### GPU via RunPod

Spins up a GPU pod, runs everything, prints results, auto-terminates. No lingering charges.

```bash
cp .env.example .env
# Add your RunPod API key and (optional) HuggingFace token

python -m benchmarks.gpu              # RTX 4090 (default)
python -m benchmarks.gpu --gpu a40    # NVIDIA A40
python -m benchmarks.gpu --gpu a100   # NVIDIA A100
```

## How It Works

TurboQuant's algorithm:

1. **Random Rotation** — Multiply by a random orthogonal matrix Pi. This makes every coordinate follow a known distribution, regardless of input data.

2. **Scalar Quantization** — Apply an optimal Lloyd-Max quantizer per coordinate. The codebook is precomputed once from the known distribution.

3. **Quantized Attention** — Instead of dequantizing keys for attention, rotate the query into quantization space and compute dot products directly against centroid lookups.

```
Quantize:   x  ──→  Pi @ x  ──→  bucketize  ──→  uint8 indices (b bits/dim)
Attention:  Q  ──→  Q @ Pi^T ──→  matmul(centroids[idx])  ──→  scores
                    (rotate once)   (no full dequantize needed)
```

The key insight: random rotation transforms any input into a well-behaved distribution, enabling simple per-coordinate quantization that is provably within 2.7x of the information-theoretic lower bound.

## Project Structure

```
turboquant/
├── turboquant/               # Core package
│   ├── __init__.py           # Public API
│   ├── core.py               # TurboQuantMSE, QJL, TurboQuantProd
│   ├── cache.py              # HuggingFace KV cache integration
│   ├── attention.py          # Quantized attention (skip dequantize)
│   └── cuda_kernels.py       # Fused Triton CUDA kernels
├── benchmarks/
│   ├── local.py              # CPU/MPS benchmark
│   └── gpu.py                # RunPod GPU benchmark
├── results/                  # Benchmark results (JSON)
│   └── a40_smollm2_1.7b.json
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

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
