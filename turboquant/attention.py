"""
Quantized Attention: compute Q @ K_hat^T directly from compressed indices.

The standard approach dequantizes all keys, then does attention:
    K_hat = centroids[idx] @ Pi        # O(S*D^2)  ← expensive
    scores = Q @ K_hat.T               # O(S*D)

This module skips full dequantization by exploiting the structure:
    Q @ K_hat^T = Q @ (centroids[idx] @ Pi)^T
               = Q @ Pi^T @ centroids[idx]^T
               = Q_rot @ centroids[idx]^T

So we:
    1. Pre-rotate Q once:  Q_rot = Q @ Pi^T            # O(D^2), done once
    2. Look up centroids:  C = centroids[idx]           # O(S*D), uint8 gather
    3. Dot product:        scores = Q_rot @ C^T         # O(S*D)

The win: step 2 loads uint8 indices (b bits) from HBM instead of float16 keys
(16 bits), reducing memory bandwidth by 4-16x. The centroid table (2^b entries)
fits in L1/L2 cache.

For the value side, we still dequantize V normally (needed for weighted sum).
Norms are folded into the scores as a diagonal scaling.
"""

import math
import torch
import numpy as np

from turboquant.core import _lloyd_max_gaussian

try:
    from turboquant.cuda_kernels import FusedQuantizedAttentionCUDA, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    FusedQuantizedAttentionCUDA = None


class QuantizedAttention:
    """
    Compute attention scores directly on TurboQuant-compressed keys.

    Instead of: dequantize -> full matmul
    Does:       rotate Q -> gather centroids -> matmul (4-16x less memory bandwidth)
    """

    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device
        self.scale = 1.0 / math.sqrt(head_dim)

        d = head_dim
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R))
        ds[ds == 0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device)

        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device)
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device)

        self.Pi_T = self.Pi.T.contiguous()

        self._fused = None
        if HAS_TRITON and device.type == "cuda" and FusedQuantizedAttentionCUDA is not None:
            self._fused = FusedQuantizedAttentionCUDA(bit_width, head_dim, device, rotation_seed)

    @torch.no_grad()
    def quantize_keys(self, K):
        """
        Quantize key vectors for later use in quantized attention.

        Args:
            K: (batch, heads, seq_len, head_dim) float keys

        Returns:
            idx:   (batch, heads, seq_len, head_dim) uint8 indices
            norms: (batch, heads, seq_len) float32 norms
        """
        flat = K.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi_T  # Pi @ x (row-vector convention)
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(K.shape), norms.squeeze(-1).view(K.shape[:-1])

    @torch.no_grad()
    def dequantize(self, idx, norms):
        """Standard dequantize for values (V still needs full reconstruction)."""
        orig_shape = idx.shape
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(orig_shape)

    @torch.no_grad()
    def quantized_attention_scores(self, Q_tensor, K_idx, K_norms, dtype=torch.float16,
                                    use_fused=True):
        """
        Compute scaled attention scores Q @ K_hat^T without full dequantization.

        When a Triton fused kernel is available and use_fused=True, delegates to
        the kernel that never materializes the (kv_len, D) centroid tensor.
        Otherwise falls back to PyTorch gather + matmul.

        Args:
            Q_tensor: (batch, heads, q_len, head_dim) float query
            K_idx:    (batch, heads, kv_len, head_dim) uint8 quantized key indices
            K_norms:  (batch, heads, kv_len) float32 key norms
            dtype:    compute dtype (float16 for speed)
            use_fused: if True, use Triton kernel when available

        Returns:
            scores: (batch, heads, q_len, kv_len) attention scores (pre-softmax)
        """
        if use_fused and self._fused is not None:
            return self._fused.fused_attention_scores(Q_tensor, K_idx, K_norms)

        Q_f32 = Q_tensor.float()
        Q_rot = Q_f32 @ self.Pi_T
        C_K = self.centroids[K_idx.long()]
        Q_rot = Q_rot.to(dtype)
        C_K = C_K.to(dtype)
        raw = torch.matmul(Q_rot, C_K.transpose(-2, -1))
        scores = raw * K_norms.unsqueeze(-2).to(dtype) * self.scale
        return scores

    @torch.no_grad()
    def full_quantized_attention(self, Q_tensor, K_idx, K_norms, V_idx, V_norms,
                                  dtype=torch.float16):
        """
        Full attention with quantized keys AND quantized values.

        Q @ K_hat^T is computed without dequantizing K (fast path).
        softmax(scores) @ V_hat still requires dequantizing V.

        Args:
            Q_tensor: (B, H, q_len, D) queries
            K_idx:    (B, H, kv_len, D) quantized key indices
            K_norms:  (B, H, kv_len) key norms
            V_idx:    (B, H, kv_len, D) quantized value indices
            V_norms:  (B, H, kv_len) value norms

        Returns:
            output:   (B, H, q_len, D) attention output
            scores:   (B, H, q_len, kv_len) attention weights (after softmax)
        """
        # Fast-path attention scores (no K dequantization)
        scores = self.quantized_attention_scores(Q_tensor, K_idx, K_norms, dtype)

        # Softmax
        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype)

        # V must be dequantized for the weighted sum (no shortcut here)
        V_hat = self.dequantize(V_idx, V_norms).to(dtype)

        # Weighted sum
        output = torch.matmul(attn_weights, V_hat)  # (B, H, q_len, D)

        return output, attn_weights


def benchmark_quantized_attention(device, head_dim=64, num_heads=32, bit_width=4,
                                   seq_lengths=None, iterations=200, warmup=10):
    """
    Benchmark: standard attention vs quantized attention.

    Measures the attention score computation: Q @ K^T
    - Baseline: Q @ K^T with float16 K
    - TQ (dequant): dequantize K, then Q @ K^T  (our old approach)
    - TQ (quantized attn): Q_rot @ centroids[idx]^T * norms  (new approach)

    Returns dict of results per sequence length.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    qa = QuantizedAttention(bit_width, head_dim, device)
    results = {}

    for sl in seq_lengths:
        Q_f = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16, device=device)
        K_f = torch.randn(1, num_heads, sl, head_dim, dtype=torch.float16, device=device)

        K_idx, K_norms = qa.quantize_keys(K_f)

        # === Baseline: standard Q @ K^T ===
        for _ in range(warmup):
            _ = torch.matmul(Q_f, K_f.transpose(-2, -1))
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            t0.record()
            for _ in range(iterations):
                _ = torch.matmul(Q_f, K_f.transpose(-2, -1))
            t1.record()
            torch.cuda.synchronize()
            baseline_ms = t0.elapsed_time(t1) / iterations
        else:
            import time
            st = time.time()
            for _ in range(iterations):
                _ = torch.matmul(Q_f, K_f.transpose(-2, -1))
            baseline_ms = (time.time() - st) / iterations * 1000

        # === Old approach: dequantize then matmul ===
        for _ in range(warmup):
            K_hat = qa.dequantize(K_idx, K_norms).to(torch.float16)
            _ = torch.matmul(Q_f, K_hat.transpose(-2, -1))
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            t0.record()
            for _ in range(iterations):
                K_hat = qa.dequantize(K_idx, K_norms).to(torch.float16)
                _ = torch.matmul(Q_f, K_hat.transpose(-2, -1))
            t1.record()
            torch.cuda.synchronize()
            dequant_ms = t0.elapsed_time(t1) / iterations
        else:
            st = time.time()
            for _ in range(iterations):
                K_hat = qa.dequantize(K_idx, K_norms).to(torch.float16)
                _ = torch.matmul(Q_f, K_hat.transpose(-2, -1))
            dequant_ms = (time.time() - st) / iterations * 1000

        # === New approach: quantized attention (no dequantize) ===
        for _ in range(warmup):
            _ = qa.quantized_attention_scores(Q_f, K_idx, K_norms)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            t0.record()
            for _ in range(iterations):
                _ = qa.quantized_attention_scores(Q_f, K_idx, K_norms)
            t1.record()
            torch.cuda.synchronize()
            qattn_ms = t0.elapsed_time(t1) / iterations
        else:
            st = time.time()
            for _ in range(iterations):
                _ = qa.quantized_attention_scores(Q_f, K_idx, K_norms)
            qattn_ms = (time.time() - st) / iterations * 1000

        results[sl] = {
            "baseline_ms": round(baseline_ms, 4),
            "dequant_then_matmul_ms": round(dequant_ms, 4),
            "quantized_attn_ms": round(qattn_ms, 4),
            "speedup_vs_baseline": round(baseline_ms / qattn_ms, 2) if qattn_ms > 0 else 0,
            "speedup_vs_dequant": round(dequant_ms / qattn_ms, 2) if qattn_ms > 0 else 0,
        }

    return results


def verify_correctness(device, head_dim=64, num_heads=4, seq_len=128, bit_width=4):
    """Verify that quantized attention produces the same scores as dequantize-then-matmul."""
    qa = QuantizedAttention(bit_width, head_dim, device)

    Q_f = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32, device=device)
    K_f = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

    K_idx, K_norms = qa.quantize_keys(K_f)

    # Method 1: dequantize then matmul
    K_hat = qa.dequantize(K_idx, K_norms)
    scores_ref = torch.matmul(Q_f, K_hat.transpose(-2, -1)) * qa.scale

    # Method 2: quantized attention
    scores_qa = qa.quantized_attention_scores(Q_f, K_idx, K_norms, dtype=torch.float32)

    max_err = (scores_ref - scores_qa).abs().max().item()
    mean_err = (scores_ref - scores_qa).abs().mean().item()

    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "scores_match": max_err < 1e-4,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== Correctness Verification ===")
    v = verify_correctness(device)
    print(f"  Max error:  {v['max_error']:.2e}")
    print(f"  Mean error: {v['mean_error']:.2e}")
    print(f"  Match: {'PASS' if v['scores_match'] else 'FAIL'}")

    print("\n=== Attention Speedup Benchmark ===")
    seq_lens = [512, 1024, 2048, 4096]
    if device.type == "cuda":
        seq_lens += [8192, 16384]

    results = benchmark_quantized_attention(device, seq_lengths=seq_lens)
    print(f"\n{'SeqLen':>8} | {'Baseline':>10} | {'Dequant+MM':>12} | {'Quant Attn':>12} | {'vs Base':>8} | {'vs Dequant':>10}")
    print("-" * 75)
    for sl, r in sorted(results.items()):
        print(f"{sl:>8} | {r['baseline_ms']:>9.3f}ms | {r['dequant_then_matmul_ms']:>11.3f}ms | "
              f"{r['quantized_attn_ms']:>11.3f}ms | {r['speedup_vs_baseline']:>7.2f}x | {r['speedup_vs_dequant']:>9.2f}x")
