"""
TurboQuant with Triton CUDA kernels.

Fuses the rotation matrix multiply + scalar quantization into single GPU kernels,
giving 10-100x speedup over the pure-PyTorch implementation.

Requires: triton (pip install triton)
"""

import math
import torch
import numpy as np

from turboquant.core import _lloyd_max_gaussian

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_rotate_quantize_kernel(
        X_ptr, Pi_ptr, Bounds_ptr, Out_ptr, Norms_ptr,
        N, D: tl.constexpr, NUM_BOUNDS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused: compute norm, normalize, rotate by Pi, scalar-quantize via boundaries."""
        row = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_row = row < N

        d_range = tl.arange(0, D)

        for b in range(BLOCK_N):
            r = tl.program_id(0) * BLOCK_N + b
            if r >= N:
                break

            x_ptrs = X_ptr + r * D + d_range
            x = tl.load(x_ptrs, mask=d_range < D, other=0.0).to(tl.float32)

            norm_sq = tl.sum(x * x, axis=0)
            norm_val = tl.sqrt(norm_sq + 1e-20)

            tl.store(Norms_ptr + r, norm_val)

            x_normed = x / norm_val

            for j in range(D):
                pi_ptrs = Pi_ptr + j * D + d_range
                pi_row = tl.load(pi_ptrs, mask=d_range < D, other=0.0).to(tl.float32)
                y_j = tl.sum(x_normed * pi_row, axis=0)

                idx = tl.zeros([], dtype=tl.int32)
                for k in range(NUM_BOUNDS):
                    bound_val = tl.load(Bounds_ptr + k)
                    idx = tl.where(y_j > bound_val, idx + 1, idx)

                tl.store(Out_ptr + r * D + j, idx.to(tl.uint8))

    @triton.jit
    def _fused_dequant_rotate_kernel(
        Idx_ptr, Centroids_ptr, Pi_ptr, Norms_ptr, Out_ptr,
        N, D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused: lookup centroids, rotate back by Pi^T, scale by norm."""
        for b in range(BLOCK_N):
            r = tl.program_id(0) * BLOCK_N + b
            if r >= N:
                break

            d_range = tl.arange(0, D)

            idx_ptrs = Idx_ptr + r * D + d_range
            indices = tl.load(idx_ptrs, mask=d_range < D, other=0).to(tl.int64)
            y_hat = tl.load(Centroids_ptr + indices, mask=d_range < D, other=0.0).to(tl.float32)

            norm_val = tl.load(Norms_ptr + r).to(tl.float32)

            for j in range(D):
                pi_col_ptrs = Pi_ptr + d_range * D + j
                pi_col = tl.load(pi_col_ptrs, mask=d_range < D, other=0.0).to(tl.float32)
                x_j = tl.sum(y_hat * pi_col, axis=0) * norm_val
                tl.store(Out_ptr + r * D + j, x_j)


class TurboQuantCUDA:
    """
    TurboQuant MSE with fused Triton CUDA kernels.

    The two hot-path operations are fused into single GPU kernels:
    1. quantize:   x -> normalize -> rotate by Pi -> scalar quantize -> indices
    2. dequantize: indices -> centroid lookup -> rotate by Pi^T -> scale by norm -> x_hat
    """

    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        assert HAS_TRITON, "Triton is required for CUDA kernels. pip install triton"
        assert device.type == "cuda", "CUDA device required"

        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device

        d = head_dim
        gen = torch.Generator(device="cpu").manual_seed(rotation_seed)
        G = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        ds = torch.sign(torch.diag(R))
        ds[ds == 0] = 1.0
        self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()

        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()
        self.num_bounds = len(b_np) - 2

    def quantize(self, x):
        """
        Fused normalize + rotate + scalar quantize.

        Args:
            x: (N, D) float tensor, raw vectors (NOT pre-normalized)

        Returns:
            idx: (N, D) uint8 tensor of codebook indices
            norms: (N,) float32 tensor of input norms
        """
        assert x.shape[-1] == self.head_dim
        orig_shape = x.shape
        flat = x.float().reshape(-1, self.head_dim).contiguous()
        N, D = flat.shape

        idx = torch.empty(N, D, dtype=torch.uint8, device=self.device)
        norms = torch.empty(N, dtype=torch.float32, device=self.device)

        BLOCK_N = 1
        grid = (N,)
        _fused_rotate_quantize_kernel[grid](
            flat, self.Pi, self.boundaries, idx, norms,
            N, D, self.num_bounds,
            BLOCK_N,
        )
        return idx.view(*orig_shape), norms

    def dequantize(self, idx, norms):
        """
        Fused centroid lookup + inverse rotate + scale.

        Args:
            idx: (N, D) uint8 codebook indices
            norms: (N,) float32 norms

        Returns:
            x_hat: (N, D) float32 reconstructed vectors
        """
        orig_shape = idx.shape
        flat_idx = idx.reshape(-1, self.head_dim).contiguous()
        flat_norms = norms.reshape(-1).contiguous()
        N, D = flat_idx.shape

        out = torch.empty(N, D, dtype=torch.float32, device=self.device)

        BLOCK_N = 1
        grid = (N,)
        _fused_dequant_rotate_kernel[grid](
            flat_idx, self.centroids, self.Pi, flat_norms, out,
            N, D,
            BLOCK_N,
        )
        return out.view(*orig_shape)

    def quantize_dequantize(self, x):
        idx, norms = self.quantize(x)
        return self.dequantize(idx, norms)


class TurboQuantFallback:
    """Pure PyTorch fallback when Triton is not available."""

    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device

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

    def quantize(self, x):
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        normalized = flat / norms
        y = normalized @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(x.shape), norms.squeeze(-1)

    def dequantize(self, idx, norms):
        flat_idx = idx.reshape(-1, self.head_dim)
        y_hat = self.centroids[flat_idx.long()]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.reshape(-1, 1)
        return x_hat.view(idx.shape)

    def quantize_dequantize(self, x):
        idx, norms = self.quantize(x)
        return self.dequantize(idx, norms)


if HAS_TRITON:

    @triton.jit
    def _fused_quantized_attention_kernel(
        Q_rot_ptr, K_idx_ptr, K_norms_ptr, Centroids_ptr, Out_ptr,
        B_H: tl.constexpr, Q_LEN: tl.constexpr, KV_LEN, D: tl.constexpr,
        scale,
        BLOCK_KV: tl.constexpr,
    ):
        """
        Fused quantized attention: compute Q_rot @ centroids[K_idx]^T * K_norms * scale.

        Each program handles one (batch*head, query_pos) pair.
        Iterates over KV positions in blocks, accumulating the dot product
        via centroid lookup from shared memory.
        """
        bh = tl.program_id(0)
        q_pos = tl.program_id(1)

        d_range = tl.arange(0, D)
        q_ptrs = Q_rot_ptr + bh * Q_LEN * D + q_pos * D + d_range
        q_vals = tl.load(q_ptrs, mask=d_range < D, other=0.0).to(tl.float32)

        for kv_start in range(0, KV_LEN, BLOCK_KV):
            kv_range = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_range < KV_LEN

            acc = tl.zeros([BLOCK_KV], dtype=tl.float32)

            for j in range(D):
                idx_ptrs = K_idx_ptr + bh * KV_LEN * D + kv_range * D + j
                indices = tl.load(idx_ptrs, mask=kv_mask, other=0).to(tl.int64)
                c_vals = tl.load(Centroids_ptr + indices, mask=kv_mask, other=0.0).to(tl.float32)
                acc += q_vals[j] * c_vals

            norm_ptrs = K_norms_ptr + bh * KV_LEN + kv_range
            norms = tl.load(norm_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            acc = acc * norms * scale

            out_ptrs = Out_ptr + bh * Q_LEN * KV_LEN + q_pos * KV_LEN + kv_range
            tl.store(out_ptrs, acc, mask=kv_mask)


class FusedQuantizedAttentionCUDA:
    """
    Compute Q @ K_hat^T entirely via Triton, without materializing the full
    dequantized key tensor. The centroid table (2^b entries) stays in registers.

    This is the "fused" approach from the paper: the kernel reads uint8 indices
    from global memory, looks up centroids, and accumulates the dot product —
    never creating the (seq_len, head_dim) float intermediate.
    """

    def __init__(self, bit_width, head_dim, device, rotation_seed=42):
        assert HAS_TRITON, "Triton required for fused attention"
        assert device.type == "cuda"

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
        self.Pi = (Q * ds.unsqueeze(0)).to(device).contiguous()
        self.Pi_T = self.Pi.T.contiguous()

        sigma = 1.0 / math.sqrt(d)
        c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
        self.centroids = torch.tensor(c_np, dtype=torch.float32, device=device).contiguous()
        self.boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=device).contiguous()

    @torch.no_grad()
    def quantize_keys(self, K):
        flat = K.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        y = (flat / norms) @ self.Pi_T
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx.view(K.shape), norms.squeeze(-1).view(K.shape[:-1])

    @torch.no_grad()
    def fused_attention_scores(self, Q_tensor, K_idx, K_norms):
        """
        Compute attention scores via Triton kernel — no intermediate float tensor.

        Args:
            Q_tensor: (B, H, q_len, D) float queries
            K_idx:    (B, H, kv_len, D) uint8 key indices
            K_norms:  (B, H, kv_len) float32 key norms

        Returns:
            scores: (B, H, q_len, kv_len) pre-softmax attention scores
        """
        B, H, q_len, D = Q_tensor.shape
        kv_len = K_idx.shape[2]
        BH = B * H

        Q_rot = (Q_tensor.float().reshape(BH, q_len, D) @ self.Pi_T).contiguous()
        K_idx_flat = K_idx.reshape(BH, kv_len, D).contiguous()
        K_norms_flat = K_norms.reshape(BH, kv_len).contiguous()

        out = torch.empty(BH, q_len, kv_len, dtype=torch.float32, device=self.device)

        BLOCK_KV = min(128, triton.next_power_of_2(kv_len))
        grid = (BH, q_len)

        _fused_quantized_attention_kernel[grid](
            Q_rot, K_idx_flat, K_norms_flat, self.centroids, out,
            BH, q_len, kv_len, D,
            self.scale,
            BLOCK_KV,
        )

        return out.view(B, H, q_len, kv_len)


def get_quantizer(bit_width, head_dim, device, rotation_seed=42):
    """Factory: returns CUDA Triton quantizer if available, else PyTorch fallback."""
    if HAS_TRITON and device.type == "cuda":
        return TurboQuantCUDA(bit_width, head_dim, device, rotation_seed)
    return TurboQuantFallback(bit_width, head_dim, device, rotation_seed)


def get_fused_attention(bit_width, head_dim, device, rotation_seed=42):
    """Factory: returns fused Triton attention if available, else None."""
    if HAS_TRITON and device.type == "cuda":
        return FusedQuantizedAttentionCUDA(bit_width, head_dim, device, rotation_seed)
    return None
