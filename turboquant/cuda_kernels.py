"""
TurboQuant with Triton CUDA kernels.

Fuses the rotation matrix multiply + scalar quantization into single GPU kernels,
giving 10-100x speedup over the pure-PyTorch implementation.

Requires: triton (pip install triton)
"""

import math
import torch
import numpy as np
from scipy.stats import norm

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _lloyd_max_gaussian(num_levels, sigma=1.0, max_iter=200):
    k = num_levels
    centroids = np.array([sigma * norm.ppf((2 * i + 1) / (2 * k)) for i in range(k)])
    for _ in range(max_iter):
        boundaries = np.empty(k + 1)
        boundaries[0], boundaries[k] = -np.inf, np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
        new_c = np.empty(k)
        for i in range(k):
            lo, hi = boundaries[i], boundaries[i + 1]
            lo_c, hi_c = max(lo, -6 * sigma), min(hi, 6 * sigma)
            num_val = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
            den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
            new_c[i] = num_val / den if den > 1e-15 else (lo_c + hi_c) / 2.0
        if np.allclose(centroids, new_c, atol=1e-12):
            break
        centroids = new_c
    boundaries = np.empty(k + 1)
    boundaries[0], boundaries[k] = -np.inf, np.inf
    for i in range(1, k):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
    return centroids, boundaries


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


def get_quantizer(bit_width, head_dim, device, rotation_seed=42):
    """Factory: returns CUDA Triton quantizer if available, else PyTorch fallback."""
    if HAS_TRITON and device.type == "cuda":
        return TurboQuantCUDA(bit_width, head_dim, device, rotation_seed)
    return TurboQuantFallback(bit_width, head_dim, device, rotation_seed)
