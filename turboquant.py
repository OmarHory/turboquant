"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Faithful implementation of the TurboQuant algorithm from:
  Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate", ICLR 2026. arXiv:2504.19874

Implements:
  1. TurboQuant_mse  (Algorithm 1) — MSE-optimal quantizer via random rotation
     + Lloyd-Max scalar quantization on the induced Beta/Gaussian distribution
  2. QJL            (Definition 1) — 1-bit Quantized Johnson-Lindenstrauss
  3. TurboQuant_prod (Algorithm 2) — Unbiased inner-product quantizer combining
     TurboQuant_mse on (b-1) bits + QJL on the residual
"""

import math
import torch
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


def _lloyd_max_gaussian(num_levels: int, sigma: float = 1.0, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal Lloyd-Max quantizer centroids and boundaries for N(0, sigma^2).

    In the paper (Section 3.1), each coordinate of the randomly rotated vector
    follows Beta((d-1)/2, (d-1)/2) scaled to [-1,1], which converges to N(0,1/d)
    in high dimensions. We solve the 1D k-means (Eq. 4) for this Gaussian.

    Returns:
        centroids: sorted array of 2^b centroid values
        boundaries: sorted array of 2^b + 1 boundary values (including -inf, +inf)
    """
    k = num_levels
    centroids = np.array([sigma * norm.ppf((2 * i + 1) / (2 * k)) for i in range(k)])

    for _ in range(max_iter):
        boundaries = np.empty(k + 1)
        boundaries[0] = -np.inf
        boundaries[k] = np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        new_centroids = np.empty(k)
        for i in range(k):
            lo, hi = boundaries[i], boundaries[i + 1]
            lo_c = max(lo, -6 * sigma)
            hi_c = min(hi, 6 * sigma)
            num = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
            den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
            new_centroids[i] = num / den if den > 1e-15 else (lo_c + hi_c) / 2.0

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    boundaries = np.empty(k + 1)
    boundaries[0] = -np.inf
    boundaries[k] = np.inf
    for i in range(1, k):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids, boundaries


def _generate_random_rotation(d: int, seed: int = 42, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix.
    Paper Section 3.1: Pi is a random rotation matrix in R^{d x d}.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def _generate_jl_matrix(d: int, seed: int = 137, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate the random Gaussian projection matrix S for QJL.
    Paper Definition 1: S in R^{d x d} with i.i.d. N(0,1) entries.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    S = torch.randn(d, d, generator=gen, dtype=torch.float32)
    return S.to(device)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant quantizer."""
    bit_width: int = 3
    head_dim: int = 64
    rotation_seed: int = 42
    jl_seed: int = 137
    device: torch.device = torch.device("cpu")


class TurboQuantMSE:
    """
    Algorithm 1 from the paper: TurboQuant_mse — MSE-optimal vector quantizer.

    Steps:
      1. Rotate input x by random orthogonal Pi -> y = Pi @ x
      2. Scalar-quantize each coordinate of y using precomputed Lloyd-Max codebook
      3. Dequantize: look up centroids, rotate back x_hat = Pi^T @ y_hat
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        d = config.head_dim
        b = config.bit_width
        num_levels = 2 ** b

        self.Pi = _generate_random_rotation(d, seed=config.rotation_seed, device=config.device)
        self.Pi_T = self.Pi.T

        sigma = 1.0 / math.sqrt(d)
        centroids_np, boundaries_np = _lloyd_max_gaussian(num_levels, sigma=sigma)

        self.centroids = torch.tensor(centroids_np, dtype=torch.float32, device=config.device)
        finite_bounds = boundaries_np[1:-1]
        self.boundaries = torch.tensor(finite_bounds, dtype=torch.float32, device=config.device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input vectors.

        Args:
            x: (..., d) tensor of input vectors

        Returns:
            idx: (..., d) tensor of uint8 indices into the codebook
        """
        orig_dtype = x.dtype
        x_f32 = x.float()
        y = x_f32 @ self.Pi.T  # rotate: y = Pi @ x, but x is row-vector
        idx = torch.bucketize(y, self.boundaries).to(torch.uint8)
        return idx

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vectors from quantized indices.

        Args:
            idx: (..., d) tensor of uint8 codebook indices

        Returns:
            x_hat: (..., d) tensor of reconstructed vectors
        """
        y_hat = self.centroids[idx.long()]
        x_hat = y_hat @ self.Pi  # rotate back: x_hat = Pi^T @ y_hat
        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip: quantize then immediately dequantize."""
        return self.dequantize(self.quantize(x))


class QJL:
    """
    Definition 1 from the paper: Quantized Johnson-Lindenstrauss transform.

    Q_qjl(x)      = sign(S @ x)
    Q_qjl_inv(z)  = (sqrt(pi/2) / d) * S^T @ z

    This is a 1-bit quantizer with zero memory overhead that provides
    unbiased inner product estimates.
    """

    def __init__(self, d: int, seed: int = 137, device: torch.device = torch.device("cpu")):
        self.d = d
        self.S = _generate_jl_matrix(d, seed=seed, device=device)
        self.S_T = self.S.T
        self.scale = math.sqrt(math.pi / 2) / d

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize to sign bits.

        Args:
            x: (..., d) input vectors

        Returns:
            signs: (..., d) tensor of {-1, +1} stored as int8
        """
        x_f32 = x.float()
        projected = x_f32 @ self.S.T  # S @ x but x is row-vector
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1
        return signs

    def dequantize(self, signs: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Dequantize sign bits back to vectors.

        Args:
            signs: (..., d) tensor of {-1, +1}
            gamma: (...) tensor of residual norms

        Returns:
            x_hat: (..., d) reconstructed contribution
        """
        z = signs.float()
        x_hat = self.scale * z @ self.S  # S^T @ z but z is row-vector
        x_hat = x_hat * gamma.unsqueeze(-1)
        return x_hat


class TurboQuantProd:
    """
    Algorithm 2 from the paper: TurboQuant_prod — unbiased inner-product quantizer.

    Two-stage approach:
      Stage 1: Apply TurboQuant_mse at bit-width (b-1) to minimize MSE
      Stage 2: Apply QJL to the residual (1 bit) for unbiased inner products

    Total bit-width = (b-1) + 1 = b bits per coordinate.
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        mse_config = TurboQuantConfig(
            bit_width=max(1, config.bit_width - 1),
            head_dim=config.head_dim,
            rotation_seed=config.rotation_seed,
            jl_seed=config.jl_seed,
            device=config.device,
        )
        self.mse_quantizer = TurboQuantMSE(mse_config)
        self.qjl = QJL(config.head_dim, seed=config.jl_seed, device=config.device)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize using two-stage approach.

        Args:
            x: (..., d) input vectors

        Returns:
            idx:   (..., d) MSE quantizer indices (uint8)
            signs: (..., d) QJL sign bits (int8)
            gamma: (...) residual norms (float32)
        """
        idx = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(idx)
        residual = x.float() - x_hat_mse
        gamma = torch.norm(residual, dim=-1)
        safe_residual = residual.clone()
        nz = gamma > 1e-10
        safe_residual[nz] = safe_residual[nz] / gamma[nz].unsqueeze(-1)
        signs = self.qjl.quantize(safe_residual)
        return idx, signs, gamma

    def dequantize(self, idx: torch.Tensor, signs: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from two-stage quantization.

        Args:
            idx:   (..., d) MSE quantizer indices
            signs: (..., d) QJL sign bits
            gamma: (...) residual norms

        Returns:
            x_hat: (..., d) reconstructed vectors
        """
        x_mse = self.mse_quantizer.dequantize(idx)
        x_qjl = self.qjl.dequantize(signs, gamma)
        return x_mse + x_qjl

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip: quantize then dequantize."""
        idx, signs, gamma = self.quantize(x)
        return self.dequantize(idx, signs, gamma)


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute mean squared error between original and reconstructed vectors."""
    return ((original.float() - reconstructed.float()) ** 2).mean().item()


def compute_inner_product_error(x: torch.Tensor, y: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Compute mean squared inner product error: E[|<y,x> - <y,x_hat>|^2]."""
    ip_orig = (y.float() * x.float()).sum(dim=-1)
    ip_recon = (y.float() * x_hat.float()).sum(dim=-1)
    return ((ip_orig - ip_recon) ** 2).mean().item()


def compute_memory_bytes(bit_width: int, num_elements: int, include_gamma: bool = False, d: int = 64) -> int:
    """
    Compute actual memory in bytes for quantized storage.

    For TurboQuant_mse: bit_width * num_elements / 8
    For TurboQuant_prod: (bit_width - 1) * num_elements / 8 (MSE indices)
                         + num_elements / 8 (QJL sign bits)
                         + (num_elements / d) * 4 (gamma floats, one per vector)
    Baseline FP16: num_elements * 2
    """
    if include_gamma:
        mse_bits = (bit_width - 1) * num_elements
        qjl_bits = num_elements
        gamma_bytes = (num_elements // d) * 4
        return (mse_bits + qjl_bits) // 8 + gamma_bytes
    else:
        return (bit_width * num_elements) // 8


if __name__ == "__main__":
    print("=== TurboQuant Unit Test ===\n")
    d = 64
    n = 1000

    torch.manual_seed(0)
    X = torch.randn(n, d)
    X = X / X.norm(dim=-1, keepdim=True)

    Y = torch.randn(n, d)
    Y = Y / Y.norm(dim=-1, keepdim=True)

    print(f"Test vectors: {n} unit vectors in R^{d}\n")

    for b in [2, 3, 4]:
        print(f"--- Bit-width b={b} ---")

        cfg = TurboQuantConfig(bit_width=b, head_dim=d)
        mse_q = TurboQuantMSE(cfg)
        prod_q = TurboQuantProd(cfg)

        X_hat_mse = mse_q.quantize_dequantize(X)
        X_hat_prod = prod_q.quantize_dequantize(X)

        mse_err_mse = compute_mse(X, X_hat_mse)
        mse_err_prod = compute_mse(X, X_hat_prod)

        ip_err_mse = compute_inner_product_error(X, Y, X_hat_mse)
        ip_err_prod = compute_inner_product_error(X, Y, X_hat_prod)

        # Paper bounds (Table from Theorem 1 and 2)
        paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
        paper_prod = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

        ub_mse = paper_mse.get(b, math.sqrt(3) * math.pi / 2 / (4 ** b))
        ub_prod = paper_prod.get(b, math.sqrt(3) * math.pi ** 2 / d / (4 ** b))

        print(f"  TurboQuant_mse  MSE: {mse_err_mse:.6f}  (paper upper bound: {ub_mse})")
        print(f"  TurboQuant_prod MSE: {mse_err_prod:.6f}")
        print(f"  TurboQuant_mse  IP error: {ip_err_mse:.6f}")
        print(f"  TurboQuant_prod IP error: {ip_err_prod:.6f}  (paper UB: {ub_prod/d:.6f})")

        ip_bias_mse = ((Y * X).sum(-1) - (Y * X_hat_mse).sum(-1)).mean().item()
        ip_bias_prod = ((Y * X).sum(-1) - (Y * X_hat_prod).sum(-1)).mean().item()
        print(f"  IP bias (mse):  {ip_bias_mse:.6f}")
        print(f"  IP bias (prod): {ip_bias_prod:.6f}  (should be ~0)")
        print()
