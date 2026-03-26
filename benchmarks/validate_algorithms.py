"""
Validate TurboQuant algorithms against theoretical bounds from:
  Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate", ICLR 2026. arXiv:2504.19874

Tests TurboQuantMSE, QJL, and TurboQuantProd at bit-widths 1–4 on synthetic
unit vectors, checking measured MSE / inner-product distortion against the
paper's upper bounds (Tables in Theorems 1 & 2).

Run from the project root:
    python -m benchmarks.validate_algorithms
"""

import math
import torch

from turboquant.core import (
    TurboQuantConfig,
    TurboQuantMSE,
    QJL,
    TurboQuantProd,
    compute_mse,
    compute_inner_product_error,
)

PAPER_MSE_BOUNDS = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
PAPER_IP_BOUNDS = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

BIT_WIDTHS = [1, 2, 3, 4]
DIMS = [64, 128]
N_VECTORS = 10_000
BIAS_TOLERANCE = 0.02
SEED = 0


def _make_unit_vectors(n: int, d: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    return X / X.norm(dim=-1, keepdim=True)


def _header(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


def _table_header() -> None:
    print(
        f"{'bits':>5}  {'dim':>4}  {'algo':<16}  {'measured':>10}  "
        f"{'bound':>10}  {'ratio':>7}  {'result':>6}"
    )
    print("-" * 72)


def _table_row(bits: int, dim: int, algo: str, measured: float, bound: float, passed: bool) -> None:
    ratio = measured / bound if bound > 0 else float("inf")
    tag = "PASS" if passed else "FAIL"
    print(
        f"{bits:>5}  {dim:>4}  {algo:<16}  {measured:>10.6f}  "
        f"{bound:>10.6f}  {ratio:>7.3f}  {tag:>6}"
    )


def validate_mse(results: list[bool]) -> None:
    """Check E[||x - x_hat||^2] <= paper upper bound for TurboQuantMSE."""
    _header("TurboQuantMSE — MSE distortion vs. paper bounds")
    _table_header()

    for d in DIMS:
        X = _make_unit_vectors(N_VECTORS, d, SEED)
        for b in BIT_WIDTHS:
            cfg = TurboQuantConfig(bit_width=b, head_dim=d)
            quantizer = TurboQuantMSE(cfg)
            X_hat = quantizer.quantize_dequantize(X)
            mse = compute_mse(X, X_hat)
            bound = PAPER_MSE_BOUNDS[b]
            passed = mse <= bound * 1.05  # 5 % slack for finite-sample noise
            results.append(passed)
            _table_row(b, d, "TurboQuantMSE", mse, bound, passed)


def validate_qjl(results: list[bool]) -> None:
    """Check QJL 1-bit MSE and inner-product error."""
    _header("QJL — 1-bit quantizer distortion")
    _table_header()

    for d in DIMS:
        X = _make_unit_vectors(N_VECTORS, d, SEED)
        Y = _make_unit_vectors(N_VECTORS, d, SEED + 1)
        qjl = QJL(d, device=torch.device("cpu"))

        signs = qjl.quantize(X)
        gamma = torch.ones(N_VECTORS)
        X_hat = qjl.dequantize(signs, gamma)

        mse = compute_mse(X, X_hat)
        bound = PAPER_MSE_BOUNDS[1]
        passed = mse <= bound * 1.10
        results.append(passed)
        _table_row(1, d, "QJL", mse, bound, passed)

        ip_err = compute_inner_product_error(X, Y, X_hat)
        ip_bound = PAPER_IP_BOUNDS[1] / d
        passed = ip_err <= ip_bound * 1.10
        results.append(passed)
        _table_row(1, d, "QJL (IP err)", ip_err, ip_bound, passed)


def validate_prod(results: list[bool]) -> None:
    """Check TurboQuantProd inner-product error, bias, and recall."""
    _header("TurboQuantProd — inner-product error vs. paper bounds")
    _table_header()

    for d in DIMS:
        X = _make_unit_vectors(N_VECTORS, d, SEED)
        Y = _make_unit_vectors(N_VECTORS, d, SEED + 1)

        for b in BIT_WIDTHS:
            if b < 2:
                continue
            cfg = TurboQuantConfig(bit_width=b, head_dim=d)
            quantizer = TurboQuantProd(cfg)
            X_hat = quantizer.quantize_dequantize(X)

            ip_err = compute_inner_product_error(X, Y, X_hat)
            ip_bound = PAPER_IP_BOUNDS[b] / d
            passed = ip_err <= ip_bound * 1.25
            results.append(passed)
            _table_row(b, d, "TurboQuantProd", ip_err, ip_bound, passed)


def validate_prod_bias(results: list[bool]) -> None:
    """Verify TurboQuantProd inner-product estimator is approximately unbiased."""
    _header("TurboQuantProd — inner-product bias (should be ~0)")
    print(f"{'bits':>5}  {'dim':>4}  {'mean bias':>12}  {'|bias|':>10}  {'tol':>8}  {'result':>6}")
    print("-" * 56)

    for d in DIMS:
        X = _make_unit_vectors(N_VECTORS, d, SEED)
        Y = _make_unit_vectors(N_VECTORS, d, SEED + 1)

        for b in BIT_WIDTHS:
            if b < 2:
                continue
            cfg = TurboQuantConfig(bit_width=b, head_dim=d)
            quantizer = TurboQuantProd(cfg)
            X_hat = quantizer.quantize_dequantize(X)

            ip_true = (Y * X).sum(dim=-1)
            ip_recon = (Y * X_hat).sum(dim=-1)
            bias = (ip_true - ip_recon).mean().item()
            passed = abs(bias) < BIAS_TOLERANCE
            results.append(passed)
            tag = "PASS" if passed else "FAIL"
            print(
                f"{b:>5}  {d:>4}  {bias:>12.6f}  {abs(bias):>10.6f}  "
                f"{BIAS_TOLERANCE:>8.4f}  {tag:>6}"
            )


def validate_recall(results: list[bool]) -> None:
    """
    Measure 1@k recall for nearest-neighbor search with TurboQuantProd.
    For each query in Y, find the nearest vector in X under exact inner product,
    then check whether the top-k approximate results contain the true nearest.
    """
    _header("TurboQuantProd — nearest-neighbor recall@k")
    K_VALUES = [1, 10]
    d = 64
    n_db = N_VECTORS
    n_queries = 500

    X = _make_unit_vectors(n_db, d, SEED)
    Y = _make_unit_vectors(n_queries, d, SEED + 2)

    exact_sims = Y @ X.T
    exact_top1 = exact_sims.argmax(dim=-1)

    print(f"{'bits':>5}  {'k':>4}  {'recall@k':>10}  {'result':>6}")
    print("-" * 36)

    for b in [2, 3, 4]:
        cfg = TurboQuantConfig(bit_width=b, head_dim=d)
        quantizer = TurboQuantProd(cfg)
        X_hat = quantizer.quantize_dequantize(X)

        approx_sims = Y @ X_hat.T
        for k in K_VALUES:
            topk_approx = approx_sims.topk(k, dim=-1).indices
            hits = (topk_approx == exact_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
            min_recall = {1: {2: 0.05, 3: 0.20, 4: 0.40}, 10: {2: 0.30, 3: 0.70, 4: 0.90}}
            passed = hits >= min_recall[k][b]
            results.append(passed)
            tag = "PASS" if passed else "FAIL"
            print(f"{b:>5}  {k:>4}  {hits:>10.4f}  {tag:>6}")


def main() -> None:
    print("TurboQuant Algorithm Validation")
    print(f"Paper: ICLR 2026, arXiv:2504.19874")
    print(f"Vectors: n={N_VECTORS}, dims={DIMS}")

    results: list[bool] = []
    validate_mse(results)
    validate_qjl(results)
    validate_prod(results)
    validate_prod_bias(results)
    validate_recall(results)

    total = len(results)
    passed = sum(results)
    failed = total - passed
    _header("SUMMARY")
    print(f"  {passed}/{total} checks passed, {failed} failed")
    if failed == 0:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")


if __name__ == "__main__":
    main()
