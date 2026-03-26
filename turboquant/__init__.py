"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Implementation of TurboQuant from ICLR 2026 (arXiv:2504.19874).
Compresses LLM KV caches to 3-4 bits with zero accuracy loss.
"""

from turboquant.core import (
    TurboQuantConfig,
    TurboQuantMSE,
    QJL,
    TurboQuantProd,
    compute_mse,
    compute_inner_product_error,
    compute_memory_bytes,
)
from turboquant.cache import TurboQuantCache, TurboQuantLayer, get_baseline_kv_memory
from turboquant.attention import QuantizedAttention

__all__ = [
    "TurboQuantConfig",
    "TurboQuantMSE",
    "QJL",
    "TurboQuantProd",
    "TurboQuantCache",
    "TurboQuantLayer",
    "QuantizedAttention",
    "get_baseline_kv_memory",
    "compute_mse",
    "compute_inner_product_error",
    "compute_memory_bytes",
]
