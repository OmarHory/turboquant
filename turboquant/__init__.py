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
from turboquant.cache import (
    TurboQuantCache,
    TurboQuantLayer,
    TQLayerFused,
    get_baseline_kv_memory,
    detect_outlier_channels,
    effective_bit_width,
)
from turboquant.packing import pack_indices, unpack_indices, packed_size_bytes, compression_ratio
from turboquant.attention import QuantizedAttention

__all__ = [
    "TurboQuantConfig",
    "TurboQuantMSE",
    "QJL",
    "TurboQuantProd",
    "TurboQuantCache",
    "TurboQuantLayer",
    "TQLayerFused",
    "QuantizedAttention",
    "get_baseline_kv_memory",
    "detect_outlier_channels",
    "effective_bit_width",
    "compute_mse",
    "compute_inner_product_error",
    "compute_memory_bytes",
    "pack_indices",
    "unpack_indices",
    "packed_size_bytes",
    "compression_ratio",
]
