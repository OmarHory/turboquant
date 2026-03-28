"""
KV Cache patching for HuggingFace models using TurboQuant.

Implements the paper's KV cache quantization strategy (Section 4.2-4.3):
  - Uses TurboQuant_mse for KV cache (not _prod, which is for vector search)
  - Per-channel outlier separation: top-k coordinate channels get higher bit-width
  - Quantizes both prefill and generation tokens

Section 4.3: "32 outlier channels are quantized at 3 bits, while the remaining
96 channels use 2 bits" for the 2.5-bit configuration on head_dim=128.

KV cache tensor shape: [batch_size, num_heads, seq_len, head_dim]
"""

import math
import torch
from typing import Any
from transformers.cache_utils import DynamicCache, DynamicLayer
from turboquant.core import TurboQuantMSE, TurboQuantConfig
from turboquant.packing import pack_indices, unpack_indices, packed_size_bytes


def detect_outlier_channels(key_states: torch.Tensor, num_outliers: int) -> torch.Tensor:
    """
    Identify outlier coordinate channels within each head based on L2 norm
    magnitude across the sequence dimension.

    Per the paper Section 4.3, outlier channels are those with the highest
    per-coordinate variance/magnitude. We measure RMS per channel across
    batch, heads, and seq_len, then select the top-k channels.

    Args:
        key_states: (batch, heads, seq_len, head_dim)
        num_outliers: number of channels to flag as outliers

    Returns:
        bool tensor of shape (head_dim,) — True for outlier channels
    """
    rms_per_channel = key_states.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
    _, top_indices = rms_per_channel.topk(min(num_outliers, rms_per_channel.shape[0]))
    mask = torch.zeros(rms_per_channel.shape[0], dtype=torch.bool, device=key_states.device)
    mask[top_indices] = True
    return mask


def effective_bit_width(head_dim: int, num_outliers: int, base_bits: int, outlier_bits: int) -> float:
    """Compute the effective bits per channel for a mixed-precision config."""
    regular = head_dim - num_outliers
    return (regular * base_bits + num_outliers * outlier_bits) / head_dim


class TurboQuantLayer(DynamicLayer):
    """
    DynamicLayer replacement that stores KV states with TurboQuant compression.

    Per-channel outlier-aware quantization (Section 4.3):
      - Each head_dim-length vector is split by channel index into outlier
        and regular subsets.
      - Outlier channels (highest RMS magnitude) get `outlier_bits` precision.
      - Regular channels get `bit_width` precision.
      - Two independent TurboQuant instances with matching sub-dimensions.

    Supported fractional effective bit-widths:
      - 2.5-bit: head_dim=128, 32 outlier channels at 3b, 96 regular at 2b
      - 3.5-bit: head_dim=128, 32 outlier channels at 4b, 96 regular at 3b
      - Arbitrary configurations for head_dim=64 etc.
    """

    def __init__(self, head_dim: int, bit_width: int, num_outlier_channels: int = 0,
                 outlier_bits: int = 0, device: torch.device | None = None,
                 use_packing: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.num_outlier_channels = num_outlier_channels
        self.outlier_bits = outlier_bits
        self.use_packing = use_packing

        dev = device or torch.device("cpu")

        self.regular_dim = head_dim - num_outlier_channels if num_outlier_channels > 0 else head_dim
        self.outlier_dim = num_outlier_channels

        self.quantizer = TurboQuantMSE(TurboQuantConfig(
            bit_width=bit_width, head_dim=self.regular_dim, device=dev,
        ))

        if num_outlier_channels > 0 and outlier_bits > bit_width:
            self.outlier_quantizer = TurboQuantMSE(TurboQuantConfig(
                bit_width=outlier_bits, head_dim=self.outlier_dim, device=dev,
                rotation_seed=43,
            ))
        else:
            self.outlier_quantizer = None
            self.regular_dim = head_dim

        self._key_data: list[dict] = []
        self._val_data: list[dict] = []
        self._cached_keys: torch.Tensor | None = None
        self._cached_values: torch.Tensor | None = None
        self._cache_dirty = True
        self._channel_mask: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

        if self.outlier_quantizer is not None and self._channel_mask is None:
            self._channel_mask = detect_outlier_channels(
                key_states, self.num_outlier_channels
            )

    def _quantize_tensor(self, x: torch.Tensor) -> dict:
        """
        Quantize [batch, heads, seq, head_dim] with per-channel outlier split.

        Each vector is split by channel index: outlier channels are quantized
        with the outlier quantizer, regular channels with the base quantizer.
        Both quantizers operate on their respective sub-dimensions independently.
        """
        shape = x.shape
        batch, heads, seq, dim = shape

        if self.outlier_quantizer is not None and self._channel_mask is not None:
            regular_mask = ~self._channel_mask
            x_f = x.float()

            regular = x_f[..., regular_mask]  # (B, H, S, regular_dim)
            outlier = x_f[..., self._channel_mask]  # (B, H, S, outlier_dim)

            r_flat = regular.reshape(-1, self.regular_dim)
            r_norms = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            r_idx = self.quantizer.quantize(r_flat / r_norms)

            o_flat = outlier.reshape(-1, self.outlier_dim)
            o_norms = o_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            o_idx = self.outlier_quantizer.quantize(o_flat / o_norms)

            if self.use_packing:
                r_idx = pack_indices(r_idx.flatten(), self.bit_width)
                o_idx = pack_indices(o_idx.flatten(), self.outlier_bits)

            return {
                'regular_idx': r_idx, 'regular_norms': r_norms.squeeze(-1),
                'regular_numel': batch * heads * seq * self.regular_dim,
                'outlier_idx': o_idx, 'outlier_norms': o_norms.squeeze(-1),
                'outlier_numel': batch * heads * seq * self.outlier_dim,
                'full_shape': shape,
                'bhs': batch * heads * seq,
            }
        else:
            flat = x.float().reshape(-1, self.head_dim)
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            idx = self.quantizer.quantize(flat / norms)

            if self.use_packing:
                idx = pack_indices(idx.flatten(), self.bit_width)

            return {
                'idx': idx, 'norms': norms.squeeze(-1),
                'idx_numel': flat.shape[0] * self.head_dim,
                'full_shape': shape,
            }

    def _dequantize_all(self, data_list: list[dict]) -> torch.Tensor:
        if not data_list:
            return torch.tensor([], dtype=self.dtype, device=self.device)

        parts = []
        for data in data_list:
            if 'regular_idx' in data:
                r_idx = data['regular_idx']
                o_idx = data['outlier_idx']

                if self.use_packing:
                    r_idx = unpack_indices(r_idx, self.bit_width, data['regular_numel'])
                    r_idx = r_idx.reshape(-1, self.regular_dim)
                    o_idx = unpack_indices(o_idx, self.outlier_bits, data['outlier_numel'])
                    o_idx = o_idx.reshape(-1, self.outlier_dim)

                r_hat = self.quantizer.dequantize(r_idx)
                r_hat = r_hat * data['regular_norms'].unsqueeze(-1)

                o_hat = self.outlier_quantizer.dequantize(o_idx)
                o_hat = o_hat * data['outlier_norms'].unsqueeze(-1)

                shape = data['full_shape']
                result = torch.zeros(shape, dtype=torch.float32, device=self.device)
                regular_mask = ~self._channel_mask
                result[..., regular_mask] = r_hat.reshape(shape[0], shape[1], shape[2], self.regular_dim)
                result[..., self._channel_mask] = o_hat.reshape(shape[0], shape[1], shape[2], self.outlier_dim)
                parts.append(result.to(self.dtype))
            else:
                idx = data['idx']
                if self.use_packing:
                    idx = unpack_indices(idx, self.bit_width, data['idx_numel'])
                    idx = idx.reshape(-1, self.head_dim)

                flat = self.quantizer.dequantize(idx)
                flat = flat * data['norms'].unsqueeze(-1)
                parts.append(flat.reshape(data['full_shape']).to(self.dtype))

        return torch.cat(parts, dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._key_data.append(self._quantize_tensor(key_states))
        self._val_data.append(self._quantize_tensor(value_states))
        self._cache_dirty = True

        return self._get_full_keys(), self._get_full_values()

    def _get_full_keys(self) -> torch.Tensor:
        if self._cache_dirty or self._cached_keys is None:
            self._cached_keys = self._dequantize_all(self._key_data)
            self._cached_values = self._dequantize_all(self._val_data)
            self._cache_dirty = False
        return self._cached_keys

    def _get_full_values(self) -> torch.Tensor:
        if self._cache_dirty or self._cached_values is None:
            self._get_full_keys()
        return self._cached_values

    def get_seq_length(self) -> int:
        if not self.is_initialized or not self._key_data:
            return 0
        return sum(d['full_shape'][-2] for d in self._key_data)

    def get_max_cache_shape(self) -> int:
        return -1

    def get_memory_bytes(self) -> int:
        """True memory footprint including packed index storage + float32 norms."""
        total_idx_bytes = 0
        total_norm_bytes = 0
        for data in self._key_data + self._val_data:
            if 'regular_idx' in data:
                if self.use_packing:
                    total_idx_bytes += data['regular_idx'].numel()  # already packed bytes
                    total_idx_bytes += data['outlier_idx'].numel()
                else:
                    total_idx_bytes += packed_size_bytes(data['regular_idx'].numel(), self.bit_width)
                    total_idx_bytes += packed_size_bytes(data['outlier_idx'].numel(), self.outlier_bits)
                total_norm_bytes += data['regular_norms'].numel() * 4
                total_norm_bytes += data['outlier_norms'].numel() * 4
            else:
                if self.use_packing:
                    total_idx_bytes += data['idx'].numel()
                else:
                    total_idx_bytes += packed_size_bytes(data['idx'].numel(), self.bit_width)
                total_norm_bytes += data['norms'].numel() * 4
        return total_idx_bytes + total_norm_bytes

    def get_effective_bits_per_value(self) -> float:
        if self.outlier_quantizer is not None:
            return effective_bit_width(
                self.head_dim, self.num_outlier_channels,
                self.bit_width, self.outlier_bits,
            )
        return float(self.bit_width)

    @property
    def keys(self):
        return self._get_full_keys()

    @keys.setter
    def keys(self, value):
        pass

    @property
    def values(self):
        return self._get_full_values()

    @values.setter
    def values(self, value):
        pass


class TurboQuantCache(DynamicCache):
    """
    DynamicCache using TurboQuant-compressed layers.
    Drop-in replacement: pass as past_key_values to model.generate().
    """

    def __init__(self, config=None, head_dim: int = 64, bit_width: int = 3,
                 num_layers: int = 24, num_outlier_channels: int = 0,
                 outlier_bits: int = 0, device: torch.device | None = None,
                 use_packing: bool = False):
        super().__init__(config=config)
        self.head_dim = head_dim
        self.bit_width = bit_width

        self.layers = [
            TurboQuantLayer(
                head_dim=head_dim,
                bit_width=bit_width,
                num_outlier_channels=num_outlier_channels,
                outlier_bits=outlier_bits,
                device=device,
                use_packing=use_packing,
            )
            for _ in range(num_layers)
        ]

    def get_memory_bytes(self) -> int:
        return sum(
            layer.get_memory_bytes()
            for layer in self.layers
            if isinstance(layer, TurboQuantLayer)
        )

    def get_effective_bits(self) -> float:
        for layer in self.layers:
            if isinstance(layer, TurboQuantLayer):
                return layer.get_effective_bits_per_value()
        return float(self.bit_width)


class TQLayerFused(DynamicLayer):
    """
    Alternative KV cache layer that stores compressed indices without
    dequantizing for HuggingFace compatibility. Instead exposes
    get_quantized_keys() / get_quantized_values() for use with
    QuantizedAttention / fused Triton kernels.

    This layer is for speed benchmarks — it does NOT produce float key/value
    tensors for standard HuggingFace attention. Use TurboQuantLayer for
    quality evaluation.
    """

    def __init__(self, head_dim: int, bit_width: int, device: torch.device):
        super().__init__()
        self.head_dim = head_dim
        self.bit_width = bit_width
        self._dev = device

        self.quantizer = TurboQuantMSE(TurboQuantConfig(
            bit_width=bit_width, head_dim=head_dim, device=device,
        ))

        self._k_idx: list[torch.Tensor] = []
        self._k_norms: list[torch.Tensor] = []
        self._v_idx: list[torch.Tensor] = []
        self._v_norms: list[torch.Tensor] = []
        self._shapes: list[tuple] = []

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def _quantize(self, x: torch.Tensor):
        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        idx = self.quantizer.quantize(flat / norms)
        return idx, norms.squeeze(-1)

    def _dequantize(self, idx: torch.Tensor, norms: torch.Tensor, shape: tuple):
        flat = self.quantizer.dequantize(idx)
        return (flat * norms.unsqueeze(-1)).reshape(shape).to(self.dtype)

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        ki, kn = self._quantize(key_states)
        vi, vn = self._quantize(value_states)
        self._k_idx.append(ki)
        self._k_norms.append(kn)
        self._v_idx.append(vi)
        self._v_norms.append(vn)
        self._shapes.append(key_states.shape)

        keys = self._dequantize_all_keys()
        values = self._dequantize_all_values()
        return keys, values

    def _dequantize_all_keys(self):
        parts = [self._dequantize(i, n, s) for i, n, s in
                 zip(self._k_idx, self._k_norms, self._shapes)]
        return torch.cat(parts, dim=-2) if parts else torch.tensor([], dtype=self.dtype)

    def _dequantize_all_values(self):
        parts = [self._dequantize(i, n, s) for i, n, s in
                 zip(self._v_idx, self._v_norms, self._shapes)]
        return torch.cat(parts, dim=-2) if parts else torch.tensor([], dtype=self.dtype)

    def get_quantized_keys(self):
        """
        Return (K_idx, K_norms) concatenated across all stored steps.

        K_idx:   (B, H, total_seq, D) uint8
        K_norms: (B, H, total_seq) float32
        """
        if not self._k_idx:
            return None, None
        all_idx = torch.cat(self._k_idx, dim=0)
        all_norms = torch.cat(self._k_norms, dim=0)
        B, H, _, D = self._shapes[0]
        total_seq = sum(s[2] for s in self._shapes)
        return all_idx.view(B, H, total_seq, D), all_norms.view(B, H, total_seq)

    def get_quantized_values(self):
        if not self._v_idx:
            return None, None
        all_idx = torch.cat(self._v_idx, dim=0)
        all_norms = torch.cat(self._v_norms, dim=0)
        B, H, _, D = self._shapes[0]
        total_seq = sum(s[2] for s in self._shapes)
        return all_idx.view(B, H, total_seq, D), all_norms.view(B, H, total_seq)

    def get_seq_length(self):
        return sum(s[2] for s in self._shapes) if self._shapes else 0

    def get_max_cache_shape(self):
        return -1

    @property
    def keys(self):
        return self._dequantize_all_keys()

    @keys.setter
    def keys(self, value):
        pass

    @property
    def values(self):
        return self._dequantize_all_values()

    @values.setter
    def values(self, value):
        pass


def get_baseline_kv_memory(cache: DynamicCache) -> int:
    """Calculate memory used by a standard (unquantized) DynamicCache."""
    total = 0
    for layer in cache.layers:
        if hasattr(layer, 'keys') and hasattr(layer.keys, 'numel'):
            k = layer.keys
            v = layer.values
            if k.numel() > 0:
                total += k.numel() * k.element_size()
            if v.numel() > 0:
                total += v.numel() * v.element_size()
    return total
