"""
KV Cache patching for HuggingFace models using TurboQuant.

Implements the paper's KV cache quantization strategy (Section 4.2-4.3):
  - Uses TurboQuant_mse for KV cache (not _prod, which is for vector search)
  - Outlier channel separation: top-k outlier channels get higher bit-width
  - Quantizes both prefill and generation tokens

KV cache tensor shape: [batch_size, num_heads, seq_len, head_dim]
"""

import torch
from typing import Any
from transformers.cache_utils import DynamicCache, DynamicLayer
from turboquant import TurboQuantMSE, TurboQuantConfig


class TurboQuantLayer(DynamicLayer):
    """
    DynamicLayer replacement that stores KV states with TurboQuant compression.

    Uses TurboQuant_mse with optional outlier channel separation:
    - Outlier channels (highest variance) get `outlier_bits` precision
    - Regular channels get `regular_bits` precision
    - Effective bit-width = weighted average

    Per the paper Section 4.3: "32 outlier channels are quantized at 3 bits,
    while the remaining 96 channels use 2 bits" for the 2.5-bit configuration.
    """

    def __init__(self, head_dim: int, bit_width: int, num_outlier_channels: int = 0,
                 outlier_bits: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.num_outlier_channels = num_outlier_channels
        self.outlier_bits = outlier_bits

        config = TurboQuantConfig(
            bit_width=bit_width,
            head_dim=head_dim,
            device=torch.device("cpu"),
        )
        self.quantizer = TurboQuantMSE(config)

        if num_outlier_channels > 0 and outlier_bits > bit_width:
            outlier_config = TurboQuantConfig(
                bit_width=outlier_bits,
                head_dim=head_dim,
                device=torch.device("cpu"),
                rotation_seed=43,
            )
            self.outlier_quantizer = TurboQuantMSE(outlier_config)
        else:
            self.outlier_quantizer = None

        self._key_data = []
        self._val_data = []

        self._cached_keys = None
        self._cached_values = None
        self._cache_dirty = True

        self._outlier_mask = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

        if self.outlier_quantizer is not None and self._outlier_mask is None:
            # Detect outlier heads based on key magnitude variance
            # Shape: [batch, num_heads, seq_len, head_dim]
            var_per_head = key_states.float().var(dim=-1).mean(dim=(0, 2))  # [num_heads]
            _, top_indices = var_per_head.topk(min(self.num_outlier_channels, var_per_head.shape[0]))
            self._outlier_mask = torch.zeros(var_per_head.shape[0], dtype=torch.bool)
            self._outlier_mask[top_indices] = True

    def _quantize_tensor(self, x: torch.Tensor):
        """
        Quantize [batch, heads, new_tokens, head_dim] tensor.
        Returns (indices, norms, shape, outlier_indices) where outlier_indices
        is None if no outlier separation is used.
        """
        shape = x.shape
        batch, heads, seq, dim = shape
        flat = x.float().reshape(-1, self.head_dim)  # [batch*heads*seq, head_dim]

        norms = flat.norm(dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-10)
        normalized = flat / safe_norms

        if self.outlier_quantizer is not None and self._outlier_mask is not None:
            x_reshaped = x.float()  # [batch, heads, seq, dim]
            regular_mask = ~self._outlier_mask

            regular = x_reshaped[:, regular_mask]  # [batch, n_regular, seq, dim]
            outlier = x_reshaped[:, self._outlier_mask]  # [batch, n_outlier, seq, dim]

            r_flat = regular.reshape(-1, dim)
            r_norms = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            r_norm_flat = r_flat / r_norms
            r_idx = self.quantizer.quantize(r_norm_flat)

            o_flat = outlier.reshape(-1, dim)
            o_norms = o_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            o_norm_flat = o_flat / o_norms
            o_idx = self.outlier_quantizer.quantize(o_norm_flat)

            return {
                'regular_idx': r_idx, 'regular_norms': r_norms.squeeze(-1),
                'regular_shape': regular.shape,
                'outlier_idx': o_idx, 'outlier_norms': o_norms.squeeze(-1),
                'outlier_shape': outlier.shape,
                'full_shape': shape,
            }
        else:
            idx = self.quantizer.quantize(normalized)
            return {
                'idx': idx, 'norms': norms.squeeze(-1),
                'full_shape': shape,
            }

    def _dequantize_all(self, data_list):
        """Dequantize and concatenate all stored chunks."""
        if not data_list:
            return torch.tensor([], dtype=self.dtype, device=self.device)

        parts = []
        for data in data_list:
            if 'regular_idx' in data:
                r_flat = self.quantizer.dequantize(data['regular_idx'])
                r_flat = r_flat * data['regular_norms'].unsqueeze(-1)
                regular = r_flat.reshape(data['regular_shape'])

                o_flat = self.outlier_quantizer.dequantize(data['outlier_idx'])
                o_flat = o_flat * data['outlier_norms'].unsqueeze(-1)
                outlier = o_flat.reshape(data['outlier_shape'])

                full_shape = data['full_shape']
                result = torch.zeros(full_shape, dtype=torch.float32)
                regular_mask = ~self._outlier_mask
                result[:, regular_mask] = regular
                result[:, self._outlier_mask] = outlier
                parts.append(result.to(self.dtype))
            else:
                flat = self.quantizer.dequantize(data['idx'])
                flat = flat * data['norms'].unsqueeze(-1)
                part = flat.reshape(data['full_shape']).to(self.dtype)
                parts.append(part)

        return torch.cat(parts, dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        k_data = self._quantize_tensor(key_states)
        v_data = self._quantize_tensor(value_states)
        self._key_data.append(k_data)
        self._val_data.append(v_data)
        self._cache_dirty = True

        keys = self._get_full_keys()
        values = self._get_full_values()
        return keys, values

    def _get_full_keys(self):
        if self._cache_dirty or self._cached_keys is None:
            self._cached_keys = self._dequantize_all(self._key_data)
            self._cached_values = self._dequantize_all(self._val_data)
            self._cache_dirty = False
        return self._cached_keys

    def _get_full_values(self):
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
        """
        Calculate effective memory for quantized storage.
        Counts actual bits: b bits per index value, plus norms overhead.
        """
        total_bits = 0
        total_norm_bytes = 0

        for data in self._key_data + self._val_data:
            if 'regular_idx' in data:
                total_bits += data['regular_idx'].numel() * self.bit_width
                total_bits += data['outlier_idx'].numel() * self.outlier_bits
                total_norm_bytes += data['regular_norms'].numel() * 4
                total_norm_bytes += data['outlier_norms'].numel() * 4
            else:
                total_bits += data['idx'].numel() * self.bit_width
                total_norm_bytes += data['norms'].numel() * 4

        return total_bits // 8 + total_norm_bytes

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
                 outlier_bits: int = 0):
        super().__init__(config=config)
        self.head_dim = head_dim
        self.bit_width = bit_width

        self.layers = [
            TurboQuantLayer(
                head_dim=head_dim,
                bit_width=bit_width,
                num_outlier_channels=num_outlier_channels,
                outlier_bits=outlier_bits,
            )
            for _ in range(num_layers)
        ]

    def get_memory_bytes(self) -> int:
        """Total effective memory used by all quantized layers."""
        return sum(
            layer.get_memory_bytes()
            for layer in self.layers
            if isinstance(layer, TurboQuantLayer)
        )

    def get_effective_bits(self) -> float:
        """Calculate effective bits per value across all layers."""
        total_elements = 0
        total_bits = 0
        for layer in self.layers:
            if isinstance(layer, TurboQuantLayer):
                for data in layer._key_data + layer._val_data:
                    if 'regular_idx' in data:
                        total_elements += data['regular_idx'].numel() + data['outlier_idx'].numel()
                        total_bits += data['regular_idx'].numel() * layer.bit_width
                        total_bits += data['outlier_idx'].numel() * layer.outlier_bits
                    else:
                        total_elements += data['idx'].numel()
                        total_bits += data['idx'].numel() * layer.bit_width
        return total_bits / total_elements if total_elements > 0 else 0


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
