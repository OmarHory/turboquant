"""
Bit-packing for sub-byte TurboQuant indices.

TurboQuant produces b-bit codebook indices (b in {2, 3, 4}) stored as uint8,
wasting 8-b bits per element. This module packs indices tightly to achieve
the theoretical memory footprint.

Packing schemes:
  b=4: 2 values per byte.  val[0] in bits [0:4], val[1] in bits [4:8].
  b=2: 4 values per byte.  val[i] in bits [2i : 2i+2].
  b=3: 8 values into 3 bytes (24-bit group). Each value occupies 3 consecutive
       bits across the group, packed LSB-first. The last group is zero-padded
       if num_elements is not a multiple of 8.
  b=8: passthrough — no packing needed.
"""

import math
import torch


def packed_size_bytes(num_elements: int, bit_width: int) -> int:
    """Number of bytes required to store num_elements values at the given bit width."""
    if bit_width == 8:
        return num_elements
    return math.ceil(num_elements * bit_width / 8)


def compression_ratio(bit_width: int, baseline_bits: int = 16) -> float:
    """Theoretical compression ratio vs a baseline representation (default FP16)."""
    return baseline_bits / bit_width


def pack_indices(idx: torch.Tensor, bit_width: int) -> torch.Tensor:
    """
    Pack b-bit index values (stored as uint8) into tightly packed uint8 bytes.

    Args:
        idx: Flat uint8 tensor of codebook indices, each in [0, 2^bit_width).
        bit_width: Bits per index (2, 3, 4, or 8).

    Returns:
        Packed uint8 tensor of length packed_size_bytes(len(idx), bit_width).
    """
    if bit_width == 8:
        return idx.clone()

    flat = idx.flatten().to(torch.uint8)
    n = flat.numel()
    device = flat.device

    if bit_width == 4:
        padded_n = n + (n % 2)
        if padded_n != n:
            flat = torch.cat([flat, torch.zeros(padded_n - n, dtype=torch.uint8, device=device)])
        lo = flat[0::2]
        hi = flat[1::2]
        return (lo | (hi << 4)).to(torch.uint8)

    if bit_width == 2:
        remainder = n % 4
        if remainder:
            flat = torch.cat([flat, torch.zeros(4 - remainder, dtype=torch.uint8, device=device)])
        v0 = flat[0::4]
        v1 = flat[1::4]
        v2 = flat[2::4]
        v3 = flat[3::4]
        return (v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)).to(torch.uint8)

    if bit_width == 3:
        # 8 values -> 24 bits -> 3 bytes per group
        group_size = 8
        remainder = n % group_size
        if remainder:
            flat = torch.cat([flat, torch.zeros(group_size - remainder, dtype=torch.uint8, device=device)])

        groups = flat.reshape(-1, group_size).to(torch.int32)
        # Build a 24-bit integer per group, then split into 3 bytes
        accum = torch.zeros(groups.shape[0], dtype=torch.int32, device=device)
        for i in range(group_size):
            accum |= groups[:, i] << (3 * i)

        b0 = (accum & 0xFF).to(torch.uint8)
        b1 = ((accum >> 8) & 0xFF).to(torch.uint8)
        b2 = ((accum >> 16) & 0xFF).to(torch.uint8)

        return torch.stack([b0, b1, b2], dim=1).flatten()

    raise ValueError(f"Unsupported bit_width={bit_width}. Must be 2, 3, 4, or 8.")


def unpack_indices(packed: torch.Tensor, bit_width: int, num_elements: int) -> torch.Tensor:
    """
    Unpack tightly packed uint8 bytes back to individual b-bit values as uint8.

    Args:
        packed: Packed uint8 tensor from pack_indices.
        bit_width: Bits per index (2, 3, 4, or 8).
        num_elements: Original number of elements before packing.

    Returns:
        uint8 tensor of length num_elements with values in [0, 2^bit_width).
    """
    if bit_width == 8:
        return packed[:num_elements].clone()

    device = packed.device

    if bit_width == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        return torch.stack([lo, hi], dim=1).flatten()[:num_elements].to(torch.uint8)

    if bit_width == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=1).flatten()[:num_elements].to(torch.uint8)

    if bit_width == 3:
        group_bytes = packed.reshape(-1, 3).to(torch.int32)
        accum = group_bytes[:, 0] | (group_bytes[:, 1] << 8) | (group_bytes[:, 2] << 16)

        mask = 0x07
        vals = torch.stack([(accum >> (3 * i)) & mask for i in range(8)], dim=1)
        return vals.flatten()[:num_elements].to(torch.uint8)

    raise ValueError(f"Unsupported bit_width={bit_width}. Must be 2, 3, 4, or 8.")


if __name__ == "__main__":
    print("=== Bit-Packing Round-Trip Tests ===\n")

    torch.manual_seed(0)
    test_sizes = [1, 7, 8, 15, 16, 100, 1024, 4096]

    all_passed = True
    for b in [2, 3, 4, 8]:
        max_val = 2 ** b
        print(f"--- bit_width={b} (max index value={max_val - 1}) ---")

        for n in test_sizes:
            idx = torch.randint(0, max_val, (n,), dtype=torch.uint8)
            packed = pack_indices(idx, b)
            unpacked = unpack_indices(packed, b, n)

            ok = torch.equal(idx, unpacked)
            expected_bytes = packed_size_bytes(n, b)
            actual_bytes = packed.numel()

            status = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False
            print(f"  n={n:5d}  packed={actual_bytes:5d} bytes  expected={expected_bytes:5d}  {status}")

        ratio = compression_ratio(b)
        print(f"  Compression vs FP16: {ratio:.1f}x\n")

    if torch.cuda.is_available():
        print("--- CUDA round-trip ---")
        device = torch.device("cuda")
        for b in [2, 3, 4]:
            idx = torch.randint(0, 2 ** b, (4096,), dtype=torch.uint8, device=device)
            packed = pack_indices(idx, b)
            unpacked = unpack_indices(packed, b, 4096)
            ok = torch.equal(idx, unpacked)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False
            print(f"  bit_width={b} on CUDA: {status}")
        print()

    print("ALL PASSED" if all_passed else "SOME TESTS FAILED")
