"""
Pure Python (NumPy/PyTorch) bit packing for ARCQ codes.
Replaces the CUDA extension to avoid architecture mismatch issues.
"""
import torch
import numpy as np

def pack_codes_py(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int64 codes into uint8 bit-packed format.
    
    Args:
        codes: [out_f, in_f] int64 tensor
        bits: 2, 3, or 4
    
    Returns:
        uint8 tensor [out_f, packed_cols] where packed_cols = ceil(in_f * bits / 8)
    """
    assert codes.dim() == 2, f"codes must be 2D, got {codes.dim()}"
    assert bits in (2, 3, 4), f"bits must be 2/3/4, got {bits}"
    
    # Vectorized packing: work on CPU as numpy
    codes_np = codes.cpu().numpy().astype(np.uint32)
    out_f, in_f = codes_np.shape
    mask = (1 << bits) - 1
    
    # Compute bit offsets for each column
    bit_offsets = np.arange(in_f, dtype=np.uint32) * bits
    byte_idx = bit_offsets // 8
    shift = bit_offsets % 8
    
    total_bits = in_f * bits
    packed_cols = (total_bits + 7) // 8
    
    # Pack: low-byte contributions (all codes map to their primary byte)
    packed = np.zeros((out_f, packed_cols), dtype=np.uint8)
    values = codes_np & mask
    
    # Write low byte for each code
    for j in range(in_f):
        bi = byte_idx[j]
        sh = shift[j]
        packed[:, bi] |= (values[:, j] << sh).astype(np.uint8)
        # High byte if code crosses byte boundary
        if sh + bits > 8:
            packed[:, bi + 1] |= ((values[:, j] << sh) >> 8).astype(np.uint8)
    
    return torch.from_numpy(packed).to(dtype=torch.uint8, device=codes.device)


def unpack_codes_py(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack uint8 bit-packed tensor to int64 codes.
    
    Args:
        packed: uint8 [out_f, packed_cols]
        bits: 2, 3, or 4
        length: in_features (number of codes per row)
    
    Returns:
        int64 [out_f, length]
    """
    packed_np = packed.cpu().numpy().astype(np.uint8)
    out_f = packed_np.shape[0]
    mask = (1 << bits) - 1
    
    bit_offsets = np.arange(length, dtype=np.uint32) * bits
    byte_idx = bit_offsets // 8
    shift = bit_offsets % 8
    
    codes = np.zeros((out_f, length), dtype=np.int64)
    for j in range(length):
        bi = byte_idx[j]
        sh = shift[j]
        val = packed_np[:, bi].astype(np.uint32) >> sh
        if sh + bits > 8:
            val |= packed_np[:, bi + 1].astype(np.uint32) << (8 - sh)
        codes[:, j] = val & mask
    
    return torch.from_numpy(codes).to(dtype=torch.int64, device=packed.device)
