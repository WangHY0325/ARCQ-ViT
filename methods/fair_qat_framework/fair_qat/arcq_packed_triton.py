"""
Triton kernel for ARCQ weight unpack: packed uint8 → fp16 weight.
Fused: bit-extract + codebook gather + center/scale + fp16 cast.
No intermediate int32 codes buffer → ~41 MB instead of ~127 MB peak.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _unpack_arcq_weight_row(
    packed_ptr,          # uint8 [out_f, packed_cols]
    codebook_ptr,        # fp32 [levels]
    center_ptr,          # fp32 [out_f, groups]
    scale_ptr,           # fp32 [out_f, groups]
    code_sum_ptr,        # fp32 [out_f, groups]
    out_ptr,             # fp16 [out_f, in_f]
    packed_cols: tl.constexpr,
    in_f: tl.constexpr,
    groups: tl.constexpr,
    group_size: tl.constexpr,
    bits: tl.constexpr,
    levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program = one output row. Process in BLOCK_SIZE element tiles."""
    row = tl.program_id(0)

    for block_start in range(0, in_f, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_f

        # ---- Step 1: bit-extract codes from packed bytes ----
        bit_offsets = offsets * bits
        byte_idx = bit_offsets >> 3
        shift = bit_offsets & 7
        code_mask = (1 << bits) - 1

        # Load packed bytes (gather by byte_idx)
        # Triton indirect load: ptr + row*packed_cols + byte_idx
        ptr_base = packed_ptr + row * packed_cols
        packed_vals = tl.load(ptr_base + byte_idx, mask=mask, other=0).to(tl.int32)
        codes = (packed_vals >> shift) & code_mask

        # Handle cross-byte for bits=3 (when shift + 3 > 8, i.e., shift >= 6)
        if bits == 3:
            cross_mask = shift + 3 > 8
            # Triton 3.1: no tl.any — always load next byte, mask the result
            next_vals = tl.load(ptr_base + byte_idx + 1, mask=mask, other=0).to(tl.int32)
            hi_shift = 8 - shift
            hi_mask_bits = (1 << (shift + 3 - 8)) - 1
            hi_bits = (next_vals & hi_mask_bits) << hi_shift
            codes = tl.where(cross_mask & mask, codes | hi_bits, codes)

        # Clamp codes to valid range
        codes = codes & code_mask

        # ---- Step 2: codebook lookup ----
        cb_vals = tl.load(codebook_ptr + codes, mask=mask, other=0.0)

        # ---- Step 3: apply center/scale with code_sum correction ----
        g_idx = offsets // group_size
        meta_idx = row * groups + g_idx

        c = tl.load(center_ptr + meta_idx, mask=mask, other=0.0)
        s = tl.load(scale_ptr + meta_idx, mask=mask, other=1.0)
        cs = tl.load(code_sum_ptr + meta_idx, mask=mask, other=0.0)

        # w_fp16 = (cb_val - code_sum/group_size) * scale + center
        result = (cb_vals - cs / group_size) * s + c

        # ---- Step 4: write fp16 ----
        tl.store(out_ptr + row * in_f + offsets, result.to(tl.float16), mask=mask)


def unpack_weight_triton(
    packed: torch.Tensor,          # uint8 [out_f, packed_cols]
    codebook: torch.Tensor,         # fp32 [levels]
    center: torch.Tensor,           # fp32 [out_f, groups]
    scale: torch.Tensor,            # fp32 [out_f, groups]
    code_sum: torch.Tensor,         # fp32 [out_f, groups]
    bits: int,
    group_size: int,
    out: torch.Tensor | None = None,  # optional pre-allocated fp16 buffer
) -> torch.Tensor:
    """Unpack ARCQ packed weight to fp16 using Triton fused kernel.
    
    Args:
        out: Optional pre-allocated fp16 buffer [out_f, in_f]. 
             If provided, writes into it (reuses memory).
    """
    device = packed.device
    out_f, packed_cols = packed.shape
    in_f = center.shape[1] * group_size  # groups * group_size = in_features
    groups = center.shape[1]
    levels = codebook.numel()

    # Ensure contiguous tensors on device
    packed = packed.contiguous()
    codebook = codebook.contiguous().float()
    center = center.contiguous().float()
    scale = scale.contiguous().float()
    code_sum = code_sum.contiguous().float()

    # Allocate or reuse fp16 output
    if out is None:
        out = torch.empty(out_f, in_f, dtype=torch.float16, device=device)

    # Grid: one program per row
    BLOCK_SIZE = 128
    grid = (out_f,)

    _unpack_arcq_weight_row[grid](
        packed, codebook, center, scale, code_sum, out,
        packed_cols=packed_cols, in_f=in_f, groups=groups,
        group_size=group_size, bits=bits, levels=levels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _dequant_activation_triton_kernel(
    x_ptr,              # fp32 [rows, cols]
    thresholds_ptr,     # fp32 [levels-1]
    codebook_ptr,       # fp32 [levels]
    out_ptr,            # fp32 [rows, cols]
    rows,
    cols: tl.constexpr,
    groups: tl.constexpr,
    group_size: tl.constexpr,
    levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program = one group. Fused quantize+dequant, zero intermediates."""
    pid = tl.program_id(0)
    total_groups = rows * groups
    if pid >= total_groups:
        return

    row = pid // groups
    g = pid % groups
    base = row * cols + g * group_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < group_size

    # Load group
    x_vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # Center
    mu = tl.sum(x_vals) / group_size

    # Scale
    residual = x_vals - mu
    var = tl.sum(residual * residual) / group_size
    s = tl.sqrt(var + 1e-6)

    # Normalize
    t = residual / s

    # Bucketize: find code index via threshold scan (levels-1 comparisons)
    code = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    for lvl in range(levels - 1):
        th = tl.load(thresholds_ptr + lvl)
        code = tl.where(t > th, lvl + 1, code)

    # Codebook lookup
    cb_vals = tl.load(codebook_ptr + code, mask=mask, other=0.0)

    # Mean-preserve correction
    code_mean = tl.sum(cb_vals) / group_size

    # Reconstruct
    x_hat = mu + s * (cb_vals - code_mean)

    # Write
    tl.store(out_ptr + base + offsets, x_hat, mask=mask)


def dequant_activation_triton(
    x: torch.Tensor,
    thresholds: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ARCQ activation quantize+dequant using Triton. Zero intermediates."""
    rows, cols = x.shape
    groups = cols // group_size
    levels = codebook.numel()
    
    x = x.contiguous()
    thresholds = thresholds.contiguous().float()
    codebook = codebook.contiguous().float()
    if out is None:
        out = torch.empty_like(x)
    else:
        if out.shape != x.shape:
            raise ValueError(f"dequant_activation_triton out shape {tuple(out.shape)} != input shape {tuple(x.shape)}")
        if out.dtype != x.dtype:
            raise ValueError(f"dequant_activation_triton out dtype {out.dtype} != input dtype {x.dtype}")
        if not out.is_contiguous():
            raise ValueError("dequant_activation_triton out must be contiguous")
    
    BLOCK_SIZE = min(128, triton.next_power_of_2(group_size))
    grid = (rows * groups,)
    
    _dequant_activation_triton_kernel[grid](
        x, thresholds, codebook, out,
        rows=rows, cols=cols, groups=groups,
        group_size=group_size, levels=levels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
