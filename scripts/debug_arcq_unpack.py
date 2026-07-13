"""
Debug: Compare ARCQ internal quantize vs manual unpack.
Runs ARCQ's own quantizer on the loaded weight and codebook,
then compares with the manual center/scale/bucketize approach.
"""
import os, sys
import torch
import numpy as np

BASE = "/gpool/home/wanghongyang/WangHY/QuEST/AAAI"
_PROJ = os.path.join(BASE, "methods", "fair_qat_framework")
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

CKPT_W4A4 = f"{BASE}/runs/cifar100_qat/pcarcq_nodc/deit_small/w4a4_w_distill/best.pt"
CKPT_W3A3 = f"{BASE}/runs/cifar100_qat/pcarcq_nodc/deit_small/w3a3/best.pt"


def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean


def main():
    sd = load_state(CKPT_W4A4)
    
    # Test layer: blocks.0.attn.qkv
    layer = "blocks.0.attn.qkv"
    w = sd[f"{layer}.linear.weight"].float()           # FP32 weight
    cb = sd[f"{layer}.weight_quant._pc_cb"].float()  # codebook
    
    print(f"Layer: {layer}")
    print(f"Weight shape: {tuple(w.shape)}, codebook: {cb.tolist()}")
    print(f"Codebook sorted: {torch.sort(cb).values.tolist()}")
    
    # Create the actual ARCQ quantizer and load its state
    from quant.dcarcq import ARCQWeightQuantizer
    
    q = ARCQWeightQuantizer(bits=4, group_size=64, freeze_codebook=True)
    
    # Load the codebook into the quantizer's buffer
    q._pc_cb.copy_(cb)
    # Also set _cached_cb to make it think codebook is already built
    q._cached_cb = cb.clone()
    q._cb_built = True
    
    # Run quantizer on the weight
    with torch.no_grad():
        w_q_arcq = q(w)
    
    # Now my manual approach
    gs = 64
    out_f, in_f = w.shape
    usable = (in_f // gs) * gs
    
    w2 = w.reshape(-1, in_f)[:, :usable]
    rows, C = w2.shape
    G = C // gs
    w3 = w2.reshape(rows, G, gs)
    
    # Match ARCQ exactly: per-row per-group
    center_m = w3.mean(dim=-1, keepdim=True)         # [rows, G, 1]
    residual_m = w3 - center_m
    scale_m = residual_m.square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
    t_m = residual_m / scale_m                        # [rows, G, gs]
    
    # Bucketize with codebook (UNSORTED codebook, matching ARCQ)
    cb_sorted = torch.sort(cb).values
    thresholds = (cb_sorted[:-1] + cb_sorted[1:]) / 2.0
    codes_m = torch.bucketize(t_m, thresholds)        # [rows, G, gs]
    t_hat_m = cb_sorted[codes_m]
    
    # Mean preserve
    t_hat_m = t_hat_m - t_hat_m.mean(dim=-1, keepdim=True)
    
    w_recon_m = (center_m + scale_m * t_hat_m).reshape(rows, C)
    
    # Full shape
    if in_f > usable:
        pad_w = w.reshape(-1, in_f)[:, usable:]
        w_recon_full = torch.cat([w_recon_m, pad_w], dim=-1).reshape(out_f, in_f)
    else:
        w_recon_full = w_recon_m.reshape(out_f, in_f)
    
    err_vs_arcq = (w_q_arcq - w_recon_full).abs().max().item()
    err_vs_original = (w - w_recon_full).abs().max().item()
    
    print(f"\nManual vs ARCQ quantizer max error: {err_vs_arcq:.6e}")
    print(f"Manual vs original weight max error:  {err_vs_original:.6e}")
    
    # Also test: manual with SAME center/scale as ARCQ
    # The ARCQ quantizer computes center/scale internally during forward
    # We computed them independently above. They should match.
    # Let's verify by extracting center/scale from a special ARCQ forward
    
    print(f"\nARCQ quantized weight shape: {tuple(w_q_arcq.shape)}")
    print(f"Manual reconstructed shape:   {tuple(w_recon_full.shape)}")
    
    if err_vs_arcq < 1e-5:
        print("\n✓ Manual unpack matches ARCQ quantizer (using non-PC for test)")
    else:
        print(f"\n✗ MISMATCH: diff = {err_vs_arcq:.6e}")

if __name__ == "__main__":
    main()
