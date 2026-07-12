"""
DC-DDFZ Quantizer for ViT — v2 (vectorized, shared codebook).

Theory:
- Each group has its own center p and RMS scale s.
- All groups in a tensor/layer share ONE DDFZ codebook derived from the
  global residual distribution (sampled for speed).
- Encoding: vectorized bucketize on all groups at once.
- No per-group Python loop.

Author: Generated for QuEST ViT experiments.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  Moment-matched lookup tables (precomputed globally once)
# ============================================================================

_N_BETA = 512
_beta_grid: torch.Tensor = None
_kappa_grid: torch.Tensor = None
_tau_table: dict = {}  # {bits: tensor}
_SUPPORTED_TAU_BITS = [1, 2, 3, 4, 5, 6, 8]
_TAU_N_U = 4096


def _init_moment_tables():
    global _beta_grid, _kappa_grid, _tau_table
    if _beta_grid is not None:
        return

    _beta_grid = torch.linspace(0.75, 8.0, _N_BETA)
    _kappa_grid = torch.empty(_N_BETA)
    for i in range(_N_BETA):
        b = float(_beta_grid[i])
        _kappa_grid[i] = math.exp(
            math.lgamma(5.0 / b) + math.lgamma(1.0 / b) - 2.0 * math.lgamma(3.0 / b)
        )

    for bi in _SUPPORTED_TAU_BITS:
        L_val = 2 ** bi
        eps_b_val = 1.0 / (2.0 * L_val * L_val)
        target = 1.0 - eps_b_val
        taus = torch.full((_N_BETA,), 8.0)
        for i in range(_N_BETA):
            b = float(_beta_grid[i])
            alpha = math.sqrt(math.exp(
                math.lgamma(1.0 / b) - math.lgamma(3.0 / b)
            ))
            logZ = math.log(b) - math.log(alpha) - math.lgamma(1.0 / b)
            Z = math.exp(logZ)
            u_max = 8.0
            for _ in range(6):
                u = torch.linspace(0.0, u_max, _TAU_N_U)
                pdf = (b / Z) * torch.exp(-((u / alpha) ** b))
                cdf = torch.cumsum(pdf, dim=0)
                cdf = cdf / cdf[-1]
                if cdf[-1] >= target:
                    break
                u_max *= 2.0
            mask = cdf >= target
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0][0]
                taus[i] = u[idx]
        _tau_table[bi] = taus


def _lookup_beta(kurt: float) -> float:
    _init_moment_tables()
    k = float(kurt)
    kclamped = max(float(_kappa_grid.min()), min(float(_kappa_grid.max()), k))
    idx = (torch.abs(_kappa_grid - kclamped)).argmin().item()
    return float(_beta_grid[idx])


def _lookup_tau(beta: float, bits: int) -> float:
    _init_moment_tables()
    bi = int(bits)
    if bi not in _tau_table:
        return 2.5
    idx = (torch.abs(_beta_grid - float(beta))).argmin().item()
    tau_val = float(_tau_table[bi][idx])
    return max(0.5, min(8.0, tau_val))


# ============================================================================
#  Tail-aware K
# ============================================================================

def _compute_tail_k(x: torch.Tensor, tau: float) -> float:
    if x.numel() < 2:
        return tau
    x_mean = x.mean().item()
    x_std = x.std().item()
    candidate = x_mean + 3.0 * x_std
    return min(candidate, tau * 1.2)


# ============================================================================
#  DDFZ Quantizer — v2 (vectorized, shared codebook)
# ============================================================================

class DDFZQuantizer(nn.Module):
    """Distribution-Driven Free-trainable Zero-point Quantizer.

    Applies group-wise DDFZ quantization to the last dimension.

    V2 design:
      - Per-group center/scale (p, s).
      - ONE codebook shared by all groups in this tensor.
      - Vectorized bucketize.
      - Optional stats sampling for speed.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        freeze_codebook: bool = False,
        stats_sample_groups: int = 4096,
        mean_preserve: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.freeze_codebook = bool(freeze_codebook)
        self.stats_sample_groups = int(stats_sample_groups)
        self.mean_preserve = bool(mean_preserve)
        self.eps = float(eps)

        # Cached codebook + thresholds
        self.register_buffer("_cached_cb", None)
        self._cb_built = False

        # Diagnostics
        self.last_stats = {}
        self._step_count = 0
        self._log_every = 50  # log stats every N steps

    # ------------------------------------------------------------------
    #  One codebook from sampled normalized residual
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_codebook_from_flat(self, t_flat: torch.Tensor) -> torch.Tensor:
        """Build a DDFZ codebook from the full sampled residual distribution.

        Args:
            t_flat: 1D tensor of normalized residuals (sampled from all groups).

        Returns:
            cb: shape [2^bits], sorted ascending, on same device/dtype.
        """
        L = 2 ** self.bits
        if L < 2:
            return torch.tensor([0.0], device=t_flat.device, dtype=torch.float32)

        device = t_flat.device
        n = t_flat.numel()

        # ---- Moments ----
        t2 = t_flat * t_flat
        m2 = t2.mean().item()
        m4 = (t2 * t2).mean().item()
        kurt = m4 / max(m2 * m2, self.eps)

        beta = _lookup_beta(kurt)
        tau = _lookup_tau(beta, self.bits)
        gamma = max(0.5, min(2.0, 2.0 / beta))

        # ---- Clip to [-tau, tau] for main stats ----
        if self.bits >= 3:
            t_stat = t_flat.clamp(-tau, tau)
        else:
            t_stat = t_flat  # use all stats for low-bit

        # ---- Positive / negative statistics ----
        pos_mask = t_stat > 1e-8
        neg_mask = t_stat < -1e-8

        p_pos = pos_mask.float().mean().item()
        p_neg = neg_mask.float().mean().item()

        sigma_pos = t_stat[pos_mask].std().item() if pos_mask.any() else 1.0
        sigma_neg = t_stat[neg_mask].abs().std().item() if neg_mask.any() else 1.0

        # ---- Distortion mass allocation ----
        mp = (max(p_pos, 1e-8)) ** (1.0 / 3.0) * sigma_pos ** (2.0 / 3.0)
        mn = (max(p_neg, 1e-8)) ** (1.0 / 3.0) * sigma_neg ** (2.0 / 3.0)
        total_m = mp + mn + 1e-8

        # ---- Allocate levels ----
        if self.bits == 1:
            n_neg, n_pos = 1, 1
        elif self.bits == 2:
            # Low-bit: prefer zero-free or scarce, use all 4 levels
            n_pos = max(1, round(L * mp / total_m))
            n_neg = L - n_pos
            if n_neg < 1:
                n_neg = 1
                n_pos = L - n_neg
        else:
            # Bits >= 3: reserve one zero code
            n_pos = max(1, round(L * mp / total_m))
            n_neg = L - n_pos - 1
            if n_neg < 1:
                n_neg = 1
                n_pos = L - n_neg - 1

        # ---- Tail-aware K factors ----
        K_pos = _compute_tail_k(t_stat[pos_mask], tau) if pos_mask.any() else tau
        K_neg = _compute_tail_k(-t_stat[neg_mask], tau) if neg_mask.any() else tau

        # ---- Build codebook ----
        t_dtype = t_flat.dtype

        if self.bits == 1:
            cb = torch.tensor([-K_neg, K_pos], device=device, dtype=torch.float32)
            cb = cb.sort()[0]
        elif self.bits == 2:
            # Zero-free scarce: no explicit zero code
            pos_levels = torch.linspace(0.05, K_pos, n_pos, device=device, dtype=torch.float32)
            neg_levels = torch.linspace(0.05, K_neg, n_neg, device=device, dtype=torch.float32)
            neg_levels = -neg_levels.flip(0)
            cb = torch.cat([neg_levels, pos_levels])
            # Pad or trim to exact length
            if cb.numel() < L:
                delta = (K_pos - 0.05) / max(n_pos - 1, 1)
                extra = torch.linspace(K_pos + delta, K_pos + delta * (L - cb.numel()),
                                       L - cb.numel(), device=device, dtype=torch.float32)
                cb = torch.cat([cb, extra])
            cb = cb[:L].sort()[0]
        else:
            # Bits >= 3: negative codes, zero, positive codes
            pos_levels = torch.linspace(0.0, K_pos, n_pos + 1, device=device, dtype=torch.float32)[1:]
            neg_levels = torch.linspace(0.0, K_neg, n_neg + 1, device=device, dtype=torch.float32)[1:]
            neg_levels = -neg_levels.flip(0)
            zero = torch.zeros(1, device=device, dtype=torch.float32)
            cb = torch.cat([neg_levels, zero, pos_levels])
            if cb.numel() > L:
                cb = cb[:L]
            elif cb.numel() < L:
                pad = L - cb.numel()
                half = pad // 2
                # Extend negative side
                if n_neg > 1 and half > 0:
                    d_neg = (neg_levels[-1] - neg_levels[0]) / max(n_neg - 1, 1)
                    ex = torch.linspace(neg_levels[0] - d_neg * half,
                                        neg_levels[0] - self.eps, half,
                                        device=device, dtype=torch.float32)
                    cb = torch.cat([ex, cb])
                rest = L - cb.numel()
                if rest > 0:
                    d_pos = (pos_levels[-1] - pos_levels[0]) / max(n_pos - 1, 1)
                    ex = torch.linspace(pos_levels[-1] + self.eps,
                                        pos_levels[-1] + d_pos * rest, rest,
                                        device=device, dtype=torch.float32)
                    cb = torch.cat([cb, ex])
                cb = cb[:L]
            cb = cb.sort()[0]

        # Diagnostics
        self.last_stats.update({
            "beta": round(beta, 3),
            "tau": round(tau, 3),
            "gamma": round(gamma, 3),
            "K_neg": round(K_neg, 3),
            "K_pos": round(K_pos, 3),
            "n_neg": n_neg,
            "n_pos": n_pos,
            "codebook_min": round(float(cb.min()), 4),
            "codebook_max": round(float(cb.max()), 4),
            "p_pos": round(p_pos, 4),
            "p_neg": round(p_neg, 4),
        })

        return cb.to(dtype=t_dtype)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 8:
            return x

        original_shape = x.shape
        C = original_shape[-1]

        # ---- Reshape + pad ----
        x2 = x.reshape(-1, C)  # [rows, C]
        if C % self.group_size != 0:
            pad = self.group_size - (C % self.group_size)
            x2 = F.pad(x2, (0, pad))
        else:
            pad = 0

        rows, C_padded = x2.shape
        n_groups = C_padded // self.group_size
        x3 = x2.reshape(rows, n_groups, self.group_size)  # [rows, G, gs]

        # ---- Per-group center & scale ----
        center = x3.mean(dim=-1, keepdim=True)                  # [rows, G, 1]
        residual = x3 - center
        scale = residual.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)  # [rows, G, 1]
        t = residual / scale                                      # normalized residual

        # ---- Build ONE shared codebook ----
        t_groups = t.reshape(-1, self.group_size)       # [rows*G, gs]
        total_groups = t_groups.shape[0]
        t_sampled = t_groups  # placeholder for frozen path

        if self.freeze_codebook and self._cb_built:
            cb = self._cached_cb
        else:
            with torch.no_grad():
                if total_groups > self.stats_sample_groups:
                    step = max(1, total_groups // self.stats_sample_groups)
                    t_sampled = t_groups[::step]
                else:
                    t_sampled = t_groups

                cb = self._build_codebook_from_flat(t_sampled.flatten())

            if self.freeze_codebook:
                self._cached_cb = cb
                self._cb_built = True

        # ---- Vectorized encode/decode ----
        thresholds = (cb[:-1] + cb[1:]) / 2.0
        codes = torch.bucketize(t, thresholds)                   # [rows, G, gs]
        t_hat = cb[codes]                                        # vectorized lookup

        # ---- Group-level mean preservation ----
        if self.mean_preserve:
            t_hat = t_hat - t_hat.mean(dim=-1, keepdim=True)

        # ---- Reconstruct ----
        x_hat = center + scale * t_hat                           # [rows, G, gs]

        # ---- Unpad + reshape ----
        x_hat_flat = x_hat.reshape(rows, C_padded)
        if pad > 0:
            x_hat_flat = x_hat_flat[:, :C]
        x_hat = x_hat_flat.reshape(original_shape)

        # ---- STE ----
        y = x + (x_hat - x).detach()

        # ---- Periodic diagnostics ----
        self._step_count += 1
        if self._step_count % self._log_every == 0:
            self.last_stats.update({
                "bits": self.bits,
                "num_groups": total_groups,
                "stat_groups": t_sampled.shape[0],
                "code_used": 2 ** self.bits if self.bits < 8 else 0,
            })

        return y


# ============================================================================
#  Specialized subclasses
# ============================================================================

class DDFZWeightQuantizer(DDFZQuantizer):
    """Weight DDFZ quantizer — codebook frozen after first forward."""

    def __init__(self, bits: int = 4, group_size: int = 64,
                 stats_sample_groups: int = 4096, **kwargs):
        super().__init__(
            bits=bits,
            group_size=group_size,
            freeze_codebook=True,
            stats_sample_groups=stats_sample_groups,
            **kwargs,
        )


class DDFZActQuantizer(DDFZQuantizer):
    """Activation DDFZ quantizer — codebook rebuilt each forward."""

    def __init__(self, bits: int = 4, group_size: int = 64,
                 stats_sample_groups: int = 4096, **kwargs):
        super().__init__(
            bits=bits,
            group_size=group_size,
            freeze_codebook=False,
            stats_sample_groups=stats_sample_groups,
            **kwargs,
        )


# ============================================================================
#  PC-DDFZ Quantizer — Phase-Compiled codebook
# ============================================================================

class DDFZPCQuantizer(DDFZQuantizer):
    """
    Phase-Compiled DDFZ for ViT.

    Same quantization formula as DDFZQuantizer:
        x -> group mean p
        r = x - p
        s = RMS(r)
        t = r / s
        t_hat = codebook_quant(t)
        y = p + s * t_hat

    Difference:
        Codebook is compiled only at selected local forward steps.
        Between compile steps, reuse cached codebook.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        freeze_codebook: bool = False,
        stats_sample_groups: int = 4096,
        mean_preserve: bool = True,
        eps: float = 1e-6,
        phase_compile: bool = True,
        compile_steps=None,
        pc_log_every: int = 200,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            freeze_codebook=freeze_codebook,
            stats_sample_groups=stats_sample_groups,
            mean_preserve=mean_preserve,
            eps=eps,
        )
        self.phase_compile = bool(phase_compile)
        self.compile_steps = self._parse_compile_steps(compile_steps)
        self.pc_log_every = int(pc_log_every)

        self._pc_ready = False
        self._pc_step = 0
        self._pc_compile_count = 0
        self._pc_last_compile_step = -1

        self.register_buffer("_pc_cb", torch.empty(0))
        self.register_buffer("_pc_thresholds", torch.empty(0))

    def _parse_compile_steps(self, compile_steps):
        if compile_steps is None or (
            isinstance(compile_steps, str)
            and compile_steps.strip().lower() == "auto"
        ):
            env_steps = os.environ.get("DDFZ_PC_COMPILE_STEPS", "").strip()
            if env_steps:
                return {
                    int(x.strip())
                    for x in env_steps.split(",")
                    if x.strip()
                }
            return {0, 250, 750, 1500, 3000, 6000, 10000}
        if isinstance(compile_steps, str):
            return {int(x.strip()) for x in compile_steps.split(",") if x.strip()}
        if isinstance(compile_steps, (list, tuple, set)):
            return {int(x) for x in compile_steps}
        raise TypeError(f"Unsupported compile_steps type: {type(compile_steps)}")

    @torch.no_grad()
    def _compile_codebook(self, t_groups: torch.Tensor):
        total_groups = t_groups.shape[0]
        t_sampled = t_groups
        if total_groups > self.stats_sample_groups:
            step = max(1, total_groups // self.stats_sample_groups)
            t_sampled = t_groups[::step]

        cb = self._build_codebook_from_flat(t_sampled.flatten())
        thresholds = (cb[:-1] + cb[1:]) / 2.0

        self._pc_cb = cb.detach().clone()
        self._pc_thresholds = thresholds.detach().clone()
        self._pc_ready = True
        self._pc_compile_count += 1
        self._pc_last_compile_step = self._pc_step

        self.last_stats.update({
            "pc_enabled": True,
            "pc_step": self._pc_step,
            "pc_compile_count": self._pc_compile_count,
            "pc_last_compile_step": self._pc_last_compile_step,
            "num_groups": int(total_groups),
            "stat_groups": int(t_sampled.shape[0]),
        })

        if self._pc_step % max(1, self.pc_log_every) == 0 or self._pc_compile_count <= 3:
            print(
                f"[VIT_PCDDFZ_COMPILE] bits={self.bits} "
                f"step={self._pc_step} count={self._pc_compile_count} "
                f"groups={total_groups} stat_groups={t_sampled.shape[0]} "
                f"policy={self.last_stats.get('n_neg', 'na')}/{self.last_stats.get('n_pos', 'na')} "
                f"beta={self.last_stats.get('beta', 'na')} "
                f"gamma={self.last_stats.get('gamma', 'na')} "
                f"K_neg={self.last_stats.get('K_neg', 'na')} "
                f"K_pos={self.last_stats.get('K_pos', 'na')}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 8:
            return x

        if not self.phase_compile:
            return super().forward(x)

        original_shape = x.shape
        C = original_shape[-1]

        x2 = x.reshape(-1, C)
        if C % self.group_size != 0:
            pad = self.group_size - (C % self.group_size)
            x2 = F.pad(x2, (0, pad))
        else:
            pad = 0

        rows, C_padded = x2.shape
        n_groups = C_padded // self.group_size
        x3 = x2.reshape(rows, n_groups, self.group_size)

        center = x3.mean(dim=-1, keepdim=True)
        residual = x3 - center
        scale = (residual.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        t = residual / scale

        t_groups = t.reshape(-1, self.group_size)

        should_compile = (not self._pc_ready) or (self._pc_step in self.compile_steps)
        if should_compile:
            with torch.no_grad():
                self._compile_codebook(t_groups)

        if not self._pc_ready or self._pc_cb.numel() == 0:
            with torch.no_grad():
                self._compile_codebook(t_groups)

        cb = self._pc_cb.to(device=x.device, dtype=t.dtype)
        thresholds = self._pc_thresholds.to(device=x.device, dtype=t.dtype)

        codes = torch.bucketize(t, thresholds)
        t_hat = cb[codes]

        if self.mean_preserve:
            t_hat = t_hat - t_hat.mean(dim=-1, keepdim=True)

        x_hat = center + scale * t_hat
        x_hat_flat = x_hat.reshape(rows, C_padded)
        if pad > 0:
            x_hat_flat = x_hat_flat[:, :C]
        x_hat_out = x_hat_flat.reshape(original_shape)

        y = x + (x_hat_out - x).detach()

        self._step_count += 1
        if self._step_count % self._log_every == 0:
            self.last_stats.update({
                "bits": self.bits,
                "pc_enabled": True,
                "pc_step": self._pc_step,
                "pc_compile_count": self._pc_compile_count,
                "num_groups": int(t_groups.shape[0]),
                "code_used": int(torch.unique(codes.detach()).numel()),
            })

        self._pc_step += 1
        return y


class DDFZPCWeightQuantizer(DDFZPCQuantizer):
    """Weight PC-DDFZ quantizer — codebook frozen after first build."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        stats_sample_groups: int = 4096,
        freeze_codebook: bool = True,
        compile_steps=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            freeze_codebook=freeze_codebook,
            stats_sample_groups=stats_sample_groups,
            compile_steps=compile_steps,
            **kwargs,
        )


class DDFZPCActQuantizer(DDFZPCQuantizer):
    """Activation PC-DDFZ quantizer — codebook rebuilt at compile steps."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        stats_sample_groups: int = 4096,
        compile_steps=None,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            freeze_codebook=False,
            stats_sample_groups=stats_sample_groups,
            compile_steps=compile_steps,
            **kwargs,
        )
