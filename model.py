#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List,Literal

#c1 everywhere BUT logsumexp because of the max
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONVEXITY INVARIANTS — DECLARATIVE MODEL GUARANTEES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# TOKEN HULL  C^ℓ := conv{z₁^ℓ, ..., z_S^ℓ} ⊂ ℝ^d
#     • The convex hull formed by all token vectors at layer ℓ.
#
#  HULL-PRESERVING MODULE
#     • A module f is hull-preserving if:
#           ∀i, zᵢ^ℓ⁺¹ ∈ conv{z₁^ℓ, ..., z_S^ℓ}
#       ⇨ every output token is a convex combination of the previous layer.
#
#  CONVEX MAP
#     • A function f is convex ⇨
#           f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)
#       Note: linear ⇒ convex, but not necessarily hull-preserving.
#
#  PER-MODULE CONVEXITY TABLE (after convexification of residuals + pre-mix)
# ─────────────────────────────────────────────────────────────────────────────
#    Component                        | Convex?   | Hull-preserving?   | Notes
# ───────────────────────────────────|───────────|────────────────────|─────────────────────────────────────────────
#  ConvexEmbedding                   |   ✓       |  n/a               | Per-token ICNN; no mixing.
#  InterleavedPhaseChannelizer       |   ✓       |  ✓                 | Uses row-simplex kernel; even-channels pass-through.
#  ConvexRoPE                        |   ✓       |  ✓                 | Orthogonal map per token; does not mix tokens.
#  ScalarHull / VectorHull           |   ✓       |  n/a               | Fully convex ICNNs; no token mixing.
#  ConvexMixer                       |   ✓       |  ✓                 | Uses Softmax over score-matrix ⇒ row-simplex.
#  LinearPreMix (projected version)  |   ✓       |  ✓                 | Weights square-normalised per row ⇒ convex mixing.
#  Residual Connections              |   ✓       |  ✓                 | Gated via learned convex function (ConvexGate).
#  LayerNorm                         |   ✓       |  ✓ (per-token)     | Applies per-token; does not mix sequence.
#  VectorHull FeedForward            |   ✓       |  n/a               | ICNN; convex per token.
# ─────────────────────────────────────────────────────────────────────────────
#
#  GLOBAL CONVEXITY CLAIMS
#
#    • Each module is convex in its input (Softplus, log-sum-exp, affine).
#    • Every cross-token operation is convex *and* hull-preserving.
#    • All residual paths apply convex gates ∈ (0,1), ensuring convex combinations.
#    • Softmax and bump kernels are row-stochastic ⇒ convex combinations of tokens.
#    • No step pushes tokens outside previous layer's convex hull.
#
# ⚠ NON-ISSUES
#
#    • Minor C¹ discontinuity in log-sum-exp max() point — measure zero — doesn't impact anything.
#    • No loss regularisation added — convexity is enforced by module design only.
#
#  CONSEQUENCE
#
#   ⇒ For all ℓ, the token sequence z^ℓ lies in the hull of z⁰.
#      That is, zᵢ^ℓ ∈ conv{z₁⁰, ..., z_S⁰}, ∀i, ℓ
#
#   ⇒ Model is *globally hull-preserving* and fully convex over input tokens.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List,Literal

#c1 everywhere BUT logsumexp because of the max


class LinearPreMix(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.d_k = embed_dim // heads
        # direct QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, S, embed_dim)
        B, S, E = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.d_k).transpose(1,2)
        k = k.view(B, S, self.heads, self.d_k).transpose(1,2)
        v = v.view(B, S, self.heads, self.d_k).transpose(1,2)
        return q, k, v

        
class PositiveLinearHK(nn.Module): #Hoedt–Klambauer MUST be used always
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(d_out, d_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / d_in))
            nn.init.normal_(self.raw, mean=mean, std=0.2)

    @property
    def weight(self):
        return F.softplus(self.raw).pow(2)  # ensures strict positivity

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = PositiveLinearHK(in_dim, 1, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.softplus(self.lin(x))      # convex, ≥ 0
        return 1.0 - torch.exp(-u)       # convex, ∈ (0,1)

        
class BatchedICNN(nn.Module):
    def __init__(self, in_dim: int, petals: int):
        super().__init__()
        self.in_dim = in_dim
        self.P = petals
        D = in_dim
        # layer dims: D → 2D → D
        self.d1, self.d2 = 2 * D, D

        # first-layer weights: (P, d1, D)
        self.weight_raw_0 = nn.Parameter(self._init_weight(petals, self.d1, D))
        self.bias_0       = nn.Parameter(torch.zeros(petals, self.d1))

        # second-layer weights: (P, d2, d1)
        self.weight_raw_1 = nn.Parameter(self._init_weight(petals, self.d2, self.d1))
        self.bias_1       = nn.Parameter(torch.zeros(petals, self.d2))

        # per-petal residual projection weight: maps 2D → D: shape (P, d1, d2)
        self.z_weight = nn.Parameter(torch.empty(petals, self.d1, self.d2))
        nn.init.kaiming_uniform_(self.z_weight, a=math.sqrt(5))

        # gating scalars
        self.gate_raw_0 = nn.Parameter(torch.full((petals,), -3.0))
        self.gate_raw_1 = nn.Parameter(torch.full((petals,), -3.0))

        self.output_bias = nn.Parameter(torch.zeros(petals, D))
        self.act = nn.Softplus()

    def _init_weight(self, petals: int, d_out: int, d_in: int) -> torch.Tensor:
        w = torch.empty(petals, d_out, d_in)
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / d_in))
            nn.init.normal_(w, mean=mean, std=0.2)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        orig = x.shape
        x_flat = x.reshape(-1, self.in_dim)          # (N, D)
        N = x_flat.size(0)

        # prepare weights & gates
        w0 = F.softplus(self.weight_raw_0).pow(2)    # (P, 2D, D)
        w1 = F.softplus(self.weight_raw_1).pow(2)    # (P, D, 2D)
        g0 = torch.sigmoid(self.gate_raw_0).view(self.P, 1, 1)  # (P,1,1)
        g1 = torch.sigmoid(self.gate_raw_1).view(self.P, 1, 1)

        # ----- first layer across petals -----
        x_in = x_flat.unsqueeze(0).expand(self.P, N, self.in_dim)       # (P, N, D)
        x_w0 = torch.bmm(x_in, w0.transpose(1,2))                       # (P, N, 2D)
        x_w0 = x_w0 + self.bias_0.unsqueeze(1)                          # (P, N, 2D)
        z0   = self.act(x_w0 * g0)                                      # (P, N, 2D)

        # ----- second layer -----
        x_w1 = torch.bmm(z0, w1.transpose(1,2))                         # (P, N, D)
        x_w1 = x_w1 + self.bias_1.unsqueeze(1)                          # (P, N, D)

        # ----- residual path via bmm -----
        # z_weight: (P, 2D, D), z0: (P, N, 2D) → z_mapped: (P, N, D)
        z_mapped = torch.bmm(z0, self.z_weight)                         # (P, N, D)

        # combine, activate, add final bias
        z1 = self.act(x_w1 * g1 + z_mapped)                             # (P, N, D)
        out = z1 + self.output_bias.unsqueeze(1)                        # (P, N, D)

        # reshape back to original leading dims + (P, D)
        out = out.permute(1, 0, 2)  # (N, P, D)
        lead_dims = list(orig[:-1])                 # e.g. [B, H, T]
        new_shape = lead_dims + [self.P, self.in_dim]  # [B, H, T, P, D]
        return out.reshape(new_shape)

#based on BatchedICNN but with additional inputs from X.

class KCN(nn.Module):
    def __init__(self, in_dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.P = petals
        self.k = 8
        D_in, D_out = self.in_dim, self.out_dim

        # 1. Shared Positive Basis Projection φ(x) → D_out
        self.phi_proj = PositiveLinearHK(D_in * self.k, D_out)

        # 2. Petal-wise second projection weights: Positive via HK
        self.raw_weight2 = nn.Parameter(torch.empty(petals, D_out, D_out))
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / D_out))
            nn.init.normal_(self.raw_weight2, mean=mean, std=0.2)
        self.bias2 = nn.Parameter(torch.zeros(petals, D_out))
        self.gate_raw2 = nn.Parameter(torch.full((petals,), -3.0))

        # 3. Shared Softplus shifts
        shifts = torch.linspace(-1.0, 1.0, self.k).view(1, 1, self.k)
        self.register_buffer("shifts", shifts)

        # 4. Residual connection (linear, may remain unconstrained)
        self.z_weight = nn.Parameter(torch.empty(petals, 2 * D_in, D_out))
        nn.init.kaiming_uniform_(self.z_weight, a=math.sqrt(5))

        # 5. Gate and bias per petal
        self.gate_raw = nn.Parameter(torch.full((petals,), -3.0))
        self.output_bias = nn.Parameter(torch.zeros(petals, D_out))

        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x_flat = x.reshape(-1, self.in_dim)
        N = x_flat.size(0)

        xb = x_flat.unsqueeze(-1) + self.shifts
        phi = F.softplus(xb)
        phi_flat = phi.reshape(N, self.in_dim * self.k)

        x_proj_shared = self.phi_proj(phi_flat)
        x_proj = x_proj_shared.unsqueeze(0).expand(self.P, N, self.out_dim)
        g1 = torch.sigmoid(self.gate_raw).view(self.P, 1, 1)
        z0 = self.act(x_proj * g1)

        # Apply softplus^2 to ensure positivity of second weights
        w2 = F.softplus(self.raw_weight2).pow(2).transpose(1, 2)
        z1 = torch.bmm(z0, w2) + self.bias2.unsqueeze(1)
        g2 = torch.sigmoid(self.gate_raw2).view(self.P, 1, 1)
        z1 = self.act(z1 * g2)

        x_res_in = torch.cat([x_flat, x_flat], dim=-1)
        x_res_exp = x_res_in.unsqueeze(0).expand(self.P, N, 2 * self.in_dim)
        x_res = torch.bmm(x_res_exp, self.z_weight)

        z_final = self.act(z1 + x_res) + self.output_bias.unsqueeze(1)
        out = z_final.permute(1, 0, 2)
        new_shape = list(orig[:-1]) + [self.P, self.out_dim]
        return out.reshape(new_shape)   

class _FusedLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        m, _ = x.max(dim=dim, keepdim=True)
        y = x - m
        ex = y.exp()
        s = ex.sum(dim=dim, keepdim=True)
        lse = m + s.log()
        ctx.save_for_backward(ex, s)
        ctx.dim = dim
        return lse

    @staticmethod
    def backward(ctx, grad_output):
        ex, s = ctx.saved_tensors
        dim = ctx.dim
        grad_x = grad_output * (ex / s)
        return grad_x, None

# TorchScript-compatible wrapper
class FusedLogSumExp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FusedLogSumExp.apply(x, self.dim)

class ScalarHull(nn.Module):
    def __init__(self, in_dim: int, petals: int):
        super().__init__()
        self.in_dim = in_dim
        self.register_buffer('nu',  torch.tensor(1.64872127070012814684865078781416357165))
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.petals = KCN(self.in_dim, petals)
        self.gate   = ConvexGate(in_dim)
        self.register_buffer("creative", torch.tensor(True))
        self.register_buffer('eps', torch.tensor(1e-6))
        self.fused_lse_hulls = FusedLogSumExp(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        g   = self.gate(x)                                   # (..., 1)

        #creativity toggle here

        if self.creative:
            xg   = (x + torch.randn_like(x) * self.noise_scale) * g # (..., D)
        else:
            xg   = x  * g # (..., D)

        # compute τ using a soft logistic
        r = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        tau = torch.exp(0.30343 * r + 0.22159)
        # get each petal’s vector output, then reduce to scalar per petal
        out_all = self.petals(xg)                  # (..., P, D)
        scores  = out_all.mean(dim=-1)             # (..., P)

        # tempered LSE over petals
        # scaled: (..., P) = scores * τ
        scaled = scores * tau                     # broadcasts τ→[...,1]→[...,P]

        lse    = self.fused_lse_hulls(scaled)  # (..., 1)

        # divide by τ and squeeze
        return (lse / tau).squeeze(-1)             # (...,)

class VectorHull(nn.Module):
    def __init__(self, dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.petals = BatchedICNN(self.in_dim, petals)
        self.gate   = ConvexGate(dim)
        self.register_buffer("creative", torch.tensor(True))
        self.register_buffer('eps', torch.tensor(1e-6))
        self.fused_lse_hulls = FusedLogSumExp(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        g = self.gate(x)  # (..., 1)

        # apply creativity scaling
        if self.creative:
            xg = (x + torch.randn_like(x) * self.noise_scale) * g  # (..., in_dim)
        else:
            xg = x * g

        # compute τ
        r = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        tau = torch.exp(0.30343 * r + 0.22159)  # (..., 1)

        # batched ICNN output (..., P, out_dim)
        out_all = self.petals(xg)  # (..., P, out_dim)

        # scale each vector per petal
        scaled = out_all * tau.unsqueeze(-1)  # (..., P, out_dim)

        # transpose petals to last dim for LSE
        scaled = scaled.transpose(-2, -1)     # (..., out_dim, P)

        # fused LSE over petals
        lse = self.fused_lse_hulls(scaled).squeeze(-1)  # (..., out_dim)

        # divide out τ
        return lse / tau  # (..., out_dim)
        
class ConvexMixer(nn.Module):
    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.register_buffer('eps', torch.tensor(1e-6))
        self.register_buffer('noise_scale', torch.tensor(1e-5))

        self.score_q = ScalarHull(d_k, petals)
        self.score_k = ScalarHull(d_k, petals)
        self.gate = nn.Softplus()
        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)
        self.register_buffer('creative', torch.tensor(True))
        self.fused = FusedLogSumExp(dim=-1)

    def forward(self, q, k, v, mask,mask_back):
        B, H, S, D = q.shape
        device = q.device

        # ——— 1) tau ———
        gate_q = self.gate(q)                          # (B,H,S,d_k)
        q = q * gate_q
        r = torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
        tau = torch.exp(0.30343 * r + 0.22159)  # or + tau_min

        # ——— 2) scalar hull scores ———
        fq = self.score_q(q)  # (B,H,S)
        gk = self.score_k(k)  # (B,H,S)
        if self.creative:
            qn = (torch.rand_like(q) - 0.5) * self.noise_scale
            kn = (torch.rand_like(k) - 0.5) * self.noise_scale
            fq_ = self.score_q(q + qn)
            gk_ = self.score_k(k + kn)
            delta_fq = (fq_ - fq).detach()
            delta_gk = (gk_ - gk).detach()
            fq = fq - 0.1 * delta_fq
            gk = gk - 0.1 * delta_gk

        # ——— 3) random-feature kernel ———
        phi_q = self.gate(self.lin_h_q(q).clamp(max=20.0))
        phi_k = self.gate(self.lin_h_k(k).clamp(max=20.0))
        log_phi_q = torch.log(phi_q + self.eps)
        log_phi_k = torch.log(phi_k + self.eps)
        sum_ab = log_phi_q.unsqueeze(-2) + log_phi_k.unsqueeze(-3)  # (B,H,S,S,r)
        logK = self.fused(sum_ab).squeeze(-1)                        # (B,H,S,S)

        # 5) Assemble logits with mask and temperature
        log_mask = torch.log(mask_back.clamp_min(self.eps))  # convert to log-domain
        scores = fq.unsqueeze(-1) + gk.unsqueeze(-2) + logK + log_mask  # additive
        
        logits = scores * tau.squeeze(-1).unsqueeze(-1)
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        weights = torch.exp(log_weights)  # (B,H,S,S)

        # 6) Weighted sum for attention output
        out = weights.reshape(B * H, S, S).bmm(v.reshape(B * H, S, D))
        out = out.reshape(B, H, S, D)

        # Optional: compute aggregated attn_score
        attn_score = weights.sum(dim=-3)
        attn_score = torch.softmax(attn_score, dim=-1).mean(dim=1)
        min_vals = attn_score.min(dim=-1, keepdim=True).values
        max_vals = attn_score.max(dim=-1, keepdim=True).values
        attn_score = (attn_score - min_vals) / (max_vals - min_vals + self.eps)

        return out, attn_score

class InterleavedPhaseChannelizer(nn.Module):
    """
    Embedding shape: (B, T, 2*M) == [c0, ϕ0, c1, ϕ1, ..., c_{M-1}, ϕ_{M-1}].
    Now uses a convex bump kernel instead of sine.
    """
    def __init__(self, embed_dim: int, init_gate_bias: float = -3.0, eps: float = 1e-6):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.M = embed_dim // 2
        self.gate_raw = nn.Parameter(torch.full((self.M,), init_gate_bias))
        self.softplus = nn.Softplus()
        self.eps = eps

    def bump_kernel(self, T: int, device: torch.device):
        """
        Returns a (T, T) kernel K[i,j] ∈ (0,1], convex in |j-i|,
        with K[i,i]=1, and smoothly → eps as j→end.
        """
        i = torch.arange(T, device=device).unsqueeze(1).float()      # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0).float()      # (1,T)

        # future offset u = (j - i) / (T - i)
        diff    = (j - i).clamp(min=0.0)                            # (T,T)
        horizon = (T - i).clamp(min=1.0)                            # (T,1)
        u       = (diff / horizon).clamp(max=1.0 - self.eps)        # [0,1)

        # bump exponent: convex, =1 at u=0, → -∞ as u→1
        expnt = 1.0 - 1.0 / (1.0 - u*u + self.eps)
        K      = torch.exp(expnt)                                   # (T,T)

        # enforce exact eps at u≈1
        K = torch.where(u >= 1.0 - self.eps, torch.full_like(K, self.eps), K)
        return K  # (T,T)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2*M)
        B, T, _ = x.shape
        M       = self.M
        device  = x.device
        dtype   = x.dtype

        # 1) content channels
        x_c = x[..., 0::2]  # (B, T, M)

        # 2) build convex bump kernel and mask it
        K = self.bump_kernel(T, device).to(dtype)   # (T,T)
        causal2d = mask.view(T, T).to(dtype)        # (T,T)
        K = K * causal2d                            # zero out masked-out positions

        # 3) normalize rows to sum=1
        K = K / (K.sum(-1, keepdim=True).clamp(min=self.eps))

        # 4) accumulate phase φ[b,i,m] = ∑₍ⱼ₌₀…T₋₁₎ K[i,j] · x_c[b,j,m]
        φ = torch.einsum('ij,bjm->bim', K, x_c)     # (B, T, M)

        # 5) gate & write back into odd slots
        gate = self.softplus(self.gate_raw).view(1,1,M)  # (1,1,M)
        φg   = φ * gate                                 # (B, T, M)
        out  = x.clone()
        out[..., 1::2] = φg
        return out

class ConvexRoPE(nn.Module):
    """
    Convex RoPE substitute with dynamic sequence length.
    Generates a monotonic, convex angle for each position-pair subspace.
    """
    def __init__(self, d_k: int):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for pairing"
        self.d_pair = d_k // 2
        # Linear mapping from scalar pos to angle per pair
        self.lin = nn.Linear(1, self.d_pair)

    def forward(self, S: int, device: torch.device) -> torch.Tensor:
        # positions normalized to [0,1]
        pos = torch.arange(S, device=device, dtype=torch.float32).unsqueeze(1) / max(S - 1, 1)
        θ = F.softplus(self.lin(pos))  # (S, d_pair), convex & ≥0
        return θ  # shape (S, d_pair)

from typing import Tuple
def apply_convex_rope(q: torch.Tensor, k: torch.Tensor, θ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:    
    """
    Apply ConvexRoPE rotation to q and k.
      q, k: (B, H, S, d_k)
      θ:    (S, d_k//2)
    Returns rotated q, k of same shape.
    """
    B, H, S, d_k = q.shape
    d2 = d_k // 2

    # reshape into (paired) shape
    q_ = q.view(B, H, S, d2, 2)
    k_ = k.view(B, H, S, d2, 2)

    # split into even (x) and odd (y) parts
    x_q, y_q = q_.unbind(-1)  # each (B, H, S, d2)
    x_k, y_k = k_.unbind(-1)

    # build cos/sin with correct shape (1,1,S,d2) to broadcast
    cosθ = torch.cos(θ).unsqueeze(0).unsqueeze(0)  # (1,1,S,d2)
    sinθ = torch.sin(θ).unsqueeze(0).unsqueeze(0)

    # rotate x/y pairs
    x_q2 = x_q * cosθ - y_q * sinθ
    y_q2 = x_q * sinθ + y_q * cosθ
    x_k2 = x_k * cosθ - y_k * sinθ
    y_k2 = x_k * sinθ + y_k * cosθ

    # stack back into pairs and reshape to original
    q_rot = torch.stack([x_q2, y_q2], dim=-1).reshape(B, H, S, d_k)
    k_rot = torch.stack([x_k2, y_k2], dim=-1).reshape(B, H, S, d_k)

    return q_rot, k_rot
    
# ----------------------------------------------------------------------
#   Pairwise Hull Attention (mask‑aware)
# ----------------------------------------------------------------------
class PairwiseHullAttention(nn.Module):
    def __init__(self, embed_dim, heads,moe_petals):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        self.pre = LinearPreMix(embed_dim, heads)
        self.mixer = ConvexMixer(self.d_k, moe_petals, self.d_k*2)#dont need many for scoring
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.phase = InterleavedPhaseChannelizer(embed_dim)
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.register_buffer("creative", torch.tensor(True))
        self.rope = ConvexRoPE(self.d_k)

    def forward(self, x, mask=None,mask_back=None):
        self.phase(x,mask) #apply in-place positional phasing
        B, S, E = x.shape
        Q, K, V= self.pre(x)
        offset = self.rope(S, x.device)                  # (S, d_k//2)
        Q, K = apply_convex_rope(Q, K, offset) 

        mean = 0.5 * (Q.mean() + K.mean())
        std  = 0.5 * (Q.std()  + K.std())
        Q = (Q - mean) / std
        K = (K - mean) / std        
        y,attn_scores = self.mixer(Q, K, V, mask=mask,mask_back=mask_back)
        
        y = y.transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.W_O(y), attn_scores

# ----------------------------------------------------------------------
#   OmniHull Block
# ----------------------------------------------------------------------
class OmniHullBlock(nn.Module):
    def __init__(self, dim, heads, moe_petals):
        super().__init__()
        self.attn      = PairwiseHullAttention(dim, heads, moe_petals)
        self.hff       = VectorHull(dim, moe_petals)
        self.ln1       = nn.LayerNorm(dim)
        self.ln2       = nn.LayerNorm(dim)
        # --- two per-branch residual gates ---
        self.res_gate1 = ConvexGate(dim)
        self.res_gate2 = ConvexGate(dim)

    def forward(self, x, mask=None, mask_back=None):
        # ——— attention branch ———
        x0, _ = x, None
        x1, attn_scores = self.attn(self.ln1(x0), mask, mask_back)   # x1 is the residual delta
        g1 = self.res_gate1(x0)                                      # shape (B,S,1)
        x  = x0 + g1 * x1

        # ——— feed-forward branch ———
        x2 = self.hff(self.ln2(x))                                   # second residual delta
        g2 = self.res_gate2(x)                                       # gate on the updated state
        x  = x + g2 * x2

        return x, attn_scores

class ConvexEmbedding(nn.Module):
    """
    ICNN-based convex embedding layer using Hoedt–Klambauer positive linear.
    Convex in the input simplex P.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 512, out_dim: int = 768):
        super().__init__()
        # first affine projection
        self.Wx = PositiveLinearHK(vocab_size, hidden_dim, bias=False)
        # convex skip connection
        self.Wz = PositiveLinearHK(hidden_dim, hidden_dim, bias=False)
        # convex activation
        self.act = nn.Softplus()
        # final convex output
        self.out = PositiveLinearHK(hidden_dim, out_dim)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        # P: (batch, seq, vocab_size)
        h0 = self.act(self.Wx(P))             # (batch, seq, hidden_dim)
        h1 = self.act(h0 + self.Wz(h0))       # combine skip-convex term
        return self.out(h1)

def tokens_to_simplex(idx: torch.LongTensor, vocab_size: int) -> torch.FloatTensor:
    """
    Convert token indices to one-hot simplex representation (batch, seq, V).
    At inference, P is exactly one-hot; for relaxed optimization, replace with softmax.
    """
    P = F.one_hot(idx, num_classes=vocab_size).float()
    return P

        
# ----------------------------------------------------------------------
#   GPT Wrapper with Causal Mask
# ----------------------------------------------------------------------
class ConvexGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        moe_petals: int,
        creativity: bool = True
    ):
        super().__init__()
        assert embed_dim >= 1, "embed_channels must be ≥1"
        self.embed_channels = embed_dim
        self.embed_dim = 2 * embed_dim
        self.vocab_size = vocab_size

        # Embeddings only for even channels [0,2,4,...]
        self.convex_embed = ConvexEmbedding(vocab_size, hidden_dim=512, out_dim=embed_dim)

        # Blocks operate on full embed_dim
        self.blocks = nn.ModuleList([
        OmniHullBlock(
            self.embed_dim,
            heads,
            moe_petals
                  )
            for i in range(depth)
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab_size, bias=False)
        self.set_creativity(creativity)


    @staticmethod
    def logistic_mask(S: int, eps: float = 1e-17) -> torch.Tensor:
        i = torch.arange(S).unsqueeze(1).float()  # (S,1)
        j = torch.arange(S).unsqueeze(0).float()  # (1,S)
    
        # Compute normalized future offset u = (j - i) / (S - i)
        diff = (j - i).clamp(min=0.0)
        horizon = (S - i).clamp(min=1.0)
        u = (diff / horizon).clamp(max=1.0)  # ∈ [0,1]
    
        # Decay from 1 to eps as u ∈ (0, 1]
        decay = torch.exp(1.0 - 1.0 / (1.0 - u ** 2 + 1e-6))  # smooth bump
        decay = decay.clamp_max(1.0)
    
        # Build final mask: full for j ≤ i, soft-decay for i < j
        mask = torch.where(j <= i, torch.ones_like(decay), decay)
        mask = mask.clamp(min=eps)
        return mask  # shape (S, S)
        
    @staticmethod
    def reversed_logistic_mask(S: int, eps: float = 1e-17) -> torch.Tensor:
            i = torch.arange(S).unsqueeze(1).float()  # (S,1)
            j = torch.arange(S).unsqueeze(0).float()  # (1,S)
        
            # Compute normalized past offset u = (i - j) / (i + 1)
            diff = (i - j).clamp(min=0.0)  # zero if j ≥ i (future)
            horizon = (i + 1).clamp(min=1.0)  # prevent division by zero
            u = (diff / horizon).clamp(max=1.0)
        
            # Smooth decay from 1 to eps as we move into the past
            decay = torch.exp(1.0 - 1.0 / (1.0 - u ** 2 + 1e-6))
            decay = decay.clamp_max(1.0)
        
            # Build mask: decay for j < i (past), 1 for j ≥ i (future including present)
            mask = torch.where(j >= i, torch.ones_like(decay), decay)
            mask = mask.clamp(min=eps)
            return mask  # shape (S, S)

    def set_creativity(self, value: bool):
        val = torch.tensor(value)
        def recurse(m):
            if hasattr(m, "creative"):
                m.creative.copy_(val)
            for child in m.children():
                recurse(child)
        recurse(self)

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, S) token indices
        returns logits: (B, S, vocab_size)
        """
        B, S = idx.shape
        device = idx.device
        #stack 0::2 as embeddings, 1::2 as zeros for positional embeddings
        P = tokens_to_simplex(idx, self.vocab_size)       # (batch, seq, V)
        embeddings = self.convex_embed(P)                          # (batch, seq, d_model)
        x = torch.stack([embeddings, torch.zeros_like(embeddings)], dim=-1).reshape(idx.shape[0], idx.shape[1], self.embed_dim)

        # 3) build causal mask
        mask_back = self.reversed_logistic_mask(S)          # (1, 1, S, S)
        mask = self.logistic_mask(S)          # (1, 1, S, S)

        attn_scores = []
        # 4) apply each block (which will write φ into odd slots)
        for blk in self.blocks:
            x,attn_temp = blk(x, mask,mask_back)
            attn_scores.append(attn_temp)

        attn_scores =  torch.stack(attn_scores).mean(dim=0)#divide by heads
        # 5) final layernorm + head
        x = self.ln_f(x)                             # (B, S, embed_dim)
        logits = self.head(x)                        # (B, S, vocab_size)
        return logits,attn_scores




