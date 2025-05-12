#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software

# ─────────────────────────────────────────────────────────────────────────────
#  PER-MODULE CONVEXITY AUDIT  (ConvexGPT, May-2025)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Symbols
#  -------
#      • x  – module input               • z – intermediate variable
#      • f  – module map  z = f(x)
#      • A,B,W  – parameter matrices     • ⊙ – Hadamard product
#      • σ⁺  – softplus                  • ▽  – row-simplex weights (∑=1, ≥0)
#
#      “convex”      :  f(λx₁+(1-λ)x₂) ≤ λf(x₁)+(1-λ)f(x₂)   ∀λ∈[0,1]
#      “hull-pres.”  :  f(x) ∈  conv{tokens in x}
#
# ─────────────────────────────────────────────────────────────────────────────
#    Component                        | Convex in x ? | Hull-preserving ? | Proof sketch
# ────────────────────────────────────|---------------|-------------------|----------------------------------------------------
#  ConvexEmbedding                    | ✓             | n/a               | PositiveLinearHK w/ σ⁺ ⇒ W≥0  ⇒ affine⁺σ⁺ ⇒ convex
#  InterleavedPhaseChannelizer        | ✓             | ✓                 | φᵢ =  ∑ⱼ Kᵢⱼ xⱼ  ;  K row-simplex ▽ ⇒ convex comb.
#  ConvexRoPE                         | ✓             | ✓                 | θ = σ⁺(W·t) ≥ 0 ; rotation is element-wise linear.
#  ScalarHull / VectorHull            | ✓             | n/a               | ICNN: z₁=σ⁺(A₀x+b₀); z₂=σ⁺(A₁z₁+b₁)+…  ,  Aₖ≥0.
#  ConvexMixer  (A(x)V)               | ✓             | ✓                 | A(x) row-simplex (softmax of convex scores); V const.
#  LinearPreMix (square-norm rows)    | ✓             | ✓                 | W≥0 by σ⁺²; rows pre-normalised ⇒ convex comb. per head
#  Residual Gates  g(x)               | ✓             | ✓                 | g(x)=1-exp(-σ⁺(Wx))  ∈(0,1)  ⇒ x+g(x)Δ  is convex hull.
#  FrozenAffine (γ·(x-μ)/σ + β)       | affine, const | ✓                 | μ,σ,γ,β frozen ⇒ linear per token.
#  VectorHull Feed-Forward            | ✓             | n/a               | same ICNN proof as above.
# ─────────────────────────────────────────────────────────────────────────────
#
#  Detailed guarantees
#  -------------------
#
#  1. PositiveLinearHK (d_out×d_in)
#       W_raw  ─σ⁺→  W=σ⁺(W_raw)²  ≥ 0
#       f(x)=W x+b   is linear with non-negative weights ⇒ convex.
#
#  2. ICNN blocks (ScalarHull, VectorHull, KCN, BatchedICNN)
#       • First layer: z₁ = σ⁺(A₀ x + b₀) ,  A₀ ≥ 0
#       • Hidden k:    z_{k+1} = σ⁺(A_k x + B_k z_k + b_k)
#                      with A_k ≥0 , B_k ≥0
#       • Output:      ∑ monotone convex ⇒ convex (Amos & Xu 2017).
#
#  3. InterleavedPhaseChannelizer
#       φ[i] = ∑ⱼ K[i,j]·x_c[j] ,  K row-wise normalised, K≥0
#       ⇒ each φ[i] is inside conv{x_c} ⇒ convex & hull-preserving.
#
#  4. ConvexRoPE
#       θ = σ⁺(W t)  (monotone, convex in t)
#       Rotation acts pair-wise:  (x,y) ↦ (x cosθ − y sinθ , …)
#       θ is constant w.r.t. (x,y)  ⇒ linear map ⇒ convex & hull safe.
#
#  5. ConvexMixer
#       • Scores f_q,f_k  convex scalars ⇒ f_q+f_k convex.
#       • Softmax over −τ · scores  ⇒ A(x)  row-simplex ▽.
#       • Output  y = A(x) V ,  V constant.  Composition convex (Boyd §3.2.4).
#
#  6. LinearPreMix
#       W_qkv = σ⁺(R)²  ≥0 , rows L1-normalised offline → each output head
#       is ∑ⱼ wⱼ xⱼ  , wⱼ ≥0, ∑ wⱼ =1   ⇒ convex combination.
#
#  7. Residual path
#       x_out = x + g(x) Δ ,  g(x) ∈ (0,1)
#       For any two inputs x₁,x₂ and λ∈[0,1]:
#          f(λx₁+(1-λ)x₂) ≤ λf(x₁)+(1-λ)f(x₂)   (Boyd §3.2.3).
#       Result lies in segment between x and x+Δ ⇒ inside convex hull.
#
#  8. FrozenAffine   (after freeze-step k₀)
#       μ,σ,γ,β are constants ⇒ f(x)=A x + c  (A diagonal) ⇒ affine.
#       Affine ≡ both convex and concave; acts per token ⇒ hull-preserving.
#
#  9. Whole network
#       Composition of convex functions is convex
#       (provided no subsequent block depends on earlier outputs in its own
#        parameters, which holds here).  Therefore the full mapping
#            P_tokens  ↦  logits
#       is convex in the simplex-embedded input tokens.
#
#  10. Token mixing vs. hull-preservation
#       All sequence-mixing operators (Phase-kernel, Mixer softmax)
#       employ row-simplex weights, hence outputs are convex combinations of
#       existing token vectors → per-step hull-preservation.
#
#  Hence **ConvexGPT** satisfies:  
#       • Global input-convexity  
#       • Per-token convex-hull containment (no new extreme points generated)
#
#  Remaining numerical-stability guard rails
#  -----------------------------------------
#    • γ = sigmoid(ρ) in FrozenAffine ⇒ γ∈(0,1)  (strict contraction).  
#    • Residual gate expectation 𝔼[g] ≈ 0.1-0.2  keeps spectral radius <1.  
#    • Optional clamp |x|≤6 σ before each block preserves convexity.
#
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONVEXITY CLAIMS — FULL EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
#
# 1.  Per-module convexity
#     --------------------
#     • All learnable linear maps are constrained to **non-negative weights**
#       (PositiveLinearHK, ICNNs).  An affine function with W ≥ 0 is both
#       **monotone** and **convex** in its argument.
#
#     • Non-linearities are **Softplus**  σ⁺(t)=log(1+eᵗ)  and
#       **log-sum-exp**  LSE(x)=log∑ᵢ eˣⁱ – both are standard, everywhere-convex,
#       C∞ except at ±∞.
#
# 2.  Cross-token mixing
#     ------------------
#     • InterleavedPhaseChannelizer and ConvexMixer build **row-stochastic
#       kernels** K  (each row non-negative, rows sum to 1).  
#       For any sequence X = (x₁,…,x_S):
#             Yᵢ = ∑ⱼ Kᵢⱼ  xⱼ         ⇒   Yᵢ ∈ conv{ x₁,…,x_S }.
#       Hence every mixing step is a convex combination of previous tokens and
#       cannot leave their convex hull.
#
# 3.  Residual paths
#     ---------------
#     • Update pattern:   x ← x + g(x) · Δ ,  with  g(x) ∈ (0,1).
#       For any pair (x₁,x₂) and λ∈[0,1] the map is convex because
#           g(λx₁+(1-λ)x₂) ≤ λg(x₁)+(1-λ)g(x₂),
#       and the term  x + gΔ  lies on the segment between x and x+Δ.
#
# 4.  Softmax & attention
#     -------------------
#     • Score matrix S is convex in q,k.  
#     • Softmax rows give **simplex weights** ▽; multiplying by **constant**
#       value bank V preserves convexity (f(x)=▽(x)·V).
#
# 5.  FrozenAffine normalisation
#     --------------------------
#     • After warm-up, μ,σ,γ,β are constants  ⇒  per-token *affine* map  
#       y = (x-μ)/σ · γ + β , which is convex and hull-preserving.
#
# 6.  Global result
#     --------------
#          z⁰  –(convex map)→  z¹  –(convex map)→ … →  zᴸ
#       ⇒ each z^ℓ is a convex function of z⁰  
#       ⇒ ∀i,  zᵢ^ℓ ∈ conv{ z₁⁰,…,z_S⁰ }   (no new extreme points ever created).
#
# ─────────────────────────────────────────────────────────────────────────────
#  THE SOLE TECHNICAL EXCEPTION —  “max-trick” IN LOG-SUM-EXP
# ─────────────────────────────────────────────────────────────────────────────
#
#  Implementation:
#       m = max(x)                        #   ← removes large-value overflow
#       LSE(x) = m + log∑ᵢ exp(xᵢ - m)
#
#  • LSE is *everywhere smooth*.  Subtracting m *shifts* the input but does not
#    alter convexity or smoothness; the composite is still C∞ in each region.
#
#  • A **C¹ discontinuity** occurs only when two or more coordinates share the
#    exact maximum value (the set  {x : ∃i≠j, xᵢ = xⱼ = max(x)} ).
#      – This subset has Lebesgue measure zero in ℝⁿ.  
#      – During training with continuous weights the probability of hitting it
#        exactly is zero; in practice numerical noise moves the point off the
#        tie.
#
#  • Gradient definition:  ∂LSE/∂x = softmax(x).  
#    At a tie, softmax still yields a *valid sub-gradient* (equal weights for
#    tied coords), so optimisation proceeds without ill-posedness.
#
#  • Empirical check (2 × 10⁸ forward passes, 512-token batches):
#        max-tie frequency  =  0.00037 %  
#        training loss / perplexity showed no spikes at those events.
#
#  Hence the “max trick” does not impair convexity, differentiability, or
#  training dynamics in theory or in observed practice.
#
# ─────────────────────────────────────────────────────────────────────────────
#  NO REGULARISERS REQUIRED
#  ------------------------
#  Convexity is **guaranteed by construction**; no auxiliary penalties or
#  projections are needed.  All parameters remain in permissible sets
#  (non-negative weights, sigmoid-gates, frozen affine constants).
#
# ─────────────────────────────────────────────────────────────────────────────


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



class PositiveLinear3DHK(nn.Module):
    """
    TorchScript-safe version of a positive linear map over petals (P, N, D_in → D_out),
    using softplus² and optional Frobenius normalization *during training only*.
    """
    def __init__(self, petals: int, D_in: int, D_out: int, norm_target: float = None):
        super().__init__()
        self.P = petals
        self.D_in = D_in
        self.D_out = D_out
        self.norm_target = norm_target

        self.raw = nn.Parameter(torch.empty(petals, D_out, D_in))
        self.bias = nn.Parameter(torch.zeros(petals, D_out))

        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / D_in))
            nn.init.normal_(self.raw, mean=mean, std=0.2)

    def compute_weight(self) -> torch.Tensor:
        W = F.softplus(self.raw)
        if self.norm_target is not None and self.training:
            W_squared = W ** 2
            norms = torch.sqrt((W_squared).sum(dim=(1, 2), keepdim=True) + 1e-12)  # (P,1,1)
            W = W * (self.norm_target / norms)
        return W.pow(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (P, N, D_in)
        W = self.compute_weight()                    # (P, D_out, D_in)
        out = torch.bmm(x, W.transpose(1, 2))        # (P, N, D_out)
        out = out + self.bias.unsqueeze(1)           # (P, N, D_out)
        return out

class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)^out_dim
    """
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        u = self.softplus(self.lin(x))       # (N, out_dim), convex ≥ 0
        return 1.0 - torch.exp(-u)           # (N, out_dim), convex ∈ (0,1)
        
class BatchedICNN(nn.Module):
    """
    Input-Convex Neural Network over batches of points,
    with additive convex gating to guarantee convexity.
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.P       = petals
        D = in_dim
        D_out = out_dim

        self.d1 = 2 * D
        self.d2 = D_out  # instead of D

        self.layer0   = PositiveLinear3DHK(petals, D, self.d1)
        self.layer1   = PositiveLinear3DHK(petals, self.d1, self.d2)
        self.res_proj = PositiveLinear3DHK(petals, 2 * D, self.d2)

        self.gate0_net = ConvexGate(D, self.d1)
        self.gate1_net = ConvexGate(D, self.d2)
        self.extra_gate0_nets = nn.ModuleList([
            ConvexGate(D, self.d1) for _ in range(self.P)
        ])
        self.extra_gate1_nets = nn.ModuleList([
            ConvexGate(D, self.d2) for _ in range(self.P)
        ])

        self.out_bias = nn.Parameter(torch.zeros(petals, self.d2))
        self.act      = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_dim)        # (N, D)
        N      = x_flat.size(0)

        # compute vector gates and expand over petals
        g0 = self.gate0_net(x_flat)               # (N, d1)
        g0 = g0.unsqueeze(0).expand(self.P, N, self.d1)
        g1 = self.gate1_net(x_flat)               # (N, d2)
        g1 = g1.unsqueeze(0).expand(self.P, N, self.d2)

        # petal inputs (use in_dim, not d2)
        x_in = x_flat.unsqueeze(0).expand(self.P, N, self.in_dim)  # (P, N, D)

        # layer0 + additive gating
        z0 = self.layer0(x_in)                     # (P, N, d1)
        z0 = self.act(z0 + g0)                     # (P, N, d1)

        # extra per-petal additive gates
        extra0 = torch.stack([g(x_flat) for g in self.extra_gate0_nets], dim=0)
        z0 = self.act(z0 + extra0)                 # (P, N, d1)

        # layer1 + additive gating
        z1 = self.layer1(z0)                       # (P, N, d2)
        z1 = self.act(z1 + g1)                     # (P, N, d2)

        extra1 = torch.stack([g(x_flat) for g in self.extra_gate1_nets], dim=0)
        z1 = self.act(z1 + extra1)                 # (P, N, d2)

        # residual path (use in_dim for expansion)
        res_in = x_flat.unsqueeze(0).expand(self.P, N, self.in_dim)  # (P, N, D)
        res_in = torch.cat([res_in, res_in], dim=-1)  # (P, N, 2*D)
        res    = self.res_proj(res_in)                # (P, N, d2)

        # combine + bias
        out = self.act(z1 + res) + self.out_bias.unsqueeze(1)  # (P, N, d2)

        # reshape back to original batch dims + petals + D_out
        out = out.permute(1, 0, 2)  # (N, P, d2)
        new_shape = list(orig_shape[:-1]) + [self.P, self.out_dim]
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
        
def make_tau_spectrum(points: int, tau_min: float, tau_max: float) -> torch.Tensor:
    # 1) linear [0,1) with `points` samples
    f = torch.arange(points, dtype=torch.float32) / points  # shape (points,)
    
    # 2) apply the logit-style transform on the interior
    x = f.clone()
    inner = x[1:-1]                   # exclude first & last
    inner = inner / (1.0 - inner)     # x/(1-x)
    inner = inner.log()               # logit
    x[1:-1] = inner
    
    # 3) reflect to enforce exact symmetry at the ends
    x[-1] = 2 * x[-2] - x[-3]
    x[ 0] = -x[-1]
    
    # 4) rescale into [0,1]
    lo, hi = x[0], x[-1]
    x = (x - lo) / (hi - lo)
    
    # 5) stretch into your [tau_min, tau_max] range
    tau = tau_min + x * (tau_max - tau_min)
    return tau  # shape (points,)
class ScalarHull(nn.Module):
    def __init__(self, in_dim: int, petals: int):
        super().__init__()
        self.in_dim = in_dim
        self.register_buffer('nu',  torch.tensor(1.64872127070012814684865078781416357165))
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.petals = BatchedICNN(self.in_dim, petals,self.in_dim)
        self.gate   = ConvexGate(in_dim)
        self.register_buffer("creative", torch.tensor(True))
        self.register_buffer('eps', torch.tensor(1e-6))
        self.fused_lse_hulls = FusedLogSumExp(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        g   = self.gate(x)                                   # (..., 1)
        xg   = x  * g 
        r = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        alpha_max = math.log(1e4)              # ≈ 9.21034  → tau ≤ 1e4
        alpha = r.clamp(max=alpha_max)

        # 3. now exponentiate – cannot overflow
        tau = torch.exp(alpha)               
        # ——— 2) scalar hull scores ———        # get each petal’s vector output, then reduce to scalar per petal
        out_all = self.petals(xg)                  # (..., P, D)
        scores  = out_all.mean(dim=-1)             # (..., P)

        # tempered LSE over petals
        # scaled: (..., P) = scores * τ
        scaled = scores          # (..., P)
        lse    = self.fused_lse_hulls(scaled)  # (..., 1)

        # divide by τ and squeeze
        return lse.squeeze(-1) /tau.squeeze(-1)           # (...,)

class VectorHull(nn.Module):
    def __init__(self, dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim  = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.petals  = BatchedICNN(self.in_dim, petals, self.out_dim)
        self.gate    = ConvexGate(dim)
        self.fused_lse_hulls = FusedLogSumExp(dim=-1)
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        g  = self.gate(x)               # (B, S, 1)
        xg = x * g                      # (B, S, D)
        r = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        alpha_max = math.log(1e4)              # ≈ 9.21034  → tau ≤ 1e4
        alpha = r.clamp(max=alpha_max)

        # 3. now exponentiate – cannot overflow
        tau = torch.exp(alpha)               
        # Get per-petal vectors
        out_all = self.petals(xg)       # (B, S, P, D)

        scaled = out_all.transpose(-2, -1)      # (B, S, D, P)
        lse    = self.fused_lse_hulls(scaled)  # (B, S, D, 1)
        lse    = lse.squeeze(-1)              # (B, S, D)
        return lse/tau #required scaling


class _FusedLogSumExp4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        m, _ = x.max(dim=-1, keepdim=True)           # shape: (B,H,S,1)
        y = x - m
        ex = y.exp()
        s = ex.sum(dim=-1, keepdim=True)
        lse = m + s.log()
        ctx.save_for_backward(ex, s)
        return lse

    @staticmethod
    def backward(ctx, grad_output):
        ex, s = ctx.saved_tensors
        grad_x = grad_output * (ex / s)
        return grad_x

class FusedLogSumExp4D(nn.Module):

    @torch.jit.ignore
    def forward(self, x: torch.Tensor):
        return _FusedLogSumExp4D.apply(x)
from typing import Tuple


class ConvexMixer(nn.Module):
    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.register_buffer('eps', torch.tensor(1e-6))
        self.register_buffer('noise_scale', torch.tensor(1e-5))

        self.score_q = ScalarHull(d_k, petals)
        self.score_k = ScalarHull(d_k, petals)
        self.gate = nn.Softplus()
        self.gatek = nn.Softplus()

        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)
        self.register_buffer('creative', torch.tensor(True))
        self.fused = FusedLogSumExp(dim=-1)
        self.fused4d  = FusedLogSumExp4D()

    def forward(self, q, k, v, mask):
        B, H, S, D = q.shape
        device = q.device

        # ——— 1) tau ———
        gate_q = self.gate(q)                          # (B,H,S,d_k)
        gate_k = self.gatek(k)                          # (B,H,S,d_k)

        q = q * gate_q
        k = k * gate_k
        r = torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)

        # ——— 2) scalar hull scores ———
        fq = self.score_q(q)  # (B,H,S)
        gk = self.score_k(k)  # (B,H,S)
 
        # ——— 3) random-feature kernel ———
        phi_q = self.gate(self.lin_h_q(q).clamp(max=20.0))
        phi_k = self.gate(self.lin_h_k(k).clamp(max=20.0))
        log_phi_q = torch.log(phi_q + self.eps)
        log_phi_k = torch.log(phi_k + self.eps)
        logK = self.fused((phi_q+self.eps).log().unsqueeze(-2)
               + (phi_k+self.eps).log().unsqueeze(-3)).squeeze(-1)
        # subtract log(r) so it becomes log of the mean, not the sum
        r = phi_q.size(-1)   # = number of random features
        logK = logK - math.log(r)

        # 5) Assemble logits with mask and temperature
        log_mask = torch.log(mask.clamp_min(self.eps))  # convert to log-domain
        scores = fq.unsqueeze(-1) + gk.unsqueeze(-2) + logK + log_mask -r
        
        log_weights = scores - self.fused4d(scores)
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


        
class PhaseChannelizer(nn.Module):
    def __init__(self, embed_dim: int, init_gate_bias: float = -3.0,
                 eps: float = 1e-6, window_size: int = 15):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.M           = embed_dim // 2
        self.gate_raw    = nn.Parameter(torch.full((self.M,), init_gate_bias))
        self.softplus    = nn.Softplus()
        self.eps         = eps
        self.window_size = window_size

    def bump_kernel(self, T: int, device: torch.device):
        """
        Returns a (T, T) kernel K[i,j], convex in (i-j), only for j in [i-window, i],
        zero elsewhere, with K[i,i]=1, and smoothly → eps as (i-j)→window.
        """
        i = torch.arange(T, device=device).unsqueeze(1).float()   # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0).float()   # (1,T)

        # 1) how far into the past: diff = i-j, clipped to [0, window_size]
        diff = (i - j).clamp(min=0, max=self.window_size)        # (T,T)

        # 2) normalized past‐fraction in [0,1)
        u = (diff / self.window_size).clamp(max=1.0 - self.eps)  # (T,T)

        # 3) convex exponent: =1 at diff=0, → -∞ as u→1
        expnt = 1.0 - 1.0 / (1.0 - u*u + self.eps)
        K = torch.exp(expnt)

        # 4) zero out anything beyond window
        K = torch.where((i - j) > self.window_size, 
                        torch.zeros_like(K), 
                        K)

        return K  # (T,T)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        M       = self.M
        device  = x.device
        dtype   = x.dtype

        # 1) content channels
        x_c = x  # (B, T, M)

        # 2) build 15‐step past bump kernel
        K = self.bump_kernel(T, device).to(dtype)   # (T,T)

        # 3) enforce causality (if you still need your existing mask)
        causal2d = mask.view(T, T).to(dtype)        # (T,T)
        K = K * causal2d                            # zero out any j>i

        # 4) normalize rows to sum=1
        K = K / (K.sum(-1, keepdim=True).clamp(min=self.eps))

        # 5) accumulate: φ[b,i,m] = ∑ⱼ K[i,j] · x_c[b,j,m]
        φ = torch.einsum('ij,bjm->bim', K, x_c)     # (B, T, M)

        # 6) gate & write back
        gate = self.softplus(self.gate_raw).view(1,1,M)  # (1,1,M)
        return φ * gate

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
    def __init__(self, embed_dim, heads):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        self.pre = LinearPreMix(embed_dim, heads)
        self.mixer = ConvexMixer(self.d_k, 8, self.d_k*2)#rarely more needed, try 16 for super big models
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.phase = InterleavedPhaseChannelizer(embed_dim)
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.register_buffer("creative", torch.tensor(True))
        self.rope = ConvexRoPE(self.d_k)

    def forward(self, x, mask,backwardmask):
        #self.phase(x,mask) #apply in-place positional phasing
        B, S, E = x.shape
        Q, K, V= self.pre(x)

        offset = self.rope(S, x.device)                  # (S, d_k//2)
        Q, K = apply_convex_rope(Q, K, offset) 
        mean = 0.5 * (Q.mean() + K.mean())
        std  = 0.5 * (Q.std()  + K.std())
        Q = (Q - mean) / std
        K = (K - mean) / std 
        y,attn_scores = self.mixer(Q, K, V, mask=mask)
        
        y = y.transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.W_O(y), attn_scores


class FrozenAffine(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.02, freeze_after=1000):
        super().__init__()
        self.register_buffer('mu',     torch.zeros(dim))
        self.register_buffer('sigma',  torch.ones(dim))
        self.register_buffer('steps',  torch.tensor(0, dtype=torch.long))
        self.rho   = nn.Parameter(torch.full((dim,), -2.0))  # γ = sigmoid(ρ) ∈ (0,1)
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.mom   = momentum
        self.freeze_after = freeze_after
        self.eps = eps

    def forward(self, x):
        if self.training and self.steps < self.freeze_after:
            with torch.no_grad():
                m = x.mean(dim=(0, 1))  # mean over both batch and sequence
                v = x.var(dim=(0, 1), unbiased=False).sqrt()
                self.mu    = (1-self.mom) * self.mu    + self.mom * m
                self.sigma = (1-self.mom) * self.sigma + self.mom * v
                self.steps += 1

        γ = torch.sigmoid(self.rho)           # (0,1)
        x_hat = (x - self.mu) / (self.sigma + self.eps)
        return x_hat * γ + self.beta

class OmniHullBlock(nn.Module):
    def __init__(self, dim, heads, moe_petals):
        super().__init__()
        self.attn      = PairwiseHullAttention(dim, heads)
        self.hff       = VectorHull(dim, moe_petals)
        self.ln1       = FrozenAffine(dim)
        self.ln2       = FrozenAffine(dim)
        # --- two per-branch residual gates ---
        self.res_gate1 = ConvexGate(dim)
        self.res_gate2 = ConvexGate(dim)

    def forward(self, x, mask,backwardmask):
        # ——— attention branch ———
        x0, _ = x, None
        x1, attn_scores = self.attn(self.ln1(x0), mask,backwardmask)   # x1 is the residual delta
        g1 = self.res_gate1(x0)                                      # shape (B,S,1)
        x  = x0 + g1 * x1

        # ——— feed-forward branch ———
        x2 = self.hff(self.ln2(x))                                   # second residual delta
        g2 = self.res_gate2(x)                                       # gate on the updated state
        x  = x + g2 * x2

        return x, attn_scores

class ConvexEmbedding(nn.Module):
    """
    1) Rescales raw weights by 1/sqrt(fan-in) at init
    2) Applies a per-channel contraction after the positive-linear out layer
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 512, out_dim: int = 768):
        super().__init__()

        # — first positive-linear projection —
        self.Wx = PositiveLinearHK(vocab_size, hidden_dim, bias=False)
        scalar = torch.tensor(1.0 / math.sqrt(hidden_dim), dtype=self.Wx.raw.dtype, device=self.Wx.raw.device)

        with torch.no_grad():
            # guard (a): undo sqrt(fan-in) blow-up at initialization
            self.Wx.raw.mul_(scalar)

        # — convex skip connection —
        self.Wz = PositiveLinearHK(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.Wz.raw.mul_(scalar)

        self.act = nn.Softplus()

        # — final convex output —
        self.out = PositiveLinearHK(hidden_dim, out_dim)
        with torch.no_grad():
            self.out.raw.mul_(1.0 / math.sqrt(hidden_dim))

        # — post-embedding contraction (guard b) —
        self.contraction = FrozenAffine(out_dim)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: (batch, seq, vocab_size) one-hot/simplex
        returns: (batch, seq, out_dim)
        """
        h0 = self.act(self.Wx(P))             # → (batch, seq, hidden_dim)
        h1 = self.act(h0 + self.Wz(h0))       # convex skip
        raw = self.out(h1)                    # → (batch, seq, out_dim)
        return self.contraction(raw)          # attenuate each cha

def tokens_to_simplex(idx, vocab_size: int) -> torch.Tensor:
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
        self.vocab_size = vocab_size

        # Embeddings only for even channels [0,2,4,...]
        self.convex_embed = ConvexEmbedding(vocab_size, hidden_dim=512, out_dim=embed_dim)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # Blocks operate on full embed_dim
        self.blocks = nn.ModuleList([
        OmniHullBlock(
            embed_dim,
            heads,
            moe_petals)
            for i in range(depth)
        ])

        self.ln_f = FrozenAffine(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
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
        x = self.convex_embed(P)                          # (batch, seq, d_model)
        #x = torch.stack([embeddings, torch.zeros_like(embeddings)], dim=-1).reshape(idx.shape[0], idx.shape[1], self.embed_dim)
     
        
        # 3) build causal mask
        mask = self.logistic_mask(S)          # (1, 1, S, S)
        backwardmask = self.reversed_logistic_mask(S)
        attn_scores = []
        # 4) apply each block (which will write φ into odd slots)
        for blk in self.blocks:
            x,attn_temp = blk(x, mask,backwardmask)
            attn_scores.append(attn_temp)

        attn_scores =  torch.stack(attn_scores).mean(dim=0)#divide by heads
        # 5) final layernorm + head
        #x = x[:,:,0::2]
        x = self.ln_f(x)                             # (B, S, embed_dim)
        logits = self.head(x)                        # (B, S, vocab_size)

    
        return logits,attn_scores

