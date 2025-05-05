#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List,Literal


class SqSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softplus(x).pow(2)

        
class S4DFFT(nn.Module):
    """
    Diagonal State‑Space (S4D) layer with length‑agnostic FFT or recurrent scan.

      x : (B,T,D)  ➜  y : (B,T,D)
    """

    def __init__(
        self,
        d_model: int,
        N: int          = 64,          # # diagonal modes
        init: str       = "hippoD",    # 'hippoD' | 'inverse' | 'linear'
        short_thresh: int = 512,       # switch to recurrent if T ≤ this
        tau_min: float  = 1e-4,        # clamp on exp(log_tau)
    ):
        super().__init__()
        assert N % 2 == 0, "N must be even (conjugate pairs)."

        self.d_model, self.N = d_model, N
        self.tau_min = tau_min

        # unconstrained parameters for N/2 distinct modes
        self.log_tau = nn.Parameter(torch.randn(N // 2))
        self.freq    = nn.Parameter(torch.randn(N // 2))
        self.B       = nn.Parameter(torch.randn(N // 2))
        self.C       = nn.Parameter(torch.randn(N // 2))

        # input/output projections
        self.in_proj  = nn.Linear(d_model, N // 2, bias=False)
        self.out_proj = nn.Linear(N // 2, d_model, bias=False)

        # learnable global time‑scale Δt  (log‑domain)
        self.log_dt = nn.Parameter(torch.zeros(()))

        self._init_modes(init)

    # ----- initialisers --------------------------------------------------
    def _init_modes(self, kind: Literal["hippoD", "inverse", "linear"]):
        n = torch.arange(self.N // 2)
        with torch.no_grad(): 
            self.log_tau.fill_(math.log(0.5))
            if kind == "hippoD":
                self.freq.copy_(math.pi * (2*n + 1) / 2)
            elif kind == "inverse":
                self.freq.copy_((self.N / math.pi) / (2*n + 1))
            elif kind == "linear":
                self.freq.copy_(math.pi * n)
            else:
                raise ValueError(kind)
            nn.init.normal_(self.B,  mean=1.0, std=0.2)
            nn.init.normal_(self.C,  std=1.0 / math.sqrt(self.N/2))

    # ---------------------------------------------------------------------------
    # Real‑only kernel builder
    # ---------------------------------------------------------------------------
    def _kernel_fft(self, T: int):
        """
        Return RFFT(K) where K is the real convolution kernel of length T.
          output: (N, L/2+1) complex
        Everything up to the final rfft is real‑typed.
        """
        L   = self._next_pow_two(2 * T)

        dt   = torch.exp(self.log_dt)                      # scalar
        tau  = torch.exp(self.log_tau).clamp(min=self.tau_min)   # (N/2,)
        angle = self.freq * dt                                   # (N/2,)

        # |lam|  = exp(-tau*dt)            (real)
        # arg(lam)= angle                  (real)
        lam_mag = torch.exp(-tau * dt)                         # (N/2,)
        log_gain = (self.B.abs() + 1e-9).log() + \
                  (self.C.abs() + 1e-9).log()                 # (N/2,)

        i = torch.arange(T, device=tau.device)                 # (T,)

        # amplitude term  (N/2,T)   — still real
        amp = torch.exp(log_gain[:, None] + i[None] * torch.log(lam_mag)[:, None])

        # phase term
        phase = i[None] * angle[:, None]                       # (N/2,T)

        K_half = amp * torch.cos(phase)                        # (N/2,T) real

        # build full length‑N kernel (conjugate pair ⇒ symmetry in mode index)
        K_full = torch.cat([K_half, K_half.flip(0)], dim=0)     # (N,T) real

        return torch.fft.rfft(K_full, n=L, dim=-1)             # (N,L/2+1) complex

    @staticmethod
    def _next_pow_two(n: int) -> int:
        # smallest power of two ≥ n, in O(1) bit ops
        # (from Hacker’s Delight)
        n = n - 1
        n = n | (n >> 1)
        n = n | (n >> 2)
        n = n | (n >> 4)
        n = n | (n >> 8)
        n = n | (n >> 16)
        n = n | (n >> 16)

        # if you worry about >32‐bit dims, add: n |= (n >> 32)
        return n + 1

    # ----- forward (FFT or scan) ----------------------------------------
    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        x_proj  = self.in_proj(x)                               # (B,T,N/2)
        x_modes = torch.cat([x_proj, x_proj.flip(-1)], dim=-1)  # (B,T,N)  real

        L  = self._next_pow_two(2 * T)
        Uf = torch.fft.rfft(x_modes, n=L, dim=1).transpose(1, 2)   # (B,N,L/2+1)

        Kf = self._kernel_fft(T)                                   # (N,L/2+1)
        Yf = Uf * Kf[None]                                         # broadcast

        y_modes = torch.fft.irfft(Yf, n=L, dim=2)[..., :T]          # (B,N,T)
        y_modes = y_modes.transpose(1, 2)                          # (B,T,N)
        y       = y_modes[..., : self.N // 2]                       # (B,T,N/2)
        return self.out_proj(y)

class S4PreMix(nn.Module):
    def __init__(self, embed_dim: int, heads: int, petals: int):
        super().__init__()
        # compute per-head and inner dimensions
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.d_k = embed_dim // heads
        assert self.d_k % 2 == 0 , "self.d_dk must be divisible by 2"

        # choose number of modes = d_k
        self.N_modes = self.d_k//2 #cannot meaningfully use more than self.dk, optimizing for half- a low pass. 
        # S4D preprocessing
        self.s4d = S4DFFT(d_model=self.d_k, N=self.N_modes)
        # QKV projection at inner_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, S, embed_dim)
        B, S, E = x.shape
        # apply per-head S4 after projecting to embed_dim
        x = x.view(B * self.heads, S, self.d_k)
        x = self.s4d(x)
        x = x.view(B, S, E)
        # compute QKV
        qkv = self.qkv(x)  # (B, S, 3*E)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        q = q.view(B, S, self.heads, self.d_k).transpose(1,2)
        k = k.view(B, S, self.heads, self.d_k).transpose(1,2)
        v = v.view(B, S, self.heads, self.d_k).transpose(1,2)
        return q, k, v

class LinearPreMix(nn.Module):
    def __init__(self, embed_dim: int, heads: int, petals: int):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.d_k = embed_dim // heads
        # direct QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, S, embed_dim)
        B, S, E = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.d_k).transpose(1,2)
        k = k.view(B, S, self.heads, self.d_k).transpose(1,2)
        v = v.view(B, S, self.heads, self.d_k).transpose(1,2)
        return q, k, v


# ----------  Positive weight layer with Hoedt–Klambauer init ----------
# ---------------------------------------------------------------------
# PositiveLinear – strictly‑positive weights with HL init + safe interface
# ---------------------------------------------------------------------
class PositiveLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.raw  = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.empty(d_out)) if bias else None
        with torch.no_grad():
            nn.init.normal_(self.raw, mean=math.log(math.sqrt(2/d_in)), std=0.2)
            if self.bias is not None: self.bias.zero_()

    @property
    def weight(self):                        # strictly positive
        return F.softplus(self.raw).pow(2)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# ---------------- ICNN petal -----------------------------------------
class ICNN(nn.Module):
    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = [PositiveLinear(dim if i==0 else h, h) for i, h in enumerate(hidden_dims)]
        layers.append(PositiveLinear(hidden_dims[-1], dim))        # keep dimension
        self.layers = nn.ModuleList(layers)
        self.softplus = SqSoftplus()

    def forward(self, x):                                          # (..., D)
        z = x
        for layer in self.layers:
            z = self.softplus(layer(z))
        return z


class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1, bias=True)
        self.softplus = SqSoftplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.softplus(self.lin(x))      # convex, ≥ 0
        return 1.0 - torch.exp(-u)       # convex, ∈ (0,1)

class ScalarHull(nn.Module):
    """
    Scalar‐valued convex hull with a bounded convex gate,
    where the LSE temperature is τ = sqrt(‖xg‖² + ν),
    yielding a Student-t taper.
      x : (..., D)  →  y : (...,)   (scalar output)
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], petals: int):
        super().__init__()
        self.nu = 2.71828
        # each petal is a genuine ICNN: R^D→R^D→R
        self.petals = nn.ModuleList(ICNN(in_dim, hidden_dims) for _ in range(petals))
        # convex & bounded gate: g(x) ∈ (0,1)
        self.gate = ConvexGate(in_dim)
        self.eps  = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        g   = self.gate(x)                                   # (..., 1)
        xg  = (x + torch.randn_like(x) * self.eps) * g       # (..., D)
        # 1) magnitude per example: r = sqrt(mean(xg_i²))
        r   = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        # 2) temperature τ = sqrt(r² + ν)
        tau = torch.sqrt(r.pow(2) + self.nu)                 # (..., 1)
        # 3) compute each petal’s scalar score (mean over features)
        scores = [petal(xg).mean(dim=-1, keepdim=True) for petal in self.petals]  # list of (...,1)
        S      = torch.cat(scores, dim=-1)                   # (..., P)
        # 4) tempered log-sum-exp and rescale: out = (1/τ)·logsumexp(τ·S)
        lse = torch.logsumexp(S * tau, dim=-1, keepdim=True) # (..., 1)
        out = lse / tau                                       # (..., 1)
        return out.squeeze(-1)                                # (...,)


class VectorHull(nn.Module):
    """
    Vector‐valued convex hull with a bounded convex gate,
    where the LSE temperature is τ = sqrt(‖xg‖² + ν),
    yielding a Student-t taper.
      x : (..., D)  →  y : (..., D)
    """
    def __init__(self, dim: int, hidden_dims: List[int], petals: int):
        super().__init__()
        self.nu = 2.71828
        self.petals = nn.ModuleList(ICNN(dim, hidden_dims) for _ in range(petals))
        self.gate   = ConvexGate(dim)
        self.eps    = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        g    = self.gate(x)                                # (..., 1)
        xg   = (x + torch.randn_like(x) * self.eps) * g    # (..., D)
        # 1) magnitude per token: r = sqrt(mean(xg_i²))
        r    = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        # 2) temperature τ = sqrt(r² + ν)
        tau  = torch.sqrt(r.pow(2) + self.nu)              # (..., 1)
        # 3) get all petal outputs: shape (..., D, P)
        outs = torch.stack([petal(xg) for petal in self.petals], dim=-1)  # (..., D, P)
        # 4) tempered LSE per feature: out_j = (1/τ)·logsumexp(τ·outs_j)
        scaled = outs * tau.unsqueeze(-2)                  # (..., D, P)
        lse    = torch.logsumexp(scaled, dim=-1)           # (..., D)
        out    = lse / tau                                 # (..., D)
        return out
        
class ValueICNN(nn.Module):
    """
    Produce a vector-valued value embedding via a convex ICNN hull.
      x : (..., embed_dim) -> v : (..., embed_dim)
    """
    def __init__(self, embed_dim: int, hidden_dims: List[int], petals: int):
        super().__init__()
        # a VectorHull already returns a vector of same dim
        self.vh = VectorHull(embed_dim, hidden_dims, petals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        B, S, E = x.shape
        # flatten tokens, run through hull, then restore shape
        v = self.vh(x.reshape(-1, E))    # (B*S, E)
        return v.view(B, S, E)

class ConvexPositionalBias(nn.Module):
    """
    Bias(i,j) = - w * |i-j|    with   w ≥ 0  (learned per head)
    Convex in positional indices; monotone non‑increasing with distance.
    """
    def __init__(self, heads):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(heads))   # raw parameter
        self.softplus = SqSoftplus()

    def forward(self, S: int):
        device = self.w_raw.device
        w = self.softplus(self.w_raw)                      # (H,)
        pos  = torch.arange(S, device=device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S,S)
        bias = - w[:, None, None] * dist                # (H,S,S)
        return bias
        

class ConvexMixer(nn.Module):
    """Numerically‑robust replacement for ConvexSoftMixer.

    The design goals are:
    * keep the convex, non‑negative kernel factorisation (phi_q, phi_k ≥ 0)
    * combine with per‑token convex scores f_q, g_k and extra positional bias
    * avoid `exp` overflow by working in log‑space up to the final softmax
    * expose a temperature parameter for controllable sharpness
    """

    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(1.0 / math.sqrt(d_k)))

        self.score_q = ScalarHull(d_k, [d_k], petals)
        self.score_k = ScalarHull(d_k, [d_k], petals)
        self.gate = SqSoftplus()
        # non‑negative random‑feature maps
        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_score) -> torch.Tensor:  
        """q,k,v: (B, H, S, d_k)  *extra_score*: (1|B, 1|H, S, S)"""
        B, H, S, D = q.shape

        # === 1. Convex token scores ========================================
        f_q = self.score_q(q)               # (B,H,S)
        g_k = self.score_k(k)               # (B,H,S)

        # === 2. Non‑negative kernel =======================================
        # use softplus to guarantee positivity but cap the raw values before exp
        raw_phi_q = self.lin_h_q(q).clamp(max=20.0)
        raw_phi_k = self.lin_h_k(k).clamp(max=20.0)
        phi_q = self.gate(raw_phi_q)      # (B,H,S,r)
        phi_k = self.gate(raw_phi_k)       # (B,H,S,r)
        # Instead of einsum…
        # kernel = torch.einsum('bhsr,bhtr->bhst', phi_q, phi_k) + 1e-6
        
        # Do this:
        phi_k_t = phi_k.transpose(-1, -2)                            # (B,H,r,S)
        kernel  = torch.matmul(phi_q, phi_k_t) + 1e-6               # (B,H,S,S)        
        log_kernel = kernel.log()           # (B,H,S,S)

        # === 3. Combine all scores in log‑space ============================
        logit = (f_q.unsqueeze(-1) + g_k.unsqueeze(-2) + log_kernel) / self.temperature
        if extra_score is not None:
            logit = logit + extra_score

        # === 4. Row‑wise softmax with log‑sum‑exp trick ====================
        logit = logit - logit.max(dim=-1, keepdim=True).values      # stabilise
        weight = logit.exp()
        weight = weight / weight.sum(dim=-1, keepdim=True)          # (B,H,S,S)

        # === 5. Weighted sum ==============================================
        out = torch.einsum('bhij,bhjd->bhid', weight, v)            # (B,H,S,d_k)
        return out


class InterleavedPhaseChannelizer(nn.Module):
    """
    Embedding shape: (B, T, 2*M) == [c0, ϕ0, c1, ϕ1, ..., c_{M-1}, ϕ_{M-1}].
    This module:
      1. Extracts content channels ci at even indices.
      2. Builds a deterministic distance kernel W[i,j] = 1/(1 + |i-j|), applies the causal mask.
      3. Computes φ for each content channel: φ[b,i,m] = sum_j W[i,j] * x[b,j,2*m].
      4. Gates each φ-channel via a softplus-activated learnable scalar.
      5. Writes gated φ into the corresponding odd slots ϕm.
    """
    def __init__(self, embed_dim: int, init_gate_bias: float = -3.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.M = embed_dim // 2
        # one raw gate per channel
        self.gate_raw = nn.Parameter(torch.full((self.M,), init_gate_bias))
        self.softplus = SqSoftplus()

    def forward(self,
                x: torch.Tensor,   # (B, T, 2*M)
                mask: torch.Tensor # (1, 1, T, T) boolean causal+padding mask
    ) -> torch.Tensor:
        B, T, D2 = x.shape
        M = self.M
        assert D2 == 2 * M

        device = x.device
        dtype = x.dtype

        # 1) extract content slots ci
        x_c = x[..., 0::2]                  # (B, T, M)

        # 2) build deterministic distance kernel W[i,j] = 1 / (1 + |i-j|)
        pos = torch.arange(T, device=device, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
        W = 1.0 / (dist + 1.0)                               # (T, T)

        # 3) apply the causal+padding mask
        causal2d = mask.view(T, T)                          # (T, T)
        W = W * causal2d.to(dtype)

        # 4) normalize each row
        row_sum = W.sum(-1, keepdim=True).clamp(min=1e-6)    # (T, 1)
        W = W / row_sum                                     # (T, T)

        # 5) accumulate φ[b,i,m] = sum_j W[i,j] * x_c[b,j,m]
        #    use einsum: 'ij,bjm->bim'
        phi = torch.einsum('ij,bjm->bim', W, x_c)           # (B, T, M)

        # 6) gate each channel
        gate = self.softplus(self.gate_raw).view(1, 1, M)   # (1, 1, M)
        gated_phi = gate * phi                              # (B, T, M)

        # 7) write into odd slots ϕm
        out = x.clone()
        out[..., 1::2] = gated_phi

        return out
        
# ----------------------------------------------------------------------
#   Pairwise Hull Attention (mask‑aware)
# ----------------------------------------------------------------------
class PairwiseHullAttention(nn.Module):
    def __init__(self, embed_dim, heads, petals, use_s4d=False):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        if use_s4d:
            self.pre = S4PreMix(embed_dim, heads, petals)
        else:
            self.pre = LinearPreMix(embed_dim, heads, petals)
                
        self.mixer = ConvexMixer(self.d_k, petals, self.d_k*2)
        self.pos = ConvexPositionalBias(heads)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.phase = InterleavedPhaseChannelizer(embed_dim)

    def forward(self, x, mask=None):
        x = self.phase(x,mask)
        B, S, E = x.shape
        Q, K, V= self.pre(x)
        mean = 0.5 * (Q.mean() + K.mean())
        std  = 0.5 * (Q.std()  + K.std())
        Q = (Q - mean) / std
        K = (K - mean) / std
        
        bias = self.pos(S).unsqueeze(0)
        if mask is not None:
            bias = bias.masked_fill(~mask, float('-inf'))
        # pass bias as extra_score keyword
        y = self.mixer(Q, K, V, extra_score=bias)

        y = y.transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.W_O(y)

# ----------------------------------------------------------------------
#   OmniHull Block
# ----------------------------------------------------------------------
class OmniHullBlock(nn.Module):
    def __init__(self, dim, heads, petals, use_s4d=False):
        super().__init__()
        self.attn = PairwiseHullAttention(dim, heads, petals, use_s4d=use_s4d)
        self.hff  = VectorHull(dim, [dim * 2], petals)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.a1, self.a2   = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mix(x, y, a_raw):
        alpha = F.softplus(a_raw) / (1 + F.softplus(a_raw))
        return (1 - alpha) * x + alpha * y

    def forward(self, x: torch.Tensor, mask = None):
        x = self._mix(x, self.attn(self.ln1(x), mask), self.a1)
        x = self._mix(x, self.hff(self.ln2(x)),         self.a2)
        return x

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
        petals: int
    ):
        super().__init__()
        assert embed_dim >= 1, "embed_channels must be ≥1"
        self.embed_channels = embed_dim
        self.embed_dim = 2 * embed_dim

        # Embeddings only for even channels [0,2,4,...]
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # Blocks operate on full embed_dim
        self.blocks = nn.ModuleList([
            OmniHullBlock(
                self.embed_dim, heads, petals,
                use_s4d=(0 < i < depth - 1)
            )
            for i in range(depth)
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(S: int, device: torch.device) -> torch.Tensor:
        # shape (1, 1, S, S) where True = allowed
        return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device)) \
                   .unsqueeze(0).unsqueeze(1)

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, S) token indices
        returns logits: (B, S, vocab_size)
        """
        B, S = idx.shape
        device = idx.device

        # 1) embed into M dims
        x_small = self.token_emb(idx)                # (B, S, M)

        # 2) interleave zeros on odd positions → full embed_dim = 2*M
        #    result x[..., 0::2] = x_small, x[..., 1::2] = 0
        x = x_small.new_zeros(B, S, self.embed_dim)  # init zeros
        x[..., 0::2] = x_small                        # fill even slots

        # 3) build causal mask
        mask = self._causal_mask(S, device)          # (1, 1, S, S)

        # 4) apply each block (which will write φ into odd slots)
        for blk in self.blocks:
            x = blk(x, mask)

        # 5) final layernorm + head
        x = self.ln_f(x)                             # (B, S, embed_dim)
        logits = self.head(x)                        # (B, S, vocab_size)
        return logits




#training,eval loop 
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from typing import Literal
device    = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.jit.script
def wolf_update(p: torch.Tensor,
                g: torch.Tensor,
                state_p: torch.Tensor,
                lr: float):
    # define your constants here instead of capturing them
    etcerta: float = 0.367879441
    et:      float = 1.0 - etcerta

    # same logic as before
    update    = state_p * et + g * etcerta
    new_state = state_p * et + update * etcerta
    sign_agree = torch.sign(update) * torch.sign(g)
    update    = update + (torch.rand_like(update)*2 - 1) * etcerta * update
    p_new     = torch.where(sign_agree > 0, p - lr * update, p)
    return p_new, new_state

class Wolf(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['p'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state_p = self.state[p]['p']
                p_new, new_state = wolf_update(p.data, p.grad, state_p, lr)
                p.data.copy_(p_new)
                state_p.copy_(new_state)
        return loss

# 1) Load data and meta as before
data_dir  = os.path.dirname(base_dir)
train_ids = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
val_ids   = np.fromfile(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# 2) Compute data‐marginal q[v]
counts = np.bincount(train_ids, minlength=vocab_size).astype(float)
q = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=device)  # [V]

# 3) Dataset + DataLoader
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = torch.from_numpy(data).long()
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

block_size = 256
train_loader = DataLoader(CharDataset(train_ids, block_size),
                          batch_size=32, shuffle=True, drop_last=True)
val_loader   = DataLoader(CharDataset(val_ids,   block_size),
                          batch_size=32, shuffle=False, drop_last=True)

# 4) Model, optimizer, loss
virgin = ConvexGPT(vocab_size = vocab_size,embed_dim  = 64,depth  = 2,heads = 2,petals = 8)

print("Number of parameters: ", sum(p.numel() for p in virgin.parameters()))
model = torch.jit.script(virgin)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-3)#or adam, but i prefer the WOLF.
criterion = nn.CrossEntropyLoss()
losses = []
# 6) Train / eval functions
def train_epoch():
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Forward
        logits = model(xb)                 # (B, T, V)
        B, T, V = logits.shape
        p = F.softmax(logits, dim=-1)      # (B, T, V)

        # 1) Standard CE
        loss = criterion(logits.view(B*T, V),
                            yb.view(B*T))
        # Backprop
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#because things do explode sometimes
        optimizer.step()
        print(loss.item())
        total_loss += loss.item()
        losses.append(loss.item())
    return total_loss / len(train_loader)

@torch.no_grad()
def eval_epoch():
    model.eval()
    total_loss = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        B, T, V = logits.shape
        total_loss += criterion(logits.view(B*T,V),
                                yb.view(B*T)).item()
    return total_loss / len(val_loader)

# 7) Run training
num_epochs = 10
for epoch in range(1, num_epochs+1):
    train_loss = train_epoch() 
    val_loss   = eval_epoch()
    print(f"Epoch {epoch:2d} | train: {train_loss:.4f} | val: {val_loss:.4f}")



# --- helpers ---------------------------------------------------------
def fenchel_decode(logits, tau=1.0, iters=3):
    """Fenchel‑dual KL‑regularised projection of -logits (energy)."""
    energy = -logits                        # (B,V)
    p = torch.full_like(energy, 1.0 / energy.size(-1))  # uniform start
    for _ in range(iters):
        p = torch.softmax((-energy / tau) + p.log(), dim=-1)
    return p
    

# --- generation ------------------------------------------------------
use_fenchel   = False          # flip to compare
tau           = 1.0           # λ  (temperature analogue)
max_new_tokens = 4000
top_k          = 25
block_size     = 128
temperature    = 1.0

bcontext_str = "To be, or not to be,"
context_ids = torch.tensor([[ stoi[c] for c in bcontext_str ]],
                           dtype=torch.long)
context_ids = context_ids.to(device)


generated = context_ids.clone()  # (1,T0)
model.eval()
with torch.no_grad():
  for _ in range(max_new_tokens):
    input_ids = generated[:, -block_size:]        # casual block
    logits = model(input_ids)[:, -1, :] / temperature
    # top‑k mask
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -1e10

    if use_fenchel:
        probs = fenchel_decode(logits, tau=tau, iters=3)
    else:
        probs = torch.softmax(logits, dim=-1)

    next_id = torch.multinomial(probs, num_samples=1)   # (1,1)
    generated = torch.cat([generated, next_id], dim=1)

print('> ', ''.join(itos[i] for i in generated[0].tolist()))

