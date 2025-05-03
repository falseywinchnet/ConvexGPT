#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List, Literal

# --------‑‑‑‑ helper ---------------------------------------------------
@torch.jit.script   
def next_pow_two(x: int) -> int:
    t = torch.tensor(float(x))            # scalar tensor
    k = torch.ceil(torch.log2(t)).int()   # round‑up exponent
    return int((2 ** k).item())           # convert back to Python int

# ---------------------------------------------------------------------
#  S4DFFT  (augmented)
# ---------------------------------------------------------------------
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


    def _kernel_fft(self, T: int):
        """
        Return RFFT(K) where K is the real convolution kernel of length T.
          output: (N, L/2+1) complex
        Everything up to the final rfft is real‑typed.
        """
        L   = next_pow_two(2 * T)

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

    # ----- forward (FFT or scan) ----------------------------------------
    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        x_proj  = self.in_proj(x)                               # (B,T,N/2)
        x_modes = torch.cat([x_proj, x_proj.flip(-1)], dim=-1)  # (B,T,N)  real

        L  = next_pow_two(2 * T)
        Uf = torch.fft.rfft(x_modes, n=L, dim=1).transpose(1, 2)   # (B,N,L/2+1)

        Kf = self._kernel_fft(T)                                   # (N,L/2+1)
        Yf = Uf * Kf[None]                                         # broadcast

        y_modes = torch.fft.irfft(Yf, n=L, dim=2)[..., :T]          # (B,N,T)
        y_modes = y_modes.transpose(1, 2)                          # (B,T,N)
        y       = y_modes[..., : self.N // 2]                       # (B,T,N/2)
        return self.out_proj(y)


class S4PreMix(nn.Module):
    def __init__(self, embed_dim, inner_dim, heads, N_modes=64):
        super().__init__()
        assert inner_dim % heads == 0
        self.heads = heads
        self.d_k   = inner_dim // heads

        # ---- NEW: linear up‑proj to inner_dim -------------------------
        self.up   = nn.Linear(embed_dim, inner_dim, bias=False)

        # ---- S4D operates at reduced width ---------------------------
        self.s4d  = S4DFFT(d_model=inner_dim, N=N_modes)
        self.qkv  = nn.Linear(inner_dim, inner_dim * 3, bias=False)

    def forward(self, x):                       # x: (B,S,E)
        z  = self.s4d(self.up(x))               # (B,S,inner_dim)
        q, k, v = self.qkv(z).chunk(3, dim=-1)  # each (B,S,inner_dim)

        B, S, _ = x.shape
        new_shape = (B, S, self.heads, self.d_k)

        # safe reshape regardless of contiguity
        q = q.reshape(new_shape).transpose(1, 2)   # (B,H,S,d_k)
        k = k.reshape(new_shape).transpose(1, 2)
        v = v.reshape(new_shape).transpose(1, 2)
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
        return F.softplus(self.raw)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# ---------------- ICNN petal -----------------------------------------
class ICNN(nn.Module):
    def __init__(self, dim, hidden_dims):
        super().__init__()
        layers = [PositiveLinear(dim if i==0 else h, h) for i, h in enumerate(hidden_dims)]
        layers.append(PositiveLinear(hidden_dims[-1], dim))        # keep dimension
        self.layers = nn.ModuleList(layers)
        self.ReLU = nn.Softplus()

    def forward(self, x):                                          # (..., D)
        z = x
        for layer in self.layers:
            z = self.ReLU(layer(z))
        return z



#locally convex gate
class ConvexGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1, bias=True)
        self.ReLU = nn.Softplus()

    def forward(self, x):                       # (...,D)
        u = self.ReLU(self.lin(x))             # convex, ≥0
        return 1.0 - torch.exp(-u)              # convex, ∈(0,1)


class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1, bias=True)
        self.ReLU = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.ReLU(self.lin(x))      # convex, ≥ 0
        return 1.0 - torch.exp(-u)       # convex, ∈ (0,1)

class ScalarHull(nn.Module):
    """
    Scalar-valued convex hull with a bounded convex gate.
      x : (..., D)  ->  y : (...,)     (scalar output)
    """
    def __init__(self, in_dim: int, hidden: List[int], petals: int):
        super().__init__()
        # convex ICNN petals
        self.petals = nn.ModuleList(ICNN(in_dim, hidden) for _ in range(petals))
        # convex & bounded gate
        self.gate   = ConvexGate(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g      = self.gate(x)                                          # (...,1)
        xg     = x * g                                                 # (...,D)
        scores = [p(xg).mean(-1, keepdim=True) for p in self.petals]    # list of (...,1)
        return torch.logsumexp(torch.cat(scores, dim=-1), dim=-1)      # (...,)

class VectorHull(nn.Module):
    """
    Vector-valued convex hull with a bounded convex gate.
      x : (..., D)  ->  y : (..., D)
    """
    def __init__(self, dim: int, hidden: List[int], petals: int):
        super().__init__()
        # convex ICNN petals
        self.petals = nn.ModuleList(ICNN(dim, hidden) for _ in range(petals))
        # convex & bounded gate
        self.gate   = ConvexGate(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g    = self.gate(x)                                           # (...,1)
        xg   = x * g                                                 # (...,D)
        outs = torch.stack([p(xg) for p in self.petals], dim=-1)      # (...,D,P)
        return torch.logsumexp(outs, dim=-1)                         # (...,D)

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


class LinearPreMix(nn.Module):

    def __init__(self, embed_dim, inner_dim, heads,N_modes):
        super().__init__()
        assert inner_dim % heads == 0
        self.heads = heads
        self.d_k   = inner_dim // heads

        self.up   = nn.Linear(embed_dim, inner_dim,  bias=False)   # optional width lift
        self.qkv  = nn.Linear(inner_dim, inner_dim * 3, bias=False)

    def forward(self, x):               # x:(B,S,E)
        z        = self.up(x)           # (B,S,inner_dim)
        q, k, v  = self.qkv(z).chunk(3, dim=-1)

        B, S, _  = x.shape
        new_shape= (B, S, self.heads, self.d_k)

        q = q.reshape(new_shape).transpose(1,2)  # (B,H,S,d_k)
        k = k.reshape(new_shape).transpose(1,2)
        v = v.reshape(new_shape).transpose(1,2)
        return q, k, v


class ConvexPositionalBias(nn.Module):
    """
    Bias(i,j) = - w * |i-j|    with   w ≥ 0  (learned per head)
    Convex in positional indices; monotone non‑increasing with distance.
    """
    def __init__(self, heads):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(heads))   # raw parameter
        self.ReLU = nn.Softplus()

    def forward(self, S: int):
        device = self.w_raw.device
        w = self.ReLU(self.w_raw)                      # (H,)
        pos  = torch.arange(S, device=device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S,S)
        bias = - w[:, None, None] * dist                # (H,S,S)
        return bias

    """
efficient softmax version
class ConvexHullMixer(nn.Module):

    y[b,h,i,d] = Σ_j softmax( (q·kᵀ)/√d + bias )[b,h,i,j] · V[b,h,j,d]

    Implemented with `torch.matmul` to avoid fragile `einsum` shape errors.
    def __init__(self, d_k: int):
        super().__init__()
        self.val = nn.Linear(d_k, d_k, bias=False)

    def forward(
        self,
        q: torch.Tensor,              # (B, H, S, d)
        k: torch.Tensor,              # (B, H, S, d)
        v: torch.Tensor,              # (B, H, S, d)
        extra_score: torch.Tensor | None = None  # (1|B, H, S, S)
    ) -> torch.Tensor:               # returns (B, H, S, d)
        B, H, S, D = q.shape
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # (B,H,S,S)
        if extra_score is not None:
            scores = scores + extra_score  # broadcast OK

        attn = F.softmax(scores, dim=-1)
        y    = torch.matmul(attn, self.val(v))  # (B,H,S,d)
        return y
     """

#softmax-free mostly convex version
class ConvexSoftMixer(nn.Module):
    def __init__(self, d_k, proj_dim, petals):
        super().__init__()
        self.score_q = ICNN(d_k, [d_k], petals)      # f(q)
        self.score_k = ICNN(d_k, [d_k], petals)      # g(k)
        self.lin_h   = nn.Linear(d_k, proj_dim, bias=False)
        self.lin_v   = nn.Linear(d_k, proj_dim, bias=False)

    def forward(self, q, k, v):          # (B,H,S,d_k)
        B,H,S,D = q.shape
        f_q = self.score_q(q)            # (B,H,S)
        g_k = self.score_k(k)            # (B,H,S)

        # random‑feature kernel   exp(<q,k>) ≈ φ(q)·φ(k)
        phi_q = torch.exp(self.lin_h(q))     # (B,H,S,r)
        phi_k = torch.exp(self.lin_h(k))     # weight‑sharing ok
        scores = torch.einsum('bhsr,bhtr->bhst', phi_q, phi_k)  # (B,H,S,S)

        s_ij = f_q.unsqueeze(3) + g_k.unsqueeze(2) + scores.log()
        # pre‑compute value feature
        u_jd = self.lin_v(v)                             # (B,H,S,r)
        logits = s_ij.unsqueeze(-1) + u_jd.unsqueeze(2)  # (B,H,S,S,r)
        y = torch.logsumexp(logits, dim=3) - math.log(S) # (B,H,S,r)
        return y  

class PairwiseHullAttention(nn.Module):
    def __init__(self, embed_dim, heads, petals, inner_dim=128, use_s4d=False):
        super().__init__()
        assert inner_dim % heads == 0
        self.heads = heads
        self.d_k   = inner_dim // heads

        premix_cls = S4PreMix if use_s4d else LinearPreMix
        self.pre   = premix_cls(embed_dim, inner_dim, heads, N_modes=64)
        self.mixer = ConvexHullMixer(self.d_k)
        self.pos   = ConvexPositionalBias(heads)
        self.W_O   = nn.Linear(inner_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x    : (B,S,E)
        # mask : (1|B,1|H,S,S) with True = keep
        B, S, _ = x.shape
        q, k, v = self.pre(x)                               # (B,H,S,d)

        bias = self.pos(S).unsqueeze(0)                     # (1,H,S,S)
        if mask is not None:
            mask_bool = mask.to(torch.bool)
            bias      = bias.masked_fill(~mask_bool, float('-inf'))

        y = self.mixer(q, k, v, extra_score=bias)           # (B,H,S,d)
        out = y.transpose(1, 2).reshape(B, S, -1)           # (B,S,inner_dim)
        return self.W_O(out)

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = self._mix(x, self.attn(self.ln1(x), mask), self.a1)
        x = self._mix(x, self.hff(self.ln2(x)),         self.a2)
        return x

class ConvexGPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, depth: int, heads: int, petals: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            OmniHullBlock(embed_dim, heads, petals, use_s4d=(i < depth - 1))
            for i in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(S: int, device: torch.device) -> torch.Tensor:
        # shape (1, 1, S, S) where True = allowed
        return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(1)

    def forward(self, idx: torch.Tensor):
        B, S = idx.shape
        device = idx.device
        x = self.token_emb(idx)
        mask = self._causal_mask(S, device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))



'''
def fenchel_decode(logits, tau=1.0, iters=3):
    """Fenchel‑dual KL‑regularised projection of -logits (energy)."""
    energy = -logits                        # (B,V)
    p = torch.full_like(energy, 1.0 / energy.size(-1))  # uniform start
    for _ in range(iters):
        p = torch.softmax((-energy / tau) + p.log(), dim=-1)
    return p

# --- generation ------------------------------------------------------
use_fenchel   = True         
tau           = 1.0           # λ  (temperature analogue)
max_new_tokens = 200
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
    logits = model(input_ids)                     # (1,cur_T,V)
    logits = logits[:, -1, :] / temperature       # (1,V)

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
'''


