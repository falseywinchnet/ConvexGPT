#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List, Literal

# ---------------- Diagonal State‑Space (S4D) layer ----------------
class S4DFFT(nn.Module):
    """
    Diagonal State‑Space (S4D) layer with length‑agnostic FFT or recurrent scan.
      Input x: (B, T, D) -> output y: (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,               # number of diagonal modes (must be even)
        init: str = "hippoD",     # initialization scheme for frequencies
        short_thresh: int = 512,   # switch to recurrent if T <= this (unused here)
        tau_min: float = 1e-4,     # min time‑scale clamp
    ):
        super().__init__()
        assert N % 2 == 0, "N must be even (conjugate pairs)."

        # Model dims
        self.d_model, self.N = d_model, N
        self.tau_min = tau_min

        # Parameters for N/2 distinct complex modes
        # log_tau: controls decay rate
        # freq: angular frequency of oscillation
        # B, C: input/output gains per mode
        self.log_tau = nn.Parameter(torch.randn(N // 2))    # shape (N/2,)
        self.freq    = nn.Parameter(torch.randn(N // 2))    # shape (N/2,)
        self.B       = nn.Parameter(torch.randn(N // 2))    # shape (N/2,)
        self.C       = nn.Parameter(torch.randn(N // 2))    # shape (N/2,)

        # Projections: map from d_model to N/2 modes and back
        self.in_proj  = nn.Linear(d_model, N // 2, bias=False)
        self.out_proj = nn.Linear(N // 2, d_model, bias=False)

        # Global time‑step Δt (learned in log-domain)
        self.log_dt = nn.Parameter(torch.zeros(()))         # scalar

        # Initialize mode parameters
        self._init_modes(init)

    def _init_modes(self, kind: Literal["hippoD", "inverse", "linear"]):
        """Initialize frequencies and gains based on scheme."""
        n = torch.arange(self.N // 2)
        with torch.no_grad():
            # set default decay half-life
            self.log_tau.fill_(math.log(0.5))                # tau = 0.5
            # set mode frequencies
            if kind == "hippoD":
                self.freq.copy_(math.pi * (2*n + 1) / 2)      # odd multiples of π/2
            elif kind == "inverse":
                self.freq.copy_((self.N / math.pi) / (2*n + 1))
            elif kind == "linear":
                self.freq.copy_(math.pi * n)
            else:
                raise ValueError(kind)
            # gains initialized with small noise
            nn.init.normal_(self.B,  mean=1.0, std=0.2)
            nn.init.normal_(self.C,  std=1.0 / math.sqrt(self.N/2))

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return smallest power of two ≥ n."""
        n -= 1
        n |= n >> 1; n |= n >> 2; n |= n >> 4
        n |= n >> 8; n |= n >> 16; n |= n >> 16
        return n + 1

    def _kernel_fft(self, T: int):
        """
        Compute RFFT of real convolution kernel K of length T.
        Returns complex tensor of shape (N, L//2+1) where L = next_pow_two(2*T).
        """
        L = self._next_pow_two(2 * T)

        # scalar Δt and per-mode tau, angle
        dt   = torch.exp(self.log_dt)                              # scalar > 0
        tau  = torch.exp(self.log_tau).clamp(min=self.tau_min)    # (N/2,)
        angle= self.freq * dt                                     # (N/2,)

        # decay magnitude per mode: exp(-tau*dt)
        lam_mag = torch.exp(-tau * dt)                            # (N/2,)
        # combine input/output gains in log-space
        log_gain = (self.B.abs()+1e-9).log() + (self.C.abs()+1e-9).log()  # (N/2,)

        # time indices 0..T-1
        i = torch.arange(T, device=tau.device)                    # (T,)

        # amplitude: (N/2, T)
        amp   = torch.exp(log_gain[:,None] + i[None]*torch.log(lam_mag)[:,None])
        # phase: (N/2, T)
        phase = i[None] * angle[:,None]

        # real part of each half-kernel: (N/2, T)
        K_half = amp * torch.cos(phase)
        # mirror modes for full real kernel: (N, T)
        K_full = torch.cat([K_half, K_half.flip(0)], dim=0)

        # return RFFT: shape (N, L//2+1), complex dtype
        return torch.fft.rfft(K_full, n=L, dim=-1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        Return y: (B, T, d_model)
        """
        B, T, _ = x.shape
        # project to N/2 modes: x_proj (B, T, N/2)
        x_proj = self.in_proj(x)
        # create full N modes by mirroring: (B, T, N)
        x_modes= torch.cat([x_proj, x_proj.flip(-1)], dim=-1)

        # FFT over time dim: result Uf shape (B, N, L//2+1)
        L  = self._next_pow_two(2 * T)
        Uf = torch.fft.rfft(x_modes, n=L, dim=1).transpose(1,2)

        # get kernel FFT: (N, L//2+1)
        Kf = self._kernel_fft(T)
        # elementwise multiply in freq domain
        Yf = Uf * Kf[None]

        # inverse FFT back, truncate to T: (B, N, T)
        y_modes = torch.fft.irfft(Yf, n=L, dim=2)[...,:T]
        # transpose to (B, T, N)
        y_modes = y_modes.transpose(1,2)
        # take first N/2 modes and project out: y (B, T, d_model)
        y = y_modes[..., : self.N//2]
        return self.out_proj(y)

# ----------- Pre-mix modules for attention input -------------------
class S4PreMix(nn.Module):
    """
    Apply per-head S4DFFT preprocessing before QKV projection.
    """
    def __init__(self, embed_dim: int, heads: int, petals: int, use_s4d=True):
        super().__init__()
        assert embed_dim % heads == 0
        self.heads = heads
        self.d_k   = embed_dim // heads
        self.N_modes = self.d_k  # N=modes per head
        # S4D layer per head on sequences
        self.s4d = S4DFFT(d_model=self.d_k, N=self.N_modes)
        # linear projection to QKV: maps embed_dim -> 3*embed_dim
        self.qkv= nn.Linear(embed_dim, 3*embed_dim, bias=False)

    def forward(self, x: torch.Tensor):  # x: (B, S, E)
        B, S, E = x.shape
        # reshape into (B*heads, S, d_k)
        x = x.view(B*self.heads, S, self.d_k)
        x = self.s4d(x)                  # apply S4D
        x = x.view(B, S, E)
        # compute QKV
        qkv = self.qkv(x)                # (B, S, 3E)
        q,k,v = qkv.chunk(3, dim=-1)
        # split heads: each of (B, S, heads, d_k) -> transpose -> (B, heads, S, d_k)
        q = q.view(B,S,self.heads,self.d_k).transpose(1,2)
        k = k.view(B,S,self.heads,self.d_k).transpose(1,2)
        v = v.view(B,S,self.heads,self.d_k).transpose(1,2)
        return q, k, v

class LinearPreMix(nn.Module):
    """
    Standard linear QKV projection without S4D.
    """
    def __init__(self, embed_dim: int, heads: int, petals: int, use_s4d=False):
        super().__init__()
        assert embed_dim % heads == 0
        self.heads = heads
        self.d_k   = embed_dim // heads
        self.qkv   = nn.Linear(embed_dim, 3*embed_dim, bias=False)

    def forward(self, x: torch.Tensor):  # x: (B, S, E)
        B,S,E = x.shape
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=-1)
        # reshape & transpose same as S4PreMix
        q = q.view(B,S,self.heads,self.d_k).transpose(1,2)
        k = k.view(B,S,self.heads,self.d_k).transpose(1,2)
        v = v.view(B,S,self.heads,self.d_k).transpose(1,2)
        return q, k, v

# -------- Positive weight linear layer with Hoedt–Klambauer init -------
class PositiveLinear(nn.Module):
    """Linear layer with strictly positive weights via softplus(raw).
    Initial raw ~ N(log(sqrt(2/d_in)), .2)
    """
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(d_out, d_in))  # raw weights
        self.bias= nn.Parameter(torch.empty(d_out)) if bias else None
        with torch.no_grad():
            nn.init.normal_(self.raw, mean=math.log(math.sqrt(2/d_in)), std=0.2)
            if self.bias is not None: self.bias.zero_()

    @property
    def weight(self):
        return F.softplus(self.raw)  # strictly positive

    def forward(self, x):  # x: (..., d_in)
        return F.linear(x, self.weight, self.bias)

# ------------------- Input Convex Neural Net (ICNN) -------------------
class ICNN(nn.Module):
    """ICNN producing convex mapping with positive-linear weights.
    """
    def __init__(self, dim, hidden_dims: List[int]):
        super().__init__()
        layers = []
        for i,h in enumerate(hidden_dims):
            in_dim = dim if i==0 else hidden_dims[i-1]
            layers.append(PositiveLinear(in_dim, h))
        layers.append(PositiveLinear(hidden_dims[-1], dim))  # output same dim
        self.layers = nn.ModuleList(layers)
        self.softplus= nn.Softplus()

    def forward(self, x: torch.Tensor):  # x: (..., dim)
        z = x
        for layer in self.layers:
            # each layer: positive-linear + Softplus => convex
            z = self.softplus(layer(z))
        return z  # convex output

# ----------------- Convex & bounded gating function ------------------
class ConvexGate(nn.Module):
    """
    g(x) = 1 - exp(-Softplus(Wx + b)) in (0,1), convex in x.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):  # x: (..., D)
        u = self.softplus(self.lin(x))  # u ≥ 0, convex
        return 1.0 - torch.exp(-u)       # output in (0,1)

# ----------------- Convex hull aggregation modules ------------------
class ScalarHull(nn.Module):
    """
    Scalar-valued convex hull: combine ICNN petals via logsumexp.
    Input x: (..., D) -> output: (...,)
    """
    def __init__(self, in_dim: int, hidden: List[int], petals: int):
        super().__init__()
        self.petals = nn.ModuleList(ICNN(in_dim, hidden) for _ in range(petals))
        self.gate   = ConvexGate(in_dim)

    def forward(self, x: torch.Tensor):  # x: (..., D)
        g    = self.gate(x)                          # (...,1)
        xg   = x * g                                 # gated input
        scores = [p(xg).mean(-1, keepdim=True) for p in self.petals]
        # logsumexp over petals: convex hull
        return torch.logsumexp(torch.cat(scores, dim=-1), dim=-1)

class VectorHull(nn.Module):
    """
    Vector-valued convex hull: like ScalarHull but returns D-dimensional.
    Input x: (..., D) -> output: (..., D)
    """
    def __init__(self, dim: int, hidden: List[int], petals: int):
        super().__init__()
        self.petals= nn.ModuleList(ICNN(dim, hidden) for _ in range(petals))
        self.gate  = ConvexGate(dim)

    def forward(self, x: torch.Tensor):  # x: (..., D)
        g    = self.gate(x)                         # (...,1)
        xg   = x * g                                # (..., D)
        outs = torch.stack([p(xg) for p in self.petals], dim=-1)  # (..., D, P)
        return torch.logsumexp(outs, dim=-1)        # (..., D)

class ValueICNN(nn.Module):
    """Wrap VectorHull for sequence: input (B,S,E) -> (B,S,E)."""
    def __init__(self, embed_dim: int, hidden_dims: List[int], petals: int):
        super().__init__()
        self.vh = VectorHull(embed_dim, hidden_dims, petals)

    def forward(self, x: torch.Tensor):  # x: (B,S,E)
        B,S,E = x.shape
        v = self.vh(x.view(-1, E))  # flatten tokens -> (B*S, E)
        return v.view(B, S, E)

class ConvexPositionalBias(nn.Module):
    """
    Computes bias per head: - w * |i-j|, convex in distance.
    Returns (H, S, S).
    """
    def __init__(self, heads: int):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(heads))
        self.softplus= nn.Softplus()

    def forward(self, S: int):
        w    = self.softplus(self.w_raw)  # (H,), ≥0
        # broadcast to (H,S,S)
        i    = torch.arange(S, device=w.device, dtype=torch.float32)
        dist = (i.unsqueeze(0) - i.unsqueeze(1)).abs()  # (S,S)
        return -w[:,None,None] * dist                    # (H,S,S)

class ConvexMixer(nn.Module):
    """
    Attention mixer: convex scores + non-neg kernel + positional bias.
    Numerically stable log-space computations.
    """
    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.temperature = (1.0 / math.sqrt(d_k))  # scaling

        # per-token convex scores f_q, g_k
        from typing import List
        class ScalarHullInline(nn.Module):
            def __init__(self, d: int, hidden: List[int], petals: int):
                super().__init__()
                # petals x hidden dims
                self.petals = nn.ModuleList([
                    nn.Sequential(nn.Linear(d, h), nn.Softplus(), nn.Linear(h, 1))
                    for h in hidden for _ in range(petals)
                ])
            def forward(self, x):  # (..., d)
                outs = torch.stack([p(x) for p in self.petals], dim=-1).squeeze(-2)
                return torch.logsumexp(outs, dim=-1)

        self.score_q = ScalarHullInline(d_k, [d_k], petals)
        self.score_k = ScalarHullInline(d_k, [d_k], petals)

        # random-feature maps -> non-negative kernel factors
        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)

    def forward(self, q, k, v, extra_score):
        # q,k,v: (B,H,S,d_k), extra_score: (1|B,1|H,S,S)
        B,H,S,D = q.shape
        # 1. convex token logits
        f_q = self.score_q(q)          # (B,H,S)
        g_k = self.score_k(k)          # (B,H,S)
        # 2. kernel matrix via RF features
        phi_q = F.softplus(self.lin_h_q(q).clamp(max=20.0))  # (B,H,S,r)
        phi_k = F.softplus(self.lin_h_k(k).clamp(max=20.0))  # (B,H,S,r)
        kernel= torch.matmul(phi_q, phi_k.transpose(-1,-2)) + 1e-6  # (B,H,S,S)
        log_kernel = kernel.log()
        # 3. combine logits: (f_q + g_k + log_kernel)/temp + extra_score
        logit= (f_q.unsqueeze(-1) + g_k.unsqueeze(-2) + log_kernel)/self.temperature
        if extra_score is not None:
            logit = logit + extra_score
        # 4. row-wise softmax (stable)
        logit = logit - logit.max(dim=-1, keepdim=True).values
        weight= logit.exp()
        weight= weight / weight.sum(dim=-1, keepdim=True)  # (B,H,S,S)
        # 5. weighted sum: output (B,H,S,d_k)
        return torch.einsum('bhij,bhjd->bhid', weight, v)

class PairwiseHullAttention(nn.Module):
    """
    Full attention block: pre-mix, convex mixer, bias, output proj.
    """
    def __init__(self, embed_dim: int, heads: int, petals: int, use_s4d=False):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k   = embed_dim // heads
        # choose pre-mix strategy
        self.pre   = S4PreMix(embed_dim, heads, petals, True) if use_s4d else \
                    LinearPreMix(embed_dim, heads, petals)
        self.mixer = ConvexMixer(self.d_k, petals, self.d_k*2)
        self.pos   = ConvexPositionalBias(heads)
        self.W_O   = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):  # x: (B,S,E)
        B,S,E = x.shape
        # project to q,k,v: each (B,H,S,d_k)
        q,k,v = self.pre(x)
        # positional bias (H,S,S) -> broadcast to (B,H,S,S)
        bias = self.pos(S).unsqueeze(0)
        if mask is not None:
            bias = bias.masked_fill(~mask, float('-inf'))
        # apply mixer, returns (B,H,S,d_k)
        y = self.mixer(q, k, v, extra_score=bias)
        # reshape back: (B,S,E)
        y = y.transpose(1,2).reshape(B,S,E)
        return self.W_O(y)  # final linear projection

class OmniHullBlock(nn.Module):
    """Transformer block with convex attention and convex feed-forward hull."""
    def __init__(self, dim: int, heads: int, petals: int, use_s4d=False):
        super().__init__()
        self.attn = PairwiseHullAttention(dim, heads, petals, use_s4d)
        self.hff  = VectorHull(dim, [dim*2], petals)  # convex FFN
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # mixing weights a1,a2: raw, convert to alpha in (0,1)
        self.a1 = nn.Parameter(torch.zeros(()))
        self.a2 = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mix(x, y, a_raw):
        # alpha = softplus(a_raw)/(1+softplus(a_raw)) in (0,1)
        alpha = F.softplus(a_raw)/(1+F.softplus(a_raw))
        return (1-alpha)*x + alpha*y

    def forward(self, x: torch.Tensor, mask=None):  # x: (B,S,E)
        # attention with residual mixing
        x = self._mix(x, self.attn(self.ln1(x), mask), self.a1)
        # convex feed-forward with residual mixing
        x = self._mix(x, self.hff(self.ln2(x)), self.a2)
        return x

class ConvexGPT(nn.Module):
    """GPT-like causal model using OmniHull blocks."""
    def __init__(self, vocab_size: int, embed_dim: int, depth: int, heads: int, petals: int):
        super().__init__()
        # token embedding: (vocab_size -> embed_dim)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        # stack of OmniHullBlocks
        self.blocks = nn.ModuleList([
            OmniHullBlock(embed_dim, heads, petals, use_s4d=(i < depth-1))
            for i in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(S: int, device: torch.device) -> torch.Tensor:
        # Lower-triangular mask (S,S), True allows attend
        return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))       
            .unsqueeze(0).unsqueeze(1)    # shape (1,1,S,S)

    def forward(self, idx: torch.Tensor):  # idx: (B, S)
        B,S = idx.shape
        device = idx.device
        x = self.token_emb(idx)         # (B,S,embed_dim)
        mask = self._causal_mask(S, device)
        for blk in self.blocks:
            x = blk(x, mask)
        # final norm + linear head: (B,S,vocab_size)
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


