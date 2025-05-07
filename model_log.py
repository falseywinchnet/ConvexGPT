#copyright joshuah rainstar 2025 joshuah.rainstar@gmail.com
#licensed under convexgpt license if you do not agree please close this and delete any files related to this program

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List,Literal



        
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
        log_lam_mag = lam_mag.log()
        scaled_i = i[None] * log_lam_mag[:, None]
        amp = torch.exp(log_gain[:, None] + scaled_i)

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
    def __init__(self, embed_dim: int, heads: int):
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
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # reshape to heads
        q = q.view(B, S, self.heads, self.d_k).transpose(1,2)
        k = k.view(B, S, self.heads, self.d_k).transpose(1,2)
        v = v.view(B, S, self.heads, self.d_k).transpose(1,2)
        return q, k, v

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


class LogBatchedHull(nn.Module):
    def __init__(self, dim: int, petals: int, kind: int):
        """
        Hull module with log-space ICNN aggregation.
        kind = 0 for scalar output, 1 for vector output.
        """
        super().__init__()
        assert kind in (0, 1), "kind must be 0 (scalar) or 1 (vector)"
        self.kind = kind
        self.dim = dim
        self.petals = petals

        # Batched ICNN weights for log-space output
        D, P = dim, petals
        self.d1, self.d2 = 2 * D, D
        self.weight_raw_0 = nn.Parameter(self._init_weight(P, self.d1, D))
        self.bias_0       = nn.Parameter(torch.zeros(P, self.d1))
        self.weight_raw_1 = nn.Parameter(self._init_weight(P, self.d2, self.d1))
        self.bias_1       = nn.Parameter(torch.zeros(P, self.d2))
        self.z_weight     = nn.Parameter(torch.empty(P, self.d1, self.d2))
        nn.init.kaiming_uniform_(self.z_weight, a=math.sqrt(5))
        self.gate_raw_0   = nn.Parameter(torch.full((P,), -3.0))
        self.gate_raw_1   = nn.Parameter(torch.full((P,), -3.0))
        self.output_bias  = nn.Parameter(torch.zeros(P, D))

        self.act = nn.Softplus()
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.register_buffer('eps', torch.tensor(1e-6))
        self.register_buffer('creative', torch.tensor(True))

        # Gating layer (in log-space)
        self.gate_layer = nn.Linear(D, 1, bias=True)

    def _init_weight(self, petals: int, d_out: int, d_in: int) -> torch.Tensor:
        w = torch.empty(petals, d_out, d_in)
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / d_in))
            nn.init.normal_(w, mean=mean, std=0.2)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D)
        B, H, S, D = x.shape
        P = self.petals
    
        # 1) Gating (stable log1p alternative)
        gate_pre = F.softplus(self.gate_layer(x))               # (B,H,S,1)
        # log(1 − exp(−gate_pre)) == −Softplus(−gate_pre)
        log_gate = -F.softplus(-gate_pre)                       # (B,H,S,1)
        xg = x + log_gate                                       # (B,H,S,D)
    
        # 2) τ in log-domain
        r   = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B,H,S,1)
        tau = torch.exp(0.30343 * r + 0.22159).clamp(max=20.0)                             # (B,H,S,1)
    
        # 3) Flatten for per-petal ICNN
        x_flat = xg.reshape(B*H*S, D)                            # (N, D) with N=B·H·S
    
        # 4) First layer (stable: softplus → clamp_min → log)
        w0 = F.softplus(self.weight_raw_0).pow(2)                # (P,2D,D)
        g0 = torch.sigmoid(self.gate_raw_0).view(P,1,1)          # (P,1,1)
        x0 = x_flat.unsqueeze(0).expand(P, -1, -1)               # (P,N,D)
        z0 = F.softplus(torch.bmm(x0, w0.transpose(1,2)) * g0).clamp_min(self.eps).log()# (P,N,2D)
    
        # 5) Second layer + residual
        w1 = F.softplus(self.weight_raw_1).pow(2)                # (P,D,2D)
        g1 = torch.sigmoid(self.gate_raw_1).view(P,1,1)          # (P,1,1)
        x1 = torch.bmm(z0.exp(), w1.transpose(1,2)) + self.bias_1.unsqueeze(1)  # (P,N,D)
        res = torch.bmm(z0.exp(), self.z_weight)                 # (P,N,D)
        z1 = torch.log(F.softplus(x1 * g1 + res) + self.eps)     # (P,N,D)
    
        # 6) Add bias, unflatten
        out = z1 + self.output_bias.unsqueeze(1)                 # (P,N,D)
        out = out.permute(1,0,2).contiguous().view(B, H, S, P, D) # (B,H,S,P,D)
    
        # 7) Scale by τ
        out = out * tau.unsqueeze(-1)                            # (B,H,S,P,D)
    
        # 8) Transpose for LSE over petals
        out = out.transpose(-2, -1)                              # (B,H,S,D,P)
    
        # 9) Log-sum-exp
        m   = out.max(dim=-1, keepdim=True)[0]                   # (B,H,S,D,1)
        lse = m + (out - m).exp().sum(dim=-1, keepdim=True).log()# (B,H,S,D,1)
        lse = lse.squeeze(-1)                                    # (B,H,S,D)
    
        # 10) Divide by τ (still has shape (B,H,S,1))
        result_log = lse / tau                                   # (B,H,S,D)
    
        # 11) Final collapse if scalar
        if self.kind == 0:
            return result_log.mean(dim=-1)                       # (B,H,S)
        else:
            return result_log                      
        
class LogConvexPositionalBias(nn.Module):
    """
    log_bias(i,j) = log(exp(-w * |i-j|)) = -w * |i-j|
    Directly outputs log-domain positional bias.
    """
    def __init__(self, heads):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(heads))  # unconstrained
        self.softplus = nn.Softplus()

    def forward(self, S: int) -> torch.Tensor:
        device = self.w_raw.device
        w = self.softplus(self.w_raw)  # (H,)
        pos = torch.arange(S, device=device, dtype=torch.float32)
        dist = (pos[None, :] - pos[:, None]).abs()  # (S, S)
        log_bias = -w[:, None, None] * dist  # (H, S, S)
        return log_bias
    


class LogConvexMixer(nn.Module):
    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.register_buffer('eps', torch.tensor(1e-6))
        self.register_buffer('noise_scale', torch.tensor(1e-5))

        self.score_q = LogBatchedHull(d_k, petals, kind=0)  # log-domain scalar hull
        self.score_k = LogBatchedHull(d_k, petals, kind=0)  # log-domain scalar hull
        self.gate = nn.Softplus()
        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)

        self.register_buffer("creative", torch.tensor(True))

    def forward(
        self,
        q: torch.Tensor,           # (B, H, S, d_k)
        k: torch.Tensor,           # (B, H, S, d_k)
        v: torch.Tensor,           # (B, H, S, d_k)
        extra_score: torch.Tensor, # (B, H, S, S)
        mask: torch.Tensor         # (B, H, S, S)
    ) -> torch.Tensor:            # returns (B, H, S, d_k)

        B, H, S, D = q.shape

        # 1. Temperature scaling (tau)
        gate_q = self.gate(q)
        q = q * gate_q
        r = torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
        tau = torch.exp(0.30343 * r + 0.22159).clamp(max=20.0)  # (B, H, S, 1)

        # 2. Scalar ICNN hull scores (already in log-space)
        fq = self.score_q(q)  # (B, H, S), log-domain
        gk = self.score_k(k)  # (B, H, S), log-domain

        if self.creative:
            qn = (torch.rand_like(q) - 0.5) * self.noise_scale
            kn = (torch.rand_like(k) - 0.5) * self.noise_scale
            fq_ = self.score_q(q + qn)
            gk_ = self.score_k(k + kn)
            fq = fq - 0.1 * (fq_ - fq).detach()
            gk = gk - 0.1 * (gk_ - gk).detach()

        # 3. Random feature kernel in log-domain
        log_phi_q = torch.log(F.softplus(self.lin_h_q(q).clamp(max=20.0)) + self.eps)
        log_phi_k = torch.log(F.softplus(self.lin_h_k(k).clamp(max=20.0)) + self.eps)
        sum_ab = log_phi_q.unsqueeze(-2) + log_phi_k.unsqueeze(-3)  # (B, H, S, S, r)

        # Manual fused log-sum-exp over feature axis
        m = sum_ab.max(dim=-1, keepdim=True)[0]  # (B, H, S, S, 1)
        logK = m + (sum_ab - m).exp().sum(dim=-1, keepdim=True).log()  # (B, H, S, S, 1)
        logK = logK.squeeze(-1)  # (B, H, S, S)
        # 4. Attention scores (still in log-domain)
        scores = fq.unsqueeze(-1) + gk.unsqueeze(-2) + logK + extra_score + mask


        # 5. Apply temperature and softmax in log-space
        tau4 = tau.squeeze(-1)  # (B, H, S)
        logits = scores * tau4.unsqueeze(-1)  # (B, H, S, S)
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        weights = log_weights.exp()  # (B, H, S, S)

        # 6. Weighted sum of values
        # 6. Weighted sum of values (do it in value-space, then re-log)
        w_mat   = weights.reshape(B * H, S, S)           # (B·H, S, S)
        v_val   = v.exp().reshape(B * H, S, D)           # back to value-space
        out_val = w_mat.bmm(v_val)                       # (B·H, S, D)
        # prevent log(0) and clamp extreme values
        out_log = (out_val + self.eps).clamp(min=self.eps).log()
        return out_log.reshape(B, H, S, D)


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
        self.softplus = nn.Softplus()

    def forward(self,
                x: torch.Tensor,   # (B, T, 2*M)
                mask: torch.Tensor # (1, 1, T, T) boolean causal+padding mask
    ) -> torch.Tensor: #return nothing
        B, T, D2 = x.shape #1128
        M = self.M
        assert D2 == 2 * M 

        device = x.device
        dtype = x.dtype

        # 1) extract content slots ci
        x_c = x[..., 0::2]                  # (B, T, M)

        # 2) build deterministic distance kernel W[i,j] = 1 / (1 + |i-j|)
        # 2–3) build and apply convex mask to distance kernel
        with torch.no_grad():
            pos = torch.arange(T, device=device, dtype=dtype)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            W = 1.0 / (dist + 1.0)                              # (T, T)
            if mask is not None:
                W = W * mask.view(T, T).to(dtype)              # soft gate
            W = W / W.sum(-1, keepdim=True).clamp(min=1e-6)    # row-normalize

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

        # 7) write into odd slots ϕm. gated_phi is computed from content-only paths. therefore: this is safe.
        x = x.clone()
        x[..., 1::2] = gated_phi
        return x

        
# ----------------------------------------------------------------------
#   Pairwise Hull Attention (mask‑aware)
# ----------------------------------------------------------------------
class LogPairwiseHullAttention(nn.Module):
    def __init__(self, embed_dim, heads, moe_petals, use):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        self.eps = 1e-6

        # Premixer remains in value-space
        self.pre = S4PreMix(embed_dim, heads) if use == 0 else LinearPreMix(embed_dim, heads)

        # Log-domain mixer
        self.mixer = LogConvexMixer(self.d_k, moe_petals, r=2 * self.d_k)

        # Positional bias (in log space)
        self.pos = LogConvexPositionalBias(heads)

        # Log-domain phase channelizer
        self.phase = InterleavedPhaseChannelizer(embed_dim)

        # Output projection
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x_log: torch.Tensor, mask: torch.Tensor = None,mask_exp: torch.Tensor = None) -> torch.Tensor:
        B, S, E = x_log.shape
        x = self.phase(x_log.exp(),mask_exp)

        # --- Step 1: In-place log-domain φ-channelization ---
        Q_val, K_val, V_val = self.pre(x)

        mean = 0.5 * (Q_val.mean() + K_val.mean())
        std = 0.5 * (Q_val.std() + K_val.std())
        Q_val = (Q_val - mean) / std
        K_val = (K_val - mean) / std
        
        Q_log = Q_val.clamp_min(self.eps).log()
        K_log = K_val.clamp_min(self.eps).log()
        V_log = V_val.clamp_min(self.eps).log()

        # --- Step 3: Positional bias in log-space ---
        bias = self.pos(S).unsqueeze(0)  # (1, H, S, S)

        # --- Step 4: Hull attention mixer ---
        y_log = self.mixer(Q_log, K_log, V_log, extra_score=bias, mask=mask)  # (B, H, S, d_k)

        # --- Step 5: Recombine heads + output proj ---
        y_log = y_log.transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.W_O(y_log)

# ----------------------------------------------------------------------
#   OmniHull Block
# ----------------------------------------------------------------------
class LogOmniHullBlock(nn.Module):
    def __init__(self, dim, heads, moe_petals, use=0):
        super().__init__()
        self.attn = LogPairwiseHullAttention(dim, heads, moe_petals, use=use)
        self.hff  = LogBatchedHull(dim, petals=moe_petals, kind=1)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.a1, self.a2   = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))

    @staticmethod
    def _log_mix(log_x: torch.Tensor, log_y: torch.Tensor, a_raw: torch.Tensor) -> torch.Tensor:
        """
        Convex mix of two log-domain tensors with bounded alpha and mean-centering.

        Args:
          log_x: (B, S, D) log-values from the identity path
          log_y: (B, S, D) log-values from the new path
          a_raw: scalar or tensor broadcastable to (B, S, D) controlling mix

        Returns:
          (B, S, D) mixed log-values, mean-centered over S to prevent drift.
        """
        # 1) Bound alpha to [0.1, 0.9]
        alpha = 0.1 + 0.8 * torch.sigmoid(a_raw)       # ensures no branch is dropped
        log_alpha = torch.log(alpha + 1e-6)
        log_one_minus = torch.log(1.0 - alpha + 1e-6)

        # 2) Core convex combination in log-space
        log_mix = torch.logaddexp(log_one_minus + log_x,
                                  log_alpha      + log_y)

        # 3) Mean-center over the sequence dimension (dim=-2)
        mean_mix = log_mix.mean(dim=-2, keepdim=True)
        return log_mix - mean_mix

    def forward(self, x_log: torch.Tensor, mask=None,mask_exp=None) -> torch.Tensor:
        mean_log = x_log.mean(dim=-1, keepdim=True)  # (B, S, 1)
        x_log = x_log - mean_log#perform normalization
        x_log = self._log_mix(x_log, self.attn(x_log, mask,mask_exp), self.a1)
        x_ = x_log.unsqueeze(1)            # (B, 1, S, D)
        y_ = self.hff(x_)                  # (B, 1, S, D)
        x_log = self._log_mix(x_log, y_.squeeze(1), self.a2)
        return x_log

# ----------------------------------------------------------------------
#   GPT Wrapper with Causal Mask
# ----------------------------------------------------------------------
class LogConvexGPT(nn.Module):
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

        # Embeddings only for even channels [0,2,4,...]
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # Blocks operate on full embed_dim
        self.blocks = nn.ModuleList([
        LogOmniHullBlock(
            self.embed_dim,
            heads,
            moe_petals,
            use=1 if i == 0 or i == depth - 1 else 0
                  )
            for i in range(depth)
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab_size, bias=False)
        self.set_creativity(creativity)

    @staticmethod
    def _causal_mask(S: int, device: torch.device, steepness: float = 50.0):
        pos = torch.arange(S, device=device, dtype=torch.float)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # Positive for future positions
        # Exponential barrier (always convex)
        mask = torch.exp(-steepness * F.softplus(diff))
        return mask.unsqueeze(0).unsqueeze(1)

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
        print("you shouldnt run this model. it has serious mode collapse issues. ")
        assert False
        B, S = idx.shape
        device = idx.device
        #stack 0::2 as embeddings, 1::2 as zeros for positional embeddings
        E = self.token_emb.weight                             # (V, D)
        E_proc = (1 + torch.tanh(E)).log()                    # log-domain safe
        embeddings = F.embedding(idx, E_proc)                 # (B, S, D)
        x = torch.stack([embeddings, torch.zeros_like(self.token_emb(idx))], dim=-1).reshape(idx.shape[0], idx.shape[1], 2 * self.token_emb.embedding_dim)
    

        # 3) build causal mask
        mask = self._causal_mask(S, device)          # (1, 1, S, S)
        mask_exp = mask.exp()
        # 4) apply each block (which will write φ into odd slots)
        for blk in self.blocks:
            x = blk(x, mask,mask_exp)

        # 5) final layernorm + head
        x = self.ln_f(x.exp())                             # (B, S, embed_dim)
        logits = self.head(x)                        # (B, S, vocab_size)
        return logits




def train_epoch_log():
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # --- Model forward pass ---
        logits = model(xb)                   # (B, T, V), in value space (i.e. exp-domain)
        log_probs = F.log_softmax(logits, dim=-1)
        B, T, V = logits.shape
        loss = F.nll_loss(log_probs.view(B*T, V), yb.view(B*T))


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(loss.item())
        total_loss += loss.item()
        losses.append(loss.item())
    return total_loss / len(train_loader)


def fenchel_decode(logits, tau=1.0, iters=3):
    """Fenchel‑dual KL‑regularised projection of -logits (energy)."""
    energy = -logits                        # (B,V)
    p = torch.full_like(energy, 1.0 / energy.size(-1))  # uniform start
    for _ in range(iters):
        p = torch.softmax((-energy / tau) + p.log(), dim=-1)
    return p


use_fenchel     = False        # True to use Fenchel decode
tau             = 1.0          # Fenchel temperature
max_new_tokens  = 512
top_k           = 25
block_size      = 256
temperature     = 1.0

context_str = "To be, or not to be,"
context_ids = torch.tensor([[stoi[c] for c in context_str]], dtype=torch.long, device=device)
generated = context_ids.clone()

model.eval()
with torch.no_grad():
    for _ in range(max_new_tokens):
        input_ids = generated[:, -block_size:]                # (1, T)
        logits = model(input_ids)[:, -1, :] / temperature     # (1, V), still log-values

        log_probs = F.log_softmax(logits, dim=-1)             # Normalize

        if top_k is not None:
            v, _ = torch.topk(log_probs, top_k)
            log_probs[log_probs < v[:, [-1]]] = -1e10         # Top-k filter in log-space

        if use_fenchel:
            probs = fenchel_decode(log_probs, tau=tau, iters=3)
        else:
            probs = log_probs.exp()                           # Back to value-space

        next_id = torch.multinomial(probs, num_samples=1)     # Sample
        generated = torch.cat([generated, next_id], dim=1)

print('> ', ''.join(itos[i] for i in generated[0].tolist()))
