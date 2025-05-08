#copyright joshuah.rainstar@gmail.com 2025
#protected under ConvexGPT license and copyright -proprietary software

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List,Literal

#c1 everywhere BUT logsumexp because of the max
#yes, efficient,Kolmogorov-Arnold Networks without quadratic explosion, delivered here
        
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



'''
Canonical φ(x) builds φ ∈ ℝ^{B, D_in, K}.

When K=8, D_in=8192, B=128 → φ alone = 8.3M floats.
FFT variant avoids this explicit K-expansion and instead processes one padded array per batch element (ℝ^{B, L} and L ≈ D_in).
caveat: it must be trained. it varies by d_in only. so you need pre-learned kernels.
in practice the error is fairly contained. 

naively, our cost is O(N * D_in * D_in*2 * K) for this model since d_out is always twice d_in and K is always 8:
that's a quadratic term times 16: O(N*embed_dim^2*16).
This fft form scales as : O(N * D_in * (log D_in + K))
In practice, this converges on quasi-linear growth and enables much more massive KAN utilization with minimal cost, provided
you pre-learn the kernels. however, since x varies, how do you prelearn the kernel? simple. you zero-mean it. this ensures the kernels are scale invariant.
This means our more economical KAN module can be quickly instantiated with desired behaviors that approximate nonlinear transformations.
secondly, use fixed scales. in principle, this is a reasonable configuration.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Configuration
D_in = 64
D_out = 128
K = 8
batch_size = 128
epochs = 5000
lr = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
from torch.optim.optimizer import Optimizer
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

class FixedLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)

# Canonical φ(x) with fixed shifts
class CanonicalPhi(nn.Module):
    def __init__(self, shifts, W, b, D_in):
        super().__init__()
        self.shifts = shifts
        self.W = W
        self.b = b
        self.norm = FixedLayerNorm(D_in)

    def forward(self, x):
        x_norm = self.norm(x)  # Normalize x
        xb = x_norm.unsqueeze(-1) + self.shifts.view(1, 1, K)  # (N, D_in, K)
        phi = F.softplus(xb)  # (N, D_in, K)
        return torch.einsum("nik,dik->nd", phi, self.W) + self.b  # (N, D_out)



# FFT encoder with decoder (trainable linear layer)
class FFTSharedKFilters(nn.Module):
    def __init__(self, K_shifts, D_in, shifts_values):
        super().__init__()
        self.K_shifts = K_shifts
        self.D_in = D_in
        self.D_out = 2 * D_in
        self.L = D_in
        self.Lf = self.L // 2 + 1

        self.log_amp_shifts = nn.Parameter(torch.randn(self.K_shifts, self.Lf) * 0.01)
        self.phase_shifts = nn.Parameter(torch.randn(self.K_shifts, self.Lf) * 0.01)
        self.final_bias = nn.Parameter(torch.zeros(self.D_out))
        self.norm = FixedLayerNorm(D_in)  # Apply normalization

    def forward(self, x):
        N = x.shape[0]
        x_norm = self.norm(x)  # Normalize x

        Xf = torch.fft.rfft(x_norm, n=self.L, dim=-1)  # (N, Lf)
        Kf_shifts = torch.exp(self.log_amp_shifts) * torch.exp(-1j * self.phase_shifts)
        Yf_k = Xf.unsqueeze(1) * Kf_shifts.unsqueeze(0)  # (N, K_shifts, Lf)
        x_time_k = torch.fft.irfft(Yf_k, n=self.L, dim=-1)  # (N, K_shifts, D_in)
        phi_k_d = F.softplus(x_time_k)  # (N, K_shifts, D_in)
        S_n_learned = phi_k_d.sum(dim=[1, 2])  # (N,)
        return S_n_learned.unsqueeze(-1) + self.final_bias.unsqueeze(0)  # (N, D_out)
        
        return pred

shifts = torch.linspace(0.0, 1.0, 8).to(device)

# Initialize models and training targets
fft_model = FFTSharedKFilters(8, D_in, D_out).to(device)
W_ref = nn.Parameter(torch.ones(D_out, D_in, 8), requires_grad=False).to(device)
b_ref = nn.Parameter(torch.zeros(D_out), requires_grad=False).to(device)
canonical_phi_model = CanonicalPhi(shifts, W_ref, b_ref, D_in).to(device)

opt = torch.optim.AdamW(fft_model.parameters(), lr=3e-2)
losses = []

start_time = time.time()
for epoch in range(1, epochs + 1):
    x = torch.randn(batch_size, D_in, device=device)
    with torch.no_grad():
        target  = canonical_phi_model(x)

    pred = fft_model(x)
    loss = F.mse_loss(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}, loss {loss.item():.4e}")
end_time = time.time()


# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label="FFT Encoder + Linear")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training FFT Encoder to Match φ(x) with Trainable Linear Layer")
plt.grid(True)
plt.legend()
plt.show()

# Report parameters
param_count = sum(p.numel() for p in fft_model.parameters())
print(f"Trainable parameters: {param_count}")
print(f"Training time: {end_time - start_time:.2f} seconds")





'''
class KCN(nn.Module):
    def __init__(self, in_dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.P       = petals
        self.k       = 8 #empirically and theoretically, shouldnt exceed min(batch,petals), not important
        D_in, D_out = self.in_dim, self.out_dim

        # — edge-wise basis weights — (P, D_out, D_in, k)
        # we'll later flatten the last two dims to match BatchedICNN’s use of bmm
        self.weight = nn.Parameter(
            torch.stack([
                self._init_weight(1, D_out, D_in)[0] for _ in range(self.k)
            ], dim=-1)  # shape: (D_out, D_in, k)
        )
        self.bias = nn.Parameter(torch.zeros(D_out))


        self.weight2 = nn.Parameter(self._init_weight(petals, D_out, D_out))
        self.bias2   = nn.Parameter(torch.zeros(petals, D_out))
        self.gate_raw2 = nn.Parameter(torch.full((petals,), -3.0))

        # shared Softplus shifts: shape (1, D_in, k)
        shifts = torch.linspace(-1.0, 1.0, self.k ).view(1, 1, self.k )  # shape: (1, 1, k)
        self.register_buffer("shifts", shifts)              # broadcast when used

        # residual projection exactly like BatchedICNN: (P, 2*D_in, D_out)
        self.z_weight = nn.Parameter(torch.empty(petals, 2 * D_in, D_out))
        nn.init.kaiming_uniform_(self.z_weight, a=math.sqrt(5))

        # gating scalars per petal
        self.gate_raw = nn.Parameter(torch.full((petals,), -3.0))

        # final bias per petal/output
        self.output_bias = nn.Parameter(torch.zeros(petals, D_out))
        self.act = nn.Softplus()
   
    
    def _init_weight(self, petals: int, d_out: int, d_in: int) -> torch.Tensor:
        w = torch.empty(petals, d_out, d_in)
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / d_in))
            nn.init.normal_(w, mean=mean, std=0.2)
        return w
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D_in)
        orig = x.shape
        x_flat = x.reshape(-1, self.in_dim)       # (N, D_in)
        N = x_flat.size(0)

        # ------------------------------------------------------------
        # 1) compute basis activations φ(x): (N, D_in, k)
        xb  = x_flat.unsqueeze(-1) + self.shifts       # (N, D_in, k)
        phi = F.softplus(xb)                           # (N, D_in, k)

        # flatten φ for bmm
        # reshape to (N, D_in * k)
        phi_flat = phi.reshape(N, self.in_dim * self.k)  

        # reshape weight to (D_in * k, D_out)
        weight_flat = self.weight.reshape(self.in_dim * self.k, self.out_dim)

        # shared projection
        x_proj_shared = phi_flat @ weight_flat + self.bias  # (N, D_out)

        # duplicate to all petals
        x_proj = x_proj_shared.unsqueeze(0).expand(self.P, N, self.out_dim)  # (P, N, D_out)
        g1 = torch.sigmoid(self.gate_raw).view(self.P, 1, 1)
        z0 = self.act(x_proj * g1)  # (P, N, D_out)

        # ------------------------------------------------------------
        # 3) second projection
        w2 = F.softplus(self.weight2).pow(2).transpose(1, 2)  # (P, D_out, D_out)
        z1 = torch.bmm(z0, w2) + self.bias2.unsqueeze(1)      # (P, N, D_out)
        g2 = torch.sigmoid(self.gate_raw2).view(self.P, 1, 1)
        z1 = self.act(z1 * g2)

        # ------------------------------------------------------------
        # 4) residual projection
        x_res_in = torch.cat([x_flat, x_flat], dim=-1)                       # (N, 2*D_in)
        x_res_exp = x_res_in.unsqueeze(0).expand(self.P, N, 2 * self.in_dim)  # (P, N, 2*D_in)
        x_res = torch.bmm(x_res_exp, self.z_weight)                          # (P, N, D_out)

        # ------------------------------------------------------------
        # 5) combine and output
        z_final = self.act(z1 + x_res) + self.output_bias.unsqueeze(1)       # (P, N, D_out)
        out = z_final.permute(1, 0, 2)                                       # (N, P, D_out)
        new_shape = list(orig[:-1]) + [self.P, self.out_dim]
        return out.reshape(new_shape)      


class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.softplus(self.lin(x))      # convex, ≥ 0
        return 1.0 - torch.exp(-u)       # convex, ∈ (0,1)

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
        self.petals = KCN(self.in_dim, petals, out_dim=self.out_dim)
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

class ConvexPositionalBias(nn.Module):
    """
    Bias(i,j) = - w * |i-j|    with   w ≥ 0  (learned per head)
    Convex in positional indices; monotone non‑increasing with distance.
    """
    def __init__(self, heads):
        super().__init__()
        self.w_raw = nn.Parameter(torch.zeros(heads))   # raw parameter
        self.softplus = nn.Softplus()

    def forward(self, S: int):
        device = self.w_raw.device
        w = self.softplus(self.w_raw)                      # (H,)
        pos  = torch.arange(S, device=device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S,S)
        bias = - w[:, None, None] * dist                # (H,S,S)
        return bias
    
        
class ConvexMixer(nn.Module):
    def __init__(self, d_k: int, petals: int, r: int):
        super().__init__()
        self.register_buffer('nu',  torch.tensor(1.64872127070012814684865078781416357165))
        self.register_buffer('eps',   torch.tensor(1e-6))
        self.register_buffer('noise_scale', torch.tensor(1e-5))

        self.score_q = ScalarHull(d_k, petals)
        self.score_k = ScalarHull(d_k, petals)
        self.gate    = nn.Softplus()
        self.lin_h_q = nn.Linear(d_k, r, bias=False)
        self.lin_h_k = nn.Linear(d_k, r, bias=False)
        self.register_buffer("creative", torch.tensor(True))
        self.fused = FusedLogSumExp(dim=-1)                          # or cache this module

    def forward(self,
                q: torch.Tensor,           # (B,H,S,d_k)
                k: torch.Tensor,           # (B,H,S,d_k)
                v: torch.Tensor,           # (B,H,S,d_k)
                extra_score: torch.Tensor, # (B,H,S,S)
                mask: torch.Tensor         # (B,H,S,S)
               ) -> torch.Tensor:          # returns (B,H,S,d_k)

        B, H, S, D = q.shape

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
        # 1) compute elementwise sums in log-space
        a     = torch.log(phi_q + self.eps)                     # (B,H,S,r)
        b     = torch.log(phi_k + self.eps)                     # (B,H,S,r)
        sum_ab = a.unsqueeze(-2) + b.unsqueeze(-3)              # (B,H,S,S,r)
        
        # 2) log-sum-exp over the feature axis via your fused LSE
        logK  = self.fused(sum_ab).squeeze(-1)                       # (B,H,S,S)

        # ——— 4) build & mask scores ———
        log_gate = torch.log(mask.clamp_min(self.eps))         # (1,1,S,S) or (B,H,S,S)
        extra_score = extra_score + log_gate                   # soft gating of positional bias
        scores = fq.unsqueeze(-1) + gk.unsqueeze(-2) + logK + extra_score  # (B,H,S,S)

        # ——— 5) 4-D tempered LSE hull, in log space ———
        tau4 = tau.squeeze(-1)                # (B,H,S)
        logits = scores * tau4.unsqueeze(-1)
        logits = logits + torch.log(mask.clamp_min(self.eps))
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        weights = torch.exp(log_weights)                                      # (B,H,S,S)

        # ——— 6) single batched bmm for output ———
        w_mat = weights.reshape(B * H, S, S)  # (B*H, S, S)
        v_mat = v.reshape(B * H, S, D)        # (B*H, S, D)
        out   = w_mat.bmm(v_mat)              # (B*H, S, D)
        return out.reshape(B, H, S, D)
        
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
    ) -> None: #return nothing
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

        
# ----------------------------------------------------------------------
#   Pairwise Hull Attention (mask‑aware)
# ----------------------------------------------------------------------
class PairwiseHullAttention(nn.Module):
    def __init__(self, embed_dim, heads,moe_petals, use):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        if use==0:
            self.pre = S4PreMix(embed_dim, heads)
        else:
            self.pre = LinearPreMix(embed_dim, heads)
        self.mixer = ConvexMixer(self.d_k, moe_petals, self.d_k*2)#dont need many for scoring
        self.pos = ConvexPositionalBias(heads)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.phase = InterleavedPhaseChannelizer(embed_dim)
        self.register_buffer('noise_scale', torch.tensor(1e-5))
        self.register_buffer("creative", torch.tensor(True))

    def forward(self, x, mask=None):
        self.phase(x,mask) #apply in-place positional phasing
        B, S, E = x.shape
        Q, K, V= self.pre(x)
        mean = 0.5 * (Q.mean() + K.mean())
        std  = 0.5 * (Q.std()  + K.std())
        Q = (Q - mean) / std
        K = (K - mean) / std
        
        bias = self.pos(S).unsqueeze(0)
        
        y = self.mixer(Q, K, V, extra_score=bias, mask=mask)
        
        y = y.transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.W_O(y)

# ----------------------------------------------------------------------
#   OmniHull Block
# ----------------------------------------------------------------------
class OmniHullBlock(nn.Module):
    def __init__(self, dim, heads, moe_petals, use=0):
        super().__init__()
        self.attn = PairwiseHullAttention(dim, heads, moe_petals, use=use)
        self.hff  = VectorHull(dim, moe_petals)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.a1, self.a2   = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))

    def _mix(self, x, y):
        return 0.9 * x + 0.1 * y

    def forward(self, x: torch.Tensor, mask=None):
        x1 = self.attn(self.ln1(x), mask)
        x1 = self.hff(self.ln2(x1))
        x = self._mix(x1,x)
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
        OmniHullBlock(
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

    #@staticmethod
    #def _causal_mask(S: int, device: torch.device) -> torch.Tensor:
     #   # shape (1, 1, S, S) where True = allowed
     #   return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device)) \
     #              .unsqueeze(0).unsqueeze(1)
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
        B, S = idx.shape
        device = idx.device
        #stack 0::2 as embeddings, 1::2 as zeros for positional embeddings
        E = self.token_emb.weight                             # (V, D)
        E_proc = (torch.tanh(E))                    # log-domain safe
        embeddings = F.embedding(idx, E_proc)                 # (B, S, D)
        x = torch.stack([embeddings, torch.zeros_like(self.token_emb(idx))], dim=-1).reshape(idx.shape[0], idx.shape[1], 2 * self.token_emb.embedding_dim)

        # 3) build causal mask
        mask = self._causal_mask(S, device)          # (1, 1, S, S)

        # 4) apply each block (which will write φ into odd slots)
        for blk in self.blocks:
            x = blk(x, mask)

        # 5) final layernorm + head
        x = self.ln_f(x)                             # (B, S, embed_dim)
        logits = self.head(x)                        # (B, S, vocab_size)
        return logits


