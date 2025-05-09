#copyright joshuah.rainstar@gmail.com 2025
#protected under license and copyright -proprietary software

import torch
from torch.optim.optimizer import Optimizer

'''
  Through the misty dawns of Preserve Planet 889a, a highly advanced predator moves with an eerie precision that betrays its origins. 
  It hunts not by chance or mere pursuit, but through a higher mathematics of flesh, a calculus of predation. 
  The beast's neural pathways, under the extreme pressure of an always changing topology on a planet deformed by random spacetime ripples,
  evolved to solve trajectory equations no human mathematician could formulate on the fly, allowing them to intersect with prey at the exact point
  where escape became theoretical rather than possible. Each muscle contraction serves a purpose; each calorie burned achieves maximum effect. 
  A perfect killing machine, refined through millennia of natural selection, as successful as the dragonfly.

    
  A necessary adaptation comes with temporal camouflaging prey, an adaptive but rare pest animal with the power to move with extreme deception.
  and more importantly the ability to superimpose their presence holographically to defy spatial predictions, with almost quantum effectiveness.
  Facing these circumstances, successful hunts plummet, triggering a metamorphosis encoded for scenarios requiring extreme risk taking for survival. 
  What emerges from the crystalline chrysalis bears only a superficial resemblance to the fearsome predator that entered it, a brightly reflective, flimsy thing.
  Most striking are the membranous structures extending from its flanks,not only true wings but also thermal sensing surfaces that integrate energy cost dynamics.
  Now fragile, these fly by night in silence absolute with equilibrium matched body temperature over the most difficult terrain hunt the most challenging prey.
  The Direwolf.

'''
def direwolf_update(p: torch.Tensor,
                g: torch.Tensor,
                state_p: torch.Tensor,
                lr: float):

    etcerta: float = 0.367879441
    et:      float = 1.0 - etcerta

    # same logic as before
    update    = state_p * et + g * etcerta
    new_state = state_p * et + update * etcerta
    sign_agree = torch.sign(update) * torch.sign(g)
    update    = update + (torch.rand_like(update)*2 - 1) * etcerta * update
    g_new     =  -lr * update
    return g_new, new_state
class Direwolf(Optimizer):
        def __init__(self, model, lr=1e-3):
            defaults = dict(lr=lr)
            super().__init__(model, defaults)
            self.model = model
            self.state = {}
            self.lr = lr
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.state[param] = {
                        "p" : torch.zeros_like(param.data),
                        "temp": torch.zeros_like(param.data),
                        "res_pos": torch.zeros_like(param.data),
                        "res_neg": torch.zeros_like(param.data),
                        "elasticity": torch.zeros_like(param.data),
                        "last_grad_sign": torch.zeros_like(param.data),
                        "frozen": torch.zeros_like(param.data, dtype=torch.bool)
                    }

        # 1. Temperature thresholds (normalized energy ratios)
        T_CRYSTALLIZE   = 0.1787775950  # When bulk free energy per vol drops below 0.17878 of formation energy,
                                          # nucleation becomes thermodynamically favorable (1 – 1/e).
        
        T_MIN           = 0.0119188     # The thermal‐equilibrium “floor” (1/83.901) at triple point energy,
                                          # below which parameters are fully consolidated (frozen).
        
        PERC_CRYSTAL    = 0.59274621    # ≈ pc for 2D bond percolation on ℤ².
                                          # When ≥59.27% of temps < T_MIN, the petal percolates into a contiguous “ice” phase.
        
        # 2. Resistance & Elasticity Dynamics
        DELTA_RESIST    = 0.003603      # Resistance increment per unidirectional update.
                                          # Derived as one‐tenth of the per‐step energy dissipation (γ),
                                          # grounding accumulation rate in phonon‐conduction energetics, subdiffusive viscoelastic stiffening.
        
        GAMMA_DECAY     = 0.03603       # Fractional resistance decay on gradient sign flip.
                                          # Matches the ratio of a single‐step conductive energy transfer in 130fs
                                          # (1.71×10⁻²⁰ J) to total formation energy (4.75×10⁻¹⁹ J).
        
        ELASTICITY_K    = 0.0119188     # Elasticity gain per unit |Δx|.
                                          # Equal to T_MIN so that each unit shift restores the floor thermal energy,
                                          # modeling phonon fluidity injection from gradient work.
        
        # 3. Cryogenic Freezing
        CRYO_DIVISOR    = 0.00360230547        # Divider applied to temp upon freezing.
                                          # Ratio of formation energy to per‐step conduction influx,
                                          # i.e. ~E_form/E_conduction (4.75e−19 J / 1.71e−20 J).
        
        # 4. Per‐step physical time constant
        TAU_DIFFUSE     = 1.31e-13      # Characteristic thermal‐diffusion time (seconds)
                                          # for adjacent ice‐lattice molecules (2.75 Å spacing),
                                          # from τ = L²/α with α_ice ≈ 1.1×10⁻⁶ m²/s.
    
        @torch.no_grad()
        def step(self):
            loss = closure() if closure is not None else None
            
            for param in self.model.parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                s = self.state[param]
                state_p = s["p"]
                g_new, new_state = direwolf_update(param.data, param.grad.data, state_p, lr)
                state_p.copy_(new_state)
                
                dx = grad.clone()
                current_sign = torch.sign(grad)
                last_sign = s["last_grad_sign"]
    
                same_sign = (current_sign == last_sign)
                flip_sign = ~same_sign
    
                # Accumulate directional resistance
                s["res_pos"][same_sign & (current_sign > 0)] += self.DELTA_RESIST
                s["res_neg"][same_sign & (current_sign < 0)] += self.DELTA_RESIST
    
                # Resistance to apply when gradient flips
                resistance = torch.where(current_sign > 0, s["res_pos"], s["res_neg"])
                g_new[flip_sign] = g_new[flip_sign] / (1 + resistance[flip_sign])
    
                # Dissipate old resistance, build new on flip
                s["res_pos"][flip_sign & (current_sign < 0)] *= (1 - self.GAMMA_DECAY)
                s["res_neg"][flip_sign & (current_sign > 0)] *= (1 - self.GAMMA_DECAY)
                s["res_pos"][flip_sign & (current_sign > 0)] += self.DELTA_RESIST
                s["res_neg"][flip_sign & (current_sign < 0)] += self.DELTA_RESIST
    
                # Elasticity increases with param motion
                s["elasticity"] = s["elasticity"] * (1 - self.GAMMA_DECAY) + self.ELASTICITY_K * Δx.abs()
    
                # Temperature loss from dissipation
                s["temp"] *= (1 - self.GAMMA_DECAY)
    
                # Temperature gain from motion scaled by total resistance
                total_resist = s["res_pos"] + s["res_neg"]
                s["temp"] += dx.abs() * total_resist
    
                # Check crystallization condition
                below_Tmin = s["temp"] < self.T_MIN
                below_Tcryst = s["temp"] < self.T_CRYSTALLIZE
    
                frozen_mask = (below_Tmin.float().mean() > self.PERC_CRYSTAL) and below_Tcryst.all()
                if frozen_mask:
                    s["frozen"][:] = True
                    # Apply cryo division only to those below T_CRYSTALLIZE
                    s["temp"][below_Tcryst] *= self.CRYO_DIVISOR
    
                # Store current gradient sign
                s["last_grad_sign"] = current_sign
    
                # Temperature scale (saturates at T_CRYSTALLIZE)
                temp_scale = s["temp"] / (self.T_CRYSTALLIZE + 1e-8)
                temp_scale = temp_scale.clamp(0.0, 1.0)
                
                # Elasticity scale
                elas_scale = 1.0 + s["elasticity"]
                
                # Frozen mask: drop motion
                frozen_mask = s["frozen"]
                cryo_scale = torch.where(frozen_mask, 1.0 * self.CRYO_DIVISOR, 1.0)
                
                # Final scaled update
                scaled_update = g_new * temp_scale * elas_scale * cryo_scale
                param.data.add_(scaled_update)
    
                # Write adjusted gradients back to param
                param.data.copy_(param.data-grad)

        def get_temperature_map(self):
            return {k: self.state[k]["temp"].clone().detach() for k in self.state}
    
        def get_frozen_mask(self):
            return {k: self.state[k]["frozen"].clone() for k in self.state}
