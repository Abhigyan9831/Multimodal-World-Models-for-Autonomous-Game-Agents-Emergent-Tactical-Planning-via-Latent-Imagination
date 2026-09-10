# utils/rnd.py
"""
Random Network Distillation (RND) — Curiosity Bonus.

How it works:
  - TARGET network: small MLP with FIXED random weights (never trained).
  - PREDICTOR network: same architecture but TRAINED to match target output.
  - Curiosity reward = MSE(predictor(obs), target(obs))
    → High error = state is NEW / surprising → NPC explores it more.
    → Low error  = state seen many times    → NPC already knows it.

Stops the NPC camping in one spot. IEG/DeepMind use this in combat agents.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class RNDModule(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int = 64, lr: float = 1e-3,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device

        def _mlp():
            return nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, embed_dim)
            )

        # Target is frozen random — never updated
        self.target    = _mlp().to(device)
        self.predictor = _mlp().to(device)

        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_dim   = obs_dim

    @torch.no_grad()
    def curiosity_reward(self, obs_vec: torch.Tensor) -> float:
        """
        obs_vec: (1, obs_dim) float32 tensor.
        Returns a scalar curiosity bonus (higher = more novel state).
        """
        target_out    = self.target(obs_vec)
        predictor_out = self.predictor(obs_vec)
        return float(((predictor_out - target_out) ** 2).mean().item())

    def update(self, obs_vec: torch.Tensor) -> float:
        """Train predictor to match target. Returns the MSE loss."""
        target_out    = self.target(obs_vec).detach()
        predictor_out = self.predictor(obs_vec)
        loss = ((predictor_out - target_out) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
