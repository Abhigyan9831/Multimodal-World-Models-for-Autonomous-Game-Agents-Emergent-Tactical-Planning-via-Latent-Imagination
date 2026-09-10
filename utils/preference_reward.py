# utils/preference_reward.py
import torch
import torch.nn as nn
import numpy as np

class PreferenceRewardModel(nn.Module):
    """A tiny neural network that learns to predict human preference."""
    def __init__(self, obs_dim=1038, device='cuda'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, obs):
        return self.net(obs)

    def train_preference(self, good_obs, bad_obs):
        """Good obs gets +1, Bad obs gets -1. Trains the model."""
        good_obs = torch.tensor(good_obs, dtype=torch.float32, device=self.device)
        bad_obs = torch.tensor(bad_obs, dtype=torch.float32, device=self.device)
        
        pred_good = self.forward(good_obs)
        pred_bad = self.forward(bad_obs)
        
        # Bradley-Terry loss: We want pred_good > pred_bad
        loss = -torch.log(torch.sigmoid(pred_good - pred_bad))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_reward(self, obs):
        """Query the preference model for a reward signal."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            return self.forward(obs_tensor).item()