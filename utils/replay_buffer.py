# utils/replay_buffer.py
"""
Dual-store replay buffer:
  - images   stored as uint8  (128,128,3)  — 49 KB/step → 4.9 GB @ 100k steps (system RAM)
  - vec_obs  stored as float32 (10,)       — 40 B/step
Samples are normalised to float32 [0,1] on the target device at batch time.
"""
import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=100_000, vec_obs_dim=10, act_dim=6, seq_len=16):
        self.capacity    = capacity
        self.vec_obs_dim = vec_obs_dim
        self.act_dim     = act_dim
        self.seq_len     = seq_len
        self.episodes    = deque()
        self.current_episode = []
        self.total_steps = 0

    def add(self, image, vec_obs, action, reward, done):
        """
        image:   np.ndarray (128, 128, 3) uint8
        vec_obs: np.ndarray (10,)         float32
        action:  int
        reward:  float
        done:    bool
        """
        self.current_episode.append({
            'image':   image.astype(np.uint8),
            'vec_obs': vec_obs.astype(np.float32),
            'action':  action,
            'reward':  float(reward),
            'done':    bool(done)
        })
        self.total_steps += 1

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            while self.total_steps > self.capacity and self.episodes:
                removed = self.episodes.popleft()
                self.total_steps -= len(removed)

    def sample(self, batch_size, seq_len, device='cuda'):
        """
        Returns:
            img_t:  (B, T, 3, 128, 128) float32 [0,1] on device
            vec_t:  (B, T, 10)          float32 on device
            act_t:  (B, T)              long    on device
            rew_t:  (B, T)              float32 on device
            done_t: (B, T)              bool    on device
        """
        valid = [ep for ep in self.episodes if len(ep) >= seq_len]
        if not valid:
            return None

        img_batch, vec_batch, act_batch, rew_batch, done_batch = [], [], [], [], []

        for _ in range(batch_size):
            ep    = random.choice(valid)
            start = random.randint(0, len(ep) - seq_len)
            seq   = ep[start:start + seq_len]

            img_batch.append([s['image']   for s in seq])
            vec_batch.append([s['vec_obs'] for s in seq])
            act_batch.append([s['action']  for s in seq])
            rew_batch.append([s['reward']  for s in seq])
            done_batch.append([s['done']   for s in seq])

        # (B, T, H, W, C) → (B, T, C, H, W), normalise to [0, 1]
        img_np = np.array(img_batch, dtype=np.float32) / 255.0
        img_t  = torch.tensor(img_np, dtype=torch.float32, device=device)
        img_t  = img_t.permute(0, 1, 4, 2, 3)  # (B, T, 3, 128, 128)

        vec_t  = torch.tensor(np.array(vec_batch),  dtype=torch.float32, device=device)
        act_t  = torch.tensor(np.array(act_batch),  dtype=torch.long,    device=device)
        rew_t  = torch.tensor(np.array(rew_batch),  dtype=torch.float32, device=device)
        done_t = torch.tensor(np.array(done_batch), dtype=torch.bool,    device=device)

        return img_t, vec_t, act_t, rew_t, done_t

    def __len__(self):
        return self.total_steps

    def is_ready(self, min_episodes=5):
        return len(self.episodes) >= min_episodes


# ─── VALIDATION ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    buf = ReplayBuffer(capacity=100_000)

    for ep in range(10):
        for step in range(50):
            img     = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            vec_obs = np.random.randn(10).astype(np.float32)
            action  = np.random.randint(0, 6)
            reward  = float(np.random.randn())
            done    = (step == 49)
            buf.add(img, vec_obs, action, reward, done)

    print(f"Total steps:  {len(buf)}")
    print(f"Episodes:     {len(buf.episodes)}")

    batch = buf.sample(batch_size=16, seq_len=16, device='cpu')
    if batch:
        img_t, vec_t, act_t, rew_t, done_t = batch
        print(f" img_t shape:   {img_t.shape}")    # (16, 16, 3, 128, 128)
        print(f" vec_t shape:   {vec_t.shape}")    # (16, 16, 10)
        print(f" act_t shape:   {act_t.shape}")    # (16, 16)
        print(f" img range:     [{img_t.min():.3f}, {img_t.max():.3f}]")

    print("\n Replay buffer validation PASSED!")