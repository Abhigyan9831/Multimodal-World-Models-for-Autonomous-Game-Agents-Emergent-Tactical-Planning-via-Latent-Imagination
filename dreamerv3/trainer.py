# dreamerv3/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dreamerv3.config import DreamerConfig
from dreamerv3.model import DreamerV3


class DreamerTrainer:
    def __init__(self, config: DreamerConfig, device='cuda'):
        self.config = config
        self.device = device

        self.model = DreamerV3(config).to(device)

        # ── World optimizer: encoder + RSSM + reward decoder (joint) ──
        self.world_optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.rssm.parameters()) +
            list(self.model.reward_decoder.parameters()),
            lr=config.lr, eps=1e-8
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=config.lr, eps=1e-8
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=config.lr, eps=1e-8
        )

        self.scaler = torch.cuda.amp.GradScaler()
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f" DreamerTrainer initialized on {device}")

    def action_to_onehot(self, action):
        if isinstance(action, int):
            oh = torch.zeros(1, self.config.act_dim, device=self.device)
            oh[0, action] = 1.0
        else:
            B  = action.shape[0]
            oh = torch.zeros(B, self.config.act_dim, device=self.device)
            oh.scatter_(1, action.unsqueeze(1), 1.0)
        return oh

    def encode_batch(self, img_b, vec_b):
        """
        img_b: (B, T, 3, 128, 128) float32 [0,1]
        vec_b: (B, T, 10)          float32
        returns obs_b: (B, T, 522)
        """
        B, T, C, H, W = img_b.shape
        imgs_flat = img_b.reshape(B * T, C, H, W)
        vecs_flat = vec_b.reshape(B * T, self.config.vector_obs_dim)
        obs_flat  = self.model.encode_obs(imgs_flat, vecs_flat)   # (B*T, 522)
        return obs_flat.reshape(B, T, self.config.obs_dim)

    def train_world_model(self, img_b, vec_b, actions, rewards, dones):
        """Train ConvEncoder + RSSM + RewardDecoder jointly."""
        img_b   = img_b.to(self.device)
        vec_b   = vec_b.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        with torch.cuda.amp.autocast():
            obs = self.encode_batch(img_b, vec_b)   # (B, T, 522)
            B, T, _ = obs.shape

            state      = self.model.rssm.initial_state(B, self.device)
            kl_loss    = 0.0
            reward_loss = 0.0

            for t in range(T):
                obs_t = obs[:, t]
                act_t = self.action_to_onehot(actions[:, t])
                rew_t = rewards[:, t]

                new_state = self.model.rssm(obs_t, act_t, state)

                post  = new_state['post_logits'].float()
                prior = new_state['prior_logits'].float()
                kl_loss += F.kl_div(
                    F.log_softmax(prior, dim=-1),
                    F.softmax(post,  dim=-1),
                    reduction='batchmean'
                )

                pred_reward = self.model.reward_decoder(new_state)
                reward_loss += F.mse_loss(pred_reward.float(), rew_t.float())

                state = {k: new_state[k].detach() for k in ['deter', 'stoch']}

            total_loss = reward_loss + 0.1 * kl_loss

        self.world_optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.world_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters()) +
            list(self.model.rssm.parameters()) +
            list(self.model.reward_decoder.parameters()),
            self.config.grad_clip
        )
        self.scaler.step(self.world_optimizer)
        self.scaler.update()

        return {
            'world_model_loss': total_loss.item(),
            'reward_loss':      reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss,
            'kl_divergence':    kl_loss.item()     if isinstance(kl_loss,    torch.Tensor) else kl_loss
        }

    def train_actor_critic(self, img_b, vec_b, actions, rewards):
        """Train Actor + Critic using imagined rollouts from the first step of each batch."""
        img_b = img_b[:, 0].to(self.device)   # (B, 3, 128, 128) — first timestep
        vec_b = vec_b[:, 0].to(self.device)   # (B, 10)
        B     = img_b.shape[0]

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                obs_0 = self.model.encode_obs(img_b, vec_b)   # (B, 522)

            state            = self.model.rssm.initial_state(B, self.device)
            imagined_rewards = []
            imagined_values  = []
            log_probs        = []

            # Seed state with a real observation
            dummy_act = torch.zeros(B, self.config.act_dim, device=self.device)
            state = self.model.rssm(obs_0, dummy_act, state)

            for _ in range(self.config.imagination_horizon):
                action, dist = self.model.actor(state)
                act_onehot   = self.action_to_onehot(action)

                next_state = self.model.rssm.imagine(act_onehot, state)
                value      = self.model.critic(state)
                reward     = self.model.reward_decoder(state)

                imagined_rewards.append(reward)
                imagined_values.append(value)
                log_probs.append(dist.log_prob(action))

                state = {k: next_state[k].detach() for k in ['deter', 'stoch']}

            rewards_t  = torch.stack(imagined_rewards, dim=1)
            values_t   = torch.stack(imagined_values,  dim=1)
            logprobs_t = torch.stack(log_probs,        dim=1)

            gamma   = 0.99
            returns = torch.zeros_like(rewards_t)
            ret     = values_t[:, -1]
            for t in reversed(range(self.config.imagination_horizon)):
                ret          = rewards_t[:, t] + gamma * ret
                returns[:, t] = ret

            advantages  = (returns - values_t).detach()
            actor_loss  = -(logprobs_t * advantages).mean()
            critic_loss = F.mse_loss(values_t.float(), returns.float())

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.grad_clip)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.config.grad_clip)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        return {
            'actor_loss':  actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def save_checkpoint(self, episode, metrics=None):
        for fname in [f'checkpoint_ep{episode}.pt', 'checkpoint_latest.pt']:
            path = os.path.join(self.config.checkpoint_dir, fname)
            torch.save({
                'episode':         episode,
                'model_state':     self.model.state_dict(),
                'world_optimizer': self.world_optimizer.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer':self.critic_optimizer.state_dict(),
                'scaler':          self.scaler.state_dict(),
                'metrics':         metrics or {}
            }, path)
        print(f" Checkpoint saved: episode {episode}")

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, 'checkpoint_latest.pt')
        if not os.path.exists(path):
            print("No checkpoint found — starting fresh.")
            return 0
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.world_optimizer.load_state_dict(ckpt['world_optimizer'])
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])
            episode = ckpt['episode']
            print(f" Resumed from episode {episode}")
            return episode
        except RuntimeError as e:
            print(f" Checkpoint architecture incompatible with new Pure DreamerV3 CNN model:\n  {e}")
            print(" Starting fresh training for Pure DreamerV3 (new checkpoints will be saved automatically).")
            return 0



# ─── VALIDATION ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config  = DreamerConfig()
    trainer = DreamerTrainer(config)

    B, T = config.batch_size, config.seq_len
    img_b   = torch.rand(B, T, 3, 128, 128)
    vec_b   = torch.randn(B, T, config.vector_obs_dim)
    actions = torch.randint(0, config.act_dim, (B, T))
    rewards = torch.randn(B, T)
    dones   = torch.zeros(B, T, dtype=torch.bool)

    wm = trainer.train_world_model(img_b, vec_b, actions, rewards, dones)
    ac = trainer.train_actor_critic(img_b, vec_b, actions, rewards)

    print(f"World loss: {wm['world_model_loss']:.4f}  KL: {wm['kl_divergence']:.4f}")
    print(f"Actor loss: {ac['actor_loss']:.4f}  Critic: {ac['critic_loss']:.4f}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\n Trainer validation PASSED!")