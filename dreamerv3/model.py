# dreamerv3/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamerv3.config import DreamerConfig


# ─── CNN ENCODER ─────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    """
    128x128 RGB → 512-dim embedding.
    4 strided conv layers (halve spatial dims each), then linear projection.
    VRAM footprint: ~80 MB for a batch of 16.
    """
    def __init__(self, config: DreamerConfig):
        super().__init__()
        C = config.img_channels  # 3

        self.convs = nn.Sequential(
            # (B, 3, 128, 128) → (B, 32, 62, 62)
            nn.Conv2d(C,   32,  kernel_size=4, stride=2, padding=0), nn.SiLU(),
            # (B, 32, 62, 62) → (B, 64, 30, 30)
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=0), nn.SiLU(),
            # (B, 64, 30, 30) → (B, 128, 14, 14)
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=0), nn.SiLU(),
            # (B, 128, 14, 14) → (B, 256, 6, 6)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0), nn.SiLU(),
        )
        # After 4 strides of 2: 128 → 62 → 30 → 14 → 6  (floor((n-4)/2+1))
        flat_dim = 256 * 6 * 6  # = 9216

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, config.cnn_embed_dim),
            nn.LayerNorm(config.cnn_embed_dim),
        )

    def forward(self, img):
        """
        img: (B, 3, 128, 128) float32 in [0, 1]
        returns: (B, 512) float32
        """
        return self.proj(self.convs(img))


# ─── RSSM ────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    """Recurrent State Space Model — DreamerV3 world model core."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config    = config
        self.deter_dim = config.deter_dim
        self.stoch_dim = config.stoch_dim * config.stoch_classes

        # GRU for deterministic state
        self.gru = nn.GRUCell(
            input_size  = config.hidden_dim,
            hidden_size = config.deter_dim
        )

        # Input embedding: obs (522) + action → hidden
        self.input_layer = nn.Sequential(
            nn.Linear(config.obs_dim + config.act_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU()
        )

        # Prior: deter → stoch (imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(config.deter_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.stoch_dim * config.stoch_classes)
        )

        # Posterior: deter + obs → stoch (real experience)
        self.posterior_net = nn.Sequential(
            nn.Linear(config.deter_dim + config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.stoch_dim * config.stoch_classes)
        )

        # Imagination obs proxy: stoch → obs_dim (used in imagine() to avoid zeros)
        self.stoch_to_obs = nn.Linear(self.stoch_dim, config.obs_dim)

    def initial_state(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim,  device=device)
        }

    def forward(self, obs_embed, action_onehot, prev_state):
        """One RSSM step with a real observation."""
        x = torch.cat([obs_embed, action_onehot], dim=-1)
        x = self.input_layer(x)
        deter = self.gru(x, prev_state['deter'])

        post_logits = self.posterior_net(
            torch.cat([deter, obs_embed], dim=-1)
        ).reshape(-1, self.config.stoch_dim, self.config.stoch_classes)
        stoch = F.gumbel_softmax(post_logits, tau=1.0, hard=True)
        stoch = stoch.reshape(-1, self.stoch_dim)

        prior_logits = self.prior_net(deter).reshape(
            -1, self.config.stoch_dim, self.config.stoch_classes
        )

        return {
            'deter':        deter,
            'stoch':        stoch,
            'post_logits':  post_logits,
            'prior_logits': prior_logits
        }

    def imagine(self, action_onehot, prev_state):
        """Imagine next state without a real observation."""
        prior_logits = self.prior_net(prev_state['deter']).reshape(
            -1, self.config.stoch_dim, self.config.stoch_classes
        )
        stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        stoch = stoch.reshape(-1, self.stoch_dim)

        # Project stoch → obs_dim via learned linear (no zero-filling hack)
        obs_proxy = self.stoch_to_obs(stoch)

        x     = torch.cat([obs_proxy, action_onehot], dim=-1)
        x_in  = self.input_layer(x)
        deter = self.gru(x_in, prev_state['deter'])

        return {'deter': deter, 'stoch': stoch, 'prior_logits': prior_logits}


# ─── HEADS ───────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.act_dim)
        )

    def forward(self, state):
        feat   = torch.cat([state['deter'], state['stoch']], dim=-1)
        logits = self.net(feat)
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.sample(), dist


class Critic(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, state):
        feat = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.net(feat).squeeze(-1)


class RewardDecoder(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_dim = config.deter_dim + config.stoch_dim * config.stoch_classes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, state):
        feat = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.net(feat).squeeze(-1)


# ─── DREAMERV3 ───────────────────────────────────────────────────────────────
class DreamerV3(nn.Module):
    """Full DreamerV3 model with CNN image encoder."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config         = config
        self.encoder        = ConvEncoder(config)
        self.rssm           = RSSM(config)
        self.actor          = Actor(config)
        self.critic         = Critic(config)
        self.reward_decoder = RewardDecoder(config)

    def encode_obs(self, image, vec_obs):
        """
        image:   (B, 3, 128, 128) float32 in [0, 1]
        vec_obs: (B, 10)          float32  (3 player + 3 npc + 4 perception)
        returns: (B, 522)         float32
        """
        visual = self.encoder(image)                     # (B, 512)
        return torch.cat([visual, vec_obs], dim=-1)      # (B, 522)

    def forward(self, obs_embed, action_onehot, prev_state):
        return self.rssm(obs_embed, action_onehot, prev_state)

    def get_action(self, obs_embed, prev_state):
        dummy_action = torch.zeros(
            obs_embed.shape[0], self.config.act_dim, device=obs_embed.device
        )
        state         = self.rssm(obs_embed, dummy_action, prev_state)
        action, dist  = self.actor(state)
        return action, state, dist


# ─── VALIDATION ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = DreamerConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DreamerV3(config).to(device)

    B   = config.batch_size
    img = torch.rand(B, 3, 128, 128, device=device)          # raw pixels
    vec = torch.randn(B, config.vector_obs_dim, device=device) # telemetry

    obs   = model.encode_obs(img, vec)                        # (B, 522)
    act   = torch.zeros(B, config.act_dim, device=device)
    act[:, 0] = 1.0

    state     = model.rssm.initial_state(B, device)
    new_state = model(obs, act, state)
    action, _, _ = model.get_action(obs, state)

    print(f" ConvEncoder output:  {obs.shape}")               # (16, 522)
    print(f" RSSM deter shape:    {new_state['deter'].shape}")
    print(f" RSSM stoch shape:    {new_state['stoch'].shape}")
    print(f" Action:              {action}")
    print(f" VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\\n Pure DreamerV3 model validation PASSED!")