from dataclasses import dataclass

@dataclass
class DreamerConfig:
    
    img_channels: int = 3
    img_height:   int = 128
    img_width:    int = 128
    cnn_embed_dim: int = 512      

    
    vector_obs_dim: int = 15      # 3 player + 3 npc + 4 AI perception + 1 threat + 4 opponent history
    obs_dim:        int = 527     # 512 CNN + 15 vector

    
    act_dim: int = 6              # F, B, L, R, Stay, Jump

    # ── World Model ──────────────────────────────────────────────────
    deter_dim:      int = 256
    stoch_dim:      int = 32
    stoch_classes:  int = 32
    hidden_dim:     int = 256
    num_layers:     int = 2

    
    batch_size:           int   = 16
    seq_len:              int   = 16
    lr:                   float = 3e-4
    grad_clip:            float = 100.0
    imagination_horizon:  int   = 15

    
    replay_capacity: int = 100_000

    
    log_every:  int = 10
    save_every: int = 100
    checkpoint_dir: str = r"E:\NPC_Brain\checkpoints"