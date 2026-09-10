# utils/rewards.py
import numpy as np
import time

class RewardFunction:
    """
    Smart Combat Reward Function.
    Fixed faults from v1:
      - Idle penalty no longer gated behind threat_score (was never firing at ep<50)
      - Cover reward no longer gated behind threat_score (was never firing at ep<50)
      - Survival flat +0.2/step replaced with tiny +0.05 + movement bonus (was incentivising camping)
      - Per-step movement reward added (rewards NOT standing still)
      - Stronger death (-15) and damage (-4) + extra -2 if hit while stationary in open
    """
    def __init__(self):
        self.last_move_time  = time.time()
        self.last_position   = None
        self.stationary_time = 0.0
        self.prev_step_pos   = None   # for per-step movement delta

    def _update_stationary(self, x, y):
        now = time.time()
        if self.last_position is None:
            self.last_position = (x, y)
            self.last_move_time = now
            self.stationary_time = 0.0
            return False
        dist = np.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
        if dist > 10.0:
            self.last_position = (x, y)
            self.last_move_time = now
            self.stationary_time = 0.0
        else:
            self.stationary_time = now - self.last_move_time
        return self.stationary_time > 3.0

    def compute(self, game_state, action, prev_game_state=None,
                ai_perception=None, threat_score=0.0, player_dist=0.0):
        reward = 0.0
        info   = {}

        x, y   = game_state['npc_x'], game_state['npc_y']
        alive  = game_state['alive']

        # ── 1. Death Penalty ─────────────────────────────────────────
        if not alive:
            reward -= 15.0
            info['event'] = 'died'
            return reward, info

        # ── 2. Tiny survival baseline (no longer +0.2 camping magnet) ─
        reward += 0.05

        # ── 3. Per-step movement reward (caps at +0.5) ────────────────
        if self.prev_step_pos is not None:
            step_dist = np.sqrt((x - self.prev_step_pos[0])**2 + (y - self.prev_step_pos[1])**2)
            reward += min(step_dist / 200.0, 0.5)
            info['step_dist'] = round(step_dist, 1)
        self.prev_step_pos = (x, y)

        # ── 4. Damage Avoidance ───────────────────────────────────────
        got_shot = game_state.get('got_shot', False)
        if got_shot:
            reward -= 4.0
            info['got_shot'] = True

        # ── 5. Stationary check (always — no threat_score gate) ───────
        is_stationary = self._update_stationary(x, y)

        # ── 6. Cover detection (always — no threat_score gate) ────────
        actors_seen  = 0.0
        nearest_dist = 9999.0
        if ai_perception is not None:
            actors_seen  = float(ai_perception[0]) * 5.0
            nearest_dist = float(ai_perception[1]) * 5000.0

        # NPC is in cover = player cannot see it
        in_cover = (actors_seen == 0.0)
        info['in_cover'] = in_cover

        if in_cover:
            # Scale cover reward by how close player is (more reward = more pressure)
            if player_dist < 2000.0:
                reward += 1.5   # active cover under pressure
                info['cover_tier'] = 'hot'
            else:
                reward += 0.5   # general good positioning
                info['cover_tier'] = 'safe'
        else:
            # Exposed — penalise if player is nearby
            if player_dist < 400.0:
                reward -= 1.5
                info['danger_close'] = True

        # ── 7. Idle penalty (always active, no threat_score gate) ────
        if is_stationary:
            if in_cover:
                # Holding cover still is fine — small reward
                reward += 0.4
                info['holding_cover'] = True
            else:
                # Standing idle in the open — punish hard
                reward -= 1.5
                info['stationary_in_open'] = True
                # Double-punish if also taking damage while frozen
                if got_shot:
                    reward -= 2.0
                    info['hit_while_idle'] = True

        info['total_reward'] = round(reward, 4)
        return reward, info

    def reset(self):
        self.last_move_time  = time.time()
        self.last_position   = None
        self.stationary_time = 0.0
        self.prev_step_pos   = None


# ─── VALIDATION ──────────────────────────────────────────────────────
if __name__ == "__main__":
    reward_fn = RewardFunction()
    fake_state = {
        'npc_x': 61690.0, 'npc_y': -24820.0, 'npc_z': 88.0,
        'player_x': 60000.0, 'player_y': -24000.0, 'player_z': 88.0,
        'health': 100.0, 'max_health': 100.0,
        'is_dead': False, 'alive': True, 'got_shot': False
    }
    reward, info = reward_fn.compute(fake_state, action=0, player_dist=2100.0)
    print(f"Reward: {reward:.2f}")
    print(f"Info: {info}")