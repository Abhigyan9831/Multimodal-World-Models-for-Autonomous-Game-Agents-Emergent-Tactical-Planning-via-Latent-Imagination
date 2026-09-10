# utils/opponent_model.py
import numpy as np


class PlayerHistoryTracker:
    """
    Opponent Modelling — tracks player's recent movement and shooting behaviour.

    Adds 4 values to the observation vector so the World Model can PREDICT
    what the player will do next, rather than just react:
        player_vx      — player X velocity (normalised)
        player_vy      — player Y velocity (normalised)
        player_moving_toward_npc — 1.0 if closing in, 0.0 if retreating
        player_shot_recently     — 1.0 if player fired in last step
    """

    VEL_SCALE = 500.0   # normalise velocity to [-1, 1] range

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_px = None
        self._prev_py = None
        self._shot_flag = 0.0

    def update(self, game_state: dict) -> np.ndarray:
        """
        Call once per step with the latest game_state.
        Returns a (4,) float32 array ready to append to the obs vector.
        """
        px, py = game_state['player_x'], game_state['player_y']
        nx, ny = game_state['npc_x'],    game_state['npc_y']

        if self._prev_px is None:
            self._prev_px, self._prev_py = px, py
            return np.zeros(4, dtype=np.float32)

        # --- velocity ---
        vx = (px - self._prev_px) / self.VEL_SCALE
        vy = (py - self._prev_py) / self.VEL_SCALE
        vx = float(np.clip(vx, -1.0, 1.0))
        vy = float(np.clip(vy, -1.0, 1.0))

        # --- moving toward NPC? ---
        prev_dist = np.sqrt((self._prev_px - nx)**2 + (self._prev_py - ny)**2)
        curr_dist = np.sqrt((px - nx)**2 + (py - ny)**2)
        moving_toward = 1.0 if curr_dist < prev_dist else 0.0

        # --- shot recently (set externally via mark_shot) ---
        shot = self._shot_flag
        self._shot_flag = 0.0   # reset after one step

        self._prev_px, self._prev_py = px, py
        return np.array([vx, vy, moving_toward, shot], dtype=np.float32)

    def mark_shot(self):
        """Call this when the NPC receives damage so the history captures it."""
        self._shot_flag = 1.0
