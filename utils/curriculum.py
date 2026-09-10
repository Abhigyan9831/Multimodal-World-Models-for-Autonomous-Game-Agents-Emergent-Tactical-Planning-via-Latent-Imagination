# utils/curriculum.py
from collections import deque


class CurriculumManager:
    """
    Auto-scales training difficulty by adjusting STEP_DELAY based on win rate.

    Win  = NPC survived the full episode (not killed).
    Loss = NPC was killed.

    Rules (Tencent IEG style):
        win_rate > WIN_THRESH  → decrease STEP_DELAY (NPC reacts faster → harder)
        win_rate < LOSS_THRESH → increase STEP_DELAY (NPC gets more thinking time → easier)

    STEP_DELAY is clamped between MIN_DELAY and MAX_DELAY.
    """

    WIN_THRESH  = 0.70   # 70 % wins  → increase difficulty
    LOSS_THRESH = 0.20   # 20 % wins  → decrease difficulty
    WINDOW      = 20     # rolling window size (episodes)
    STEP        = 0.02   # how much to adjust STEP_DELAY each change
    MIN_DELAY   = 0.10   # fastest reaction (100 ms)
    MAX_DELAY   = 0.50   # slowest reaction (500 ms)

    def __init__(self, initial_delay: float = 0.3):
        self.step_delay = initial_delay
        self._history   = deque(maxlen=self.WINDOW)

    def record(self, survived: bool):
        """Call at end of every episode. survived=True means NPC was NOT killed."""
        self._history.append(1 if survived else 0)

    def update(self) -> float:
        """
        Adjusts step_delay based on recent win rate.
        Returns the (possibly updated) step_delay.
        """
        if len(self._history) < self.WINDOW:
            return self.step_delay   # not enough data yet

        win_rate = sum(self._history) / len(self._history)

        if win_rate > self.WIN_THRESH:
            self.step_delay = max(self.MIN_DELAY,
                                  self.step_delay - self.STEP)
            print(f"  [CURRICULUM] Win rate {win_rate:.0%} → harder | STEP_DELAY={self.step_delay:.2f}s")
        elif win_rate < self.LOSS_THRESH:
            self.step_delay = min(self.MAX_DELAY,
                                  self.step_delay + self.STEP)
            print(f"  [CURRICULUM] Win rate {win_rate:.0%} → easier | STEP_DELAY={self.step_delay:.2f}s")

        return self.step_delay

    @property
    def win_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
