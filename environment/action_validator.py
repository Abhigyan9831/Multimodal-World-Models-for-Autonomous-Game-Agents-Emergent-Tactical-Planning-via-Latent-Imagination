# environment/action_validator.py
import numpy as np

class ActionValidator:
    def __init__(self, bounds_x=(-510.0, 3190.0), bounds_y=(-1000.0, 3400.0)):
        self.min_x, self.max_x = bounds_x
        self.min_y, self.max_y = bounds_y

    def is_valid(self, target_x, target_y):
        # Check map bounds
        if not (self.min_x <= target_x <= self.max_x and self.min_y <= target_y <= self.max_y):
            return False
        return True

    def clip_to_bounds(self, target_x, target_y):
        target_x = np.clip(target_x, self.min_x, self.max_x)
        target_y = np.clip(target_y, self.min_y, self.max_y)
        return target_x, target_y