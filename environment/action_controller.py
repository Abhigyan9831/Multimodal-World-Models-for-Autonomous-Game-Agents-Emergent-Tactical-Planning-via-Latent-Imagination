# environment/action_controller.py
import numpy as np

class ActionController:
    def __init__(self, step_size=200.0):
        self.step_size = step_size
        # 0=F, 1=B, 2=L, 3=R, 4=Stay, 5=Jump
        self.actions = {
            0: (self.step_size, 0),
            1: (-self.step_size, 0),
            2: (0, -self.step_size),
            3: (0, self.step_size),
            4: (0, 0),
            5: (self.step_size, 0) 
        }

    def get_target(self, current_x, current_y, action_int):
        dx, dy = self.actions[action_int]
        return current_x + dx, current_y + dy