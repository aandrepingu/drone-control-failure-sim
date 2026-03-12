import numpy as np
from .base import BaseController
from .pid import PID

class QuadrotorRLController(BaseController):
    def __init__(self, policy):
        self.policy = policy
    
    def reset(self):
        pass

    def compute_control(self, state, target):
        obs = np.concatenate([
            state["pos"],
            state["euler"],
            [target["z"]],
            [target["roll"]],
            [target["pitch"]],
            [target["yaw"]]
        ])

        action = self.policy(obs)

        return action