import numpy as np
from .base import BaseController
from .pid import PID

class QuadrotorRLController(BaseController):
    def __init__(self, policy, dt: float):
        self.policy = policy
        self.dt = dt
        self._prev_vel = None
    
    def reset(self):
        self._prev_vel = None

    def compute_control(self, state, target):
        pos = state["pos"]
        vel = state["vel"]
        ang_disp = state["euler"]
        ang_vel = state["ang_vel"]

        if self._prev_vel is None:
            accel = np.zeros(3, dtype=np.float64)
        else:
            accel = (vel - self._prev_vel) / self.dt
        self._prev_vel = vel.copy()

        obs = np.concatenate([pos, vel, accel, ang_disp, ang_vel])
        action = self.policy(obs)
        return action