import copy
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from sim.model import load_model
from sim.sim_config import SimConfig
from sim.sim_loop import Simulator


class DroneEnv(gym.Env):
    """Gymnasium environment for quadrotor control and failure simulation."""
    metadata = {"render_modes": ["human"], "render_fps": 500}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt: float = 0.002,
        max_steps: int = 2000,
        pos_range=(-0.5, 0.5),
        vel_range=(-0.2, 0.2),
        tilt_range=(-0.3, 0.3),
        yaw_range=(-0.3, 0.3),
        target=None,
        failures=None,
        failure_factory=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.max_steps = max_steps
        self.pos_range = np.array(pos_range, dtype=np.float64)
        self.vel_range = np.array(vel_range, dtype=np.float64)
        self.tilt_range = np.array(tilt_range, dtype=np.float64)
        self.yaw_range = np.array(yaw_range, dtype=np.float64)

        self.target = target or {"z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.failures_template = failures
        self.failure_factory = failure_factory

        model, data = load_model()
        self.sim = Simulator(model, data, SimConfig.random(
            self.pos_range, self.vel_range, self.tilt_range, self.yaw_range
        ), controller=None, failures=[])
        self.ctrl_low = self.sim.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.sim.model.actuator_ctrlrange[:, 1].copy()

        # Action: normalized motor thrusts in [0, 1]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        # Observation: pos(3) + vel(3) + euler(3) + ang_vel(3) + target(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        self.viewer = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = SimConfig.random(self.pos_range, self.vel_range, self.tilt_range, self.yaw_range)
        self.sim.config = cfg
        self.sim.time = 0.0
        self.sim.apply_config(cfg)
        self.sim.target = dict(self.target)
        self.steps = 0

        if self.failure_factory is not None:
            self.sim.failures = self.failure_factory()
        elif self.failures_template is not None:
            self.sim.failures = copy.deepcopy(self.failures_template)
        else:
            self.sim.failures = []

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, 0.0, 1.0)
        motor_cmds = self.ctrl_low + action * (self.ctrl_high - self.ctrl_low)
        self.sim.data.ctrl[:] = motor_cmds

        for f in self.sim.failures:
            f.apply(self.sim, self.sim.time)

        mujoco.mj_step(self.sim.model, self.sim.data)
        self.sim.time += self.dt
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        self.viewer.sync()

    def _get_obs(self):
        state = self.sim.get_state()
        target = np.array(
            [self.target["z"], self.target["roll"], self.target["pitch"], self.target["yaw"]],
            dtype=np.float64,
        )
        return np.concatenate(
            [state["pos"], state["vel"], state["euler"], state["ang_vel"], target]
        ).astype(np.float32)

    def _compute_reward(self, action):
        state = self.sim.get_state()
        pos_target = np.array([0.0, 0.0, self.target["z"]], dtype=np.float64)
        att_target = np.array(
            [self.target["roll"], self.target["pitch"], self.target["yaw"]],
            dtype=np.float64,
        )
        pos_err = state["pos"] - pos_target
        att_err = _wrap_angles(state["euler"] - att_target)
        ang_vel = state["ang_vel"]

        reward = 1.0
        reward -= 1.5 * float(np.dot(pos_err, pos_err))
        reward -= 0.5 * float(np.dot(att_err, att_err))
        reward -= 0.05 * float(np.dot(ang_vel, ang_vel))
        reward -= 0.001 * float(np.dot(action, action))
        return reward

    def _check_terminated(self):
        state = self.sim.get_state()
        z = state["pos"][2]
        roll, pitch, _ = state["euler"]
        if z < 0.0:
            return True
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            return True
        if np.linalg.norm(state["pos"]) > 5.0:
            return True
        return False


def _wrap_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

