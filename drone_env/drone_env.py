import gymnasium as gym
import mujoco
import numpy as np

class DroneEnv(gym.Env):
    """Custom Gymnasium environment for quadrotor control failure simulation."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Action: 4 motor thrusts
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Observation: position + velocity + orientation + angular rates
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("drone_env/quadrotor.xml")
        self.data = mujoco.MjData(self.model)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self):
        # Placeholder reward
        return 0.0

    def _check_done(self):
        # Placeholder termination condition
        return False

