from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from sim.controllers.quadrotor_pid import QuadrotorPIDController
from sim.core.sim_loop import SimLoop
from sim.core.state import *
from sim.modules.control import ControlModule
from sim.modules.fault import FaultModule
from sim.modules.mujoco_dynamics import MujocoDynamicsModule
from sim.modules.sensors import SensorModule
from sim.sim_config import SimConfig

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def load_model(xml_name="quadrotor.xml"):
    """
    Load model from xml file and extract the mujoco MjData object.

    :param xml_name: name of the xml model you want to load
    """
    xml_path = ASSETS / xml_name
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def launch_viewer(model, data):
    return mujoco.viewer.launch_passive(model, data)


class DroneEnv(gym.Env):
    """Custom Gymnasium environment for quadrotor control failure simulation."""

    # metadata = {"render_modes": ["human"]}
    def init_modules(
        self,
        state_board: StateBoard,
    ):
        """
        Initialize sim modules with shared state objects.

        Pipeline Data Flow
        1. MujocoDynamicsModule

        - Steps MuJoCo physics.
        - Writes Ground Truth (true position, velocity, orientation quaternion, acceleration, and angular velocity) to VehicleState.

        2. SensorModule

        - Reads Ground Truth from VehicleState.
        - Applies noise, biases, latency, and drift.
        - Writes Sensor Data to VehicleState.

        3. ControlModule

        - Reads Sensor Data (or an estimated state derived from sensors).
        - Calculates body torques / motor commands.
        - Writes raw command output to VehicleState.

        4. FaultModule

        - Intercepts commands, applies actuator loss or scaling failures.
        - Writes modified motor commands to VehicleState, which DynamicsModule applies on the next tick.

        5. TrajectoryCaptureModule

        - Manages and captures trajectory over a certain horizon

        6. TelemetryModule

        - Logs state information to parquet

        7. RewardModule

        - Computes reward for RL purposes
        """
        # dynamics module
        dynamics_module = MujocoDynamicsModule(
            model=self.mj_model,
            data=self.mj_data,
            dynamics_state=state_board.dynamics_state,
            actuator_state=state_board.actuator_state,
        )
        self.sim_loop.add_module(dynamics_module)

        # sensor module
        sensor_module = SensorModule(
            dynamics_state=state_board.dynamics_state,
            sensor_data=state_board.sensor_data,
        )
        self.sim_loop.add_module(sensor_module)

        # control module
        controller = QuadrotorPIDController(
            mass=self.mj_model.body_mass.sum(), dt=0.002
        )
        control_module = ControlModule(
            controller=controller,
            sensor_data=state_board.sensor_data,
            control_targets=state_board.control_targets,
            actuator_state=state_board.actuator_state,
        )
        self.sim_loop.add_module(control_module)

        # fault module
        fault_module = FaultModule(
            actuator_state=state_board.actuator_state,
            fault_status=state_board.fault_status,
        )
        self.sim_loop.add_module(fault_module)

    def apply_init_config(
        self,
        config: SimConfig,
    ):
        self.mj_data.qpos[0:3] = config.position
        self.mj_data.qpos[3:7] = config.quat
        self.mj_data.qvel[0:3] = config.velocity

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def generate_init_config(self) -> SimConfig:
        pos_range = np.array([-0.5, 0.5])
        vel_range = np.array([0.5, 2.0])
        tilt_range = np.array([-1, 1])
        yaw_range = np.array([-1, 1])

        return SimConfig.random(pos_range, vel_range, tilt_range, yaw_range)

    def __init__(self):
        super().__init__()
        # Action: 4 motor thrusts
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Observation: position + velocity + orientation + angular rates
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path("drone_env/quadrotor.xml")
        self.mj_data = mujoco.MjData(self.model)
        viewer = launch_viewer(self.mj_model, self.mj_data)

        self.sim_loop = SimLoop(viewer)
        state_board = StateBoard()
        self.apply_init_config(self.generate_init_config())

        self.init_modules(state_board)

        print("Gym env initialization complete")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
