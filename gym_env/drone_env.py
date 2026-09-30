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
    ):
        """
        Initialize sim modules with shared state objects.

        Pipeline Data Flow

        FaultModule

        - Intercepts commands, applies actuator loss or scaling failures.
        - Writes modified motor commands to VehicleState, which MujocoDynamicsModule applies immediately after

        MujocoDynamicsModule

        - Steps MuJoCo physics.
        - Writes Ground Truth (true position, velocity, orientation quaternion, acceleration, and angular velocity) to VehicleState.

        SensorModule

        - Reads Ground Truth from VehicleState.
        - Applies noise, biases, latency, and drift.
        - Writes Sensor Data to VehicleState.

        RewardModule

        - Computes reward for RL purposes
        """

        # fault module
        fault_module = FaultModule(
            actuator_state=self.state_board.actuator_state,
            fault_status=self.state_board.fault_status,
        )
        self.sim_loop.add_module(fault_module)
        # dynamics module
        dynamics_module = MujocoDynamicsModule(
            model=self.mj_model,
            data=self.mj_data,
            dynamics_state=self.state_board.dynamics_state,
            actuator_state=self.state_board.actuator_state,
        )
        self.sim_loop.add_module(dynamics_module)

        # sensor module
        sensor_module = SensorModule(
            dynamics_state=self.state_board.dynamics_state,
            sensor_data=self.state_board.sensor_data,
        )
        self.sim_loop.add_module(sensor_module)

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

        return SimConfig.random(
            pos_range, vel_range, tilt_range, yaw_range, rng=self.np_random
        )

    def __init__(self, render:bool|None, model, data, fault_prob = 0.4):
        super().__init__()
        # Action: 4 motor thrusts
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Observation: position + velocity + orientation + angular rates
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.fault_prob = fault_prob

        # Load MuJoCo model
        self.mj_model = model
        self.mj_data = data
        viewer = launch_viewer(self.mj_model, self.mj_data) if render else None

        self.sim_loop = SimLoop(viewer)
        self.state_board = StateBoard()
        self.apply_init_config(self.generate_init_config())

        self.init_modules()

        print("Gym env initialization complete")

    def reset(self, seed=None, options=None):
        self.sim_timestamp=0
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for module in self.sim_loop.modules:
            module.reset()
        fault_index = -1
        fault_start_time = -1
        inject_fault = self.np_random.random() < self.fault_prob
        if inject_fault:
            fault_index = self.np_random.integers(0, 4)
            fault_start_time = self.np_random.uniform(1000, 5000)

        if options and 'fault_start_time' in options:
            inject_fault = True
            fault_index = options['fault_index']
            fault_start_time = options['fault_start_time']
        
        if inject_fault:
            self.state_board.fault_status.fault_active = True
            thrusts = np.ones(4, dtype=float)
            thrusts[fault_index] = 0.0
            self.state_board.fault_status.actuator_effectiveness[:] = thrusts
            self.state_board.fault_status.fault_start_time = fault_start_time
        # generate and apply new init config
        config = self.generate_init_config()
        self.apply_init_config(config)

        # pass new seed to sensor module and update sensor readings
        sensor_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim_loop.modules[2].apply_seed(sensor_seed)
        self.sim_loop.modules[2].HandleDispatch(0)
        
        return self._get_obs(), {}

    def step(self, action):
        # apply thrusts from action
        self.state_board.actuator_state.motor_thrusts[:] = action

        self.sim_loop.step(self.sim_timestamp)
        self.sim_timestamp += 2
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, False, {}

    def _get_obs(self):
        """
        State in the form of:
        [x, y, z, vx, vy, vz, roll, pitch, yaw, roll rate, pitch rate, yaw_rate]
        """
        # placeholder
        sensors = self.state_board.sensor_data

        obs = np.concatenate(
            [sensors.gps_position, sensors.gps_velocity, sensors.estimated_attitude, sensors.imu_gyro]
        ).astype(np.float32)

        return obs

    def _compute_reward(self):
        # Placeholder reward
        return 0.0

    def _check_done(self):
        # Placeholder termination condition
        if self.sim_loop.viewer:
            return not self.sim_loop.viewer.is_running()
        return False
