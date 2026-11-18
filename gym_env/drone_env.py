from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.controllers.quadrotor_pid import QuadrotorPIDController
from sim.core.sim_loop import SimLoop
from sim.core.state import *
from sim.modules.control import ControlModule
from sim.modules.fault import FaultModule
from sim.modules.mujoco_dynamics import MujocoDynamicsModule
from sim.modules.sensors import SensorModule
from sim.modules.wind import WindModule
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
        self.fault_module = FaultModule(
            actuator_state=self.state_board.actuator_state,
            fault_status=self.state_board.fault_status,
        )
        self.sim_loop.add_module(self.fault_module)
        # wind module
        self.wind_module = WindModule(
            wind_state=self.state_board.wind_state,
            dynamics_state=self.state_board.dynamics_state,
        )
        self.sim_loop.add_module(self.wind_module)
        # dynamics module
        self.dynamics_module = MujocoDynamicsModule(
            model=self.mj_model,
            data=self.mj_data,
            dynamics_state=self.state_board.dynamics_state,
            actuator_state=self.state_board.actuator_state,
            wind_state=self.state_board.wind_state,
        )
        self.sim_loop.add_module(self.dynamics_module)

        # sensor module
        self.sensor_module = SensorModule(
            dynamics_state=self.state_board.dynamics_state,
            sensor_data=self.state_board.sensor_data,
        )
        self.sim_loop.add_module(self.sensor_module)

    def apply_init_config(
        self,
        config: SimConfig,
    ):
        self.mj_data.qpos[0:3] = config.position
        self.mj_data.qpos[3:7] = config.quat
        self.mj_data.qvel[0:3] = config.velocity

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # sync state board to new initial data
        self.state_board.dynamics_state.position[:] = self.mj_data.qpos[0:3]
        quat_wxyz = self.mj_data.qpos[3:7]
        self.state_board.dynamics_state.quaternion[:] = quat_wxyz
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self.state_board.dynamics_state.euler_angles[:] = R.from_quat(
            quat_xyzw
        ).as_euler("xyz")
        self.state_board.dynamics_state.linear_velocity[:] = self.mj_data.qvel[0:3]
        self.state_board.dynamics_state.angular_velocity[:] = self.mj_data.qvel[3:6]

        # forward wind force into WindState and WindModule
        self.state_board.wind_state.force[:] = config.wind_force
        if hasattr(self, "wind_module"):
            self.wind_module.reset(config.wind_force)

    def generate_init_config(self, options: dict | None = None) -> SimConfig:
        return SimConfig.sample(rng=self.np_random, options=options)

    def __init__(
        self,
        render: bool | None,
        model,
        data,
        fault_prob=0.4,
        substeps: int = 10,
    ):
        super().__init__()
        # Action: 4 motor thrusts in [0.0, 7.0] N
        self.action_space = gym.spaces.Box(
            low=0.0, high=7.0, shape=(4,), dtype=np.float32
        )
        # Observation: position + velocity + orientation + angular rates
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.fault_prob = fault_prob
        self.substeps = substeps
        self.max_sim_time_ms = 10_000
        self.sim_timestamp = 0

        # Load MuJoCo model
        self.mj_model = model
        self.mj_data = data
        viewer = launch_viewer(self.mj_model, self.mj_data) if render else None

        self.sim_loop = SimLoop(viewer)
        self.state_board = StateBoard()
        self.init_modules()
        self.apply_init_config(self.generate_init_config())

        print("Gym env initialization complete")

    def set_target(self, target_state: dict):
        if "pos" in target_state and target_state["pos"] is not None:
            self.state_board.control_targets.target_pos[:] = target_state["pos"]
        if "yaw" in target_state and target_state["yaw"] is not None:
            self.state_board.control_targets.target_attitude[2] = target_state["yaw"]

    def _setup_fault(self, options: dict | None = None):
        fault_index = -1
        fault_start_time = -1
        inject_fault = self.np_random.random() < self.fault_prob
        if inject_fault:
            fault_index = int(self.np_random.integers(0, 4))
            fault_start_time = float(self.np_random.uniform(1000, 5000))

        if options and "fault_start_time" in options:
            inject_fault = True
            fault_index = options.get("fault_index", 0)
            fault_start_time = options["fault_start_time"]

        if inject_fault:
            self.state_board.fault_status.fault_active = True
            thrusts = np.ones(4, dtype=float)
            thrusts[fault_index] = 0.0
            self.state_board.fault_status.actuator_effectiveness[:] = thrusts
            self.state_board.fault_status.fault_start_time = fault_start_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_timestamp = 0
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # call self.state_board.reset()
        self.state_board.reset()
        self._setup_fault(options=options)

        # sample episode configuration
        config = SimConfig.sample(rng=self.np_random, options=options)

        # apply config and reset simulation modules cleanly
        self.apply_init_config(config)
        for module in self.sim_loop.modules:
            if isinstance(module, SensorModule):
                module.reset(rng=self.np_random)
            else:
                module.reset()

        if options and options.get("disable_wind", False):
            self.wind_module.set_constant_wind(np.zeros(3))

        if options and options.get("planar_init", False):
            target_pos = self.state_board.control_targets.target_pos
            pos_bound = options.get("pos_bound", 0.4)
            vel_bound = options.get("vel_bound", 0.3)
            pitch_bound = options.get("pitch_bound", 0.2)
            pitch_rate_bound = options.get("pitch_rate_bound", 0.3)

            x_init = float(target_pos[0] + self.np_random.uniform(-pos_bound, pos_bound))
            y_init = float(target_pos[1])
            z_init = float(target_pos[2] + self.np_random.uniform(-pos_bound, pos_bound))

            vx = float(self.np_random.uniform(-vel_bound, vel_bound))
            vy = 0.0
            vz = float(self.np_random.uniform(-vel_bound, vel_bound))

            theta = float(self.np_random.uniform(-pitch_bound, pitch_bound))
            q_pitch = float(self.np_random.uniform(-pitch_rate_bound, pitch_rate_bound))

            quat = np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0], dtype=float)

            self.mj_data.qpos[0:3] = [x_init, y_init, z_init]
            self.mj_data.qpos[3:7] = quat
            self.mj_data.qvel[0:3] = [vx, vy, vz]
            self.mj_data.qvel[3:6] = [0.0, q_pitch, 0.0]

            mujoco.mj_forward(self.mj_model, self.mj_data)

            # sync state board dynamics state
            self.state_board.dynamics_state.position[:] = self.mj_data.qpos[0:3]
            quat_wxyz = self.mj_data.qpos[3:7]
            self.state_board.dynamics_state.quaternion[:] = quat_wxyz
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            self.state_board.dynamics_state.euler_angles[:] = R.from_quat(
                quat_xyzw
            ).as_euler("xyz")
            self.state_board.dynamics_state.linear_velocity[:] = self.mj_data.qvel[0:3]
            self.state_board.dynamics_state.angular_velocity[:] = self.mj_data.qvel[3:6]

        # evaluate initial sensor readings and return (obs, info)
        self.sensor_module.HandleDispatch(0)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.state_board.actuator_state.motor_thrusts[:] = np.clip(action, 0.0, 7.0)
        for _ in range(self.substeps):
            if self.sim_loop.viewer and not self.sim_loop.viewer.is_running():
                break
            self.sim_loop.step(self.sim_timestamp)
            self.sim_timestamp += 2
        obs = self._get_obs()
        terminated = self._check_done()
        reward = self._compute_reward()
        truncated = self.sim_timestamp >= self.max_sim_time_ms
        return obs, reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        """
        State in the form of:
        [x, y, z, vx, vy, vz, roll, pitch, yaw, roll rate, pitch rate, yaw_rate]
        """
        sensors = self.state_board.sensor_data

        obs = np.concatenate(
            [
                sensors.gps_position,
                sensors.gps_velocity,
                sensors.estimated_attitude,
                sensors.imu_gyro,
            ]
        ).astype(np.float32)

        return obs

    def _is_crashed(self) -> bool:
        dyn = self.state_board.dynamics_state
        target = self.state_board.control_targets.target_pos
        ex = dyn.position[0] - target[0]
        ez = dyn.position[2] - target[2]
        theta = dyn.euler_angles[1]
        phi = dyn.euler_angles[0]
        return bool(abs(ex) > 2.0 or abs(ez) > 2.0 or abs(theta) > 1.0 or abs(phi) > 1.0)

    def _compute_reward(self) -> float:
        dyn = self.state_board.dynamics_state
        target = self.state_board.control_targets.target_pos
        ex = dyn.position[0] - target[0]
        ez = dyn.position[2] - target[2]
        vx = dyn.linear_velocity[0]
        vz = dyn.linear_velocity[2]
        theta = dyn.euler_angles[1]
        q = dyn.angular_velocity[1]

        reward = 2.0 - (
            2.5 * (ex ** 2)
            + 4.0 * (ez ** 2)
            + 0.3 * (vx ** 2)
            + 0.5 * (vz ** 2)
            + 2.0 * (theta ** 2)
            + 0.1 * (q ** 2)
        )

        if abs(ex) < 0.1 and abs(ez) < 0.1 and abs(theta) < 0.1:
            reward += 2.0

        if self._is_crashed():
            reward -= 50.0

        return float(reward)

    def _check_done(self) -> bool:
        if self.sim_loop.viewer and not self.sim_loop.viewer.is_running():
            return True
        return self._is_crashed()

    def _get_info(self) -> dict:
        dyn = self.state_board.dynamics_state
        target = self.state_board.control_targets.target_pos
        ex = float(dyn.position[0] - target[0])
        ez = float(dyn.position[2] - target[2])
        return {
            "pos": dyn.position.copy(),
            "target_pos": target.copy(),
            "ex": ex,
            "ez": ez,
            "dist": float(np.sqrt(ex**2 + ez**2)),
        }

    def close(self):
        """Cleanly close MuJoCo viewer and release graphics resources."""
        if hasattr(self, "sim_loop") and self.sim_loop is not None:
            self.sim_loop.close()
