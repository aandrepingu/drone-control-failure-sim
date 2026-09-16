import time
from pathlib import Path

import mujoco
import numpy as np

from sim.controllers.quadrotor_pid import QuadrotorPIDController
from sim.core.sim_loop import SimLoop
from sim.core.sim_module import SimModule
from sim.core.state import *
from sim.failures.thrust_loss import ThrustLoss
from sim.modules.control import ControlModule
from sim.modules.mujoco_dynamics import MujocoDynamicsModule
from sim.modules.sensors import SensorModule
from sim.modules.fault import FaultModule
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


def init_modules(
    sim_loop: SimLoop,
    model: mujoco.MjModel,
    data: mujoco.MjData,
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
        model=model,
        data=data,
        dynamics_state=state_board.dynamics_state,
        actuator_state=state_board.actuator_state,
    )
    sim_loop.add_module(dynamics_module)

    # sensor module
    sensor_module = SensorModule(
        dynamics_state=state_board.dynamics_state, sensor_data=state_board.sensor_data
    )
    sim_loop.add_module(sensor_module)

    # control module
    controller = QuadrotorPIDController(mass=model.body_mass.sum(), dt=0.002)
    control_module = ControlModule(
        controller=controller,
        sensor_data=state_board.sensor_data,
        control_targets=state_board.control_targets,
        actuator_state=state_board.actuator_state,
    )
    sim_loop.add_module(control_module)

    # fault module
    fault_module = FaultModule(
        actuator_state=state_board.actuator_state, fault_status=state_board.fault_status
    )
    sim_loop.add_module(fault_module)


if __name__ == "__main__":
    model, data = load_model()

    # pos_range = np.array([-0.5, 0.5])
    # vel_range = np.array([0.5, 2.0])
    # tilt_range = np.array([-1, 1])
    # yaw_range = np.array([-1, 1])

    # cfg = SimConfig.random(pos_range,vel_range,tilt_range, yaw_range)
    # controller = QuadrotorPIDController(mass, dt=0.002)
    # sim = Simulator(model, data, cfg,controller, failures)
    viewer = launch_viewer(model, data)
    sim_loop = SimLoop(viewer)

    state_board = StateBoard()
    init_modules(sim_loop, model, data, state_board)

    sim_loop.run_forever()
