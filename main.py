import mujoco
import mujoco.viewer
import numpy as np

from gym_env.drone_env import *
from sim.controllers.quadrotor_pid import QuadrotorPIDController
from sim.core.sim_loop import SimLoop
from sim.core.state import *
from sim.modules.control import ControlModule
from sim.modules.fault import FaultModule
from sim.modules.mujoco_dynamics import MujocoDynamicsModule
from sim.modules.sensors import SensorModule
from sim.sim_config import SimConfig



if __name__ == "__main__":
    model, data = load_model()

    pos_range = np.array([-0.5, 0.5])
    vel_range = np.array([0.5, 2.0])
    tilt_range = np.array([-1, 1])
    yaw_range = np.array([-1, 1])

    cfg = SimConfig.random(pos_range, vel_range, tilt_range, yaw_range)
    # controller = QuadrotorPIDController(mass, dt=0.002)
    # sim = Simulator(model, data, cfg,controller, failures)
    viewer = launch_viewer(model, data)
    sim_loop = SimLoop(viewer)

    state_board = StateBoard()
    apply_init_config(cfg, model, data)
    init_modules(sim_loop, model, data, state_board)

    sim_loop.run_forever()
