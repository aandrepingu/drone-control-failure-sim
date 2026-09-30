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
    env = DroneEnv(render=True, model=model, data=data)
    mass = model.body_mass.sum()
    obs, info = env.reset(options={"""'fault_start_time' : 3000, 'fault_index': np.random.randint(low=0,high=4)"""})

    done = False
    action = None
    pos = None
    euler = None
    gyro = None
    target_pos = np.zeros(3)
    target_yaw = None
    target_state = {"pos": target_pos, "yaw": target_yaw}

    controller = QuadrotorPIDController(mass=mass, dt=0.002)
    while not done:
        # target_state = trajectory.get_target(env.time)

        # # Update env with target so it can calculate reward correctly
        # env.set_target(target_state)

        # calculate control action

        action = controller.compute_control(obs=obs, info={}, target_state=target_state)

        # step plant forward
        # this will also render the sim if render=True within gym env
        obs, reward, done, truncated, info = env.step(action)
        pos = obs[0:3]
        euler = obs[6:9]
        body_rates = obs[9:12]

        # telemetry.log(obs, action, target_state, info)
