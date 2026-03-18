from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
from sim.failures.stuck_actuator import StuckActuator
from sim.sim_config import SimConfig
from sim.controllers.quadrotor_rl import QuadrotorRLController
import numpy as np
import time

def make_hover_policy(mass):
    hover_thrust = mass * 9.81 / 4.0

    def policy(obs):
        return np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float64)

    return policy


if __name__ == '__main__':
    model, data = load_model()
    mass = model.body_mass.sum()
    dt = 0.002

    pos = np.array([0.0, 0.0, 1.0])
    vel = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    cfg = SimConfig(pos, vel, quat)

    policy = make_hover_policy(mass)
    controller = QuadrotorRLController(policy, dt=dt)

    failures = []
    sim = Simulator(model, data, cfg, controller, failures)

    viewer = launch_viewer(model, data)
    while viewer.is_running():
        sim.step(dt=dt)
        viewer.sync()
        time.sleep(dt)