
from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
from sim.sim_config import SimConfig
from sim.controllers.quadrotor_pid import QuadrotorPIDController
import numpy as np
import time

model, data = load_model()
mass = model.body_mass.sum()
print(mass)
# proof-of-concept of randomized failures
# failures = [
#     ThrustLoss(start_time=np.random.uniform(low=3,high=6),alpha=np.random.uniform(low=0,high=1),prop=np.random.choice(range(4)))
#     for _ in range(3)]
failures = []

pos_range = np.array([-0.5, 0.5])
vel_range = np.array([0.5, 2.0])
tilt_range = np.array([-0.5, 0.5])
yaw_range = np.array([-0.5, 0.5])

cfg = SimConfig.random(pos_range,vel_range,tilt_range, yaw_range) 
controller = QuadrotorPIDController(mass, dt=0.002)
sim = Simulator(model, data, cfg,controller, failures)

viewer = launch_viewer(model, data)
while viewer.is_running():
    sim.step()
    viewer.sync()
    time.sleep(0.002)
