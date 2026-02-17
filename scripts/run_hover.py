
from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
from sim.sim_config import SimConfig
import numpy as np
import time

model, data = load_model()

# proof-of-concept of randomized failures
failures = [
    ThrustLoss(start_time=np.random.uniform(low=0,high=3),alpha=np.random.uniform(low=0,high=1),prop=np.random.choice(range(4)))
    for _ in range(3)]

pos_range = np.array([-0.5, 0.5])
vel_range = np.array([0.5, 2.0])
tilt_range = np.array([0.1, 0.5])
yaw_range = np.array([0.1, 0.5])

cfg = SimConfig.random(pos_range,vel_range,tilt_range, yaw_range) 

sim = Simulator(model, data, cfg,failures)

viewer = launch_viewer(model, data)
while viewer.is_running():
    sim.step()
    viewer.sync()
    time.sleep(0.002)
