
from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
# from sim.scenarios.single_motor_loss import build_scenario
import time

model, data = load_model()
failures = [ThrustLoss(start_time=3, alpha=0.01,prop=0)]

sim = Simulator(model, data, failures)
viewer = launch_viewer(model, data)
while viewer.is_running():
    sim.step()
    viewer.sync()
    time.sleep(0.002)
