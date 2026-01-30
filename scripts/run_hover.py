
from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
# from sim.scenarios.single_motor_loss import build_scenario
import time

model, data = load_model()
# failures = build_scenario()

sim = Simulator(model, data, )
viewer = launch_viewer(model, data)

while viewer.is_running():
    sim.step()
    time.sleep(0.002)
