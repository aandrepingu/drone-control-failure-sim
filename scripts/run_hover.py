
from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
from sim.sim_config import SimConfig
from sim.controllers.quadrotor_pid import QuadrotorPIDController
import numpy as np
import time
import csv
from datetime import datetime

DT = 0.002
DURATION = 30.0
TOTAL_STEPS = int(DURATION / DT)

model, data = load_model()
mass = model.body_mass.sum()
print(mass)
# proof-of-concept of randomized failures
# failures = [
#     ThrustLoss(start_time=np.random.uniform(low=3,high=6),alpha=np.random.uniform(low=0,high=1),prop=np.random.choice(range(4)))
#     for _ in range(3)]
# failures = [ThrustLoss(start_time=np.random.uniform(low=1.0,high=3.0),alpha = 0.01,prop=np.random.choice(range(4)))]
failures = [
    ThrustLoss(
        start_time=np.random.uniform(low=1.0, high=3.0),
        alpha=0.1,
        prop=np.random.choice(range(4))
    )
]
pos_range = np.array([-0.5, 0.5])
vel_range = np.array([0.5, 2.0])
tilt_range = np.array([-1, 1])
yaw_range = np.array([-1, 1])

cfg = SimConfig.random(pos_range,vel_range,tilt_range, yaw_range) 
controller = QuadrotorPIDController(mass, dt=DT)
sim = Simulator(model, data, cfg, controller, failures)

# timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename   = f"trajectory_{timestamp}.csv"
filename = "thrust_loss_low_alpha.csv"
fieldnames = [
    "step", "time",
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "motor_0", "motor_1", "motor_2", "motor_3"
]

viewer = launch_viewer(model, data)

"""
while viewer.is_running():
    sim.step()
    viewer.sync()
    time.sleep(0.002)
"""

with open(filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for step in range(TOTAL_STEPS):
        if not viewer.is_running():
            break

        sim.step()
        viewer.sync()

        # -- read state from MuJoCo data --
        pos     = data.qpos[0:3]        # x, y, z
        quat    = data.qpos[3:7]        # w, x, y, z
        vel     = data.qvel[0:3]        # linear velocity
        ang_vel = data.qvel[3:6]        # angular velocity
        motors  = data.ctrl[:]          # motor commands (4 rotors)

        writer.writerow({
            "step":      step,
            "time":      round(step * DT, 4),
            "pos_x":     pos[0],     "pos_y":     pos[1],     "pos_z":     pos[2],
            "vel_x":     vel[0],     "vel_y":     vel[1],     "vel_z":     vel[2],
            "quat_w":    quat[0],    "quat_x":    quat[1],    "quat_y":    quat[2],  "quat_z": quat[3],
            "ang_vel_x": ang_vel[0], "ang_vel_y": ang_vel[1], "ang_vel_z": ang_vel[2],
            "motor_0":   motors[0],  "motor_1":   motors[1],  "motor_2":   motors[2], "motor_3": motors[3],
        })

        time.sleep(DT)

print(f"Trajectory saved to {filename}  ({TOTAL_STEPS} steps, {DURATION}s)")