from sim.model import load_model
from sim.sim_loop import Simulator
from sim.viewer import launch_viewer
from sim.failures.thrust_loss import ThrustLoss
from sim.sim_config import SimConfig
from sim.controllers.quadrotor_pid import QuadrotorPIDController
import numpy as np
import mujoco
import time
import csv
import json
import os
from datetime import datetime

DT = 0.002
DURATION = 30.0
TOTAL_STEPS = int(DURATION / DT)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"runs/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

FIELDNAMES = [
    "step", "time",
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "motor_0", "motor_1", "motor_2", "motor_3"
]


def make_quat(tilt_axis, tilt_angle, yaw_angle):
    """Compose a yaw + tilt quaternion exactly as SimConfig.random does."""
    tilt_axis = np.array(tilt_axis, dtype=float)
    tilt_axis[2] = 0.0
    tilt_axis /= np.linalg.norm(tilt_axis)

    q_tilt = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_tilt, tilt_axis, tilt_angle)

    q_yaw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), yaw_angle)

    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, q_yaw, q_tilt)
    return quat

SCENARIOS = [
    {
        "id": "s01_hover_stable",
        "description": "Near-origin, nearly level, zero velocity — baseline, no failure",
        "failure": None,
        "position": np.array([0.0,  0.0,  1.0]),
        "velocity": np.array([0.0,  0.0,  0.0]),
        "tilt_axis": np.array([1.0,  0.0,  0.0]),  # tiny forward tilt
        "tilt_angle": 0.05,
        "yaw_angle": 0.0,
    },
    {
        "id": "s02_diagonal_flight",
        "description": "Off-centre start, diagonal velocity, moderate tilt — no failure",
        "failure": None,
        "position": np.array([-0.5, -0.5,  0.5]),
        "velocity": np.array([ 0.8,  0.8,  0.4]),
        "tilt_axis": np.array([ 1.0,  1.0,  0.0]),
        "tilt_angle": 0.25,
        "yaw_angle": 0.3,
    },
    {
        "id": "s03_aggressive_tilt_partial_failure",
        "description": "Large tilt + lateral velocity, prop 2 fails at t=2s (alpha=0.5)",
        "failure": {"prop": 2, "alpha": 0.5, "start_time": 2.0},
        "position": np.array([ 0.3, -0.3,  1.5]),
        "velocity": np.array([ 1.0,  0.0, -0.3]),
        "tilt_axis": np.array([ 1.0, -0.5,  0.0]),
        "tilt_angle": 0.6,
        "yaw_angle": -0.5,
    },
]

def run_scenario(scenario, model, data, viewer):
    sid = scenario["id"]
    print(f"\n{'='*60}")
    print(f"Running : {sid}")
    print(f"  {scenario['description']}")

    quat = make_quat(scenario["tilt_axis"], scenario["tilt_angle"], scenario["yaw_angle"])
    cfg = SimConfig(
        position=scenario["position"],
        velocity=scenario["velocity"],
        quat=quat,
    )

    failures = (
        [ThrustLoss(start_time=scenario["failure"]["start_time"],
                    alpha=scenario["failure"]["alpha"],
                    prop=scenario["failure"]["prop"])]
        if scenario["failure"] else []
    )

    mass = model.body_mass.sum()
    controller = QuadrotorPIDController(mass, dt=DT)
    sim = Simulator(model, data, cfg, controller, failures)

    csv_path = os.path.join(OUT_DIR, f"{sid}.csv")
    meta_path = os.path.join(OUT_DIR, f"{sid}_meta.json")

    meta = {
        "scenario_id": sid,
        "description": scenario["description"],
        "duration_s": DURATION,
        "dt": DT,
        "total_steps": TOTAL_STEPS,
        "position": scenario["position"].tolist(),
        "velocity": scenario["velocity"].tolist(),
        "tilt_axis": scenario["tilt_axis"].tolist(),
        "tilt_angle": scenario["tilt_angle"],
        "yaw_angle": scenario["yaw_angle"],
        "quat": quat.tolist(),
        "failure": scenario["failure"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for step in range(TOTAL_STEPS):
            if not viewer.is_running():
                print("Viewer closed - aborting.")
                return False

            sim.step()
            viewer.sync()

            pos = data.qpos[0:3]
            quat_s = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            motors = data.ctrl[:]

            writer.writerow({
                "step": step,
                "time": round(step * DT, 4),
                "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
                "quat_w": quat_s[0], "quat_x": quat_s[1], "quat_y": quat_s[2], "quat_z": quat_s[3],
                "ang_vel_x": ang_vel[0], "ang_vel_y": ang_vel[1], "ang_vel_z": ang_vel[2],
                "motor_0": motors[0], "motor_1": motors[1], "motor_2": motors[2], "motor_3": motors[3],
            })

            time.sleep(DT)

    print(f"CSV {csv_path}")
    print(f"Meta {meta_path}")
    return True

model, data = load_model()
viewer = launch_viewer(model, data)

for scenario in SCENARIOS:
    ok = run_scenario(scenario, model, data, viewer)
    if not ok:
        break

print(f"\nAll done. Output: {OUT_DIR}/")