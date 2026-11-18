import argparse
import os

import mujoco
import mujoco.viewer
import numpy as np

from gym_env.drone_env import DroneEnv, load_model
from sim.controllers.quadrotor_pid import QuadrotorPIDController
from sim.controllers.tile_q_controller import PlanarTileQController

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drone controller in MuJoCo viewer")
    parser.add_argument(
        "--controller",
        type=str,
        default="tile_q",
        choices=["tile_q", "pid"],
        help="Controller type: 'tile_q' or 'pid' (default: tile_q)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/planar_tile_q_best.npz",
        help="Path to trained weights file for tile_q controller",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless without opening MuJoCo viewer",
    )
    args = parser.parse_args()

    render = not args.no_render
    model, data = load_model()
    env = DroneEnv(render=render, model=model, data=data, fault_prob=0.0, substeps=10)
    mass = float(model.body_mass.sum())

    target_pos = np.array([0.0, 0.0, 0.0])
    target_state = {"pos": target_pos, "yaw": 0.0}
    env.set_target(target_state)

    if args.controller == "tile_q":
        # Initialize and load trained Tile-Coded Q-Learning Controller
        controller = PlanarTileQController(mass=mass, dt=0.02, training=False)
        if os.path.exists(args.weights):
            controller.load(args.weights)
            print(f"Loaded controller weights from {args.weights}")
        else:
            print(f"Warning: Checkpoint {args.weights} not found. Running with initial weights.")
    else:
        controller = QuadrotorPIDController(mass=mass, dt=0.02)
        print("Using QuadrotorPIDController")

    obs, info = env.reset(
        options={
            "planar_init": True,
            "disable_wind": True,
        }
    )
    controller.reset()

    done = False
    truncated = False
    total_reward = 0.0

    try:
        while not (done or truncated):
            if env.sim_loop.viewer and not env.sim_loop.viewer.is_running():
                break
            env.set_target(target_state)
            action = controller.compute_control(obs=obs, info=info, target_state=target_state)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Evaluation finished | Total Reward: {total_reward:.2f} | Final Pos: {obs[0:3]}")
    finally:
        env.close()
