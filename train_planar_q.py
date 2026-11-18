import argparse
import os
import time
from collections import deque

import numpy as np

from gym_env.drone_env import DroneEnv, load_model
from sim.controllers.tile_q_controller import PlanarTileQController


def train(
    num_episodes: int = 3000,
    substeps: int = 10,
    checkpoint_dir: str = "checkpoints",
    load_weights: str | None = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "planar_tile_q_best.npz")
    final_checkpoint_path = os.path.join(checkpoint_dir, "planar_tile_q_final.npz")

    # Load physics model and initialize headless environment
    model, data = load_model()
    env = DroneEnv(
        render=False,
        model=model,
        data=data,
        fault_prob=0.0,
        substeps=substeps,
    )
    mass = float(model.body_mass.sum())

    target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target_state = {"pos": target_pos, "yaw": 0.0}
    env.set_target(target_state)

    # Instantiate Q-learning controller
    controller = PlanarTileQController(
        mass=mass,
        dt=0.02,
        num_tilings=16,
        alpha=0.20,
        gamma=0.98,
        epsilon=0.30,
        training=True,
    )

    if load_weights and os.path.exists(load_weights):
        controller.load(load_weights)
        print(f"Loaded existing weights from {load_weights}")

    # Epsilon decay schedule parameters
    eps_start = 0.30
    eps_min = 0.02
    eps_decay_episodes = 2200

    # Metric tracking
    reward_window = deque(maxlen=50)
    length_window = deque(maxlen=50)
    dist_window = deque(maxlen=50)

    best_score = -float("inf")
    start_time = time.time()

    print("=" * 75)
    print("Starting 6D Planar Quadrotor Tile-Coded Q-Learning Training")
    print(f"Total Episodes: {num_episodes} | Substeps: {substeps} (50 Hz control rate)")
    print(f"Goal: Hover at target {target_pos} | Checkpoints -> {checkpoint_dir}/")
    print("=" * 75)

    for episode in range(num_episodes):
        # Curriculum Schedule:
        # Episodes 0-800: small initial perturbations around target
        # Episodes 800-2400: smoothly ramp perturbations to full scale
        # Episodes 2400+: full perturbations
        prog = min(1.0, max(0.0, (episode - 800) / 1600.0))
        pos_bound = float(0.15 + (0.50 - 0.15) * prog)
        vel_bound = float(0.10 + (0.35 - 0.10) * prog)
        pitch_bound = float(0.08 + (0.25 - 0.08) * prog)
        pitch_rate_bound = float(0.10 + (0.30 - 0.10) * prog)

        options = {
            "planar_init": True,
            "disable_wind": True,
            "pos_bound": pos_bound,
            "vel_bound": vel_bound,
            "pitch_bound": pitch_bound,
            "pitch_rate_bound": pitch_rate_bound,
        }

        # Update exploration rate
        decay_factor = min(episode, eps_decay_episodes) / float(eps_decay_episodes)
        controller.epsilon = float(
            max(eps_min, eps_start * ((eps_min / eps_start) ** decay_factor))
        )

        obs, info = env.reset(options=options)
        env.set_target(target_state)
        controller.reset()

        ep_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = controller.compute_control(
                obs=obs, info=info, target_state=target_state
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller.update(
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                target_state=target_state,
            )
            obs = next_obs
            ep_reward += reward
            step_count += 1

        final_dist = info.get("dist", 0.0)
        reward_window.append(ep_reward)
        length_window.append(step_count)
        dist_window.append(final_dist)

        # Logging every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = float(np.mean(reward_window))
            avg_length = float(np.mean(length_window))
            avg_dist = float(np.mean(dist_window))
            elapsed = time.time() - start_time
            sps = (episode + 1) / max(elapsed, 1e-3)

            print(
                f"Ep {episode + 1:4d}/{num_episodes:4d} | "
                f"eps: {controller.epsilon:.3f} | "
                f"Avg Rew: {avg_reward:7.1f} | "
                f"Avg Len: {avg_length:5.1f}/500 | "
                f"Avg Dist: {avg_dist:.3f}m | "
                f"{sps:.1f} ep/s"
            )

            # Combined score prioritizing survival length and high reward
            score = avg_length * 2.0 + avg_reward
            if score > best_score:
                best_score = score
                controller.save(best_checkpoint_path)
                print(
                    f"  --> Saved new best model checkpoint: {best_checkpoint_path} "
                    f"(Avg Len: {avg_length:.1f}, Avg Rew: {avg_reward:.1f})"
                )

    # Save final policy
    controller.save(final_checkpoint_path)
    if not os.path.exists(best_checkpoint_path):
        controller.save(best_checkpoint_path)

    total_time = time.time() - start_time
    print("=" * 75)
    print(
        f"Training completed in {total_time:.1f} s ({total_time / 60.0:.2f} min). Final weights saved to {final_checkpoint_path}"
    )
    print("=" * 75)
    return controller


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 6D Planar Quadrotor Tile-Coded Q-Learning Agent"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Number of training episodes (default: 3000)",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Physics substeps per RL step (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save weights (default: checkpoints)",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to initial weights file to resume training from",
    )
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        substeps=args.substeps,
        checkpoint_dir=args.checkpoint_dir,
        load_weights=args.load_weights,
    )
