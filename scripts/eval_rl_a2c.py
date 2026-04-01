import os
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env.drone_env import DroneEnv


def make_env():
    return DroneEnv(render_mode="human")

def plot_trajectories(all_trajectories):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for traj in all_trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Quadrotor Trajectories (10 Runs)")

    plt.show()
    plt.savefig("trajectories_a2c.png")

def main():
    env = DummyVecEnv([make_env])

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = A2C.load("a2c_quadrotor", env=env)

    num_episodes = 1
    max_steps = 10000

    all_trajectories = []

    for ep in range(num_episodes):
        obs = env.reset()

        trajectory = []
        ep_rewards = []

        for step in range(max_steps):
            if step % 1000 == 0:
                print(f"Running step {step}")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
  
            pos = obs[0][:3]
            trajectory.append(pos)
            ep_rewards.append(reward[0])

        all_trajectories.append(np.array(trajectory))
        print(f"Episode {ep} length: {len(trajectory)}, total reward: {sum(ep_rewards)}")

    np.save("quadrotor_trajectories.npy", all_trajectories)

    print("Saved trajectories to quadrotor_trajectories.npy")

    plot_trajectories(all_trajectories)

if __name__ == "__main__":
    main()
