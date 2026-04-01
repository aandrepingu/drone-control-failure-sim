import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from drone_env.drone_env import DroneEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        return True
    
def make_env():
    return DroneEnv(render_mode="human")

if __name__ == '__main__':
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env = VecMonitor(env)

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=5,
        gamma=0.99,
        learning_rate=7e-4,
        ent_coef=0.0,
    )

    reward_logger = RewardLoggerCallback()

    model.learn(total_timesteps=300_000)
    model.save("a2c_quadrotor")
    env.save("a2c_quadrotor_vecnorm.pkl")

    rewards = reward_logger.episode_rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    window = 20
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), rewards_smooth, label=f"{window}-episode MA", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards During Training")
    plt.grid(True)
    plt.legend()
    plt.savefig("rewards_a2c.png")
    plt.show()