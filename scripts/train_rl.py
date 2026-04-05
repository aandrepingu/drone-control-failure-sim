import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

from drone_env.drone_env import DroneEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if "episode" in info.keys():
                    self.episode_rewards.append(info["episode"]["r"])
        return True

def make_env():
    return DroneEnv(
        render_mode=None,
        goal_pos=np.array([0.0, 0.0, 1.0]),
        pos_range=(-0.3, 0.3),   
        vel_range=(-0.1, 0.1),   
        tilt_range=(-0.1, 0.1),  
        yaw_range=(-0.1, 0.1),   
        warmup_steps=20,
    )

if __name__ == '__main__':
    NUM_ENVS = 4
    TOTAL_STEPS = 500_000

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0
    )

    reward_logger = RewardLoggerCallback()

    model.learn(total_timesteps=TOTAL_STEPS, callback=reward_logger)
    model.save("ppo_quadrotor")
    env.save("ppo_quadrotor_vecnorm.pkl")

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
    plt.savefig("rewards_ppo.png")
    plt.show()