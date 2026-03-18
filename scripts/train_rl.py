import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from drone_env.drone_env import DroneEnv

def make_env():
    return DroneEnv(render_mode="human")

if __name__ == '__main__':
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env = VecMonitor(env)

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

    model.learn(total_timesteps=300_000)
    model.save("ppo_quadrotor")
    env.save("ppo_quadrotor_vecnorm.pkl")