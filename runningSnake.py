import os
import gym
from stable_baselines3 import PPO
from custom_env import SnakeEnv
import time
env = SnakeEnv()
env.reset()

TIMESTEPS = 10000
models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)



model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=logdir)
for i in range(1, 1000000000000000000000000000):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
