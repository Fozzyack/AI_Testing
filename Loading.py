import os
import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
env = gym.make("LunarLander-v2")
env.reset()
TIMESTEPS = 10000
env.reset()
models_dir = "models/PPO"
model_path = f"{models_dir}/1490000"


model = PPO.load(model_path, env = env)

for step in range(10):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward, done)
env.close()
