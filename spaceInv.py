import gym
import random
import time
from stable_baselines3 import PPO
env = gym.make('SpaceInvaders-v0', render_mode='human')

model = PPO("MlpPolicy", env, verbose = 1)

for stip in range(500):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward, done)
env.close()
