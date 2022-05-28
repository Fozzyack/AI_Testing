import os
import gym
from stable_baselines3 import PPO
env = gym.make("LunarLander-v2")
env.reset()

TIMESTEPS = 10000
env.reset()
models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)



model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=logdir)
for i in range(1, 150):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

'''for stip in range(500):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward, done)'''
env.close()
