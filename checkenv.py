from stable_baselines3.common.env_checker import check_env
from custom_env import SnakeEnv


env = SnakeEnv()
print(env.reset())
print(env.observation_space)
# It will check your custom environment and output additional warnings if needed
check_env(env)
