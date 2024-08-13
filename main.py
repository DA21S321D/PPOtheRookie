import gymnasium as gym
from PPO import PPO
env = gym.make('Pendulum-v1', render_mode='human')
model = PPO(env)
model.learn(1000000000)