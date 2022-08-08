###
# robosumo requires a lower version of gym:
# !pip install gym==0.16
###

import robosumo.envs
import gym
import numpy as np

env_name = "RoboSumo-Ant-vs-Ant-v0"
env = gym.make(env_name)
print(env.observation_space, env.action_space)
observation = env.reset()
print(observation)
for i in range(10):
    env.render(mode='rgb_array')
    action = np.random.uniform(0,1,(2, 8))
    print(action)
    observation, reward, done, infos = env.step(action)

