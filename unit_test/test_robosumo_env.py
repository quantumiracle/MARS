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
for _ in range(3):
    observation = env.reset()
    # print(observation)
    for i in range(10000):
        env.render(mode='human')  # only 'human' mode will successfully render this; 'rgb_image' or 'rgb_array' will not
        action = np.random.uniform(-1,1,(2, 8))
        # print(action)
        observation, reward, done, infos = env.step(action)
        # print(observation[0].shape)
        print(done, reward)
        if np.any(done):
            break

