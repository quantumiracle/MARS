import random
import gym
import numpy as np

class LaserTagWrapper():
    """ Wrap single agent OpenAI gym atari game to be multi-agent version """
    def __init__(self, env):
        super(LaserTagWrapper, self).__init__()
        self.env = env
        self.agents = ['1', '2']
        self.num_agents = len(self.agents)
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = self.env.action_space
        self.action_spaces = {name: self.action_space for name in self.agents}
    
    @property
    def spec(self):
        return self.env.spec

    def reset(self):
        obs = self.env.reset()
        return obs

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        self.env.render()

    def step(self, actions):
        action_dict = {}
        for ag, ac in zip(self.agents, actions):
            action_dict[ag] = ac
        obs, reward, done, info = self.env.step(action_dict)
        return obs, reward, [done, done], [info, info] 

    def close(self):
        self.env.close()