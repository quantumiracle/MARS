import gym
import numpy as np

class MDPWrapper():
    """ 
    """   
    def __init__(self, env):  
        super(MDPWrapper, self).__init__()
        self.env = env
        self.agents = ['agent0', 'agent1']
        self.num_agents = len(self.agents)      
        # for observation, discrete to box, fake space
        self.observation_space = gym.spaces.Box(low=0.0, high=env.observation_space.n, shape=(1,))
        self.observation_spaces = {a:self.observation_space for a in self.agents}
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.action_spaces = {a:self.action_space for a in self.agents}
        self.curr_step = 0

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        obs = self.env.reset()
        self.curr_step = 0
        return [[obs, obs]]

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        self.curr_step += 1
        a = self.env.action_map(action)
        obs, r, done, info = self.env.step(a)
        if self.env.max_transition is not None and self.curr_step >= self.env.max_transition:
            done = True
        return [[obs], [obs]], [r, -r], [done, done], [info, info]

    def visualize_MDP(self, ):
        self.env.visualize_MDP()


if __name__ == '__main__':
    # single agent version
    # env = CombinatorialLock2Player(2, wrapped=False).env
    # obs = env.reset()
    # print(obs)
    # done = False
    # while not done:
    #     obs, r, done, _ = env.step(0)
    #     print(obs, r, done)

    # two agent version
    # env = CombinatorialLock2PlayerWrapper(env)
    env = CombinatorialLock2Player(2, wrapped=True).env
    print(env.observation_space, env.action_space)
    # env.render()
    obs = env.reset()
    print(obs)
    done = False
    while not np.any(done):
        obs, r, done, _ = env.step([0,1])
        print(obs, r, done)
