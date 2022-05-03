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
        self.max_transition=  self.env.max_transition
        self.OneHotObs = False
        self.RichObs = True if 'RichObs' in env.__class__.__name__ else False
        try:
            self.OneHotObs = self.env.OneHotObs
        except:
            pass

        if self.RichObs:  # rich observation env uses Box observation space instead Discrete
            self.observation_space = env.observation_space
        else: # others have a Discrete observation space, fake it a Box
            if self.OneHotObs:
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(env.observation_space.n,))
            else:
                self.observation_space = gym.spaces.Box(low=0.0, high=env.observation_space.n, shape=(1,))
        self.observation_spaces = {ag: self.observation_space for ag in self.agents}
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.action_spaces = {a:self.action_space for a in self.agents}
        self.curr_step = 0
        self.NEsolver()

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        obs = self.env.reset()
        self.curr_step = 0
        if self.RichObs or self.OneHotObs:
            return [obs, obs]
        else:
            return [[obs], [obs]]

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action, *args, **kwargs):
        self.curr_step += 1
        a = self.env.action_map(action)
        obs, r, done, info = self.env.step(a, *args, **kwargs)
        if self.env.max_transition is not None and self.curr_step >= self.env.max_transition:
            done = True

        if self.RichObs or self.OneHotObs:
            return [obs, obs], [r, -r], [done, done], [info, info]
        else: 
            return [[obs], [obs]], [r, -r], [done, done], [info, info]

    def visualize_MDP(self, ):
        self.env.visualize_MDP()

    def NEsolver(self, *args, **kwargs):
        try: 
            self.Nash_v, self.Nash_q, self.Nash_strategies = self.env.NEsolver(*args, **kwargs)
            # print(self.Nash_strategies)
            return self.Nash_v, self.Nash_q, self.Nash_strategies

        except:
            pass



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
