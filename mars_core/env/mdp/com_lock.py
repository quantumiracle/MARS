from blackhc import mdp
from blackhc.mdp import example
import gym
import numpy as np

class CombinatorialLock2Player():
    def __init__(self, layers=2, wrapped=False):
        self.action_set = ['a0', 'a1']
        num_actions = len(self.action_set)
        self.layers = 5
        self.pos_reward = 1
        self.spec = mdp.MDPSpec()
        self.env = self._construct_game()
        setattr(self.env, 'num_actions', num_actions)
        if wrapped:
            self.wrap2player()

    def _construct_game(self, ):
        start = self.spec.state('start')
        mid_states = []
        for i in range(self.layers):
            # each layer has a good and bad state
            if i == self.layers-1:
                mid_states.append([self.spec.state(f'good_{i+1}', terminal_state=True), self.spec.state(f'bad_{i+1}', terminal_state=True)])
            else:
                mid_states.append([self.spec.state(f'good_{i+1}'), self.spec.state(f'bad_{i+1}')])

        mutual_actions = []
        for a in self.action_set:
            for b in self.action_set:
                mutual_actions.append(self.spec.action())

        self.spec.transition(start, mutual_actions[0], mdp.NextState(mid_states[0][0]))
        self.spec.transition(start, mutual_actions[3], mdp.NextState(mid_states[0][0]))
        self.spec.transition(start, mutual_actions[1], mdp.NextState(mid_states[0][1]))
        self.spec.transition(start, mutual_actions[2], mdp.NextState(mid_states[0][1]))

        for i in range(self.layers-1):
            if i == self.layers-2:
                self.spec.transition(mid_states[i][0], mutual_actions[0], mdp.Reward(self.pos_reward))
                self.spec.transition(mid_states[i][0], mutual_actions[3], mdp.Reward(self.pos_reward))

            self.spec.transition(mid_states[i][0], mutual_actions[0], mdp.NextState(mid_states[i+1][0]))
            self.spec.transition(mid_states[i][0], mutual_actions[3], mdp.NextState(mid_states[i+1][0]))
            self.spec.transition(mid_states[i][0], mutual_actions[1], mdp.NextState(mid_states[i+1][1]))
            self.spec.transition(mid_states[i][0], mutual_actions[2], mdp.NextState(mid_states[i+1][1]))

            self.spec.transition(mid_states[i][1], mutual_actions[0], mdp.NextState(mid_states[i+1][1]))
            self.spec.transition(mid_states[i][1], mutual_actions[3], mdp.NextState(mid_states[i+1][1]))
            self.spec.transition(mid_states[i][1], mutual_actions[1], mdp.NextState(mid_states[i+1][1]))
            self.spec.transition(mid_states[i][1], mutual_actions[2], mdp.NextState(mid_states[i+1][1]))

        return self._to_gym_env()

    def _to_gym_env(self):
        env = self.spec.to_env()
        return env

    def visualize_MDP(self,):
        spec_graph = self.spec.to_graph()
        spec_png = mdp.graph_to_png(spec_graph)

        mdp.display_mdp(self.spec)

    def wrap2player(self, ):
        """
        Wrap the single player game to a two player game.
        """
        self.env = CombinatorialLock2PlayerWrapper(self.env)


class CombinatorialLock2PlayerWrapper():
    """ 
    """   
    def __init__(self, env):  
        super(CombinatorialLock2PlayerWrapper, self).__init__()
        self.env = env
        # self.action_spaces = self.env.action_spaces
        self.agents = ['agent0', 'agent1']
        self.num_agents = len(self.agents)      
        # for observation, discrete to box, fake space
        self.observation_space = gym.spaces.Box(low=0.0, high=env.observation_space.n, shape=(1,))
        self.observation_spaces = {a:self.observation_space for a in self.agents}
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.action_spaces = {a:self.action_space for a in self.agents}

    @property
    def spec(self):
        return self.env.spec

    def reset(self, observation=None):
        obs = self.env.reset()
        return [[obs, obs]]

    def seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action):
        """Action map from one player to two player.
            p2 0  1
        p1
        0      0  1
        1      2  3
        """
        if np.sum(action) == 0:
            a = 0
        elif np.sum(action) == 2:
            a = 3
        elif action[0] < action[1]:
            a = 1
        elif action[0]> action[1]:
            a = 2
        else:
            raise NotImplementedError
        obs, r, done, info = self.env.step(a)
        return [[obs], [obs]], [r, -r], [done, done], [info, info]


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
