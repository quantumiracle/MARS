from blackhc import mdp
from .mdp_wrapper import MDPWrapper
import numpy as np
import itertools
import ecos
import numpy as np
from scipy.sparse import csr_matrix
from .utils.nash_solver import NashEquilibriumECOSSolver

class RandomMDP():
    """A random MDP game for two players. The action sets for the two players can be different.
    """
    def __init__(self, wrapped=False):
        self.two_action_sets = [['a0', 'a1'], ['b0', 'b1',]]
        self.num_actions = [len(a) for a in self.two_action_sets]
        self.num_agents = 2
        self.max_transition = None # maximum number of transition steps, no limit here
        self.num_non_terminal_states = 3
        self.reward_range = [-2,2]  # range of uniform distribution for random rewards
        self.spec = mdp.MDPSpec()
        self.env = self._construct_game()
        assert self.num_actions[0] == self.num_actions[1]
        setattr(self.env, 'num_actions', self.num_actions[0])  # when two players have the same size action sets
        setattr(self.env, 'action_map', self.action_map)
        setattr(self.env, 'visualize_MDP', self.visualize_MDP)
        setattr(self.env, 'max_transition', self.max_transition)
        self.NEsolver(verbose=False)
        if wrapped:
            self.wrap2player()

    def _set_payoff(self,):
        """Construct random payoff matrix for each state.
        """
        p1_payoff = [] # only give player 1 payoff, since it's zero-sum game, player 2's is the inverse.
        for j in range(len(self.state_set)-1):
            p1_payoff.append(np.random.uniform(*self.reward_range, size=np.sum(self.num_actions)))
        return p1_payoff

    def _construct_game(self, ):
        self.action_sets = []
        for per_player_actions in self.num_actions:
            index_actions = np.arange(per_player_actions)
            self.action_sets.append([str(a) for a in index_actions])
        print("Action set for each player: ", self.action_sets)

        actions = list(itertools.product(*self.action_sets))
        single_player_actions = [''.join(a) for a in actions]
        print("Merged action set: ", single_player_actions)
        single_player_action_set = []
        for a in single_player_actions:
            single_player_action_set.append(self.spec.action('A'+a))

        self.state_set =[]
        for i in range(self.num_non_terminal_states):
            self.state_set.append(self.spec.state(f"s{i}"))
        self.state_set.append(self.spec.state(f"s{i+1}", terminal_state=True))
        print(self.state_set)

        self.p1_payoff = self._set_payoff()

        for j in range(len(self.state_set)-1):
            for i, a in enumerate(single_player_action_set):
                self.spec.transition(self.state_set[j], a, mdp.NextState(self.state_set[j+1]))
                self.spec.transition(self.state_set[j], a, mdp.Reward(self.p1_payoff[j][i]))  # random reward for each transition
        
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
        self.env = MDPWrapper(self.env)

    def action_map(self, action):
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
        return a

    def NEsolver(self, verbose=False):
        """This will solve the Nash equilibrium for each game state.
        The NE of later state will not affect the NE strategy of previous state in this game, just
        the NE value of previous state will be added by the NE value of later state, for both pure NE 
        or mixed NE. 
        This is caused by that different actions/strategies lead to exactly the same next state, the same
        NE value will be back-propagated through every action entry symmetrically. A zero-sum matrix plus a 
        constant gives the same NE strategy.
        """
        next_state_NE_value = 0.
        ne_list=[]
        ne_value_list=[]
        for s, s_payoff in zip(self.state_set[-2::-1], self.p1_payoff[::-1]):  # inverse order (except the terminal state)
            s_payoff = s_payoff.reshape(self.num_actions)
            if verbose:
                print(f'Payoff matrix of {s}: \n {s_payoff}')
            s_payoff = s_payoff+next_state_NE_value # NE[Q(s,a)] <- NE[R(s,a)+NE[Q(s_, a_)]]
            ne = NashEquilibriumECOSSolver(s_payoff)
            ne_value = ne[0]@s_payoff@ne[1].T
            next_state_NE_value = ne_value # later state NE value will be passed to the payoff matrix of previous state
            ne_list.append(ne)
            ne_value_list.append(ne_value)
            if verbose:
                print('Nash equilibrium strategy: ', ne)
                print('Nash equilibrium value for player 1: ', ne_value)
        print(self.state_set[-2::-1], ne_list, ne_value_list)
        return self.state_set[-2::-1], ne_list, ne_value_list
            

if __name__ == '__main__':
    # single agent version
    # env = CombinatorialLock(2, wrapped=False).env
    # obs = env.reset()
    # print(obs)
    # done = False
    # while not done:
    #     obs, r, done, _ = env.step(0)
    #     print(obs, r, done)

    # two agent version
    # env = CombinatorialLock2PlayerWrapper(env)
    env = RandomMDP(wrapped=True).env
    env.visualize_MDP()
    print(env.observation_space, env.action_space)
    # env.render()
    obs = env.reset()
    print(obs)
    done = False
    while not np.any(done):
        obs, r, done, _ = env.step([0,1])
        print(obs, r, done)
