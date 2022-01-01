from blackhc import mdp
import numpy as np
from .mdp_wrapper import MDPWrapper


class Attack():
    def __init__(self, wrapped=False):
        self.action_set = ['R', 'L', 'B', 'A']
        self.num_actions = len(self.action_set)
        self.num_agents = 2
        self.max_transition = 100  # maximum number of transition steps
        self.spec = mdp.MDPSpec()
        self.env = self._construct_game()
        setattr(self.env, 'num_actions', self.num_actions)
        setattr(self.env, 'action_map', self.action_map)
        setattr(self.env, 'max_transition', self.max_transition)
        if wrapped:
            self.wrap2player()

    def _construct_game(self, ):
        s0 = self.spec.state('s0')
        s1 = self.spec.state('s1')
        s2 = self.spec.state('s2')

        s3 = self.spec.state('end', terminal_state=True)
        actions = []
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                actions.append(self.spec.action(self.action_set[j]+self.action_set[i]))

        #RR
        self.spec.transition(s0, actions[0], mdp.NextState(s0))
        self.spec.transition(s1, actions[0], mdp.NextState(s1))
        self.spec.transition(s2, actions[0], mdp.NextState(s2))
        #LR
        self.spec.transition(s0, actions[1], mdp.NextState(s2))
        self.spec.transition(s1, actions[1], mdp.NextState(s0))
        self.spec.transition(s2, actions[1], mdp.NextState(s1))
        #BR
        self.spec.transition(s0, actions[2], mdp.NextState(s1))
        self.spec.transition(s1, actions[2], mdp.NextState(s2))
        self.spec.transition(s2, actions[2], mdp.NextState(s0))
        self.spec.transition(s0, actions[2], mdp.Reward(-0.5))
        self.spec.transition(s1, actions[2], mdp.Reward(-0.5))
        self.spec.transition(s2, actions[2], mdp.Reward(-.5))
        #AR
        self.spec.transition(s0, actions[3], mdp.NextState(s1))
        self.spec.transition(s1, actions[3], mdp.NextState(s3))
        self.spec.transition(s2, actions[3], mdp.NextState(s0))
        self.spec.transition(s0, actions[3], mdp.Reward(-1.))
        self.spec.transition(s1, actions[3], mdp.Reward(-1.))
        self.spec.transition(s2, actions[3], mdp.Reward(1.))
        #RL
        self.spec.transition(s0, actions[4], mdp.NextState(s1))
        self.spec.transition(s1, actions[4], mdp.NextState(s2))
        self.spec.transition(s2, actions[4], mdp.NextState(s0))
        #LL
        self.spec.transition(s0, actions[5], mdp.NextState(s0))
        self.spec.transition(s1, actions[5], mdp.NextState(s1))
        self.spec.transition(s2, actions[5], mdp.NextState(s2))
        #BL
        self.spec.transition(s0, actions[6], mdp.NextState(s2))
        self.spec.transition(s1, actions[6], mdp.NextState(s0))
        self.spec.transition(s2, actions[6], mdp.NextState(s1))
        self.spec.transition(s0, actions[6], mdp.Reward(-0.5))
        self.spec.transition(s1, actions[6], mdp.Reward(-0.5))
        self.spec.transition(s2, actions[6], mdp.Reward(-0.5))
        #AL
        self.spec.transition(s0, actions[7], mdp.NextState(s3))
        self.spec.transition(s1, actions[7], mdp.NextState(s0))
        self.spec.transition(s2, actions[7], mdp.NextState(s1))
        self.spec.transition(s0, actions[7], mdp.Reward(-1.))
        self.spec.transition(s1, actions[7], mdp.Reward(1.))
        self.spec.transition(s2, actions[7], mdp.Reward(-1.))
        #RB
        self.spec.transition(s0, actions[8], mdp.NextState(s2))
        self.spec.transition(s1, actions[8], mdp.NextState(s0))
        self.spec.transition(s2, actions[8], mdp.NextState(s1))
        self.spec.transition(s0, actions[8], mdp.Reward(0.5))
        self.spec.transition(s1, actions[8], mdp.Reward(0.5))
        self.spec.transition(s2, actions[8], mdp.Reward(0.5))
        #LB
        self.spec.transition(s0, actions[9], mdp.NextState(s1))
        self.spec.transition(s1, actions[9], mdp.NextState(s2))
        self.spec.transition(s2, actions[9], mdp.NextState(s0))
        self.spec.transition(s0, actions[9], mdp.Reward(0.5))
        self.spec.transition(s1, actions[9], mdp.Reward(0.5))
        self.spec.transition(s2, actions[9], mdp.Reward(0.5))
        #BB
        self.spec.transition(s0, actions[10], mdp.NextState(s0))
        self.spec.transition(s1, actions[10], mdp.NextState(s1))
        self.spec.transition(s2, actions[10], mdp.NextState(s2))
        #AB
        self.spec.transition(s0, actions[11], mdp.NextState(s0))
        self.spec.transition(s1, actions[11], mdp.NextState(s1))
        self.spec.transition(s2, actions[11], mdp.NextState(s2))
        self.spec.transition(s0, actions[11], mdp.Reward(0.5))
        #RA
        self.spec.transition(s0, actions[12], mdp.NextState(s3))
        self.spec.transition(s1, actions[12], mdp.NextState(s0))
        self.spec.transition(s2, actions[12], mdp.NextState(s1))
        self.spec.transition(s0, actions[12], mdp.Reward(1.))
        self.spec.transition(s1, actions[12], mdp.Reward(-1.))
        self.spec.transition(s2, actions[12], mdp.Reward(1.))
        #LA
        self.spec.transition(s0, actions[13], mdp.NextState(s1))
        self.spec.transition(s1, actions[13], mdp.NextState(s3))
        self.spec.transition(s2, actions[13], mdp.NextState(s1))
        self.spec.transition(s0, actions[13], mdp.Reward(1.))
        self.spec.transition(s1, actions[13], mdp.Reward(1.))
        self.spec.transition(s2, actions[13], mdp.Reward(-1.))
        #BA
        self.spec.transition(s0, actions[14], mdp.NextState(s0))
        self.spec.transition(s1, actions[14], mdp.NextState(s1))
        self.spec.transition(s2, actions[14], mdp.NextState(s2))
        self.spec.transition(s0, actions[14], mdp.Reward(-0.5))
        #AA
        self.spec.transition(s0, actions[15], mdp.NextState(s3))
        self.spec.transition(s1, actions[15], mdp.NextState(s3))
        self.spec.transition(s2, actions[15], mdp.NextState(s3))

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
            p2 0R  1L  2B  3A
        p1
        0R      0   4   8  12
        1L      1   5   9  13
        2B      2   6  10  14
        3A      3   7  11  15
        """
        a = action[1]*self.num_actions+action[0]

        return a


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
    env = Attack(wrapped=True).env
    print(env.observation_space, env.action_space)
    # env.render()
    obs = env.reset()
    print(obs)
    done = False
    while not np.any(done):
        obs, r, done, _ = env.step([0,1])
        print(obs, r, done)
