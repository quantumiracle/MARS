from .base import MarkovGameSolver
import numpy as np
import itertools

class SelfPlay(MarkovGameSolver):
    def __init__(self, env, save_path, solve_episodes):
        super(SelfPlay, self).__init__(env, save_path, solve_episodes)

    def add_best_response(self, add_to_player, br_against_player, action_seq, condition_seq):
        new_policy = self.get_best_response_policy(br_against_player, np.array(action_seq), condition_seq)  # get best response against the max player, max_player_action_prob_sequence: [#policies, H]
        add_to_player['policy_set'].append(new_policy) # add new policy to policy set to form mixture

    def update_meta_strategy(self, max_player, min_player):
        # one-hot (only latest model has probability mass = 1) distribution
        max_policies = len(max_player['policy_set'])
        meta_strategy = np.zeros(max_policies)
        meta_strategy[-1] = 1
        max_player['meta_strategy'] = meta_strategy

        min_policies = len(min_player['policy_set'])
        meta_strategy = np.zeros(min_policies)
        meta_strategy[-1] = 1
        min_player['meta_strategy'] = meta_strategy



class QLearningSelfPlay(SelfPlay):
    def __init__(self, env, save_path, solve_episodes):
        super(QLearningSelfPlay, self).__init__(env, save_path, solve_episodes)
        """ (Sampling version) Fictitious selfplay with best response calculated by Q-learning
        """

    def solve(self, epsilon=0.2, gamma=0.99, br_itrs = 10, tau=0.0001):
        # ini_max_policy = np.array(self.num_trans*self.num_states*[1./self.num_actions*np.ones(self.num_actions)]) # shape: [num_transition, num_states, num_action]
        # ini_min_policy = np.array(self.num_trans*self.num_states*[1./self.num_actions*np.ones(self.num_actions)])

        ini_max_q = self.get_markov_q()
        ini_min_q = self.get_markov_q()

        max_player = {
            'side': 'max',
            'q_set': [ini_max_q],
            'meta_strategy': np.array([1.])
        }
        min_player = {
            'side': 'min',
            'q_set': [ini_min_q],
            'meta_strategy': np.array([1.])
        }
        fixed_side = 'max'
        exploitability_records = []

        for i in range(self.solve_episodes):
            obs = self.env.reset()
            done = False

            if i % br_itrs == 0:  # switch side
                if fixed_side == 'max':
                    if i != 0: # store new br table, and generate new meta strategy
                        min_player['q_set'].append(br_q)
                        num_qs = len(min_player['q_set'])
                        min_player['meta_strategy'] = 1./num_qs*np.ones(num_qs)
                    
                    # update side
                    fixed_side = 'min'

                else:
                    if i != 0: # store new br table, and generate new meta strategy
                        max_player['q_set'].append(br_q)
                        num_qs = len(max_player['q_set'])
                        max_player['meta_strategy'] = 1./num_qs*np.ones(num_qs)
                    
                    # update side
                    fixed_side = 'max'

                br_q = self.get_markov_q()  # init new best response side Q table

            # sample policy for the current episode
            if fixed_side == 'max':
                max_q_at_episode = self.sample_policy(max_player['q_set'], max_player['meta_strategy'])
            else:
                min_q_at_episode = self.sample_policy(min_player['q_set'], min_player['meta_strategy'])

            while not np.any(done):
                if fixed_side == 'max':
                    # fixed side uses greedy choice
                    max_action = self.get_greedy_action(max_q_at_episode[obs[0][0]])
                    # br side uses e-greedy choice
                    if np.random.random() > epsilon:
                        min_action = self.get_greedy_action(br_q[obs[0][0]])
                    else:
                        min_action = self.get_random_action(self.num_actions)
                else:
                    if np.random.random() > epsilon:
                        max_action = self.get_greedy_action(br_q[obs[0][0]])
                    else:
                        max_action = self.get_random_action(self.num_actions)

                    min_action = self.get_greedy_action(min_q_at_episode[obs[0][0]])

                action = [max_action, min_action]
                next_obs, r, done, _ = self.env.step(action)

                # update best response side
                if fixed_side  == 'max': # br is min
                    if done[1]: # there is no action and reward for terminal state, so cannot get q value for its next state
                        target = r[1]
                    else:
                        target = r[1] + gamma * np.max(br_q[next_obs[0][0]])
                    br_q[obs[0][0], min_action] = tau*target + (1-tau)*br_q[obs[0][0], min_action]

                else:
                    if done[0]: 
                        target = r[0]
                    else:
                        target = r[0] + gamma * np.max(br_q[next_obs[0][0]])
                    br_q[obs[0][0], max_action] = tau*target + (1-tau)*br_q[obs[0][0], max_action]

                obs = next_obs

            ##  get exploitability of the max player
            if i % 10 == 0:
                # first average q, then get greedy policy (the resulting policy is deterministic)
                # average_q = self.weighted_average_q_table(max_player['q_set'], max_player['meta_strategy'])
                # mix_policy = self.q_table_to_greedy_policy(average_q)

                # first get greedy policy, then average policies according to meta strategy
                greedy_policies = [self.q_table_to_greedy_policy(q) for q in max_player['q_set']]
                mix_policy = self.weighted_average_q_table(greedy_policies, max_player['meta_strategy'])
                exploitability = self.best_response_value_given_markov_policy(mix_policy)
                print(f'itr: {i}, exploitability: {exploitability}', )
                exploitability_records.append(exploitability)
                np.save(self.save_path, exploitability_records)