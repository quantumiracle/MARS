from .base import MarkovGameSolver
import numpy as np
from mars.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumCVXPYSolver
import itertools

class DoubleOralce(MarkovGameSolver):
    def __init__(self, env, save_path, solve_episodes):
        super(DoubleOralce, self).__init__(env, save_path, solve_episodes)
        self.evaluation_matrix = np.array([[0]])

    def add_best_response(self, add_to_player, br_against_player, action_seq, condition_seq):
        new_policy = self.get_best_response_policy(br_against_player, np.array(action_seq), condition_seq)  # get best response against the max player, max_player_action_prob_sequence: [#policies, H]
        add_to_player['policy_set'].append(new_policy) # add new policy to policy set to form mixture

        # update evaluation matrix
        eval_scores = []
        for p in br_against_player['policy_set']:
            score = self.policy_against_policy_value(max_policy=p, min_policy=new_policy)
            eval_scores.append(score)  # score always from the max player's view
        self.evaluation_matrix = self.update_matrix(self.evaluation_matrix, add_to_player['side'], eval_scores)

    def update_meta_strategy(self, max_player, min_player):
        # meta-Nash distribution
        if len(max_player['policy_set']) * len(min_player['policy_set']) >= 2:  # enough policies to get Nash
            (max_player['meta_strategy'], min_player['meta_strategy']), _ = NashEquilibriumCVXPYSolver(self.evaluation_matrix)
        else: # uniform for initialization only
            max_policies = len(max_player['policy_set'])
            max_player['meta_strategy'] = 1./max_policies*np.ones(max_policies)
            min_policies = len(min_player['policy_set'])
            min_player['meta_strategy'] = 1./min_policies*np.ones(min_policies)


class OracleDoubleOralce(DoubleOralce):
    def __init__(self, env, save_path, solve_episodes):
        super(OracleDoubleOralce, self).__init__(env, save_path, solve_episodes)
        
    def get_posterior_policy(self, policy_set, meta_prior, side):
        posterior_policy = self.create_expand_policy(zero_ini=True)  # zero initiate to sum later

        def get_all_traj_for_transition(num_transition, single_side=False):  # TESTED
            """
            step = 1:
            [
                [s0,a0,b0], [s0,a0,b1],...
            ]
            step = 2:
            [
                [s0,a0,b0,s1,a1,b1], [s1,a0,b0,s1,a1,b1],
            ]
            ...
            """
            if single_side: # lack of one action
                ranges = (num_transition-1)*[range(self.num_states), range(self.num_actions), range(self.num_actions)] + [range(self.num_states), range(self.num_actions)]
            else:
                ranges = num_transition*[range(self.num_states), range(self.num_actions), range(self.num_actions)]
            all_possible_trajs = list(itertools.product(*ranges))

            return all_possible_trajs

        
        def split_traj(traj, side):  # TESTED
            """
            s0, a0, b0, s1, a1, b1, ... sn-1, an-1, bn-1, sn ->
            side = max:
            [], [s0,a0], [s0,a0,b0,s1,a1], [s0,a0,b0,s1,a1,b1,s2,a2], ...
            side = min:
            [s0,b0], [s0,a0,b0,s1,b1], [s0,a0,b0,s1,a1,b1,s2,b2], ...
            """
            split_trajs = []
            i=2
            while i<=len(traj):  # max length same as original traj but lack of one action (due to the side choice)
                if side == 'max':
                    split_trajs.append(traj[:i])
                else:
                    split_trajs.append(np.concatenate([traj[:i-1], [traj[i]]]))
                i = i+3 # 3 due to (s,a,b)

            return split_trajs


        def get_likelihood(policy, side):
            likelihoods = self.create_expand_Q() # (s,a,b) rather than (s,a) or (s,b)
            for i in range(self.num_trans):
                # number of transitions is i+1
                # each transition is (s,a,b)
                # each trajectory of transitions is [(s1,a1,b1), (s2,a2,b2), ...] of length num_transition
                trajs = get_all_traj_for_transition(num_transition=i+1)   
                for traj in trajs:
                    # split trajectories to be a powerset, 
                    # and return only one-side action for the last transition,
                    # b.c. in likelihood there is only one-side probability
                    trajs_list = split_traj(traj, side) 
                    likelihood = 1.
                    for j, t in enumerate(trajs_list):  # this product corresponding to likelihood for h step (not h-1)
                        likelihood = likelihood*policy[j][tuple(t)]  # scalar
                    likelihoods[i][tuple(traj)] = likelihood  # needs to know this likelihood stored for transition i should be used for posterior transition i+1
            return likelihoods

        likelihoods = []
        for pi_i in policy_set:
            likelihoods.append(get_likelihood(pi_i, side))

        def get_denoms(likelihoods, rho):
            denoms = self.create_expand_Q(zero_ini=True)
            for j, rho_j in enumerate(rho):  # over policy in policy set
                for i in range(len(likelihoods[j])):  # over transition
                    trajs = get_all_traj_for_transition(i+1) 
                    for traj in trajs:
                        denoms[i][tuple(traj)] += rho_j*likelihoods[j][i][tuple(traj)]

            return denoms

        denoms = get_denoms(likelihoods, meta_prior)

        for pi_i, rho_i, likelihood_i, in zip(policy_set, meta_prior, likelihoods):  # loop over policy set

            for i, (p,) in enumerate(zip(pi_i)): # loop over transition
                taus = get_all_traj_for_transition(i+1, single_side=True)
                for tau in taus:
                    if i==0:
                        posterior_policy[i][tuple(tau)] += p[tuple(tau)]*rho_i # posterior = prior for the first transition
                    else:
                        tau_h_1 = tau[:-2]  # remove the last (s,a/b) for the likelihood and denominator since they take trajectory until (h-1)
                        # import pdb; pdb.set_trace()
                        posterior_policy[i][tuple(tau)] += p[tuple(tau)]*rho_i*likelihood_i[i-1][tuple(tau_h_1)]/denoms[i-1][tuple(tau_h_1)]

        return posterior_policy

    def add_best_response(self, add_to_player, br_against_player):
        new_policy = self.get_best_response_policy(br_against_player)  # get best response against the max player, max_player_action_prob_sequence: [#policies, H]
        add_to_player['policy_set'].append(new_policy) # add new policy to policy set to form mixture

        eval_scores = []
        for p in br_against_player['policy_set']:
            score = self.policy_against_policy_value(max_policy=p, min_policy=new_policy)
            eval_scores.append(score)  # score always from the max player's view
        self.evaluation_matrix = self.update_matrix(self.evaluation_matrix, add_to_player['side'], eval_scores)


    def solve(self, ):
        ini_max_policy = self.create_expand_policy()
        ini_min_policy = self.create_expand_policy()
        max_player = {
            'side': 'max',
            'policy_set': [ini_max_policy],
            'meta_strategy': np.array([1.])
        }
        min_player = {
            'side': 'min',
            'policy_set': [ini_min_policy],
            'meta_strategy': np.array([1.])
        }

        print('policy shape: ')
        for p in ini_max_policy:
            print(p.shape)

        exploitability_records = []

        for i in range(self.solve_episodes):

            # iterative update for each side
            if i % 2 == 0:  # udpate min player, best response against max player
                self.add_best_response(min_player, max_player)
            else: 
                self.add_best_response(max_player, min_player)

            self.update_meta_strategy(max_player, min_player)

            if i % 10 == 0:
                exploitability = self.get_best_response_value(max_player)  # best response of the max player
                print(f'itr: {i}, exploitability: {exploitability}', )
                exploitability_records.append(exploitability)
                np.save(self.save_path, exploitability_records)



class QLearningDoubleOralce(DoubleOralce):
    def __init__(self, env, save_path, solve_episodes):
        super(QLearningDoubleOralce, self).__init__(env, save_path, solve_episodes)
        """ (Sampling version) Double oracle with best response calculated by Q-learning
        """

    def update_meta_strategy(self, max_player, min_player):
        # meta-Nash distribution
        if len(max_player['q_set']) * len(min_player['q_set']) >= 2:  # enough policies to get Nash
            (max_player['meta_strategy'], min_player['meta_strategy']), _ = NashEquilibriumCVXPYSolver(self.evaluation_matrix)
        else: # uniform for initialization only
            max_qs = len(max_player['q_set'])
            max_player['meta_strategy'] = 1./max_qs*np.ones(max_qs)
            min_qs = len(min_player['q_set'])
            min_player['meta_strategy'] = 1./min_qs*np.ones(min_qs)

    def policy_against_policy_value(self, max_policy, min_policy):
        V = []
        Q = []
        max_policy = max_policy.reshape(self.num_trans, self.num_states, self.num_actions)
        min_policy = min_policy.reshape(self.num_trans, self.num_states, self.num_actions)

        for tm, rm, max_pm, min_pm in zip(self.trans_matrices[::-1], self.reward_matrices[::-1], max_policy[::-1], min_policy[::-1]): # inverse enumerate 
            if len(V) > 0:
                rm = np.array(rm)+np.array(V[-1])  # broadcast sum on rm's last dim, last one in Nash_v is for the next state
            q_values = np.einsum("ijk,ijk->ij", tm, rm)  # transition prob * reward for the last dimension in (state, action, next_state)
            q_values = q_values.reshape(-1, self.num_actions, self.num_actions) # action list to matrix
            Q.append(q_values)
            values = []
            for max_p, min_p, q in zip(max_pm, min_pm, q_values):
                value = np.min(max_p@q@min_p)
                values.append(value)  # each value is a Nash equilibrium value on one state
            V.append(values)  # (trans, state)
        V = V[::-1]  # (#trans, #states)
        Q = Q[::-1]

        avg_init_v = np.mean(V[0])  # average best response value of initial states; minus for making it positive
        return avg_init_v

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
                        # update evaluation matrix
                        eval_scores = []
                        for q in max_player['q_set']:
                            p=self.q_table_to_greedy_policy(q)
                            br_p=self.q_table_to_greedy_policy(br_q)
                            score = self.policy_against_policy_value(max_policy=p, min_policy=br_p)
                            eval_scores.append(score)  # score always from the max player's view
                        self.evaluation_matrix = self.update_matrix(self.evaluation_matrix, min_player['side'], eval_scores)

                    # udpate side
                    fixed_side = 'min'

                else:
                    if i != 0: # store new br table, and generate new meta strategy
                        max_player['q_set'].append(br_q)
                        # update evaluation matrix
                        eval_scores = []
                        for q in min_player['q_set']:
                            p=self.q_table_to_greedy_policy(q)
                            br_p=self.q_table_to_greedy_policy(br_q)
                            score = self.policy_against_policy_value(max_policy=br_p, min_policy=p)
                            eval_scores.append(score)  # score always from the max player's view
                        self.evaluation_matrix = self.update_matrix(self.evaluation_matrix, max_player['side'], eval_scores)

                    # udpate side
                    fixed_side = 'max'

                self.update_meta_strategy(max_player, min_player)
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