from .base import MarkovGameSolver
import numpy as np
import itertools

class FictitiousSelfPlay(MarkovGameSolver):
    def __init__(self, env, save_path, solve_episodes):
        super(FictitiousSelfPlay, self).__init__(env, save_path, solve_episodes)

    def add_best_response(self, add_to_player, br_against_player, action_seq, condition_seq):
        new_policy = self.get_best_response_policy(br_against_player, np.array(action_seq), condition_seq)  # get best response against the max player, max_player_action_prob_sequence: [#policies, H]
        add_to_player['policy_set'].append(new_policy) # add new policy to policy set to form mixture

    def update_meta_strategy(self, max_player, min_player):
        # unifor distribution
        max_policies = len(max_player['policy_set'])
        max_player['meta_strategy'] = 1./max_policies*np.ones(max_policies)
        min_policies = len(min_player['policy_set'])
        min_player['meta_strategy'] = 1./min_policies*np.ones(min_policies)

class OracleFictitiousSelfPlay(FictitiousSelfPlay):
    def __init__(self, env, save_path, solve_episodes):
        super(FictitiousSelfPlay, self).__init__(env, save_path, solve_episodes)
        

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

