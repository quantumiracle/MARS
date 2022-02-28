import numpy as np
import copy

class MarkovGameSolver:
    def __init__(self, env, save_path, solve_episodes):
        self.num_states = env.env.num_states
        self.num_trans = env.env.max_transition
        self.num_actions = env.env.num_actions
        self.trans_matrices = env.env.trans_prob_matrices
        self.reward_matrices = env.env.reward_matrices
        self.env = env
        self.save_path = save_path
        self.solve_episodes = solve_episodes


    def sample_from_categorical(self, dist):
        """
        sample once from a categorical distribution, return the entry index.
        dist: should be a list or array of probabilities for a categorical distribution
        """
        sample_id = np.argmax(np.random.multinomial(1, dist))
        sample_prob = dist[sample_id]
        return sample_id, sample_prob

    def unified_state(self, s):
        unified_s = s[0]%self.num_states
        return unified_s

    def create_expand_policy(self, zero_ini=False):
        """
        Returns:
        [   [state_dim, action_dim],
            [state_dim, action_dim, action_dim, state_dim, action_dim],
            [state_dim, action_dim, action_dim, state_dim, action_dim, action_dim, state_dim, action_dim],
            ...
        ]
        """
        policies = []
        intial_dim = [self.num_states]
        for i in range(self.num_trans):
            if zero_ini:
                policy =  (1./self.num_actions) * np.zeros((*intial_dim, self.num_actions))
            else:
                policy =  (1./self.num_actions) * np.ones((*intial_dim, self.num_actions))
            # incremental shape
            intial_dim.extend([self.num_actions, self.num_actions, self.num_states])
            policies.append(policy)

        return policies

    def create_expand_value(self,):
        """
        Returns:
        [   [state_dim],
            [state_dim, action_dim, action_dim, state_dim],
            [state_dim, action_dim, action_dim, state_dim, action_dim, action_dim, state_dim],
            ...
        ]    
        """
        values = []
        intial_dim = [self.num_states]
        for i in range(self.num_trans):
            value =  (1./self.num_actions) * np.ones(intial_dim)
            # incremental shape
            intial_dim.extend([self.num_actions, self.num_actions, self.num_states])
            values.append(value)

        return values

    def create_expand_Q(self, zero_ini=False):
        """
        Returns:
        [   [state_dim, action_dim, action_dim],
            [state_dim, action_dim, action_dim, state_dim, action_dim, action_dim],
            [state_dim, action_dim, action_dim, state_dim, action_dim, action_dim, state_dim, action_dim, action_dim],
            ...
        ]
        """
        values = []
        intial_dim = [self.num_states, self.num_actions, self.num_actions]
        for i in range(self.num_trans):
            if zero_ini:
                value =  (1./self.num_actions) * np.zeros(intial_dim)
            else:
                value =  (1./self.num_actions) * np.ones(intial_dim)
            # incremental shape
            intial_dim.extend([self.num_states, self.num_actions, self.num_actions])
            values.append(value)

        return values

    def get_posterior_policy(self, policy_set, meta_prior, side, likelihood, con_seq, only_update_observed=True):
        posterior_policy = self.create_expand_policy(zero_ini=True)  # zero initiate to sum later
        denom = meta_prior @ likelihood  # [#policy] * [#policy, H] -> [H]
        if only_update_observed:  # only update observed sequences, others use prior
            for pi_i, rho_i in zip(policy_set, meta_prior):  # loop over policy
                for i, (p,) in enumerate(zip(pi_i)): # loop over transition 
                    posterior_policy[i] += np.array(p*rho_i)  # all entries using prior

            for pi_i in policy_set:  # loop over policy
                for i, s in enumerate(con_seq): # loop over transition 
                    posterior_policy[i][tuple(s)] = 0  # clear the observed entries

            for pi_i, rho_i, likelihood_per_policy in zip(policy_set, meta_prior, likelihood):  # loop over policy
                for i, (p, d, l, s) in enumerate(zip(pi_i, denom, likelihood_per_policy, con_seq)): # loop over transition 
                    posterior_policy[i][tuple(s)] += np.array(p[tuple(s)]*rho_i*l/d)  # only update observed

        else:  # update all entries
            for pi_i, rho_i, likelihood_per_policy in zip(policy_set, meta_prior, likelihood):  # loop over policy
                for i, (p, d, l) in enumerate(zip(pi_i, denom, likelihood_per_policy)): # loop over transition
                    posterior_policy[i] += np.array(p*rho_i*l/d)  # sum over policies in mixture

        return posterior_policy

    def get_prior_policy(self, policy_set, meta_prior,):
        prior_policy = self.create_expand_policy(zero_ini=True)  # zero initiate to sum later
        for pi_i, rho_i in zip(policy_set, meta_prior):  # loop over policy
            for i, (p,) in enumerate(zip(pi_i)): # loop over transition 
                prior_policy[i] += np.array(p*rho_i)  # all entries using prior
        return prior_policy

    def broadcast_shape(self, input, output_shape):
        """ broadcast input to have the same shape as output_to_be """
        len_input_shape = len(np.array(input).shape)
        incrs_shape = output_shape[:-len_input_shape]+len_input_shape*(1,)
        output = np.tile(input, incrs_shape)
        return output

    def get_best_response_policy(self, player, *args, **kwargs):
        given_policy = self.get_posterior_policy(player['policy_set'], player['meta_strategy'], player['side'], *args, **kwargs)
        br_policy = self.create_expand_policy()  # best response policy
        br_v = self.create_expand_value()
        br_q = self.create_expand_Q()

        for i in range(self.num_trans-1, -1, -1):  # inverse indexing
            tm = self.trans_matrices[i]
            rm = self.reward_matrices[i]

            rm_ = np.array(rm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)
            tm_ = np.array(tm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)

            expand_rm = self.broadcast_shape(rm_, br_q[i].shape+(self.num_states,))  # broadcast the shape of reward matrix to be expand Q shape plus additional state dimension
            if i == self.num_trans-1:
                expand_tm = self.broadcast_shape(tm_, expand_rm.shape)
                br_q[i] =  np.sum(expand_rm*expand_tm, axis=-1)
            else:
                v = br_v[i+1]
                v_before_trans = expand_rm + v  # expand_rm and v are same shape
                expand_tm = self.broadcast_shape(tm_, v_before_trans.shape)  # get the same shape as v_before_trans
                br_q[i] = np.sum(v_before_trans*expand_tm, axis=-1)  # only sum over last dim: state

            if player['side'] == 'max':
                mu_dot_q = np.einsum('...i, ...ij->...j', given_policy[i], br_q[i])
                arg_id = np.argmin(mu_dot_q, axis=-1)  # min player takes minimum as best response against max player's policy
                br_v[i] = np.min(mu_dot_q, axis=-1)
            else:
                q_dot_nu = np.einsum('...ij, ...j->...i', br_q[i], given_policy[i])
                arg_id = np.argmax(q_dot_nu, axis=-1)   # vice versa    
                br_v[i] = np.max(q_dot_nu, axis=-1)     
            
            br_policy[i] = (np.arange(self.num_actions) == arg_id[...,None]).astype(int)  # from extreme (min/max) idx to one-hot simplex
            # print(br_policy[i].shape, br_q[i].shape)

        return br_policy

    def best_response_value(self, policy, side='max'):
        """
        Formulas for calculating best response values:
        1. Nash strategies: (\pi_a^*, \pi_b^*) = \min \max Q(s,a,b), 
            where Q(s,a,b) = r(s,a,b) + \gamma \min \max Q(s',a',b') (this is the definition of Nash Q-value);
        2. Best response (of max player) value: Br V(s) = \min_b \pi(s,a) Br Q(s,a,b)  (Br Q is the oracle best response Q value)
        """
        br_v = self.create_expand_value()
        br_q = self.create_expand_Q()

        for i in range(self.num_trans-1, -1, -1):  # inverse indexing
            tm = self.trans_matrices[i]
            rm = self.reward_matrices[i]

            rm_ = np.array(rm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)
            tm_ = np.array(tm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)
            expand_rm = self.broadcast_shape(rm_, br_q[i].shape+(self.num_states,))
            if i == self.num_trans-1:
                expand_tm = self.broadcast_shape(tm_, expand_rm.shape)
                br_q[i] =  np.sum(expand_rm*expand_tm, axis=-1)
            else:
                v = br_v[i+1]
                v_before_trans = expand_rm + v  # expand_rm and v are same shape
                expand_tm = self.broadcast_shape(tm_, v_before_trans.shape)  # get the same shape as v_before_trans
                br_q[i] = np.sum(v_before_trans*expand_tm, axis=-1)  # only sum over last dim: state

            if side == 'max':
                mu_dot_q = np.einsum('...i, ...ij->...j', policy[i], br_q[i])
                br_v[i] = np.min(mu_dot_q, axis=-1)
            else:
                q_dot_nu = np.einsum('...ij, ...j->...i', br_q[i], policy[i])
                br_v[i] = np.max(q_dot_nu, axis=-1)         

        avg_init_br_v = -np.mean(br_v[0])  # average best response value of initial states; minus for making it positive
        return avg_init_br_v

    def get_best_response_value(self, player):
        # should still get the posterior policy as the mixture policy to exploit!
        prior_mixture_policy = self.get_prior_policy(player['policy_set'], player['meta_strategy'])
        exploitability = self.best_response_value(prior_mixture_policy, player['side'])

        return exploitability

    def policy_against_policy_value(self, max_policy, min_policy):
        """
        Formulas for calculating best response values:
        1. Nash strategies: (\pi_a^*, \pi_b^*) = \min \max Q(s,a,b), 
            where Q(s,a,b) = r(s,a,b) + \gamma \min \max Q(s',a',b') (this is the definition of Nash Q-value);
        2. Best response (of max player) value: Br V(s) = \min_b \pi(s,a) Br Q(s,a,b)  (Br Q is the oracle best response Q value)
        """
        eval_v = self.create_expand_value()
        eval_q = self.create_expand_Q()

        for i in range(self.num_trans-1, -1, -1):  # inverse indexing
            tm = self.trans_matrices[i]
            rm = self.reward_matrices[i]

            rm_ = np.array(rm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)
            tm_ = np.array(tm).reshape(self.num_states, self.num_actions, self.num_actions, self.num_states)
            expand_rm = self.broadcast_shape(rm_, eval_q[i].shape+(self.num_states,))
            if i == self.num_trans-1:
                expand_tm = self.broadcast_shape(tm_, expand_rm.shape)
                eval_q[i] =  np.sum(expand_rm*expand_tm, axis=-1)
            else:
                v = eval_v[i+1]
                v_before_trans = expand_rm + v  # expand_rm and v are same shape
                expand_tm = self.broadcast_shape(tm_, v_before_trans.shape)  # get the same shape as v_before_trans
                eval_q[i] = np.sum(v_before_trans*expand_tm, axis=-1)  # only sum over last dim: state

            mu_dot_q = np.einsum('...i, ...ij->...j', max_policy[i], eval_q[i])
            mu_dot_q_dot_nu = np.einsum('...kj, ...kj->...k', mu_dot_q, min_policy[i])
            eval_v[i] = mu_dot_q_dot_nu    

        avg_init_eval_v = np.mean(eval_v[0])  # average best response value of initial states; minus for making it positive
        return avg_init_eval_v

    def update_matrix(self, matrix, side, row):
        """
        Adding a new policy to the league will add a row or column to the present evaluation matrix, after it has
        been evaluated against all policies in the opponent's league. Whether adding a row or column depends on whether
        it is the row player (idx=0) or column player (idx=1). 
        For example, if current evaluation matrix is:
        2  1
        -1 3
        After adding a new policy for row player with evaluated average episode reward (a,b) against current two policies in 
        the league of the column player, it gives: 
        it gives:
        2   1
        -1  3
        a   b    
        """ 
        if side == 'max': # for row player
            if matrix.shape[0] == 0: 
                matrix = np.array([row])
            else:  # add row
                matrix = np.vstack([matrix, row])
        else: # for column player
            if matrix.shape[0] == 0:  # the first update
                matrix = np.array([row]) 
            else:  # add column
                matrix=np.hstack([matrix, np.array([row]).T])

        return matrix

    def sample_policy(self, policy_set, dist):
        policy_id, _ = self.sample_from_categorical(dist)
        policy = policy_set[policy_id]   
        return policy 

    def solve(self, ):
        self.evaluation_matrix = np.array([[0]])
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
            s = self.env.reset()
            done = False
            max_player_observed_sequence = []
            min_player_observed_sequence = []
            max_player_conditional_sequence = []
            min_player_conditional_sequence = []
            max_player_action_prob_sequence = [[] for _ in max_player['policy_set']]
            min_player_action_prob_sequence = [[] for _ in min_player['policy_set']]
            step = 0

            # sample policy from meta strategy for the current episode
            max_policy = self.sample_policy(max_player['policy_set'], max_player['meta_strategy'])
            min_policy = self.sample_policy(min_player['policy_set'], min_player['meta_strategy'])

            while not np.any(done):    
                # get observed sequence
                max_player_observed_sequence.extend([self.unified_state(s[0])])
                min_player_observed_sequence.extend([self.unified_state(s[1])])

                # get action distribution given specific observed history
                max_policy_to_choose = max_policy[step][tuple(max_player_observed_sequence)]  # a=np.ones([1,2]), b=(0,0), a[b] -> 1.
                min_policy_to_choose = min_policy[step][tuple(min_player_observed_sequence)]

                # choose action
                max_action, _ = self.sample_from_categorical(max_policy_to_choose)
                min_action, _ = self.sample_from_categorical(min_policy_to_choose)

                # roullout info for mixture policy
                if i % 2 == 0:  # update min player side
                    for p_id, p in enumerate(max_player['policy_set']):  # get trajectory probabilities for each policy in policy set
                        max_p = p[step][tuple(max_player_observed_sequence)]
                        # _, max_a_prob = sample_from_categorical(max_p) # this is wrong
                        max_a_prob = max_p[max_action]  # get the probability of real action (rollout) under each policy in policy set
                        if step == 0:
                            max_player_action_prob_sequence[p_id].append(max_a_prob)
                        else:
                            max_player_action_prob_sequence[p_id].append(max_player_action_prob_sequence[p_id][-1]*max_a_prob)  # [p1, p1p2, p1p2p3, ...]
                    max_player_conditional_sequence.append(copy.deepcopy(max_player_observed_sequence))

                else:
                    for p_id, p in enumerate(min_player['policy_set']):
                        min_p = p[step][tuple(min_player_observed_sequence)]
                        # _, min_a_prob = sample_from_categorical(min_p)  # this is wrong
                        min_a_prob = min_p[min_action]  # get the probability of real action (rollout) under each policy in policy set
                        if step == 0:
                            min_player_action_prob_sequence[p_id].append(min_a_prob)
                        else:
                            min_player_action_prob_sequence[p_id].append(min_player_action_prob_sequence[p_id][-1]*min_a_prob)  # [p1, p1p2, p1p2p3, ...]
                    min_player_conditional_sequence.append(copy.deepcopy(min_player_observed_sequence))

                action = [max_action, min_action]
                max_player_observed_sequence.extend(action)
                min_player_observed_sequence.extend(action)

                s_, r, done, _  = self.env.step(action)  
                s = s_

                step += 1

            # iterative update for each side
            if i % 2 == 0:  # udpate min player, best response against max player
                self.add_best_response(min_player, max_player, max_player_action_prob_sequence, max_player_conditional_sequence)
            else: 
                self.add_best_response(max_player, min_player, min_player_action_prob_sequence, min_player_conditional_sequence)

            self.update_meta_strategy(max_player, min_player)

            if i % 10 == 0:
                exploitability = self.get_best_response_value(max_player)  # best response of the max player
                print(f'itr: {i}, exploitability: {exploitability}', )
                exploitability_records.append(exploitability)
                np.save(self.save_path, exploitability_records)
