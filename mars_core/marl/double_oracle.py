import os
import numpy as np
from datetime import datetime
import sys
sys.path.append("..")
from rl.algorithm.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumCVXPYSolver

class NXDOMetaLearner():
    """
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Neural Extensive-Form Double Oracle (NXDO) 

    Ref: https://arxiv.org/pdf/2103.06426.pdf
    """
    def __init__(self, logger, args, save_checkpoint=True):
        super(NXDOMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.last_update_epi= 0
        self.saved_checkpoints = []
        self.evaluation_matrix = np.array([[0]])  # the evaluated utility matrix (N*N) of policy league with N policies

        logger.add_extr_log('matrix_equilibrium')

    def step(self, model, logger, env, args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        """
        # score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        # score_avg_window = 10 # the length of window for averaging the score values
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
             - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])

        # this is an indicator that best response policy is found
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and logger.current_episode - self.last_update_epi > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(logger.current_episode)) # save all checkpoints
                self.saved_checkpoints.append(str(logger.current_episode))

            logger.additional_logs.append(f'Score delta: {score_delta}, udpate the opponent.')
            self.last_update_epi = logger.current_episode

            model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

            ### update the opponent with epsilon meta Nash policy
            # evaluate the N*N utility matrix, N is the number of currently saved models
            added_row = []
            if len(self.saved_checkpoints) == 1:
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+self.saved_checkpoints[-1])
            elif len(self.saved_checkpoints) > 1:
                eval_agents = model.Kwargs['eval_models']  # these agents are evaluation models (for evaluation purpose only)
                env = model.Kwargs['eval_env']
                eval_agents[0].load_model(self.model_path+self.saved_checkpoints[-1]) # current model
                for previous_model_id in self.saved_checkpoints[:-1]:
                    eval_agents[1].load_model(self.model_path+previous_model_id)
                    added_row.append(self.evaluate(env, eval_agents, args)[0])
                # print('row: ', added_row)
                self.update_matrix(np.array(added_row)) # add new evaluation results to matrix
                # print('matrix: ', self.evaluation_matrix)
                # rollout with NFSP to learn meta strategy or directly calculate the Nash from the matrix
                self.nash_meta_strategy, _ = NashEquilibriumECOSSolver(self.evaluation_matrix)
                # the solver returns the equilibrium strategies for both players, just take one; it should be the same due to the symmetric poicy space
                self.nash_meta_strategy = self.nash_meta_strategy[0]
                # print('nash: ', self.nash_meta_strategy)
                logger.extr_logs.append(f'Current episode: {logger.current_episode}, utitlity matrix: {self.evaluation_matrix}, Nash stratey: {self.nash_meta_strategy}')

        # sample from Nash meta policy in a episode-wise manner
        if len(self.saved_checkpoints) > 1:
            sample_hist = np.random.multinomial(1, self.nash_meta_strategy)
            policy_idx = np.squeeze(np.where(sample_hist>0))
            # print('points: ', self.saved_checkpoints, policy_idx)
            model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+self.saved_checkpoints[policy_idx])

    def update_matrix(self, row):
        """
        Adding a new policy to the league will add a row to the present evaluation matrix,
        and also a column as the inverse of row value together with a 0. 
        For example, if current evaluation matrix is:
        0  1
        -1 0
        After adding a new policy with evaluated average episode reward (a,b) against current two policies, 
        it gives:
        0   1 -a
        -1  0 -b
        a   b  0      
        """ 
        self.evaluation_matrix = np.vstack((self.evaluation_matrix, row))
        col = np.concatenate((-row, [0]))
        self.evaluation_matrix = np.hstack((self.evaluation_matrix, col.reshape(len(col), 1)))
        return self.evaluation_matrix

    def evaluate(self, env, agents, args, eval_episodes=3):
        agent_dim = len(agents)
        avg_epi_rewards = np.zeros(agent_dim)
        for epi in range(eval_episodes):
            obs = env.reset()
            rewards = np.zeros(agent_dim)
            overall_steps = 0

            for step in range(args.max_steps_per_episode):        
                overall_steps += 1
                obs_to_store = obs.swapaxes(
                    0, 1
                ) if args.num_envs > 1 else obs  # transform from (envs, agents, dim) to (agents, envs, dim)
                actions = []
                for state, agent in zip(obs_to_store, agents):
                    action = agent.choose_action(state, Greedy=True)
                    actions.append(action)

                obs_, reward, done, info = env.step(actions)
                obs = obs_
                rewards += np.array(reward)

                if np.any(
                done
                    ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                    break

            avg_epi_rewards += rewards

        avg_epi_rewards = avg_epi_rewards/eval_episodes

        return avg_epi_rewards


class NXDO2SideMetaLearner(NXDOMetaLearner):
    """
    This is a two-side version, which means the
    agents on both sides of the game will maintain 
    a policy sets for each of them. The update of 
    two sides follows an iterative manner.

    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Neural Extensive-Form Double Oracle (NXDO) 

    Ref: https://arxiv.org/pdf/2103.06426.pdf
    """
    def __init__(self, logger, args, save_checkpoint=True):
        # super(NXDO2SideMetaLearner, self).__init__(logger, args, save_checkpoint)
        self.model_path = logger.model_dir

        # get names
        self.current_learnable_model_idx = int(args.marl_spec['trainable_agent_idx'])
        self.current_fixed_opponent_idx = int(args.marl_spec['opponent_idx'])

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.last_update_epi= 0
        self.saved_checkpoints = [[] for _ in range(2)] # for both player
        self.evaluation_matrix = np.array([])  # the evaluated utility matrix (N*N) of policy league with N policies
        self.nash_meta_strategy = None
        logger.add_extr_log('matrix_equilibrium')

    def _switch_charac(self, model):
        """ Iteratively calculate the meta-Nash equilibrium and learn the best response, so switch the characters after each update."""
        # change learnable and not learnable to achieve iterative learning
        idx = self.current_learnable_model_idx
        self.current_learnable_model_idx = self.current_fixed_opponent_idx 
        self.current_fixed_opponent_idx = idx
        self._change_learnable_list(model)

    def _change_learnable_list(self, model):
        """ Change the learnable list according to current learnable/fixed index. """
        if not self.current_fixed_opponent_idx in model.not_learnable_list:
            model.not_learnable_list.append(self.current_fixed_opponent_idx)
        if self.current_learnable_model_idx in model.not_learnable_list:
            model.not_learnable_list.remove(self.current_learnable_model_idx)

    def step(self, model, logger, env, args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        # score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        # score_avg_window = 10 # use the same average window as logging for score delta
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[logger.keys[self.current_learnable_model_idx]][-score_avg_window:])\
             - np.mean(logger.epi_rewards[logger.keys[self.current_fixed_opponent_idx]][-score_avg_window:])

        # this is an indicator that best response policy is found
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and logger.current_episode - self.last_update_epi > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                save_path = self.model_path+str(logger.current_episode)+'_'+str(self.current_learnable_model_idx)
                model.agents[self.current_learnable_model_idx].save_model(save_path) # save all checkpoints
                self.saved_checkpoints[self.current_learnable_model_idx].append(str(logger.current_episode))

            logger.additional_logs.append(f'Score delta: {score_delta}, save the model to {save_path}.')
            self.last_update_epi = logger.current_episode

            ### update the opponent with epsilon meta Nash policy
            # evaluate the N*N utility matrix, N is the number of currently saved models
            added_row = []
            if len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 1:
                eval_agents = model.Kwargs['eval_models']  # these agents are evaluation models (for evaluation purpose only)
                env = model.Kwargs['eval_env']
                eval_agents[0].load_model(self.model_path+self.saved_checkpoints[self.current_learnable_model_idx][-1]+'_'+str(self.current_learnable_model_idx)) # load current model
                for opponent_model_id in self.saved_checkpoints[self.current_fixed_opponent_idx]:
                    eval_agents[1].load_model(self.model_path+opponent_model_id+'_'+str(self.current_fixed_opponent_idx))
                    added_row.append(self.evaluate(env, eval_agents, args)[0])  # reward for current learnable model
                # print('row: ', added_row, self.current_learnable_model_idx)
                self.update_matrix(self.current_learnable_model_idx, np.array(added_row)) # add new evaluation results to matrix
                # print('matrix: ', self.evaluation_matrix)
                if len(self.saved_checkpoints[self.current_fixed_opponent_idx])*len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 4: # no need for NE when (1,1), (1,2)
                    # rollout with NFSP to learn meta strategy or directly calculate the Nash from the matrix
                    # self.nash_meta_strategy = NashEquilibriumECOSSolver(self.evaluation_matrix) # present implementation cannot solve non-square matrix
                    self.nash_meta_strategy = NashEquilibriumCVXPYSolver(self.evaluation_matrix) # cvxpy can solve non-square matrix, just a bit slower, but nxdo doesn't solve Nash often
                    # the solver returns the equilibrium strategies for both players, just take one; it should be the same due to the symmetric poicy space
                    self.nash_meta_strategy = self.nash_meta_strategy[self.current_learnable_model_idx]
                    # print('nash: ', self.nash_meta_strategy)
                    logger.extr_logs.append(f'Current episode: {logger.current_episode}, utitlity matrix: {self.evaluation_matrix}, Nash stratey: {self.nash_meta_strategy}')

            self._switch_charac(model)
            model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # sample from Nash meta policy in a episode-wise manner
        if len(self.saved_checkpoints[self.current_fixed_opponent_idx])*len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 4 and self.nash_meta_strategy is not None:
            sample_hist = np.random.multinomial(1, self.nash_meta_strategy)  # meta nash policy is a distribution over the policy set, sample one policy from it according to meta nash for each episode
            policy_idx = np.squeeze(np.where(sample_hist>0))
            # print('points: ', self.saved_checkpoints, policy_idx)
            model.agents[self.current_fixed_opponent_idx].load_model(self.model_path+self.saved_checkpoints[self.current_fixed_opponent_idx][policy_idx]+'_'+str(self.current_fixed_opponent_idx))

    def update_matrix(self, idx, row):
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
        if idx == 0: # for row player
            if self.evaluation_matrix.shape[0] == 0: 
                self.evaluation_matrix = np.array([row])
            else:  # add row
                self.evaluation_matrix = np.vstack([self.evaluation_matrix, row])
        elif idx == 1: # for column player
            if self.evaluation_matrix.shape[0] == 0:  # the first update
                self.evaluation_matrix = -np.array([row])  # the minus is due to evaluation matrix is always from the first player's perspective, but not necessarily the current_learnable_model
            else:  # add column
                self.evaluation_matrix=np.hstack([self.evaluation_matrix, -np.array([row]).T]) # same reason for minus

        return self.evaluation_matrix
