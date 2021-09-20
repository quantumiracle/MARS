import os
import numpy as np
from datetime import datetime
import sys
sys.path.append("..")
from rl.algorithm.equilibrium_solver import NashEquilibriumECOSSolver

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

    def step(self, model, logger, env, args, min_update_interval = 20):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        # score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        score_avg_window = 10 # use the same average window as logging for score delta

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

            ### update the opponent with epsilon meta Nash policy
            # evaluate the N*N utility matrix, N is the number of currently saved models
            added_row = []
            if len(self.saved_checkpoints) == 1:
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+self.saved_checkpoints[-1])
            elif len(self.saved_checkpoints) > 1:
                agents = model.Kwargs['eval_models']
                env = model.Kwargs['eval_env']
                agents[0].load_model(self.model_path+self.saved_checkpoints[-1]) # current model
                for previous_model_id in self.saved_checkpoints[:-1]:
                    agents[1].load_model(self.model_path+previous_model_id)
                    added_row.append(self.evaluate(env, agents, args)[0])
                # print('row: ', added_row)
                self.update_matrix(np.array(added_row)) # add new evaluation results to matrix
                # print('matrix: ', self.evaluation_matrix)
                # rollout with NFSP to learn meta strategy or directly calculate the Nash from the matrix
                self.nash_meta_strategy = NashEquilibriumECOSSolver(self.evaluation_matrix)
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
        After adding a new policy with evaluated average episode reward (a,b) against current 2 policy, 
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

            avg_epi_rewards += rewards

        avg_epi_rewards = avg_epi_rewards/eval_episodes

        return avg_epi_rewards