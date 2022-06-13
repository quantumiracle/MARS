import numpy as np
from mars.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumCVXPYSolver
from mars.marl.meta_learner import MetaLearner
from mars.env.import_env import make_env
from mars.rl.agents.dqn import DQN
from mars.rl.agents.ppo import PPO


class PSROSymMetaLearner(MetaLearner):
    """
    This is a version for symmetric agents.
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Policy Space Response Oracle (PSRO) based on Double Oracle 

    Ref: https://arxiv.org/abs/1711.00832
    """
    def __init__(self, logger, args, save_checkpoint=True):
        super(PSROSymMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]
        self.current_learnable_model_idx = int(args.marl_spec['trainable_agent_idx'])
        self.current_fixed_opponent_idx = int(args.marl_spec['opponent_idx'])

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.meta_step = self.last_meta_step = 0
        self.evaluation_matrix = np.array([[0]])  # the evaluated utility matrix (N*N) of policy league with N policies
        
        # although there is only one learnable agent, the checkpoints and strategies are replicated for both players,
        # so as to match with the asymetric version 
        self.saved_checkpoints = [[] for _ in range(2)] 
        self.meta_strategies = [[] for _ in range(2)] 

        logger.add_extr_log('matrix_equilibrium')
        ori_num_envs = args.num_envs
        self.num_envs = args.num_envs = 1
        self.eval_env = make_env(args)
        args.multiprocess = False
        eval_model1 = eval(args.algorithm)(self.eval_env, args)
        eval_model2 = eval(args.algorithm)(self.eval_env, args)
        args.multiprocess = True
        args.num_envs = ori_num_envs
        self.eval_models = [eval_model1, eval_model2]

    def step(self, model, logger, env, args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        """
        self.meta_step += 1
        # score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        # score_avg_window = 10 # the length of window for averaging the score values
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values


        # save current best response if qualified
        step_diff = self.meta_step - self.last_meta_step
        if step_diff > max(min_update_interval, len(self.saved_checkpoints[self.current_fixed_opponent_idx])):  # almost ensure ergodicity of opponent's policy set
            score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
                - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])

            if score_delta  > self.args.marl_spec['selfplay_score_delta']\
                and self.meta_step - self.last_meta_step > min_update_interval:
                # update the opponent with current model, assume they are of the same type
                if self.save_checkpoint:
                    model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(self.meta_step)) # save all checkpoints
                    self.saved_checkpoints[self.current_learnable_model_idx].append(str(self.meta_step))
                    self.saved_checkpoints[self.current_fixed_opponent_idx].append(str(self.meta_step))

                logger.additional_logs.append(f'Score delta: {score_delta}, udpate the opponent.')
                self.last_meta_step = self.meta_step

                # model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

                ### update the opponent with epsilon meta Nash policy
                # evaluate the N*N utility matrix, N is the number of currently saved models
                added_row = []
                saved_checkpoints = self.saved_checkpoints[self.current_learnable_model_idx]
                if len(saved_checkpoints) == 1:
                    model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+saved_checkpoints[-1])
                elif len(saved_checkpoints) > 1:
                    eval_agents = self.eval_models  # these agents are evaluation models (for evaluation purpose only)
                    env = self.eval_env
                    eval_agents[0].load_model(self.model_path+saved_checkpoints[-1]) # current model
                    for previous_model_id in saved_checkpoints[:-1]:
                        eval_agents[1].load_model(self.model_path+previous_model_id)
                        added_row.append(self.evaluate(env, eval_agents, args)[0])
                    self.update_matrix(np.array(added_row)) # add new evaluation results to matrix
                    # rollout with NFSP to learn meta strategy or directly calculate the Nash from the matrix
                    # the solver returns the equilibrium strategies for both players, just take one; it should be the same due to the symmetric poicy space
                    self.meta_strategies, _ = NashEquilibriumECOSSolver(self.evaluation_matrix)
                    logger.extr_logs.append(f'Current meta step: {self.meta_step}, utitlity matrix: {self.evaluation_matrix}, Nash stratey: {self.meta_strategy}')

        # sample from Nash meta policy in a episode-wise manner
        saved_checkpoints = self.saved_checkpoints[self.current_fixed_opponent_idx]
        if len(saved_checkpoints) > 1:
            self._replace_agent_with_meta(model, self.args.marl_spec['opponent_idx'], saved_checkpoints)

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
                ) if self.num_envs > 1 else obs  # transform from (envs, agents, dim) to (agents, envs, dim)
                actions = []
                for state, agent in zip(obs_to_store, agents):
                    action = agent.choose_action(state, Greedy=True)
                    actions.append(action)

                obs_, reward, done, info = env.step(actions)
                obs = obs_
                try:
                    rewards += np.array(reward)
                except:
                    print("Reward sum error in evaluation.")

                if np.any(
                done
                    ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                    break

            avg_epi_rewards += rewards

        avg_epi_rewards = avg_epi_rewards/eval_episodes

        return avg_epi_rewards


class PSROMetaLearner(PSROSymMetaLearner):
    """
    This is a asymmetric/two-side version, which means the
    agents on both sides of the game will maintain 
    a policy sets for each of them. The update of 
    two sides follows an iterative manner.

    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Policy Space Response Oracle (PSRO) based on Double Oracle 

    Ref: https://arxiv.org/abs/1711.00832
    """
    def __init__(self, logger, args, save_checkpoint=True):
        self.model_path = logger.model_dir

        # get names
        self.current_learnable_model_idx = int(args.marl_spec['trainable_agent_idx'])
        self.current_fixed_opponent_idx = int(args.marl_spec['opponent_idx'])

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.meta_step = self.last_meta_step = 0
        self.saved_checkpoints = [[] for _ in range(2)] # for both player
        self.meta_strategies = [[] for _ in range(2)] # for both player
        self.evaluation_matrix = np.array([])  # the evaluated utility matrix (N*N) of policy league with N policies
        logger.add_extr_log('matrix_equilibrium')
        ori_num_envs = args.num_envs
        self.num_envs = args.num_envs = 1
        self.eval_env = make_env(args)
        args.multiprocess = False
        eval_model1 = eval(args.algorithm)(self.eval_env, args)
        eval_model2 = eval(args.algorithm)(self.eval_env, args)
        args.multiprocess = True
        args.num_envs = ori_num_envs
        self.eval_models = [eval_model1, eval_model2]

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
        self.meta_step += 1
        # score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        # score_avg_window = 10 # use the same average window as logging for score delta
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[logger.keys[self.current_learnable_model_idx]][-score_avg_window:])\
             - np.mean(logger.epi_rewards[logger.keys[self.current_fixed_opponent_idx]][-score_avg_window:])

        # this is an indicator that best response policy is found
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
            and self.meta_step - self.last_meta_step > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                save_path = self.model_path+str(self.meta_step)+'_'+str(self.current_learnable_model_idx)
                model.agents[self.current_learnable_model_idx].save_model(save_path) # save all checkpoints
                self.saved_checkpoints[self.current_learnable_model_idx].append(str(self.meta_step))

            logger.additional_logs.append(f'Score delta: {score_delta}, save the model to {save_path}.')
            self.last_meta_step = self.meta_step 

            ### update the opponent with epsilon meta Nash policy
            # evaluate the N*N utility matrix, N is the number of currently saved models
            added_row = []
            if len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 1:
                eval_agents = self.eval_models # these agents are evaluation models (for evaluation purpose only)
                env = self.eval_env
                eval_agents[0].load_model(self.model_path+self.saved_checkpoints[self.current_learnable_model_idx][-1]+'_'+str(self.current_learnable_model_idx)) # load current model
                for opponent_model_id in self.saved_checkpoints[self.current_fixed_opponent_idx]:
                    eval_agents[1].load_model(self.model_path+opponent_model_id+'_'+str(self.current_fixed_opponent_idx))
                    added_row.append(self.evaluate(env, eval_agents, args)[0])  # reward for current learnable model
                # print('row: ', added_row, self.current_learnable_model_idx)
                self.update_matrix(self.current_learnable_model_idx, np.array(added_row)) # add new evaluation results to matrix

                if len(self.saved_checkpoints[self.current_learnable_model_idx])*len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 4: # no need for NE when (1,1), (1,2)
                    # rollout with NFSP to learn meta strategy or directly calculate the Nash from the matrix
                    # self.meta_strategies, _ = NashEquilibriumECOSSolver(self.evaluation_matrix) # present implementation cannot solve non-square matrix
                    self.meta_strategies, _ = NashEquilibriumCVXPYSolver(self.evaluation_matrix) # cvxpy can solve non-square matrix, just a bit slower, but psro doesn't solve Nash often
                    # the solver returns the equilibrium strategies for both players, just take one; it should be the same due to the symmetric poicy space
                    # self.meta_strategies = self.meta_strategies[self.current_learnable_model_idx]
                    logger.extr_logs.append(f'Current episode: {self.meta_step}, utitlity matrix: {self.evaluation_matrix}, Nash stratey: {self.meta_strategies}')
            self._switch_charac(model)
            # model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # sample from Nash meta policy in a episode-wise manner
        current_policy_checkpoints = self.saved_checkpoints[self.current_fixed_opponent_idx] 
        if len(self.saved_checkpoints[self.current_learnable_model_idx])*len(self.saved_checkpoints[self.current_fixed_opponent_idx]) >= 4 \
            and self.meta_strategies is not None:
            self._replace_agent_with_meta(model, self.current_fixed_opponent_idx, current_policy_checkpoints, postfix = '_'+str(self.current_fixed_opponent_idx))

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
