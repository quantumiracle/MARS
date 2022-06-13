import numpy as np
from .meta_learner import MetaLearner

class SelfPlaySymMetaLearner(MetaLearner):
    """
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Selfplay is the learning scheme following iterative best response,
    where the agent always learn against the latest version of its historical policies.
    The policy is stored and updated to its opponent when its average recent performance
    achieves a certain threshold score over its opponent or for a long enough training period.
    """
    def __init__(self, logger, args, save_checkpoint=True):
        super(SelfPlaySymMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.meta_step = self.last_meta_step = 0

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        self.meta_step += 1
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
             - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and self.meta_step - self.last_meta_step > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                # model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(logger.current_episode)) # save all checkpoints
                # model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+str(logger.current_episode))

                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+'1')  # save only the latest checkpoint
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+'1')  # this step is actually initialize the next learnable model, which is not necessary
            logger.additional_logs.append(f'Score delta: {score_delta}, udpate the opponent.')

            self.last_meta_step = self.meta_step

            ### The reinit seems to hurt agent performance!
            # model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model
        
        # if (self.meta_step - self.last_meta_step) > agent_reinit_interval:
        #     model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model
        #     self.last_meta_step = self.meta_step

class SelfPlayMetaLearner(SelfPlaySymMetaLearner):

    def __init__(self, logger, args, save_checkpoint=True):
        self.model_path = logger.model_dir

        # get names
        self.current_learnable_model_idx = int(args.marl_spec['trainable_agent_idx'])
        self.current_fixed_opponent_idx = int(args.marl_spec['opponent_idx'])

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.meta_step = self.last_meta_step = 0

    def _switch_charac(self, model):
        """ Iteratively update."""
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

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        self.meta_step += 1
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[logger.keys[self.current_learnable_model_idx]][-score_avg_window:])\
             - np.mean(logger.epi_rewards[logger.keys[self.current_fixed_opponent_idx]][-score_avg_window:])

        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and self.meta_step - self.last_meta_step > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                save_path = self.model_path+'1'+'_'+str(self.current_learnable_model_idx)  # 1 is the model save id; selfplay does not need to track all previous models
                model.agents[self.current_learnable_model_idx].save_model(save_path)  # save only the latest checkpoint
                model.agents[self.current_fixed_opponent_idx].load_model(save_path)  # this step is actually initialize the next learnable model, which is not necessary
            logger.additional_logs.append(f'Score delta: {score_delta}, update the opponent.')

            self.last_meta_step = self.meta_step
            
            self._switch_charac(model)
            # model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # if (self.meta_step - self.last_meta_step) > agent_reinit_interval:
        #     model.agents[self.current_learnable_model_idx].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model
        #     self.last_meta_step = self.meta_step


class FictitiousSelfPlaySymMetaLearner(MetaLearner):
    """
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Fictitious selfplay is the learning scheme that the agent always learn
    against the average of the historical best response policy. If the policies can 
    not be explicitly averaged, then a league of historical policies needs to be maintained.
    The policy is stored and saved into the league when its average recent performance 
    since the last checkpoint achieves a certain threshold score over its opponent 
    or for a long enough training period.
    """
    def __init__(self, logger, args, save_checkpoint=True):
        super(FictitiousSelfPlaySymMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]
        self.current_learnable_model_idx = int(args.marl_spec['trainable_agent_idx'])
        self.current_fixed_opponent_idx = int(args.marl_spec['opponent_idx'])

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.meta_step = self.last_meta_step = 0
        # although there is only one learnable agent, the checkpoints and strategies are replicated for both players,
        # so as to match with the asymetric version 
        self.saved_checkpoints = [[] for _ in range(2)] 
        self.meta_strategies = [[] for _ in range(2)] 

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.
        
        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        self.meta_step += 1
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values
        
        # save current best response if qualified
        step_diff = self.meta_step - self.last_meta_step
        if step_diff > max(min_update_interval, len(self.saved_checkpoints[self.current_fixed_opponent_idx])):  # almost ensure ergodicity of opponent's policy set
            score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
                - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])
            if score_delta  > self.args.marl_spec['selfplay_score_delta']:
                if self.save_checkpoint:
                    save_path = self.model_path+str(self.meta_step)
                    model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(save_path) # save all checkpoints
                    # update for both players
                    self.saved_checkpoints[self.current_learnable_model_idx].append(str(self.meta_step))
                    self.saved_checkpoints[self.current_fixed_opponent_idx].append(str(self.meta_step))
                    logger.additional_logs.append(f'Score delta: {score_delta}, save the model to {save_path}.')

                self.last_meta_step = self.meta_step
                # model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # load a model for each episode to achieve an empiral average policy
        current_policy_checkpoints = self.saved_checkpoints[self.current_fixed_opponent_idx]  # load checkpoints from opponent's available policy set
        if len(current_policy_checkpoints) > 0: # the policy set has one or more policies to sample from
            # update for both players
            self.meta_strategies[self.current_learnable_model_idx] = np.ones(len(current_policy_checkpoints))/len(current_policy_checkpoints)  # uniformly distributed
            self.meta_strategies[self.current_fixed_opponent_idx] = np.ones(len(current_policy_checkpoints))/len(current_policy_checkpoints)  # uniformly distributed
            self._replace_agent_with_meta(model, self.args.marl_spec['opponent_idx'], current_policy_checkpoints)

        # if (self.meta_step - self.last_meta_step) > agent_reinit_interval:
        #     model.agents[self.args.marl_spec['trainable_agent_idx']].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model
        #     self.last_meta_step = self.meta_step

class FictitiousSelfPlayMetaLearner(FictitiousSelfPlaySymMetaLearner):
    """
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.

    Fictitious selfplay is the learning scheme that the agent always learn
    against the average of the historical best response policy. If the policies can 
    not be explicitly averaged, then a league of historical policies needs to be maintained.
    The policy is stored and saved into the league when its average recent performance 
    since the last checkpoint achieves a certain threshold score over its opponent 
    or for a long enough training period.
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

    def _switch_charac(self, model):
        """ Iteratively update."""
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

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.
        
        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        self.meta_step += 1
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        # save current best response if qualified
        step_diff = self.meta_step - self.last_meta_step
        if step_diff > max(min_update_interval, len(self.saved_checkpoints[self.current_fixed_opponent_idx])):  # almost ensure ergodicity of opponent's policy set
            score_delta = np.mean(logger.epi_rewards[logger.keys[self.current_learnable_model_idx]][-score_avg_window:])\
                - np.mean(logger.epi_rewards[logger.keys[self.current_fixed_opponent_idx]][-score_avg_window:])

            if score_delta  > self.args.marl_spec['selfplay_score_delta']\
                and self.meta_step - self.last_meta_step > min_update_interval:
                # update the opponent with current model, assume they are of the same type
                if self.save_checkpoint:
                    save_path = self.model_path+str(self.meta_step)+'_'+str(self.current_learnable_model_idx)
                    model.agents[self.current_learnable_model_idx].save_model(save_path) # save all checkpoints
                    self.saved_checkpoints[self.current_learnable_model_idx].append(str(self.meta_step))
                    logger.additional_logs.append(f'Score delta: {score_delta}, save the model to {save_path}.')

                self.last_meta_step = self.meta_step
                self._switch_charac(model)
                # model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # load a model for each episode to achieve an empiral average policy
        current_policy_checkpoints = self.saved_checkpoints[self.current_fixed_opponent_idx]  # use the learnable to get best response of the policy set of the fixed agent
        if len(current_policy_checkpoints) > 0:  # the policy set has one or more policies to sample from
            self.meta_strategies[self.current_fixed_opponent_idx] = np.ones(len(current_policy_checkpoints))/len(current_policy_checkpoints)  # uniformly distributed         
            self._replace_agent_with_meta(model, self.current_fixed_opponent_idx, current_policy_checkpoints, postfix = '_'+str(self.current_fixed_opponent_idx))

        # if (self.meta_step - self.last_meta_step) > agent_reinit_interval:
        #     model.agents[self.current_learnable_model_idx].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model
        #     self.last_meta_step = self.meta_step
