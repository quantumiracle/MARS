import os
import numpy as np
from datetime import datetime
class SelfPlayMetaLearner():
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
        super(SelfPlayMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.last_update_epi= 0

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.

        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
             - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and logger.current_episode - self.last_update_epi > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                # model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(logger.current_episode)) # save all checkpoints
                # model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+str(logger.current_episode))

                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+'1')  # save only the latest checkpoint
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+'1')
            logger.additional_logs.append(f'Score delta: {score_delta}, udpate the opponent.')

            self.last_update_epi = logger.current_episode

            model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model
        
        if (logger.current_episode - self.last_update_epi) > agent_reinit_interval:
            model.agents[self.current_learnable_model_idx].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model

class FictitiousSelfPlayMetaLearner():
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
        super(FictitiousSelfPlayMetaLearner, self).__init__()
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.last_update_epi= 0
        self.saved_checkpoints = []

    def step(self, model, logger, *Args):
        """
        A meta learner step (usually at the end of each episode), update models if available.
        
        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        agent_reinit_interval = 1000 # after a long time of unimproved performance against the opponent, reinit the agent
        score_avg_window = self.args.marl_spec['score_avg_window']  # mininal opponent update interval in unit of episodes
        min_update_interval = self.args.marl_spec['min_update_interval'] # the length of window for averaging the score values

        score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
             - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and logger.current_episode - self.last_update_epi > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(logger.current_episode)) # save all checkpoints
                self.saved_checkpoints.append(str(logger.current_episode))
                logger.additional_logs.append(f'Score delta: {score_delta}, save the model to {self.model_path+self.saved_checkpoints[-1]}.')

            self.last_update_epi = logger.current_episode

            model.agents[self.current_learnable_model_idx].reinit(nets_init=False, buffer_init=True, schedulers_init=True)  # reinitialize the model

        # load a model for each episode to achieve an empiral average policy
        if len(self.saved_checkpoints) > 0:
            random_checkpoint = np.random.choice(self.saved_checkpoints)
            model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+random_checkpoint)
            # logger.additional_logs.append(f'Load the random opponent model from {self.model_path+random_checkpoint}.')

        if (logger.current_episode - self.last_update_epi) > agent_reinit_interval:
            model.agents[self.current_learnable_model_idx].reinit(nets_init=True, buffer_init=True, schedulers_init=True)  # reinitialize the model
