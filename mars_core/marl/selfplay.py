import os
import numpy as np
from datetime import datetime
class SelfPlayMetaLearner():
    """
    Meta learn is the  for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.
    """
    def __init__(self, logger, args, save_checkpoint=True):
        super(SelfPlayMetaLearner, self).__init__()
        # create model checkpoint save directory
        # now = datetime.now()
        # dt_string = now.strftime("%d%m%Y%H%M%S")
        # self.model_path = f'../model/{args.env_type}_{args.env_name}_marl_method_{dt_string}/'
        # os.makedirs(self.model_path, exist_ok=True)
        self.model_path = logger.model_dir

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args
        self.last_update_epi= 0

    def step(self, model, logger, min_update_interval = 20):
        """
        params: 
            :min_update_interval: mininal opponent update interval in unit of episodes
        """
        score_avg_window = self.args.log_avg_window # use the same average window as logging for score delta
        score_delta = np.mean(logger.epi_rewards[self.model_name][-score_avg_window:])\
             - np.mean(logger.epi_rewards[self.opponent_name][-score_avg_window:])
        print(score_delta)
        if score_delta  > self.args.marl_spec['selfplay_score_delta']\
             and logger.current_episode - self.last_update_epi > min_update_interval:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path+str(logger.current_episode))
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path+str(logger.current_episode))

            self.last_update_epi = logger.current_episode

