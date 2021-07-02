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
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        self.model_path = f'../model/{args.env_type}_{args.env_name}_marl_method_{dt_string}'
        os.makedirs(self.model_path, exist_ok=True)

        # get names
        self.model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
        self.opponent_name = logger.keys[args.marl_spec['opponent_idx']]

        self.save_checkpoint = save_checkpoint
        self.args = args

    def step(self, model, logger):
        score_delta = logger.epi_rewards[model_name][-1] - logger.epi_rewards[opponent_name][-1]
        if score_delta  > self.args.marl_spec['selfplay_score_delta']:
            # update the opponent with current model, assume they are of the same type
            if self.save_checkpoint:
                model.agents[self.args.marl_spec['trainable_agent_idx']].save_model(self.model_path)
                model.agents[self.args.marl_spec['opponent_idx']].load_model(self.model_path)



