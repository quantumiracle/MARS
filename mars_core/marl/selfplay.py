from datetime import datetime


def selfpaly_meta_learn(model, logger, args, save_checkpoint=True):
    """
    Meta learn is the function for MARL meta strategy, 
    which assigns the policy update schedule on a level higher
    than policy update itself in standard RL.
    """
    model_name = logger.keys[args.marl_spec['trainable_agent_idx']]
    opponent_name = logger.keys[args.marl_spec['opponent_idx']]
    score_delta = logger.epi_rewards[model_name][-1] - logger.epi_rewards[opponent_name][-1]
    if score_delta  > args.marl_spec['selfplay_score_delta']:
        # update the opponent with current model, assume they are of the same type
        if save_checkpoint:
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")

            path = f'../model/{args.env_type}_{args.env_name}_marl_method_{dt_string}'
            os.makedirs(path, exist_ok=True)

            model.agents[args.marl_spec['trainable_agent_idx']].save_model(path)
            model.agents[args.marl_spec['opponent_idx']].load_model(path)


