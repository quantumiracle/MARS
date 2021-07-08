import numpy as np
from utils.logger import init_logger
from marl.meta_learner import init_meta_learner


def rollout(env, model, args):
    print("Arguments: ", args)
    overall_steps = 0
    logger = init_logger(env, args)
    meta_learner = init_meta_learner(logger, args)
    for epi in range(args.max_episodes):
        obs = env.reset()
        for step in range(args.max_steps_per_episode):
            overall_steps += 1
            action_ = model.choose_action(obs)
            model.scheduler_step(overall_steps)
            if isinstance(action_[0], tuple): # action item contains additional information like log probability
                action, other_info = [], []
                for (a, info) in action_:
                    action.append(a)
                    other_info.append(info)
            else:
                action = action_
                other_info = None
            obs_, reward, done, info = env.step(action)

            if args.render:
                env.render()

            if other_info is None: 
                sample = [obs, action, reward, obs_, done]
            else:
                sample = [obs, action, reward, obs_, other_info, done]
            model.store(sample)

            obs = obs_
            logger.log_reward(reward)

            if np.any(
                    done
            ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                logger.log_episode_reward(step)
                break

            if not args.algorithm_spec['episodic_update'] and \
                 model.ready_to_update and overall_steps > args.train_start_frame:
                if args.update_itr >= 1:
                    for _ in range(args.update_itr):
                        loss = model.update()  # only log loss for once, loss is a list
                elif overall_steps * args.update_itr % 1 == 0:
                    loss = model.update()
                if overall_steps % 1000 == 0:  # loss logging interval
                    logger.log_loss(loss)

        if model.ready_to_update:
            if args.algorithm_spec['episodic_update']:
                loss = model.update()
                logger.log_loss(loss)

            meta_learner.step(
                model,
                logger)  # metalearner for selfplay need just one step per episode

        if epi % args.log_interval == 0:
            logger.print_and_save()
            if not args.marl_method:
                model.save_model(logger.model_dir)
