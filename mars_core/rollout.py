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
        epi_reward = 0
        for step in range(args.max_steps_per_episode):
            overall_steps += 1
            action = model.choose_action(obs)
            model.scheduler_step(overall_steps)
            obs_, reward, done, info = env.step(action)
            if args.render:
                env.render()

            sample = [obs, action, reward, obs_, done]
            model.store(sample)

            obs = obs_
            logger.log_reward(reward)

            if np.any(
                    done
            ):  # if any player in a game is done, the game episode done; may not be correct for some envs
                logger.log_episode_reward(step)
                break

            if model.ready_to_update and overall_steps > args.train_start_frame:
                if args.update_itr >= 1:
                    for _ in range(args.update_itr):
                        loss = model.update()
                        logger.log_loss(loss)
                elif overall_steps * args.update_itr % 1 == 0:
                    loss = model.update()
                    logger.log_loss(loss)

        meta_learner.step(
            model,
            logger)  # metalearner for selfplay need just one step per episode

        if epi % args.log_interval == 0:
            logger.print_and_save()
