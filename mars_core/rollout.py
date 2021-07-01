from utils.logger import init_logger

def rollout(env, model, args):
    print("Arguments: ", args)
    overall_steps = 0
    logger = init_logger(env, args)
    for epi in range(args.max_episodes):
        obs = env.reset()
        epi_reward = 0
        for step in range(args.max_steps_per_episode):
            overall_steps += 1
            action = model.choose_action(obs)
            model.scheduler_step(overall_steps)
            obs_, reward, done, info = env.step(action)
            obs = obs_
            logger.log_reward(reward)
            if args.render:
                env.render()

            done_ = [done, done] # TODO
            sample = [obs, action, reward, obs_, done_]
            model.store(sample)

            if done:
                break
            
            if model.ready_to_update:
                if args.update_itr >= 1:
                    for _ in range(args.update_itr):
                        loss = model.update()
                        logger.log_loss(loss)
                elif overall_steps*args.update_itr % 1 == 0:
                    loss = model.update()
                    logger.log_loss(loss)

        logger.log_episode_reward()

        if epi % args.log_interval == 0:
            logger.print(epi)
