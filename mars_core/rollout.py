def rollout(env, model, args):
    print(args)

    for epi in range(args.max_episodes):
        obs = env.reset()
        for step in range(args.max_steps_per_episode):
            action = model.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            obs = obs_
            env.render()

            sample = [obs, action, reward, obs_, done]
            model.store(sample)

            if done:
                break