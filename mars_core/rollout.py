def rollout(env, model):
    o = env.reset()
    max_frames = 10000
    for step in range(max_frames):
        a = model.choose_action(o)
        o_, r, d, _ = env.step(a)
        o = o_
        env.render()

        if d:
            break