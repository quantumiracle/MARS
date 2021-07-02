from selfplay import selfpaly_meta_learn

def call_meta_learner(args, *kargs):
    if args.marl_method = 'selfplay':
        return selfpaly_meta_learn(args, *kargs)

    else:
        raise NotImplementedError
