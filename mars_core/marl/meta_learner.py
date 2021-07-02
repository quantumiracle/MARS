from .selfplay import SelfPlayMetaLearner

class MetaLearner():
    def __init__(self,):
        pass
        
    def step(self, *kargs):
        pass

def init_meta_learner(logger, args, *kargs):
    if args.marl_method == 'selfplay':
        return SelfPlayMetaLearner(logger, args, *kargs)

    else:
        return MetaLearner()
