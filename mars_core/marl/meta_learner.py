from .selfplay import SelfPlayMetaLearner, FictitiousSelfPlayMetaLearner

class MetaLearner():
    def __init__(self,):
        pass
        
    def step(self, *kargs):
        pass

def init_meta_learner(logger, args, *kargs):
    if args.marl_method == 'selfplay':
        return SelfPlayMetaLearner(logger, args, *kargs)

    if args.marl_method == 'fictitious_selfplay':
        return FictitiousSelfPlayMetaLearner(logger, args, *kargs)

    else:
        return MetaLearner()
