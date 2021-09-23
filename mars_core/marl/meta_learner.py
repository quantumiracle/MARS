from .selfplay import SelfPlayMetaLearner, FictitiousSelfPlayMetaLearner
from .double_oracle import NXDOMetaLearner, NXDO2SideMetaLearner

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

    if args.marl_method == 'nxdo':
        return NXDOMetaLearner(logger, args, *kargs)

    if args.marl_method == 'nxdo2':
        return NXDO2SideMetaLearner(logger, args, *kargs)

    else:
        return MetaLearner()
