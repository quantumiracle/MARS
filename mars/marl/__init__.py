import numpy as np
from .selfplay import SelfPlayMetaLearner, SelfPlaySymMetaLearner, FictitiousSelfPlayMetaLearner, FictitiousSelfPlaySymMetaLearner
from .double_oracle import PSROSymMetaLearner, PSROMetaLearner
from .meta_learner import MetaLearner

def init_meta_learner(logger, args, *kargs):
    if args.marl_method == 'selfplay':
        return SelfPlayMetaLearner(logger, args, *kargs)

    elif args.marl_method == 'selfplay_sym':
        return SelfPlaySymMetaLearner(logger, args, *kargs)

    elif args.marl_method == 'fictitious_selfplay':
        return FictitiousSelfPlayMetaLearner(logger, args, *kargs)

    elif args.marl_method == 'fictitious_selfplay_sym':
        return FictitiousSelfPlaySymMetaLearner(logger, args, *kargs)

    elif args.marl_method == 'psro':
        return PSROMetaLearner(logger, args, *kargs)

    elif args.marl_method == 'prso_sym':
        return PSROSymMetaLearner(logger, args, *kargs)

    else:
        return MetaLearner()
