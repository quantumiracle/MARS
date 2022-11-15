from .com_lock import CombinatorialLock
from .attack import Attack
from .arbitrary_mdp import ArbitraryMDP
from .arbitrary_richobs_mdp import ArbitraryRichObsMDP
from .mdp_wrapper import MDPWrapper

try:
    attack = Attack(True).env
    combinatorial_lock = CombinatorialLock(5, True).env
    arbitrary_mdp = MDPWrapper(ArbitraryMDP())
    arbitrary_richobs_mdp = MDPWrapper(ArbitraryRichObsMDP())

except:
    pass