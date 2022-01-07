# These methods only store one-side models; otherwise two-side models are saved.
SelfplayBasedMethods = ['selfplay', 'fictitious_selfplay', 'nxdo'] 

# These methods have a meta strategy
MetaStrategyMethods = ['fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 

# These methods needs a meta step
MetaStepMethods = ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 

# These methods are based on Nash
NashBasedMethods = ['nash_dqn', 'nash_dqn_exploiter', 'nash_ppo']