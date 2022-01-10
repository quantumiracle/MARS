# These methods only store one-side models; otherwise two-side models are saved.
SelfplayBasedMethods = ['selfplay', 'fictitious_selfplay', 'nxdo'] 

# These methods have a meta strategy: when test/exploit, needs a meta_learner as one agent.
MetaStrategyMethods = ['fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 

# These methods needs a meta step during training.
MetaStepMethods = ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 

# These methods are based on Nash
NashBasedMethods = ['nash_dqn', 'nash_dqn_exploiter', 'nash_ppo']