# These methods only store one-side models; otherwise two-side models are saved.
SelfplayBasedMethods = ['selfplay', 'fictitious_selfplay', 'nxdo'] 

# # These methods require an evaluation model.
# EvaluationModelMethods = ['nxdo', 'nxdo2'] 

# There methods have a meta strategy
MetaStrategyMethods = ['fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 

# There methods needs a meta step
MetaStepMethods = ['selfplay', 'selfplay2', 'fictitious_selfplay', 'fictitious_selfplay2', 'nxdo', 'nxdo2'] 
