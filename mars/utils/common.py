# These methods only store one-side models; otherwise two-side models are saved.
SelfplayBasedMethods = ['selfplay_sym', 'fictitious_selfplay_sym', 'psro_sym'] 

# These methods have a meta strategy: when test/exploit, needs a meta_learner as one agent.
MetaStrategyMethods = ['fictitious_selfplay', 'fictitious_selfplay_sym', 'psro', 'psro_sym'] 

# These methods needs a meta step during training.
MetaStepMethods = ['selfplay', 'selfplay_sym', 'fictitious_selfplay', 'fictitious_selfplay_sym', 'psro', 'psro_sym'] 

# These methods are based on Nash
NashBasedMethods = ['nash_dqn', 'nash_dqn_exploiter', 'nash_dqn_factorized', 'nash_ppo', 'nash_actor_critic']

# These methods follow on-policy update
OnPolicyMethods = ['ppo', 'nash_ppo']