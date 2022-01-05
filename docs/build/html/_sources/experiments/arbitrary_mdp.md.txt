# Arbitrary MDP Test

To run the test: 

`python run_nash_dqn_simple_mdp.py`

`python run_nash_dqn_exploiter_simple_mdp.py`

The test is on an arbitrary 2-agent MDP environment with 3 transitions, 3 states per timestep, 3 actions per state for each player. The reward function $r(s,a, s^\prime)$ and transition function $T(s^\prime|s,a)$ are all randomly sampled. Here we allow the dependency on next state $s^\prime$, such that the update rule $Q(s,a)=\mathbb{E}_{s'\sim T(s'|s,a)}[r(s,a,s')+\gamma V(s')]$ can be stochastic (large batch size is essential for stabilize the convergence in this case).

Nash DQN:

<p float="left">
<img src="img/arbitrary_mdp/nash_dqn_test2.png" alt="drawing" width=700/>
</p>


Nash DQN Exploiter:

<p float="left">
<img src="img/arbitrary_mdp/nash_dqn_exploiter_test2.png" alt="drawing" width=700/>
</p>

