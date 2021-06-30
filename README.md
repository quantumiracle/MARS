# marl_torch

A library for running multi-agent reinforcement learning (MARL) on PettingZoo environments.

## Files:

`test_zoo.py`: testing the pettingzoo environments;

`test_mp_env.py`: testing the vectorized wrappers (from tianshou and openai baseline) for parallel sampling;

**Single Agent**

`train_single_pong.py`

`train_single_slimevolley.py`

For training PPO against the environment baseline:

1. `python train_single_pong.py --train`
2. `python train_single_slimevolley.py --train`

**Two Agents**

`train_pettingzoo.py`: single process version.

`train_pettingzoo_mp_vecenv.py`: multi-process version with vectorized environments.

For PettingZoo or SlimeVolley:

​	1.`python train_pettingzoo_mp_vecenv.py --env pong_v1 --ram --num-envs 3 --selfplay`

​	2.`python train_pettingzoo_mp_vecenv.py --env slimevolley_v0 --num-envs 3 --selfplay`



Some deprecated scripts:

`train_pettingzoo_mp.py`:  multi-process version without vectorized environments, but each process contains a environment and a model; it's less efficient in training. 

`train_pettingzoo_mp_vecenv_baseline.py`: multi-process version with vectorized environments (using baseline wrappers), there exists some problem when different parallel envs end at different timestep.